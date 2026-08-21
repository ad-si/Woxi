//! A native `CodeParser`` context.
//!
//! `CodeParser`` is the paclet the Wolfram Language ships for reading source
//! *as source* — as a stream of tokens carrying the positions they came
//! from, rather than as the expression they evaluate to. Editors use it to
//! find syntax errors, and templating languages use it to cut a file at
//! boundaries the reader can see but the evaluator cannot.
//!
//! What is implemented here is the concrete side of that: a tokenizer that
//! covers the whole input, so every character of the source belongs to
//! exactly one token and the tokens' `Source` spans tile the file. That is
//! what [`code_concrete_parse`] hands back, and it is enough to answer the
//! questions the concrete tree is asked — where the newlines are, where a
//! comment starts, which span a token occupies.
//!
//! [`code_parse`] is the abstract side. Woxi's own parser decides whether
//! the source reads at all; when it does not, the failure is reported as the
//! `ErrorNode` the abstract tree is expected to carry, positioned where the
//! reader gave up.

use crate::syntax::Expr;

/// How a node reports the piece of source it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convention {
  /// `Source -> {{line, column}, {line, column}}` — the default.
  LineColumn,
  /// `Source -> {first, last}`, both 1-based character indices, and both
  /// inclusive, so a one-character token reports the same index twice.
  CharacterIndex,
}

impl Convention {
  /// The convention a `CodeParser`SourceConvention -> "…"` option names.
  /// An unrecognized name leaves the default in place, as it does in the
  /// Wolfram Language.
  pub fn from_option_value(value: &str) -> Self {
    match value {
      "SourceCharacterIndex" => Self::CharacterIndex,
      _ => Self::LineColumn,
    }
  }
}

/// One token of the source, with where it came from.
struct Token {
  /// The `Token`…` symbol naming what kind of token this is.
  kind: &'static str,
  /// The exact source text, so the tokens concatenate back to the input.
  text: String,
  /// 1-based character index of the first character.
  first: usize,
  /// 1-based character index just past the last character.
  past: usize,
  /// 1-based line and column of the first character.
  start_line_column: (usize, usize),
  /// 1-based line and column just past the last character.
  end_line_column: (usize, usize),
}

impl Token {
  /// The `Source` value this token reports under `convention`.
  fn source(&self, convention: Convention) -> Expr {
    match convention {
      Convention::CharacterIndex => Expr::List(
        vec![
          Expr::Integer(self.first as i128),
          // `past` is one beyond the token; the convention names the last
          // character itself, so an empty token reports its own start.
          Expr::Integer(self.past.max(self.first + 1) as i128 - 1),
        ]
        .into(),
      ),
      Convention::LineColumn => Expr::List(
        vec![
          line_column(self.start_line_column),
          line_column(self.end_line_column),
        ]
        .into(),
      ),
    }
  }

  /// `LeafNode[Token`kind, "text", <|Source -> …|>]`.
  fn to_leaf_node(&self, convention: Convention) -> Expr {
    Expr::FunctionCall {
      name: "LeafNode".to_string(),
      args: vec![
        Expr::Identifier(self.kind.to_string()),
        Expr::String(self.text.clone()),
        Expr::Association(vec![(
          Expr::Identifier("Source".to_string()),
          self.source(convention),
        )]),
      ]
      .into(),
    }
  }
}

/// `{line, column}` as the Wolfram Language writes it.
fn line_column((line, column): (usize, usize)) -> Expr {
  Expr::List(
    vec![Expr::Integer(line as i128), Expr::Integer(column as i128)].into(),
  )
}

/// The operators the tokenizer knows, longest first so that `===` is read as
/// one token rather than `==` followed by `=`. The name is the `Token`…`
/// symbol the Wolfram Language's own tokenizer reports.
const OPERATORS: &[(&str, &str)] = &[
  (";;", "Token`SemiSemi"),
  ("===", "Token`EqualEqualEqual"),
  ("=!=", "Token`EqualBangEqual"),
  ("//.", "Token`SlashSlashDot"),
  ("//@", "Token`SlashSlashAt"),
  ("/;", "Token`SlashSemi"),
  ("^:=", "Token`CaretColonEqual"),
  ("+=", "Token`PlusEqual"),
  ("-=", "Token`MinusEqual"),
  ("*=", "Token`StarEqual"),
  ("/=", "Token`SlashEqual"),
  ("^=", "Token`CaretEqual"),
  (":=", "Token`ColonEqual"),
  ("==", "Token`EqualEqual"),
  ("!=", "Token`BangEqual"),
  ("<=", "Token`LessEqual"),
  (">=", "Token`GreaterEqual"),
  ("&&", "Token`AmpAmp"),
  ("||", "Token`BarBar"),
  ("++", "Token`PlusPlus"),
  ("--", "Token`MinusMinus"),
  ("->", "Token`MinusGreater"),
  (":>", "Token`ColonGreater"),
  ("/.", "Token`SlashDot"),
  ("/@", "Token`SlashAt"),
  ("//", "Token`SlashSlash"),
  ("@@@", "Token`AtAtAt"),
  ("@@", "Token`AtAt"),
  ("@*", "Token`AtStar"),
  ("/*", "Token`SlashStar"),
  ("[[", "Token`OpenSquareOpenSquare"),
  ("]]", "Token`CloseSquareCloseSquare"),
  ("<|", "Token`LessBar"),
  ("|>", "Token`BarGreater"),
  ("::", "Token`ColonColon"),
  ("~~", "Token`TildeTilde"),
  ("...", "Token`DotDotDot"),
  ("..", "Token`DotDot"),
  ("<<", "Token`LessLess"),
  (">>>", "Token`GreaterGreaterGreater"),
  (">>", "Token`GreaterGreater"),
  ("(", "Token`OpenParen"),
  (")", "Token`CloseParen"),
  ("[", "Token`OpenSquare"),
  ("]", "Token`CloseSquare"),
  ("{", "Token`OpenCurly"),
  ("}", "Token`CloseCurly"),
  (",", "Token`Comma"),
  (";", "Token`Semi"),
  (":", "Token`Colon"),
  ("+", "Token`Plus"),
  ("-", "Token`Minus"),
  ("*", "Token`Star"),
  ("/", "Token`Slash"),
  ("^", "Token`Caret"),
  ("=", "Token`Equal"),
  ("<", "Token`Less"),
  (">", "Token`Greater"),
  ("!", "Token`Bang"),
  ("&", "Token`Amp"),
  ("|", "Token`Bar"),
  ("@", "Token`At"),
  ("~", "Token`Tilde"),
  ("?", "Token`Question"),
  (".", "Token`Dot"),
  ("'", "Token`SingleQuote"),
];

/// Split `source` into tokens covering every character of it.
///
/// Nothing is dropped: whitespace, newlines and comments are tokens like any
/// other, because a concrete tree is meant to be able to reproduce the file
/// it was read from.
fn tokenize(source: &str) -> Vec<Token> {
  let chars: Vec<char> = source.chars().collect();
  let mut tokens = Vec::new();
  let mut i = 0usize;
  let mut line = 1usize;
  let mut column = 1usize;

  while i < chars.len() {
    let start = i;
    let start_line = line;
    let start_column = column;
    let kind = scan_one(&chars, &mut i);
    // A scanner that consumed nothing would loop forever; treat the
    // character as its own unknown token instead.
    if i == start {
      i += 1;
    }
    let text: String = chars[start..i].iter().collect();
    for ch in &chars[start..i] {
      if *ch == '\n' {
        line += 1;
        column = 1;
      } else {
        column += 1;
      }
    }
    tokens.push(Token {
      kind,
      text,
      first: start + 1,
      past: i + 1,
      start_line_column: (start_line, start_column),
      end_line_column: (line, column),
    });
  }
  tokens
}

/// Consume one token starting at `*i`, returning what kind it was.
fn scan_one(chars: &[char], i: &mut usize) -> &'static str {
  let ch = chars[*i];

  // A newline is its own token — the one a templating reader looks for.
  if ch == '\n' {
    *i += 1;
    return "Token`Newline";
  }
  if ch == '\r' {
    *i += 1;
    if chars.get(*i) == Some(&'\n') {
      *i += 1;
    }
    return "Token`Newline";
  }
  if ch == ' ' || ch == '\t' {
    while matches!(chars.get(*i), Some(' ' | '\t')) {
      *i += 1;
    }
    return "Token`Whitespace";
  }

  // `(* … *)`, which nests.
  if ch == '(' && chars.get(*i + 1) == Some(&'*') {
    *i += 2;
    let mut depth = 1usize;
    while *i < chars.len() && depth > 0 {
      if chars[*i] == '(' && chars.get(*i + 1) == Some(&'*') {
        depth += 1;
        *i += 2;
      } else if chars[*i] == '*' && chars.get(*i + 1) == Some(&')') {
        depth -= 1;
        *i += 2;
      } else {
        *i += 1;
      }
    }
    return if depth == 0 {
      "Token`Comment"
    } else {
      "Token`Error`UnterminatedComment"
    };
  }

  if ch == '"' {
    *i += 1;
    while *i < chars.len() {
      match chars[*i] {
        '\\' => *i += 2,
        '"' => {
          *i += 1;
          return "Token`String";
        }
        _ => *i += 1,
      }
    }
    *i = chars.len();
    return "Token`Error`UnterminatedString";
  }

  if ch.is_ascii_digit() {
    return scan_number(chars, i);
  }

  // A symbol may carry a context (`` Foo`bar ``) and may start with `$`.
  if ch.is_alphabetic() || ch == '$' || ch == '`' {
    while matches!(chars.get(*i), Some(c) if c.is_alphanumeric() || *c == '$' || *c == '`')
    {
      *i += 1;
    }
    return "Token`Symbol";
  }

  if ch == '_' {
    let mut unders = 0usize;
    while chars.get(*i) == Some(&'_') {
      unders += 1;
      *i += 1;
    }
    // `_.` is the optional-default pattern, one token in its own right.
    if unders == 1 && chars.get(*i) == Some(&'.') {
      *i += 1;
      return "Token`UnderDot";
    }
    return match unders {
      1 => "Token`Under",
      2 => "Token`UnderUnder",
      _ => "Token`UnderUnderUnder",
    };
  }

  if ch == '#' {
    *i += 1;
    let sequence = chars.get(*i) == Some(&'#');
    if sequence {
      *i += 1;
    }
    while matches!(chars.get(*i), Some(c) if c.is_alphanumeric()) {
      *i += 1;
    }
    return if sequence {
      "Token`HashHash"
    } else {
      "Token`Hash"
    };
  }

  if ch == '%' {
    while chars.get(*i) == Some(&'%') {
      *i += 1;
    }
    while matches!(chars.get(*i), Some(c) if c.is_ascii_digit()) {
      *i += 1;
    }
    return "Token`Percent";
  }

  // `\[Alpha]` and friends are one character of source spelled out.
  if ch == '\\' && chars.get(*i + 1) == Some(&'[') {
    *i += 2;
    while *i < chars.len() && chars[*i] != ']' {
      *i += 1;
    }
    if *i < chars.len() {
      *i += 1;
    }
    return "Token`Symbol";
  }

  for (text, name) in OPERATORS {
    let candidate: Vec<char> = text.chars().collect();
    if chars[*i..].starts_with(&candidate) {
      *i += candidate.len();
      return name;
    }
  }

  *i += 1;
  "Token`Error`UnhandledCharacter"
}

/// Consume a number: digits, an optional fraction, an optional exponent, and
/// the Wolfram Language's `base^^digits` and `` `precision `` forms.
fn scan_number(chars: &[char], i: &mut usize) -> &'static str {
  let mut is_real = false;
  while matches!(chars.get(*i), Some(c) if c.is_ascii_digit()) {
    *i += 1;
  }
  // `16^^ff` — a based number, whose digits may include letters.
  if chars.get(*i) == Some(&'^') && chars.get(*i + 1) == Some(&'^') {
    *i += 2;
    while matches!(chars.get(*i), Some(c) if c.is_alphanumeric() || *c == '.') {
      *i += 1;
    }
    return "Token`Integer";
  }
  // A dot is part of the number only when a digit follows it; `1..` is a
  // repeated pattern, not a malformed real.
  if chars.get(*i) == Some(&'.')
    && matches!(chars.get(*i + 1), Some(c) if c.is_ascii_digit())
  {
    is_real = true;
    *i += 1;
    while matches!(chars.get(*i), Some(c) if c.is_ascii_digit()) {
      *i += 1;
    }
  }
  // `` 1.5`20 `` — a precision mark.
  if chars.get(*i) == Some(&'`') {
    is_real = true;
    *i += 1;
    while chars.get(*i) == Some(&'`') {
      *i += 1;
    }
    while matches!(chars.get(*i), Some(c) if c.is_ascii_digit() || *c == '.') {
      *i += 1;
    }
  }
  // `1*^6` — the scientific-notation operator.
  if chars.get(*i) == Some(&'*') && chars.get(*i + 1) == Some(&'^') {
    is_real = true;
    *i += 2;
    if matches!(chars.get(*i), Some('+' | '-')) {
      *i += 1;
    }
    while matches!(chars.get(*i), Some(c) if c.is_ascii_digit()) {
      *i += 1;
    }
  }
  if is_real {
    "Token`Real"
  } else {
    "Token`Integer"
  }
}

/// `CodeParser`CodeTokenize[source]` — the tokens, as leaf nodes.
pub fn code_tokenize(source: &str, convention: Convention) -> Expr {
  Expr::List(
    tokenize(source)
      .iter()
      .map(|t| t.to_leaf_node(convention))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// `CodeParser`CodeConcreteParse[source]` — the concrete tree.
///
/// The children are the tokens themselves. A fuller implementation would
/// group them under the operator nodes they belong to; what callers ask the
/// concrete tree for, and what this answers, is where each piece of the
/// source sits.
pub fn code_concrete_parse(source: &str, convention: Convention) -> Expr {
  Expr::FunctionCall {
    name: "ContainerNode".to_string(),
    args: vec![
      Expr::Identifier("String".to_string()),
      code_tokenize(source, convention),
      Expr::Association(vec![]),
    ]
    .into(),
  }
}

/// `CodeParser`CodeParse[source]` — the abstract tree.
///
/// Source that reads gives a container of the expressions it read. Source
/// that does not gives one carrying an `ErrorNode`, positioned where the
/// reader stopped, which is what a syntax check looks for.
pub fn code_parse(source: &str, convention: Convention) -> Expr {
  // The same reader the interpreter uses, so what `CodeParse` calls a
  // syntax error is exactly what running the file would have called one.
  let prepared = crate::insert_statement_separators(source);
  let children = match crate::parse(&prepared) {
    Ok(pairs) => pairs
      .filter(|pair| !matches!(pair.as_rule(), crate::Rule::EOI))
      .map(|pair| Expr::FunctionCall {
        name: "CodeParser`Abstract`Node".to_string(),
        args: vec![crate::syntax::pair_to_expr(pair)].into(),
      })
      .collect::<Vec<_>>(),
    Err(error) => vec![error_node(source, &error.to_string(), convention)],
  };
  Expr::FunctionCall {
    name: "ContainerNode".to_string(),
    args: vec![
      Expr::Identifier("String".to_string()),
      Expr::List(children.into()),
      Expr::Association(vec![]),
    ]
    .into(),
  }
}

/// An `ErrorNode` describing why `source` would not read.
///
/// The position comes from the parser's own report when it names one, and
/// otherwise from the end of the source — a reader that ran out of input
/// failed there.
fn error_node(source: &str, message: &str, convention: Convention) -> Expr {
  let (line, column) = parse_error_position(message)
    .unwrap_or_else(|| end_of_source_position(source));
  let index = character_index_of(source, line, column);
  let span = match convention {
    Convention::CharacterIndex => Expr::List(
      vec![Expr::Integer(index as i128), Expr::Integer(index as i128)].into(),
    ),
    Convention::LineColumn => Expr::List(
      vec![line_column((line, column)), line_column((line, column))].into(),
    ),
  };
  Expr::FunctionCall {
    name: "ErrorNode".to_string(),
    args: vec![
      Expr::Identifier("Token`Error`UnexpectedCharacter".to_string()),
      Expr::String(message.to_string()),
      Expr::Association(vec![(Expr::Identifier("Source".to_string()), span)]),
    ]
    .into(),
  }
}

/// The `line:column` a pest parse error names, when it names one.
fn parse_error_position(message: &str) -> Option<(usize, usize)> {
  let arrow = message.find("--> ")?;
  let rest = &message[arrow + 4..];
  let end = rest.find('\n').unwrap_or(rest.len());
  let (line, column) = rest[..end].trim().split_once(':')?;
  Some((line.trim().parse().ok()?, column.trim().parse().ok()?))
}

/// The line and column just past the last character of `source`.
fn end_of_source_position(source: &str) -> (usize, usize) {
  let mut line = 1usize;
  let mut column = 1usize;
  for ch in source.chars() {
    if ch == '\n' {
      line += 1;
      column = 1;
    } else {
      column += 1;
    }
  }
  (line, column)
}

/// The 1-based character index of a line and column in `source`.
fn character_index_of(source: &str, line: usize, column: usize) -> usize {
  let mut current_line = 1usize;
  let mut index = 1usize;
  for ch in source.chars() {
    if current_line == line {
      return index + column - 1;
    }
    if ch == '\n' {
      current_line += 1;
    }
    index += 1;
  }
  index
}
