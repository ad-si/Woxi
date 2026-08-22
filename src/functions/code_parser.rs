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
  /// The symbol naming what kind of token this is. Most are `Token`…`, but
  /// the kinds the language already has a name for — `Symbol`, `Integer`,
  /// `Real`, `Rational`, `String`, `Whitespace` — are reported under it.
  kind: String,
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

  /// `LeafNode[kind, "text", <|Source -> …|>]`.
  fn to_leaf_node(&self, convention: Convention) -> Expr {
    Expr::FunctionCall {
      name: "LeafNode".to_string(),
      args: vec![
        Expr::Identifier(self.kind.clone()),
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
  ("//=", "Token`SlashSlashEqual"),
  ("/;", "Token`SlashSemi"),
  ("/:", "Token`SlashColon"),
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
  ("<->", "Token`LessMinusGreater"),
  ("|->", "Token`BarMinusGreater"),
  ("->", "Token`MinusGreater"),
  (":>", "Token`ColonGreater"),
  ("/.", "Token`SlashDot"),
  ("/@", "Token`SlashAt"),
  ("//", "Token`SlashSlash"),
  ("@@@", "Token`AtAtAt"),
  ("@@", "Token`AtAt"),
  ("@*", "Token`AtStar"),
  ("/*", "Token`SlashStar"),
  ("<|", "Token`LessBar"),
  ("|>", "Token`BarGreater"),
  ("<>", "Token`LessGreater"),
  ("**", "Token`StarStar"),
  ("::[", "Token`ColonColonOpenSquare"),
  ("::", "Token`ColonColon"),
  ("??", "Token`QuestionQuestion"),
  ("!!", "Token`BangBang"),
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

/// The named characters the Wolfram Language reads as operators, so that
/// `\[Rule]` is a token of its own rather than a letter of a symbol. The
/// token it reports is `Token`LongName`` followed by the name.
const LONG_NAME_OPERATORS: &[&str] = &[
  "Not",
  "PlusMinus",
  "CenterDot",
  "Times",
  "Divide",
  "OpenCurlyQuote",
  "CloseCurlyQuote",
  "OpenCurlyDoubleQuote",
  "CloseCurlyDoubleQuote",
  "InvisibleTimes",
  "LeftArrow",
  "UpArrow",
  "RightArrow",
  "DownArrow",
  "LeftRightArrow",
  "UpDownArrow",
  "UpperLeftArrow",
  "UpperRightArrow",
  "LowerRightArrow",
  "LowerLeftArrow",
  "LeftTeeArrow",
  "UpTeeArrow",
  "RightTeeArrow",
  "DownTeeArrow",
  "LeftVector",
  "DownLeftVector",
  "RightUpVector",
  "LeftUpVector",
  "RightVector",
  "DownRightVector",
  "RightDownVector",
  "LeftDownVector",
  "RightArrowLeftArrow",
  "UpArrowDownArrow",
  "LeftArrowRightArrow",
  "ReverseEquilibrium",
  "Equilibrium",
  "DoubleLeftArrow",
  "DoubleUpArrow",
  "DoubleRightArrow",
  "DoubleDownArrow",
  "DoubleLeftRightArrow",
  "DoubleUpDownArrow",
  "LeftArrowBar",
  "RightArrowBar",
  "DownArrowUpArrow",
  "ForAll",
  "PartialD",
  "Exists",
  "NotExists",
  "Del",
  "Element",
  "NotElement",
  "ReverseElement",
  "NotReverseElement",
  "SuchThat",
  "Product",
  "Coproduct",
  "Sum",
  "Minus",
  "MinusPlus",
  "DivisionSlash",
  "Backslash",
  "SmallCircle",
  "Sqrt",
  "CubeRoot",
  "Proportional",
  "Divides",
  "DoubleVerticalBar",
  "NotDoubleVerticalBar",
  "And",
  "Or",
  "Integral",
  "ContourIntegral",
  "DoubleContourIntegral",
  "ClockwiseContourIntegral",
  "CounterClockwiseContourIntegral",
  "Therefore",
  "Because",
  "Colon",
  "Proportion",
  "Tilde",
  "VerticalTilde",
  "NotTilde",
  "EqualTilde",
  "TildeEqual",
  "NotTildeEqual",
  "TildeFullEqual",
  "NotTildeFullEqual",
  "TildeTilde",
  "NotTildeTilde",
  "CupCap",
  "HumpDownHump",
  "HumpEqual",
  "DotEqual",
  "NotEqual",
  "Congruent",
  "NotCongruent",
  "LessEqual",
  "GreaterEqual",
  "LessFullEqual",
  "GreaterFullEqual",
  "NotLessFullEqual",
  "NotGreaterFullEqual",
  "LessLess",
  "GreaterGreater",
  "NotCupCap",
  "NotLess",
  "NotGreater",
  "NotLessEqual",
  "NotGreaterEqual",
  "LessTilde",
  "GreaterTilde",
  "NotLessTilde",
  "NotGreaterTilde",
  "LessGreater",
  "GreaterLess",
  "NotLessGreater",
  "NotGreaterLess",
  "Precedes",
  "Succeeds",
  "PrecedesSlantEqual",
  "SucceedsSlantEqual",
  "PrecedesTilde",
  "SucceedsTilde",
  "NotPrecedes",
  "NotSucceeds",
  "Subset",
  "Superset",
  "NotSubset",
  "NotSuperset",
  "SubsetEqual",
  "SupersetEqual",
  "NotSubsetEqual",
  "NotSupersetEqual",
  "UnionPlus",
  "SquareSubset",
  "SquareSuperset",
  "SquareSubsetEqual",
  "SquareSupersetEqual",
  "SquareIntersection",
  "SquareUnion",
  "CirclePlus",
  "CircleMinus",
  "CircleTimes",
  "CircleDot",
  "RightTee",
  "LeftTee",
  "DownTee",
  "UpTee",
  "DoubleRightTee",
  "LeftTriangle",
  "RightTriangle",
  "LeftTriangleEqual",
  "RightTriangleEqual",
  "Xor",
  "Nand",
  "Nor",
  "Wedge",
  "Vee",
  "Intersection",
  "Union",
  "Diamond",
  "Star",
  "LessEqualGreater",
  "GreaterEqualLess",
  "NotPrecedesSlantEqual",
  "NotSucceedsSlantEqual",
  "NotSquareSubsetEqual",
  "NotSquareSupersetEqual",
  "NotPrecedesTilde",
  "NotSucceedsTilde",
  "NotLeftTriangle",
  "NotRightTriangle",
  "NotLeftTriangleEqual",
  "NotRightTriangleEqual",
  "LeftCeiling",
  "RightCeiling",
  "LeftFloor",
  "RightFloor",
  "Cap",
  "Cup",
  "LeftAngleBracket",
  "RightAngleBracket",
  "Perpendicular",
  "LongLeftArrow",
  "LongRightArrow",
  "LongLeftRightArrow",
  "DoubleLongLeftArrow",
  "DoubleLongRightArrow",
  "DoubleLongLeftRightArrow",
  "UpArrowBar",
  "DownArrowBar",
  "LeftRightVector",
  "RightUpDownVector",
  "DownLeftRightVector",
  "LeftUpDownVector",
  "LeftVectorBar",
  "RightVectorBar",
  "RightUpVectorBar",
  "RightDownVectorBar",
  "DownLeftVectorBar",
  "DownRightVectorBar",
  "LeftUpVectorBar",
  "LeftDownVectorBar",
  "LeftTeeVector",
  "RightTeeVector",
  "RightUpTeeVector",
  "RightDownTeeVector",
  "DownLeftTeeVector",
  "DownRightTeeVector",
  "LeftUpTeeVector",
  "LeftDownTeeVector",
  "UpEquilibrium",
  "ReverseUpEquilibrium",
  "RoundImplies",
  "LeftTriangleBar",
  "RightTriangleBar",
  "Equivalent",
  "LessSlantEqual",
  "GreaterSlantEqual",
  "NestedLessLess",
  "NestedGreaterGreater",
  "PrecedesEqual",
  "SucceedsEqual",
  "DoubleLeftTee",
  "LeftDoubleBracket",
  "RightDoubleBracket",
  "LeftAssociation",
  "RightAssociation",
  "TwoWayRule",
  "Piecewise",
  "ImplicitPlus",
  "AutoLeftMatch",
  "AutoRightMatch",
  "InvisiblePrefixScriptBase",
  "InvisiblePostfixScriptBase",
  "Transpose",
  "Conjugate",
  "ConjugateTranspose",
  "HermitianConjugate",
  "VerticalBar",
  "NotVerticalBar",
  "Distributed",
  "Conditioned",
  "UndirectedEdge",
  "DirectedEdge",
  "ContinuedFractionK",
  "TensorProduct",
  "TensorWedge",
  "ProbabilityPr",
  "ExpectationE",
  "PermutationProduct",
  "NotEqualTilde",
  "NotHumpEqual",
  "NotHumpDownHump",
  "NotLeftTriangleBar",
  "NotRightTriangleBar",
  "NotLessLess",
  "NotNestedLessLess",
  "NotLessSlantEqual",
  "NotGreaterGreater",
  "NotNestedGreaterGreater",
  "NotGreaterSlantEqual",
  "NotPrecedesEqual",
  "NotSucceedsEqual",
  "NotSquareSubset",
  "NotSquareSuperset",
  "Equal",
  "VerticalSeparator",
  "VectorGreater",
  "VectorGreaterEqual",
  "VectorLess",
  "VectorLessEqual",
  "Limit",
  "MaxLimit",
  "MinLimit",
  "Cross",
  "Function",
  "Xnor",
  "DiscreteShift",
  "DifferenceDelta",
  "DiscreteRatio",
  "RuleDelayed",
  "Square",
  "Rule",
  "Implies",
  "ShortRightArrow",
  "ShortLeftArrow",
  "ShortUpArrow",
  "ShortDownArrow",
  "Application",
  "LeftBracketingBar",
  "RightBracketingBar",
  "LeftDoubleBracketingBar",
  "RightDoubleBracketingBar",
  "CapitalDifferentialD",
  "DifferentialD",
  "InvisibleComma",
  "InvisibleApplication",
  "LongEqual",
];

/// The named characters that are spelled-out whitespace.
const LONG_NAME_WHITESPACE: &[&str] = &[
  "RawTab",
  "RawSpace",
  "NonBreakingSpace",
  "ThickSpace",
  "ThinSpace",
  "VeryThinSpace",
  "MediumSpace",
  "NoBreak",
  "SpaceIndicator",
  "InvisibleSpace",
  "NegativeVeryThinSpace",
  "NegativeThinSpace",
  "NegativeMediumSpace",
  "NegativeThickSpace",
  "COMPATIBILITYNoBreak",
  "AutoSpace",
  "Continuation",
  "RoundSpaceIndicator",
  "PageBreakAbove",
  "PageBreakBelow",
  "DiscretionaryPageBreakAbove",
  "DiscretionaryPageBreakBelow",
  "AlignmentMarker",
];

/// The named characters that spell a token the plain syntax also has,
/// paired with the token they stand for.
const LONG_NAME_TOKENS: &[(&str, &str)] = &[
  ("NewLine", "Token`Newline"),
  ("RawReturn", "Token`Newline"),
  ("RawExclamation", "Token`Bang"),
  ("RawNumberSign", "Token`Hash"),
  ("RawPercent", "Token`Percent"),
  ("RawAmpersand", "Token`Amp"),
  ("RawQuote", "Token`SingleQuote"),
  ("RawLeftParenthesis", "Token`OpenParen"),
  ("RawRightParenthesis", "Token`CloseParen"),
  ("RawStar", "Token`Star"),
  ("RawPlus", "Token`Plus"),
  ("RawComma", "Token`Comma"),
  ("RawDash", "Token`Minus"),
  ("RawDot", "Token`Dot"),
  ("RawSlash", "Token`Slash"),
  ("RawColon", "Token`Colon"),
  ("RawSemicolon", "Token`Semi"),
  ("RawLess", "Token`Less"),
  ("RawEqual", "Token`Equal"),
  ("RawGreater", "Token`Greater"),
  ("RawQuestion", "Token`Question"),
  ("RawAt", "Token`At"),
  ("RawLeftBracket", "Token`OpenSquare"),
  ("RawRightBracket", "Token`CloseSquare"),
  ("RawWedge", "Token`Caret"),
  ("RawUnderscore", "Token`Under"),
  ("RawLeftBrace", "Token`OpenCurly"),
  ("RawVerticalBar", "Token`Bar"),
  ("RawRightBrace", "Token`CloseCurly"),
  ("RawTilde", "Token`Tilde"),
  ("LineSeparator", "Token`Newline"),
  ("ParagraphSeparator", "Token`Newline"),
  ("IndentingNewLine", "Token`Newline"),
  ("DiscretionaryLineSeparator", "Token`Newline"),
  ("DiscretionaryParagraphSeparator", "Token`Newline"),
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

/// The backslash forms that open a piece of typeset input written linearly,
/// each of which is a token of its own.
const LINEAR_SYNTAX: &[(char, &str)] = &[
  ('!', "Token`LinearSyntax`Bang"),
  (')', "Token`LinearSyntax`CloseParen"),
  ('*', "Token`LinearSyntax`Star"),
  ('%', "Token`LinearSyntax`Percent"),
  ('+', "Token`LinearSyntax`Plus"),
  ('/', "Token`LinearSyntax`Slash"),
  ('@', "Token`LinearSyntax`At"),
  ('^', "Token`LinearSyntax`Caret"),
  ('_', "Token`LinearSyntax`Under"),
  ('&', "Token`LinearSyntax`Amp"),
  ('`', "Token`LinearSyntax`BackTick"),
  (' ', "Token`LinearSyntax`Space"),
];

/// Consume one token starting at `*i`, returning what kind it was.
fn scan_one(chars: &[char], i: &mut usize) -> String {
  let ch = chars[*i];

  // A newline is its own token — the one a templating reader looks for.
  if ch == '\n' {
    *i += 1;
    return "Token`Newline".to_string();
  }
  if ch == '\r' {
    *i += 1;
    if chars.get(*i) == Some(&'\n') {
      *i += 1;
    }
    return "Token`Newline".to_string();
  }
  // Every space is a token of its own: two spaces are two tokens, not one
  // run, because that is how the Wolfram Language reports them.
  if ch == ' ' || ch == '\t' {
    *i += 1;
    return "Whitespace".to_string();
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
      "Token`Comment".to_string()
    } else {
      "Token`Error`UnterminatedComment".to_string()
    };
  }

  if ch == '"' {
    *i += 1;
    while *i < chars.len() {
      match chars[*i] {
        '\\' => *i += 2,
        '"' => {
          *i += 1;
          return "String".to_string();
        }
        _ => *i += 1,
      }
    }
    *i = chars.len();
    return "Token`Error`UnterminatedString".to_string();
  }

  // A number starts at a digit, or at the dot of `.5`.
  if ch.is_ascii_digit()
    || (ch == '.' && matches!(chars.get(*i + 1), Some(c) if c.is_ascii_digit()))
  {
    return scan_number(chars, i).to_string();
  }

  // A named character that is not letterlike is a token in its own right,
  // written either as `\[Rule]` or as the character it stands for.
  if let Some((name, past)) = read_long_name(chars, *i) {
    if let Some(kind) = long_name_token(&name) {
      *i = past;
      return kind;
    }
  } else if !ch.is_ascii()
    && let Some(kind) = long_name_of(ch).and_then(long_name_token)
  {
    *i += 1;
    return kind;
  }

  // A symbol may carry a context (`` Foo`bar ``), may start with `$`, and
  // may spell any of its letters as a named character.
  if symbol_piece(chars, *i, true).is_some() {
    while let Some(width) = symbol_piece(chars, *i, false) {
      *i += width;
    }
    return "Symbol".to_string();
  }

  if ch == '_' {
    let mut unders = 0usize;
    while chars.get(*i) == Some(&'_') && unders < 3 {
      unders += 1;
      *i += 1;
    }
    // `_.` is the optional-default pattern, one token in its own right.
    if unders == 1 && chars.get(*i) == Some(&'.') {
      *i += 1;
      return "Token`UnderDot".to_string();
    }
    return match unders {
      1 => "Token`Under".to_string(),
      2 => "Token`UnderUnder".to_string(),
      _ => "Token`UnderUnderUnder".to_string(),
    };
  }

  // `#`, `##` — the slot marks. The number that may follow is a token of
  // its own, so `#1` is a slot mark and then a `1`.
  if ch == '#' {
    *i += 1;
    if chars.get(*i) == Some(&'#') {
      *i += 1;
      return "Token`HashHash".to_string();
    }
    return "Token`Hash".to_string();
  }

  // `%`, `%%`, `%%%` — the out marks. One is `Token`Percent`, any longer
  // run is `Token`PercentPercent`, and a following number is its own token.
  if ch == '%' {
    let start = *i;
    while chars.get(*i) == Some(&'%') {
      *i += 1;
    }
    return if *i - start == 1 {
      "Token`Percent".to_string()
    } else {
      "Token`PercentPercent".to_string()
    };
  }

  // `\!`, `\*`, … open linear syntax; `\.41` and `\041` spell a character
  // by its code, and are read as whatever that character is.
  if ch == '\\' {
    if let Some((_, kind)) = LINEAR_SYNTAX
      .iter()
      .find(|(c, _)| chars.get(*i + 1) == Some(c))
    {
      *i += 2;
      return kind.to_string();
    }
    if let Some((spelled, past)) = read_character_escape(chars, *i) {
      let mut escape = [0u8; 4];
      let text = &*spelled.encode_utf8(&mut escape);
      if let Some((_, name)) = OPERATORS.iter().find(|(op, _)| *op == text) {
        *i = past;
        return name.to_string();
      }
    }
  }

  for (text, name) in OPERATORS {
    let candidate: Vec<char> = text.chars().collect();
    if chars[*i..].starts_with(&candidate) {
      *i += candidate.len();
      return name.to_string();
    }
  }

  *i += 1;
  "Token`Error`UnhandledCharacter".to_string()
}

/// How many characters of a symbol sit at `i`, or `None` when what is there
/// cannot be part of one. `first` excludes the digits a symbol may not start
/// with.
fn symbol_piece(chars: &[char], i: usize, first: bool) -> Option<usize> {
  let ch = *chars.get(i)?;
  if ch.is_ascii_alphabetic() || ch == '$' || ch == '`' {
    return Some(1);
  }
  if !first && ch.is_ascii_digit() {
    return Some(1);
  }
  if let Some((name, past)) = read_long_name(chars, i) {
    return long_name_token(&name).is_none().then_some(past - i);
  }
  if let Some((spelled, past)) = read_character_escape(chars, i) {
    return spelled.is_alphanumeric().then_some(past - i);
  }
  if !ch.is_ascii() {
    // A character written as itself: letterlike if no name classifies it
    // as anything else.
    return long_name_of(ch)
      .and_then(long_name_token)
      .is_none()
      .then_some(1);
  }
  None
}

/// The `\[Name]` written at `i`, with the index just past its `]`.
fn read_long_name(chars: &[char], i: usize) -> Option<(String, usize)> {
  if chars.get(i) != Some(&'\\') || chars.get(i + 1) != Some(&'[') {
    return None;
  }
  let mut end = i + 2;
  while matches!(chars.get(end), Some(c) if c.is_ascii_alphanumeric()) {
    end += 1;
  }
  if chars.get(end) != Some(&']') || end == i + 2 {
    return None;
  }
  Some((chars[i + 2..end].iter().collect(), end + 1))
}

/// The character a `\.41` (hexadecimal) or `\041` (octal) escape spells,
/// with the index just past it.
fn read_character_escape(chars: &[char], i: usize) -> Option<(char, usize)> {
  if chars.get(i) != Some(&'\\') {
    return None;
  }
  let (radix, digits, start) = if chars.get(i + 1) == Some(&'.') {
    (16u32, 2usize, i + 2)
  } else {
    (8u32, 3usize, i + 1)
  };
  let spelled: String = chars.get(start..start + digits)?.iter().collect();
  if !spelled.chars().all(|c| c.is_digit(radix)) {
    return None;
  }
  let code = u32::from_str_radix(&spelled, radix).ok()?;
  Some((char::from_u32(code)?, start + digits))
}

/// The token a named character stands for, or `None` when it is letterlike
/// and so belongs to whatever symbol it was written in.
fn long_name_token(name: &str) -> Option<String> {
  if let Some((_, kind)) = LONG_NAME_TOKENS.iter().find(|(n, _)| *n == name) {
    return Some(kind.to_string());
  }
  if LONG_NAME_WHITESPACE.contains(&name) {
    return Some("Whitespace".to_string());
  }
  LONG_NAME_OPERATORS
    .contains(&name)
    .then(|| format!("Token`LongName`{name}"))
}

/// The name of the character `ch`, when it is one the classification above
/// knows — the Wolfram Language reads `\[Rule]` and `→` the same way.
fn long_name_of(ch: char) -> Option<&'static str> {
  let mut buffer = [0u8; 4];
  let text = &*ch.encode_utf8(&mut buffer);
  LONG_NAME_TOKENS
    .iter()
    .map(|(name, _)| *name)
    .chain(LONG_NAME_WHITESPACE.iter().copied())
    .chain(LONG_NAME_OPERATORS.iter().copied())
    .find(|name| crate::syntax::named_char_to_unicode(name) == Some(text))
}

/// Consume a number: digits, an optional fraction, an optional exponent, and
/// the Wolfram Language's `base^^digits` and `` `precision `` forms.
///
/// The kind reported is the kind of the value: `1*^-6` is the exact
/// `1/1000000`, so the Wolfram Language calls it a `Rational`.
fn scan_number(chars: &[char], i: &mut usize) -> &'static str {
  let mut is_real = false;
  while matches!(chars.get(*i), Some(c) if c.is_ascii_digit()) {
    *i += 1;
  }
  // `16^^ff` — a based number, whose digits may include letters, and which
  // is a real exactly when those digits carry a point.
  if chars.get(*i) == Some(&'^') && chars.get(*i + 1) == Some(&'^') {
    *i += 2;
    while matches!(chars.get(*i), Some(c) if c.is_alphanumeric() || *c == '.') {
      is_real |= chars[*i] == '.';
      *i += 1;
    }
    return if is_real { "Real" } else { "Integer" };
  }
  // A trailing dot makes a real of its own — `1.` — unless another dot
  // follows it, because `1..` is a repeated pattern.
  if chars.get(*i) == Some(&'.') && chars.get(*i + 1) != Some(&'.') {
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
  // `1*^6` — the scientific-notation operator. An exact mantissa keeps the
  // number exact: a positive exponent scales it to an integer, a negative
  // one to a rational.
  if chars.get(*i) == Some(&'*') && chars.get(*i + 1) == Some(&'^') {
    *i += 2;
    let negative = chars.get(*i) == Some(&'-');
    if matches!(chars.get(*i), Some('+' | '-')) {
      *i += 1;
    }
    while matches!(chars.get(*i), Some(c) if c.is_ascii_digit()) {
      *i += 1;
    }
    if !is_real {
      return if negative { "Rational" } else { "Integer" };
    }
  }
  if is_real { "Real" } else { "Integer" }
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
