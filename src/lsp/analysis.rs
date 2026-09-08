//! Static analysis of Wolfram Language source for the language server.
//!
//! The language server must answer questions about code that is still being
//! typed, i.e. code that usually does not parse. So instead of the pest
//! grammar it works on a forgiving token stream: strings and comments are
//! single tokens, everything else is classified just precisely enough to
//! find symbols and assignments, to highlight them ([`semantic_tokens`])
//! and to say which of them are misspellings of a built-in
//! ([`spelling_suggestion`]). The grammar is still used — but only to
//! report syntax errors (see [`diagnostics`]).

use crate::evaluator::functions::{
  ImplementationStatus, implementation_status,
};

/// What a [`Token`] is, at the granularity the server needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
  /// A symbol name, e.g. `Sin`, `myVar`, `` System`Sin ``.
  Symbol,
  /// A number, e.g. `1`, `2.5`, `` 1.5`20 ``, `2^^1011`, `1.*^-6`.
  Number,
  /// A string literal, including its quotes.
  Str,
  /// An opening bracket: `(`, `[`, `{` or `<|`.
  Open,
  /// A closing bracket: `)`, `]`, `}` or `|>`.
  Close,
  /// Anything else: operators, `,`, `;`, `#`, `&`, …
  Operator,
  /// A `(* … *)` comment, including its delimiters.
  ///
  /// [`tokenize`] drops these — everything that reasons about the code
  /// itself is easier to write without them — so only [`scan`], and with
  /// it the highlighter and the formatter, ever sees one.
  Comment,
}

/// A lexical token, addressed by byte offsets into the source it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
  pub kind: TokenKind,
  pub start: usize,
  pub end: usize,
}

impl Token {
  /// The source text this token covers.
  pub fn text<'a>(&self, source: &'a str) -> &'a str {
    &source[self.start..self.end]
  }
}

/// Multi-character operators, longest first so that greedy matching never
/// splits e.g. `===` into `==` and `=`.
const OPERATORS: &[&str] = &[
  "^:=", "//@", "//.", "//=", "|->", "===", "=!=", "@@@", "...", ">>>", ";;",
  ":=", "::", "=.", "==", "<=", ">=", "!=", "->", ":>", "/.", "/;", "/@", "/:",
  "//", "@@", "@*", "**", "++", "--", "+=", "-=", "*=", "/=", "^=", "&&", "||",
  "<>", "~~", "..", "<<", ">>", "<|", "|>", "*^", "^^",
];

fn is_symbol_start(c: char) -> bool {
  c.is_alphabetic() || c == '$'
}

fn is_symbol_continuation(c: char) -> bool {
  c.is_alphanumeric() || c == '$' || c == '`'
}

/// Split `source` into tokens, dropping whitespace but keeping
/// `(* comments *)`.
///
/// Never fails: unterminated strings and comments simply run to the end of
/// the input, which is exactly the state a file is in while being typed.
pub fn scan(source: &str) -> Vec<Token> {
  let chars: Vec<(usize, char)> = source.char_indices().collect();
  let offset_at = |i: usize| chars.get(i).map_or(source.len(), |&(o, _)| o);
  let char_at = |i: usize| chars.get(i).map(|&(_, c)| c);

  let mut tokens = Vec::new();
  let mut i = 0;
  while i < chars.len() {
    let (start, c) = chars[i];

    if c.is_whitespace() {
      i += 1;
      continue;
    }

    // `(* … *)` comments, which nest in the Wolfram Language.
    if c == '(' && char_at(i + 1) == Some('*') {
      let mut depth = 1_usize;
      i += 2;
      while i < chars.len() && depth > 0 {
        match (char_at(i), char_at(i + 1)) {
          (Some('('), Some('*')) => {
            depth += 1;
            i += 2;
          }
          (Some('*'), Some(')')) => {
            depth -= 1;
            i += 2;
          }
          _ => i += 1,
        }
      }
      tokens.push(Token {
        kind: TokenKind::Comment,
        start,
        end: offset_at(i),
      });
      continue;
    }

    if c == '"' {
      i += 1;
      while i < chars.len() {
        match char_at(i) {
          Some('\\') => i += 2,
          Some('"') => {
            i += 1;
            break;
          }
          _ => i += 1,
        }
      }
      i = i.min(chars.len());
      tokens.push(Token {
        kind: TokenKind::Str,
        start,
        end: offset_at(i),
      });
      continue;
    }

    // A number, which may start with its decimal point: `.5` is `0.5`.
    if c.is_ascii_digit()
      || (c == '.' && char_at(i + 1).is_some_and(|c| c.is_ascii_digit()))
    {
      while char_at(i).is_some_and(|c| c.is_ascii_digit() || c == '.') {
        i += 1;
      }
      // Precision/accuracy marks (`` 1.5`20 ``), base notation
      // (`2^^1011`) and scientific notation (`1.*^-6`). Nothing else may
      // be glued to a number: `2x` is a product, not one token.
      loop {
        match (char_at(i), char_at(i + 1)) {
          (Some('`'), _) => {
            i += 1;
            while char_at(i).is_some_and(|c| c.is_ascii_digit() || c == '.') {
              i += 1;
            }
          }
          (Some('^'), Some('^')) => {
            i += 2;
            while char_at(i)
              .is_some_and(|c| c.is_ascii_alphanumeric() || c == '.')
            {
              i += 1;
            }
          }
          (Some('*'), Some('^')) => {
            i += 2;
            if char_at(i).is_some_and(|c| c == '+' || c == '-') {
              i += 1;
            }
            while char_at(i).is_some_and(|c| c.is_ascii_digit()) {
              i += 1;
            }
          }
          _ => break,
        }
      }
      tokens.push(Token {
        kind: TokenKind::Number,
        start,
        end: offset_at(i),
      });
      continue;
    }

    if is_symbol_start(c) {
      i += 1;
      while char_at(i).is_some_and(is_symbol_continuation) {
        i += 1;
      }
      tokens.push(Token {
        kind: TokenKind::Symbol,
        start,
        end: offset_at(i),
      });
      continue;
    }

    // Slots (`#`, `##2`, `#name`) and output references (`%`, `%%%`,
    // `%3`) are single tokens: their parts only mean anything glued
    // together, and the name of `#name` is not a symbol of its own.
    if c == '#' || c == '%' {
      while char_at(i) == Some(c) {
        i += 1;
      }
      if c == '#' {
        while char_at(i).is_some_and(is_symbol_continuation) {
          i += 1;
        }
      } else {
        while char_at(i).is_some_and(|c| c.is_ascii_digit()) {
          i += 1;
        }
      }
      tokens.push(Token {
        kind: TokenKind::Operator,
        start,
        end: offset_at(i),
      });
      continue;
    }

    let rest = &source[start..];
    let matched = OPERATORS.iter().find(|op| rest.starts_with(**op));
    let text = match matched {
      Some(op) => *op,
      None => &source[start..offset_at(i + 1)],
    };
    let kind = match text {
      "(" | "[" | "{" | "<|" => TokenKind::Open,
      ")" | "]" | "}" | "|>" => TokenKind::Close,
      _ => TokenKind::Operator,
    };
    let end = start + text.len();
    tokens.push(Token { kind, start, end });
    while i < chars.len() && offset_at(i) < end {
      i += 1;
    }
  }

  tokens
}

/// Split `source` into tokens, dropping whitespace and `(* comments *)`.
///
/// This is [`scan`] without the comments: a comment can appear between any
/// two tokens, so code that looks at what follows what — finding
/// assignments, deciding where a statement ends — would have to skip them
/// at every step.
pub fn tokenize(source: &str) -> Vec<Token> {
  scan(source)
    .into_iter()
    .filter(|token| token.kind != TokenKind::Comment)
    .collect()
}

/// Whether a definition assigns to a symbol on its own (`x = 1`) or to a
/// symbol applied to arguments (`f[x_] := …`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionKind {
  Function,
  Variable,
}

/// An assignment found in a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
  pub name: String,
  pub kind: DefinitionKind,
  /// The assignment operator used, e.g. `=`, `:=`, `^:=`.
  pub operator: String,
  /// Byte range of the defined symbol itself.
  pub name_range: (usize, usize),
  /// Byte range of the whole definition, e.g. all of `f[x_] := x^2`.
  pub full_range: (usize, usize),
  /// Bracket nesting depth of the assignment. Only depth 0 definitions are
  /// top-level; deeper ones are `Module`/`Block` locals and the like.
  pub depth: usize,
}

/// The operators that bind a value to the expression on their left.
const ASSIGNMENT_OPERATORS: &[&str] = &["=", ":=", "^=", "^:="];

/// Find every assignment in `source`.
///
/// Definitions inside `Module[{i = 1}, …]` are reported too (with a nonzero
/// `depth`), because "go to definition" on a local variable should land on
/// its initialiser just as much as on a top-level one.
pub fn find_definitions(source: &str, tokens: &[Token]) -> Vec<Definition> {
  let mut definitions = Vec::new();
  let mut depths = vec![0_usize; tokens.len()];
  let mut depth = 0_usize;
  for (i, token) in tokens.iter().enumerate() {
    if token.kind == TokenKind::Close {
      depth = depth.saturating_sub(1);
    }
    depths[i] = depth;
    if token.kind == TokenKind::Open {
      depth += 1;
    }
  }

  for (i, token) in tokens.iter().enumerate() {
    if token.kind != TokenKind::Symbol {
      continue;
    }
    let mut j = i + 1;
    let mut kind = DefinitionKind::Variable;
    // Any number of argument lists: `f[x_] := …` and `f[x_][y_] := …`.
    while tokens.get(j).is_some_and(|t| t.text(source) == "[") {
      let Some(close) = matching_bracket(tokens, j) else {
        break;
      };
      j = close + 1;
      kind = DefinitionKind::Function;
    }
    let Some(operator) = tokens.get(j) else {
      continue;
    };
    let operator_text = operator.text(source);
    if !ASSIGNMENT_OPERATORS.contains(&operator_text) {
      continue;
    }
    let full_end = statement_end(source, tokens, j + 1).unwrap_or(operator.end);
    definitions.push(Definition {
      name: token.text(source).to_string(),
      kind,
      operator: operator_text.to_string(),
      name_range: (token.start, token.end),
      full_range: (token.start, full_end),
      depth: depths[i],
    });
  }

  definitions
}

/// Index of the token closing the bracket opened at `open`.
fn matching_bracket(tokens: &[Token], open: usize) -> Option<usize> {
  let mut depth = 0_usize;
  for (i, token) in tokens.iter().enumerate().skip(open) {
    match token.kind {
      TokenKind::Open => depth += 1,
      TokenKind::Close => {
        // Saturating, so that unbalanced input — which is what a file
        // being typed usually is — cannot panic the server.
        depth = depth.saturating_sub(1);
        if depth == 0 {
          return Some(i);
        }
      }
      _ => {}
    }
  }
  None
}

/// Byte offset where the statement starting at token `from` ends.
///
/// A statement ends at a top-level `;`, or at a line break that is not a
/// continuation (neither the line before nor the line after starts or ends
/// with an operator), or at the end of the input.
fn statement_end(source: &str, tokens: &[Token], from: usize) -> Option<usize> {
  let mut depth = 0_usize;
  let mut last_end = None;
  for i in from..tokens.len() {
    let token = tokens[i];
    if i > from && depth == 0 {
      let previous = tokens[i - 1];
      let between = &source[previous.end..token.start];
      let continues =
        matches!(previous.kind, TokenKind::Operator | TokenKind::Open)
          || matches!(token.kind, TokenKind::Operator | TokenKind::Close);
      if between.contains('\n') && !continues {
        break;
      }
    }
    match token.kind {
      TokenKind::Open => depth += 1,
      TokenKind::Close => depth = depth.saturating_sub(1),
      _ => {
        if depth == 0 && token.text(source) == ";" {
          break;
        }
      }
    }
    last_end = Some(token.end);
  }
  last_end
}

/// The symbol token containing `offset`, if any.
///
/// The end of a token counts as inside it, so a cursor placed directly
/// after a name (where editors most often put it) still resolves.
pub fn symbol_at(tokens: &[Token], offset: usize) -> Option<Token> {
  tokens
    .iter()
    .find(|t| {
      t.kind == TokenKind::Symbol && t.start <= offset && offset <= t.end
    })
    .copied()
}

/// Every occurrence of the symbol `name` in the token stream.
pub fn occurrences(source: &str, tokens: &[Token], name: &str) -> Vec<Token> {
  tokens
    .iter()
    .filter(|t| t.kind == TokenKind::Symbol && t.text(source) == name)
    .copied()
    .collect()
}

/// Severity of a [`Diagnostic`], mirroring the LSP enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
  Error = 1,
  Warning = 2,
  Information = 3,
}

/// A problem found in a document, addressed by byte offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
  pub range: (usize, usize),
  pub severity: Severity,
  pub code: &'static str,
  pub message: String,
}

/// Analyse `source` and report syntax errors and unsupported built-ins.
///
/// A syntax error is fatal for the whole document, so it is reported alone:
/// past the error the token stream is not trustworthy enough to say
/// anything about which symbols are actually being called.
pub fn diagnostics(
  source: &str,
  tokens: &[Token],
  definitions: &[Definition],
) -> Vec<Diagnostic> {
  if source.trim().is_empty() {
    return Vec::new();
  }
  if let Some(error) = syntax_error(source) {
    return vec![error];
  }
  let mut diagnostics = unsupported_builtins(source, tokens, definitions);
  diagnostics.extend(misspelled_symbols(source, tokens, definitions));
  diagnostics.sort_by_key(|diagnostic| diagnostic.range);
  diagnostics
}

/// Run the Wolfram grammar over `source` and turn a parse failure into a
/// diagnostic.
///
/// Only input that no path of the interpreter accepts is reported. The
/// interpreter preprocesses its input before parsing it — line
/// continuations are joined and statement separators are inserted — and it
/// falls back to the single-expression reader, so plenty of code the bare
/// grammar rejects still runs. Reporting that as an error would put a red
/// squiggle under working scripts.
fn syntax_error(source: &str) -> Option<Diagnostic> {
  use pest::error::InputLocation;

  // The raw parse is what carries a usable location: the preprocessed
  // text has different offsets than the document the editor shows.
  let error = crate::parse(source).err()?;

  let normalized = if source.contains('\r') {
    source.replace("\r\n", "\n").replace('\r', "\n")
  } else {
    source.to_string()
  };
  let trimmed = normalized.trim();
  if crate::parse(&crate::insert_statement_separators(trimmed)).is_ok()
    || crate::syntax::string_to_expr(trimmed).is_ok()
  {
    return None;
  }
  let (start, end) = match error.location {
    InputLocation::Pos(pos) => {
      let pos = pos.min(source.len());
      // Highlight the rest of the offending line; a zero-width range at
      // the end of the file is what an incomplete expression deserves.
      let end = source[pos..]
        .find('\n')
        .map_or(source.len(), |offset| pos + offset);
      (pos, end)
    }
    InputLocation::Span((start, end)) => {
      (start.min(source.len()), end.min(source.len()))
    }
  };
  Some(Diagnostic {
    range: (start, end),
    severity: Severity::Error,
    code: "syntax",
    message: format!("Syntax error: {}", error.variant.message()),
  })
}

/// Warn about calls to Wolfram Language symbols that Woxi cannot evaluate.
///
/// This is the one thing an editor cannot tell the user about a Woxi
/// script and Woxi can: the code is valid Wolfram Language, but this
/// interpreter will not run it.
fn unsupported_builtins(
  source: &str,
  tokens: &[Token],
  definitions: &[Definition],
) -> Vec<Diagnostic> {
  let mut diagnostics = Vec::new();
  for (i, token) in tokens.iter().enumerate() {
    if token.kind != TokenKind::Symbol {
      continue;
    }
    let name = token.text(source);
    // A symbol the document defines itself shadows the built-in.
    if definitions.iter().any(|d| d.name == name) {
      continue;
    }
    // `f::usage` names a message, not the symbol `usage`.
    if tokens
      .get(i.wrapping_sub(1))
      .is_some_and(|t| t.text(source) == "::")
    {
      continue;
    }
    let message = match implementation_status(name) {
      Some(ImplementationStatus::NotImplemented) => {
        format!("`{name}` is not implemented in Woxi yet")
      }
      Some(ImplementationStatus::NotPlanned) => {
        format!("`{name}` is not supported by Woxi")
      }
      _ => continue,
    };
    diagnostics.push(Diagnostic {
      range: (token.start, token.end),
      severity: Severity::Warning,
      code: "unsupported",
      message,
    });
  }
  diagnostics
}

/// Maps byte offsets to line/character positions and back.
///
/// LSP counts characters in UTF-16 code units by default, which is neither
/// the byte offsets the analysis uses nor Rust's `char` count, so every
/// conversion goes through here.
#[derive(Debug, Clone)]
pub struct LineIndex {
  /// Byte offset of the start of each line.
  line_starts: Vec<usize>,
}

impl LineIndex {
  pub fn new(source: &str) -> Self {
    let mut line_starts = vec![0];
    line_starts.extend(
      source
        .char_indices()
        .filter(|&(_, c)| c == '\n')
        .map(|(i, c)| i + c.len_utf8()),
    );
    Self { line_starts }
  }

  /// The zero-based line and UTF-16 character of a byte `offset`.
  pub fn position(&self, source: &str, offset: usize) -> (u32, u32) {
    let offset = offset.min(source.len());
    let line = self
      .line_starts
      .partition_point(|&start| start <= offset)
      .saturating_sub(1);
    let line_start = self.line_starts[line];
    let character = source[line_start..offset]
      .chars()
      .map(char::len_utf16)
      .sum::<usize>();
    (line as u32, character as u32)
  }

  /// The byte offset of a zero-based line and UTF-16 character.
  ///
  /// Positions past the end of a line (or of the document) clamp to it, as
  /// the specification requires.
  pub fn offset(&self, source: &str, line: u32, character: u32) -> usize {
    let Some(&line_start) = self.line_starts.get(line as usize) else {
      return source.len();
    };
    let line_end = self
      .line_starts
      .get(line as usize + 1)
      .map_or(source.len(), |&next| next);
    // Exclude the line terminator, so a character past the end of a line
    // clamps to that line's last character instead of the next line.
    let line_end = line_start
      + source[line_start..line_end]
        .trim_end_matches(['\n', '\r'])
        .len();
    let mut remaining = character as usize;
    let mut offset = line_start;
    for c in source[line_start..line_end].chars() {
      if remaining < c.len_utf16() {
        break;
      }
      remaining -= c.len_utf16();
      offset += c.len_utf8();
    }
    offset.min(line_end)
  }
}

/// The semantic token types this server reports, in the order the LSP
/// legend lists them: a token's type is sent as an index into this table.
pub const SEMANTIC_TOKEN_TYPES: &[&str] = &[
  "comment",
  "string",
  "number",
  "operator",
  "function",
  "variable",
  "parameter",
  "property",
];

/// The semantic token modifiers this server reports, in legend order: a
/// token's modifiers are sent as a bit set over this table.
pub const SEMANTIC_TOKEN_MODIFIERS: &[&str] =
  &["declaration", "defaultLibrary"];

/// Bit of `declaration` in a [`SemanticToken`]'s modifier set: the symbol
/// is being defined here rather than used.
pub const MODIFIER_DECLARATION: u32 = 1;
/// Bit of `defaultLibrary` in a [`SemanticToken`]'s modifier set: the
/// symbol is a built-in rather than something the file introduces.
pub const MODIFIER_DEFAULT_LIBRARY: u32 = 2;

/// One highlighted piece of source, addressed by byte offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticToken {
  pub range: (usize, usize),
  /// Index into [`SEMANTIC_TOKEN_TYPES`].
  pub token_type: u32,
  /// Bit set over [`SEMANTIC_TOKEN_MODIFIERS`].
  pub modifiers: u32,
}

/// Index of `name` in [`SEMANTIC_TOKEN_TYPES`].
fn semantic_type(name: &str) -> u32 {
  SEMANTIC_TOKEN_TYPES
    .iter()
    .position(|type_name| *type_name == name)
    .expect("semantic token type is part of the legend") as u32
}

/// Classify every token of `source` for semantic highlighting.
///
/// The editor's own grammar colors the file by its shapes; this says what
/// the shapes *mean* — which names are built-ins, which the file defines
/// itself, and which are only the parameters of a definition.
///
/// Tokens never span a line: an editor may reject a multi-line one, so a
/// comment or string that covers several lines is reported per line.
pub fn semantic_tokens(
  source: &str,
  definitions: &[Definition],
) -> Vec<SemanticToken> {
  let tokens = scan(source);
  let parameters = parameter_bindings(source, &tokens, definitions);
  let mut semantic = Vec::new();
  let mut previous: Option<Token> = None;
  for (i, token) in tokens.iter().enumerate() {
    let (token_type, modifiers) = match token.kind {
      TokenKind::Comment => (semantic_type("comment"), 0),
      TokenKind::Str => (semantic_type("string"), 0),
      TokenKind::Number => (semantic_type("number"), 0),
      TokenKind::Open | TokenKind::Close | TokenKind::Operator => {
        // A slot is the argument of a pure function, so it is highlighted
        // like the parameter of a named one: `#`, `#1`, `#name`.
        if token.text(source).starts_with('#') {
          (semantic_type("parameter"), 0)
        } else {
          (semantic_type("operator"), 0)
        }
      }
      TokenKind::Symbol => {
        classify_symbol(source, &tokens, i, previous, definitions, &parameters)
      }
    };
    for range in split_at_line_breaks(source, (token.start, token.end)) {
      semantic.push(SemanticToken {
        range,
        token_type,
        modifiers,
      });
    }
    previous = Some(*token);
  }
  semantic
}

/// The semantic type and modifiers of the symbol token at `index`.
fn classify_symbol(
  source: &str,
  tokens: &[Token],
  index: usize,
  previous: Option<Token>,
  definitions: &[Definition],
  parameters: &[(String, (usize, usize))],
) -> (u32, u32) {
  let token = tokens[index];
  let name = token.text(source);

  // `f::usage` names a message of `f`, not a symbol of its own.
  if previous.is_some_and(|t| t.text(source) == "::") {
    return (semantic_type("property"), 0);
  }

  // A name bound by a pattern (`f[x_] := x^2`) is a parameter wherever it
  // appears inside the definition that binds it.
  if parameters.iter().any(|(parameter, (start, end))| {
    parameter == name && *start <= token.start && token.end <= *end
  }) {
    let declares = tokens
      .get(index + 1)
      .is_some_and(|next| next.start == token.end && next.text(source) == "_");
    let modifiers = if declares { MODIFIER_DECLARATION } else { 0 };
    return (semantic_type("parameter"), modifiers);
  }

  // A symbol the file defines itself shadows any built-in of that name.
  if let Some(definition) = definitions
    .iter()
    .find(|definition| definition.name == name)
  {
    let token_type = match definition.kind {
      DefinitionKind::Function => semantic_type("function"),
      DefinitionKind::Variable => semantic_type("variable"),
    };
    let declares = definitions
      .iter()
      .any(|definition| definition.name_range == (token.start, token.end));
    let modifiers = if declares { MODIFIER_DECLARATION } else { 0 };
    return (token_type, modifiers);
  }

  if implementation_status(name).is_some() {
    return (semantic_type("function"), MODIFIER_DEFAULT_LIBRARY);
  }
  (semantic_type("variable"), 0)
}

/// The names bound by the patterns of each definition, paired with the
/// range of the definition binding them.
///
/// `f[x_] := x^2` binds `x` for the whole of the definition, so both the
/// `x_` and the `x` in the body are parameters rather than free symbols.
fn parameter_bindings(
  source: &str,
  tokens: &[Token],
  definitions: &[Definition],
) -> Vec<(String, (usize, usize))> {
  let mut bindings: Vec<(String, (usize, usize))> = Vec::new();
  for definition in definitions {
    if definition.kind != DefinitionKind::Function {
      continue;
    }
    let (start, end) = definition.full_range;
    // Tokens are ordered, so the definition's own tokens are a slice of
    // them: a file of many definitions must not cost one pass over the
    // whole token stream each.
    let first = tokens.partition_point(|token| token.start < start);
    for (i, token) in tokens.iter().enumerate().skip(first) {
      if token.end > end {
        break;
      }
      if token.kind != TokenKind::Symbol {
        continue;
      }
      // Only a `_` glued to the name binds it: `x_`, `x__`, `x_Integer`.
      let binds = tokens.get(i + 1).is_some_and(|next| {
        next.start == token.end && next.text(source) == "_"
      });
      let name = token.text(source).to_string();
      if binds && !bindings.contains(&(name.clone(), (start, end))) {
        bindings.push((name, (start, end)));
      }
    }
  }
  bindings
}

/// Split a byte range at the line breaks it contains, dropping the breaks
/// themselves and any line the range covers no characters of.
fn split_at_line_breaks(
  source: &str,
  (start, end): (usize, usize),
) -> Vec<(usize, usize)> {
  let text = &source[start..end];
  if !text.contains('\n') {
    return vec![(start, end)];
  }
  let mut ranges = Vec::new();
  let mut line_start = start;
  for (offset, c) in text.char_indices() {
    if c != '\n' {
      continue;
    }
    let mut line_end = start + offset;
    if source[line_start..line_end].ends_with('\r') {
      line_end -= 1;
    }
    if line_end > line_start {
      ranges.push((line_start, line_end));
    }
    line_start = start + offset + 1;
  }
  if end > line_start {
    ranges.push((line_start, end));
  }
  ranges
}

/// Shortest symbol name a spelling suggestion is offered for. Below it
/// every name is close to some built-in, so a suggestion says nothing.
const MIN_SPELLCHECK_LENGTH: usize = 4;

/// Number of edits a name may be away from a built-in and still be taken
/// for a misspelling of it. Longer names get a wider radius: a typo is
/// about as likely in a long name as in a short one, but an unrelated name
/// is far less likely to land within two edits of a long built-in.
fn spelling_tolerance(length: usize) -> usize {
  if length >= 8 { 2 } else { 1 }
}

/// Every `System`` symbol name, sorted, held once: the spell checker walks
/// the whole list for each name it checks, and rebuilding it there would
/// cost more than the comparisons do.
static BUILTIN_NAMES: std::sync::LazyLock<Vec<&'static str>> =
  std::sync::LazyLock::new(
    crate::evaluator::functions::all_builtin_symbol_names,
  );

/// The built-in symbol `name` was most likely meant to be, if any.
///
/// Only capitalized names are checked: every `System`` symbol starts with
/// an uppercase letter, and the Wolfram Language's own convention reserves
/// lowercase names for the user's variables — so a lowercase `list` is a
/// variable, not a misspelling of `List`.
pub fn spelling_suggestion(name: &str) -> Option<&'static str> {
  if name.len() < MIN_SPELLCHECK_LENGTH
    || !name.chars().all(|c| c.is_ascii_alphanumeric())
    || !name.starts_with(|c: char| c.is_ascii_uppercase())
    // A name ending in a digit is a numbered one of the author's own —
    // `Option1`, `Option2` — not a slip of the finger.
    || name.ends_with(|c: char| c.is_ascii_digit())
    || crate::evaluator::functions::is_builtin_symbol(name)
  {
    return None;
  }
  let tolerance = spelling_tolerance(name.len());
  let mut best: Option<(usize, &'static str)> = None;
  for candidate in BUILTIN_NAMES.iter().copied() {
    // Only names that could plausibly be the one meant are measured: this
    // runs on every keystroke against every `System`` symbol, and the
    // distance itself is far more expensive than the comparison. A name
    // too different in length cannot be within `tolerance` edits at all,
    // and one that starts with another letter has nothing to have been
    // recognized by.
    if candidate.len().abs_diff(name.len()) > tolerance
      || !candidate.as_bytes()[0].eq_ignore_ascii_case(&name.as_bytes()[0])
    {
      continue;
    }
    // A trailing `Q` marks a predicate and a trailing `s` a plural or a
    // domain — `BooleanQ`, `Booleans`, `Integers` — so a name that is a
    // built-in without one of them is a related name of the author's,
    // not a typo of it.
    if ["Q", "s"].iter().any(|suffix| {
      candidate.strip_suffix(suffix) == Some(name)
        || name.strip_suffix(suffix) == Some(candidate)
    }) {
      continue;
    }
    let distance = edit_distance(name, candidate);
    if distance > tolerance {
      continue;
    }
    // The list is sorted, so an equally close candidate never displaces
    // the first one and the suggestion is deterministic.
    if best.is_none_or(|(best_distance, _)| distance < best_distance) {
      best = Some((distance, candidate));
    }
  }
  best.map(|(_, candidate)| candidate)
}

/// The distance between two symbol names, as the interpreter's own
/// `DamerauLevenshteinDistance` computes it: the number of insertions,
/// deletions, substitutions and transpositions between them. Swapping two
/// letters — `Lenght` for `Length` — is the typo a spell checker has to
/// catch, and plain Levenshtein counts it as two edits rather than one.
fn edit_distance(a: &str, b: &str) -> usize {
  let distance =
    crate::functions::string_ast::damerau_levenshtein_distance_ast(&[
      crate::syntax::Expr::String(a.to_string()),
      crate::syntax::Expr::String(b.to_string()),
    ]);
  match distance {
    Ok(crate::syntax::Expr::Integer(distance)) => {
      distance.unsigned_abs() as usize
    }
    // The distance of two strings always evaluates; a name that somehow
    // does not is simply not suggested for.
    _ => usize::MAX,
  }
}

/// Point out symbols that are one typo away from a built-in.
///
/// An undefined symbol is not an error in the Wolfram Language — it stands
/// for itself — so this only fires where a built-in is close enough that
/// the name was almost certainly meant to be it, and a code action offers
/// the correction. Symbols the file defines or binds as a pattern are
/// deliberate names and left alone.
fn misspelled_symbols(
  source: &str,
  tokens: &[Token],
  definitions: &[Definition],
) -> Vec<Diagnostic> {
  let parameters = parameter_bindings(source, tokens, definitions);
  // The same name usually occurs more than once, and looking a name up is
  // the expensive part of this.
  let mut suggestions: Vec<(&str, Option<&'static str>)> = Vec::new();
  let mut diagnostics = Vec::new();
  for (i, token) in tokens.iter().enumerate() {
    if token.kind != TokenKind::Symbol {
      continue;
    }
    let name = token.text(source);
    if definitions.iter().any(|definition| definition.name == name)
      || parameters.iter().any(|(parameter, _)| parameter == name)
    {
      continue;
    }
    // `f::usage` names a message, not the symbol `usage`.
    if tokens
      .get(i.wrapping_sub(1))
      .is_some_and(|t| t.text(source) == "::")
    {
      continue;
    }
    let suggestion = if let Some((_, suggestion)) =
      suggestions.iter().find(|(seen, _)| *seen == name)
    {
      *suggestion
    } else {
      let suggestion = spelling_suggestion(name);
      suggestions.push((name, suggestion));
      suggestion
    };
    let Some(suggestion) = suggestion else {
      continue;
    };
    diagnostics.push(Diagnostic {
      range: (token.start, token.end),
      severity: Severity::Information,
      code: "spelling",
      message: format!(
        "`{name}` is not a known symbol; did you mean `{suggestion}`?"
      ),
    });
  }
  diagnostics
}
