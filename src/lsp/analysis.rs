//! Static analysis of Wolfram Language source for the language server.
//!
//! The language server must answer questions about code that is still being
//! typed, i.e. code that usually does not parse. So instead of the pest
//! grammar it works on a forgiving token stream: comments and strings are
//! skipped, everything else is classified just precisely enough to find
//! symbols and assignments. The grammar is still used — but only to report
//! syntax errors (see [`diagnostics`]).

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
  "^:=", "//@", "//.", "|->", "===", "=!=", "@@@", "...", ";;", ":=", "::",
  "=.", "==", "<=", ">=", "!=", "->", ":>", "/.", "/;", "/@", "//", "@@", "@*",
  "**", "++", "--", "+=", "-=", "*=", "/=", "^=", "&&", "||", "<>", "~~", "..",
  "<|", "|>", "*^", "^^",
];

fn is_symbol_start(c: char) -> bool {
  c.is_alphabetic() || c == '$'
}

fn is_symbol_continuation(c: char) -> bool {
  c.is_alphanumeric() || c == '$' || c == '`'
}

/// Split `source` into tokens, dropping whitespace and `(* comments *)`.
///
/// Never fails: unterminated strings and comments simply run to the end of
/// the input, which is exactly the state a file is in while being typed.
pub fn tokenize(source: &str) -> Vec<Token> {
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

    if c.is_ascii_digit() {
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
  unsupported_builtins(source, tokens, definitions)
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
