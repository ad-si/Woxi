//! A whitespace formatter for Wolfram Language source.
//!
//! It normalizes the horizontal whitespace of a file — the spaces between
//! tokens and the indentation of each line — and nothing else. Where the
//! line breaks go is the author's decision: `f[x_] := Module[{y = 1}, y]`
//! written on one line stays on one line, and one written across six stays
//! across six. That keeps the formatter useful on the file an editor
//! reformats on every save, since it can never explode a carefully laid
//! out expression or fold one the author meant to keep open.
//!
//! Because the line structure is preserved, the formatted text has exactly
//! as many lines as the input, which is what lets the server answer a
//! range formatting request by simply reformatting the whole document and
//! reporting the edits of the lines the request asked about.
//!
//! Formatting a file must never change what it does. Every result is
//! therefore checked against its input before it is returned: the two must
//! tokenize to the same tokens, separated by the same line breaks, no two
//! neighbouring atoms may have been glued into one, and — for a file that
//! reads as an expression at all — the two must read as the same one. A
//! result that fails the check is dropped and the source returned
//! unchanged, so the worst a formatting bug can do is leave a file alone.

use super::analysis::{Token, TokenKind, scan};

/// How to indent, as an editor's `FormattingOptions` describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
  /// Width of one indentation level, in spaces.
  pub tab_size: usize,
  /// Indent with spaces rather than tabs.
  pub insert_spaces: bool,
}

impl Default for Options {
  fn default() -> Self {
    // Two spaces: the width the Wolfram Language's own front end uses for
    // a nested expression, and narrow enough for the deep nesting that
    // `Module[{…}, If[…, …]]` produces.
    Self {
      tab_size: 2,
      insert_spaces: true,
    }
  }
}

impl Options {
  /// One level of indentation.
  fn indent(&self) -> String {
    if self.insert_spaces {
      " ".repeat(self.tab_size)
    } else {
      "\t".to_string()
    }
  }
}

/// Format `source`, or return it unchanged if the result would not be the
/// same code (see the module documentation).
pub fn format(source: &str, options: &Options) -> String {
  let formatted = format_unchecked(source, options);
  if preserves_code(source, &formatted) {
    formatted
  } else {
    source.to_string()
  }
}

/// Format `source` without the safety check.
fn format_unchecked(source: &str, options: &Options) -> String {
  let tokens = scan(source);
  let indent = options.indent();
  // Whatever the file already ends its lines with stays: reformatting a
  // file must not rewrite every line of it on a Windows checkout.
  let newline = if source.contains("\r\n") {
    "\r\n"
  } else {
    "\n"
  };
  let mut out = String::new();
  let mut depth = 0_usize;
  let mut previous: Option<Token> = None;

  for (i, token) in tokens.iter().enumerate() {
    let gap = match previous {
      Some(previous) => &source[previous.end..token.start],
      None => &source[..token.start],
    };
    let line_breaks = gap.matches('\n').count();

    // A closing bracket ends the block it closes, so the line it starts is
    // indented like the line that opened it.
    if token.kind == TokenKind::Close {
      depth = depth.saturating_sub(1);
    }

    if line_breaks > 0 || previous.is_none() {
      for _ in 0..line_breaks {
        out.push_str(newline);
      }
      // An expression the previous line left unfinished — `x = 1 +` — is
      // indented one level further, the way its continuation is written.
      let continues = previous.is_some_and(|previous| {
        // A `,` or `;` ends what it separates: the next line starts a new
        // argument or statement at the depth it already had.
        previous.kind == TokenKind::Operator
          && !matches!(previous.text(source), ";" | ",")
      });
      let level = depth + usize::from(continues);
      for _ in 0..level {
        out.push_str(&indent);
      }
    } else if let Some(previous) = previous
      && needs_space(source, &tokens, i, previous)
    {
      out.push(' ');
    }

    out.push_str(token.text(source));
    if token.kind == TokenKind::Open {
      depth += 1;
    }
    previous = Some(*token);
  }

  // Whatever followed the last token: its line breaks are kept, its
  // trailing spaces are not.
  let trailer = match previous {
    Some(previous) => &source[previous.end..],
    None => source,
  };
  for _ in 0..trailer.matches('\n').count() {
    out.push_str(newline);
  }
  out
}

/// Whether a space belongs between the token at `index` and the token
/// before it, `previous`, when the two share a line.
fn needs_space(
  source: &str,
  tokens: &[Token],
  index: usize,
  previous: Token,
) -> bool {
  let token = tokens[index];
  let before = previous.text(source);
  let text = token.text(source);
  let is_atom = |kind| {
    matches!(kind, TokenKind::Symbol | TokenKind::Number | TokenKind::Str)
  };

  // Two adjacent atoms are a product (`2 x`) or a concatenation of
  // strings; gluing them would silently change the expression.
  if is_atom(previous.kind) && is_atom(token.kind) {
    return true;
  }

  // Brackets hug what they enclose: `f[x, y]`, `{1, 2}`, `<|a -> 1|>`.
  if previous.kind == TokenKind::Open || token.kind == TokenKind::Close {
    return false;
  }
  // A separator hugs what precedes it and is followed by a space.
  if text == "," || text == ";" {
    return false;
  }
  if before == "," || before == ";" {
    return true;
  }
  // A comment stands apart from the code around it.
  if previous.kind == TokenKind::Comment || token.kind == TokenKind::Comment {
    return true;
  }
  // An argument list or part specification hugs what it applies to:
  // `f[x]`, `list[[1]]`, `f[x][y]`, `#[[1]]`.
  if text == "["
    && (is_atom(previous.kind)
      || previous.kind == TokenKind::Close
      || before.starts_with('#')
      || before.starts_with('%'))
  {
    return false;
  }

  // `#"key"` is a slot named by a string and `# "key"` is a slot times
  // one, so here too the author's spacing stands.
  if before.starts_with('#') && token.kind == TokenKind::Str {
    return previous.end < token.start;
  }

  // A backslash introduces a named character (`\[Alpha]`) or continues a
  // line; anything may follow it but a space.
  if before == "\\" {
    return false;
  }

  // A repetition binds the pattern it repeats — but a number swallows a
  // dot that follows it, so `{1 ..}` has to keep its space.
  if text == ".." || text == "..." {
    return previous.kind == TokenKind::Number;
  }

  // Operators that bind their operands tightly enough to be written
  // without spaces: `x_Integer`, `x^2`, `f::usage`, `_?NumberQ`, `n_:1`,
  // `f''[x]`.
  const TIGHT: &[&str] = &["_", "^", "::", "?", ":", "..", "...", "'"];
  if TIGHT.contains(&text) {
    return false;
  }
  if TIGHT.contains(&before) {
    return match before {
      // A `_` only continues into a head or another underscore: the `+`
      // of `x_ + y_` is an ordinary operator and keeps its spaces.
      "_" => !(token.kind == TokenKind::Symbol || text == "_" || text == "."),
      // A derivative's primes belong to the function they differentiate,
      // and so does the argument list after them: `f''[x]`.
      "'" => !(text == "'" || text == "["),
      // `..` and `...` are postfix, so only what precedes them is tight.
      ".." | "..." => true,
      _ => false,
    };
  }

  // `!` is `Not` before its operand and `Factorial` after it.
  if text == "!" {
    return !is_postfix_position(source, previous);
  }
  if before == "!" {
    return is_postfix_position_at(source, tokens, index.wrapping_sub(2));
  }
  // `++` and `--` likewise increment before or after their variable.
  if text == "++" || text == "--" {
    return !is_postfix_position(source, previous);
  }
  if before == "++" || before == "--" {
    return is_postfix_position_at(source, tokens, index.wrapping_sub(2));
  }
  // A unary sign belongs to what it signs: `-x`, `a + -b`. Before a
  // number the author's spacing stands, because the reader folds the sign
  // into the literal only when the two are written together — `f[-1]`
  // passes an integer where `f[- 1]` applies `Minus` to one.
  if (before == "-" || before == "+")
    && !is_postfix_position_at(source, tokens, index.wrapping_sub(2))
  {
    return token.kind == TokenKind::Number && previous.end < token.start;
  }

  true
}

/// Whether `token` can end an expression, which is what decides whether an
/// operator following it is postfix (`i++`) or prefix (`++i`), and whether
/// a sign is binary (`a - b`) or unary (`a + -b`).
fn is_postfix_position(source: &str, token: Token) -> bool {
  if matches!(
    token.kind,
    TokenKind::Symbol | TokenKind::Number | TokenKind::Str | TokenKind::Close
  ) {
    return true;
  }
  // The operators that are an expression, or complete one, by themselves:
  // a blank (`_`), a repetition (`a ..`), a pure function's `&`, a
  // factorial, an increment, and the slots and output references.
  let text = token.text(source);
  matches!(text, "_" | ".." | "..." | "&" | "!" | "++" | "--" | "'")
    || text.starts_with('#')
    || text.starts_with('%')
}

/// [`is_postfix_position`] for the token at `index`, where no token at all
/// — the start of the file — is a prefix position.
fn is_postfix_position_at(
  source: &str,
  tokens: &[Token],
  index: usize,
) -> bool {
  // A comment says nothing about the expression it sits in, so look past
  // it to the code before it.
  let mut index = index;
  while tokens
    .get(index)
    .is_some_and(|token| token.kind == TokenKind::Comment)
  {
    let Some(next) = index.checked_sub(1) else {
      return false;
    };
    index = next;
  }
  tokens
    .get(index)
    .copied()
    .is_some_and(|token| is_postfix_position(source, token))
}

/// Whether `formatted` is the same code as `source`.
///
/// The two must produce the same tokens in the same order, separated by
/// the same number of line breaks — the Wolfram Language ends a statement
/// at a newline, so a line break gained or lost is a change of meaning —
/// and no pair of neighbouring atoms may have been run together.
fn preserves_code(source: &str, formatted: &str) -> bool {
  let original = scan(source);
  let result = scan(formatted);
  if original.len() != result.len() {
    return false;
  }
  let is_atom = |kind| {
    matches!(kind, TokenKind::Symbol | TokenKind::Number | TokenKind::Str)
  };
  for (i, (before, after)) in original.iter().zip(result.iter()).enumerate() {
    if before.kind != after.kind || before.text(source) != after.text(formatted)
    {
      return false;
    }
    let (original_gap, result_gap) = match i.checked_sub(1) {
      Some(previous) => (
        &source[original[previous].end..before.start],
        &formatted[result[previous].end..after.start],
      ),
      None => (&source[..before.start], &formatted[..after.start]),
    };
    if original_gap.matches('\n').count() != result_gap.matches('\n').count() {
      return false;
    }
    if result_gap.is_empty()
      && i > 0
      && is_atom(before.kind)
      && is_atom(original[i - 1].kind)
    {
      return false;
    }
  }
  let original_trailer = original.last().map_or(source, |t| &source[t.end..]);
  let result_trailer = result.last().map_or(formatted, |t| &formatted[t.end..]);
  if original_trailer.matches('\n').count()
    != result_trailer.matches('\n').count()
  {
    return false;
  }
  same_expression(source, formatted)
}

/// Whether `source` and `formatted` read as the very same expression.
///
/// The tokens can agree and the code still not mean the same thing: a
/// named character (`\[Alpha]`) is four tokens that only mean anything
/// with nothing between them, and `#"key"` is a named slot right up until
/// a space turns it into a slot times a string. Reading both texts and
/// comparing what they read as is the check that catches all of those at
/// once.
///
/// A file that does not read at all — the usual state of one being typed
/// — has nothing to compare, and the token and line checks above are what
/// stands for it there.
fn same_expression(source: &str, formatted: &str) -> bool {
  let read = |text: &str| {
    crate::syntax::string_to_expr(&crate::insert_statement_separators(
      text.trim(),
    ))
    .ok()
    // `Expr` has no equality of its own, and its `Debug` form is the
    // whole tree.
    .map(|expression| format!("{expression:?}"))
  };
  match (read(source), read(formatted)) {
    (Some(before), Some(after)) => before == after,
    // The file read before and does not now: whatever the formatter did,
    // it did not just move whitespace.
    (Some(_), None) => false,
    (None, _) => true,
  }
}
