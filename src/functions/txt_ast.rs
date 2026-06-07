use crate::syntax::Expr;

/// Strip a single trailing newline (either `"\n"` or `"\r\n"`) so that
/// `Import["file.txt"]` produces the same `StringLength` as wolframscript.
pub fn strip_trailing_newline(mut s: String) -> String {
  if s.ends_with('\n') {
    s.pop();
    if s.ends_with('\r') {
      s.pop();
    }
  }
  s
}

/// Split a string into lines without the final line separator. All trailing
/// empty lines are dropped so the result matches wolframscript's
/// `Import[file, "Lines"]`, which also strips them.
fn split_lines(s: &str) -> Vec<&str> {
  let mut out: Vec<&str> = s
    .split('\n')
    .map(|line| line.strip_suffix('\r').unwrap_or(line))
    .collect();
  while out.last() == Some(&"") {
    out.pop();
  }
  out
}

/// Auto-convert a token to Integer / BigInteger / Real / String, matching
/// wolframscript's behaviour for the `"Data"` element of a plain-text file.
fn token_to_expr(tok: &str) -> Expr {
  if tok.is_empty() {
    return Expr::String(String::new());
  }
  if let Ok(n) = tok.parse::<i128>() {
    return Expr::Integer(n);
  }
  // Integer too large for i128 — try arbitrary precision before falling
  // back to a Real/String, so that tokens like the 122-digit rows in
  // the Carl-Gauss `prime.txt` round-trip as Integer.
  if !tok.contains('.') && !tok.contains('e') && !tok.contains('E') {
    use std::str::FromStr;
    if let Ok(big) = num_bigint::BigInt::from_str(tok) {
      return Expr::BigInteger(big);
    }
  }
  if let Ok(f) = tok.parse::<f64>()
    && (tok.contains('.') || tok.contains('e') || tok.contains('E'))
  {
    return Expr::Real(f);
  }
  Expr::String(tok.to_string())
}

/// Dispatch an `Import[..., element]` call for a plain-text source.
/// Returns `None` if the element name is not one of the supported ones, so
/// the caller can fall back to returning the expression unevaluated.
///
/// Supported elements (matching wolframscript):
/// - `"Plaintext"` / `"String"` / `"Text"` → full file content (trailing `\n` stripped).
/// - `"Lines"` → list of line strings.
/// - `"Words"` → flat list of whitespace-separated tokens.
/// - `"Data"`  → if every non-empty line has the same whitespace-token
///               count, a list-of-lists with numeric tokens auto-converted;
///               otherwise a list of line strings (same as `"Lines"`).
/// - `"Table"` → list-of-lists of auto-converted tokens, one row per line,
///               regardless of whether the rows are uniform in length.
pub fn import_element(content: &str, element: &str) -> Option<Expr> {
  match element {
    "Plaintext" | "String" | "Text" => {
      Some(Expr::String(strip_trailing_newline(content.to_string())))
    }
    "Lines" => {
      let lines = split_lines(content);
      Some(Expr::List(
        lines
          .into_iter()
          .map(|l| Expr::String(l.to_string()))
          .collect(),
      ))
    }
    "Words" => {
      let words: Vec<Expr> = content
        .split_whitespace()
        .map(|w| Expr::String(w.to_string()))
        .collect();
      Some(Expr::List(words.into()))
    }
    "Data" => {
      let lines = split_lines(content);
      // Tokenise each line; track whether every line has the same token
      // count so we can decide between a table and a list of strings.
      let tokenised: Vec<Vec<&str>> = lines
        .iter()
        .map(|l| l.split_whitespace().collect())
        .collect();
      let uniform = !tokenised.is_empty()
        && tokenised.iter().all(|row| row.len() == tokenised[0].len())
        && !tokenised[0].is_empty();

      if uniform {
        let rows: Vec<Expr> = tokenised
          .iter()
          .map(|row| {
            Expr::List(row.iter().map(|tok| token_to_expr(tok)).collect())
          })
          .collect();
        Some(Expr::List(rows.into()))
      } else {
        Some(Expr::List(
          lines
            .into_iter()
            .map(|l| Expr::String(l.to_string()))
            .collect(),
        ))
      }
    }
    "Table" => {
      let lines = split_lines(content);
      let rows: Vec<Expr> = lines
        .iter()
        .map(|line| {
          Expr::List(line.split_whitespace().map(token_to_expr).collect())
        })
        .collect();
      Some(Expr::List(rows.into()))
    }
    _ => None,
  }
}
