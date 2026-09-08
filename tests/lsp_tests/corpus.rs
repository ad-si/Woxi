//! The language server run over every script in `tests/scripts`.
//!
//! The unit tests above check the rules; this checks them against real
//! code. Formatting hundreds of scripts is the only way to find the
//! construct a spacing rule breaks, and running the spell checker over
//! them is the only way to see how often it cries wolf — which, on code
//! that is known to work, must be never.

use woxi::lsp::analysis::{
  Token, diagnostics, find_definitions, scan, tokenize,
};
use woxi::lsp::format::{Options, format};

/// Every `.wls` script in the repository, as `(name, source)`.
fn scripts() -> Vec<(String, String)> {
  let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/scripts");
  let mut scripts: Vec<(String, String)> = std::fs::read_dir(dir)
    .expect("tests/scripts is readable")
    .filter_map(|entry| {
      let path = entry.ok()?.path();
      if path.extension()?.to_str()? != "wls" {
        return None;
      }
      let name = path.file_name()?.to_str()?.to_string();
      Some((name, std::fs::read_to_string(&path).ok()?))
    })
    .collect();
  scripts.sort();
  assert!(scripts.len() > 100, "expected the script corpus to be here");
  scripts
}

/// The `(kind, text)` of every token of `source`, comments included.
fn tokens_of(source: &str) -> Vec<(woxi::lsp::analysis::TokenKind, &str)> {
  scan(source)
    .into_iter()
    .map(|token: Token| (token.kind, token.text(source)))
    .collect()
}

#[test]
fn formatting_every_script_keeps_it_the_same_code() {
  for (name, source) in scripts() {
    let result = format(&source, &Options::default());
    assert_eq!(
      tokens_of(&result),
      tokens_of(&source),
      "formatting changed the tokens of {name}"
    );
    assert_eq!(
      result.split('\n').count(),
      source.split('\n').count(),
      "formatting changed the line count of {name}"
    );
    if woxi::parse(&source).is_ok() {
      assert!(woxi::parse(&result).is_ok(), "{name} no longer parses");
    }
    assert_eq!(
      format(&result, &Options::default()),
      result,
      "formatting {name} is not idempotent"
    );
  }
}

#[test]
fn no_working_script_is_accused_of_a_misspelling() {
  let mut accused = Vec::new();
  for (name, source) in scripts() {
    let tokens = tokenize(&source);
    let definitions = find_definitions(&source, &tokens);
    for diagnostic in diagnostics(&source, &tokens, &definitions) {
      if diagnostic.code == "spelling" {
        accused.push(format!("{name}: {}", diagnostic.message));
      }
    }
  }
  assert!(
    accused.is_empty(),
    "spelling hints on working code: {accused:#?}"
  );
}
