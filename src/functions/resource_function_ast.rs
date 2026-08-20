//! `ResourceFunction["Name"]` — fetch and load a named resource from the
//! Wolfram Function Repository on first use.
//!
//! Woxi bundles no catalog of resource functions: every name is resolved
//! against the Function Repository's public, unauthenticated pages — the
//! same ones a browser sees at
//! `https://resources.wolframcloud.com/FunctionRepository/resources/<Name>/`
//! — and its published "Definition" cell is evaluated through the normal
//! interpreter, the same way `Get` loads a package (see
//! `evaluator::dispatch::io_functions::evaluate_source`). The resource's own
//! exported name is rewritten to the private
//! `` FunctionRepository`$<hash>`Name `` symbol the resource system itself
//! reports (its `SymbolName` search-result field), so evaluating the
//! definition does not itself define the bare, unqualified name — matching
//! a real kernel, where only `ResourceFunction["Name"]` reaches it. Any
//! helper symbols the definition introduces are not similarly namespaced,
//! since Woxi's `Begin`/`End` do not yet route new definitions into a
//! context; a known gap shared with the rest of the context system.
//!
//! No network access, an unknown or ambiguous name, or a definition Woxi's
//! language subset can't evaluate all leave the call symbolic — matching a
//! kernel with no internet connection — rather than erroring.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn cache() -> &'static Mutex<HashMap<String, Option<String>>> {
  static CACHE: OnceLock<Mutex<HashMap<String, Option<String>>>> =
    OnceLock::new();
  CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolve `name` to the private symbol its definition was loaded under,
/// fetching and evaluating it on first use. Returns `None` if it could not
/// be found, downloaded, parsed, or evaluated. Cached in-process for the
/// rest of the run so repeated calls don't refetch.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_resource_function(name: &str) -> Option<String> {
  if let Some(hit) = cache().lock().unwrap().get(name) {
    return hit.clone();
  }
  let resolved = fetch_and_load(name);
  cache()
    .lock()
    .unwrap()
    .insert(name.to_string(), resolved.clone());
  resolved
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_and_load(name: &str) -> Option<String> {
  let symbol_name = search_symbol_name(name)?;
  let download_url = find_definition_download_url(name)?;
  let notebook_src = curl_get_text(&download_url)?;
  let nb = crate::notebook::parse_notebook(&notebook_src).ok()?;
  let definition_src = extract_definition_source(&nb)?;
  let private_src = rewrite_bare_name(&definition_src, name, &symbol_name);
  crate::evaluator::dispatch::io_functions::evaluate_source(&private_src)
    .ok()?;
  Some(symbol_name)
}

/// Run `curl` against `url` and return its stdout as text, or `None` on any
/// failure (offline, DNS, timeout, non-2xx, non-UTF8 body). Short timeouts
/// keep an offline `ResourceFunction[...]` call from stalling a script.
#[cfg(not(target_arch = "wasm32"))]
fn curl_get_text(url: &str) -> Option<String> {
  let output = std::process::Command::new("curl")
    .args([
      "-s",
      "-S",
      "-L",
      "--compressed",
      "--connect-timeout",
      "5",
      "--max-time",
      "15",
    ])
    .arg(url)
    .output()
    .ok()?;
  if !output.status.success() {
    return None;
  }
  String::from_utf8(output.stdout).ok()
}

/// Look up `name` in the Function Repository's public search API and return
/// the private `SymbolName` a real kernel loads it under, e.g.
/// `` FunctionRepository`$9543894814e342208989d9067167bd41`BarycentricCoordinates ``.
/// Only an exact, case-sensitive `Name` match counts.
#[cfg(not(target_arch = "wasm32"))]
fn search_symbol_name(name: &str) -> Option<String> {
  let url = format!(
    "https://www.wolframcloud.com/obj/resourcesystem/api/1.0/SearchResources?ResourceTypes=Function&Format=json&Count=20&Query={}",
    url_encode(name)
  );
  let body = curl_get_text(&url)?;
  let json: serde_json::Value = serde_json::from_str(&body).ok()?;
  let resources = json.get("Resources")?.as_array()?;
  resources.iter().find_map(|r| {
    if r.get("Name")?.as_str()? != name {
      return None;
    }
    r.get("SymbolName")?.as_str().map(str::to_string)
  })
}

/// Scrape the resource's public page for the link to download its
/// definition notebook — the search API returns no clean field for it.
#[cfg(not(target_arch = "wasm32"))]
fn find_definition_download_url(name: &str) -> Option<String> {
  let url = format!(
    "https://resources.wolframcloud.com/FunctionRepository/resources/{}/",
    url_encode(name)
  );
  let html = curl_get_text(&url)?;
  let re = regex::Regex::new(
    r#"https://www\.wolframcloud\.com/download/[0-9a-fA-F-]+\?[^"'\s]*-definition"#,
  )
  .ok()?;
  re.find(&html).map(|m| m.as_str().replace("&amp;", "&"))
}

/// Percent-encode a resource name for use in a URL path/query component.
#[cfg(not(target_arch = "wasm32"))]
fn url_encode(s: &str) -> String {
  let mut out = String::with_capacity(s.len());
  for b in s.bytes() {
    match b {
      b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
        out.push(b as char);
      }
      _ => out.push_str(&format!("%{b:02X}")),
    }
  }
  out
}

/// Pull the "Definition" section's Input cell(s) — the resource's actual
/// code — out of a downloaded definition notebook, joined by newlines.
#[cfg(not(target_arch = "wasm32"))]
fn extract_definition_source(nb: &crate::notebook::Notebook) -> Option<String> {
  use crate::notebook::{Cell, CellEntry, CellStyle};
  let mut cells: Vec<Cell> = Vec::new();
  for entry in &nb.cells {
    match entry {
      CellEntry::Single(c) => cells.push(c.clone()),
      CellEntry::Group(g) => cells.extend(g.cells.iter().cloned()),
    }
  }
  let is_heading = |c: &Cell| {
    matches!(
      c.style,
      CellStyle::Title
        | CellStyle::Section
        | CellStyle::Subsection
        | CellStyle::Subsubsection
    )
  };
  let start = cells
    .iter()
    .position(|c| is_heading(c) && c.content.trim() == "Definition")?;
  let mut src = String::new();
  for cell in &cells[start + 1..] {
    if is_heading(cell) {
      break;
    }
    if matches!(cell.style, CellStyle::Input | CellStyle::Code) {
      if !src.is_empty() {
        src.push('\n');
      }
      src.push_str(&cell.content);
    }
  }
  (!src.is_empty()).then_some(src)
}

/// Replace standalone occurrences of the resource's bare exported
/// identifier with its private, context-qualified symbol, so evaluating the
/// definition doesn't itself define the bare name in `Global`` — matches
/// Wolfram identifier syntax (letters, digits, `$`) so it doesn't touch a
/// longer identifier that merely contains `name` as a substring (e.g. a
/// helper `BarycentricObject` next to a resource named
/// `BarycentricCoordinates`) or one extended with a `$` suffix.
#[cfg(not(target_arch = "wasm32"))]
fn rewrite_bare_name(src: &str, name: &str, symbol_name: &str) -> String {
  let is_ident_char = |c: char| c.is_ascii_alphanumeric() || c == '$';
  let mut out = String::with_capacity(src.len());
  let mut rest = src;
  while let Some(idx) = rest.find(name) {
    let before_ok = rest[..idx]
      .chars()
      .next_back()
      .is_none_or(|c| !is_ident_char(c));
    let after = &rest[idx + name.len()..];
    let after_ok = after.chars().next().is_none_or(|c| !is_ident_char(c));
    out.push_str(&rest[..idx]);
    out.push_str(if before_ok && after_ok {
      symbol_name
    } else {
      name
    });
    rest = after;
  }
  out.push_str(rest);
  out
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
  use super::*;
  use crate::notebook::{Cell, CellEntry, CellStyle};

  fn cell(style: CellStyle, content: &str) -> Cell {
    Cell {
      style,
      content: content.to_string(),
      collapsed: false,
    }
  }

  #[test]
  fn rewrite_bare_name_replaces_exact_identifier_occurrences() {
    let src = "ClearAll[Foo]\nFoo::tag=\"msg\";\nFoo[x_]:=x+1;\nFoo[2]";
    let out = rewrite_bare_name(src, "Foo", "Ctx`$hash`Foo");
    assert_eq!(
      out,
      "ClearAll[Ctx`$hash`Foo]\nCtx`$hash`Foo::tag=\"msg\";\nCtx`$hash`Foo[x_]:=x+1;\nCtx`$hash`Foo[2]"
    );
  }

  #[test]
  fn rewrite_bare_name_does_not_touch_longer_identifiers() {
    // "FooBar" and "BarFoo" are distinct identifiers from "Foo" — a plain
    // substring replace would wrongly mangle them.
    let src = "FooBar[x_]:=x; BarFoo[x_]:=x; Foo[x_]:=x";
    let out = rewrite_bare_name(src, "Foo", "Priv`Foo");
    assert_eq!(out, "FooBar[x_]:=x; BarFoo[x_]:=x; Priv`Foo[x_]:=x");
  }

  #[test]
  fn rewrite_bare_name_does_not_touch_dollar_suffixed_variant() {
    // Wolfram identifiers include `$`, so "Foo$1" is one token, not "Foo"
    // followed by "$1".
    let src = "Foo$1=1; Foo[x_]:=x";
    let out = rewrite_bare_name(src, "Foo", "Priv`Foo");
    assert_eq!(out, "Foo$1=1; Priv`Foo[x_]:=x");
  }

  #[test]
  fn extract_definition_source_collects_input_cells_until_next_heading() {
    let nb = crate::notebook::Notebook {
      cells: vec![
        CellEntry::Single(cell(CellStyle::Title, "MyResource")),
        CellEntry::Single(cell(CellStyle::Text, "A description")),
        CellEntry::Single(cell(CellStyle::Section, "Definition")),
        CellEntry::Single(cell(CellStyle::Input, "ClearAll[MyResource]")),
        CellEntry::Single(cell(CellStyle::Input, "MyResource[x_]:=x+1")),
        CellEntry::Single(cell(CellStyle::Section, "Documentation")),
        CellEntry::Single(cell(CellStyle::Input, "MyResource[5]")),
      ],
    };
    let src = extract_definition_source(&nb).unwrap();
    assert_eq!(src, "ClearAll[MyResource]\nMyResource[x_]:=x+1");
  }

  #[test]
  fn extract_definition_source_none_without_definition_section() {
    let nb = crate::notebook::Notebook {
      cells: vec![
        CellEntry::Single(cell(CellStyle::Title, "MyResource")),
        CellEntry::Single(cell(CellStyle::Input, "MyResource[5]")),
      ],
    };
    assert!(extract_definition_source(&nb).is_none());
  }

  #[test]
  fn url_encode_escapes_non_alphanumeric_bytes() {
    assert_eq!(url_encode("Foo Bar"), "Foo%20Bar");
    assert_eq!(url_encode("A-b_c.d~e"), "A-b_c.d~e");
    assert_eq!(url_encode("a`b"), "a%60b");
  }
}
