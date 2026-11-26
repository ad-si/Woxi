use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// functions.csv format: name,description,implementation_status,effect_level,version,rank
static FUNCTIONS_CSV: &str = include_str!("../../functions.csv");

/// How completely Woxi implements a `System`` symbol, taken from the
/// `implementation_status` column of functions.csv.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplementationStatus {
  /// ✅ — implemented.
  Implemented,
  /// 🚧 — partially implemented: some forms or options are missing.
  Partial,
  /// 🚫 — out of scope for Woxi (e.g. it needs the Wolfram Cloud).
  NotPlanned,
  /// Listed in functions.csv, but not implemented (yet).
  NotImplemented,
}

/// Implementation status of every symbol in functions.csv.
static IMPLEMENTATION_STATUS: LazyLock<
  HashMap<&'static str, ImplementationStatus>,
> = LazyLock::new(|| {
  FUNCTIONS_CSV
    .lines()
    .skip(1)
    .filter_map(|line| {
      // The description may contain (quoted) commas, so read the four
      // trailing columns from the right.
      let fields: Vec<&str> = line.rsplitn(5, ',').collect();
      let name = fields.last()?.split(',').next()?.trim();
      let status = fields.get(3).unwrap_or(&"").trim();
      if name.is_empty() || name == "-----" {
        return None;
      }
      let status = if status.starts_with("✅") {
        ImplementationStatus::Implemented
      } else if status.starts_with("🚧") {
        ImplementationStatus::Partial
      } else if status.starts_with("🚫") {
        ImplementationStatus::NotPlanned
      } else {
        ImplementationStatus::NotImplemented
      };
      Some((name, status))
    })
    .collect()
});

/// Look up how completely Woxi implements a built-in symbol.
/// `None` for names that are not `System`` symbols at all.
pub fn implementation_status(name: &str) -> Option<ImplementationStatus> {
  IMPLEMENTATION_STATUS.get(name).copied()
}

/// Set of known Wolfram Language function names (from functions.csv)
/// that are NOT yet implemented in Woxi.
static KNOWN_WOLFRAM_FUNCTIONS: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| {
    IMPLEMENTATION_STATUS
      .iter()
      .filter(|(_, status)| **status != ImplementationStatus::Implemented)
      .map(|(name, _)| *name)
      .collect()
  });

/// Check if a function name is a known Wolfram Language function
/// that hasn't been implemented yet.
pub fn is_known_wolfram_function(name: &str) -> bool {
  !starts_lowercase(name) && KNOWN_WOLFRAM_FUNCTIONS.contains(name)
}

/// Whether `name` starts with a lowercase ASCII letter. No `System`` symbol
/// does (see `no_builtin_starts_lowercase`), so the names of most user
/// symbols — `x`, `g`, `list$12` — are told apart from them without hashing
/// the name and probing the tables, which every evaluated call and every
/// comparison of two symbols otherwise does.
fn starts_lowercase(name: &str) -> bool {
  name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
}

/// Every symbol name in functions.csv, regardless of whether Woxi implements
/// it or whether its row carries a description. `BUILTIN_FUNCTION_INFO` skips
/// rows without a description or effect level, which is the wrong question to
/// ask about a symbol's attributes.
static ALL_BUILTIN_NAMES: LazyLock<rustc_hash::FxHashSet<&'static str>> =
  LazyLock::new(|| {
    FUNCTIONS_CSV
      .lines()
      .skip(1)
      .filter_map(|line| {
        let name = line.split(',').next()?.trim();
        if name.is_empty() || name == "-----" {
          None
        } else {
          Some(name)
        }
      })
      .collect()
  });

/// The formal symbols `\[FormalA]`, `\[FormalCapitalOmega]`,
/// `\[FormalScriptX]`, …: single private-use glyphs that live in `System``
/// and are `Protected`, so they can serve as bound variables that user code
/// cannot accidentally assign to.
static FORMAL_SYMBOLS: LazyLock<rustc_hash::FxHashSet<&'static str>> =
  LazyLock::new(|| {
    crate::named_characters::NAMED_CHARACTERS
      .iter()
      .filter(|(name, _)| name.starts_with("Formal"))
      .map(|(_, glyph)| *glyph)
      .collect()
  });

/// True if `name` is a `System`` symbol, i.e. listed in functions.csv or
/// one of the formal symbols.
pub fn is_builtin_symbol(name: &str) -> bool {
  !starts_lowercase(name)
    && (ALL_BUILTIN_NAMES.contains(name) || FORMAL_SYMBOLS.contains(name))
}

/// Built-ins whose `effect_level` in functions.csv says that evaluating
/// them does more than compute a value: they print, change state or touch
/// files and the network (`effectful`, `stateful`, `io`).
static SIDE_EFFECTING_BUILTINS: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| {
    FUNCTIONS_CSV
      .lines()
      .skip(1)
      .filter_map(|line| {
        // Read the trailing columns from the right, as the description may
        // contain (quoted) commas.
        let fields: Vec<&str> = line.rsplitn(5, ',').collect();
        let name = fields.last()?.split(',').next()?.trim();
        let effect_level = fields.get(2)?.trim();
        matches!(effect_level, "effectful" | "stateful" | "io").then_some(name)
      })
      .collect()
  });

/// True if evaluating the built-in `name` has side effects beyond
/// returning a value (printing, assignment, file or network access).
pub fn is_side_effecting_builtin(name: &str) -> bool {
  SIDE_EFFECTING_BUILTINS.contains(name)
}

/// Information about a built-in Wolfram Language function.
pub struct BuiltinFunctionInfo {
  pub description: &'static str,
}

/// Strip RFC 4180 quoting from a CSV field and collapse doubled quotes.
fn unquote_csv_field(field: &'static str) -> &'static str {
  match field.strip_prefix('"').and_then(|f| f.strip_suffix('"')) {
    Some(inner) if inner.contains("\"\"") => {
      Box::leak(inner.replace("\"\"", "\"").into_boxed_str())
    }
    Some(inner) => inner,
    None => field,
  }
}

/// Registry of all function info from functions.csv.
static BUILTIN_FUNCTION_INFO: LazyLock<
  HashMap<&'static str, BuiltinFunctionInfo>,
> = LazyLock::new(|| {
  FUNCTIONS_CSV
    .lines()
    .skip(1)
    .filter_map(|line| {
      // Columns: name,description,status,effect_level,version,rank. The
      // description may be quoted (RFC 4180) and contain commas, so read the
      // four trailing columns from the right.
      let fields: Vec<&str> = line.rsplitn(5, ',').collect();
      let head = fields.last()?;
      let (name, description) = head.split_once(',').unwrap_or((head, ""));
      let name = name.trim();
      let description = unquote_csv_field(description.trim());
      let effect_level = fields.get(2).unwrap_or(&"").trim();
      if name.is_empty() || name == "-----" {
        return None;
      }
      if description.is_empty() && effect_level.is_empty() {
        return None;
      }
      Some((name, BuiltinFunctionInfo { description }))
    })
    .collect()
});

/// Look up built-in function info by name.
pub fn get_builtin_function_info(
  name: &str,
) -> Option<&'static BuiltinFunctionInfo> {
  BUILTIN_FUNCTION_INFO.get(name)
}

/// Get all built-in function names from functions.csv.
pub fn get_builtin_function_names() -> Vec<&'static str> {
  let mut names: Vec<&str> = BUILTIN_FUNCTION_INFO.keys().copied().collect();
  names.sort_unstable();
  names
}

/// Get every Wolfram Language function name that appears in
/// `functions.csv` — whether or not Woxi has implemented it.
pub fn known_wolfram_function_names() -> Vec<&'static str> {
  let mut names: Vec<&str> = KNOWN_WOLFRAM_FUNCTIONS.iter().copied().collect();
  names.sort_unstable();
  names
}

/// `tests/SUMMARY.md` is the mdbook table of contents that lists every
/// documented built-in. Embedding it lets `Information[…]` resolve a symbol
/// name to its public documentation URL without runtime file I/O (works in
/// WASM too).
static SUMMARY_MD: &str = include_str!("../../tests/SUMMARY.md");

/// Map of built-in symbol name → public documentation URL on
/// <https://woxi.ad-si.com>. Built once from `SUMMARY.md` entries of the
/// form `` [`Sin`](math/elementary/Sin.md) ``.
static DOC_URL_MAP: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
  let re = regex::Regex::new(r"\[`([A-Za-z][A-Za-z0-9]*)`\]\(([^)]+)\.md\)")
    .expect("valid SUMMARY.md regex");
  let mut map = HashMap::new();
  for cap in re.captures_iter(SUMMARY_MD) {
    let name = cap.get(1).unwrap().as_str().to_string();
    let path = cap.get(2).unwrap().as_str();
    map
      .entry(name)
      .or_insert_with(|| format!("https://woxi.ad-si.com/docs/{path}"));
  }
  map
});

/// Look up the public documentation URL for a built-in symbol, if one exists.
pub fn get_doc_url(name: &str) -> Option<&'static str> {
  DOC_URL_MAP.get(name).map(std::string::String::as_str)
}

/// Every `System`` symbol name, whether or not Woxi implements it — what
/// `Names` enumerates and what context resolution must leave unqualified.
pub fn all_builtin_symbol_names() -> Vec<&'static str> {
  let mut names: Vec<&str> = ALL_BUILTIN_NAMES.iter().copied().collect();
  names.sort_unstable();
  names
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `is_builtin_symbol` and `is_known_wolfram_function` answer for a name
  /// starting with a lowercase letter without looking it up.
  #[test]
  fn no_builtin_starts_lowercase() {
    for name in ALL_BUILTIN_NAMES
      .iter()
      .chain(FORMAL_SYMBOLS.iter())
      .chain(IMPLEMENTATION_STATUS.keys())
    {
      assert!(!starts_lowercase(name), "{name}");
    }
  }

  #[test]
  fn builtin_symbols_are_recognized() {
    for name in ["Plus", "Sin", "$Version", "Pi", "E", "I", "List"] {
      assert!(is_builtin_symbol(name), "{name}");
    }
    for name in ["x", "g", "list$12", "plus", "", "myFunction", "Zzyzx"] {
      assert!(!is_builtin_symbol(name), "{name}");
      assert!(!is_known_wolfram_function(name), "{name}");
    }
  }
}
