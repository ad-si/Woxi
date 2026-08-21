//! Paclet support: the directory registry behind `PacletDirectoryLoad` /
//! `PacletDirectoryUnload` and the context → file resolution that
//! `Needs`, `Get` and `FindFile` share.
//!
//! A paclet is a directory holding a `PacletInfo.wl` (or `PacletInfo.m`)
//! file with a `PacletObject[<|…|>]` expression. Its `"Kernel"` extensions
//! declare which contexts the paclet provides and where their `.wl`/`.m`
//! files live, so requesting one of those contexts loads the right file:
//!
//! ```wolfram
//! PacletObject[<|"Name" -> "MyPaclet", "Version" -> "1.0.0",
//!   "Extensions" -> {{"Kernel", "Root" -> "Kernel",
//!                     "Context" -> {"MyPaclet`"}}}|>]
//! ```

use std::cell::RefCell;
use std::path::{Component, Path, PathBuf};

use crate::syntax::Expr;

thread_local! {
  /// Directories registered by `PacletDirectoryLoad`, in registration order.
  static PACLET_DIRECTORIES: RefCell<Vec<String>> =
    const { RefCell::new(Vec::new()) };
}

/// The directories currently registered as paclet directories.
pub fn loaded_directories() -> Vec<String> {
  PACLET_DIRECTORIES.with(|d| d.borrow().clone())
}

/// Forget every registered paclet directory — a fresh session has none.
pub fn clear_loaded_directories() {
  PACLET_DIRECTORIES.with(|d| d.borrow_mut().clear());
}

/// Register `dirs` as paclet directories and return the full list of loaded
/// directories. A directory that does not exist is reported with
/// `PacletDirectoryLoad::nodir` and not registered; an already registered one
/// keeps its position, matching wolframscript.
pub fn load_directories(dirs: &[String]) -> Vec<String> {
  for dir in dirs {
    let expanded = expand_directory(dir);
    if !Path::new(&expanded).is_dir() {
      crate::emit_message_to_stdout(&format!(
        "PacletDirectoryLoad::nodir: Directory {expanded} not found."
      ));
      continue;
    }
    PACLET_DIRECTORIES.with(|d| {
      let mut d = d.borrow_mut();
      if !d.contains(&expanded) {
        d.push(expanded);
      }
    });
  }
  loaded_directories()
}

/// Unregister `dirs` and return the directories that remain loaded.
/// Unloading a directory that was never loaded is a silent no-op.
pub fn unload_directories(dirs: &[String]) -> Vec<String> {
  let expanded: Vec<String> =
    dirs.iter().map(|d| expand_directory(d)).collect();
  PACLET_DIRECTORIES.with(|d| d.borrow_mut().retain(|e| !expanded.contains(e)));
  loaded_directories()
}

/// Absolutize `dir` against the virtual working directory and fold away
/// `.` and `..` segments. Symbolic links are deliberately left alone:
/// wolframscript reports `PacletDirectoryLoad["/tmp"]` as `/tmp`, not as
/// the directory `/tmp` links to.
///
/// The result is spelled the way Woxi spells paths in strings — folding the
/// components back together would otherwise turn a `C:/dir` argument into
/// `C:\dir` on Windows, which no longer compares equal to what the caller
/// passed in and cannot be re-used as a string literal.
fn expand_directory(dir: &str) -> String {
  let joined = crate::vfs::resolve(dir);
  let mut normalized = PathBuf::new();
  for component in joined.components() {
    match component {
      Component::CurDir => {}
      Component::ParentDir => {
        normalized.pop();
      }
      other => normalized.push(other.as_os_str()),
    }
  }
  crate::utils::wolfram_path_string(&normalized)
}

/// The file that provides `context`, or `None` when nothing does.
/// Registered paclet directories are searched first, then `$Path` — the
/// order wolframscript uses.
pub fn resolve_context(context: &str) -> Option<PathBuf> {
  let segments = crate::utils::context_segments(context)?;
  for dir in loaded_directories() {
    for root in paclet_roots(Path::new(&dir)) {
      if let Some(file) = paclet_context_file(&root, context, &segments) {
        return Some(file);
      }
    }
  }
  for entry in crate::utils::search_path() {
    let base = PathBuf::from(expand_directory(&entry));
    let mut stem = base;
    for segment in &segments {
      stem.push(segment);
    }
    // `Foo.wl` wins over `Foo.m`, and both over `Foo/init.wl`.
    let candidates = [
      stem.with_extension("wl"),
      stem.with_extension("m"),
      stem.join("init.wl"),
      stem.join("init.m"),
    ];
    if let Some(file) = candidates.into_iter().find(|p| p.is_file()) {
      return Some(file);
    }
  }
  None
}

/// The paclet directories inside `dir`: `dir` itself when it is a paclet,
/// plus every immediate subdirectory that is one. A `PacletDirectoryLoad`
/// argument may be either a single paclet or a directory collecting several.
fn paclet_roots(dir: &Path) -> Vec<PathBuf> {
  let mut roots = Vec::new();
  if paclet_info_file(dir).is_some() {
    roots.push(dir.to_path_buf());
  }
  if let Ok(entries) = std::fs::read_dir(dir) {
    let mut subdirectories: Vec<PathBuf> = entries
      .flatten()
      .map(|entry| entry.path())
      .filter(|path| path.is_dir() && paclet_info_file(path).is_some())
      .collect();
    subdirectories.sort();
    roots.extend(subdirectories);
  }
  roots
}

/// The `PacletInfo.wl` / `PacletInfo.m` file of the paclet rooted at `dir`.
fn paclet_info_file(dir: &Path) -> Option<PathBuf> {
  ["PacletInfo.wl", "PacletInfo.m"]
    .into_iter()
    .map(|name| dir.join(name))
    .find(|path| path.is_file())
}

/// The file the paclet rooted at `root` provides `context` from, if any.
fn paclet_context_file(
  root: &Path,
  context: &str,
  segments: &[&str],
) -> Option<PathBuf> {
  let info = std::fs::read_to_string(paclet_info_file(root)?).ok()?;
  let parsed = crate::syntax::string_to_expr(&info).ok()?;
  for extension in kernel_extensions(&parsed) {
    let extension_root = match &option_value(&extension, "Root") {
      Some(Expr::String(sub)) => {
        let mut path = root.to_path_buf();
        for segment in sub.split(['/', '\\']).filter(|s| !s.is_empty()) {
          path.push(segment);
        }
        path
      }
      // Without a "Root" the extension is rooted at the paclet directory.
      _ => root.to_path_buf(),
    };
    let Some(file) = declared_context_file(&extension, context) else {
      continue;
    };
    // An explicitly declared file wins; otherwise the extension's
    // `init.wl`/`init.m` is loaded, and only then a file named after the
    // context. (Verified against wolframscript.)
    let candidates: Vec<PathBuf> = if let Some(name) = file {
      vec![extension_root.join(name)]
    } else {
      let mut stem = extension_root.clone();
      for segment in segments {
        stem.push(segment);
      }
      vec![
        extension_root.join("init.wl"),
        extension_root.join("init.m"),
        stem.with_extension("wl"),
        stem.with_extension("m"),
      ]
    };
    if let Some(found) = candidates.into_iter().find(|p| p.is_file()) {
      return Some(found);
    }
  }
  None
}

/// The `"Kernel"` extensions of a parsed paclet-info expression.
/// An extension is a list whose first element is the extension name, e.g.
/// `{"Kernel", "Root" -> "Kernel", "Context" -> {"MyPaclet`"}}`.
///
/// Both spellings a `PacletInfo` file may use are accepted: the current
/// `PacletObject[<|"Extensions" -> …|>]` with string keys, and the legacy
/// `Paclet[Extensions -> …]` with symbol keys that `PacletInfo.m` files
/// predating version 12 still carry.
fn kernel_extensions(paclet_object: &Expr) -> Vec<Vec<Expr>> {
  let Expr::FunctionCall { name, args } = paclet_object else {
    return Vec::new();
  };
  if args.is_empty() {
    return Vec::new();
  }
  let fields: Vec<(Expr, Expr)> = match (name.as_str(), &args[0]) {
    ("PacletObject", Expr::Association(pairs)) => {
      pairs.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
    ("Paclet", _) => args.iter().filter_map(rule_pair).collect(),
    _ => return Vec::new(),
  };
  let extensions = fields
    .iter()
    .find_map(|(key, value)| key_is(key, "Extensions").then_some(value));
  let Some(Expr::List(items)) = extensions else {
    return Vec::new();
  };
  items
    .iter()
    .filter_map(|item| match item {
      Expr::List(parts) if key_is(parts.first()?, "Kernel") => {
        Some(parts.iter().map(Expr::clone).collect())
      }
      _ => None,
    })
    .collect()
}

/// The `lhs -> rhs` (or `:>`) pair an expression carries, if it is a rule.
fn rule_pair(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some(((**pattern).clone(), (**replacement).clone())),
    _ => None,
  }
}

/// Whether `key` names `name`, written either as the string `"Name"` or —
/// in a legacy `Paclet[…]` info file — as the bare symbol `Name`.
fn key_is(key: &Expr, name: &str) -> bool {
  matches!(key, Expr::String(k) | Expr::Identifier(k) if k == name)
}

/// The value of the `name -> value` rule in an extension, if present.
fn option_value(extension: &[Expr], name: &str) -> Option<Expr> {
  extension.iter().find_map(|part| {
    let (key, value) = rule_pair(part)?;
    key_is(&key, name).then_some(value)
  })
}

/// Whether `extension` declares `context`, and the file explicitly given
/// for it. `"Context" -> {"A`", {"B`", "B.wl"}}` declares `A`` without a
/// file and `B`` with `"B.wl"`; a bare `"Context" -> "A`"` is also allowed.
fn declared_context_file(
  extension: &[Expr],
  context: &str,
) -> Option<Option<String>> {
  let declarations = option_value(extension, "Context")?;
  let entries: Vec<Expr> = match &declarations {
    Expr::List(items) => items.iter().map(Expr::clone).collect(),
    other => vec![(*other).clone()],
  };
  entries.iter().find_map(|entry| match entry {
    Expr::String(declared) if declared == context => Some(None),
    Expr::List(pair) => match (pair.first(), pair.get(1)) {
      (Some(Expr::String(declared)), Some(Expr::String(file)))
        if declared == context =>
      {
        Some(Some(file.clone()))
      }
      (Some(Expr::String(declared)), None) if declared == context => Some(None),
      _ => None,
    },
    _ => None,
  })
}
