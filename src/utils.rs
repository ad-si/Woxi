#[cfg(not(target_arch = "wasm32"))]
use rand::distributions::{Alphanumeric, Distribution};

#[cfg(not(target_arch = "wasm32"))]
fn rand_str(length: usize) -> String {
  let rng = rand::thread_rng();
  let characters: Vec<char> = Alphanumeric
    .sample_iter(rng)
    .map(std::convert::Into::into)
    .take(length)
    .collect();
  characters.iter().collect::<String>()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn create_file(
  filename_opt: Option<String>,
) -> Result<std::path::PathBuf, std::io::Error> {
  let file_path = match filename_opt {
    Some(filename) => {
      let home_dir = std::env::current_dir().unwrap();
      home_dir.join(filename)
    }
    None => std::env::temp_dir().join(rand_str(16)),
  };

  create_new_file(file_path)
}

/// Create a uniquely named empty file in the system temp directory carrying
/// `extension`, for callers that need the name to say what the bytes are.
#[cfg(not(target_arch = "wasm32"))]
pub fn create_temp_file(
  extension: &str,
) -> Result<std::path::PathBuf, std::io::Error> {
  create_new_file(
    std::env::temp_dir().join(format!("{}.{extension}", rand_str(16))),
  )
}

/// Create `path`, failing if something is already there.
#[cfg(not(target_arch = "wasm32"))]
fn create_new_file(
  path: std::path::PathBuf,
) -> Result<std::path::PathBuf, std::io::Error> {
  std::fs::OpenOptions::new()
    .create_new(true)
    .write(true)
    .truncate(true)
    .open(&path)
    .map(|_| path)
}

/// Join `sub` onto `base` with the platform's path separator.
#[cfg(not(target_arch = "wasm32"))]
fn join_path(base: &str, sub: &str) -> String {
  let sep = std::path::MAIN_SEPARATOR_STR;
  format!(
    "{}{sep}{}",
    base.trim_end_matches(['/', std::path::MAIN_SEPARATOR]),
    sub.replace('/', sep)
  )
}

/// The `$Path` package search path — the directories `Needs`, `Get` and
/// `FindFile` look through for a context's file. Modeled after
/// wolframscript's list but rooted at Woxi's directories since we don't ship
/// the full Wolfram layout.
///
/// `$Path` is an ordinary variable, and the usual way to make a package
/// findable is `AppendTo[$Path, dir]`, so a value that has been assigned wins
/// over the built-in list.
#[cfg(not(target_arch = "wasm32"))]
pub fn search_path() -> Vec<String> {
  if let Some(value) = crate::variable_value("$Path")
    && let crate::syntax::Expr::List(entries) = &value
  {
    return entries
      .iter()
      .filter_map(|entry| match entry {
        crate::syntax::Expr::String(dir) => Some(dir.clone()),
        _ => None,
      })
      .collect();
  }
  default_search_path()
}

/// The `$Path` value Woxi starts a session with.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_search_path() -> Vec<String> {
  let home = std::env::var("HOME")
    .or_else(|_| std::env::var("USERPROFILE"))
    .unwrap_or_default();
  let user_sub = if cfg!(target_os = "macos") {
    "Library/Wolfram"
  } else if cfg!(target_os = "windows") {
    "AppData\\Roaming\\Wolfram"
  } else {
    ".Wolfram"
  };
  let base_root = if cfg!(target_os = "macos") {
    "/Library/Wolfram"
  } else if cfg!(target_os = "windows") {
    "C:\\ProgramData\\Wolfram"
  } else {
    "/usr/share/Wolfram"
  };
  let user_base = join_path(&home, user_sub);
  let mut entries: Vec<String> = vec![
    join_path(&user_base, "Kernel"),
    join_path(&user_base, "Autoload"),
    join_path(&user_base, "Applications"),
    join_path(base_root, "Kernel"),
    join_path(base_root, "Autoload"),
    join_path(base_root, "Applications"),
    ".".to_string(),
  ];
  if !home.is_empty() {
    entries.push(home);
  }
  entries
}

/// Canonicalize `path` the way Wolfram spells paths.
///
/// Identical to [`std::fs::canonicalize`] except on Windows, where that
/// returns an extended-length "verbatim" path (`\\?\C:\dir`). Wolfram never
/// surfaces that form — `SetDirectory`, `AbsoluteFileName`,
/// `$TemporaryDirectory` and friends all report a plain drive or UNC path —
/// and the prefix silently breaks comparisons against paths spelled the
/// ordinary way (e.g. `$UserDocumentsDirectory`).
#[cfg(not(target_arch = "wasm32"))]
pub fn canonicalize(
  path: impl AsRef<std::path::Path>,
) -> std::io::Result<std::path::PathBuf> {
  let canonical = std::fs::canonicalize(path)?;
  #[cfg(target_os = "windows")]
  if let Some(plain) =
    canonical.to_str().and_then(strip_windows_verbatim_prefix)
  {
    return Ok(std::path::PathBuf::from(plain));
  }
  Ok(canonical)
}

/// The way Woxi spells a filesystem path inside a Wolfram Language string.
///
/// On Windows the native separator is `\`, which Woxi's string layer reads
/// back as an escape: a path through `C:\new\table` comes apart into
/// control characters the moment it is spliced into another expression, so
/// the result cannot be handed to `Get`, `FileNameSplit` or a comparison.
/// Paths are therefore surfaced with forward slashes — accepted by the
/// Windows API just as well, and the spelling `$InputFileName` already
/// uses. Elsewhere `\` is an ordinary filename character and is kept.
pub fn wolfram_path_string(path: &std::path::Path) -> String {
  let spelled = path.to_string_lossy().into_owned();
  if cfg!(target_os = "windows") {
    spelled.replace('\\', "/")
  } else {
    spelled
  }
}

/// The plain spelling of a Windows extended-length path, or `None` when
/// `path` has no verbatim prefix or cannot lose it.
///
/// `\\?\C:\dir` becomes `C:\dir` and `\\?\UNC\server\share` becomes
/// `\\server\share`. Device paths such as `\\?\Volume{…}\dir` have no plain
/// equivalent and keep the prefix.
///
/// Defined on every platform (and never called off Windows) so the mapping
/// stays under test wherever the suite runs.
pub fn strip_windows_verbatim_prefix(path: &str) -> Option<String> {
  if let Some(rest) = path.strip_prefix(r"\\?\UNC\") {
    return Some(format!(r"\\{rest}"));
  }
  let rest = path.strip_prefix(r"\\?\")?;
  // Only a drive-letter path is expressible without the prefix.
  let mut chars = rest.chars();
  match (chars.next(), chars.next()) {
    (Some(drive), Some(':')) if drive.is_ascii_alphabetic() => {
      Some(rest.to_string())
    }
    _ => None,
  }
}

/// The top-level contexts of the packages that ship with the Wolfram
/// Language, whose `Get`/`Needs` therefore always succeeds. Woxi has no
/// package system — every built-in it implements lives in one namespace —
/// so loading one of these has nothing to do and evaluates to `Null` the
/// way it does in the Wolfram Language. A symbol the package would have
/// defined and Woxi does not implement still reports itself, unevaluated,
/// at the point it is used.
const STANDARD_DISTRIBUTION_CONTEXTS: &[&str] = &[
  "BarCharts",
  "Calendar",
  "Combinatorica",
  "ComputationalGeometry",
  "ComputerArithmetic",
  "DatabaseLink",
  "Developer",
  "DifferentialEquations",
  "ErrorBarPlots",
  "FiniteFields",
  "FourierSeries",
  "FunctionApproximations",
  "Geodesy",
  "Graphics",
  "GraphUtilities",
  "HypothesisTesting",
  "Internal",
  "JLink",
  "LinearAlgebra",
  "MultivariateStatistics",
  "NETLink",
  "Notation",
  "NumericalCalculus",
  "NumericalDifferentialEquationAnalysis",
  "PhysicalConstants",
  "PieCharts",
  "PlotLegends",
  "PolyhedronOperations",
  "PrimalityProving",
  "Quaternions",
  "RegressionCommon",
  "ResourceLocator",
  "StatisticalPlots",
  "Units",
  "VariationalMethods",
  "VectorAnalysis",
  "WaveletScalogram",
];

/// The symbol segments of a context name — `"Foo\`Bar\`"` gives
/// `["Foo", "Bar"]`. `None` for anything that is not a context name: a
/// context consists of valid symbol names separated by and ending with a
/// backtick, so `"Units.m"` is an ordinary file path rather than one.
pub fn context_segments(name: &str) -> Option<Vec<&str>> {
  let body = name.strip_suffix('`')?;
  if body.is_empty() {
    return None;
  }
  let segments: Vec<&str> = body.split('`').collect();
  let is_symbol = |s: &&str| {
    !s.is_empty()
      && s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '$')
      && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '$')
  };
  segments.iter().all(is_symbol).then_some(segments)
}

/// Whether `name` is a context name — `Foo\`` or `Foo\`Bar\`` — belonging to
/// a package that ships with the Wolfram Language.
pub fn is_standard_distribution_context(name: &str) -> bool {
  context_segments(name).is_some_and(|segments| {
    STANDARD_DISTRIBUTION_CONTEXTS.contains(&segments[0])
  })
}
