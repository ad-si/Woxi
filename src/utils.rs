#[cfg(not(target_arch = "wasm32"))]
use rand::distributions::{Alphanumeric, Distribution};

#[cfg(not(target_arch = "wasm32"))]
fn rand_str(length: usize) -> String {
  let rng = rand::thread_rng();
  let characters: Vec<char> = Alphanumeric
    .sample_iter(rng)
    .map(|c| c.into())
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

  std::fs::OpenOptions::new()
    .create_new(true)
    .write(true)
    .truncate(true)
    .open(&file_path)
    .map(|_| file_path)
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
