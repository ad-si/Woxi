//! The set of scripts the language server sweeps are run over.
//!
//! Every sweep means the same thing by "the repository's scripts", so the
//! rule for what belongs to that set lives here rather than being spelled
//! out again in each test.

use std::path::PathBuf;

/// Every `.wls` script that is part of the repository, sorted by path.
///
/// `_`-prefixed scripts are excluded: they are local scratch files,
/// gitignored as `/tests/scripts/_*`, so they are neither known-good
/// Wolfram Language nor visible to anyone else running the suite. A sweep
/// that included them would pass or fail depending on what happened to be
/// lying in the working tree.
pub(super) fn script_paths() -> Vec<PathBuf> {
  let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/scripts");
  let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
    .expect("tests/scripts is readable")
    .filter_map(|entry| {
      let path = entry.ok()?.path();
      if path.extension()?.to_str()? != "wls" {
        return None;
      }
      if path.file_name()?.to_str()?.starts_with('_') {
        return None;
      }
      Some(path)
    })
    .collect();
  paths.sort();
  assert!(
    paths.len() > 100,
    "expected the script suite, found {}",
    paths.len()
  );
  paths
}
