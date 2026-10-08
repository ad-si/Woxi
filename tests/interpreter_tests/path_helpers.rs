//! Platform-independent paths for tests that read or write files. Shared by
//! all `interpreter_tests*` binaries (see `tests/interpreter_tests.rs`).
#![allow(dead_code)]

// On Windows, if a path has components starting with
// n, r, t, the backslashes look like escape sequences,
// and the path doesn't work when passed to interpret.
// Patching paths in the test cases is whack-a-mole, so
// just use a Unix-style path syntax always.
// C:/tmp/foo/bar.txt works fine on Windows.
pub(crate) fn unixify(path: &str) -> String {
  path.replace('\\', "/")
}

pub(crate) fn temp_dir() -> String {
  let mut tmp = std::env::temp_dir().display().to_string();
  if tmp.ends_with(std::path::MAIN_SEPARATOR) {
    tmp.pop();
  }
  unixify(&tmp)
}

/// A scratch path inside the platform temp directory. Never hardcode
/// `/tmp/...` in a test — it does not exist on Windows, where the
/// nightly CI runs the full unit suite.
pub(crate) fn temp_file(file: &str) -> String {
  let tmp = std::env::temp_dir().join(file);
  unixify(&tmp.display().to_string())
}

pub(crate) fn manifest_file(file: &str) -> String {
  let manifest = env!("CARGO_MANIFEST_DIR");
  unixify(&format!("{manifest}/{file}"))
}
