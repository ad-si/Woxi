//! The Wolfram working directory, and path resolution against it.
//!
//! `SetDirectory` / `ResetDirectory` move a *virtual* working directory: we
//! deliberately do NOT call `std::env::set_current_dir`, because that mutates
//! process-wide state and cargo runs tests in parallel threads within a
//! single process — mutating the real CWD from one test races against any
//! other test that resolves a relative path, causing flaky failures in CI.
//! Instead the directory is a per-thread stack; the top of the stack is what
//! `Directory[]` reports, and the process CWD is the fallback when the stack
//! is empty.
//!
//! The consequence is that `std::fs` must never see a relative path taken
//! from evaluated Wolfram code: `std::fs` resolves against the process CWD,
//! which is not where `Directory[]` points. Every such path goes through
//! [`resolve`] first — that is the single place the working directory is
//! applied.

use std::path::{Path, PathBuf};

thread_local! {
  static DIRECTORY_STACK: std::cell::RefCell<Vec<String>> =
    const { std::cell::RefCell::new(Vec::new()) };
}

/// The current working directory as `Directory[]` reports it: the top of the
/// virtual stack, or the process CWD while the stack is empty.
pub fn current_dir() -> String {
  DIRECTORY_STACK
    .with(|s| s.borrow().last().cloned())
    .unwrap_or_else(|| {
      std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
    })
}

/// Make `dir` — which must already be absolute — the working directory.
pub fn push_dir(dir: String) {
  DIRECTORY_STACK.with(|s| s.borrow_mut().push(dir));
}

/// Restore the previous working directory, returning the one left behind.
/// `None` when the stack is already empty.
pub fn pop_dir() -> Option<String> {
  DIRECTORY_STACK.with(|s| s.borrow_mut().pop())
}

/// The directory stack as `DirectoryStack[]` reports it, oldest first.
pub fn directory_stack() -> Vec<String> {
  DIRECTORY_STACK.with(|s| s.borrow().clone())
}

/// `path` as the file system should see it: absolute paths are used as
/// given, relative ones are taken against the working directory that
/// [`current_dir`] reports rather than the process CWD.
pub fn resolve(path: impl AsRef<Path>) -> PathBuf {
  let path = path.as_ref();
  if path.is_absolute() {
    path.to_path_buf()
  } else {
    PathBuf::from(current_dir()).join(path)
  }
}

/// Whether something exists at `path`, resolved against the working
/// directory.
pub fn exists(path: impl AsRef<Path>) -> bool {
  resolve(path).exists()
}

/// Whether `path`, resolved against the working directory, is a directory.
pub fn is_dir(path: impl AsRef<Path>) -> bool {
  resolve(path).is_dir()
}

/// Whether `path`, resolved against the working directory, is a regular
/// file.
pub fn is_file(path: impl AsRef<Path>) -> bool {
  resolve(path).is_file()
}
