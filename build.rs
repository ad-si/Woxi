use std::process::Command;

fn main() {
  // Re-run when the checked-out commit changes (commit, branch switch,
  // rebase). Deliberately does NOT track .git/index: staging files with
  // `git add` would otherwise re-stamp WOXI_GIT_VERSION and force a rebuild
  // and relink of the whole crate plus every test binary. The trade-off is
  // that the `-dirty` suffix reflects the working tree as of the last
  // compile triggered by a source change, which is accurate whenever it
  // matters (any rebuild re-evaluates it).
  println!("cargo:rerun-if-changed=.git/HEAD");
  // HEAD is usually a symbolic ref; track the branch ref file it points to
  // so new commits on the current branch refresh the version stamp.
  if let Ok(head) = std::fs::read_to_string(".git/HEAD")
    && let Some(git_ref) = head.strip_prefix("ref: ")
  {
    let ref_path = format!(".git/{}", git_ref.trim());
    if std::path::Path::new(&ref_path).exists() {
      println!("cargo:rerun-if-changed={}", ref_path);
    }
  }

  let version = Command::new("git")
    .args(["describe", "--always", "--long", "--tags", "--dirty"])
    .output()
    .ok()
    .and_then(|out| {
      if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
      } else {
        None
      }
    })
    .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());

  println!("cargo:rustc-env=WOXI_GIT_VERSION={}", version);
}
