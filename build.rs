use std::process::Command;

fn main() {
  // Re-run if HEAD or any ref changes.
  println!("cargo:rerun-if-changed=.git/HEAD");
  println!("cargo:rerun-if-changed=.git/index");

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
