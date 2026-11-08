//! Public distributions must agree before a release is tagged.
use std::path::Path;

#[test]
fn distribution_versions_match_the_interpreter() {
  let root = Path::new(env!("CARGO_MANIFEST_DIR"));
  let version = env!("CARGO_PKG_VERSION");
  for file in [
    "woxi-py/Cargo.toml",
    "woxi-studio/Cargo.toml",
    "jupyterlite-woxi-kernel/pyproject.toml",
  ] {
    let text = std::fs::read_to_string(root.join(file)).unwrap();
    let actual = text
      .lines()
      .find(|line| line.starts_with("version = "))
      .unwrap();
    assert_eq!(actual, format!("version = \"{version}\""), "{file}");
  }
  for file in [
    "npm/package.json",
    "jupyterlite-woxi-kernel/package.json",
    "jupyterlite-woxi-kernel/package-lock.json",
  ] {
    let text = std::fs::read_to_string(root.join(file)).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["version"], version, "{file}");
    if file.ends_with("package-lock.json") {
      assert_eq!(json["packages"][""]["version"], version, "{file}");
    }
  }
  let kernel =
    std::fs::read_to_string(root.join("jupyterlite-woxi-kernel/src/kernel.ts"))
      .unwrap();
  for field in ["implementation_version", "version"] {
    assert!(
      kernel
        .lines()
        .any(|line| line.trim() == format!("{field}: '{version}',")),
      "kernel {field}"
    );
  }
  let plist =
    std::fs::read_to_string(root.join("woxi-studio/macos/Info.plist")).unwrap();
  for key in ["CFBundleShortVersionString", "CFBundleVersion"] {
    let after_key = plist
      .split_once(&format!("<key>{key}</key>"))
      .unwrap()
      .1
      .trim_start();
    assert!(
      after_key.starts_with(&format!("<string>{version}</string>")),
      "{key}"
    );
  }
}
