// Tests for host-derived symbols (`$UserName`, `$MachineName`,
// `$HomeDirectory`, …). Expected values are computed dynamically
// from the same OS facilities the implementation in
// `src/evaluator/listable.rs` consults (env vars, `gethostname`,
// `current_dir`, `cfg!(target_os = …)`), so the assertions hold on
// any host without a controlled capture environment.
//
// What the tests catch even though the expected value is host-derived:
//   * the symbol resolves at all (no Identifier passthrough),
//   * the result is a String of the expected shape (quoting, trailing
//     slash policy, short-hostname stripping, …),
//   * the implementation reads the right source (e.g. `$UserName`
//     from `$USER`, not `$LOGNAME`),
//   * platform branches stay in sync (e.g. `$OperatingSystem` returns
//     "MacOSX" on macOS and "Unix" on Linux).

use super::*;

mod machine_specific {
  use super::*;

  fn normalise(s: &str) -> String {
    s.chars()
      .filter(|c| !c.is_whitespace() && *c != '"')
      .collect()
  }

  fn assert_eval(input: &str, expected: &str) {
    clear_state();
    let actual = match interpret(input) {
      Ok(s) => s,
      Err(e) => panic!(
        "Woxi returned error: {:?}\n  input:    {}\n  expected: {}",
        e, input, expected
      ),
    };
    if normalise(&actual) != normalise(expected) {
      panic!(
        "output mismatch\n  input:    {}\n  expected: {}\n  actual:   {}",
        input, expected, actual
      );
    }
  }

  fn host_home() -> String {
    std::env::var("HOME")
      .or_else(|_| std::env::var("USERPROFILE"))
      .expect("HOME (or USERPROFILE) must be set in the test environment")
  }

  fn host_user() -> String {
    std::env::var("USER")
      .or_else(|_| std::env::var("USERNAME"))
      .expect("USER (or USERNAME) must be set in the test environment")
  }

  /// Short hostname, stripped of trailing `.local` / `.lan` etc, to
  /// mirror `$MachineName` in `src/evaluator/listable.rs`.
  #[cfg(unix)]
  fn host_machine_name() -> String {
    let mut buf = [0u8; 256];
    let ret = unsafe {
      libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len())
    };
    assert_eq!(ret, 0, "gethostname() failed");
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let host =
      std::str::from_utf8(&buf[..len]).expect("hostname is not valid UTF-8");
    host.split('.').next().unwrap_or(host).to_string()
  }

  fn host_initial_dir() -> String {
    std::env::current_dir()
      .expect("current_dir() failed")
      .to_string_lossy()
      .into_owned()
  }

  fn host_temp_dir() -> String {
    let tmp = std::env::temp_dir();
    let canon = std::fs::canonicalize(&tmp).unwrap_or(tmp);
    let mut s = canon.to_string_lossy().into_owned();
    while s.len() > 1 && s.ends_with('/') {
      s.pop();
    }
    s
  }

  #[test]
  fn environment_home() {
    assert_eval(r#"Environment["HOME"]"#, &format!(r#""{}""#, host_home()));
  }

  #[cfg(unix)]
  #[test]
  fn machine_name() {
    assert_eval(r#"$MachineName"#, &format!(r#""{}""#, host_machine_name()));
  }

  #[test]
  fn user_name() {
    assert_eval(r#"$UserName"#, &format!(r#""{}""#, host_user()));
  }

  #[test]
  fn home_directory() {
    assert_eval(r#"$HomeDirectory"#, &format!(r#""{}""#, host_home()));
  }

  #[test]
  fn user_base_directory() {
    let sub = if cfg!(target_os = "macos") {
      "Library/Wolfram"
    } else if cfg!(target_os = "windows") {
      "AppData\\Roaming\\Wolfram"
    } else {
      ".Wolfram"
    };
    let expected =
      format!(r#""{}/{}""#, host_home().trim_end_matches('/'), sub);
    assert_eval(r#"$UserBaseDirectory"#, &expected);
  }

  #[test]
  fn base_directory() {
    let root = if cfg!(target_os = "macos") {
      "/Library/Wolfram"
    } else if cfg!(target_os = "windows") {
      "C:\\ProgramData\\Wolfram"
    } else {
      "/usr/share/Wolfram"
    };
    assert_eval(r#"$BaseDirectory"#, &format!(r#""{}""#, root));
  }

  #[test]
  fn initial_directory() {
    assert_eval(
      r#"$InitialDirectory"#,
      &format!(r#""{}""#, host_initial_dir()),
    );
  }

  #[test]
  fn temporary_directory() {
    assert_eval(
      r#"$TemporaryDirectory"#,
      &format!(r#""{}""#, host_temp_dir()),
    );
  }

  #[test]
  fn parent_directory() {
    let parent = std::path::PathBuf::from(host_initial_dir())
      .parent()
      .expect("current_dir has no parent")
      .to_string_lossy()
      .into_owned();
    assert_eval(r#"ParentDirectory[]"#, &format!(r#""{}""#, parent));
  }

  #[test]
  fn expand_file_name() {
    let expected = format!(
      r#""{}/ExampleData/sunflowers.jpg""#,
      host_initial_dir().trim_end_matches('/'),
    );
    assert_eval(r#"ExpandFileName["ExampleData/sunflowers.jpg"]"#, &expected);
  }

  #[test]
  fn operating_system() {
    let os = if cfg!(target_os = "macos") {
      "MacOSX"
    } else if cfg!(target_os = "linux") {
      "Unix"
    } else if cfg!(target_os = "windows") {
      "Windows"
    } else {
      "Unknown"
    };
    assert_eval(r#"$OperatingSystem"#, &format!(r#""{}""#, os));
  }
}
