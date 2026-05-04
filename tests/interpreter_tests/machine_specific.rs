// Tests whose expected values depend on the host machine (current
// user, home directory, hostname, OS, …). They are kept out of
// `test_cases.rs` and excluded from `make test` via the nextest
// filter `!test(machine_specific)`, because outside a controlled
// environment the host-derived values are unpredictable.
//
// To run them, use the Docker target which rebuilds a deterministic
// capture environment (USER, HOME, hostname, WORKDIR, OS) inside a
// Linux container. The expected values below are pinned to that
// environment:
//
//   make test-docker
//
// The container is configured with:
//   USER=woxi
//   HOME=/home/woxi
//   hostname=woxi-test
//   WORKDIR=/home/woxi/woxi
//   (Linux base image → `$OperatingSystem` resolves to "Unix")
//
// Cases that depend on the host's CPU architecture (`$SystemID`) or
// physical RAM (`$SystemMemory`) are intentionally not asserted here
// — `tests/interpreter_tests/math/numeric.rs` covers them with
// type/range checks (`Head[$SystemID] == String`, `$SystemMemory > 0`)
// that work on any host.

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

  #[test]
  fn environment_home() {
    assert_eval(r#"Environment["HOME"]"#, r#""/home/woxi""#);
  }

  #[test]
  fn machine_name() {
    assert_eval(r#"$MachineName"#, r#""woxi-test""#);
  }

  #[test]
  fn user_name() {
    assert_eval(r#"$UserName"#, r#""woxi""#);
  }

  #[test]
  fn home_directory() {
    assert_eval(r#"$HomeDirectory"#, r#""/home/woxi""#);
  }

  #[test]
  fn user_base_directory() {
    assert_eval(r#"$UserBaseDirectory"#, r#""/home/woxi/.Wolfram""#);
  }

  #[test]
  fn base_directory() {
    assert_eval(r#"$BaseDirectory"#, r#""/usr/share/Wolfram""#);
  }

  #[test]
  fn initial_directory() {
    assert_eval(r#"$InitialDirectory"#, r#""/home/woxi/woxi""#);
  }

  #[test]
  fn temporary_directory() {
    assert_eval(r#"$TemporaryDirectory"#, r#""/tmp""#);
  }

  #[test]
  fn parent_directory() {
    assert_eval(r#"ParentDirectory[]"#, r#""/home/woxi""#);
  }

  #[test]
  fn expand_file_name() {
    assert_eval(
      r#"ExpandFileName["ExampleData/sunflowers.jpg"]"#,
      r#""/home/woxi/woxi/ExampleData/sunflowers.jpg""#,
    );
  }

  #[test]
  fn operating_system() {
    assert_eval(r#"$OperatingSystem"#, r#""Unix""#);
  }
}
