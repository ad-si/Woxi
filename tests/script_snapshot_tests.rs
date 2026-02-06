use std::fs;
use std::path::Path;
use woxi::{
  clear_state, interpret_with_stdout, set_script_command_line, without_shebang,
};

/// Run a script and snapshot its Print[] output.
///
/// By default, runs via the woxi interpreter.
/// Set WOXI_USE_WOLFRAM=true to run via `wolframscript -file` instead,
/// validating that wolframscript produces the same output:
///
///   WOXI_USE_WOLFRAM=true cargo test script_
fn run_script_snapshot(name: &str) {
  run_script_snapshot_with_args(name, &[]);
}

fn run_script_snapshot_with_args(name: &str, args: &[&str]) {
  let path = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests/scripts")
    .join(name);
  let use_wolfram = std::env::var("WOXI_USE_WOLFRAM").as_deref() == Ok("true");

  let stdout = if use_wolfram {
    let mut cmd = std::process::Command::new("wolframscript");
    cmd.arg("-file").arg(&path);
    for arg in args {
      cmd.arg(arg);
    }
    let output = cmd.output().unwrap_or_else(|e| {
      panic!("Failed to run wolframscript on {}: {}", name, e)
    });

    assert!(
      output.status.success(),
      "wolframscript failed on {}: {}",
      name,
      String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8_lossy(&output.stdout).into_owned()
  } else {
    clear_state();

    // Set $ScriptCommandLine: first element is the script path, rest are args
    let mut cmd_line = vec![path.to_string_lossy().to_string()];
    cmd_line.extend(args.iter().map(|s| s.to_string()));
    set_script_command_line(&cmd_line);

    let content = fs::read_to_string(&path).unwrap();
    let code = without_shebang(&content);

    let result = interpret_with_stdout(&code)
      .unwrap_or_else(|e| panic!("Script {} failed: {}", name, e));

    result.stdout
  };

  insta::assert_snapshot!(name, stdout);
}

macro_rules! script_test {
  ($test_name:ident, $file:expr) => {
    #[test]
    fn $test_name() {
      run_script_snapshot($file);
    }
  };
}

script_test!(script_99_bottles_of_beer, "99_bottles_of_beer.wls");
script_test!(script_abc_problem, "abc_problem.wls");
script_test!(script_fizzbuzz_1, "fizzbuzz_1.wls");
script_test!(script_fizzbuzz_2, "fizzbuzz_2.wls");
script_test!(script_fizzbuzz_3, "fizzbuzz_3.wls");
script_test!(script_fizzbuzz_4, "fizzbuzz_4.wls");
script_test!(script_fizzbuzz_5, "fizzbuzz_5.wls");
script_test!(script_hello_world, "hello_world.wls");
script_test!(script_n_queens_problem_1, "n-queens_problem_1.wls");

#[test]
fn script_cli_args() {
  run_script_snapshot_with_args("cli_args.wls", &["5"]);
}
