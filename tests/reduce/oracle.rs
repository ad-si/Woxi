//! Tests in this module are intentionally ignored by the self-contained suite.
//! `make test-reduce-oracle` runs them when `wolframscript` is provisioned.

use std::process::Command;

use woxi::interpret;

#[test]
#[ignore = "requires wolframscript development oracle"]
fn curated_wolfram_surface_agreement() {
  let cases = [
    "Reduce[2*x + 1 == 0, x, Reals]",
    "Reduce[1/3 < x && x <= 5/7, x, Rationals]",
    "Reduce[x > 2, x, Integers]",
    "Reduce[2 < x < 5, x, Integers]",
    "Reduce[!(3*x < 1), x, Reals]",
  ];

  for input in cases {
    let woxi = interpret(input).unwrap();
    let output = Command::new("wolframscript")
      .args(["-code", input])
      .output()
      .expect("wolframscript must be available for the oracle target");
    assert!(
      output.status.success(),
      "wolframscript failed for {input}: {}",
      String::from_utf8_lossy(&output.stderr)
    );
    let wolfram = String::from_utf8(output.stdout)
      .expect("wolframscript output must be UTF-8");
    assert_eq!(woxi, wolfram.trim(), "input: {input}");
  }
}
