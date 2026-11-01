//! Regressions found while making Wolfram Demonstrations notebooks run:
//! semantic bugs that surfaced as generic Part/list/argument messages.
use super::*;

mod internal_messages {
  use super::*;

  // The elimination in Solve/Reduce tries the branch where a denominator
  // vanishes and meets `1/0` there. That is the solver's own bookkeeping:
  // the message must not be generated at all, so `Check` does not fire, it
  // is not in `$MessageList`, and a front end (which shows the captured
  // messages) reports nothing.
  const SYSTEM: &str = "Solve[{8.314 == (22.4*p)/a + (22.4*p)/b, \
    0.0036639754618530535*a == (0.009871668311944718*p)^(2/7), \
    2269.12 == 44.8*p}, {a, b, p}]";

  #[test]
  fn solve_elimination_raises_no_messages() {
    clear_state();
    let result = interpret_with_stdout(SYSTEM).unwrap();
    assert!(
      result.result.starts_with("{{a -> 223.89"),
      "{}",
      result.result
    );
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
  }

  #[test]
  fn solve_elimination_does_not_trip_check() {
    clear_state();
    assert_eq!(
      interpret(&format!("Head[Check[{SYSTEM}, failed]]")).unwrap(),
      "List"
    );
    assert_eq!(interpret(&format!("{SYSTEM}; $MessageList")).unwrap(), "{}");
  }

  #[test]
  fn user_division_by_zero_still_reported() {
    clear_state();
    let result = interpret_with_stdout("1/0").unwrap();
    assert_eq!(result.result, "ComplexInfinity");
    assert!(result.warnings.iter().any(|w| w.contains("Power::infy")));
  }
}
