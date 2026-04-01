use super::*;

mod arg_count_errors {
  use super::*;

  #[test]
  fn exact_one_arg_function() {
    // argx format: exactly 1 argument expected
    clear_state();
    let result = interpret_with_stdout("Sin[1, 2]").unwrap();
    assert_eq!(result.result, "Sin[1, 2]");
    assert!(result.warnings[0].contains(
      "Sin::argx: Sin called with 2 arguments; 1 argument is expected."
    ));
  }

  #[test]
  fn exact_two_arg_function() {
    // argrx format: exactly N (>1) arguments expected
    clear_state();
    let result = interpret_with_stdout("Divide[1, 2, 3]").unwrap();
    assert_eq!(result.result, "Divide[1, 2, 3]");
    assert!(result.warnings[0].contains(
      "Divide::argrx: Divide called with 3 arguments; 2 arguments are expected."
    ));
  }

  #[test]
  fn between_range_function() {
    // argb format: between min and max
    clear_state();
    let result = interpret_with_stdout("If[1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.result, "If[1, 2, 3, 4, 5]");
    assert!(result.warnings[0]
      .contains("If::argb: If called with 5 arguments; between 2 and 4 arguments are expected."));
  }

  #[test]
  fn or_range_function() {
    // argt format: min or max arguments expected
    clear_state();
    let result = interpret_with_stdout("Sort[1, 2, 3]").unwrap();
    assert_eq!(result.result, "Sort[1, 2, 3]");
    assert!(result.warnings[0].contains(
      "Sort::argt: Sort called with 3 arguments; 1 or 2 arguments are expected."
    ));
  }

  #[test]
  fn singular_argument_text() {
    // "1 argument" (singular) when called with 1 arg
    clear_state();
    let result = interpret_with_stdout("If[1]").unwrap();
    assert_eq!(result.result, "If[1]");
    assert!(result.warnings[0].contains("called with 1 argument;"));
  }

  #[test]
  fn no_error_for_valid_args() {
    // Valid call should produce no warnings
    clear_state();
    let result = interpret_with_stdout("Sin[1]").unwrap();
    assert!(
      result.warnings.is_empty(),
      "Expected no warnings for valid Sin[1], got: {:?}",
      result.warnings
    );
  }

  #[test]
  fn no_error_for_unknown_function() {
    // Unknown function should not trigger arg count error
    clear_state();
    let result = interpret_with_stdout("MyFunc[1, 2, 3, 4, 5]").unwrap();
    assert_eq!(result.result, "MyFunc[1, 2, 3, 4, 5]");
    assert!(
      result.warnings.iter().all(|w| !w.contains("::arg")),
      "Expected no arg count warnings for unknown function"
    );
  }

  #[test]
  fn quiet_too_many_args() {
    clear_state();
    let result = interpret_with_stdout("Quiet[1, 2, 3, 4]").unwrap();
    assert_eq!(result.result, "Quiet[1, 2, 3, 4]");
    assert!(result.warnings[0]
      .contains("Quiet::argb: Quiet called with 4 arguments; between 1 and 3 arguments are expected."));
  }
}
