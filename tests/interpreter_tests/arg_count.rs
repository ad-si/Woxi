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
  fn exact_two_arg_function_called_with_one() {
    // argr format: exactly N (>1) arguments expected but called with
    // exactly one argument. wolframscript uses `argr` here, not `argrx`.
    clear_state();
    let result = interpret_with_stdout("NumericalOrder[a]").unwrap();
    assert_eq!(result.result, "NumericalOrder[a]");
    assert!(result.warnings[0].contains(
      "NumericalOrder::argr: NumericalOrder called with 1 argument; 2 arguments are expected."
    ));
  }

  #[test]
  fn exact_two_arg_function_called_with_zero() {
    // Zero args (not one) keeps the argrx tag.
    clear_state();
    let result = interpret_with_stdout("NumericalOrder[]").unwrap();
    assert_eq!(result.result, "NumericalOrder[]");
    assert!(result.warnings[0].contains(
      "NumericalOrder::argrx: NumericalOrder called with 0 arguments; 2 arguments are expected."
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

mod singular_argument_tags {
  use super::*;

  #[test]
  fn one_argument_uses_argtu_tag() {
    // Regression: wolframscript uses ::argtu (not ::argt) when a
    // range-arity function is called with exactly one argument
    assert_eq!(interpret("Insert[x]").unwrap(), "Insert[x]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Insert::argtu: Insert called with 1 argument; 2 or 3 arguments are expected."
      )),
      "expected argtu message, got {:?}",
      msgs
    );
  }

  #[test]
  fn one_argument_uses_argbu_tag() {
    assert_eq!(interpret("Array[f]").unwrap(), "Array[f]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Array::argbu: Array called with 1 argument; between 2 and 4 arguments are expected."
      )),
      "expected argbu message, got {:?}",
      msgs
    );
  }

  #[test]
  fn insert_two_argument_operator_form_is_silent() {
    // Regression: Insert[a, b] is the valid operator form, previously
    // warned Insert::argrx
    assert_eq!(interpret("Insert[a, b]").unwrap(), "Insert[a, b]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }

  #[test]
  fn optimization_functions_accept_two_to_four_args() {
    // Regression: ArgMax/ArgMin/Maximize/Minimize/MaxValue/MinValue accept
    // between 2 and 4 arguments (f, vars[, dom][, opts]). Previously Woxi
    // declared a 2-3 range and reported the ::argt/::argtu "2 or 3 arguments"
    // message; wolframscript uses ::argb/::argbu "between 2 and 4 arguments".
    for f in [
      "ArgMax", "ArgMin", "Maximize", "Minimize", "MaxValue", "MinValue",
    ] {
      // One argument → argbu (singular form).
      clear_state();
      assert_eq!(interpret(&format!("{f}[x]")).unwrap(), format!("{f}[x]"));
      let msgs = woxi::get_captured_messages_raw();
      let expected = format!(
        "{f}::argbu: {f} called with 1 argument; \
         between 2 and 4 arguments are expected."
      );
      assert!(
        msgs.iter().any(|m| m.contains(&expected)),
        "expected argbu message for {f}, got {:?}",
        msgs
      );

      // Zero arguments → argb (plural form).
      clear_state();
      assert_eq!(interpret(&format!("{f}[]")).unwrap(), format!("{f}[]"));
      let msgs = woxi::get_captured_messages_raw();
      let expected = format!(
        "{f}::argb: {f} called with 0 arguments; \
         between 2 and 4 arguments are expected."
      );
      assert!(
        msgs.iter().any(|m| m.contains(&expected)),
        "expected argb message for {f}, got {:?}",
        msgs
      );
    }
  }
}
