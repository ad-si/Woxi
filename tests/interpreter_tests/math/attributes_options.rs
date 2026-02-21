use super::*;

mod attributes {
  use super::*;

  #[test]
  fn plus() {
    assert_eq!(
      interpret("Attributes[Plus]").unwrap(),
      "{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"
    );
  }

  #[test]
  fn hold() {
    assert_eq!(
      interpret("Attributes[Hold]").unwrap(),
      "{HoldAll, Protected}"
    );
  }

  #[test]
  fn if_func() {
    assert_eq!(
      interpret("Attributes[If]").unwrap(),
      "{HoldRest, Protected}"
    );
  }

  #[test]
  fn set_func() {
    assert_eq!(
      interpret("Attributes[Set]").unwrap(),
      "{HoldFirst, Protected, SequenceHold}"
    );
  }

  #[test]
  fn and_func() {
    assert_eq!(
      interpret("Attributes[And]").unwrap(),
      "{Flat, HoldAll, OneIdentity, Protected}"
    );
  }

  #[test]
  fn constant_e() {
    assert_eq!(
      interpret("Attributes[E]").unwrap(),
      "{Constant, Protected, ReadProtected}"
    );
  }

  #[test]
  fn sin_func() {
    assert_eq!(
      interpret("Attributes[Sin]").unwrap(),
      "{Listable, NumericFunction, Protected}"
    );
  }

  #[test]
  fn unknown_func() {
    assert_eq!(interpret("Attributes[unknownfunc]").unwrap(), "{}");
  }

  #[test]
  fn string_arg() {
    assert_eq!(
      interpret("Attributes[\"Plus\"]").unwrap(),
      "{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"
    );
  }

  #[test]
  fn non_symbol_arg_returns_unevaluated() {
    assert_eq!(
      interpret("Attributes[a + b + c]").unwrap(),
      "Attributes[a + b + c]"
    );
  }

  #[test]
  fn hold_complete() {
    assert_eq!(
      interpret("Attributes[HoldComplete]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Attributes[Unevaluated]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }
}

mod options {
  use super::*;

  #[test]
  fn set_and_get() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f]").unwrap(),
      "{a -> 1, b -> 2}"
    );
  }

  #[test]
  fn get_specific_option() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, a]").unwrap(),
      "{a -> 1}"
    );
  }

  #[test]
  fn get_second_option() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, b]").unwrap(),
      "{b -> 2}"
    );
  }

  #[test]
  fn unknown_function() {
    assert_eq!(interpret("Options[unknownfunc]").unwrap(), "{}");
  }

  #[test]
  fn option_not_found() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, c]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn overwrite_options() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> 1}; Options[f] = {a -> 10, b -> 20}; Options[f]"
      )
      .unwrap(),
      "{a -> 10, b -> 20}"
    );
  }

  #[test]
  fn single_rule() {
    assert_eq!(
      interpret("Options[g] = {x -> 42}; Options[g]").unwrap(),
      "{x -> 42}"
    );
  }

  #[test]
  fn multiple_functions() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> 1}; Options[g] = {b -> 2}; {Options[f], Options[g]}"
      )
      .unwrap(),
      "{{a -> 1}, {b -> 2}}"
    );
  }
}
