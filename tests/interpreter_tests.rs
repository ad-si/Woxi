use woxi::{
  clear_state, interpret, interpret_with_stdout, split_into_statements,
};

mod interpreter_tests {
  use super::*;

  #[test]
  fn test_split_single_expression() {
    assert_eq!(split_into_statements("1 + 2"), vec!["1 + 2"]);
  }

  #[test]
  fn test_split_multiple_lines() {
    assert_eq!(
      split_into_statements(
        "Graphics[Circle[]]\n1 + 3\nGraphics[Rectangle[]]\n5 * 8"
      ),
      vec![
        "Graphics[Circle[]]",
        "1 + 3",
        "Graphics[Rectangle[]]",
        "5 * 8"
      ]
    );
  }

  #[test]
  fn test_split_preserves_multiline_brackets() {
    assert_eq!(
      split_into_statements("Module[{a = 1},\n  a + 2\n]\n3 + 4"),
      vec!["Module[{a = 1},\n  a + 2\n]", "3 + 4"]
    );
  }

  #[test]
  fn test_split_preserves_set_delayed_continuation() {
    assert_eq!(
      split_into_statements("f[x_] :=\n  x^2\nf[3]"),
      vec!["f[x_] :=\n  x^2", "f[3]"]
    );
  }

  #[test]
  fn test_split_semicolon_lines() {
    assert_eq!(
      split_into_statements("a = 5;\nb = 10;\na + b"),
      vec!["a = 5;", "b = 10;", "a + b"]
    );
  }

  #[test]
  fn test_split_blank_lines() {
    assert_eq!(
      split_into_statements("1 + 2\n\n3 + 4"),
      vec!["1 + 2", "3 + 4"]
    );
  }

  #[test]
  fn test_split_trailing_comment_only() {
    // A trailing comment-only line should not produce a separate statement
    assert_eq!(
      split_into_statements("Sin[123]\n(* comment *)"),
      vec!["Sin[123]"]
    );
  }

  #[test]
  fn test_split_leading_comment_only() {
    // A leading comment-only line should be merged with the next code line
    assert_eq!(
      split_into_statements("(* comment *)\nSin[123]"),
      vec!["(* comment *)\nSin[123]"]
    );
  }

  #[test]
  fn test_split_comment_between_expressions() {
    // A comment between two expressions should not produce an extra statement
    assert_eq!(
      split_into_statements("1 + 1\n(* comment *)\n2 + 2"),
      vec!["1 + 1", "(* comment *)\n2 + 2"]
    );
  }

  #[test]
  fn test_split_multiple_comments_only() {
    // Multiple comment-only lines should not produce statements
    assert_eq!(split_into_statements("(* c1 *)\n(* c2 *)"), vec![""]);
  }

  #[test]
  fn test_split_preserves_multiline_association() {
    assert_eq!(
      split_into_statements(
        "a = <|\n  \"x\" -> 1,\n  \"y\" -> 2\n|>\nPrint[a]"
      ),
      vec!["a = <|\n  \"x\" -> 1,\n  \"y\" -> 2\n|>", "Print[a]"]
    );
  }

  #[test]
  fn test_split_preserves_nested_multiline_association() {
    assert_eq!(
      split_into_statements("a = <|\"x\" -> <|\n  \"n\" -> 42\n|>|>\nPrint[a]"),
      vec!["a = <|\"x\" -> <|\n  \"n\" -> 42\n|>|>", "Print[a]"]
    );
  }

  #[test]
  fn test_split_backslash_line_continuation() {
    assert_eq!(
      split_into_statements(
        "ImaginaryQ[u_] :=\\\n  Head[u]===Complex && Re[u]===0\nImaginaryQ[3 I]"
      ),
      vec![
        "ImaginaryQ[u_] :=  Head[u]===Complex && Re[u]===0",
        "ImaginaryQ[3 I]"
      ]
    );
  }

  #[test]
  fn test_split_backslash_continuation_multi() {
    assert_eq!(split_into_statements("1 +\\\n2 +\\\n3"), vec!["1 +2 +3"]);
  }

  #[test]
  fn test_comment_only_input() {
    // A standalone comment should not cause an error
    clear_state();
    let result = interpret("(* comment *)");
    assert!(result.is_err());
    assert!(matches!(result, Err(woxi::InterpreterError::EmptyInput)));
  }

  #[test]
  fn test_expression_then_comment() {
    // Expression followed by comment should evaluate the expression
    clear_state();
    assert_eq!(interpret("Sin[123]\n(* comment *)").unwrap(), "Sin[123]");
  }

  #[test]
  fn test_comment_then_expression() {
    // Comment followed by expression should evaluate the expression
    clear_state();
    assert_eq!(interpret("(* comment *)\nSin[123]").unwrap(), "Sin[123]");
  }

  #[test]
  fn test_inline_comment() {
    // Inline comment should not affect the result
    clear_state();
    assert_eq!(interpret("5 + (* inline *) 3").unwrap(), "8");
  }

  #[test]
  fn test_comment_after_condition_operator() {
    // A comment after /; should not cause an infinite loop
    clear_state();
    assert_eq!(interpret("x /; (* foo *) True").unwrap(), "x /; True");
  }

  #[test]
  fn test_comment_after_condition_in_set_delayed() {
    // SetDelayed with Condition and inline comment should work
    clear_state();
    assert_eq!(
      interpret("f[x_] := x^2 /; (* positive *) True; f[3]").unwrap(),
      "9"
    );
  }

  #[test]
  fn test_condition_in_module_as_guard() {
    // Condition inside Module should act as a guard for the function definition.
    // If test is True, return the value. If not, the overload doesn't match.
    clear_state();
    interpret("Foo[u_,x_Symbol] := Module[{}, 3 /; u == 1]").unwrap();
    // x != 1, so Foo[x, x] should remain unevaluated
    assert_eq!(interpret("Foo[x, x]").unwrap(), "Foo[x, x]");
    // 1 == 1 is True, so Foo[1, x] should return 3
    assert_eq!(interpret("Foo[1, x]").unwrap(), "3");
  }

  #[test]
  fn test_condition_in_block_as_guard() {
    // Same behavior with Block instead of Module
    clear_state();
    interpret("Bar[n_] := Block[{}, n^2 /; n > 0]").unwrap();
    assert_eq!(interpret("Bar[3]").unwrap(), "9");
    assert_eq!(interpret("Bar[-1]").unwrap(), "Bar[-1]");
  }

  #[test]
  fn test_condition_in_module_with_expression() {
    // Condition guard with non-trivial expression in Module
    clear_state();
    interpret("Sqr[n_] := Module[{}, n^2 /; n > 0]").unwrap();
    assert_eq!(interpret("Sqr[3]").unwrap(), "9");
    assert_eq!(interpret("Sqr[-1]").unwrap(), "Sqr[-1]");
  }

  #[test]
  fn test_nested_comment() {
    clear_state();
    assert_eq!(
      interpret("1 + (* outer (* inner *) outer *) 2").unwrap(),
      "3"
    );
  }

  #[test]
  fn test_nested_comment_only() {
    clear_state();
    assert!(interpret("(* outer (* inner *) *)").is_err());
  }

  #[test]
  fn test_deeply_nested_comment() {
    clear_state();
    assert_eq!(
      interpret("10 + (* a (* b (* c *) b *) a *) 5").unwrap(),
      "15"
    );
  }

  #[test]
  fn test_nested_comment_multiline() {
    clear_state();
    assert_eq!(
      interpret("1 + (* outer\n(* inner *)\nouter *) 2").unwrap(),
      "3"
    );
  }

  #[test]
  fn test_split_nested_comment() {
    assert_eq!(
      split_into_statements("1 + 1\n(* outer (* inner *) *)\n2 + 2"),
      vec!["1 + 1", "(* outer (* inner *) *)\n2 + 2"]
    );
  }

  mod algebra;
  mod arithmetic;
  mod association;
  mod calculus;
  mod cellular_automaton;
  mod column;
  mod control_flow;
  mod dataset;
  mod datetime;
  mod functions;
  mod graphics;
  mod image;
  mod interval;
  mod io;
  mod linear_algebra;
  mod list;
  mod math;
  mod property;
  mod quantity;
  mod statistics;
  mod string;
  mod syntax;
  mod tabular;
}
