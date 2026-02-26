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
  mod io;
  mod linear_algebra;
  mod list;
  mod math;
  mod quantity;
  mod statistics;
  mod string;
  mod syntax;
  mod tabular;
}
