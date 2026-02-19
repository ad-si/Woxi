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

  mod algebra;
  mod arithmetic;
  mod association;
  mod calculus;
  mod cellular_automaton;
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
}
