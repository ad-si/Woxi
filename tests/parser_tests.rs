use woxi::parse;

#[cfg(test)]
mod tests {
  use woxi::Rule;

  use super::*;
  #[test]
  fn test_parse_calculation() {
    let input = "1 + 2";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_symbolic_calculation() {
    let input = "x + 2";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_constant() {
    let input = "Sin[Pi]";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_unicode_pi() {
    let input = "Sin[π]";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_function_definition() {
    let input = "f[x_] := x^2 + 2*x + 1";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_function_call() {
    let input = "Plot[f[x], {x, -2, 2}]";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_complex_expression() {
    let input = "3*x^2 + 2*x + 1";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_list() {
    let input = "{1, 2, 3, 4, 5}";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_symbolic_list() {
    let input = "{a, b, c}";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_boolean_list() {
    let input = "{True, False, False}";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_nested_function_calls() {
    let input = "Cos[Sin[x]]";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_complex_nested_function_calls() {
    let input = "Plot[Sin[x], {x, -Pi, Pi}]";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_program() {
    let input = "1 + 2";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_expression() {
    let input = "1 + 2";
    let program = parse(input).unwrap().next().unwrap();
    let expression = program.into_inner().next().unwrap();
    assert_eq!(expression.as_rule(), Rule::Expression);
  }

  #[test]
  fn test_parse_leading_dot_real_literal() {
    let input = "Hue[.9, .3]";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_part_span() {
    let input = "{a, b, c, d}[[2;;3]]";
    let pair = parse(input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_deeply_nested_function_calls() {
    // Regression: deeply nested function calls must parse in linear time, not O(2^d).
    // At 30 levels, exponential backtracking would take hours; this must finish instantly.
    let input = "F[".repeat(30) + "x" + &"]".repeat(30);
    let pair = parse(&input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_deeply_nested_with_implicit_times() {
    // Ensure implicit multiplication still works correctly with deep nesting
    let input = "F[".repeat(20) + "x" + &"]".repeat(20) + " y";
    let pair = parse(&input).unwrap().next().unwrap();
    assert_eq!(pair.as_rule(), Rule::Program);
  }

  #[test]
  fn test_parse_unary_plus() {
    let inputs = vec!["(+q)", "+x", "+5", "1 + +2", "+x^2"];
    for input in inputs {
      let pair = parse(input).unwrap().next().unwrap();
      assert_eq!(pair.as_rule(), Rule::Program, "Failed to parse: {}", input);
    }
  }
}
