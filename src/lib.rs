use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

impl WolframParser {
  pub fn parse_wolfram(
    input: &str,
  ) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
    Self::parse(Rule::Program, input)
  }
}

pub fn parse(
  input: &str,
) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
  WolframParser::parse_wolfram(input)
}

pub fn interpret(input: &str) -> Result<f64, String> {
  let pairs = parse(input).map_err(|e| e.to_string())?;
  let expr = pairs.into_iter().next().ok_or("Empty input")?;
  evaluate_expression(expr)
}

fn evaluate_expression(
  expr: pest::iterators::Pair<Rule>,
) -> Result<f64, String> {
  match expr.as_rule() {
    Rule::Expression => {
      let mut result = 0.0;
      let mut op = '+';
      for pair in expr.into_inner() {
        match pair.as_rule() {
          Rule::Term | Rule::NumericValue => {
            let value = evaluate_term(pair)?;
            match op {
              '+' => result += value,
              '-' => result -= value,
              _ => return Err(format!("Unexpected operator: {}", op)),
            }
          }
          Rule::Operator => op = pair.as_str().chars().next().unwrap(),
          _ => {
            return Err(format!(
              "Unexpected rule in Expression: {:?}",
              pair.as_rule()
            ))
          }
        }
      }
      Ok(result)
    }
    Rule::Program => evaluate_expression(expr.into_inner().next().unwrap()),
    _ => Err(format!("Unexpected rule: {:?}", expr.as_rule())),
  }
}

fn evaluate_term(term: pest::iterators::Pair<Rule>) -> Result<f64, String> {
  match term.as_rule() {
    Rule::Term => {
      let inner = term.into_inner().next().unwrap();
      evaluate_term(inner)
    }
    Rule::NumericValue => evaluate_term(term.into_inner().next().unwrap()),
    Rule::Integer | Rule::Real => {
      term.as_str().parse::<f64>().map_err(|e| e.to_string())
    }
    Rule::Expression => evaluate_expression(term),
    _ => Err(format!("Unexpected rule in Term: {:?}", term.as_rule())),
  }
}

#[cfg(test)]
mod tests {
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
}
