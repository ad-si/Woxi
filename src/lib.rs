use pest::Parser;
use pest_derive::Parser;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

#[derive(Error, Debug)]
pub enum InterpreterError {
    #[error("Parse error: {0}")]
    ParseError(#[from] pest::error::Error<Rule>),
    #[error("Empty input")]
    EmptyInput,
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

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

pub fn interpret(input: &str) -> Result<f64, InterpreterError> {
    let pairs = parse(input)?;
    let expr = pairs.into_iter().next().ok_or(InterpreterError::EmptyInput)?;
    evaluate_expression(expr).map_err(InterpreterError::EvaluationError)
}

fn evaluate_expression(expr: pest::iterators::Pair<Rule>) -> Result<f64, String> {
    match expr.as_rule() {
        Rule::Expression => {
            let mut terms = expr.into_inner().peekable();
            let mut result = 0.0;
            let mut current_term = evaluate_term(terms.next().unwrap())?;

            while let Some(op) = terms.next() {
                let next_term = terms.next().unwrap();
                match op.as_str() {
                    "+" => {
                        result += current_term;
                        current_term = evaluate_term(next_term)?;
                    }
                    "-" => {
                        result += current_term;
                        current_term = -evaluate_term(next_term)?;
                    }
                    "*" => current_term *= evaluate_term(next_term)?,
                    "/" => {
                        let divisor = evaluate_term(next_term)?;
                        if divisor == 0.0 {
                            return Err("Division by zero".to_string());
                        }
                        current_term /= divisor;
                    }
                    _ => return Err(format!("Unexpected operator: {}", op.as_str())),
                }
            }

            Ok(result + current_term)
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
