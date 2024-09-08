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

pub fn interpret(input: &str) -> Result<String, InterpreterError> {
  let pairs = parse(input)?;
  let expr = pairs
    .into_iter()
    .next()
    .ok_or(InterpreterError::EmptyInput)?;
  evaluate_expression(expr)
    .map(|result| {
      if result == 1.0 {
        "True".to_string()
      } else if result == 0.0 {
        "False".to_string()
      } else {
        format!("{:.10}", result)
          .trim_end_matches('0')
          .trim_end_matches('.')
          .to_string()
      }
    })
    .map_err(InterpreterError::EvaluationError)
}

fn evaluate_expression(
  expr: pest::iterators::Pair<Rule>,
) -> Result<f64, String> {
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
    Rule::NumericValue => {
      let inner = term.into_inner().next().unwrap();
      evaluate_term(inner)
    }
    Rule::Constant => match term.as_str() {
      "Pi" => Ok(std::f64::consts::PI),
      _ => Err(format!("Unknown constant: {}", term.as_str())),
    },
    Rule::Integer | Rule::Real => {
      term.as_str().parse::<f64>().map_err(|e| e.to_string())
    }
    Rule::Expression => evaluate_expression(term),
    Rule::FunctionCall => evaluate_function_call(term),
    _ => Err(format!("Unexpected rule in Term: {:?}", term.as_rule())),
  }
}

fn evaluate_function_call(
  func_call: pest::iterators::Pair<Rule>,
) -> Result<f64, String> {
  let mut inner = func_call.into_inner();
  let func_name = inner.next().unwrap().as_str();
  let mut args = inner.next().unwrap().into_inner();

  match func_name {
    "Prime" => {
      let n = evaluate_term(args.next().unwrap())?;
      if n.fract() != 0.0 || n < 1.0 {
        return Err(
          "Prime function argument must be a positive integer greater than 0"
            .to_string(),
        );
      }
      Ok(nth_prime(n as usize) as f64)
    }
    "EvenQ" => {
      let n = evaluate_term(args.next().unwrap())?;
      if n.fract() == 0.0 && (n as i64) % 2 == 0 {
        Ok(1.0) // Representing "True"
      } else {
        Ok(0.0) // Representing "False"
      }
    }
    "OddQ" => {
      let n = evaluate_term(args.next().unwrap())?;
      if n.fract() == 0.0 && (n as i64) % 2 != 0 {
        Ok(1.0) // Representing "True"
      } else {
        Ok(0.0) // Representing "False"
      }
    }
    "GroupBy" => {
      // Placeholder implementation
      Ok(0.0)
    }
    _ => Err(format!("Unknown function: {}", func_name)),
  }
}

fn nth_prime(n: usize) -> usize {
  if n == 0 {
    return 0; // Return 0 for invalid input
  }
  let mut count = 0;
  let mut num = 1;
  while count < n {
    num += 1;
    if is_prime(num) {
      count += 1;
    }
  }
  num
}

fn is_prime(n: usize) -> bool {
  if n <= 1 {
    return false;
  }
  if n == 2 {
    return true;
  }
  if n % 2 == 0 {
    return false;
  }
  let sqrt_n = (n as f64).sqrt() as usize;
  for i in (3..=sqrt_n).step_by(2) {
    if n % i == 0 {
      return false;
    }
  }
  true
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
