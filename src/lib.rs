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
  let program = pairs
    .into_iter()
    .next()
    .ok_or(InterpreterError::EmptyInput)?;

  if program.as_rule() != Rule::Program {
    return Err(InterpreterError::EvaluationError(format!(
      "Expected Program, got {:?}",
      program.as_rule()
    )));
  }

  let expr = program
    .into_inner()
    .next()
    .ok_or(InterpreterError::EmptyInput)?;

  match expr.as_rule() {
    Rule::List => {
      let items: Vec<String> = expr
        .into_inner()
        .map(|item| interpret(item.as_str()))
        .collect::<Result<_, _>>()?;
      Ok(format!("{{{}}}", items.join(", ")))
    }
    Rule::Expression | Rule::Term => {
      evaluate_expression(expr).map(format_result)
    }
    Rule::FunctionCall => evaluate_function_call(expr),
    Rule::Identifier => Ok(expr.as_str().to_string()),
    _ => Err(InterpreterError::EvaluationError(format!(
      "Unexpected rule: {:?}",
      expr.as_rule()
    ))),
  }
}

fn format_result(result: f64) -> String {
  if result.fract() == 0.0 {
    let int_result = result as i64;
    match int_result {
      1 => "True".to_string(),
      0 => "False".to_string(),
      _ => int_result.to_string(),
    }
  } else {
    format!("{:.10}", result)
      .trim_end_matches('0')
      .trim_end_matches('.')
      .to_string()
  }
}

fn evaluate_expression(
  expr: pest::iterators::Pair<Rule>,
) -> Result<f64, InterpreterError> {
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
              return Err(InterpreterError::EvaluationError(
                "Division by zero".to_string(),
              ));
            }
            current_term /= divisor;
          }
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Unexpected operator: {}",
              op.as_str()
            )))
          }
        }
      }

      Ok(result + current_term)
    }
    Rule::Program => evaluate_expression(expr.into_inner().next().unwrap()),
    _ => Err(InterpreterError::EvaluationError(format!(
      "Unexpected rule: {:?}",
      expr.as_rule()
    ))),
  }
}

fn evaluate_term(
  term: pest::iterators::Pair<Rule>,
) -> Result<f64, InterpreterError> {
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
      _ => Err(InterpreterError::EvaluationError(format!(
        "Unknown constant: {}",
        term.as_str()
      ))),
    },
    Rule::Integer => term
      .as_str()
      .parse::<f64>()
      .map_err(|e| InterpreterError::EvaluationError(e.to_string())),
    Rule::Real => {
      if let Ok(value) = term.as_str().parse::<f64>() {
        Ok(value)
      } else {
        Err(InterpreterError::EvaluationError(
          "invalid float literal".to_string(),
        ))
      }
    }
    Rule::Expression => evaluate_expression(term),
    Rule::FunctionCall => evaluate_function_call(term).and_then(|s| {
      s.parse::<f64>()
        .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
    }),
    Rule::List => Ok(0.0), // Placeholder for list evaluation
    Rule::Identifier => match term.as_str() {
      "True" => Ok(1.0),
      "False" => Ok(0.0),
      _ => Ok(0.0), // Return 0.0 for other identifiers
    },
    _ => Err(InterpreterError::EvaluationError(format!(
      "Unexpected rule in Term: {:?}",
      term.as_rule()
    ))),
  }
}

fn evaluate_function_call(
  func_call: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  let mut inner = func_call.into_inner();
  let func_name = inner.next().unwrap().as_str();
  let mut args = inner.next().unwrap().into_inner();

  match func_name {
    "Prime" => {
      let n = evaluate_term(args.next().unwrap())?;
      if n.fract() != 0.0 || n < 1.0 {
        return Err(InterpreterError::EvaluationError(
          "Prime function argument must be a positive integer greater than 0"
            .to_string(),
        ));
      }
      Ok(nth_prime(n as usize).to_string())
    }
    "EvenQ" | "OddQ" => {
      let arg = args.next().unwrap();
      let n = evaluate_term(arg)?;
      if n.fract() != 0.0 {
        return Ok("False".to_string());
      }
      let is_even = (n as i64) % 2 == 0;
      Ok(
        if (func_name == "EvenQ" && is_even)
          || (func_name == "OddQ" && !is_even)
        {
          "True".to_string()
        } else {
          "False".to_string()
        },
      )
    }
    "First" | "Last" => {
      let list = args.next().unwrap();
      if list.as_rule() != Rule::List {
        return Err(InterpreterError::EvaluationError(format!(
          "{} function argument must be a list",
          func_name
        )));
      }
      let mut items = list.into_inner();
      let target_item = if func_name == "First" {
        items.next()
      } else {
        items.last()
      };

      match target_item {
        Some(item) => interpret(item.as_str()),
        None => {
          Err(InterpreterError::EvaluationError("Empty list".to_string()))
        }
      }
    }
    "GroupBy" => {
      // Placeholder implementation
      Err(InterpreterError::EvaluationError(
        "GroupBy function not yet implemented".to_string(),
      ))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Unknown function: {}",
      func_name
    ))),
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
