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

  evaluate_expression(expr)
}

fn format_result(result: f64) -> String {
  if result.fract() == 0.0 {
    let int_result = result as i64;
    int_result.to_string()
  }
  else {
    format!("{:.10}", result)
      .trim_end_matches('0')
      .trim_end_matches('.')
      .to_string()
  }
}

fn evaluate_expression(
  expr: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  match expr.as_rule() {
    Rule::String => Ok(expr.as_str().trim_matches('"').to_string()),
    Rule::Expression => {
      let mut inner = expr.into_inner();
      let first = inner.next().unwrap();
      if inner.clone().next().is_none() {
        if first.as_rule() == Rule::List {
          return evaluate_expression(first);
        }
        else if first.as_rule() == Rule::Identifier {
          return Ok(first.as_str().to_string());
        }
        else if first.as_rule() == Rule::FunctionCall {
          return evaluate_function_call(first);
        }
        else if first.as_rule() == Rule::Term {
          return evaluate_expression(first.into_inner().next().unwrap());
        }
      }
      let mut values: Vec<f64> = vec![evaluate_term(first)?];
      let mut ops: Vec<&str> = vec![];
      while let Some(op_pair) = inner.next() {
        let op = op_pair.as_str();
        let term = inner.next().unwrap();
        ops.push(op);
        values.push(evaluate_term(term)?);
      }
      // First pass: handle multiplication and division
      let mut i = 0;
      while i < ops.len() {
        if ops[i] == "*" {
          values[i] = values[i] * values[i + 1];
          values.remove(i + 1);
          ops.remove(i);
        }
        else if ops[i] == "/" {
          if values[i + 1] == 0.0 {
            return Err(InterpreterError::EvaluationError(
              "Division by zero".to_string(),
            ));
          }
          values[i] = values[i] / values[i + 1];
          values.remove(i + 1);
          ops.remove(i);
        }
        else {
          i += 1;
        }
      }
      // Second pass: handle addition and subtraction
      let mut result = values[0];
      for (op, &val) in ops.iter().zip(values.iter().skip(1)) {
        if *op == "+" {
          result += val;
        }
        else if *op == "-" {
          result -= val;
        }
        else {
          return Err(InterpreterError::EvaluationError(format!(
            "Unexpected operator: {}",
            op
          )));
        }
      }
      Ok(format_result(result))
    }
    Rule::Program => evaluate_expression(expr.into_inner().next().unwrap()),
    Rule::List => {
      let items: Vec<String> = expr
        .into_inner()
        .filter(|item| item.as_str() != ",")
        .map(|item| evaluate_expression(item))
        .collect::<Result<_, _>>()?;
      Ok(format!("{{{}}}", items.join(", ")))
    }
    Rule::Term => {
      let mut inner = expr.clone().into_inner();
      if let Some(first) = inner.next() {
        if first.as_rule() == Rule::FunctionCall {
          return evaluate_function_call(first);
        }
        else if first.as_rule() == Rule::List {
          return evaluate_expression(first);
        }
      }
      evaluate_term(expr).map(format_result)
    }
    Rule::FunctionCall => evaluate_function_call(expr),
    Rule::Identifier => Ok(expr.as_str().to_string()),
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
    Rule::Integer => {
      term.as_str().parse::<i64>().map(|n| n as f64).map_err(|_| {
        InterpreterError::EvaluationError("invalid integer literal".to_string())
      })
    }
    Rule::Real => term.as_str().parse::<f64>().map_err(|_| {
      InterpreterError::EvaluationError("invalid float literal".to_string())
    }),
    Rule::Expression => evaluate_expression(term).and_then(|s| {
      s.parse::<f64>()
        .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
    }),
    Rule::FunctionCall => evaluate_function_call(term).and_then(|s| {
      if s == "True" {
        Ok(1.0)
      }
      else if s == "False" {
        Ok(0.0)
      }
      else {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      }
    }),
    Rule::Identifier => match term.as_str() {
      "True" => Ok(1.0),
      "False" => Ok(0.0),
      _ => Ok(0.0), // Return 0.0 for unknown identifiers
    },
    Rule::Slot => {
      // For slot (#), we'll return 1.0 as a default value when evaluated as a term
      // It will be replaced with the actual value in the anonymous function evaluation
      Ok(1.0)
    }
    Rule::List => Err(InterpreterError::EvaluationError(
      "Cannot evaluate a list as a numeric value".to_string(),
    )),
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
  let func_name_pair = inner.next().unwrap();

  // Handle anonymous function case
  if func_name_pair.as_rule() == Rule::AnonymousFunction {
    let mut func_parts = func_name_pair.into_inner();
    let _slot = func_parts.next().unwrap();
    let operator = func_parts.next().unwrap().as_str();
    let operand = func_parts.next().unwrap();

    // Get argument - for Wolfram syntax '#^2 &[{1, 2, 3}]'
    let args = inner.next();
    if args.is_none() {
      return Err(InterpreterError::EvaluationError(
        "Expected arguments for anonymous function".to_string(),
      ));
    }

    let arg = args.unwrap();
    // Extract list from the argument
    let list = match arg.as_rule() {
      Rule::List => arg,
      Rule::Expression => {
        let mut inner_expr = arg.into_inner();
        if let Some(first) = inner_expr.next() {
          if first.as_rule() == Rule::List {
            first
          }
          else {
            return Err(InterpreterError::EvaluationError(
              "Anonymous function must be applied to a list".to_string(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "Empty expression in anonymous function arguments".to_string(),
          ));
        }
      }
      _ => {
        return Err(InterpreterError::EvaluationError(format!(
          "Anonymous function must be applied to a list, got {:?}",
          arg.as_rule()
        )))
      }
    };

    let items: Vec<_> = list
      .into_inner()
      .filter(|item| item.as_str() != ",")
      .collect();
    let mut results = Vec::new();

    for item in items {
      let item_value = evaluate_term(item.clone())?;

      let result = match operator {
        "+" => item_value + evaluate_term(operand.clone())?,
        "-" => item_value - evaluate_term(operand.clone())?,
        "*" => item_value * evaluate_term(operand.clone())?,
        "/" => {
          let denominator = evaluate_term(operand.clone())?;
          if denominator == 0.0 {
            return Err(InterpreterError::EvaluationError(
              "Division by zero".to_string(),
            ));
          }
          item_value / denominator
        }
        "^" => item_value.powf(evaluate_term(operand.clone())?),
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "Unsupported operator in anonymous function: {}",
            operator
          )))
        }
      };

      results.push(format_result(result));
    }

    return Ok(format!("{{{}}}", results.join(", ")));
  }

  // Handle regular function case
  let func_name = func_name_pair.as_str();
  // collect all arguments (ignore literal commas generated by the grammar)
  let args_pairs: Vec<pest::iterators::Pair<Rule>> =
    inner.filter(|p| p.as_str() != ",").collect();

  match func_name {
    // ----- numeric helpers --------------------------------------------------
    "Prime" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Prime expects exactly 1 argument".into(),
        ));
      }
      let n = evaluate_term(args_pairs[0].clone())?;
      if n.fract() != 0.0 || n < 1.0 {
        return Err(InterpreterError::EvaluationError(
          "Prime function argument must be a positive integer greater than 0"
            .into(),
        ));
      }
      Ok(nth_prime(n as usize).to_string())
    }
    "Sign" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Sign expects exactly 1 argument".into(),
        ));
      }
      let n = evaluate_term(args_pairs[0].clone())?;
      Ok(
        if n > 0.0 {
          "1"
        }
        else if n < 0.0 {
          "-1"
        }
        else {
          "0"
        }
        .to_string(),
      )
    }

    // ----- list helpers ------------------------------------------------------
    "Map" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Map expects exactly 2 arguments".into(),
        ));
      }
      let func_pair = &args_pairs[0];
      let list_pair = &args_pairs[1];

      // Accept both List and Expression wrapping a List for the second argument
      let list_rule = list_pair.as_rule();
      let elements: Vec<_> = if list_rule == Rule::List {
        list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect()
      }
      else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          }
          else {
            return Err(InterpreterError::EvaluationError(
              "Second argument of Map must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "Second argument of Map must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "Second argument of Map must be a list".into(),
        ));
      };

      let func_name_inner = func_pair.as_str();
      let mut mapped = Vec::new();

      for elem in elements {
        let elem_val_str = evaluate_expression(elem.clone())?;
        let num = elem_val_str.parse::<f64>().map_err(|_| {
          InterpreterError::EvaluationError(
            "Map currently supports only numeric list elements".into(),
          )
        })?;

        let mapped_val = match func_name_inner {
          "Sign" => {
            let sign = if num > 0.0 {
              1.0
            }
            else if num < 0.0 {
              -1.0
            }
            else {
              0.0
            };
            format_result(sign)
          }
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Unknown mapping function: {}",
              func_name_inner
            )))
          }
        };
        mapped.push(mapped_val);
      }
      Ok(format!("{{{}}}", mapped.join(", ")))
    }
    "EvenQ" | "OddQ" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(format!(
          "{} expects exactly 1 argument",
          func_name
        )));
      }
      let n = evaluate_term(args_pairs[0].clone())?;
      if n.fract() != 0.0 {
        return Ok("False".to_string());
      }
      let is_even = n >= 0.0 && (n as i64) % 2 == 0;
      Ok(
        if (func_name == "EvenQ" && is_even)
          || (func_name == "OddQ" && !is_even)
        {
          "True"
        }
        else {
          "False"
        }
        .to_string(),
      )
    }
    "First" | "Last" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(format!(
          "{} expects exactly 1 argument",
          func_name
        )));
      }
      let list_pair = &args_pairs[0];
      // Accept both List and Expression wrapping a List
      let list_rule = list_pair.as_rule();
      let items: Vec<_> = if list_rule == Rule::List {
        list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect()
      }
      else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          }
          else {
            return Err(InterpreterError::EvaluationError(format!(
              "{} function argument must be a list",
              func_name
            )));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(format!(
            "{} function argument must be a list",
            func_name
          )));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} function argument must be a list",
          func_name
        )));
      };
      let target = if func_name == "First" {
        items.first()
      }
      else {
        items.last()
      };
      match target {
        Some(item) => evaluate_expression(item.clone()),
        None => Err(InterpreterError::EvaluationError("Empty list".into())),
      }
    }

    // ----- list element / slice helpers ------------------------------------
    "Rest" | "Most" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(format!(
          "{} expects exactly 1 argument", func_name
        )));
      }
      // extract list items exactly like the First/Last implementation
      let list_pair = &args_pairs[0];
      let list_rule = list_pair.as_rule();
      let items: Vec<_> = if list_rule == Rule::List {
        list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect()
      }
      else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          }
          else {
            return Err(InterpreterError::EvaluationError(format!(
              "{} function argument must be a list",
              func_name
            )));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(format!(
            "{} function argument must be a list",
            func_name
          )));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} function argument must be a list",
          func_name
        )));
      };
      let slice: Vec<_> = if func_name == "Rest" {
        if items.len() <= 1 { vec![] } else { items[1..].to_vec() }
      } else {                       // Most
        if items.len() <= 1 { vec![] } else { items[..items.len() - 1].to_vec() }
      };
      let evaluated: Result<Vec<_>, _> =
        slice.into_iter().map(|p| evaluate_expression(p)).collect();
      return Ok(format!("{{{}}}", evaluated?.join(", ")));
    }

    "Take" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Take expects exactly 2 arguments".into(),
        ));
      }
      // first argument must be a list – reuse the extraction helper
      let list_pair = &args_pairs[0];
      let list_rule = list_pair.as_rule();
      let items: Vec<_> = if list_rule == Rule::List {
        list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect()
      }
      else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          }
          else {
            return Err(InterpreterError::EvaluationError(
              "Take function argument must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "Take function argument must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "Take function argument must be a list".into(),
        ));
      };
      // second argument must be a positive integer
      let n = evaluate_term(args_pairs[1].clone())?;
      if n.fract() != 0.0 || n <= 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Second argument of Take must be a positive integer".into(),
        ));
      }
      let k = std::cmp::min(n as usize, items.len());
      let evaluated: Result<Vec<_>, _> = items[..k]
        .iter()
        .cloned()
        .map(|p| evaluate_expression(p))
        .collect();
      return Ok(format!("{{{}}}", evaluated?.join(", ")));
    }

    "Part" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Part expects exactly 2 arguments".into(),
        ));
      }
      let list_pair = &args_pairs[0];
      let list_rule = list_pair.as_rule();
      let items: Vec<_> = if list_rule == Rule::List {
        list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect()
      }
      else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          }
          else {
            return Err(InterpreterError::EvaluationError(
              "Part function argument must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "Part function argument must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "Part function argument must be a list".into(),
        ));
      };
      let n = evaluate_term(args_pairs[1].clone())?;
      if n.fract() != 0.0 || n <= 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Second argument of Part must be a positive integer".into(),
        ));
      }
      let idx = (n as usize).checked_sub(1).ok_or_else(|| {
        InterpreterError::EvaluationError("Invalid index in Part".into())
      })?;
      if idx >= items.len() {
        return Err(InterpreterError::EvaluationError(
          "Index out of bounds in Part".into(),
        ));
      }
      return evaluate_expression(items[idx].clone());
    }

    "Length" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Length expects exactly 1 argument".into(),
        ));
      }
      let list_pair = &args_pairs[0];
      // Accept both List and Expression wrapping a List
      let list_rule = list_pair.as_rule();
      let list_items = if list_rule == Rule::List {
        list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect::<Vec<_>>()
      }
      else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first
              .into_inner()
              .filter(|p| p.as_str() != ",")
              .collect::<Vec<_>>()
          }
          else {
            return Err(InterpreterError::EvaluationError(
              "Length function argument must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "Length function argument must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "Length function argument must be a list".into(),
        ));
      };
      Ok(list_items.len().to_string())
    }
    "GroupBy" => Err(InterpreterError::EvaluationError(
      "GroupBy function not yet implemented".into(),
    )),
    "Print" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Print expects exactly 1 argument".into(),
        ));
      }
      // Accept string, or expression wrapping a string, or any printable value
      let arg_pair = &args_pairs[0];
      let arg_str = match arg_pair.as_rule() {
        Rule::String => arg_pair.as_str().trim_matches('"').to_string(),
        Rule::Expression => {
          let mut expr_inner = arg_pair.clone().into_inner();
          if let Some(first) = expr_inner.next() {
            if first.as_rule() == Rule::String {
              first.as_str().trim_matches('"').to_string()
            }
            else {
              evaluate_expression(arg_pair.clone())?
            }
          }
          else {
            evaluate_expression(arg_pair.clone())?
          }
        }
        _ => evaluate_expression(arg_pair.clone())?,
      };
      println!("{}", arg_str);
      Ok("Null".to_string())
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
