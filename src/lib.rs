use pest::iterators::Pair;
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
    Rule::Term => {
      let mut inner = expr.clone().into_inner();
      if let Some(first) = inner.next() {
        if first.as_rule() == Rule::FunctionCall {
          return evaluate_function_call(first);
        }
        else if first.as_rule() == Rule::List {
          return evaluate_expression(first);
        }
        else if first.as_rule() == Rule::String {
          return Ok(first.as_str().trim_matches('"').to_string());
        }
        else if first.as_rule() == Rule::Integer
          || first.as_rule() == Rule::Real
          || first.as_rule() == Rule::Constant
          || first.as_rule() == Rule::NumericValue
          || first.as_rule() == Rule::Identifier
          || first.as_rule() == Rule::Slot
        {
          return evaluate_term(first).map(format_result);
        }
      }
      evaluate_term(expr).map(format_result)
    }
    Rule::Expression => {
      // --- special case: Map operator ----------------------------------
      {
        let items: Vec<_> = expr.clone().into_inner().collect();
        if items.len() == 3
          && items[1].as_rule() == Rule::Operator
          && items[1].as_str() == "/@"
        {
          return apply_map_operator(items[0].clone(), items[2].clone());
        }
      }
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
    Rule::String => {
      // For string arguments in string functions, just return 0.0 (never used as a number)
      Ok(0.0)
    }
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

fn extract_string(pair: Pair<Rule>) -> Result<String, InterpreterError> {
  match pair.as_rule() {
    Rule::String => Ok(pair.as_str().trim_matches('"').to_string()),
    Rule::Expression | Rule::Term => {
      let mut inner = pair.clone().into_inner();
      if let Some(first) = inner.next() {
        return extract_string(first);
      }
      Err(InterpreterError::EvaluationError(
        "Expected string argument".into(),
      ))
    }
    _ => evaluate_expression(pair), // fall-back, keeps behaviour for exotic cases
  }
}

fn evaluate_function_call(
  func_call: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  // 2. Store the full textual form of the call for later error messages
  let call_text = func_call.as_str().to_string(); // keep original text
  let mut inner = func_call.into_inner();
  let func_name_pair = inner.next().unwrap();

  // Helper for boolean conversion
  fn as_bool(s: &str) -> Option<bool> {
    match s {
      "True" => Some(true),
      "False" => Some(false),
      _ => None,
    }
  }

  // ----- anonymous function -------------------------------------------------
  if func_name_pair.as_rule() == Rule::AnonymousFunction {
    // inspect the parts that form the anonymous function
    let parts: Vec<_> = func_name_pair.clone().into_inner().collect();

    // fetch the argument list / single argument of the call  →  #&[arg]
    let arg = inner.next().ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Expected arguments for anonymous function".to_string(),
      )
    })?;

    // ----------------------------------------------------------------------
    // simple identity function  (#&)
    // ----------------------------------------------------------------------
    if parts.len() == 1 {
      // just return the evaluated argument unchanged
      return evaluate_expression(arg);
    }

    // ----------------------------------------------------------------------
    // operator form  (# op term &)
    // (old behaviour, preserved)
    // ----------------------------------------------------------------------
    let operator = parts[1].as_str();
    let operand = parts[2].clone();

    // the existing code that:
    //  • expects `arg` to be a list,
    //  • iterates over its elements,
    //  • applies the operator to each element,
    //  • collects the results and returns them,
    // stays exactly as it was – just use the
    // `operator`, `operand`, and `arg` variables defined above.

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
    // ─────────────────── relational (inclusive) comparisons ──────────────────
    "GreaterEqual" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "GreaterEqual expects at least 2 arguments".into(),
        ));
      }
      let mut prev = evaluate_term(args_pairs[0].clone())?;
      for ap in args_pairs.iter().skip(1) {
        let cur = evaluate_term(ap.clone())?;
        if prev < cur {
          return Ok("False".to_string());
        }
        prev = cur;
      }
      return Ok("True".to_string());
    }

    "LessEqual" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "LessEqual expects at least 2 arguments".into(),
        ));
      }
      let mut prev = evaluate_term(args_pairs[0].clone())?;
      for ap in args_pairs.iter().skip(1) {
        let cur = evaluate_term(ap.clone())?;
        if prev > cur {
          return Ok("False".to_string());
        }
        prev = cur;
      }
      return Ok("True".to_string());
    }

    // ───────────────────────────── boolean logic ─────────────────────────────
    "And" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "And expects at least 2 arguments".into(),
        ));
      }
      for ap in &args_pairs {
        if as_bool(&evaluate_expression(ap.clone())?).unwrap_or(false) == false
        {
          return Ok("False".to_string());
        }
      }
      return Ok("True".to_string());
    }

    "Or" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "Or expects at least 2 arguments".into(),
        ));
      }
      for ap in &args_pairs {
        if as_bool(&evaluate_expression(ap.clone())?).unwrap_or(false) {
          return Ok("True".to_string());
        }
      }
      return Ok("False".to_string());
    }

    "Xor" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "Xor expects at least 2 arguments".into(),
        ));
      }
      let mut true_cnt = 0;
      for ap in &args_pairs {
        if as_bool(&evaluate_expression(ap.clone())?).unwrap_or(false) {
          true_cnt += 1;
        }
      }
      return Ok(if true_cnt % 2 == 1 { "True" } else { "False" }.to_string());
    }

    "Not" => {
      if args_pairs.len() != 1 {
        use std::io::{self, Write};
        println!(
          "\nNot::argx: Not called with {} arguments; 1 argument is expected.",
          args_pairs.len()
        );
        io::stdout().flush().ok();

        // rebuild unevaluated expression
        let mut parts = Vec::new();
        for ap in &args_pairs {
          parts.push(evaluate_expression(ap.clone())?);
        }
        return Ok(format!("Not[{}]", parts.join(", ")));
      }
      let v =
        as_bool(&evaluate_expression(args_pairs[0].clone())?).unwrap_or(false);
      return Ok(if v { "False" } else { "True" }.to_string());
    }

    // ──────────────────────────────── If ──────────────────────────────────────
    "If" => {
      // arity 2‥4
      if !(2..=4).contains(&args_pairs.len()) {
        use std::io::{self, Write};
        println!(
          "\nIf::argb: If called with {} arguments; between 2 and 4 arguments are expected.",
          args_pairs.len()
        );
        io::stdout().flush().ok();
        return Ok(call_text); // return unevaluated expression
      }

      // evaluate test
      let test_str = evaluate_expression(args_pairs[0].clone())?;
      let test_val = as_bool(&test_str);

      match (test_val, args_pairs.len()) {
        (Some(true), _) => return evaluate_expression(args_pairs[1].clone()),
        (Some(false), 2) => return Ok("Null".to_string()),
        (Some(false), 3) => return evaluate_expression(args_pairs[2].clone()),
        (Some(false), 4) => return evaluate_expression(args_pairs[2].clone()),
        (None, 2) => return Ok("Null".to_string()),
        (None, 3) => return Ok("Null".to_string()),
        (None, 4) => return evaluate_expression(args_pairs[3].clone()),
        _ => unreachable!(),
      }
    }

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
    "Plus" => {
      // ── arity check ──────────────────────────────────────────────────────
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Plus expects at least 1 argument".into(),
        ));
      }

      // ── sum all numeric arguments ────────────────────────────────────────
      let mut sum = 0.0;
      for ap in &args_pairs {
        sum += evaluate_term(ap.clone())?;
      }

      // ── return formatted result ──────────────────────────────────────────
      return Ok(format_result(sum));
    }
    "Times" => {
      // ----- arity check ---------------------------------------------------
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Times expects at least 1 argument".into(),
        ));
      }

      // ----- multiply all numeric arguments --------------------------------
      let mut product = 1.0;
      for ap in &args_pairs {
        product *= evaluate_term(ap.clone())?;
      }

      // ----- return formatted result ---------------------------------------
      return Ok(format_result(product));
    }
    "Minus" => {
      // ---- arity check ----------------------------------------------------
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Minus expects at least 1 argument".into(),
        ));
      }

      // ---- unary / argument-count handling ----------------------------------
      if args_pairs.len() == 1 {
        // unary Minus
        let v = evaluate_term(args_pairs[0].clone())?;
        return Ok(format_result(-v));
      }

      // ---- wrong number of arguments  ---------------------------------------
      // Print *with* a trailing newline to match shelltest's expected output,
      // and flush stdout to ensure the order is correct for shelltest.
      use std::io::{self, Write};
      println!(
          "\nMinus::argx: Minus called with {} arguments; 1 argument is expected.",
          args_pairs.len()
      );
      io::stdout().flush().ok();

      // build the pretty printing of the unevaluated expression:  "5 − 2"
      let mut pieces = Vec::new();
      for ap in &args_pairs {
        pieces.push(evaluate_expression(ap.clone())?); // keeps formatting (e.g. 5, 2, 3.1…)
      }
      let expr = pieces.join(" − "); // note U+2212 (minus sign) surrounded by spaces
      return Ok(expr);
    }
    "Abs" => {
      // ── arity check ────────────────────────────────────────────────────────
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Abs expects exactly 1 argument".into(),
        ));
      }
      // ── evaluate argument ──────────────────────────────────────────────────
      let n = evaluate_term(args_pairs[0].clone())?;
      // ── return absolute value, formatted like other numeric outputs ───────
      return Ok(format_result(n.abs()));
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
    "Sqrt" => {
      // ---- arity check ----------------------------------------------------
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Sqrt expects exactly 1 argument".into(),
        ));
      }
      // ---- evaluate & validate argument -----------------------------------
      let n = evaluate_term(args_pairs[0].clone())?;
      if n < 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Sqrt function argument must be non-negative".into(),
        ));
      }
      // ---- return √n, formatted like all other numeric outputs ------------
      return Ok(format_result(n.sqrt()));
    }

    // ── string functions ────────────────────────────────────────────────
    "Floor" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Floor expects exactly 1 argument".into(),
        ));
      }
      let n = evaluate_term(args_pairs[0].clone())?;
      let mut r = n.floor();
      if r == -0.0 { r = 0.0; }
      return Ok(format_result(r));
    }

    "Ceiling" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Ceiling expects exactly 1 argument".into(),
        ));
      }
      let n = evaluate_term(args_pairs[0].clone())?;
      let mut r = n.ceil();
      if r == -0.0 { r = 0.0; }
      return Ok(format_result(r));
    }

    "Round" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Round expects exactly 1 argument".into(),
        ));
      }
      let n = evaluate_term(args_pairs[0].clone())?;

      // banker’s rounding (half-to-even)
      let base  = n.trunc();
      let frac  = n - base;
      let mut r = if frac.abs() == 0.5 {
        if (base as i64) % 2 == 0 { base }               // already even
        else if n.is_sign_positive() { base + 1.0 }      // away from zero
        else { base - 1.0 }
      } else {
        n.round()
      };
      if r == -0.0 { r = 0.0; }
      return Ok(format_result(r));
    }

    "StringLength" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "StringLength expects exactly 1 argument".into(),
        ));
      }
      let s = extract_string(args_pairs[0].clone())?;
      return Ok(s.chars().count().to_string());
    }

    "StringTake" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "StringTake expects exactly 2 arguments".into(),
        ));
      }
      let s = extract_string(args_pairs[0].clone())?;
      let n = evaluate_term(args_pairs[1].clone())?;
      if n.fract() != 0.0 || n < 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Second argument of StringTake must be a non-negative integer".into(),
        ));
      }
      let k = n as usize;
      let taken: String = s.chars().take(k).collect();
      return Ok(taken);
    }

    "StringDrop" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "StringDrop expects exactly 2 arguments".into(),
        ));
      }
      let s = extract_string(args_pairs[0].clone())?;
      let n = evaluate_term(args_pairs[1].clone())?;
      if n.fract() != 0.0 || n < 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Second argument of StringDrop must be a non-negative integer".into(),
        ));
      }
      let k = n as usize;
      let dropped: String = s.chars().skip(k).collect();
      return Ok(dropped);
    }

    "StringJoin" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "StringJoin expects at least 1 argument".into(),
        ));
      }
      let mut joined = String::new();
      for ap in &args_pairs {
        joined.push_str(&extract_string(ap.clone())?);
      }
      return Ok(joined);
    }

    "StringSplit" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "StringSplit expects exactly 2 arguments".into(),
        ));
      }
      let s = extract_string(args_pairs[0].clone())?;
      let delim = extract_string(args_pairs[1].clone())?;
      let parts: Vec<String> = if delim.is_empty() {
        s.chars().map(|c| c.to_string()).collect()
      }
      else {
        s.split(&delim).map(|p| p.to_string()).collect()
      };
      return Ok(format!("{{{}}}", parts.join(", ")));
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
          "{} expects exactly 1 argument",
          func_name
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
        if items.len() <= 1 {
          vec![]
        }
        else {
          items[1..].to_vec()
        }
      }
      else {
        // Most
        if items.len() <= 1 {
          vec![]
        }
        else {
          items[..items.len() - 1].to_vec()
        }
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

    "Drop" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Drop expects exactly 2 arguments".into(),
        ));
      }
      // ----- get list items (same extraction code used in Take) -----
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
              "First argument of Drop must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "First argument of Drop must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "First argument of Drop must be a list".into(),
        ));
      };

      // ----- get n ----------------------------------------------------
      let n = evaluate_term(args_pairs[1].clone())?;
      if n.fract() != 0.0 || n < 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Second argument of Drop must be a non-negative integer".into(),
        ));
      }
      let start = std::cmp::min(n as usize, items.len());
      let slice = items[start..].to_vec();
      let evaluated: Result<Vec<_>, _> =
        slice.into_iter().map(|p| evaluate_expression(p)).collect();
      return Ok(format!("{{{}}}", evaluated?.join(", ")));
    }

    "Append" | "Prepend" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(format!(
          "{} expects exactly 2 arguments",
          func_name
        )));
      }
      // extract list items (same helper as above)
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
              "First argument of {} must be a list",
              func_name
            )));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(format!(
            "First argument of {} must be a list",
            func_name
          )));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(format!(
          "First argument of {} must be a list",
          func_name
        )));
      };

      // evaluate existing list
      let mut evaluated: Vec<String> = items
        .into_iter()
        .map(|p| evaluate_expression(p))
        .collect::<Result<_, _>>()?;

      // evaluate new element
      let new_elem = evaluate_expression(args_pairs[1].clone())?;

      if func_name == "Append" {
        evaluated.push(new_elem);
      }
      else {
        // Prepend
        evaluated.insert(0, new_elem);
      }
      return Ok(format!("{{{}}}", evaluated.join(", ")));
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
    // ----- aggregation -----------------------------------------------------
    "Total" => {
      // arity
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Total expects exactly 1 argument".into(),
        ));
      }

      // extract list items (identical logic used in Length)
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
              "Total function argument must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "Total function argument must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "Total function argument must be a list".into(),
        ));
      };

      // sum numeric values
      let mut sum = 0.0;
      for item in items {
        sum += evaluate_term(item.clone())?;
      }
      return Ok(format_result(sum));
    }
    "Divide" => {
      // ── arity check ──────────────────────────────────────────────────────
      if args_pairs.len() != 2 {
        use std::io::{self, Write};
        println!(
          "\nDivide::argrx: Divide called with {} arguments; 2 arguments are expected.",
          args_pairs.len()
        );
        io::stdout().flush().ok();

        // unevaluated expression
        let mut parts = Vec::new();
        for ap in &args_pairs {
          parts.push(evaluate_expression(ap.clone())?);
        }
        return Ok(format!("Divide[{}]", parts.join(", ")));
      }

      // ── evaluate arguments ───────────────────────────────────────────────
      let numerator   = evaluate_term(args_pairs[0].clone())?;
      let denominator = evaluate_term(args_pairs[1].clone())?;

      if denominator == 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Division by zero".into(),
        ));
      }

      // ── return formatted result ──────────────────────────────────────────
      return Ok(format_result(numerator / denominator));
    }
    // ----- boolean equality helpers ----------------------------------------
    "Equal" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "Equal expects at least 2 arguments".into(),
        ));
      }
      let first_val = evaluate_term(args_pairs[0].clone())?;
      for ap in args_pairs.iter().skip(1) {
        if evaluate_term(ap.clone())? != first_val {
          return Ok("False".to_string());
        }
      }
      return Ok("True".to_string());
    }
    "Unequal" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "Unequal expects at least 2 arguments".into(),
        ));
      }
      let mut seen = Vec::<f64>::new();
      for ap in &args_pairs {
        let v = evaluate_term(ap.clone())?;
        if seen.contains(&v) {
          return Ok("False".to_string()); // duplicate found
        }
        seen.push(v);
      }
      return Ok("True".to_string()); // all pair-wise different
    }

    // ----- relational comparison helpers -----------------------------------
    "Greater" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "Greater expects at least 2 arguments".into(),
        ));
      }
      let mut prev = evaluate_term(args_pairs[0].clone())?;
      for ap in args_pairs.iter().skip(1) {
        let current = evaluate_term(ap.clone())?;
        if !(prev > current) {
          return Ok("False".to_string());
        }
        prev = current;
      }
      return Ok("True".to_string());
    }

    "Less" => {
      if args_pairs.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "Less expects at least 2 arguments".into(),
        ));
      }
      let mut prev = evaluate_term(args_pairs[0].clone())?;
      for ap in args_pairs.iter().skip(1) {
        let current = evaluate_term(ap.clone())?;
        if !(prev < current) {
          return Ok("False".to_string());
        }
        prev = current;
      }
      return Ok("True".to_string());
    }
    "Select" => {
      // ----- arity ---------------------------------------------------------
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Select expects exactly 2 arguments".into(),
        ));
      }

      // ----- extract list --------------------------------------------------
      let list_pair = &args_pairs[0];
      let list_rule = list_pair.as_rule();
      let elems: Vec<_> = if list_rule == Rule::List {
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
              "First argument of Select must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "First argument of Select must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "First argument of Select must be a list".into(),
        ));
      };

      // ----- identify predicate -------------------------------------------
      let pred_pair = &args_pairs[1];
      let pred_name = pred_pair.as_str();

      // ----- filter --------------------------------------------------------
      let mut kept = Vec::new();
      for elem in elems {
        let passes = match pred_name {
          "EvenQ" | "OddQ" => {
            let n = evaluate_term(elem.clone())?;
            if n.fract() != 0.0 {
              false
            }
            else {
              let is_even = (n as i64) % 2 == 0;
              if pred_name == "EvenQ" {
                is_even
              }
              else {
                !is_even
              }
            }
          }
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Unknown predicate function: {}",
              pred_name
            )));
          }
        };
        if passes {
          kept.push(evaluate_expression(elem.clone())?);
        }
      }
      return Ok(format!("{{{}}}", kept.join(", ")));
    }
    "Flatten" => {
      // ----- arity ---------------------------------------------------------
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Flatten expects exactly 1 argument".into(),
        ));
      }

      // ----- obtain the list ----------------------------------------------
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
              "Flatten argument must be a list".into(),
            ));
          }
        }
        else {
          return Err(InterpreterError::EvaluationError(
            "Flatten argument must be a list".into(),
          ));
        }
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "Flatten argument must be a list".into(),
        ));
      };

      // ----- recursive flattener ------------------------------------------
      fn collect_flat<'a>(
        pair: pest::iterators::Pair<'a, Rule>,
        acc: &mut Vec<pest::iterators::Pair<'a, Rule>>,
      ) {
        match pair.as_rule() {
          Rule::List => {
            for sub in pair.into_inner().filter(|p| p.as_str() != ",") {
              collect_flat(sub, acc);
            }
          }
          Rule::Expression => {
            let mut inner = pair.clone().into_inner();
            if let Some(first) = inner.next() {
              if first.as_rule() == Rule::List && inner.next().is_none() {
                collect_flat(first, acc);
                return;
              }
            }
            acc.push(pair);
          }
          _ => acc.push(pair),
        }
      }

      // ----- flatten -------------------------------------------------------
      let mut flat_pairs = Vec::new();
      for it in items {
        collect_flat(it, &mut flat_pairs);
      }

      // ----- evaluate & format --------------------------------------------
      let evaluated: Result<Vec<_>, _> = flat_pairs
        .into_iter()
        .map(|p| evaluate_expression(p))
        .collect();
      return Ok(format!("{{{}}}", evaluated?.join(", ")));
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

fn apply_map_operator(
  func: pest::iterators::Pair<Rule>,
  list: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  // right after the function’s opening brace
  let func_core = if func.as_rule() == Rule::Term {
    func.clone().into_inner().next().unwrap()
  }
  else {
    func.clone()
  };

  // ----- obtain list items (same extraction logic used in Map) -----
  let list_rule = list.as_rule();
  let elements: Vec<_> = if list_rule == Rule::List {
    list.into_inner().filter(|p| p.as_str() != ",").collect()
  }
  else if list_rule == Rule::Expression {
    let mut inner = list.into_inner();
    if let Some(first) = inner.next() {
      if first.as_rule() == Rule::List {
        first.into_inner().filter(|p| p.as_str() != ",").collect()
      }
      else {
        return Err(InterpreterError::EvaluationError(
          "Second operand of /@ must be a list".into(),
        ));
      }
    }
    else {
      return Err(InterpreterError::EvaluationError(
        "Second operand of /@ must be a list".into(),
      ));
    }
  }
  else {
    return Err(InterpreterError::EvaluationError(
      "Second operand of /@ must be a list".into(),
    ));
  };

  // ----- identify mapped function ----------------------------------
  match func_core.as_rule() {
    Rule::Identifier => {
      let name = func_core.as_str();
      match name {
        "Sign" => {
          let mut mapped = Vec::new();
          for el in elements {
            let v = evaluate_term(el.clone())?;
            let s = if v > 0.0 {
              1.0
            }
            else if v < 0.0 {
              -1.0
            }
            else {
              0.0
            };
            mapped.push(format_result(s));
          }
          Ok(format!("{{{}}}", mapped.join(", ")))
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "Unknown mapping function: {}",
          name
        ))),
      }
    }
    Rule::AnonymousFunction => {
      let parts: Vec<_> = func_core.clone().into_inner().collect();

      // identity function  (#&)
      if parts.len() == 1 {
        let mut out = Vec::new();
        for el in &elements {
          out.push(evaluate_expression(el.clone())?);
        }
        return Ok(format!("{{{}}}", out.join(", ")));
      }

      let operator = parts[1].as_str();
      let operand = parts[2].clone();
      let mut out = Vec::new();

      for el in elements {
        let v = evaluate_term(el.clone())?;
        let res = match operator {
          "+" => v + evaluate_term(operand.clone())?,
          "-" => v - evaluate_term(operand.clone())?,
          "*" => v * evaluate_term(operand.clone())?,
          "/" => {
            let d = evaluate_term(operand.clone())?;
            if d == 0.0 {
              return Err(InterpreterError::EvaluationError(
                "Division by zero".into(),
              ));
            }
            v / d
          }
          "^" => v.powf(evaluate_term(operand.clone())?),
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Unsupported operator in anonymous function: {}",
              operator
            )))
          }
        };
        out.push(format_result(res));
      }
      return Ok(format!("{{{}}}", out.join(", ")));
    }
    _ => Err(InterpreterError::EvaluationError(
      "Left operand of /@ must be a function".into(),
    )),
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
