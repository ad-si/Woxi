use chrono::Local;
use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

#[derive(Clone)]
enum StoredValue {
  Association(Vec<(String, String)>),
  Raw(String), // keep evaluated textual value
}
thread_local! {
    static ENV: RefCell<HashMap<String, StoredValue>> = RefCell::new(HashMap::new());
    //            name         (parameter names)      body-text
    static FUNC_DEFS: RefCell<HashMap<String, (Vec<String>, String)>> = RefCell::new(HashMap::new());
}

#[derive(Error, Debug)]
pub enum InterpreterError {
  #[error("Parse error: {0}")]
  ParseError(#[from] pest::error::Error<Rule>),
  #[error("Empty input")]
  EmptyInput,
  #[error("Evaluation error: {0}")]
  EvaluationError(String),
}

/// Extended result type that includes both stdout and the result
#[derive(Debug, Clone)]
pub struct InterpretResult {
  pub stdout: String,
  pub result: String,
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

// Captured output from Print statements
thread_local! {
    static CAPTURED_STDOUT: RefCell<String> = RefCell::new(String::new());
}

/// Clears the captured stdout buffer
fn clear_captured_stdout() {
  CAPTURED_STDOUT.with(|buffer| {
    buffer.borrow_mut().clear();
  });
}

/// Appends to the captured stdout buffer
fn capture_stdout(text: &str) {
  CAPTURED_STDOUT.with(|buffer| {
    buffer.borrow_mut().push_str(text);
    buffer.borrow_mut().push('\n');
  });
}

/// Gets the captured stdout content
fn get_captured_stdout() -> String {
  CAPTURED_STDOUT.with(|buffer| buffer.borrow().clone())
}

pub fn interpret(input: &str) -> Result<String, InterpreterError> {
  // Clear the stdout capture buffer
  clear_captured_stdout();

  // Regular interpretation
  let pairs = parse(input)?;
  let mut pairs = pairs.into_iter();
  let program = pairs.next().ok_or(InterpreterError::EmptyInput)?;

  if program.as_rule() != Rule::Program {
    return Err(InterpreterError::EvaluationError(format!(
      "Expected Program, got {:?}",
      program.as_rule()
    )));
  }

  let mut last_result = None;
  let mut any_nonempty = false;
  for node in program.into_inner() {
    match node.as_rule() {
      Rule::Expression => {
        last_result = Some(evaluate_expression(node)?);
        any_nonempty = true;
      }
      Rule::FunctionDefinition => {
        store_function_definition(node)?;
        any_nonempty = true;
      }
      _ => {} // ignore semicolons, etc.
    }
  }

  if any_nonempty {
    last_result.ok_or(InterpreterError::EmptyInput)
  } else {
    Err(InterpreterError::EmptyInput)
  }
}

/// New interpret function that returns both stdout and the result
pub fn interpret_with_stdout(
  input: &str,
) -> Result<InterpretResult, InterpreterError> {
  // Clear the stdout capture buffer
  clear_captured_stdout();

  // Perform the standard interpretation
  let result = interpret(input)?;

  // Get the captured stdout
  let stdout = get_captured_stdout();

  // Return both stdout and the result
  Ok(InterpretResult { stdout, result })
}

fn format_result(result: f64) -> String {
  if result.fract() == 0.0 {
    let int_result = result as i64;
    int_result.to_string()
  } else {
    format!("{:.10}", result)
      .trim_end_matches('0')
      .trim_end_matches('.')
      .to_string()
  }
}

// Parse display-form strings like "{1, 2, 3}" into top-level comma-separated
// element strings.  Returns None if `s` is not a braced list.
fn parse_list_string(s: &str) -> Option<Vec<String>> {
  if !(s.starts_with('{') && s.ends_with('}')) {
    return None;
  }
  let inner = &s[1..s.len() - 1];
  let mut parts = Vec::new();
  let mut depth = 0usize;
  let mut start = 0usize;
  for (i, c) in inner.char_indices() {
    match c {
      '{' | '[' | '(' | '<' => depth += 1,
      '}' | ']' | ')' | '>' => {
        if depth > 0 {
          depth -= 1;
        }
      }
      ',' if depth == 0 => {
        parts.push(inner[start..i].trim().to_string());
        start = i + 1;
      }
      _ => {}
    }
  }
  if start < inner.len() {
    parts.push(inner[start..].trim().to_string());
  }
  Some(parts)
}

fn store_function_definition(pair: Pair<Rule>) -> Result<(), InterpreterError> {
  // FunctionDefinition  :=  Identifier "[" Pattern "]" ":=" Expression
  let mut inner = pair.into_inner();
  let func_name = inner.next().unwrap().as_str().to_owned(); // Identifier
  let pattern = inner.next().unwrap(); // Pattern
  let param = pattern.as_str().trim_end_matches('_').to_owned();
  let body_pair = inner.next().unwrap(); // Expression
  let body_txt = body_pair.as_str().to_owned();

  FUNC_DEFS.with(|m| {
    m.borrow_mut().insert(func_name, (vec![param], body_txt));
  });
  Ok(())
}

fn eval_association(
  pair: Pair<Rule>,
) -> Result<(Vec<(String, String)>, String), InterpreterError> {
  let mut pairs = Vec::new();
  let mut disp_parts = Vec::new();
  for item in pair
    .into_inner()
    .filter(|p| p.as_rule() == Rule::AssociationItem)
  {
    let mut inner = item.into_inner();
    let key_pair = inner.next().unwrap();
    let val_pair = inner.next().unwrap();
    let key = extract_string(key_pair)?;
    let val = evaluate_expression(val_pair)?;
    disp_parts.push(format!("{} -> {}", key, val));
    pairs.push((key, val));
  }
  let disp = format!("<|{}|>", disp_parts.join(", "));
  Ok((pairs, disp))
}

fn evaluate_expression(
  expr: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  match expr.as_rule() {
    Rule::String => Ok(expr.as_str().trim_matches('"').to_string()),
    Rule::Association => {
      let (_pairs, disp) = eval_association(expr)?;
      Ok(disp)
    }
    Rule::PostfixApplication => {
      let mut inner = expr.into_inner();
      let arg = inner.next().unwrap();
      let func = inner.next().unwrap();

      if func.as_rule() == Rule::Identifier {
        let func_name = func.as_str();
        // Evaluate the argument
        let arg_value = evaluate_expression(arg)?;
        // Apply the function
        match func_name {
          "Sin" => {
            let n = arg_value.parse::<f64>().map_err(|_| {
              InterpreterError::EvaluationError(
                "Invalid argument for Sin".into(),
              )
            })?;
            Ok(format_result(n.sin()))
          }
          _ => Err(InterpreterError::EvaluationError(format!(
            "Unknown function for // operator: {}",
            func_name
          ))),
        }
      } else {
        Err(InterpreterError::EvaluationError(
          "Right operand of // must be a function".into(),
        ))
      }
    }
    Rule::NumericValue => {
      // numeric literal directly inside an expression (e.g. x == 2)
      return evaluate_term(expr).map(format_result);
    }
    Rule::Term => {
      let mut inner = expr.clone().into_inner();
      if let Some(first) = inner.next() {
        if first.as_rule() == Rule::FunctionCall {
          return evaluate_function_call(first);
        } else if first.as_rule() == Rule::List {
          return evaluate_expression(first);
        } else if first.as_rule() == Rule::Association {
          let (_pairs, disp) = eval_association(first)?;
          return Ok(disp);
        } else if first.as_rule() == Rule::String {
          return Ok(first.as_str().trim_matches('"').to_string());
        } else if first.as_rule() == Rule::Integer
          || first.as_rule() == Rule::Real
          || first.as_rule() == Rule::Constant
          || first.as_rule() == Rule::NumericValue
          || first.as_rule() == Rule::Identifier
          || first.as_rule() == Rule::Slot
        {
          return evaluate_term(first).map(format_result);
        }
        // --- handle PartExtract at Term level ---
        else if first.as_rule() == Rule::PartExtract {
          return evaluate_expression(first);
        }
      }
      evaluate_term(expr).map(format_result)
    }
    Rule::PartExtract => {
      let mut inner = expr.into_inner();
      let id_pair = inner.next().unwrap();
      let key_pair = inner.next().unwrap();
      let name = id_pair.as_str();
      let key = extract_string(key_pair)?;
      if let Some(StoredValue::Association(vec)) =
        ENV.with(|e| e.borrow().get(name).cloned())
      {
        for (k, v) in vec {
          if k == key {
            return Ok(v);
          }
        }
        return Err(InterpreterError::EvaluationError("Key not found".into()));
      }
      // If not found as association, try to evaluate as a list and use numeric index
      if let Ok(list_str) = evaluate_expression(id_pair.clone()) {
        // Try to parse as a list: {a, b, c}
        if list_str.starts_with('{') && list_str.ends_with('}') {
          let items: Vec<&str> = list_str[1..list_str.len() - 1]
            .split(',')
            .map(|s| s.trim())
            .collect();
          if let Ok(idx) = key.parse::<usize>() {
            if idx >= 1 && idx <= items.len() {
              return Ok(items[idx - 1].to_string());
            }
          }
        }
      }
      return Err(InterpreterError::EvaluationError(
        "Argument must be an association".into(),
      ));
    }
    Rule::Expression => {
      // --- special case: Map operator ----------------------------------
      {
        let items: Vec<_> = expr.clone().into_inner().collect();

        // Handle operators for function application
        if items.len() == 3 && items[1].as_rule() == Rule::Operator {
          // Handle @ operator (prefix notation)
          if items[1].as_str() == "@" {
            let func = items[0].clone();
            let arg = items[2].clone();

            if func.as_rule() == Rule::Identifier {
              let func_name = func.as_str();
              // Directly call the function with the argument value
              let arg_value = evaluate_expression(arg)?;
              // Create args_pairs similar to a normal function call
              return match func_name {
                "Sin" => {
                  let n = arg_value.parse::<f64>().map_err(|_| {
                    InterpreterError::EvaluationError(
                      "Invalid argument for Sin".into(),
                    )
                  })?;
                  Ok(format_result(n.sin()))
                }
                _ => Err(InterpreterError::EvaluationError(format!(
                  "Unknown function for @ operator: {}",
                  func_name
                ))),
              };
            } else {
              return Err(InterpreterError::EvaluationError(
                "Left operand of @ must be a function".into(),
              ));
            }
          }
          // Handle // operator (postfix notation)
          else if items[1].as_str() == "//" {
            let arg = items[0].clone();
            let func = items[2].clone();

            if func.as_rule() == Rule::Identifier {
              let func_name = func.as_str();
              // Directly call the function with the argument value
              let arg_value = evaluate_expression(arg)?;
              return match func_name {
                "Sin" => {
                  let n = arg_value.parse::<f64>().map_err(|_| {
                    InterpreterError::EvaluationError(
                      "Invalid argument for Sin".into(),
                    )
                  })?;
                  Ok(format_result(n.sin()))
                }
                _ => Err(InterpreterError::EvaluationError(format!(
                  "Unknown function for // operator: {}",
                  func_name
                ))),
              };
            } else {
              return Err(InterpreterError::EvaluationError(
                "Right operand of // must be a function".into(),
              ));
            }
          }
        }

        if items.len() == 3
          && items[1].as_rule() == Rule::Operator
          && items[1].as_str() == "/@"
        {
          return apply_map_operator(items[0].clone(), items[2].clone());
        }

        // --- handle = and := assignment operators ---
        if items.len() >= 3 && items[1].as_rule() == Rule::Operator {
          match items[1].as_str() {
            "=" => {
              // LHS must be an identifier
              let lhs = items[0].clone();
              if lhs.as_rule() != Rule::Identifier {
                return Err(InterpreterError::EvaluationError(
                  "Left-hand side of assignment must be an identifier".into(),
                ));
              }
              let name = lhs.as_str().to_string();

              // --- association assignment  (x = <| … |>) -------------------------
              if items.len() == 3 && items[2].as_rule() == Rule::Association {
                let (pairs, disp) = eval_association(items[2].clone())?;
                ENV.with(|e| {
                  e.borrow_mut().insert(name, StoredValue::Association(pairs))
                });
                return Ok(disp);
              }

              // --- generic RHS: may be any (possibly complex) expression ----------
              // evaluate everything that comes after the first '='
              let full_txt = expr.as_str();
              // Find the first '=' that is not part of an operator like '==' or '!='
              // This is a simple approach: split on the first '=' that is not preceded or followed by '='
              // (Assumes no whitespace between '='s in '==', '!=' etc.)
              let mut eq_index = None;
              let chars: Vec<char> = full_txt.chars().collect();
              for i in 0..chars.len() {
                if chars[i] == '=' {
                  let prev = if i > 0 { chars[i - 1] } else { '\0' };
                  let next = if i + 1 < chars.len() {
                    chars[i + 1]
                  } else {
                    '\0'
                  };
                  if prev != '=' && next != '=' {
                    eq_index = Some(i);
                    break;
                  }
                }
              }
              let rhs_txt = if let Some(idx) = eq_index {
                &full_txt[idx + 1..]
              } else {
                ""
              };
              let rhs_txt = rhs_txt.trim();

              let val = interpret(rhs_txt)?; // recursive evaluation
              ENV.with(|e| {
                e.borrow_mut().insert(name, StoredValue::Raw(val.clone()))
              });
              return Ok(val);
            }
            ":=" => {
              // Detect a definition like  f[x_] := body
              let lhs_pair = items[0].clone();

              // unwrap a possible Term wrapper so we can directly
              // look at the FunctionCall node
              let func_call_pair = if lhs_pair.as_rule() == Rule::Term {
                let mut inner = lhs_pair.clone().into_inner();
                let first = inner.next().unwrap();
                if first.as_rule() == Rule::FunctionCall {
                  first
                } else {
                  lhs_pair.clone()
                }
              } else {
                lhs_pair.clone()
              };

              // Only treat it as a user-function definition if the left side
              // really is a `FunctionCall` whose arguments are *patterns*
              if func_call_pair.as_rule() == Rule::FunctionCall {
                let mut fc_inner = func_call_pair.clone().into_inner();
                let func_name = fc_inner.next().unwrap().as_str().to_owned();

                // collect all pattern arguments (e.g. x_, y_, …  →  ["x","y"])
                let mut params = Vec::new();
                for arg in fc_inner.filter(|p| p.as_rule() == Rule::Pattern) {
                  params.push(arg.as_str().trim_end_matches('_').to_owned());
                }

                // Obtain the complete *text* to the right of the ":=" so we
                // don't lose operators like "* 2"
                let full_txt = expr.as_str();

                // find the first occurrence of ":="
                let rhs_txt = full_txt
                  .splitn(2, ":=")
                  .nth(1)
                  .unwrap_or("")
                  .trim()
                  .to_owned();

                // keep previously-computed parameter list (may be empty if
                // the parser didn't yield Pattern nodes)
                let body_txt = rhs_txt;

                FUNC_DEFS.with(|m| {
                  m.borrow_mut().insert(func_name, (params, body_txt));
                });

                return Ok("Null".to_string());
              }

              // not a function definition → keep previous behaviour
              return Ok("Null".to_string());
            }
            _ => { /* fall-through to the maths/other logic below */ }
          }
        }

        // --- relational operators '==' and '!=' ------------------------------
        if items.len() >= 3 && items.len() % 2 == 1 {
          let all_eq = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "==");
          let all_neq = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "!=");

          if all_eq {
            // Evaluate all sub-expressions and compare as strings
            let ref_val = evaluate_expression(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cmp_val = evaluate_expression(items[idx].clone())?;
              if cmp_val != ref_val {
                return Ok("False".to_string());
              }
            }
            return Ok("True".to_string());
          }
          if all_neq {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for idx in (0..items.len()).step_by(2) {
              let v = evaluate_expression(items[idx].clone())?;
              if !seen.insert(v) {
                return Ok("False".to_string());
              }
            }
            return Ok("True".to_string());
          }

          let all_gt = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == ">");
          let all_lt = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "<");
          let all_ge = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == ">=");
          let all_le = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "<=");

          if all_gt {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if !(prev > cur) {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          if all_lt {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if !(prev < cur) {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          if all_ge {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if prev < cur {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          if all_le {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if prev > cur {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          // --- mixed <  >  <=  >= comparisons -----------------------------
          let all_ineq = items.iter().skip(1).step_by(2).all(|p| {
            p.as_rule() == Rule::Operator
              && matches!(p.as_str(), ">" | "<" | ">=" | "<=")
          });

          if all_ineq {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (1..items.len()).step_by(2) {
              let op = items[idx].as_str();
              let cur = evaluate_term(items[idx + 1].clone())?;

              let ok = match op {
                ">" => prev > cur,
                "<" => prev < cur,
                ">=" => prev >= cur,
                "<=" => prev <= cur,
                _ => unreachable!(),
              };
              if !ok {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
        }
      }
      let mut inner = expr.into_inner();
      let first = inner.next().unwrap();
      if inner.clone().next().is_none() {
        if first.as_rule() == Rule::List {
          return evaluate_expression(first);
        } else if first.as_rule() == Rule::Identifier {
          // Evaluate identifier as in the main Rule::Identifier arm
          let id = first.as_str();
          if id == "Now" {
            return Ok("CURRENT_TIME_MARKER".to_string());
          }
          if let Some(stored) = ENV.with(|e| e.borrow().get(id).cloned()) {
            return Ok(match stored {
              StoredValue::Association(pairs) => format!(
                "<|{}|>",
                pairs
                  .iter()
                  .map(|(k, v)| format!("{} -> {}", k, v))
                  .collect::<Vec<_>>()
                  .join(", ")
              ),
              StoredValue::Raw(val) => val,
            });
          }
          return Ok(id.to_string());
        } else if first.as_rule() == Rule::FunctionCall {
          return evaluate_function_call(first);
        } else if first.as_rule() == Rule::Term {
          return evaluate_expression(first.into_inner().next().unwrap());
        } else if first.as_rule() == Rule::NumericValue {
          // Evaluate the numeric value as a number and format as string
          return evaluate_term(first).map(format_result);
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
        } else if ops[i] == "/" {
          if values[i + 1] == 0.0 {
            return Err(InterpreterError::EvaluationError(
              "Division by zero".to_string(),
            ));
          }
          values[i] = values[i] / values[i + 1];
          values.remove(i + 1);
          ops.remove(i);
        } else {
          i += 1;
        }
      }
      // Second pass: handle addition and subtraction
      let mut result = values[0];
      for (op, &val) in ops.iter().zip(values.iter().skip(1)) {
        if *op == "+" {
          result += val;
        } else if *op == "-" {
          result -= val;
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "Unexpected operator: {}",
            op
          )));
        }
      }
      Ok(format_result(result))
    }
    Rule::Program => {
      let mut last = None;
      for node in expr.into_inner() {
        match node.as_rule() {
          Rule::Expression => {
            last = Some(evaluate_expression(node)?);
          }
          Rule::FunctionDefinition => {
            store_function_definition(node)?;
            last = Some("Null".to_string());
          }
          _ => {}
        }
      }
      return last.ok_or(InterpreterError::EmptyInput);
    }
    Rule::List => {
      let items: Vec<String> = expr
        .into_inner()
        .filter(|item| item.as_str() != ",")
        .map(|item| evaluate_expression(item))
        .collect::<Result<_, _>>()?;
      Ok(format!("{{{}}}", items.join(", ")))
    }
    Rule::FunctionCall => evaluate_function_call(expr),
    Rule::Identifier => {
      let id = expr.as_str();
      if id == "Now" {
        return Ok("CURRENT_TIME_MARKER".to_string());
      }
      if let Some(stored) = ENV.with(|e| e.borrow().get(id).cloned()) {
        return Ok(match stored {
          StoredValue::Association(pairs) => format!(
            "<|{}|>",
            pairs
              .iter()
              .map(|(k, v)| format!("{} -> {}", k, v))
              .collect::<Vec<_>>()
              .join(", ")
          ),
          StoredValue::Raw(val) => val,
        });
      }
      Ok(id.to_string())
    }
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
      } else if s == "False" {
        Ok(0.0)
      } else {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      }
    }),
    Rule::Identifier => {
      match term.as_str() {
        "True" => Ok(1.0),
        "False" => Ok(0.0),
        "Now" => Err(InterpreterError::EvaluationError(
          "Identifier 'Now' cannot be directly used as a numeric value."
            .to_string(),
        )),
        id => {
          if let Some(StoredValue::Raw(val)) =
            ENV.with(|e| e.borrow().get(id).cloned())
          {
            return val
              .parse::<f64>()
              .map_err(|e| InterpreterError::EvaluationError(e.to_string()));
          }
          Ok(0.0) // unknown / non-numeric identifier
        }
      }
    }
    Rule::Slot => {
      // For slot (#), we'll return 1.0 as a default value when evaluated as a term
      // It will be replaced with the actual value in the anonymous function evaluation
      Ok(1.0)
    }
    Rule::List => Err(InterpreterError::EvaluationError(
      "Cannot evaluate a list as a numeric value".to_string(),
    )),
    Rule::PartExtract => {
      // Instead of error, evaluate as expression (so PartExtract can be handled at expression level)
      evaluate_expression(term).and_then(|s| {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      })
    }
    Rule::PostfixApplication => {
      // Evaluate as expression and convert to number
      evaluate_expression(term).and_then(|s| {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      })
    }
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
        // If the inner part is a string, extract it.
        // Otherwise, evaluate the expression and hope it's a string.
        if first.as_rule() == Rule::String {
          return Ok(first.as_str().trim_matches('"').to_string());
        }
        // Fallback to evaluate_expression if not directly a string.
        // This handles cases like `DateString[Now, "ISO" <> "DateTime"]` if StringJoin was more general
        // or if the argument is a variable that holds a string.
        return evaluate_expression(pair);
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

  // treat identifier that refers to an association variable
  if func_name_pair.as_rule() == Rule::Identifier {
    if let Some(StoredValue::Association(pairs)) =
      ENV.with(|e| e.borrow().get(func_name_pair.as_str()).cloned())
    {
      // ---- key lookup ---
      if inner.clone().filter(|p| p.as_str() != ",").count() == 1 {
        let arg_pair = inner.next().unwrap(); // only argument
        let key_str = extract_string(arg_pair)?; // must be string
        for (k, v) in &pairs {
          if *k == key_str {
            return Ok(v.clone());
          }
        }
        return Err(InterpreterError::EvaluationError("Key not found".into()));
      }
    }
  }

  // Helper for boolean conversion
  fn as_bool(s: &str) -> Option<bool> {
    match s {
      "True" => Some(true),
      "False" => Some(false),
      _ => None,
    }
  }

  // Helper for extracting association from first argument
  fn get_assoc_from_first_arg(
    args: &[Pair<Rule>],
  ) -> Result<Vec<(String, String)>, InterpreterError> {
    let p = &args[0];

    // NEW: recognise an identifier that is wrapped in an Expression
    if p.as_rule() == Rule::Expression {
      let mut inner = p.clone().into_inner();
      if let Some(first) = inner.next() {
        if first.as_rule() == Rule::Identifier && inner.next().is_none() {
          if let Some(StoredValue::Association(v)) =
            ENV.with(|e| e.borrow().get(first.as_str()).cloned())
          {
            return Ok(v);
          }
        }
      }
    }

    if p.as_rule() == Rule::Identifier {
      if let Some(StoredValue::Association(v)) =
        ENV.with(|e| e.borrow().get(p.as_str()).cloned())
      {
        return Ok(v);
      }
    }
    if p.as_rule() == Rule::Association {
      return Ok(eval_association(p.clone())?.0);
    }
    // Try to evaluate as an expression and parse as association display
    if let Ok(val) = evaluate_expression(p.clone()) {
      if val.starts_with("<|") && val.ends_with("|>") {
        let inner_val = &val[2..val.len() - 2];
        let mut pairs = Vec::new();
        // Only split on top-level commas (not inside braces or quotes)
        let mut depth = 0;
        let mut start = 0;
        let mut parts = Vec::new();
        let chars: Vec<char> = inner_val.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
          match c {
            '{' | '<' | '[' | '(' => depth += 1,
            '}' | '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
              parts.push(inner_val[start..i].to_string());
              start = i + 1;
            }
            _ => {}
          }
        }
        if start < chars.len() {
          parts.push(inner_val[start..].to_string());
        }
        for part in parts {
          let part_trimmed = part.trim();
          if let Some((k, v_str)) = part_trimmed.split_once("->") {
            pairs.push((k.trim().to_string(), v_str.trim().to_string()));
          }
        }
        return Ok(pairs);
      }
    }
    Err(InterpreterError::EvaluationError(
      "Argument must be an association".into(),
    ))
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
          } else {
            return Err(InterpreterError::EvaluationError(
              "Anonymous function must be applied to a list".to_string(),
            ));
          }
        } else {
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

  // ----- user-defined functions ------------------------------------------
  if let Some((params, body)) =
    FUNC_DEFS.with(|m| m.borrow().get(func_name).cloned())
  {
    // Patch: If the function definition has 0 parameters, but the call has 1 argument,
    // and the function was defined as f[x_] := ..., treat it as a single-parameter function.
    if params.is_empty() && args_pairs.len() == 1 {
      // Try to extract the parameter name from the function definition's body
      // (since the parser may have failed to store the param)
      // We'll use "x" as a default parameter name.
      let param = "x".to_string();
      let val = evaluate_expression(args_pairs[0].clone())?;
      let prev = ENV.with(|e| {
        e.borrow_mut()
          .insert(param.clone(), StoredValue::Raw(val.clone()))
      });
      let result = interpret(&body)?;
      ENV.with(|e| {
        let mut env = e.borrow_mut();
        if let Some(v) = prev {
          env.insert(param, v);
        } else {
          env.remove(&param);
        }
      });
      return Ok(result);
    }

    if args_pairs.len() != params.len() {
      return Err(InterpreterError::EvaluationError(format!(
        "{} called with {} arguments; {} expected",
        func_name,
        args_pairs.len(),
        params.len()
      )));
    }

    // evaluate actual arguments
    let mut arg_vals = Vec::new();
    for p in &args_pairs {
      arg_vals.push(evaluate_expression(p.clone())?);
    }

    // save previous bindings, bind new ones
    let mut prev: Vec<(String, Option<StoredValue>)> = Vec::new();
    for (param, val) in params.iter().zip(arg_vals.iter()) {
      let pv = ENV.with(|e| {
        e.borrow_mut()
          .insert(param.clone(), StoredValue::Raw(val.clone()))
      });
      prev.push((param.clone(), pv));
    }

    // evaluate body inside the extended environment
    let result = interpret(&body)?;

    // restore previous bindings
    for (param, old) in prev {
      ENV.with(|e| {
        let mut env = e.borrow_mut();
        if let Some(v) = old {
          env.insert(param, v);
        } else {
          env.remove(&param);
        }
      });
    }
    return Ok(result);
  }

  match func_name {
    "DateString" => {
      let current_time = Local::now();
      let default_format = "%a, %d %b %Y %H:%M:%S"; // e.g., Mon, 29 Jul 2024 10:30:00

      match args_pairs.len() {
        0 => { // DateString[]
          Ok(current_time.format(default_format).to_string())
        }
        1 => { // DateString[Now] or DateString["format"]
          let arg1_eval_result = evaluate_expression(args_pairs[0].clone());
          match arg1_eval_result {
            Ok(arg1_val) if arg1_val == "CURRENT_TIME_MARKER" => { // DateString[Now]
              Ok(current_time.format(default_format).to_string())
            }
            _ => { // DateString["format_string"] - evaluate_expression might fail if not a string
              let format_str = extract_string(args_pairs[0].clone())?;
              match format_str.as_str() {
                "ISODateTime" => Ok(current_time.format("%Y-%m-%dT%H:%M:%S").to_string()),
                _ => Ok(current_time.format(&format_str).to_string()), // Attempt direct chrono format
              }
            }
          }
        }
        2 => { // DateString[date_spec, "format_string"]
          let date_spec_eval = evaluate_expression(args_pairs[0].clone())?;
          if date_spec_eval != "CURRENT_TIME_MARKER" {
            return Err(InterpreterError::EvaluationError(
              "DateString: First argument currently must be Now.".into(),
            ));
          }
          // It's DateString[Now, "format_string"]
          let format_str = extract_string(args_pairs[1].clone())?;
          match format_str.as_str() {
            "ISODateTime" => Ok(current_time.format("%Y-%m-%dT%H:%M:%S").to_string()),
            _ => Ok(current_time.format(&format_str).to_string()), // Attempt direct chrono format
          }
        }
        _ => Err(InterpreterError::EvaluationError(
          "DateString: Called with invalid number of arguments. Expected 0, 1, or 2.".into(),
        )),
      }
    }
    "Keys" => {
      let asso = get_assoc_from_first_arg(&args_pairs)?;
      let keys: Vec<_> = asso.iter().map(|(k, _)| k.clone()).collect();
      return Ok(format!("{{{}}}", keys.join(", ")));
    }
    "Values" => {
      let asso = get_assoc_from_first_arg(&args_pairs)?;
      let vals: Vec<_> = asso.iter().map(|(_, v)| v.clone()).collect();
      return Ok(format!("{{{}}}", vals.join(", ")));
    }
    "KeyDropFrom" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "KeyDropFrom expects exactly 2 arguments".into(),
        ));
      }
      let mut asso = get_assoc_from_first_arg(&args_pairs)?;
      let key = extract_string(args_pairs[1].clone())?;
      asso.retain(|(k, _)| k != &key);
      let disp = format!(
        "<|{}|>",
        asso
          .iter()
          .map(|(k, v)| format!("{} -> {}", k, v))
          .collect::<Vec<_>>()
          .join(", ")
      );
      return Ok(disp);
    }
    "Set" => {
      // --- arity -----------------------------------------------------------
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Set expects exactly 2 arguments".into(),
        ));
      }

      // --- extract variable name (first arg must be an identifier) ---------
      let var_pair = &args_pairs[0];
      let var_name = match var_pair.as_rule() {
        Rule::Identifier => var_pair.as_str().to_string(),
        Rule::Expression => {
          let mut inner = var_pair.clone().into_inner();
          if let Some(first) = inner.next() {
            if first.as_rule() == Rule::Identifier && inner.next().is_none() {
              first.as_str().to_string()
            } else {
              return Err(InterpreterError::EvaluationError(
                "First argument of Set must be an identifier".into(),
              ));
            }
          } else {
            return Err(InterpreterError::EvaluationError(
              "First argument of Set must be an identifier".into(),
            ));
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "First argument of Set must be an identifier".into(),
          ))
        }
      };

      // --- evaluate & store RHS -------------------------------------------
      let rhs_pair = &args_pairs[1];
      if rhs_pair.as_rule() == Rule::Association {
        let (pairs, disp) = eval_association(rhs_pair.clone())?;
        ENV.with(|e| {
          e.borrow_mut()
            .insert(var_name, StoredValue::Association(pairs))
        });
        return Ok(disp);
      } else {
        let val = evaluate_expression(rhs_pair.clone())?;
        ENV.with(|e| {
          e.borrow_mut()
            .insert(var_name, StoredValue::Raw(val.clone()))
        });
        return Ok(val);
      }
    }
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
    "Sin" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Sin expects exactly 1 argument".into(),
        ));
      }
      let n = evaluate_term(args_pairs[0].clone())?;
      Ok(format_result(n.sin()))
    }
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
        } else if n < 0.0 {
          "-1"
        } else {
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
      if r == -0.0 {
        r = 0.0;
      }
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
      if r == -0.0 {
        r = 0.0;
      }
      return Ok(format_result(r));
    }

    "Round" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Round expects exactly 1 argument".into(),
        ));
      }
      let n = evaluate_term(args_pairs[0].clone())?;

      // banker's rounding (half-to-even)
      let base = n.trunc();
      let frac = n - base;
      let mut r = if frac.abs() == 0.5 {
        if (base as i64) % 2 == 0 {
          base
        }
        // already even
        else if n.is_sign_positive() {
          base + 1.0
        }
        // away from zero
        else {
          base - 1.0
        }
      } else {
        n.round()
      };
      if r == -0.0 {
        r = 0.0;
      }
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
      } else {
        s.split(&delim).map(|p| p.to_string()).collect()
      };
      return Ok(format!("{{{}}}", parts.join(", ")));
    }

    "StringStartsQ" => {
      // expects exactly 2 string arguments
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "StringStartsQ expects exactly 2 arguments".into(),
        ));
      }
      let s = extract_string(args_pairs[0].clone())?;
      let prefix = extract_string(args_pairs[1].clone())?;
      return Ok(
        if s.starts_with(&prefix) {
          "True"
        } else {
          "False"
        }
        .to_string(),
      );
    }

    "StringEndsQ" => {
      // expects exactly 2 string arguments
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "StringEndsQ expects exactly 2 arguments".into(),
        ));
      }
      let s = extract_string(args_pairs[0].clone())?;
      let suffix = extract_string(args_pairs[1].clone())?;
      return Ok(
        if s.ends_with(&suffix) {
          "True"
        } else {
          "False"
        }
        .to_string(),
      );
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Second argument of Map must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "Second argument of Map must be a list".into(),
          ));
        }
      } else {
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
            } else if num < 0.0 {
              -1.0
            } else {
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
    "NumberQ" => {
      // ----- arity check --------------------------------------------------
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "NumberQ expects exactly 1 argument".into(),
        ));
      }

      // Evaluate argument to string and try to parse it as f64
      let arg_str = evaluate_expression(args_pairs[0].clone())?;
      let is_number = arg_str.parse::<f64>().is_ok();
      return Ok(if is_number { "True" } else { "False" }.to_string());
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
        } else {
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(format!(
              "{} function argument must be a list",
              func_name
            )));
          }
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "{} function argument must be a list",
            func_name
          )));
        }
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} function argument must be a list",
          func_name
        )));
      };
      let target = if func_name == "First" {
        items.first()
      } else {
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(format!(
              "{} function argument must be a list",
              func_name
            )));
          }
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "{} function argument must be a list",
            func_name
          )));
        }
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} function argument must be a list",
          func_name
        )));
      };
      let slice: Vec<_> = if func_name == "Rest" {
        if items.len() <= 1 {
          vec![]
        } else {
          items[1..].to_vec()
        }
      } else {
        // Most
        if items.len() <= 1 {
          vec![]
        } else {
          items[..items.len() - 1].to_vec()
        }
      };
      let evaluated: Result<Vec<_>, _> =
        slice.into_iter().map(|p| evaluate_expression(p)).collect();
      return Ok(format!("{{{}}}", evaluated?.join(", ")));
    }

    "MemberQ" => {
      // ---------- arity ----------------------------------------------------
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "MemberQ expects exactly 2 arguments".into(),
        ));
      }

      // ---------- obtain list items (same logic as in Length/Take/etc.) ----
      let list_pair = &args_pairs[0];
      let list_rule = list_pair.as_rule();
      let items: Vec<_> = if list_rule == Rule::List {
        list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect()
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "First argument of MemberQ must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "First argument of MemberQ must be a list".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "First argument of MemberQ must be a list".into(),
        ));
      };

      // ---------- evaluate element to look for -----------------------------
      let target = evaluate_expression(args_pairs[1].clone())?;

      // ---------- search ----------------------------------------------------
      for it in items {
        if evaluate_expression(it.clone())? == target {
          return Ok("True".to_string());
        }
      }
      return Ok("False".to_string());
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Take function argument must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "Take function argument must be a list".into(),
          ));
        }
      } else {
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "First argument of Drop must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "First argument of Drop must be a list".into(),
          ));
        }
      } else {
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(format!(
              "First argument of {} must be a list",
              func_name
            )));
          }
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "First argument of {} must be a list",
            func_name
          )));
        }
      } else {
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
      } else {
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Part function argument must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "Part function argument must be a list".into(),
          ));
        }
      } else {
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first
              .into_inner()
              .filter(|p| p.as_str() != ",")
              .collect::<Vec<_>>()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Length function argument must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "Length function argument must be a list".into(),
          ));
        }
      } else {
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Total function argument must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "Total function argument must be a list".into(),
          ));
        }
      } else {
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
      let numerator = evaluate_term(args_pairs[0].clone())?;
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "First argument of Select must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "First argument of Select must be a list".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "First argument of Select must be a list".into(),
        ));
      };

      // ----- identify predicate -------------------------------------------
      let pred_pair = &args_pairs[1];
      let pred_src = pred_pair.as_str();
      let is_slot_pred = pred_pair.as_rule() == Rule::AnonymousFunction
        || (pred_src.contains('#') && pred_src.ends_with('&'));

      // ----- filter --------------------------------------------------------
      let mut kept = Vec::new();
      for elem in elems {
        let passes = if is_slot_pred {
          // build expression by substituting the Slot (#) with the element’s
          // evaluated value and dropping the trailing ‘&’
          let mut expr = pred_src.trim_end_matches('&').to_string();
          let elem_str = evaluate_expression(elem.clone())?;
          expr = expr.replace('#', &elem_str);
          // evaluate the resulting Wolfram-expression
          let res = interpret(&expr)?;
          as_bool(&res).unwrap_or(false)
        } else {
          match pred_src {
            "EvenQ" | "OddQ" => {
              let n = evaluate_term(elem.clone())?;
              if n.fract() != 0.0 {
                false
              } else {
                let even = (n as i64) % 2 == 0;
                if pred_src == "EvenQ" {
                  even
                } else {
                  !even
                }
              }
            }
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "Unknown predicate function: {}",
                pred_src
              )))
            }
          }
        };
        if passes {
          kept.push(evaluate_expression(elem.clone())?);
        }
      }
      return Ok(format!("{{{}}}", kept.join(", ")));
    }
    "AllTrue" => {
      // ---------- arity ----------------------------------------------------
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "AllTrue expects exactly 2 arguments".into(),
        ));
      }

      // ---------- obtain list items (accept real list OR a value that
      // ---------- evaluates to a displayed list string) ---------------
      let list_pair = &args_pairs[0];
      let elements: Vec<String> = match list_pair.as_rule() {
        // syntactic list
        Rule::List => list_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .map(|p| evaluate_expression(p))
          .collect::<Result<_, _>>()?,
        // expression that syntactically contains a list
        Rule::Expression => {
          let mut expr_inner = list_pair.clone().into_inner();
          if let Some(first) = expr_inner.next() {
            if first.as_rule() == Rule::List && expr_inner.next().is_none() {
              first
                .into_inner()
                .filter(|p| p.as_str() != ",")
                .map(|p| evaluate_expression(p))
                .collect::<Result<_, _>>()?
            } else {
              // fall back to runtime evaluation
              let val = evaluate_expression(list_pair.clone())?;
              parse_list_string(&val).ok_or_else(|| {
                InterpreterError::EvaluationError(
                  "First argument of AllTrue must be a list".into(),
                )
              })?
            }
          } else {
            return Err(InterpreterError::EvaluationError(
              "First argument of AllTrue must be a list".into(),
            ));
          }
        }
        // any other form – evaluate, then try to parse "{…}"
        _ => {
          let val = evaluate_expression(list_pair.clone())?;
          parse_list_string(&val).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "First argument of AllTrue must be a list".into(),
            )
          })?
        }
      };

      // ---------- identify predicate --------------------------------------
      let pred_pair = &args_pairs[1];
      let pred_src = pred_pair.as_str();
      let is_slot_pred = pred_pair.as_rule() == Rule::AnonymousFunction
        || (pred_src.contains('#') && pred_src.ends_with('&'));

      // ---------- test every element --------------------------------------
      for elem_str in elements {
        let passes = if is_slot_pred {
          // substitute # and evaluate
          let mut expr = pred_src.trim_end_matches('&').to_string();
          expr = expr.replace('#', &elem_str);
          let res = interpret(&expr)?;
          as_bool(&res).unwrap_or(false)
        } else {
          match pred_src {
            "EvenQ" | "OddQ" => {
              let expr = format!("{}[{}]", pred_src, elem_str);
              let res = interpret(&expr)?;
              as_bool(&res).unwrap_or(false)
            }
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "Unknown predicate function: {}",
                pred_src
              )))
            }
          }
        };
        if !passes {
          return Ok("False".to_string());
        }
      }
      return Ok("True".to_string());
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
      } else if list_rule == Rule::Expression {
        let mut expr_inner = list_pair.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if first.as_rule() == Rule::List {
            first.into_inner().filter(|p| p.as_str() != ",").collect()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Flatten argument must be a list".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "Flatten argument must be a list".into(),
          ));
        }
      } else {
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
      // 0 args → just output a newline and return Null
      if args_pairs.is_empty() {
        println!(); // visible newline for CLI tests
        capture_stdout(""); // keep Jupyter stdout behaviour
        return Ok("Null".to_string());
      }
      // 1 arg accepted, anything else is still an error
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Print expects at most 1 argument".into(),
        ));
      }
      // Accept string, or expression wrapping a string, or any printable value
      let arg_pair = &args_pairs[0];
      let arg_str = match arg_pair.as_rule() {
        Rule::String => {
          // Get the raw string with quotes
          let raw_str = arg_pair.as_str();
          // Unescape the string content properly
          if let Ok(unescaped) = snailquote::unescape(raw_str) {
            unescaped
          } else {
            // Fall back to trimming quotes if unescaping fails
            raw_str.trim_matches('"').to_string()
          }
        }
        Rule::Expression => {
          let mut expr_inner = arg_pair.clone().into_inner();
          if let Some(first) = expr_inner.next() {
            if first.as_rule() == Rule::String {
              // Same unescaping for string inside expression
              let raw_str = first.as_str();
              if let Ok(unescaped) = snailquote::unescape(raw_str) {
                unescaped
              } else {
                raw_str.trim_matches('"').to_string()
              }
            } else {
              evaluate_expression(arg_pair.clone())?
            }
          } else {
            evaluate_expression(arg_pair.clone())?
          }
        }
        _ => evaluate_expression(arg_pair.clone())?,
      };
      // Print to stdout for normal display
      println!("{}", arg_str);
      // Also capture for Jupyter
      capture_stdout(&arg_str);
      Ok("Null".to_string())
    }
    "RandomInteger" => {
      // 0 args  ──────────────────────────────  → {0,1}
      if args_pairs.is_empty() {
        // Return a random integer 0 or 1 as a string (not empty string!)
        let v: i64 = rand::thread_rng().gen_range(0..=1);
        return Ok(v.to_string());
      }

      // 1 or 2 args: first must be a list {min,max}
      if !(args_pairs.len() == 1 || args_pairs.len() == 2) {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger called with wrong number of arguments".into(),
        ));
      }

      // ---- extract range -------------------------------------------------
      let range_pair = &args_pairs[0];
      // accept plain List or Expression-wrapped List
      let list_items: Vec<_> = match range_pair.as_rule() {
        Rule::List => range_pair
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect(),
        Rule::Expression => {
          let mut inner = range_pair.clone().into_inner();
          if let Some(first) = inner.next() {
            if first.as_rule() == Rule::List {
              first.into_inner().filter(|p| p.as_str() != ",").collect()
            } else {
              vec![]
            }
          } else {
            vec![]
          }
        }
        _ => vec![],
      };
      if list_items.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger range must be a list with two elements".into(),
        ));
      }
      let min = evaluate_term(list_items[0].clone())?;
      let max = evaluate_term(list_items[1].clone())?;
      if min.fract() != 0.0 || max.fract() != 0.0 || min > max {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger range must be two integers in ascending order".into(),
        ));
      }
      let (min_i, max_i) = (min as i64, max as i64);

      // ---- single value --------------------------------------------------
      if args_pairs.len() == 1 {
        let v = rand::thread_rng().gen_range(min_i..=max_i);
        return Ok(v.to_string());
      }

      // ---- list of n values ---------------------------------------------
      let n = evaluate_term(args_pairs[1].clone())?;
      if n.fract() != 0.0 || n < 0.0 {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger list length must be a non-negative integer".into(),
        ));
      }
      let n_usize = n as usize;
      let mut rng = rand::thread_rng();
      let values: Vec<String> = (0..n_usize)
        .map(|_| rng.gen_range(min_i..=max_i).to_string())
        .collect();
      return Ok(format!("{{{}}}", values.join(", ")));
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
  // right after the function's opening brace
  let func_core = if func.as_rule() == Rule::Term {
    func.clone().into_inner().next().unwrap()
  } else {
    func.clone()
  };

  // ----- obtain list items (same extraction logic used in Map) -----
  let list_rule = list.as_rule();
  let elements: Vec<_> = if list_rule == Rule::List {
    list.into_inner().filter(|p| p.as_str() != ",").collect()
  } else if list_rule == Rule::Expression {
    let mut inner = list.into_inner();
    if let Some(first) = inner.next() {
      if first.as_rule() == Rule::List {
        first.into_inner().filter(|p| p.as_str() != ",").collect()
      } else {
        return Err(InterpreterError::EvaluationError(
          "Second operand of /@ must be a list".into(),
        ));
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "Second operand of /@ must be a list".into(),
      ));
    }
  } else {
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
            } else if v < 0.0 {
              -1.0
            } else {
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
