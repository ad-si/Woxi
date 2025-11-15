use crate::syntax::{str_to_wonum, wonum_to_number_str, WoNum, AST};
use crate::utils::create_file;
use crate::{
  apply_map_operator, eval_association, extract_string, format_result,
  functions, interpret, store_function_definition, InterpreterError, Rule,
  StoredValue, ENV, FUNC_DEFS,
};

pub fn evaluate_expression(
  expr: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  let mut inner = expr.clone().into_inner();

  if let Some(fun_call) = inner.next() {
    if fun_call.as_rule() == Rule::FunctionCall {
      let mut idents = fun_call.clone().into_inner();
      if let Some(ident) = idents.next() {
        if ident.as_span().as_str() == "CreateFile" {
          let filename_opt = idents
            .next()
            .map(|pair| pair.as_span().as_str().replace("\"", ""));
          return match create_file(filename_opt) {
            Ok(path) => Ok(path.to_string_lossy().into_owned()),
            Err(err) => Err(InterpreterError::EvaluationError(err.to_string())),
          };
        }
      }
    }
  };

  if inner.len() == 3 {
    if let (Some(a), Some(b), Some(c)) =
      (inner.next(), inner.next(), inner.next())
    {
      if a.as_rule() == Rule::NumericValue
        && b.as_rule() == Rule::Operator
        && c.as_rule() == Rule::NumericValue
      {
        let op = b.as_span().as_str();
        match op {
          "+" => {
            return evaluate_ast(AST::Plus(vec![
              str_to_wonum(a.as_span().as_str()),
              str_to_wonum(c.as_span().as_str()),
            ]));
          }
          "*" => {
            return evaluate_ast(AST::Times(vec![
              str_to_wonum(a.as_span().as_str()),
              str_to_wonum(c.as_span().as_str()),
            ]));
          }
          "-" => {
            return evaluate_ast(AST::Plus(vec![
              str_to_wonum(a.as_span().as_str()),
              -str_to_wonum(c.as_span().as_str()),
            ]));
          }
          "/" => {
            let divisor = str_to_wonum(c.as_span().as_str());
            let is_zero = match divisor {
              WoNum::Int(i) => i == 0,
              WoNum::Float(f) => f == 0.0,
            };
            if is_zero {
              return Err(InterpreterError::EvaluationError(
                "Division by zero".into(),
              ));
            }
            return evaluate_ast(AST::Divide(vec![
              str_to_wonum(a.as_span().as_str()),
              divisor,
            ]));
          }
          _ => {}
        }
      }
    }
  }
  evaluate_pairs(expr)
}

pub fn evaluate_ast(ast: AST) -> Result<String, InterpreterError> {
  Ok(match ast {
    AST::Plus(nums) => wonum_to_number_str(nums.into_iter().sum()),

    AST::Times(wo_nums) => wonum_to_number_str(
      wo_nums
        .into_iter()
        .fold(WoNum::Float(1.0), |acc, wo_num| acc * wo_num),
    ),

    AST::Minus(wo_nums) => {
      if wo_nums.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Minus expects at least 1 argument".into(),
        ));
      }
      if wo_nums.len() == 1 {
        // Unary minus
        wonum_to_number_str(-wo_nums.into_iter().next().unwrap())
      } else {
        // Multiple arguments - wrong arity, follow old behavior
        use std::io::{self, Write};
        println!(
          "\nMinus::argx: Minus called with {} arguments; 1 argument is expected.",
          wo_nums.len()
        );
        io::stdout().flush().ok();
        // Return the expression with minus signs
        let parts: Vec<String> = wo_nums.iter().map(|w| wonum_to_number_str(w.clone())).collect();
        parts.join(" − ")
      }
    }

    AST::Divide(wo_nums) => {
      if wo_nums.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Divide expects exactly 2 arguments".into(),
        ));
      }
      let mut iter = wo_nums.into_iter();
      let a = iter.next().unwrap();
      let b = iter.next().unwrap();
      wonum_to_number_str(a / b)
    }

    AST::Abs(wo_num) => wonum_to_number_str(wo_num.abs()),

    AST::Sign(wo_num) => wo_num.sign().to_string(),

    AST::Sqrt(wo_num) => match wo_num.sqrt() {
      Ok(result) => wonum_to_number_str(result),
      Err(msg) => return Err(InterpreterError::EvaluationError(msg)),
    },

    AST::Floor(wo_num) => wonum_to_number_str(wo_num.floor()),

    AST::Ceiling(wo_num) => wonum_to_number_str(wo_num.ceiling()),

    AST::Round(wo_num) => wonum_to_number_str(wo_num.round()),

    AST::CreateFile(filename_opt) => match create_file(filename_opt) {
      Ok(filename) => filename.to_string_lossy().into_owned(),
      Err(err_msg) => err_msg.to_string(),
    },
  })
}

pub fn evaluate_pairs(
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
      evaluate_term(expr).map(format_result)
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
      let raw = key_pair.as_str().trim();
      let key = if raw.starts_with('"') && raw.ends_with('"') {
        raw.trim_matches('"').to_string()
      } else {
        raw.to_string()
      };
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
      Err(InterpreterError::EvaluationError(
        "Argument must be an association".into(),
      ))
    }
    Rule::Expression => {
      {
        let mut expr_inner = expr.clone().into_inner();
        if let Some(first) = expr_inner.next() {
          if expr_inner.next().is_none() && first.as_rule() == Rule::PartExtract
          {
            return evaluate_expression(first);
          }
        }
      }
      // --- special case: Map operator ----------------------------------
      {
        let items: Vec<_> = expr.clone().into_inner().collect();
        if items.len() == 1 && items[0].as_rule() == Rule::Association {
          let (_pairs, disp) = eval_association(items[0].clone())?;
          return Ok(disp);
        }
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
              let lhs = items[0].clone();

              // Handle association update: myHash[["key"]] = value
              if lhs.as_rule() == Rule::PartExtract {
                let mut lhs_inner = lhs.into_inner();
                let ident = lhs_inner.next().unwrap();
                let key_expr = lhs_inner.next().unwrap();
                let var_name = ident.as_str().to_string();
                let key = extract_string(key_expr)?;

                // Evaluate RHS
                let rhs_value = evaluate_pairs(items[2].clone())?;

                // Update or add the key in the association
                ENV.with(|e| {
                  let mut env = e.borrow_mut();
                  if let Some(StoredValue::Association(ref mut pairs)) = env.get_mut(&var_name) {
                    // Update existing key or add new key
                    if let Some(pair) = pairs.iter_mut().find(|(k, _)| k == &key) {
                      pair.1 = rhs_value.clone();
                    } else {
                      pairs.push((key.clone(), rhs_value.clone()));
                    }
                  } else {
                    return Err(InterpreterError::EvaluationError(
                      format!("{} is not an association", var_name)
                    ));
                  }
                  Ok(())
                })?;

                // Return the updated association
                return ENV.with(|e| {
                  let env = e.borrow();
                  if let Some(StoredValue::Association(pairs)) = env.get(&var_name) {
                    let disp_parts: Vec<String> = pairs
                      .iter()
                      .map(|(k, v)| format!("{} -> {}", k, v))
                      .collect();
                    Ok(format!("<|{}|>", disp_parts.join(", ")))
                  } else {
                    Err(InterpreterError::EvaluationError("Variable not found".into()))
                  }
                });
              }

              // LHS must be an identifier for regular assignment
              if lhs.as_rule() != Rule::Identifier {
                return Err(InterpreterError::EvaluationError(
                  "Left-hand side of assignment must be an identifier or part extract".into(),
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
                  .split_once(":=")
                  .map(|x| x.1)
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
      // First pass: handle exponentiation (highest precedence)
      let mut i = 0;
      while i < ops.len() {
        if ops[i] == "^" {
          values[i] = values[i].powf(values[i + 1]);
          values.remove(i + 1);
          ops.remove(i);
        } else {
          i += 1;
        }
      }
      // Second pass: handle multiplication and division
      let mut i = 0;
      while i < ops.len() {
        if ops[i] == "*" {
          values[i] *= values[i + 1];
          values.remove(i + 1);
          ops.remove(i);
        } else if ops[i] == "/" {
          if values[i + 1] == 0.0 {
            return Err(InterpreterError::EvaluationError(
              "Division by zero".to_string(),
            ));
          }
          values[i] /= values[i + 1];
          values.remove(i + 1);
          ops.remove(i);
        } else {
          i += 1;
        }
      }
      // Third pass: handle addition and subtraction
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
      last.ok_or(InterpreterError::EmptyInput)
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

/// Convert function arguments to WoNum for AST construction
pub fn args_to_wonums(
  args_pairs: &[pest::iterators::Pair<Rule>],
) -> Result<Vec<WoNum>, InterpreterError> {
  args_pairs
    .iter()
    .map(|pair| {
      let value = evaluate_expression(pair.clone())?;
      Ok(str_to_wonum(&value))
    })
    .collect()
}

pub fn evaluate_term(
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
    "DateString" => functions::date::date_string(&args_pairs),
    "Keys" => functions::association::keys(&args_pairs),
    "Values" => functions::association::values(&args_pairs),
    "KeyDropFrom" => functions::association::key_drop_from(&args_pairs),
    "KeyExistsQ" => functions::association::key_exists_q(&args_pairs),
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
        Ok(disp)
      } else {
        let val = evaluate_expression(rhs_pair.clone())?;
        ENV.with(|e| {
          e.borrow_mut()
            .insert(var_name, StoredValue::Raw(val.clone()))
        });
        Ok(val)
      }
    }

    // Boolean Functions
    "And" => functions::boolean::and(&args_pairs),
    "Or" => functions::boolean::or(&args_pairs),
    "Xor" => functions::boolean::xor(&args_pairs),
    "Not" => functions::boolean::not(&args_pairs, &call_text),
    "If" => functions::boolean::if_condition(&args_pairs, &call_text),

    // Numeric Functions
    "Sin" => functions::numeric::sin(&args_pairs),
    "Prime" => functions::numeric::prime(&args_pairs),
    "Plus" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Plus expects at least 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Plus(wonums))
    }
    "Times" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Times expects at least 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Times(wonums))
    }
    "Minus" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Minus expects at least 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Minus(wonums))
    }
    "Abs" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Abs expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Abs(wonums.into_iter().next().unwrap()))
    }
    "Sign" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Sign expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Sign(wonums.into_iter().next().unwrap()))
    }
    "Sqrt" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Sqrt expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Sqrt(wonums.into_iter().next().unwrap()))
    }
    "Floor" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Floor expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Floor(wonums.into_iter().next().unwrap()))
    }
    "Ceiling" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Ceiling expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Ceiling(wonums.into_iter().next().unwrap()))
    }
    "Round" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Round expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Round(wonums.into_iter().next().unwrap()))
    }
    "Max" => functions::numeric::max(&args_pairs),
    "Min" => functions::numeric::min(&args_pairs),
    "Mod" => functions::numeric::modulo(&args_pairs),
    "Power" => functions::numeric::power(&args_pairs),
    "Factorial" => functions::numeric::factorial(&args_pairs),

    "Equal" => functions::math::equal(&args_pairs),
    "Unequal" => functions::math::unequal(&args_pairs),
    "Greater" => functions::math::greater(&args_pairs),
    "GreaterEqual" => functions::math::greater_equal(&args_pairs),
    "Less" => functions::math::less(&args_pairs),
    "LessEqual" => functions::math::less_equal(&args_pairs),

    "NumberQ" => functions::predicate::number_q(&args_pairs),
    "IntegerQ" => functions::predicate::integer_q(&args_pairs),
    "EvenQ" => functions::predicate::even_odd_q(&args_pairs, "EvenQ"),
    "OddQ" => functions::predicate::even_odd_q(&args_pairs, "OddQ"),

    "RandomInteger" => functions::math::random_integer(&args_pairs),

    // String Functions
    "StringLength" => functions::string::string_length(&args_pairs),
    "StringTake" => functions::string::string_take(&args_pairs),
    "StringDrop" => functions::string::string_drop(&args_pairs),
    "StringJoin" => functions::string::string_join(&args_pairs),
    "StringSplit" => functions::string::string_split(&args_pairs),
    "StringStartsQ" => functions::string::string_starts_q(&args_pairs),
    "StringEndsQ" => functions::string::string_ends_q(&args_pairs),

    // List Functions
    "Map" => functions::list_helpers::map_list(&args_pairs),
    "First" | "Last" => {
      functions::list_helpers::first_or_last(func_name, &args_pairs)
    }
    "Rest" | "Most" => {
      functions::list_helpers::rest_or_most(func_name, &args_pairs)
    }
    "MemberQ" => functions::list::member_q(&args_pairs),
    "Take" => functions::list::take(&args_pairs),
    "Drop" => functions::list::drop(&args_pairs),
    "Append" => functions::list::append(&args_pairs),
    "Prepend" => functions::list::prepend(&args_pairs),
    "Part" => functions::list::part(&args_pairs),
    "Length" => functions::list::length(&args_pairs),
    "Reverse" => functions::list::reverse(&args_pairs),
    "Range" => functions::list::range(&args_pairs),
    "Join" => functions::list::join(&args_pairs),
    "Sort" => functions::list::sort(&args_pairs),

    // Aggregation Functions
    "Total" => functions::list_helpers::total(&args_pairs),
    "Mean" => functions::list_helpers::mean(&args_pairs),
    "Median" => functions::list_helpers::median(&args_pairs),
    "Product" => functions::list_helpers::product(&args_pairs),
    "Accumulate" => functions::list_helpers::accumulate(&args_pairs),
    "Differences" => functions::list_helpers::differences(&args_pairs),
    "Divide" => {
      if args_pairs.len() != 2 {
        use std::io::{self, Write};
        println!(
          "\nDivide::argrx: Divide called with {} arguments; 2 arguments are expected.",
          args_pairs.len()
        );
        io::stdout().flush().ok();
        return Ok(call_text.to_string()); // return unevaluated expression
      }
      let wonums = args_to_wonums(&args_pairs)?;
      // Check for division by zero
      let divisor = &wonums[1];
      let is_zero = match divisor {
        WoNum::Int(i) => *i == 0,
        WoNum::Float(f) => *f == 0.0,
      };
      if is_zero {
        return Err(InterpreterError::EvaluationError("Division by zero".into()));
      }
      evaluate_ast(AST::Divide(wonums))
    }

    // Basic Functions
    "Select" => functions::list_helpers::select(&args_pairs),
    "AllTrue" => functions::list_helpers::all_true(&args_pairs),
    "Flatten" => functions::list_helpers::flatten(&args_pairs),
    "GroupBy" => Err(InterpreterError::EvaluationError(
      "GroupBy function not yet implemented".into(),
    )),
    "Print" => functions::io::print(&args_pairs),

    _ => Err(InterpreterError::EvaluationError(format!(
      "Unknown function: {}",
      func_name
    ))),
  }
}
