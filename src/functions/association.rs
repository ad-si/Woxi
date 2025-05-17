use pest::iterators::Pair;

use crate::{extract_string, InterpreterError, Rule};

/// Helper for extracting association from first argument
pub fn get_assoc_from_first_arg(
  args: &[Pair<Rule>],
) -> Result<Vec<(String, String)>, InterpreterError> {
  let p = &args[0];

  // NEW: recognise an identifier that is wrapped in an Expression
  if p.as_rule() == Rule::Expression {
    let mut inner = p.clone().into_inner();
    if let Some(first) = inner.next() {
      if first.as_rule() == Rule::Identifier && inner.next().is_none() {
        if let Some(crate::StoredValue::Association(v)) =
          crate::ENV.with(|e| e.borrow().get(first.as_str()).cloned())
        {
          return Ok(v);
        }
      }
    }
  }

  if p.as_rule() == Rule::Identifier {
    if let Some(crate::StoredValue::Association(v)) =
      crate::ENV.with(|e| e.borrow().get(p.as_str()).cloned())
    {
      return Ok(v);
    }
  }
  if p.as_rule() == Rule::Association {
    return Ok(crate::eval_association(p.clone())?.0);
  }
  // Try to evaluate as an expression and parse as association display
  if let Ok(val) = crate::evaluate_expression(p.clone()) {
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

/// Handle Keys[assoc] - returns a list of keys from an association
pub fn keys(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  let asso = get_assoc_from_first_arg(args_pairs)?;
  let keys: Vec<_> = asso.iter().map(|(k, _)| k.clone()).collect();
  Ok(format!("{{{}}}", keys.join(", ")))
}

/// Handle Values[assoc] - returns a list of values from an association
pub fn values(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  let asso = get_assoc_from_first_arg(args_pairs)?;
  let vals: Vec<_> = asso.iter().map(|(_, v)| v.clone()).collect();
  Ok(format!("{{{}}}", vals.join(", ")))
}

/// Handle KeyDropFrom[assoc, key] - returns a new association with the specified key dropped
pub fn key_drop_from(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyDropFrom expects exactly 2 arguments".into(),
    ));
  }
  let mut asso = get_assoc_from_first_arg(args_pairs)?;
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
  Ok(disp)
}
