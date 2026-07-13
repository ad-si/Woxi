#[allow(unused_imports)]
use super::*;

/// Replace pattern with replacement in an expression string
/// This handles symbolic replacement (e.g., replacing 'a' with 'x' in '{a, b}')
fn replace_in_expr(expr: &str, pattern: &str, replacement: &str) -> String {
  // For function call patterns like f[2], we need more sophisticated matching
  if pattern.contains('[') && pattern.contains(']') {
    // Function call pattern - do literal replacement
    expr.replace(pattern, replacement)
  } else {
    // Symbol replacement - need to be careful not to replace partial matches
    // Use word boundary-aware replacement
    let mut result = String::new();
    let mut chars = expr.chars().peekable();
    let pattern_chars: Vec<char> = pattern.chars().collect();

    while let Some(c) = chars.next() {
      // Check if we're at the start of a potential match
      let mut potential_match = vec![c];
      let mut iter_clone = chars.clone();

      // Build up potential match
      while potential_match.len() < pattern_chars.len() {
        if let Some(nc) = iter_clone.next() {
          potential_match.push(nc);
        } else {
          break;
        }
      }

      if potential_match == pattern_chars {
        // Check if this is a whole word match (not part of identifier)
        let prev_char = result.chars().last();
        let next_char = iter_clone.next();

        let prev_ok =
          prev_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let next_ok =
          next_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');

        if prev_ok && next_ok {
          // It's a whole word match, do the replacement
          result.push_str(replacement);
          // Consume the matched characters
          for _ in 1..pattern_chars.len() {
            chars.next();
          }
          continue;
        }
      }

      result.push(c);
    }

    result
  }
}

/// Check if a pattern is a Wolfram blank pattern (x_, x_?test, x_ /; cond)
pub fn parse_wolfram_pattern(pattern: &str) -> Option<WolframPattern> {
  let pattern = pattern.trim();

  // Check for conditional pattern: x_ /; condition
  if let Some(cond_idx) = pattern.find(" /; ") {
    let before_cond = &pattern[..cond_idx];
    let condition = &pattern[cond_idx + 4..];

    // The part before /; should be a simple blank pattern (x_)
    if let Some(underscore_idx) = before_cond.find('_') {
      let var_name = before_cond[..underscore_idx].trim().to_string();
      if !var_name.is_empty()
        && var_name.chars().all(|c| c.is_alphanumeric() || c == '$')
      {
        return Some(WolframPattern::Conditional {
          var_name,
          condition: condition.trim().to_string(),
        });
      }
    }
  }

  // Check for pattern test: x_?test
  if let Some(underscore_idx) = pattern.find('_') {
    let after_underscore = &pattern[underscore_idx + 1..];
    if after_underscore.starts_with('?') {
      let var_name = pattern[..underscore_idx].trim().to_string();
      let test_func = after_underscore[1..].trim().to_string();
      if !var_name.is_empty()
        && var_name.chars().all(|c| c.is_alphanumeric() || c == '$')
        && !test_func.is_empty()
      {
        return Some(WolframPattern::Test {
          var_name,
          test_func,
        });
      }
    }
  }

  // Check for head blank pattern: x_Head (e.g. x_Integer, x_String)
  // or simple blank pattern: x_
  if let Some(underscore_idx) = pattern.find('_') {
    let var_name = pattern[..underscore_idx].trim().to_string();
    let after_underscore = &pattern[underscore_idx + 1..];
    if !var_name.is_empty()
      && var_name.chars().all(|c| c.is_alphanumeric() || c == '$')
      && !pattern.contains(' ')
    {
      if after_underscore.is_empty() {
        return Some(WolframPattern::Blank { var_name });
      } else if after_underscore
        .chars()
        .all(|c| c.is_alphanumeric() || c == '$')
      {
        return Some(WolframPattern::HeadBlank {
          var_name,
          head: after_underscore.to_string(),
        });
      }
    }
  }

  None
}

/// Wolfram pattern types
pub enum WolframPattern {
  /// x_ - matches any single expression
  Blank { var_name: String },
  /// x_Head - matches if Head[x] == Head (e.g. x_Integer, x_String)
  HeadBlank { var_name: String, head: String },
  /// x_?test - matches if test[x] is True
  Test { var_name: String, test_func: String },
  /// x_ /; condition - matches if condition (with x substituted) is True
  Conditional { var_name: String, condition: String },
}

/// Replace a variable name with a value, respecting word boundaries
/// This avoids replacing 'i' inside strings like "Fizz"
fn replace_var_with_value(text: &str, var_name: &str, value: &str) -> String {
  let mut result = String::new();
  let var_chars: Vec<char> = var_name.chars().collect();
  let chars: Vec<char> = text.chars().collect();
  let mut i = 0;

  while i < chars.len() {
    // Check if we're inside a string literal
    if chars[i] == '"' {
      result.push(chars[i]);
      i += 1;
      // Copy everything until the closing quote
      while i < chars.len() && chars[i] != '"' {
        result.push(chars[i]);
        i += 1;
      }
      if i < chars.len() {
        result.push(chars[i]); // closing quote
        i += 1;
      }
      continue;
    }

    // Check if we're at the start of the variable name
    if i + var_chars.len() <= chars.len() {
      let potential_match: Vec<char> = chars[i..i + var_chars.len()].to_vec();
      if potential_match == var_chars {
        // Check word boundaries
        let prev_ok = i == 0 || {
          let prev = chars[i - 1];
          !prev.is_alphanumeric() && prev != '_' && prev != '$'
        };
        let next_ok = i + var_chars.len() >= chars.len() || {
          let next = chars[i + var_chars.len()];
          !next.is_alphanumeric() && next != '_' && next != '$'
        };

        if prev_ok && next_ok {
          result.push_str(value);
          i += var_chars.len();
          continue;
        }
      }
    }

    result.push(chars[i]);
    i += 1;
  }

  result
}

/// Apply a Wolfram pattern replacement to an expression
pub fn apply_wolfram_pattern(
  expr: &str,
  pattern: &WolframPattern,
  replacement: &str,
) -> Result<Option<String>, InterpreterError> {
  match pattern {
    WolframPattern::Blank { var_name } => {
      // x_ matches any expression - substitute var_name with expr in replacement
      let result = replace_var_with_value(replacement, var_name, expr);
      Ok(Some(result))
    }
    WolframPattern::HeadBlank { var_name, head } => {
      // x_Head matches if Head[expr] == head
      let expr_head = get_string_expr_head(expr);
      if expr_head == *head {
        let result = replace_var_with_value(replacement, var_name, expr);
        Ok(Some(result))
      } else {
        Ok(None)
      }
    }
    WolframPattern::Test {
      var_name,
      test_func,
    } => {
      // x_?test - check if test[expr] is True
      let test_expr = format!("{}[{}]", test_func, expr);
      match interpret(&test_expr) {
        Ok(result) if result == "True" => {
          let replaced = replace_var_with_value(replacement, var_name, expr);
          Ok(Some(replaced))
        }
        _ => Ok(None), // Test failed, no match
      }
    }
    WolframPattern::Conditional {
      var_name,
      condition,
    } => {
      // x_ /; condition - check if condition (with var substituted) is True
      let substituted_condition =
        replace_var_with_value(condition, var_name, expr);
      match interpret(&substituted_condition) {
        Ok(result) if result == "True" => {
          let replaced = replace_var_with_value(replacement, var_name, expr);
          Ok(Some(replaced))
        }
        _ => Ok(None), // Condition failed, no match
      }
    }
  }
}

/// Evaluate a FullForm string expression and return the result in FullForm.
/// Unlike `interpret()`, this preserves string quotes so that round-tripping
/// through string_to_expr works correctly.
pub fn evaluate_fullform(s: &str) -> Result<String, InterpreterError> {
  let expr = crate::syntax::string_to_expr(s)?;
  let result = evaluate_expr_to_expr(&expr)?;
  Ok(expr_to_string(&result))
}

/// Apply ReplaceAll with direct pattern and replacement strings
pub fn apply_replace_all_direct(
  expr: &str,
  pattern: &str,
  replacement: &str,
) -> Result<String, InterpreterError> {
  // Check if the pattern is a Wolfram pattern (x_, x_?test, x_ /; cond)
  if let Some(wolfram_pattern) = parse_wolfram_pattern(pattern) {
    // For list expressions, apply pattern to each element
    if expr.starts_with('{') && expr.ends_with('}') {
      let inner = &expr[1..expr.len() - 1];
      // Split by comma, being careful about nested structures
      let elements = split_list_elements(inner);
      let mut results = Vec::new();

      for elem in elements {
        let elem = elem.trim();
        if let Some(replaced) =
          apply_wolfram_pattern(elem, &wolfram_pattern, replacement)?
        {
          // Try to evaluate the replacement (preserve FullForm for round-tripping)
          let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
          results.push(evaluated);
        } else {
          // No match, keep original
          results.push(elem.to_string());
        }
      }

      return Ok(format!("{{{}}}", results.join(", ")));
    } else {
      // Single expression - apply pattern directly
      if let Some(replaced) =
        apply_wolfram_pattern(expr, &wolfram_pattern, replacement)?
      {
        let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
        return Ok(evaluated);
      }
      return Ok(expr.to_string());
    }
  }

  // Fall back to literal string replacement for non-pattern cases
  let result = replace_in_expr(expr, pattern, replacement);

  // Re-evaluate the result to simplify if possible
  if result != *expr {
    // Try to evaluate the result, but if it fails, return as-is
    evaluate_fullform(&result).or(Ok(result))
  } else {
    Ok(result)
  }
}

/// Split a list's inner content by commas, respecting nested structures
fn split_list_elements(inner: &str) -> Vec<String> {
  let mut elements = Vec::new();
  let mut current = String::new();
  let mut depth = 0;

  for c in inner.chars() {
    match c {
      '{' | '[' | '(' | '<' => {
        depth += 1;
        current.push(c);
      }
      '}' | ']' | ')' | '>' => {
        depth -= 1;
        current.push(c);
      }
      ',' if depth == 0 => {
        elements.push(current.trim().to_string());
        current.clear();
      }
      _ => {
        current.push(c);
      }
    }
  }

  if !current.is_empty() {
    elements.push(current.trim().to_string());
  }

  elements
}
