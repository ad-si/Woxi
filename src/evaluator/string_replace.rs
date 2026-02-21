#[allow(unused_imports)]
use super::*;

/// Replace pattern with replacement in an expression string
/// This handles symbolic replacement (e.g., replacing 'a' with 'x' in '{a, b}')
pub fn replace_in_expr(expr: &str, pattern: &str, replacement: &str) -> String {
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
pub fn replace_var_with_value(
  text: &str,
  var_name: &str,
  value: &str,
) -> String {
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

/// Apply ReplaceAll with multiple rules simultaneously.
/// For each subexpression, try each rule in order and use the first match.
pub fn apply_replace_all_direct_multi_rules(
  expr: &str,
  rules: &[(String, String)],
) -> Result<String, InterpreterError> {
  // For list expressions, apply rules to each element
  if expr.starts_with('{') && expr.ends_with('}') {
    let inner = &expr[1..expr.len() - 1];
    let elements = split_list_elements(inner);
    let mut results = Vec::new();

    for elem in elements {
      let elem = elem.trim();
      // Try matching the whole element against each rule (for Wolfram patterns etc.)
      let replaced = apply_first_matching_rule(elem, rules)?;
      if replaced != elem {
        // Whole element matched a rule — use the replacement (don't recurse into it)
        results.push(replaced);
      } else if elem.starts_with('{') || elem.contains('[') {
        // Subexpression — recurse into it to apply rules to sub-elements
        results.push(apply_replace_all_direct_multi_rules(elem, rules)?);
      } else {
        // Simple atom — no match
        results.push(elem.to_string());
      }
    }

    return Ok(format!("{{{}}}", results.join(", ")));
  }

  // For non-list expressions, first try matching the whole expression
  // against each rule (needed for Wolfram patterns like i_ /; cond)
  for (pattern, replacement) in rules {
    if let Some(wolfram_pattern) = parse_wolfram_pattern(pattern)
      && let Some(replaced) =
        apply_wolfram_pattern(expr, &wolfram_pattern, replacement)?
    {
      let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
      return Ok(evaluated);
    }
  }

  // Then apply literal symbol rules simultaneously in one pass
  let result = replace_in_expr_multi_rules(expr, rules);

  if result != *expr {
    evaluate_fullform(&result).or(Ok(result))
  } else {
    Ok(result)
  }
}

/// Try each rule in order on a single expression, return the result of
/// the first matching rule. If no rule matches, return the original.
/// This only matches the ELEMENT as a whole (exact or pattern match).
/// It does NOT do sub-expression replacement — the caller handles recursion.
pub fn apply_first_matching_rule(
  elem: &str,
  rules: &[(String, String)],
) -> Result<String, InterpreterError> {
  for (pattern, replacement) in rules {
    // Check if this is a Wolfram pattern
    if let Some(wolfram_pattern) = parse_wolfram_pattern(pattern) {
      if let Some(replaced) =
        apply_wolfram_pattern(elem, &wolfram_pattern, replacement)?
      {
        let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
        return Ok(evaluated);
      }
    } else if elem == pattern {
      // Exact match of whole element against literal pattern
      let evaluated =
        evaluate_fullform(replacement).unwrap_or(replacement.clone());
      return Ok(evaluated);
    }
  }
  // No rule matched
  Ok(elem.to_string())
}

/// Replace multiple patterns simultaneously in an expression, respecting word boundaries.
/// At each position, try each rule in order and use the first match.
pub fn replace_in_expr_multi_rules(
  expr: &str,
  rules: &[(String, String)],
) -> String {
  // Separate function-call patterns from symbol patterns
  let mut func_rules: Vec<(&str, &str)> = Vec::new();
  let mut symbol_rules: Vec<(&str, &str)> = Vec::new();

  for (pattern, replacement) in rules {
    if pattern.contains('[') && pattern.contains(']') {
      func_rules.push((pattern, replacement));
    } else {
      symbol_rules.push((pattern, replacement));
    }
  }

  // First apply function-call pattern rules (literal string replacement, first match wins)
  let mut current = expr.to_string();
  for (pattern, replacement) in &func_rules {
    let next = current.replace(pattern, replacement);
    if next != current {
      current = next;
      break; // First matching rule wins
    }
  }

  // Then apply symbol rules simultaneously in one pass
  if symbol_rules.is_empty() {
    return current;
  }

  let mut result = String::new();
  let chars: Vec<char> = current.chars().collect();
  let mut i = 0;

  while i < chars.len() {
    let mut matched = false;

    // Try each symbol rule at this position (first match wins)
    for (pattern, replacement) in &symbol_rules {
      let pat_chars: Vec<char> = pattern.chars().collect();
      if i + pat_chars.len() <= chars.len()
        && chars[i..i + pat_chars.len()] == pat_chars[..]
      {
        // Check word boundaries
        let prev_char = if i > 0 { Some(chars[i - 1]) } else { None };
        let next_char = if i + pat_chars.len() < chars.len() {
          Some(chars[i + pat_chars.len()])
        } else {
          None
        };

        let prev_ok =
          prev_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let next_ok =
          next_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');

        if prev_ok && next_ok {
          result.push_str(replacement);
          i += pat_chars.len();
          matched = true;
          break;
        }
      }
    }

    if !matched {
      result.push(chars[i]);
      i += 1;
    }
  }

  result
}

/// Split a list's inner content by commas, respecting nested structures
pub fn split_list_elements(inner: &str) -> Vec<String> {
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

/// Apply ReplaceRepeated with direct pattern and replacement strings
pub fn apply_replace_repeated_direct(
  expr: &str,
  pattern: &str,
  replacement: &str,
) -> Result<String, InterpreterError> {
  let mut current = expr.to_string();
  let max_iterations = 1000; // Prevent infinite loops

  for _ in 0..max_iterations {
    let next = replace_in_expr(&current, pattern, replacement);

    if next == current {
      // No more changes, we're done
      break;
    }

    // Re-evaluate to simplify (preserve FullForm for round-tripping)
    current = evaluate_fullform(&next).unwrap_or(next);
  }

  Ok(current)
}
