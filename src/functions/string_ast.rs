//! AST-native string functions.
//!
//! These functions work directly with `Expr` AST nodes, avoiding string round-trips.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

/// Helper to extract a string from an Expr
fn expr_to_str(expr: &Expr) -> Result<String, InterpreterError> {
  match expr {
    Expr::String(s) => Ok(s.clone()),
    Expr::Identifier(s) => Ok(s.clone()),
    Expr::Integer(n) => Ok(n.to_string()),
    Expr::BigInteger(n) => Ok(n.to_string()),
    Expr::Real(f) => Ok(crate::syntax::format_real(*f)),
    _ => {
      // Try to get string representation
      let s = crate::syntax::expr_to_string(expr);
      // If it's a quoted string, strip the quotes
      if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        Ok(s[1..s.len() - 1].to_string())
      } else {
        Ok(s)
      }
    }
  }
}

/// Helper to get integer from Expr
fn expr_to_int(expr: &Expr) -> Result<i128, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i128().ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Integer too large for this operation".into(),
        )
      })
    }
    Expr::Real(f) if f.fract() == 0.0 => Ok(*f as i128),
    _ => Err(InterpreterError::EvaluationError(
      "Expected integer argument".into(),
    )),
  }
}

/// StringLength[s] - returns the length of a string
pub fn string_length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StringLength expects exactly 1 argument".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  Ok(Expr::Integer(s.chars().count() as i128))
}

/// StringTake[s, n] - first n chars; StringTake[s, -n] - last n chars;
/// StringTake[s, {m, n}] - chars m through n; StringTake[s, {n}] - nth char
pub fn string_take_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringTake expects exactly 2 arguments".into(),
    ));
  }

  // StringTake[{s1, s2, ...}, spec] - map over list of strings
  if let Expr::List(strings) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = strings
      .iter()
      .map(|s| string_take_ast(&[s.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  let s = expr_to_str(&args[0])?;
  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;

  // StringTake[s, {spec1, spec2, ...}] where at least one spec is itself
  // a list - return a list, applying each sub-spec to the string.
  // (Scalar spec elements are dispatched individually as well.)
  if let Expr::List(elems) = &args[1]
    && elems.iter().any(|e| matches!(e, Expr::List(_)))
  {
    let results: Result<Vec<Expr>, InterpreterError> = elems
      .iter()
      .map(|e| string_take_ast(&[args[0].clone(), e.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  // StringTake[s, All] - return entire string
  if matches!(&args[1], Expr::Identifier(name) if name == "All") {
    return Ok(Expr::String(s));
  }

  match &args[1] {
    Expr::List(elems) if elems.len() == 1 => {
      // StringTake[s, {n}] - just the nth character
      let n = expr_to_int(&elems[0])?;
      let idx = if n > 0 { n - 1 } else { len + n };
      if idx < 0 || idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "StringTake index {} out of range for string of length {}",
          n, len
        )));
      }
      Ok(Expr::String(chars[idx as usize].to_string()))
    }
    Expr::List(elems) if elems.len() == 2 => {
      // StringTake[s, {m, n}] - characters m through n
      let m = expr_to_int(&elems[0])?;
      let n = expr_to_int(&elems[1])?;
      let start = if m > 0 { m - 1 } else { len + m };
      let end = if n > 0 { n - 1 } else { len + n };
      if start < 0 || end < 0 || start >= len || end >= len || start > end {
        return Err(InterpreterError::EvaluationError(format!(
          "StringTake range {{{}, {}}} out of range for string of length {}",
          m, n, len
        )));
      }
      let taken: String = chars[start as usize..=end as usize].iter().collect();
      Ok(Expr::String(taken))
    }
    Expr::List(elems) if elems.len() == 3 => {
      // StringTake[s, {m, n, step}] - characters m through n with step
      let m = expr_to_int(&elems[0])?;
      let n = expr_to_int(&elems[1])?;
      let step = expr_to_int(&elems[2])?;
      if step == 0 {
        return Err(InterpreterError::EvaluationError(
          "StringTake step cannot be 0".into(),
        ));
      }
      let start = if m > 0 { m - 1 } else { len + m };
      let end = if n > 0 { n - 1 } else { len + n };
      if start < 0 || end < 0 || start >= len || end >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "StringTake range {{{}, {}, {}}} out of range for string of length {}",
          m, n, step, len
        )));
      }
      let mut taken = String::new();
      let mut i = start;
      while (step > 0 && i <= end) || (step < 0 && i >= end) {
        if i >= 0 && i < len {
          taken.push(chars[i as usize]);
        }
        i += step;
      }
      Ok(Expr::String(taken))
    }
    // StringTake[s, UpTo[n]] - take up to n characters
    Expr::FunctionCall {
      name: up_name,
      args: up_args,
    } if up_name == "UpTo" && up_args.len() == 1 => {
      let max_n = expr_to_int(&up_args[0])?;
      let take_n = max_n.min(len) as usize;
      let taken: String = chars[..take_n].iter().collect();
      Ok(Expr::String(taken))
    }
    _ => {
      // StringTake[s, n] or StringTake[s, -n]
      let n = expr_to_int(&args[1])?;
      if n >= 0 {
        let take_n = n.min(len) as usize;
        let taken: String = chars[..take_n].iter().collect();
        Ok(Expr::String(taken))
      } else {
        let take_n = (-n).min(len) as usize;
        let taken: String = chars[len as usize - take_n..].iter().collect();
        Ok(Expr::String(taken))
      }
    }
  }
}

/// StringDrop[s, n] - drop first n chars; StringDrop[s, -n] - drop last n chars
/// StringDrop[s, {n}] - drop nth character; StringDrop[s, {m, n}] - drop chars m through n
pub fn string_drop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringDrop expects exactly 2 arguments".into(),
    ));
  }
  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0]
    && items.iter().all(|it| matches!(it, Expr::String(_)))
  {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| string_drop_ast(&[item.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }
  let s = expr_to_str(&args[0])?;
  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;

  match &args[1] {
    Expr::List(elems) if elems.len() == 1 => {
      // StringDrop[s, {n}] - drop the nth character
      let n = expr_to_int(&elems[0])?;
      let idx = if n > 0 { n - 1 } else { len + n };
      if idx < 0 || idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "StringDrop index {} out of range for string of length {}",
          n, len
        )));
      }
      let dropped: String = chars
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx as usize)
        .map(|(_, c)| c)
        .collect();
      Ok(Expr::String(dropped))
    }
    Expr::List(elems) if elems.len() == 2 => {
      // StringDrop[s, {m, n}] - drop characters m through n
      let m = expr_to_int(&elems[0])?;
      let n = expr_to_int(&elems[1])?;
      let start = if m > 0 { m - 1 } else { len + m };
      let end = if n > 0 { n - 1 } else { len + n };
      if start > end {
        // When start > end, nothing is dropped
        return Ok(Expr::String(s));
      }
      if start < 0 || end < 0 || start >= len || end >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "StringDrop range {{{}, {}}} out of range for string of length {}",
          m, n, len
        )));
      }
      let dropped: String = chars
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < start as usize || *i > end as usize)
        .map(|(_, c)| c)
        .collect();
      Ok(Expr::String(dropped))
    }
    _ => {
      // StringDrop[s, n] or StringDrop[s, -n]
      let n = expr_to_int(&args[1])?;
      if n >= 0 {
        let drop_n = n.min(len) as usize;
        let dropped: String = chars[drop_n..].iter().collect();
        Ok(Expr::String(dropped))
      } else {
        let drop_n = (-n).min(len) as usize;
        let dropped: String = chars[..len as usize - drop_n].iter().collect();
        Ok(Expr::String(dropped))
      }
    }
  }
}

/// StringJoin[s1, s2, ...] or StringJoin[{s1, s2, ...}] - concatenates strings
pub fn string_join_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "StringJoin expects at least 1 argument".into(),
    ));
  }

  let mut joined = String::new();

  // Check if single argument is a list
  if args.len() == 1
    && let Expr::List(items) = &args[0]
  {
    for item in items {
      joined.push_str(&expr_to_str(item)?);
    }
    return Ok(Expr::String(joined));
  }

  // Multiple arguments - join them all, flattening any lists
  fn collect_strings(
    expr: &Expr,
    out: &mut String,
  ) -> Result<(), InterpreterError> {
    match expr {
      Expr::List(items) => {
        for item in items {
          collect_strings(item, out)?;
        }
      }
      _ => out.push_str(&expr_to_str(expr)?),
    }
    Ok(())
  }
  for arg in args {
    collect_strings(arg, &mut joined)?;
  }
  Ok(Expr::String(joined))
}

/// Extract IgnoreCase option from rule arguments (args after the delimiter)
fn extract_ignore_case(options: &[Expr]) -> bool {
  for opt in options {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "IgnoreCase")
      && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True")
    {
      return true;
    }
  }
  false
}

/// Check if an Expr is a RegularExpression[pattern] and extract the pattern string
fn extract_regex_pattern(expr: &Expr) -> Option<String> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "RegularExpression" && args.len() == 1 =>
    {
      if let Expr::String(pat) = &args[0] {
        Some(pat.clone())
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Trim leading and trailing empty strings from a split result,
/// matching Wolfram's StringSplit behavior (interior empty strings are preserved).
fn trim_empty_strings(parts: Vec<Expr>) -> Vec<Expr> {
  let is_empty = |e: &Expr| matches!(e, Expr::String(s) if s.is_empty());
  let start = parts
    .iter()
    .position(|e| !is_empty(e))
    .unwrap_or(parts.len());
  let end = parts
    .iter()
    .rposition(|e| !is_empty(e))
    .map(|i| i + 1)
    .unwrap_or(0);
  if start >= end {
    return Vec::new();
  }
  parts[start..end].to_vec()
}

/// StringSplit[s] - splits by whitespace; StringSplit[s, delim] - splits by delimiter
/// StringSplit[s, RegularExpression[pat]] - splits by regex pattern
/// Options: IgnoreCase -> True/False
pub fn string_split_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "StringSplit expects at least 1 argument".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;

  // 1-argument form: split by whitespace, removing empty strings
  if args.len() == 1 {
    let parts: Vec<Expr> = s
      .split_whitespace()
      .map(|p| Expr::String(p.to_string()))
      .collect();
    return Ok(Expr::List(parts));
  }

  // Extract max parts (3rd arg if integer) and options from remaining args
  let (max_parts, option_start) = if args.len() >= 3 {
    if let Expr::Integer(n) = &args[2] {
      (Some(*n as usize), 3)
    } else {
      (None, 2)
    }
  } else {
    (None, 2)
  };
  let ignore_case = extract_ignore_case(&args[option_start..]);

  // Check if delimiter is a RegularExpression or string pattern
  let regex_from_pattern = if extract_regex_pattern(&args[1]).is_some() {
    extract_regex_pattern(&args[1])
  } else if !matches!(&args[1], Expr::String(_) | Expr::List(_)) {
    string_pattern_to_regex(&args[1])
  } else {
    None
  };
  if let Some(pat) = regex_from_pattern {
    let regex_pat = if ignore_case {
      format!("(?i){}", pat)
    } else {
      pat
    };
    let re = regex::Regex::new(&regex_pat).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Invalid regular expression: {}",
        e
      ))
    })?;
    let parts: Vec<Expr> = if let Some(n) = max_parts {
      re.splitn(&s, n)
        .map(|p| Expr::String(p.to_string()))
        .collect()
    } else {
      re.split(&s).map(|p| Expr::String(p.to_string())).collect()
    };
    return Ok(Expr::List(trim_empty_strings(parts)));
  }

  // Collect delimiters: either a single string or a list of strings
  let delims: Vec<String> = match &args[1] {
    Expr::List(items) => {
      let mut ds = Vec::new();
      for item in items {
        ds.push(expr_to_str(item)?);
      }
      ds
    }
    _ => vec![expr_to_str(&args[1])?],
  };

  let mut parts: Vec<Expr> = if delims.len() == 1 && delims[0].is_empty() {
    s.chars().map(|c| Expr::String(c.to_string())).collect()
  } else if delims.len() == 1 {
    if ignore_case {
      let re = regex::RegexBuilder::new(&regex::escape(&delims[0]))
        .case_insensitive(true)
        .build()
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("Regex error: {}", e))
        })?;
      let raw: Vec<Expr> = if let Some(n) = max_parts {
        re.splitn(&s, n)
          .map(|p| Expr::String(p.to_string()))
          .collect()
      } else {
        re.split(&s).map(|p| Expr::String(p.to_string())).collect()
      };
      trim_empty_strings(raw)
    } else {
      let raw: Vec<Expr> = if let Some(n) = max_parts {
        s.splitn(n, &delims[0])
          .map(|p| Expr::String(p.to_string()))
          .collect()
      } else {
        s.split(&delims[0])
          .map(|p| Expr::String(p.to_string()))
          .collect()
      };
      trim_empty_strings(raw)
    }
  } else {
    // Split by multiple delimiters: scan left to right, try longest delimiter match first
    let mut sorted_delims = delims.clone();
    sorted_delims.sort_by(|a, b| b.len().cmp(&a.len()));
    let mut result = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
      let remaining: String = chars[i..].iter().collect();
      let mut matched = false;
      for d in &sorted_delims {
        if !d.is_empty() && remaining.starts_with(d.as_str()) {
          result.push(Expr::String(current.clone()));
          current.clear();
          i += d.len();
          matched = true;
          break;
        }
      }
      if !matched {
        current.push(chars[i]);
        i += 1;
      }
    }
    result.push(Expr::String(current));
    result
  };
  // For multi-delimiter splits, apply max_parts by joining excess parts
  if let Some(n) = max_parts
    && parts.len() > n
  {
    let keep: Vec<Expr> = parts[..n - 1].to_vec();
    let rest: Vec<String> = parts[n - 1..]
      .iter()
      .map(|e| {
        if let Expr::String(s) = e {
          s.clone()
        } else {
          crate::syntax::expr_to_string(e)
        }
      })
      .collect();
    let joined = rest.join(&delims[0]);
    let mut result = keep;
    result.push(Expr::String(joined));
    parts = result;
  }
  Ok(Expr::List(parts))
}

/// StringStartsQ[s, prefix] - checks if string starts with prefix
pub fn string_starts_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringStartsQ expects 2 or 3 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let ignore_case = has_ignore_case_option(args);

  // Try regex-based pattern first
  if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    let full_pat = if ignore_case {
      format!("(?i)^(?:{})", regex_pat)
    } else {
      format!("^(?:{})", regex_pat)
    };
    let re = regex::Regex::new(&full_pat).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "True" } else { "False" }.to_string(),
    ));
  }

  let prefix = expr_to_str(&args[1])?;
  let result = if ignore_case {
    s.to_lowercase().starts_with(&prefix.to_lowercase())
  } else {
    s.starts_with(&prefix)
  };
  Ok(Expr::Identifier(
    if result { "True" } else { "False" }.to_string(),
  ))
}

/// StringEndsQ[s, suffix] - checks if string ends with suffix
pub fn string_ends_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringEndsQ expects 2 or 3 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let ignore_case = has_ignore_case_option(args);

  // Try regex-based pattern first
  if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    let full_pat = if ignore_case {
      format!("(?i)(?:{})$", regex_pat)
    } else {
      format!("(?:{})$", regex_pat)
    };
    let re = regex::Regex::new(&full_pat).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "True" } else { "False" }.to_string(),
    ));
  }

  let suffix = expr_to_str(&args[1])?;
  let result = if ignore_case {
    s.to_lowercase().ends_with(&suffix.to_lowercase())
  } else {
    s.ends_with(&suffix)
  };
  Ok(Expr::Identifier(
    if result { "True" } else { "False" }.to_string(),
  ))
}

/// StringContainsQ[s, sub] - checks if string contains substring
/// StringContainsQ[s, sub, IgnoreCase -> True] - case-insensitive
pub fn string_contains_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringContainsQ expects 2 or 3 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let ignore_case = has_ignore_case_option(args);

  // Try regex-based pattern first
  if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    let full_pat = if ignore_case {
      format!("(?i){}", regex_pat)
    } else {
      regex_pat
    };
    let re = regex::Regex::new(&full_pat).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "True" } else { "False" }.to_string(),
    ));
  }

  let sub = expr_to_str(&args[1])?;
  let result = if ignore_case {
    s.to_lowercase().contains(&sub.to_lowercase())
  } else {
    s.contains(&sub)
  };
  Ok(Expr::Identifier(
    if result { "True" } else { "False" }.to_string(),
  ))
}

/// Check if args contain IgnoreCase -> True option
fn has_ignore_case_option(args: &[Expr]) -> bool {
  for arg in args.iter().skip(2) {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && crate::syntax::expr_to_string(pattern) == "IgnoreCase"
      && crate::syntax::expr_to_string(replacement) == "True"
    {
      return true;
    }
  }
  false
}

/// StringReplace[s, pattern -> replacement] - replaces occurrences
/// StringReplace[s, {rule1, rule2, ...}] - applies multiple rules
pub fn string_replace_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringReplace expects 2 or 3 arguments".into(),
    ));
  }

  let max_replacements = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(n) => Some(*n as usize),
      _ => None,
    }
  } else {
    None
  };

  // Handle list of strings as first arg
  if let Expr::List(strings) = &args[0] {
    let mut results = Vec::new();
    for s_expr in strings {
      let mut new_args = vec![s_expr.clone()];
      new_args.extend_from_slice(&args[1..]);
      results.push(string_replace_ast(&new_args)?);
    }
    return Ok(Expr::List(results));
  }

  let s = expr_to_str(&args[0])?;

  /// A replacement rule: either a simple literal pattern or a regex pattern.
  enum ReplaceRule {
    Simple {
      pattern: String,
      replacement: String,
    },
    Regex {
      regex: regex::Regex,
      replacement: String,
    },
  }

  fn extract_rule(expr: &Expr) -> Result<ReplaceRule, InterpreterError> {
    let (pattern_expr, replacement_expr) = match expr {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "StringReplace: rules must be of the form pattern -> replacement"
            .into(),
        ));
      }
    };

    let replacement = expr_to_str(replacement_expr)?;

    // For simple string literals, use direct matching
    if let Expr::String(pat_str) = pattern_expr {
      return Ok(ReplaceRule::Simple {
        pattern: pat_str.clone(),
        replacement,
      });
    }

    // For complex patterns (Alternatives, StringExpression, etc.), use regex
    if let Some(regex_str) = string_pattern_to_regex(pattern_expr) {
      let re = regex::Regex::new(&regex_str).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "StringReplace: invalid pattern regex: {}",
          e
        ))
      })?;
      return Ok(ReplaceRule::Regex {
        regex: re,
        replacement,
      });
    }

    // Fallback: try expr_to_str for identifiers etc.
    if let Ok(pat_str) = expr_to_str(pattern_expr) {
      return Ok(ReplaceRule::Simple {
        pattern: pat_str,
        replacement,
      });
    }

    Err(InterpreterError::EvaluationError(
      "StringReplace: unsupported pattern type".into(),
    ))
  }

  // Collect all rules into a vec
  let rules: Vec<ReplaceRule> = match &args[1] {
    Expr::List(rule_list) => {
      let mut v = Vec::new();
      for rule in rule_list {
        v.push(extract_rule(rule)?);
      }
      v
    }
    rule => vec![extract_rule(rule)?],
  };

  // Scan-based replacement: scan left-to-right, at each position try each rule
  fn scan_replace(
    s: &str,
    rules: &[ReplaceRule],
    max: Option<usize>,
  ) -> String {
    let mut result = String::new();
    let mut count = 0usize;
    let mut i = 0;
    while i < s.len() {
      if max.is_some() && count >= max.unwrap() {
        result.push_str(&s[i..]);
        break;
      }
      let mut matched = false;
      for rule in rules {
        match rule {
          ReplaceRule::Simple {
            pattern,
            replacement,
          } => {
            if pattern.is_empty() {
              continue;
            }
            if s[i..].starts_with(pattern.as_str()) {
              result.push_str(replacement);
              i += pattern.len();
              count += 1;
              matched = true;
              break;
            }
          }
          ReplaceRule::Regex { regex, replacement } => {
            if let Some(m) = regex.find(&s[i..])
              && m.start() == 0
              && !m.as_str().is_empty()
            {
              result.push_str(replacement);
              i += m.len();
              count += 1;
              matched = true;
              break;
            }
          }
        }
      }
      if !matched {
        let ch = s[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
      }
    }
    result
  }

  Ok(Expr::String(scan_replace(&s, &rules, max_replacements)))
}

/// ToUpperCase[s] - converts string to uppercase
pub fn to_upper_case_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToUpperCase expects exactly 1 argument".into(),
    ));
  }
  // Thread over lists (including nested lists) like Wolfram does.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| to_upper_case_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }
  let s = expr_to_str(&args[0])?;
  Ok(Expr::String(s.to_uppercase()))
}

/// ToLowerCase[s] - converts string to lowercase
pub fn to_lower_case_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToLowerCase expects exactly 1 argument".into(),
    ));
  }
  // Thread over lists (including nested lists) like Wolfram does.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| to_lower_case_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }
  let s = expr_to_str(&args[0])?;
  Ok(Expr::String(s.to_lowercase()))
}

/// Characters[s] - converts string to list of characters
pub fn characters_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Characters expects exactly 1 argument".into(),
    ));
  }
  // Thread over lists like Wolfram does.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| characters_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }
  let s = expr_to_str(&args[0])?;
  let chars: Vec<Expr> =
    s.chars().map(|c| Expr::String(c.to_string())).collect();
  Ok(Expr::List(chars))
}

/// StringRiffle[list] or StringRiffle[list, sep] or StringRiffle[list, {left, sep, right}]
/// or StringRiffle[list, sep_outer, sep_inner, ...] for nested lists.
pub fn string_riffle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "StringRiffle expects at least 1 argument".into(),
    ));
  }

  if !matches!(&args[0], Expr::List(_)) {
    return Err(InterpreterError::EvaluationError(
      "StringRiffle: first argument must be a list".into(),
    ));
  }

  // Build the effective separator list. If none provided, use defaults
  // matching the nesting depth of the input: innermost " ", then "\n",
  // then "\n\n", "\n\n\n", ...
  let owned_default_seps: Vec<Expr>;
  let sep_slice: &[Expr] = if args.len() == 1 {
    let depth = list_depth(&args[0]);
    owned_default_seps = (0..depth)
      .rev()
      .map(|i| match i {
        0 => Expr::String(" ".to_string()),
        n => Expr::String("\n".repeat(n)),
      })
      .collect();
    &owned_default_seps
  } else {
    &args[1..]
  };

  let result = string_riffle_recursive(&args[0], sep_slice)?;
  Ok(Expr::String(result))
}

/// Returns the maximum nesting depth of Lists in `expr`.
/// A non-list has depth 0; `{a, b}` (atoms) has depth 1; `{{a}}` has depth 2.
fn list_depth(expr: &Expr) -> usize {
  match expr {
    Expr::List(items) => 1 + items.iter().map(list_depth).max().unwrap_or(0),
    _ => 0,
  }
}

/// Renders `expr` as a plain string, stripping quotes from string children.
/// Lists are rendered as `{a, b, c}` with elements separated by `", "`.
fn expr_to_plain_string(expr: &Expr) -> String {
  match expr {
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_plain_string).collect();
      format!("{{{}}}", parts.join(", "))
    }
    _ => {
      expr_to_str(expr).unwrap_or_else(|_| crate::syntax::expr_to_string(expr))
    }
  }
}

/// Recursively riffle `expr` using `seps` as the separator stack from outer
/// to inner. When `seps` runs out, non-list elements are stringified and
/// list elements are rendered with plain list notation.
fn string_riffle_recursive(
  expr: &Expr,
  seps: &[Expr],
) -> Result<String, InterpreterError> {
  match expr {
    Expr::List(items) if !seps.is_empty() => {
      // Parse the current separator: either a string or a {left, sep, right}
      // triple.
      let (left, sep, right) = match &seps[0] {
        Expr::List(triple) if triple.len() == 3 => (
          expr_to_str(&triple[0])?,
          expr_to_str(&triple[1])?,
          expr_to_str(&triple[2])?,
        ),
        other => (String::new(), expr_to_str(other)?, String::new()),
      };
      let rest = &seps[1..];
      let parts: Result<Vec<String>, _> = items
        .iter()
        .map(|it| string_riffle_recursive(it, rest))
        .collect();
      Ok(format!("{}{}{}", left, parts?.join(&sep), right))
    }
    _ => Ok(expr_to_plain_string(expr)),
  }
}

/// StringPosition[s, sub] - find all positions of substring
pub fn string_position_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringPosition expects 2 or 3 arguments".into(),
    ));
  }

  let s = expr_to_str(&args[0])?;
  let sub = expr_to_str(&args[1])?;
  let max_results = if args.len() == 3 {
    expr_to_int(&args[2]).ok().map(|n| n as usize)
  } else {
    None
  };

  if sub.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let mut positions = Vec::new();
  let s_chars: Vec<char> = s.chars().collect();
  let sub_chars: Vec<char> = sub.chars().collect();

  for i in 0..=s_chars.len().saturating_sub(sub_chars.len()) {
    if let Some(max) = max_results
      && positions.len() >= max
    {
      break;
    }
    let mut matched = true;
    for (j, &sub_char) in sub_chars.iter().enumerate() {
      if s_chars[i + j] != sub_char {
        matched = false;
        break;
      }
    }
    if matched {
      let start = (i + 1) as i128;
      let end = (i + sub_chars.len()) as i128;
      positions
        .push(Expr::List(vec![Expr::Integer(start), Expr::Integer(end)]));
    }
  }

  Ok(Expr::List(positions))
}

/// StringMatchQ[s, pattern] - test if string matches pattern
/// StringMatchQ[s, pattern, IgnoreCase -> True] - case-insensitive
pub fn string_match_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringMatchQ expects 2 or 3 arguments".into(),
    ));
  }

  let ignore_case = has_ignore_case_option(args);
  let s = if ignore_case {
    expr_to_str(&args[0])?.to_lowercase()
  } else {
    expr_to_str(&args[0])?
  };

  // Try pattern-based matching (DigitCharacter, LetterCharacter, Repeated, etc.)
  if let Some(regex_str) = string_pattern_to_regex(&args[1]) {
    let case_flag = if ignore_case { "(?i)" } else { "" };
    let full_regex = format!("{}^(?:{})$", case_flag, regex_str);
    let re = regex::Regex::new(&full_regex).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Invalid string pattern: {}",
        e
      ))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "True" } else { "False" }.to_string(),
    ));
  }

  // Try RegularExpression pattern
  if let Some(pat) = extract_regex_pattern(&args[1]) {
    let case_flag = if ignore_case { "(?i)" } else { "" };
    let full_regex = format!("{}^(?:{})$", case_flag, pat);
    let re = regex::Regex::new(&full_regex).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid regex: {}", e))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "True" } else { "False" }.to_string(),
    ));
  }

  // Fall back to string-based wildcard matching
  let pattern_str = if ignore_case {
    expr_to_str(&args[1])?.to_lowercase()
  } else {
    expr_to_str(&args[1])?
  };
  let matches = wildcard_match(&s, &pattern_str);
  Ok(Expr::Identifier(
    if matches { "True" } else { "False" }.to_string(),
  ))
}

/// Simple wildcard matching
fn wildcard_match(s: &str, pattern: &str) -> bool {
  let s_chars: Vec<char> = s.chars().collect();
  let p_chars: Vec<char> = pattern.chars().collect();

  fn is_non_uppercase(c: char) -> bool {
    !c.is_ascii_uppercase()
  }

  fn match_helper(s: &[char], p: &[char]) -> bool {
    match (s.is_empty(), p.is_empty()) {
      (true, true) => true,
      (_, true) => false,
      (true, false) => p.iter().all(|&c| c == '*'),
      (false, false) => match p[0] {
        '*' => match_helper(s, &p[1..]) || match_helper(&s[1..], p),
        '@' => {
          if !is_non_uppercase(s[0]) {
            return false;
          }
          match_helper(&s[1..], &p[1..]) || match_helper(&s[1..], p)
        }
        c => s[0] == c && match_helper(&s[1..], &p[1..]),
      },
    }
  }

  match_helper(&s_chars, &p_chars)
}

/// StringReverse[s] - reverse the string
pub fn string_reverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StringReverse expects exactly 1 argument".into(),
    ));
  }
  // Thread over lists (including nested lists) like Wolfram does.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| string_reverse_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }
  let s = expr_to_str(&args[0])?;
  Ok(Expr::String(s.chars().rev().collect()))
}

/// StringRepeat[s, n] - repeat string n times
pub fn string_repeat_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringRepeat expects 2 or 3 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringRepeat must be non-negative".into(),
    ));
  }
  let repeated = s.repeat(n as usize);
  if args.len() == 3 {
    // Third argument is max length
    let max_len = expr_to_int(&args[2])? as usize;
    let truncated: String = repeated.chars().take(max_len).collect();
    Ok(Expr::String(truncated))
  } else {
    Ok(Expr::String(repeated))
  }
}

/// StringTrim[s] or StringTrim[s, patt] - trim whitespace or pattern
pub fn string_trim_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "StringTrim expects 1 or 2 arguments".into(),
    ));
  }
  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut call = vec![item.clone()];
        call.extend(args[1..].iter().cloned());
        string_trim_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?));
  }
  let s = expr_to_str(&args[0])?;

  if args.len() == 1 {
    Ok(Expr::String(s.trim().to_string()))
  } else if let Expr::String(patt) = &args[1] {
    // Plain string literal: strip one occurrence from each end
    let trimmed = s.strip_prefix(patt.as_str()).unwrap_or(&s);
    let trimmed = trimmed.strip_suffix(patt.as_str()).unwrap_or(trimmed);
    Ok(Expr::String(trimmed.to_string()))
  } else if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    // String pattern (Repeated, Whitespace, etc.): use regex
    let start_re =
      regex::Regex::new(&format!("^(?:{})", regex_pat)).map_err(|e| {
        InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
      })?;
    let end_re =
      regex::Regex::new(&format!("(?:{})$", regex_pat)).map_err(|e| {
        InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
      })?;
    let trimmed = start_re.replace(&s, "");
    let trimmed = end_re.replace(&trimmed, "");
    Ok(Expr::String(trimmed.into_owned()))
  } else {
    // Unrecognized pattern: return unevaluated
    Ok(Expr::FunctionCall {
      name: "StringTrim".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Convert a Wolfram string pattern expression to a regex pattern string.
/// Returns None if the expression is not a recognized string pattern.
fn string_pattern_to_regex(expr: &Expr) -> Option<String> {
  match expr {
    // String literal patterns — convert Wolfram metacharacters (* and @)
    // to regex equivalents before escaping the rest.
    Expr::String(s) => {
      let mut result = String::new();
      for ch in s.chars() {
        match ch {
          '*' => result.push_str(".*"),
          '@' => result.push_str("[^A-Z]"),
          _ => result.push_str(&regex::escape(&ch.to_string())),
        }
      }
      Some(result)
    }

    // Character class patterns and blank patterns
    Expr::Identifier(name) => match name.as_str() {
      "DigitCharacter" => Some("[0-9]".to_string()),
      "LetterCharacter" => Some("[a-zA-Z\\p{L}]".to_string()),
      "WhitespaceCharacter" => Some("\\s".to_string()),
      "Whitespace" => Some("\\s+".to_string()),
      "WordCharacter" => Some("[a-zA-Z0-9\\p{L}]".to_string()),
      "HexadecimalCharacter" => Some("[0-9a-fA-F]".to_string()),
      "NumberString" => Some("[0-9]+(?:\\.[0-9]*)?".to_string()),
      "_" => Some(".".to_string()), // Blank: any single character
      "__" => Some(".+".to_string()), // BlankSequence: one or more characters
      "___" => Some(".*".to_string()), // BlankNullSequence: zero or more characters
      _ => None,
    },

    // Blank/BlankSequence/BlankNullSequence as Pattern AST nodes
    Expr::Pattern {
      name: _,
      head: None,
      blank_type,
    } => match blank_type {
      1 => Some(".".to_string()), // Blank: any single character
      2 => Some(".+".to_string()), // BlankSequence: one or more characters
      3 => Some(".*".to_string()), // BlankNullSequence: zero or more characters
      _ => None,
    },

    // Alternatives as BinaryOp (e.g. pat1 | pat2)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left,
      right,
    } => {
      let l = string_pattern_to_regex(left)?;
      let r = string_pattern_to_regex(right)?;
      Some(format!("(?:{}|{})", l, r))
    }

    // Alternatives[pat1, pat2, ...] = pat1 | pat2 | ...
    Expr::FunctionCall { name, args }
      if name == "Alternatives" && !args.is_empty() =>
    {
      let parts: Option<Vec<String>> =
        args.iter().map(string_pattern_to_regex).collect();
      parts.map(|ps| format!("(?:{})", ps.join("|")))
    }

    // A list {pat1, pat2, ...} in a string-pattern context is treated as
    // Alternatives[pat1, pat2, ...].
    Expr::List(items) if !items.is_empty() => {
      let parts: Option<Vec<String>> =
        items.iter().map(string_pattern_to_regex).collect();
      parts.map(|ps| format!("(?:{})", ps.join("|")))
    }

    // StringExpression[pat1, pat2, ...] = pat1 ~~ pat2 ~~ ...
    Expr::FunctionCall { name, args }
      if name == "StringExpression" && !args.is_empty() =>
    {
      let parts: Option<Vec<String>> =
        args.iter().map(string_pattern_to_regex).collect();
      parts.map(|ps| ps.join(""))
    }

    // Repeated[pat] = pat.. (one or more)
    Expr::FunctionCall { name, args }
      if name == "Repeated" && (args.len() == 1 || args.len() == 2) =>
    {
      let base = string_pattern_to_regex(&args[0])?;
      if args.len() == 1 {
        Some(format!("(?:{})+", base))
      } else {
        // Repeated[pat, n] means 1..n, Repeated[pat, {n}] means exactly n
        let quantifier = match &args[1] {
          Expr::Integer(n) => format!("{{1,{}}}", n),
          Expr::List(items) if items.len() == 1 => {
            if let Some(n) = crate::functions::math_ast::expr_to_i128(&items[0])
            {
              format!("{{{}}}", n)
            } else {
              return None;
            }
          }
          Expr::List(items) if items.len() == 2 => {
            let min = crate::functions::math_ast::expr_to_i128(&items[0])?;
            let max = crate::functions::math_ast::expr_to_i128(&items[1])?;
            format!("{{{},{}}}", min, max)
          }
          _ => return None,
        };
        Some(format!("(?:{}){}", base, quantifier))
      }
    }

    // RepeatedNull[pat] = pat... (zero or more)
    Expr::FunctionCall { name, args }
      if name == "RepeatedNull" && args.len() == 1 =>
    {
      string_pattern_to_regex(&args[0]).map(|r| format!("(?:{})*", r))
    }

    // RegularExpression["pattern"] - use the regex directly
    Expr::FunctionCall { name, args }
      if name == "RegularExpression" && args.len() == 1 =>
    {
      if let Expr::String(re_str) = &args[0] {
        Some(re_str.clone())
      } else {
        None
      }
    }

    _ => None,
  }
}

/// StringCases[s, patt] - find all substrings matching pattern
pub fn string_cases_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringCases expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;

  // Try pattern-based matching first
  if let Some(regex_str) = string_pattern_to_regex(&args[1]) {
    let re = regex::Regex::new(&regex_str).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Invalid string pattern: {}",
        e
      ))
    })?;
    let matches: Vec<Expr> = re
      .find_iter(&s)
      .map(|m| Expr::String(m.as_str().to_string()))
      .collect();
    return Ok(Expr::List(matches));
  }

  // Fall back to literal string matching
  let patt = expr_to_str(&args[1])?;

  let mut matches = Vec::new();
  let mut start = 0;
  while let Some(pos) = s[start..].find(&patt) {
    matches.push(Expr::String(patt.clone()));
    start = start + pos + patt.len();
  }

  Ok(Expr::List(matches))
}

/// ToString[expr] - convert expression to string
pub fn to_string_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ToString expects 1 or 2 arguments".into(),
    ));
  }

  // If the expression is MathMLForm[inner], produce MathML regardless of form argument
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "MathMLForm"
    && inner_args.len() == 1
  {
    let mathml = expr_to_mathml(&inner_args[0]);
    // InputForm includes trailing newline; default/OutputForm does not
    let is_input_form = args.len() == 2
      && matches!(&args[1], Expr::Identifier(f) if f == "InputForm");
    if is_input_form {
      return Ok(Expr::String(mathml));
    } else {
      return Ok(Expr::String(mathml.trim_end_matches('\n').to_string()));
    }
  }

  // If the expression is StandardForm[inner], produce box form representation
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "StandardForm"
    && inner_args.len() == 1
  {
    return Ok(Expr::String(expr_to_box_form(&inner_args[0])));
  }

  // If the expression is TraditionalForm[inner], produce \!\(\*FormBox[boxes, TraditionalForm]\)
  // using the same box escape notation as StandardForm but wrapped in FormBox.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "TraditionalForm"
    && inner_args.len() == 1
  {
    let box_str = expr_to_boxes(&inner_args[0]);
    // Use Unicode box markers (like StandardForm) so that:
    // - OutputForm renders as DisplayForm[FormBox[..., TraditionalForm]]
    // - InputForm renders as \!\(\*FormBox[..., TraditionalForm]\)
    return Ok(Expr::String(format!(
      "{}{}{}FormBox[{}, TraditionalForm]{}",
      BOX_START, BOX_OPEN, BOX_SEP, box_str, BOX_CLOSE
    )));
  }

  // If the expression is TeXForm[inner], produce TeX representation
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "TeXForm"
    && inner_args.len() == 1
  {
    return Ok(Expr::String(expr_to_tex(&inner_args[0])));
  }

  // If the expression is FortranForm[inner], produce Fortran representation
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "FortranForm"
    && inner_args.len() == 1
  {
    return Ok(Expr::String(expr_to_fortran(&inner_args[0])));
  }

  // If the expression is CForm[inner], produce C representation
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "CForm"
    && inner_args.len() == 1
  {
    return Ok(Expr::String(expr_to_c(&inner_args[0])));
  }

  // Check for form argument
  if args.len() == 2
    && let Expr::Identifier(form) = &args[1]
  {
    match form.as_str() {
      "InputForm" => {
        // If the expression is CForm[inner], produce C code instead
        if let Expr::FunctionCall {
          name,
          args: inner_args,
        } = &args[0]
          && name == "CForm"
          && inner_args.len() == 1
        {
          return Ok(Expr::String(expr_to_c(&inner_args[0])));
        }
        // InputForm: infix operators + quoted strings
        let s = crate::syntax::expr_to_input_form(&args[0]);
        return Ok(Expr::String(s));
      }
      "TeXForm" => {
        return Ok(Expr::String(expr_to_tex(&args[0])));
      }
      "CForm" => {
        return Ok(Expr::String(expr_to_c(&args[0])));
      }
      "FortranForm" => {
        return Ok(Expr::String(expr_to_fortran(&args[0])));
      }
      "OutputForm" => {
        let s = crate::syntax::expr_to_output_form_2d(&args[0]);
        return Ok(Expr::String(s));
      }
      _ => {}
    }
  }
  // Other forms: fall through to default (OutputForm-like) behavior

  // Special case: StringForm["template", args...] → substitute placeholders
  if let Expr::FunctionCall {
    name,
    args: sf_args,
  } = &args[0]
    && name == "StringForm"
    && !sf_args.is_empty()
    && let Expr::String(template) = &sf_args[0]
  {
    return Ok(Expr::String(format_string_form(template, &sf_args[1..])));
  }

  // Default (no form or unrecognized form): OutputForm-like
  // Resolve any nested form wrappers (TeXForm, CForm, FortranForm) first,
  // matching wolframscript behavior where ToString extracts form conversions.
  let resolved = resolve_form_wrappers(&args[0]);
  let s = crate::syntax::expr_to_output(&resolved);
  Ok(Expr::String(s))
}

/// Recursively replace TeXForm/CForm/FortranForm wrappers with their converted
/// string representation. This matches wolframscript behavior where ToString
/// extracts the converted form from nested wrappers.
fn resolve_form_wrappers(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => match name.as_str()
    {
      "TeXForm" => Expr::Identifier(expr_to_tex(&args[0])),
      "CForm" => Expr::Identifier(expr_to_c(&args[0])),
      "FortranForm" => Expr::Identifier(expr_to_fortran(&args[0])),
      _ => expr.clone(),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(resolve_form_wrappers).collect())
    }
    _ => expr.clone(),
  }
}

/// Convert a Wolfram expression to LaTeX (TeX) notation.
pub fn expr_to_tex(expr: &Expr) -> String {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::Real(f) => crate::syntax::format_real(*f),
    Expr::String(s) => format!("\\text{{{}}}", s),
    Expr::Identifier(name) | Expr::Constant(name) => tex_identifier(name),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => format!("-{}", expr_to_tex(operand)),
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      operand,
    } => format!("\\lnot {}", expr_to_tex(operand)),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let l = expr_to_tex(left);
      let r = expr_to_tex(right);
      // Check if right side starts with minus to avoid x+-y
      if r.starts_with('-') {
        format!("{}{}", l, r)
      } else {
        format!("{}+{}", l, r)
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => format!("{}-{}", expr_to_tex(left), expr_to_tex(right)),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => tex_times(left, right),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => format!("\\frac{{{}}}{{{}}}", expr_to_tex(left), expr_to_tex(right)),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => tex_power(left, right),
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => format!("{} \\land {}", expr_to_tex(left), expr_to_tex(right)),
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      left,
      right,
    } => format!("{} \\lor {}", expr_to_tex(left), expr_to_tex(right)),
    Expr::BinaryOp {
      op: BinaryOperator::StringJoin,
      left,
      right,
    } => format!("{} \\diamond {}", expr_to_tex(left), expr_to_tex(right)),
    Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left,
      right,
    } => format!("{} | {}", expr_to_tex(left), expr_to_tex(right)),
    Expr::Comparison {
      operands,
      operators,
    } => {
      use crate::syntax::ComparisonOp;
      let mut result = expr_to_tex(&operands[0]);
      for (i, op) in operators.iter().enumerate() {
        let op_tex = match op {
          ComparisonOp::Equal => "=",
          ComparisonOp::NotEqual => "\\neq ",
          ComparisonOp::Less => "<",
          ComparisonOp::LessEqual => "\\leq ",
          ComparisonOp::Greater => ">",
          ComparisonOp::GreaterEqual => "\\geq ",
          ComparisonOp::SameQ => "===",
          ComparisonOp::UnsameQ => "=!=",
        };
        result.push_str(op_tex);
        if i + 1 < operands.len() {
          result.push_str(&expr_to_tex(&operands[i + 1]));
        }
      }
      result
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_tex).collect();
      format!("\\{{{}\\}}", parts.join(","))
    }
    Expr::FunctionCall { name, args } => tex_function_call(name, args),
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!("{} \\to {}", expr_to_tex(pattern), expr_to_tex(replacement))
    }
    _ => crate::syntax::expr_to_output(expr),
  }
}

/// Convert an identifier to its TeX representation.
fn tex_identifier(name: &str) -> String {
  match name {
    "Pi" => "\\pi".to_string(),
    "E" => "e".to_string(),
    "I" => "i".to_string(),
    "Infinity" => "\\infty".to_string(),
    "True" => "\\text{True}".to_string(),
    "False" => "\\text{False}".to_string(),
    // Single letter identifiers stay as-is
    s if s.len() == 1 => s.to_string(),
    // Greek letters
    "Alpha" | "alpha" => "\\alpha".to_string(),
    "Beta" | "beta" => "\\beta".to_string(),
    "Gamma" | "gamma" => "\\gamma".to_string(),
    "Delta" | "delta" => "\\delta".to_string(),
    "Epsilon" | "epsilon" => "\\epsilon".to_string(),
    "Zeta" | "zeta" => "\\zeta".to_string(),
    "Eta" | "eta" => "\\eta".to_string(),
    "Theta" | "theta" => "\\theta".to_string(),
    "Iota" | "iota" => "\\iota".to_string(),
    "Kappa" | "kappa" => "\\kappa".to_string(),
    "Lambda" | "lambda" => "\\lambda".to_string(),
    "Mu" | "mu" => "\\mu".to_string(),
    "Nu" | "nu" => "\\nu".to_string(),
    "Xi" | "xi" => "\\xi".to_string(),
    "Omicron" | "omicron" => "o".to_string(),
    "Rho" | "rho" => "\\rho".to_string(),
    "Sigma" | "sigma" => "\\sigma".to_string(),
    "Tau" | "tau" => "\\tau".to_string(),
    "Upsilon" | "upsilon" => "\\upsilon".to_string(),
    "Phi" | "phi" => "\\phi".to_string(),
    "Chi" | "chi" => "\\chi".to_string(),
    "Psi" | "psi" => "\\psi".to_string(),
    "Omega" | "omega" => "\\omega".to_string(),
    // Multi-letter identifiers get \text{}
    s => format!("\\text{{{}}}", s),
  }
}

/// Handle binary multiplication in TeX (space-separated).
/// Delegates to `tex_times_nary` for fraction handling.
fn tex_times(left: &Expr, right: &Expr) -> String {
  tex_times_nary(&[left.clone(), right.clone()])
}

/// Check if an expression is Power[base, negative_integer]
/// and return (base, positive_exponent) if so.
fn as_neg_int_power(expr: &Expr) -> Option<(&Expr, i128)> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Integer(n) = &args[1]
        && *n < 0
      {
        return Some((&args[0], -n));
      }
      None
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Integer(n) = right.as_ref()
        && *n < 0
      {
        return Some((left.as_ref(), -n));
      }
      // Also handle UnaryOp Minus wrapping an integer
      if let Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } = right.as_ref()
        && let Expr::Integer(n) = operand.as_ref()
        && *n > 0
      {
        return Some((left.as_ref(), *n));
      }
      None
    }
    _ => None,
  }
}

/// Render a denominator factor: base^exp with exp omitted if 1.
/// Uses plain parens `(...)` for compound bases (matching wolframscript),
/// and only wraps in parens when exponent > 1.
fn tex_denom_factor(base: &Expr, pos_exp: i128) -> String {
  if pos_exp == 1 {
    // No parens needed in denominator when exponent is 1
    expr_to_tex(base)
  } else {
    let exp_str = pos_exp.to_string();
    if exp_str.len() == 1 {
      format!("{}^{}", tex_parens_plain(base), exp_str)
    } else {
      format!("{}^{{{}}}", tex_parens_plain(base), exp_str)
    }
  }
}

/// Wrap compound expressions in plain parens `(...)` for use in
/// denominators (matching wolframscript style, not `\left(\right)`).
fn tex_parens_plain(expr: &Expr) -> String {
  let needs_parens = match expr {
    Expr::BinaryOp { .. } | Expr::UnaryOp { .. } => true,
    Expr::FunctionCall { name, args } if args.len() >= 2 => {
      matches!(name.as_str(), "Plus" | "Times")
    }
    _ => false,
  };
  if needs_parens {
    format!("({})", expr_to_tex(expr))
  } else {
    expr_to_tex(expr)
  }
}

/// Handle n-ary Times in TeX, splitting into \frac when negative-integer
/// Power factors are present (matching Wolfram TeXForm).
fn tex_times_nary(args: &[Expr]) -> String {
  // Check for -1 leading factor
  let (negate, factors) = if matches!(&args[0], Expr::Integer(-1)) {
    (true, &args[1..])
  } else {
    (false, args)
  };

  // Partition into numerator factors and denominator factors
  let mut numer: Vec<String> = Vec::new();
  let mut denom: Vec<String> = Vec::new();
  for arg in factors {
    if let Some((base, pos_exp)) = as_neg_int_power(arg) {
      denom.push(tex_denom_factor(base, pos_exp));
    } else {
      numer.push(expr_to_tex(arg));
    }
  }

  let sign = if negate { "-" } else { "" };

  if denom.is_empty() {
    // No denominator factors — simple product
    let body = numer.join(" ");
    format!("{}{}", sign, body)
  } else {
    let numer_tex = if numer.is_empty() {
      "1".to_string()
    } else {
      numer.join(" ")
    };
    let denom_tex = denom.join(" ");
    format!("{}\\frac{{{}}}{{{}}}", sign, numer_tex, denom_tex)
  }
}

/// Handle power expressions in TeX
fn tex_power(base: &Expr, exp: &Expr) -> String {
  // Special case: Power[x, 1/2] → \sqrt{x}
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left,
    right,
  } = exp
    && matches!(left.as_ref(), Expr::Integer(1))
    && matches!(right.as_ref(), Expr::Integer(2))
  {
    return format!("\\sqrt{{{}}}", expr_to_tex(base));
  }
  // Power[x, Rational[1, 2]] → \sqrt{x}
  if let Expr::FunctionCall { name, args } = exp
    && name == "Rational"
    && args.len() == 2
  {
    if matches!(&args[0], Expr::Integer(1))
      && matches!(&args[1], Expr::Integer(2))
    {
      return format!("\\sqrt{{{}}}", expr_to_tex(base));
    }
    if matches!(&args[0], Expr::Integer(1)) {
      return format!(
        "\\sqrt[{}]{{{}}}",
        expr_to_tex(&args[1]),
        expr_to_tex(base)
      );
    }
  }

  // Negative integer exponent: Power[x, -n] → \frac{1}{x^n}
  if let Expr::Integer(n) = exp
    && *n < 0
  {
    return format!("\\frac{{1}}{{{}}}", tex_denom_factor(base, -n));
  }

  let base_tex = tex_base_with_parens(base);
  let exp_tex = expr_to_tex(exp);

  // Single-character exponents don't need braces in Wolfram TeXForm
  if exp_tex.chars().count() == 1 {
    format!("{}^{}", base_tex, exp_tex)
  } else {
    format!("{}^{{{}}}", base_tex, exp_tex)
  }
}

/// Wrap base in parens if needed for power
fn tex_base_with_parens(base: &Expr) -> String {
  let needs_parens = match base {
    Expr::BinaryOp { .. } | Expr::UnaryOp { .. } => true,
    Expr::FunctionCall { name, args } if args.len() >= 2 => {
      matches!(name.as_str(), "Plus" | "Times")
    }
    _ => false,
  };
  if needs_parens {
    format!("\\left({}\\right)", expr_to_tex(base))
  } else {
    expr_to_tex(base)
  }
}

/// Handle function calls in TeX
fn tex_function_call(name: &str, args: &[Expr]) -> String {
  match name {
    // Trig functions
    "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" if args.len() == 1 => {
      let fn_tex = format!("\\{}", name.to_lowercase());
      format!("{} ({})", fn_tex, expr_to_tex(&args[0]))
    }
    // Inverse trig
    "ArcSin" if args.len() == 1 => {
      format!("\\sin ^{{-1}}({})", expr_to_tex(&args[0]))
    }
    "ArcCos" if args.len() == 1 => {
      format!("\\cos ^{{-1}}({})", expr_to_tex(&args[0]))
    }
    "ArcTan" if args.len() == 1 => {
      format!("\\tan ^{{-1}}({})", expr_to_tex(&args[0]))
    }
    // Log
    "Log" if args.len() == 1 => {
      format!("\\log ({})", expr_to_tex(&args[0]))
    }
    "Log" if args.len() == 2 => {
      format!(
        "\\log _{{{}}}({})",
        expr_to_tex(&args[0]),
        expr_to_tex(&args[1])
      )
    }
    // Sqrt
    "Sqrt" if args.len() == 1 => {
      format!("\\sqrt{{{}}}", expr_to_tex(&args[0]))
    }
    // Abs
    "Abs" if args.len() == 1 => {
      format!("| {}|", expr_to_tex(&args[0]))
    }
    // Rational
    "Rational" if args.len() == 2 => {
      format!(
        "\\frac{{{}}}{{{}}}",
        expr_to_tex(&args[0]),
        expr_to_tex(&args[1])
      )
    }
    // Plus (n-ary) — use Wolfram canonical order but move pure numeric
    // constants to the end (matching Wolfram's TeXForm convention),
    // then rotate so a non-negative term leads (avoid starting with -z+x, prefer x-z)
    "Plus" if !args.is_empty() => {
      // Partition into symbolic and numeric terms, keeping relative order
      let mut symbolic: Vec<&Expr> = Vec::new();
      let mut numeric: Vec<&Expr> = Vec::new();
      for arg in args {
        if matches!(arg, Expr::Integer(_) | Expr::Real(_)) {
          numeric.push(arg);
        } else {
          symbolic.push(arg);
        }
      }
      let reordered_args: Vec<&Expr> =
        symbolic.into_iter().chain(numeric).collect();
      let tex_strs: Vec<String> =
        reordered_args.iter().map(|a| expr_to_tex(a)).collect();
      // Find first non-negative term to lead with
      let lead = tex_strs
        .iter()
        .position(|s| !s.starts_with('-'))
        .unwrap_or(0);
      let reordered: Vec<&str> = tex_strs[lead..]
        .iter()
        .chain(tex_strs[..lead].iter())
        .map(|s| s.as_str())
        .collect();
      let mut result = reordered[0].to_string();
      for t in reordered.iter().skip(1) {
        if t.starts_with('-') {
          result.push_str(t);
        } else {
          result.push('+');
          result.push_str(t);
        }
      }
      result
    }
    // Times (n-ary)
    "Times" if args.len() >= 2 => tex_times_nary(args),
    // Power
    "Power" if args.len() == 2 => tex_power(&args[0], &args[1]),
    // Exp
    "Exp" if args.len() == 1 => {
      format!("e^{{{}}}", expr_to_tex(&args[0]))
    }
    // Factorial
    "Factorial" if args.len() == 1 => {
      format!("{}!", expr_to_tex(&args[0]))
    }
    // Sum
    "Sum" if args.len() == 2 => {
      if let Expr::List(bounds) = &args[1]
        && bounds.len() >= 3
      {
        return format!(
          "\\sum _{{{}={}}}^{{{}}} {}",
          expr_to_tex(&bounds[0]),
          expr_to_tex(&bounds[1]),
          expr_to_tex(&bounds[2]),
          expr_to_tex(&args[0])
        );
      }
      format!(
        "\\text{{Sum}}({})",
        args.iter().map(expr_to_tex).collect::<Vec<_>>().join(",")
      )
    }
    // Product
    "Product" if args.len() == 2 => {
      if let Expr::List(bounds) = &args[1]
        && bounds.len() >= 3
      {
        return format!(
          "\\prod _{{{}={}}}^{{{}}} {}",
          expr_to_tex(&bounds[0]),
          expr_to_tex(&bounds[1]),
          expr_to_tex(&bounds[2]),
          expr_to_tex(&args[0])
        );
      }
      format!(
        "\\text{{Product}}({})",
        args.iter().map(expr_to_tex).collect::<Vec<_>>().join(",")
      )
    }
    // Integrate
    "Integrate" if args.len() == 2 => {
      if let Expr::List(bounds) = &args[1]
        && bounds.len() == 3
      {
        return format!(
          "\\int_{{{}}}^{{{}}} {} \\, d{}",
          expr_to_tex(&bounds[1]),
          expr_to_tex(&bounds[2]),
          expr_to_tex(&args[0]),
          expr_to_tex(&bounds[0])
        );
      }
      // Indefinite integral
      format!(
        "\\int {} \\, d{}",
        expr_to_tex(&args[0]),
        expr_to_tex(&args[1])
      )
    }
    // Derivative
    "D" if args.len() == 2 => {
      format!(
        "\\frac{{\\partial {}}}{{\\partial {}}}",
        expr_to_tex(&args[0]),
        expr_to_tex(&args[1])
      )
    }
    // Limit
    "Limit" if args.len() == 2 => {
      if let Expr::Rule {
        pattern,
        replacement,
      } = &args[1]
      {
        return format!(
          "\\lim_{{{} \\to {}}} {}",
          expr_to_tex(pattern),
          expr_to_tex(replacement),
          expr_to_tex(&args[0])
        );
      }
      format!(
        "\\text{{Limit}}({}, {})",
        expr_to_tex(&args[0]),
        expr_to_tex(&args[1])
      )
    }
    // MatrixForm
    "MatrixForm" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let mut lines = Vec::new();
        for row in rows {
          if let Expr::List(cols) = row {
            let cells: Vec<String> = cols.iter().map(expr_to_tex).collect();
            lines.push(format!(" {} \\\\", cells.join(" & ")));
          }
        }
        let ncols = if let Some(Expr::List(first_row)) = rows.first() {
          first_row.len()
        } else {
          1
        };
        let col_spec: String =
          std::iter::repeat_n("c", ncols).collect::<Vec<_>>().join("");
        format!(
          "\\left(\n\\begin{{array}}{{{}}}\n{}\\end{{array}}\n\\right)",
          col_spec,
          lines.join("\n") + "\n"
        )
      } else {
        format!("\\text{{MatrixForm}}({})", expr_to_tex(&args[0]))
      }
    }
    // Complex
    "Complex" if args.len() == 2 => {
      let re = expr_to_tex(&args[0]);
      let im = expr_to_tex(&args[1]);
      format!("{}+{} i", re, im)
    }
    // Default: render as text function name with parenthesized args
    _ => {
      let args_tex: Vec<String> = args.iter().map(expr_to_tex).collect();
      format!("\\text{{{}}}({})", name, args_tex.join(","))
    }
  }
}

// ====================================================================
// MathML generation
// ====================================================================

/// Convert a Wolfram expression to MathML (presentation MathML).
/// Returns the complete `<math>...</math>` block with proper indentation.
pub fn expr_to_mathml(expr: &Expr) -> String {
  let inner = mathml_inner(expr, 1);
  format!("<math>\n{}\n</math>\n", inner)
}

/// Render a single expression as a MathML fragment at the given indentation depth.
fn mathml_inner(expr: &Expr, depth: usize) -> String {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  let indent = " ".repeat(depth);

  match expr {
    // Numbers
    Expr::Integer(n) => format!("{}<mn>{}</mn>", indent, n),
    Expr::BigInteger(n) => format!("{}<mn>{}</mn>", indent, n),
    Expr::Real(f) => {
      format!("{}<mn>{}</mn>", indent, crate::syntax::format_real(*f))
    }

    // Strings
    Expr::String(s) => format!("{}<ms>{}</ms>", indent, mathml_escape(s)),

    // Identifiers and constants
    Expr::Identifier(name) | Expr::Constant(name) => {
      format!("{}<mi>{}</mi>", indent, mathml_identifier(name))
    }

    // Unary minus
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let inner = mathml_inner(operand, depth + 1);
      format!(
        "{}<mrow>\n{}<mo>-</mo>\n{}\n{}</mrow>",
        indent,
        " ".repeat(depth + 1),
        inner,
        indent
      )
    }

    // Binary operators
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus => {
        let l = mathml_inner(left, depth + 1);
        let r = mathml_inner(right, depth + 1);
        format!(
          "{}<mrow>\n{}\n{}<mo>+</mo>\n{}\n{}</mrow>",
          indent,
          l,
          " ".repeat(depth + 1),
          r,
          indent
        )
      }
      BinaryOperator::Minus => {
        let l = mathml_inner(left, depth + 1);
        let r = mathml_inner(right, depth + 1);
        format!(
          "{}<mrow>\n{}\n{}<mo>-</mo>\n{}\n{}</mrow>",
          indent,
          l,
          " ".repeat(depth + 1),
          r,
          indent
        )
      }
      BinaryOperator::Times => {
        let l = mathml_inner(left, depth + 1);
        let r = mathml_inner(right, depth + 1);
        format!(
          "{}<mrow>\n{}\n{}<mo>&#8290;</mo>\n{}\n{}</mrow>",
          indent,
          l,
          " ".repeat(depth + 1),
          r,
          indent
        )
      }
      BinaryOperator::Divide => {
        let l = mathml_inner(left, depth + 1);
        let r = mathml_inner(right, depth + 1);
        format!("{}<mfrac>\n{}\n{}\n{}</mfrac>", indent, l, r, indent)
      }
      BinaryOperator::Power => {
        if let Some(sqrt_arg) = crate::functions::is_sqrt(expr) {
          let b = mathml_inner(sqrt_arg, depth + 1);
          return format!("{}<msqrt>\n{}\n{}</msqrt>", indent, b, indent);
        }
        let b = mathml_inner(left, depth + 1);
        let e = mathml_inner(right, depth + 1);
        format!("{}<msup>\n{}\n{}\n{}</msup>", indent, b, e, indent)
      }
      _ => {
        // Fallback: render as operator
        let op_str = format!("{:?}", op);
        let l = mathml_inner(left, depth + 1);
        let r = mathml_inner(right, depth + 1);
        format!(
          "{}<mrow>\n{}\n{}<mo>{}</mo>\n{}\n{}</mrow>",
          indent,
          l,
          " ".repeat(depth + 1),
          op_str,
          r,
          indent
        )
      }
    },

    // Lists
    Expr::List(items) => mathml_list(items, depth),

    // Function calls
    Expr::FunctionCall { name, args } => {
      mathml_function_call(name, args, depth)
    }

    // Fallback: render via output form
    _ => {
      let s = crate::syntax::expr_to_output(expr);
      format!("{}<mi>{}</mi>", indent, mathml_escape(&s))
    }
  }
}

fn mathml_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
}

fn mathml_identifier(name: &str) -> &str {
  match name {
    "Pi" => "&#960;",
    "E" => "&#8519;",
    "I" => "&#8520;",
    "Infinity" | "DirectedInfinity" => "&#8734;",
    _ => name,
  }
}

fn mathml_list(items: &[Expr], depth: usize) -> String {
  let indent = " ".repeat(depth);
  let inner_indent = " ".repeat(depth + 1);
  let inner2_indent = " ".repeat(depth + 2);

  // Build comma-separated items inside {  }
  let mut parts = Vec::new();
  for (i, item) in items.iter().enumerate() {
    parts.push(mathml_inner(item, depth + 2));
    if i + 1 < items.len() {
      parts.push(format!("{}<mo>,</mo>", inner2_indent));
    }
  }

  format!(
    "{}<mrow>\n{}<mo>{{</mo>\n{}<mrow>\n{}\n{}</mrow>\n{}<mo>}}</mo>\n{}</mrow>",
    indent,
    inner_indent,
    inner_indent,
    parts.join("\n"),
    inner_indent,
    inner_indent,
    indent
  )
}

fn mathml_function_call(name: &str, args: &[Expr], depth: usize) -> String {
  let indent = " ".repeat(depth);
  let inner = " ".repeat(depth + 1);

  match name {
    // Rational
    "Rational" if args.len() == 2 => {
      let n = mathml_inner(&args[0], depth + 1);
      let d = mathml_inner(&args[1], depth + 1);
      format!("{}<mfrac>\n{}\n{}\n{}</mfrac>", indent, n, d, indent)
    }

    // Power
    "Power" if args.len() == 2 => {
      // Power[x, Rational[1, 2]] → Sqrt
      if let Expr::FunctionCall { name: rn, args: ra } = &args[1]
        && rn == "Rational"
        && ra.len() == 2
        && matches!(&ra[0], Expr::Integer(1))
        && matches!(&ra[1], Expr::Integer(2))
      {
        let b = mathml_inner(&args[0], depth + 1);
        return format!("{}<msqrt>\n{}\n{}</msqrt>", indent, b, indent);
      }
      let b = mathml_inner(&args[0], depth + 1);
      let e = mathml_inner(&args[1], depth + 1);
      format!("{}<msup>\n{}\n{}\n{}</msup>", indent, b, e, indent)
    }

    // Sqrt
    "Sqrt" if args.len() == 1 => {
      let b = mathml_inner(&args[0], depth + 1);
      format!("{}<msqrt>\n{}\n{}</msqrt>", indent, b, indent)
    }

    // Plus (n-ary)
    "Plus" if !args.is_empty() => {
      let mut parts = Vec::new();
      parts.push(mathml_inner(&args[0], depth + 1));
      for arg in args.iter().skip(1) {
        // Check if argument has a leading minus (Times[-1, ...])
        let is_neg = matches!(arg,
          Expr::FunctionCall { name: n, args: a }
            if n == "Times" && !a.is_empty() && matches!(&a[0], Expr::Integer(-1))
        ) || matches!(arg, Expr::Integer(n) if *n < 0)
          || matches!(
            arg,
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              ..
            }
          );

        if is_neg {
          parts.push(format!("{}<mo>-</mo>", inner));
          // Render the negated form
          match arg {
            Expr::FunctionCall { name: n, args: a }
              if n == "Times"
                && a.len() >= 2
                && matches!(&a[0], Expr::Integer(-1)) =>
            {
              if a.len() == 2 {
                parts.push(mathml_inner(&a[1], depth + 1));
              } else {
                parts.push(mathml_function_call("Times", &a[1..], depth + 1));
              }
            }
            Expr::Integer(n) if *n < 0 => {
              parts.push(format!("{}<mn>{}</mn>", inner, -n));
            }
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand,
            } => {
              parts.push(mathml_inner(operand, depth + 1));
            }
            _ => parts.push(mathml_inner(arg, depth + 1)),
          }
        } else {
          parts.push(format!("{}<mo>+</mo>", inner));
          parts.push(mathml_inner(arg, depth + 1));
        }
      }
      format!("{}<mrow>\n{}\n{}</mrow>", indent, parts.join("\n"), indent)
    }

    // Times (n-ary)
    "Times" if args.len() >= 2 => {
      // Check for leading -1
      if matches!(&args[0], Expr::Integer(-1)) {
        let rest = &args[1..];
        if rest.len() == 1 {
          let r = mathml_inner(&rest[0], depth + 1);
          return format!(
            "{}<mrow>\n{}<mo>-</mo>\n{}\n{}</mrow>",
            indent, inner, r, indent
          );
        }
        let r = mathml_function_call("Times", rest, depth + 1);
        return format!(
          "{}<mrow>\n{}<mo>-</mo>\n{}\n{}</mrow>",
          indent, inner, r, indent
        );
      }
      let mut parts = Vec::new();
      parts.push(mathml_inner(&args[0], depth + 1));
      for arg in args.iter().skip(1) {
        parts.push(format!("{}<mo>&#8290;</mo>", inner));
        parts.push(mathml_inner(arg, depth + 1));
      }
      format!("{}<mrow>\n{}\n{}</mrow>", indent, parts.join("\n"), indent)
    }

    // Trig functions (lowercase in MathML)
    "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Log" | "Exp"
      if args.len() == 1 =>
    {
      let fn_name = name.to_lowercase();
      let a = mathml_inner(&args[0], depth + 1);
      format!(
        "{}<mrow>\n{}<mi>{}</mi>\n{}<mo>&#8289;</mo>\n{}<mo>(</mo>\n{}\n{}<mo>)</mo>\n{}</mrow>",
        indent, inner, fn_name, inner, inner, a, inner, indent
      )
    }

    // Complex
    "Complex" if args.len() == 2 => {
      let re = mathml_inner(&args[0], depth + 1);
      let im = mathml_inner(&args[1], depth + 1);
      format!(
        "{}<mrow>\n{}\n{}<mo>+</mo>\n{}\n{}<mo>&#8290;</mo>\n{}<mi>&#8520;</mi>\n{}</mrow>",
        indent, re, inner, im, inner, inner, indent
      )
    }

    // Default: render as function application
    _ => {
      let args_inner: Vec<String> =
        args.iter().map(|a| mathml_inner(a, depth + 1)).collect();
      let mut parts = Vec::new();
      for (i, a) in args_inner.iter().enumerate() {
        parts.push(a.clone());
        if i + 1 < args_inner.len() {
          parts.push(format!("{}<mo>,</mo>", inner));
        }
      }
      format!(
        "{}<mrow>\n{}<mi>{}</mi>\n{}<mo>&#8289;</mo>\n{}<mo>(</mo>\n{}\n{}<mo>)</mo>\n{}</mrow>",
        indent,
        inner,
        mathml_escape_str(name),
        inner,
        inner,
        parts.join("\n"),
        inner,
        indent
      )
    }
  }
}

fn mathml_escape_str(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
}

// ====================================================================
// StandardForm box representation
// ====================================================================

// Wolfram private-use Unicode characters for box syntax.
// These are used internally in strings; InputForm converts them to \!, \(, \*, \).
pub const BOX_START: char = '\u{f7c1}'; // \!
pub const BOX_OPEN: char = '\u{f7c9}'; // \(
pub const BOX_SEP: char = '\u{f7c8}'; // \*
pub const BOX_CLOSE: char = '\u{f7c0}'; // \)

/// Convert a Wolfram expression to its StandardForm box representation.
/// Returns a string using private-use Unicode box markers internally.
/// In InputForm these render as `\!\(\*RowBox[{"..."}]\)`;
/// in OutputForm they render as `DisplayForm[RowBox[{...}]]`.
pub fn expr_to_box_form(expr: &Expr) -> String {
  let box_str = expr_to_boxes(expr);
  // If the result is NOT already a box (RowBox, FractionBox, etc.), wrap in RowBox
  if box_str.contains("Box[") {
    format!(
      "{}{}{}{}{}",
      BOX_START, BOX_OPEN, BOX_SEP, box_str, BOX_CLOSE
    )
  } else {
    format!(
      "{}{}{}RowBox[{{{}}}]{}",
      BOX_START, BOX_OPEN, BOX_SEP, box_str, BOX_CLOSE
    )
  }
}

/// Convert an expression to its box form representation.
/// All atoms are string-quoted to match Wolfram's box format.
pub fn expr_to_boxes(expr: &Expr) -> String {
  use crate::syntax::{BinaryOperator, UnaryOperator};

  match expr {
    // Simple atoms — all quoted in box form
    Expr::Integer(n) => format!("\"{}\"", n),
    Expr::BigInteger(n) => format!("\"{}\"", n),
    Expr::Real(f) => format!("\"{}\"", crate::syntax::format_real(*f)),
    Expr::String(s) => {
      format!(
        "\"\\\"{}\\\"\"",
        s.replace('\\', "\\\\").replace('"', "\\\"")
      )
    }
    Expr::Identifier(name) | Expr::Constant(name) => format!("\"{}\"", name),

    // Unary minus
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      format!("RowBox[{{\"-\", {}}}]", expr_to_boxes(operand))
    }

    // Binary operators
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus => {
        format!(
          "RowBox[{{{}, \"+\", {}}}]",
          expr_to_boxes(left),
          expr_to_boxes(right)
        )
      }
      BinaryOperator::Minus => {
        format!(
          "RowBox[{{{}, \"-\", {}}}]",
          expr_to_boxes(left),
          expr_to_boxes(right)
        )
      }
      BinaryOperator::Times => {
        format!(
          "RowBox[{{{}, \" \", {}}}]",
          expr_to_boxes(left),
          expr_to_boxes(right)
        )
      }
      BinaryOperator::Divide => {
        format!(
          "FractionBox[{}, {}]",
          expr_to_boxes(left),
          expr_to_boxes(right)
        )
      }
      BinaryOperator::Power => {
        if let Some(sqrt_arg) = crate::functions::is_sqrt(expr) {
          return format!("SqrtBox[{}]", expr_to_boxes(sqrt_arg));
        }
        format!(
          "SuperscriptBox[{}, {}]",
          expr_to_boxes(left),
          expr_to_boxes(right)
        )
      }
      _ => {
        let op_str = match op {
          BinaryOperator::And => "&&",
          BinaryOperator::Or => "||",
          BinaryOperator::StringJoin => "<>",
          BinaryOperator::Alternatives => "|",
          _ => "?",
        };
        format!(
          "RowBox[{{{}, \"{}\", {}}}]",
          expr_to_boxes(left),
          op_str,
          expr_to_boxes(right)
        )
      }
    },

    // Lists
    Expr::List(items) => {
      let mut parts = vec!["\"{{\"".to_string()];
      if !items.is_empty() {
        let inner: Vec<String> = items.iter().map(expr_to_boxes).collect();
        parts.push(format!("RowBox[{{{}}}]", inner.join(", \",\", ")));
      }
      parts.push("\"}}\"".to_string());
      format!("RowBox[{{{}}}]", parts.join(", "))
    }

    // Function calls
    Expr::FunctionCall { name, args } => box_function_call(name, args),

    // Fallback
    _ => crate::syntax::expr_to_output(expr),
  }
}

fn box_function_call(name: &str, args: &[Expr]) -> String {
  match name {
    // Rational → FractionBox
    "Rational" if args.len() == 2 => {
      format!(
        "FractionBox[{}, {}]",
        expr_to_boxes(&args[0]),
        expr_to_boxes(&args[1])
      )
    }

    // Power → SuperscriptBox
    "Power" if args.len() == 2 => {
      // Power[x, Rational[1, 2]] → SqrtBox
      if let Expr::FunctionCall { name: rn, args: ra } = &args[1]
        && rn == "Rational"
        && ra.len() == 2
        && matches!(&ra[0], Expr::Integer(1))
        && matches!(&ra[1], Expr::Integer(2))
      {
        return format!("SqrtBox[{}]", expr_to_boxes(&args[0]));
      }
      format!(
        "SuperscriptBox[{}, {}]",
        expr_to_boxes(&args[0]),
        expr_to_boxes(&args[1])
      )
    }

    // Sqrt → SqrtBox
    "Sqrt" if args.len() == 1 => {
      format!("SqrtBox[{}]", expr_to_boxes(&args[0]))
    }

    // Plus → RowBox with + separators
    "Plus" if !args.is_empty() => {
      let mut parts = Vec::new();
      parts.push(expr_to_boxes(&args[0]));
      for arg in args.iter().skip(1) {
        // Check for negative terms (Times[-1, ...])
        let is_neg = matches!(arg,
          Expr::FunctionCall { name: n, args: a }
            if n == "Times" && !a.is_empty() && matches!(&a[0], Expr::Integer(-1))
        ) || matches!(arg, Expr::Integer(n) if *n < 0);

        if is_neg {
          parts.push("\"-\"".to_string());
          match arg {
            Expr::FunctionCall { name: n, args: a }
              if n == "Times"
                && a.len() >= 2
                && matches!(&a[0], Expr::Integer(-1)) =>
            {
              if a.len() == 2 {
                parts.push(expr_to_boxes(&a[1]));
              } else {
                parts.push(box_function_call("Times", &a[1..]));
              }
            }
            Expr::Integer(n) if *n < 0 => {
              parts.push(format!("\"{}\"", -n));
            }
            _ => parts.push(expr_to_boxes(arg)),
          }
        } else {
          parts.push("\"+\"".to_string());
          parts.push(expr_to_boxes(arg));
        }
      }
      format!("RowBox[{{{}}}]", parts.join(", "))
    }

    // Times → RowBox with space separators, or FractionBox for fractions
    "Times" if args.len() >= 2 => {
      // Leading -1
      if matches!(&args[0], Expr::Integer(-1)) {
        if args.len() == 2 {
          return format!("RowBox[{{\"-\", {}}}]", expr_to_boxes(&args[1]));
        }
        let rest = box_function_call("Times", &args[1..]);
        return format!("RowBox[{{\"-\", {}}}]", rest);
      }
      // Check for fraction form: Times[..., Power[den, -1]]
      let full_expr = Expr::FunctionCall {
        name: "Times".to_string(),
        args: args.to_vec(),
      };
      let (num, den) =
        crate::functions::polynomial_ast::together::extract_num_den(&full_expr);
      if !matches!(&den, Expr::Integer(1)) {
        return format!(
          "FractionBox[{}, {}]",
          expr_to_boxes(&num),
          expr_to_boxes(&den)
        );
      }
      let parts: Vec<String> = args.iter().map(expr_to_boxes).collect();
      format!("RowBox[{{{}}}]", parts.join(", \" \", "))
    }

    // Default: function application
    _ => {
      let args_boxes: Vec<String> = args.iter().map(expr_to_boxes).collect();
      format!(
        "RowBox[{{\"{}[\", {}, \"]\"}}]",
        name,
        args_boxes.join(", \",\", ")
      )
    }
  }
}

/// Format a StringForm expression by substituting placeholders.
/// `template` is the format string, `values` are the arguments to substitute.
/// `` `` `` placeholders are replaced sequentially, `` `n` `` with the nth argument.
pub fn format_string_form(template: &str, values: &[Expr]) -> String {
  let mut result = String::new();
  let chars: Vec<char> = template.chars().collect();
  let len = chars.len();
  let mut i = 0;
  let mut seq_index = 0; // sequential placeholder counter

  while i < len {
    if chars[i] == '`' {
      // Check for `` (sequential placeholder)
      if i + 1 < len && chars[i + 1] == '`' {
        if seq_index < values.len() {
          result.push_str(&crate::syntax::expr_to_output(&values[seq_index]));
        }
        seq_index += 1;
        i += 2;
        continue;
      }
      // Check for `n` (indexed placeholder)
      let start = i + 1;
      let mut end = start;
      while end < len && chars[end].is_ascii_digit() {
        end += 1;
      }
      if end > start && end < len && chars[end] == '`' {
        let idx: usize = chars[start..end]
          .iter()
          .collect::<String>()
          .parse()
          .unwrap_or(0);
        if idx >= 1 && idx <= values.len() {
          result.push_str(&crate::syntax::expr_to_output(&values[idx - 1]));
        }
        i = end + 1;
        continue;
      }
    }
    result.push(chars[i]);
    i += 1;
  }
  result
}

/// ToExpression[s] - convert string to expression and evaluate
pub fn to_expression_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToExpression expects exactly 1 argument".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  // Parse and evaluate the string as code
  let parsed = crate::syntax::string_to_expr(&s)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

/// StringPadLeft[s, n] or StringPadLeft[s, n, pad] - pad string on left
pub fn string_pad_left_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringPadLeft expects 2 or 3 arguments".into(),
    ));
  }

  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| {
        let mut call = vec![s.clone()];
        call.extend(args[1..].iter().cloned());
        string_pad_left_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?));
  }

  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringPadLeft must be non-negative".into(),
    ));
  }

  let target_len = n as usize;
  let pad_str = if args.len() == 3 {
    let p = expr_to_str(&args[2])?;
    if p.is_empty() { " ".to_string() } else { p }
  } else {
    " ".to_string()
  };

  let char_count = s.chars().count();
  if char_count >= target_len {
    Ok(Expr::String(
      s.chars().skip(char_count - target_len).collect(),
    ))
  } else {
    // Wolfram fills entire target with right-aligned repeating pad,
    // then overlays the original string at the right end
    let pad_len = pad_str.chars().count();
    let offset = (pad_len - (target_len % pad_len)) % pad_len;
    let pad_needed = target_len - char_count;
    let padding: String = pad_str
      .chars()
      .cycle()
      .skip(offset)
      .take(pad_needed)
      .collect();
    Ok(Expr::String(format!("{}{}", padding, s)))
  }
}

/// StringPadRight[s, n] or StringPadRight[s, n, pad] - pad string on right
pub fn string_pad_right_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringPadRight expects 2 or 3 arguments".into(),
    ));
  }

  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| {
        let mut call = vec![s.clone()];
        call.extend(args[1..].iter().cloned());
        string_pad_right_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?));
  }

  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringPadRight must be non-negative".into(),
    ));
  }

  let target_len = n as usize;
  let pad_str = if args.len() == 3 {
    let p = expr_to_str(&args[2])?;
    if p.is_empty() { " ".to_string() } else { p }
  } else {
    " ".to_string()
  };

  let char_count = s.chars().count();
  if char_count >= target_len {
    Ok(Expr::String(s.chars().take(target_len).collect()))
  } else {
    // Wolfram fills entire target with left-aligned repeating pad,
    // then overlays the original string at the left end
    let pad_len = pad_str.chars().count();
    let offset = char_count % pad_len;
    let pad_needed = target_len - char_count;
    let padding: String = pad_str
      .chars()
      .cycle()
      .skip(offset)
      .take(pad_needed)
      .collect();
    Ok(Expr::String(format!("{}{}", s, padding)))
  }
}

/// StringCount[s, sub] - count occurrences of substring or pattern
pub fn string_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringCount expects exactly 2 arguments".into(),
    ));
  }

  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| string_count_ast(&[s.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  let s = expr_to_str(&args[0])?;

  // Try regex-based pattern first (handles RegularExpression, string patterns,
  // lists-of-patterns as alternatives, etc.)
  if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    let re = regex::Regex::new(&regex_pat).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
    })?;
    return Ok(Expr::Integer(re.find_iter(&s).count() as i128));
  }

  // Fallback to plain string matching
  let sub = expr_to_str(&args[1])?;
  if sub.is_empty() {
    return Ok(Expr::Integer(0));
  }
  Ok(Expr::Integer(s.matches(&sub).count() as i128))
}

/// StringFreeQ[s, sub] - check if string does NOT contain substring or pattern
pub fn string_free_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringFreeQ expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;

  // Try regex-based pattern first
  if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    let re = regex::Regex::new(&regex_pat).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "False" } else { "True" }.to_string(),
    ));
  }

  let sub = expr_to_str(&args[1])?;
  Ok(Expr::Identifier(
    if s.contains(&sub) { "False" } else { "True" }.to_string(),
  ))
}

/// ToCharacterCode[s] - converts a string to a list of character codes
pub fn to_character_code_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToCharacterCode expects exactly 1 argument".into(),
    ));
  }
  // Handle list of strings
  if let Expr::List(items) = &args[0] {
    let mut results = Vec::new();
    for item in items {
      let s = expr_to_str(item)?;
      let codes: Vec<Expr> =
        s.chars().map(|c| Expr::Integer(c as i128)).collect();
      results.push(Expr::List(codes));
    }
    return Ok(Expr::List(results));
  }
  let s = expr_to_str(&args[0])?;
  let codes: Vec<Expr> = s.chars().map(|c| Expr::Integer(c as i128)).collect();
  Ok(Expr::List(codes))
}

/// FromCharacterCode[n] or FromCharacterCode[{n1, n2, ...}] - converts character codes to a string
pub fn from_character_code_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FromCharacterCode expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => {
      let c = char::from_u32(*n as u32).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "Invalid character code: {}",
          n
        ))
      })?;
      Ok(Expr::String(c.to_string()))
    }
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      let code = n.to_u32().ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "Invalid character code: {}",
          n
        ))
      })?;
      let c = char::from_u32(code).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "Invalid character code: {}",
          n
        ))
      })?;
      Ok(Expr::String(c.to_string()))
    }
    Expr::List(items) => {
      // Check if this is a list of lists (nested)
      if !items.is_empty() && matches!(&items[0], Expr::List(_)) {
        let mut results = Vec::new();
        for item in items {
          let sub_result = from_character_code_ast(&[item.clone()])?;
          results.push(sub_result);
        }
        return Ok(Expr::List(results));
      }
      let mut result = String::new();
      for item in items {
        let code = match item {
          Expr::Integer(n) => *n as u32,
          Expr::BigInteger(n) => {
            use num_traits::ToPrimitive;
            n.to_u32().ok_or_else(|| {
              InterpreterError::EvaluationError(format!(
                "Invalid character code: {}",
                n
              ))
            })?
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "FromCharacterCode expects integer arguments".into(),
            ));
          }
        };
        let c = char::from_u32(code).ok_or_else(|| {
          InterpreterError::EvaluationError(format!(
            "Invalid character code: {}",
            code
          ))
        })?;
        result.push(c);
      }
      Ok(Expr::String(result))
    }
    _ => Err(InterpreterError::EvaluationError(
      "FromCharacterCode expects an integer or list of integers".into(),
    )),
  }
}

/// CharacterRange[c1, c2] - generates a list of characters from c1 to c2
pub fn character_range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CharacterRange expects exactly 2 arguments".into(),
    ));
  }

  let s1 = expr_to_str(&args[0])?;
  let s2 = expr_to_str(&args[1])?;

  let c1 = s1.chars().next().ok_or_else(|| {
    InterpreterError::EvaluationError(
      "CharacterRange: first argument must be a single character".into(),
    )
  })?;
  let c2 = s2.chars().next().ok_or_else(|| {
    InterpreterError::EvaluationError(
      "CharacterRange: second argument must be a single character".into(),
    )
  })?;

  let start = c1 as u32;
  let end = c2 as u32;

  if start > end {
    return Ok(Expr::List(vec![]));
  }

  let chars: Vec<Expr> = (start..=end)
    .filter_map(char::from_u32)
    .map(|c| Expr::String(c.to_string()))
    .collect();
  Ok(Expr::List(chars))
}

/// IntegerString[n] or IntegerString[n, base] or IntegerString[n, base, length] - convert integer to string
pub fn integer_string_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "IntegerString expects 1, 2, or 3 arguments".into(),
    ));
  }

  let n = expr_to_int(&args[0])?;
  let base = if args.len() >= 2 {
    expr_to_int(&args[1])? as u32
  } else {
    10
  };

  if !(2..=36).contains(&base) {
    return Err(InterpreterError::EvaluationError(
      "IntegerString: base must be between 2 and 36".into(),
    ));
  }

  // IntegerString uses absolute value (drops sign)
  let abs_n = n.unsigned_abs();

  let mut result = String::new();
  if abs_n == 0 {
    result.push('0');
  } else {
    let mut val = abs_n;
    while val > 0 {
      let digit = (val % base as u128) as u32;
      let c = char::from_digit(digit, base).unwrap();
      result.push(c);
      val /= base as u128;
    }
    result = result.chars().rev().collect();
  }

  // If length is specified, pad or truncate
  if args.len() == 3 {
    let target_len = expr_to_int(&args[2])? as usize;
    if result.len() < target_len {
      // Pad with zeros on the left
      let padding: String =
        std::iter::repeat_n('0', target_len - result.len()).collect();
      result = format!("{}{}", padding, result);
    } else if result.len() > target_len {
      // Truncate from the left (keep rightmost digits)
      result = result[result.len() - target_len..].to_string();
    }
  }

  Ok(Expr::String(result))
}

// ─── Alphabet ──────────────────────────────────────────────────────

/// Alphabet[] - Returns the list of lowercase English letters
pub fn alphabet_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Alphabet".to_string(),
      args: args.to_vec(),
    });
  }
  let letters: Vec<Expr> =
    ('a'..='z').map(|c| Expr::String(c.to_string())).collect();
  Ok(Expr::List(letters))
}

// ─── FromLetterNumber / LetterNumber ──────────────────────────────

/// FromLetterNumber[n] - give the nth letter of the English alphabet.
/// Out-of-range or 0 returns a space character.
/// Negative numbers wrap cyclically (e.g., -1 -> z).
/// Works with lists of integers too.
pub fn from_letter_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fn letter_from_int(n: i128) -> Expr {
    // Valid range: 1..=26 for positive, -26..=-1 for negative
    // Everything else (0, >26, <-26) returns space
    let pos = if (1..=26).contains(&n) {
      n as u8 // 1..=26
    } else if (-26..=-1).contains(&n) {
      (26 + n + 1) as u8 // -1->26, -2->25, ..., -26->1
    } else {
      return Expr::String(" ".to_string());
    };
    let ch = (b'a' + pos - 1) as char;
    Expr::String(ch.to_string())
  }

  match &args[0] {
    Expr::Integer(n) => Ok(letter_from_int(*n)),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|item| match item {
          Expr::Integer(n) => Ok(letter_from_int(*n)),
          _ => Ok(Expr::FunctionCall {
            name: "FromLetterNumber".to_string(),
            args: vec![item.clone()],
          }),
        })
        .collect();
      Ok(Expr::List(results?))
    }
    _ => Ok(Expr::FunctionCall {
      name: "FromLetterNumber".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// LetterNumber["c"] - give the position of a letter in the English alphabet.
/// Returns 0 for non-letter characters.
pub fn letter_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match &args[0] {
    Expr::String(s) => {
      if s.len() == 1 {
        let ch = s.chars().next().unwrap().to_ascii_lowercase();
        if ch.is_ascii_lowercase() {
          Ok(Expr::Integer((ch as i128) - ('a' as i128) + 1))
        } else {
          Ok(Expr::Integer(0))
        }
      } else {
        // For multi-character strings, return a list
        let results: Vec<Expr> = s
          .chars()
          .map(|ch| {
            let lower = ch.to_ascii_lowercase();
            if lower.is_ascii_lowercase() {
              Expr::Integer((lower as i128) - ('a' as i128) + 1)
            } else {
              Expr::Integer(0)
            }
          })
          .collect();
        Ok(Expr::List(results))
      }
    }
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|item| letter_number_ast(&[item.clone()]))
        .collect();
      Ok(Expr::List(results?))
    }
    _ => Ok(Expr::FunctionCall {
      name: "LetterNumber".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── LetterQ ───────────────────────────────────────────────────────

/// LetterQ[string] - True if string consists entirely of letters
pub fn letter_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LetterQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::String(s) => {
      let result = s.chars().all(|c| c.is_alphabetic());
      Ok(Expr::Identifier(
        if result { "True" } else { "False" }.to_string(),
      ))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

// ─── UpperCaseQ ────────────────────────────────────────────────────

/// UpperCaseQ[string] - True if string consists entirely of uppercase letters
pub fn upper_case_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "UpperCaseQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::String(s) => {
      let result = s.chars().all(|c| c.is_alphabetic() && c.is_uppercase());
      Ok(Expr::Identifier(
        if result { "True" } else { "False" }.to_string(),
      ))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

// ─── LowerCaseQ ────────────────────────────────────────────────────

/// LowerCaseQ[string] - True if string consists entirely of lowercase letters
pub fn lower_case_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LowerCaseQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::String(s) => {
      let result = s.chars().all(|c| c.is_alphabetic() && c.is_lowercase());
      Ok(Expr::Identifier(
        if result { "True" } else { "False" }.to_string(),
      ))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

// ─── StringInsert ──────────────────────────────────────────────────

/// StringInsert[string, snew, n] - inserts snew at position n in string
pub fn string_insert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "StringInsert expects exactly 3 arguments".into(),
    ));
  }

  // Handle list of strings as first argument
  if let Expr::List(strings) = &args[0] {
    let mut results = Vec::new();
    for s_expr in strings {
      let new_args = vec![s_expr.clone(), args[1].clone(), args[2].clone()];
      results.push(string_insert_ast(&new_args)?);
    }
    return Ok(Expr::List(results));
  }

  let s = expr_to_str(&args[0])?;
  let insert = expr_to_str(&args[1])?;

  // Handle list of positions
  if let Expr::List(positions) = &args[2] {
    let mut pos_list: Vec<i128> = Vec::new();
    for p in positions {
      pos_list.push(expr_to_int(p)?);
    }

    let chars: Vec<char> = s.chars().collect();
    let len = chars.len() as i128;

    // Convert all positions to 0-based indices, accounting for
    // the fact that earlier insertions shift later positions
    let mut abs_positions: Vec<usize> = Vec::new();
    for &n in &pos_list {
      let pos = if n > 0 {
        (n - 1).min(len) as usize
      } else if n < 0 {
        let p = len + 1 + n;
        p.max(0) as usize
      } else {
        return Err(InterpreterError::EvaluationError(
          "StringInsert: position cannot be 0".into(),
        ));
      };
      abs_positions.push(pos);
    }

    // Sort positions in ascending order
    abs_positions.sort();

    // Build result by inserting at each position, adjusting for previous insertions
    let mut result = String::new();
    let mut last = 0;
    for (i, &pos) in abs_positions.iter().enumerate() {
      let adjusted_pos = pos;
      if adjusted_pos > last {
        result.extend(&chars[last..adjusted_pos.min(chars.len())]);
      }
      result.push_str(&insert);
      last = adjusted_pos;
      let _ = i;
    }
    if last < chars.len() {
      result.extend(&chars[last..]);
    }
    return Ok(Expr::String(result));
  }

  let n = expr_to_int(&args[2])?;

  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;

  // Wolfram: positive n means before position n, negative means from end
  let pos = if n > 0 {
    (n - 1).min(len) as usize
  } else if n < 0 {
    // -1 means after last char, -2 means before last char, etc.
    let p = len + 1 + n;
    p.max(0) as usize
  } else {
    return Err(InterpreterError::EvaluationError(
      "StringInsert: position cannot be 0".into(),
    ));
  };

  let mut result: String = chars[..pos].iter().collect();
  result.push_str(&insert);
  result.extend(&chars[pos..]);
  Ok(Expr::String(result))
}

// ─── StringReplacePart ─────────────────────────────────────────────

/// StringReplacePart["string", "new", {m, n}] - replace characters m through n
/// StringReplacePart["string", "new", {{m1,n1},{m2,n2},...}] - replace multiple ranges
/// StringReplacePart["string", {"s1","s2",...}, {{m1,n1},{m2,n2},...}] - different replacements
pub fn string_replace_part_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "StringReplacePart expects exactly 3 arguments".into(),
    ));
  }

  let s = match expr_to_str(&args[0]) {
    Ok(s) => s,
    Err(_) => {
      return Ok(Expr::FunctionCall {
        name: "StringReplacePart".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;

  // Helper to resolve a position index (1-based, negative from end)
  let resolve_index = |n: i128| -> Result<usize, InterpreterError> {
    let idx = if n > 0 { n - 1 } else { len + n };
    if idx < 0 || idx >= len {
      return Err(InterpreterError::EvaluationError(format!(
        "StringReplacePart: index {} out of range for string of length {}",
        n, len
      )));
    }
    Ok(idx as usize)
  };

  // Parse a range {m, n} into (start, end) 0-based inclusive
  let parse_range =
    |elems: &[Expr]| -> Result<(usize, usize), InterpreterError> {
      if elems.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "StringReplacePart: range must be a list of two integers".into(),
        ));
      }
      let m = expr_to_int(&elems[0])?;
      let n = expr_to_int(&elems[1])?;
      let start = resolve_index(m)?;
      let end = resolve_index(n)?;
      Ok((start, end))
    };

  // Determine ranges and replacements
  match &args[2] {
    // Single range: {m, n}
    Expr::List(elems)
      if elems.len() == 2 && !matches!(&elems[0], Expr::List(_)) =>
    {
      let (start, end) = parse_range(elems)?;
      let replacement = expr_to_str(&args[1])?;
      let mut result = String::new();
      result.extend(&chars[..start]);
      result.push_str(&replacement);
      if end + 1 < chars.len() {
        result.extend(&chars[end + 1..]);
      }
      Ok(Expr::String(result))
    }
    // Multiple ranges: {{m1,n1}, {m2,n2}, ...}
    Expr::List(ranges) => {
      let mut range_vec: Vec<(usize, usize)> = Vec::new();
      for range in ranges {
        if let Expr::List(elems) = range {
          range_vec.push(parse_range(elems)?);
        } else {
          return Err(InterpreterError::EvaluationError(
            "StringReplacePart: expected list of ranges".into(),
          ));
        }
      }

      // Get replacements (single string or list of strings)
      let replacements: Vec<String> = match &args[1] {
        Expr::List(repls) => {
          if repls.len() != range_vec.len() {
            return Err(InterpreterError::EvaluationError(
              "StringReplacePart: number of replacements must match number of ranges".into(),
            ));
          }
          repls
            .iter()
            .map(expr_to_str)
            .collect::<Result<Vec<_>, _>>()?
        }
        _ => {
          let r = expr_to_str(&args[1])?;
          vec![r; range_vec.len()]
        }
      };

      // Sort ranges by start position, along with their replacements
      let mut indexed: Vec<(usize, (usize, usize), &str)> = range_vec
        .iter()
        .zip(replacements.iter())
        .enumerate()
        .map(|(i, (&range, repl))| (i, range, repl.as_str()))
        .collect();
      indexed.sort_by_key(|&(_, (start, _), _)| start);

      // Build result
      let mut result = String::new();
      let mut pos = 0;
      for (_, (start, end), repl) in &indexed {
        if *start > pos {
          result.extend(&chars[pos..*start]);
        }
        result.push_str(repl);
        pos = end + 1;
      }
      if pos < chars.len() {
        result.extend(&chars[pos..]);
      }
      Ok(Expr::String(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "StringReplacePart".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── StringDelete ──────────────────────────────────────────────────

/// StringDelete[string, sub] - deletes all occurrences of sub from string
pub fn string_delete_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringDelete expects exactly 2 arguments".into(),
    ));
  }
  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| string_delete_ast(&[item.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }
  let s = expr_to_str(&args[0])?;
  let sub = expr_to_str(&args[1])?;
  Ok(Expr::String(s.replace(&sub, "")))
}

// ─── Capitalize ────────────────────────────────────────────────────

/// Capitalize[string] - capitalizes the first letter of the string
pub fn capitalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Capitalize expects exactly 1 argument".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  if s.is_empty() {
    return Ok(Expr::String(s));
  }
  let mut chars = s.chars();
  let first = chars.next().unwrap().to_uppercase().to_string();
  let rest: String = chars.collect();
  Ok(Expr::String(format!("{}{}", first, rest)))
}

// ─── Decapitalize ──────────────────────────────────────────────────

/// Decapitalize[string] - lowercases the first letter of the string
pub fn decapitalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Decapitalize expects exactly 1 argument".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  if s.is_empty() {
    return Ok(Expr::String(s));
  }
  let mut chars = s.chars();
  let first = chars.next().unwrap().to_lowercase().to_string();
  let rest: String = chars.collect();
  Ok(Expr::String(format!("{}{}", first, rest)))
}

// ─── DigitQ ────────────────────────────────────────────────────────

/// DigitQ[string] - True if string consists entirely of digits
pub fn digit_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DigitQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::String(s) => {
      let result = s.chars().all(|c| c.is_ascii_digit());
      Ok(Expr::Identifier(
        if result { "True" } else { "False" }.to_string(),
      ))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

/// EditDistance[s1, s2] - Levenshtein distance between two strings
pub fn edit_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "EditDistance expects exactly 2 arguments".into(),
    ));
  }
  let s1 = expr_to_str(&args[0])?;
  let s2 = expr_to_str(&args[1])?;
  let a: Vec<char> = s1.chars().collect();
  let b: Vec<char> = s2.chars().collect();
  let n = a.len();
  let m = b.len();

  let mut dp = vec![vec![0usize; m + 1]; n + 1];
  for i in 0..=n {
    dp[i][0] = i;
  }
  for j in 0..=m {
    dp[0][j] = j;
  }
  for i in 1..=n {
    for j in 1..=m {
      let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
      dp[i][j] = (dp[i - 1][j] + 1)
        .min(dp[i][j - 1] + 1)
        .min(dp[i - 1][j - 1] + cost);
    }
  }
  Ok(Expr::Integer(dp[n][m] as i128))
}

/// LongestCommonSubsequence[s1, s2] - longest common (non-contiguous) subsequence
pub fn longest_common_subsequence_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LongestCommonSubsequence expects exactly 2 arguments".into(),
    ));
  }
  let s1 = expr_to_str(&args[0])?;
  let s2 = expr_to_str(&args[1])?;
  let chars1: Vec<char> = s1.chars().collect();
  let chars2: Vec<char> = s2.chars().collect();
  let n = chars1.len();
  let m = chars2.len();

  // DP table for longest common substring (contiguous)
  // Wolfram's LongestCommonSubsequence finds the longest contiguous match
  let mut dp = vec![vec![0usize; m + 1]; n + 1];
  let mut max_len = 0usize;
  let mut end_i = 0usize; // end position in chars1
  for i in 1..=n {
    for j in 1..=m {
      if chars1[i - 1] == chars2[j - 1] {
        dp[i][j] = dp[i - 1][j - 1] + 1;
        if dp[i][j] > max_len {
          max_len = dp[i][j];
          end_i = i;
        }
      }
    }
  }

  let result: String = chars1[end_i - max_len..end_i].iter().collect();
  Ok(Expr::String(result))
}

/// SequenceAlignment[s1, s2] — aligns two strings using Needleman-Wunsch
/// Returns a list of matching segments and {diff1, diff2} pairs.
pub fn sequence_alignment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SequenceAlignment expects exactly 2 arguments".into(),
    ));
  }

  // Handle both strings and lists
  let (is_string, chars1, chars2) = match (&args[0], &args[1]) {
    (Expr::String(s1), Expr::String(s2)) => {
      let c1: Vec<String> = s1.chars().map(|c| c.to_string()).collect();
      let c2: Vec<String> = s2.chars().map(|c| c.to_string()).collect();
      (true, c1, c2)
    }
    (Expr::List(l1), Expr::List(l2)) => {
      let c1: Vec<String> =
        l1.iter().map(crate::syntax::expr_to_output).collect();
      let c2: Vec<String> =
        l2.iter().map(crate::syntax::expr_to_output).collect();
      (false, c1, c2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SequenceAlignment".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = chars1.len();
  let m = chars2.len();

  // Needleman-Wunsch DP with match=1, mismatch=-1, gap=-1
  let mut dp = vec![vec![0i64; m + 1]; n + 1];
  for i in 0..=n {
    dp[i][0] = -(i as i64);
  }
  for j in 0..=m {
    dp[0][j] = -(j as i64);
  }
  for i in 1..=n {
    for j in 1..=m {
      let match_score = if chars1[i - 1] == chars2[j - 1] {
        1
      } else {
        -1
      };
      dp[i][j] = (dp[i - 1][j - 1] + match_score)
        .max(dp[i - 1][j] - 1)
        .max(dp[i][j - 1] - 1);
    }
  }

  // Traceback
  let mut aligned1: Vec<Option<usize>> = Vec::new(); // index into chars1 or None for gap
  let mut aligned2: Vec<Option<usize>> = Vec::new(); // index into chars2 or None for gap
  let (mut i, mut j) = (n, m);
  while i > 0 || j > 0 {
    if i > 0 && j > 0 {
      let match_score = if chars1[i - 1] == chars2[j - 1] {
        1
      } else {
        -1
      };
      if dp[i][j] == dp[i - 1][j - 1] + match_score {
        aligned1.push(Some(i - 1));
        aligned2.push(Some(j - 1));
        i -= 1;
        j -= 1;
        continue;
      }
    }
    if i > 0 && dp[i][j] == dp[i - 1][j] - 1 {
      aligned1.push(Some(i - 1));
      aligned2.push(None);
      i -= 1;
    } else {
      aligned1.push(None);
      aligned2.push(Some(j - 1));
      j -= 1;
    }
  }
  aligned1.reverse();
  aligned2.reverse();

  // Build result segments
  let mut result: Vec<Expr> = Vec::new();
  let mut k = 0;
  let len = aligned1.len();

  while k < len {
    if aligned1[k].is_some() && aligned2[k].is_some() {
      let i1 = aligned1[k].unwrap();
      let j1 = aligned2[k].unwrap();
      if chars1[i1] == chars2[j1] {
        // Matching segment
        let mut match_str = chars1[i1].clone();
        k += 1;
        while k < len
          && aligned1[k].is_some()
          && aligned2[k].is_some()
          && chars1[aligned1[k].unwrap()] == chars2[aligned2[k].unwrap()]
        {
          match_str.push_str(&chars1[aligned1[k].unwrap()]);
          k += 1;
        }
        if is_string {
          result.push(Expr::String(match_str));
        } else {
          // Re-parse as list elements
          let items: Vec<Expr> = match_str
            .split("")
            .filter(|s| !s.is_empty())
            .map(|_| {
              // This is simplified; for lists we need original exprs
              Expr::Identifier("?".to_string())
            })
            .collect();
          // Actually for lists, collect the original exprs
          let start_i = i1;
          let count = match_str.len(); // This won't work well for lists
          let _ = (start_i, count, items);
          result.push(Expr::String(match_str));
        }
      } else {
        // Mismatch
        let mut diff1 = chars1[i1].clone();
        let mut diff2 = chars2[j1].clone();
        k += 1;
        while k < len
          && aligned1[k].is_some()
          && aligned2[k].is_some()
          && chars1[aligned1[k].unwrap()] != chars2[aligned2[k].unwrap()]
        {
          diff1.push_str(&chars1[aligned1[k].unwrap()]);
          diff2.push_str(&chars2[aligned2[k].unwrap()]);
          k += 1;
        }
        if is_string {
          result
            .push(Expr::List(vec![Expr::String(diff1), Expr::String(diff2)]));
        } else {
          result
            .push(Expr::List(vec![Expr::String(diff1), Expr::String(diff2)]));
        }
      }
    } else {
      // Gap
      let mut diff1 = String::new();
      let mut diff2 = String::new();
      while k < len && (aligned1[k].is_none() || aligned2[k].is_none()) {
        if let Some(i1) = aligned1[k] {
          diff1.push_str(&chars1[i1]);
        }
        if let Some(j1) = aligned2[k] {
          diff2.push_str(&chars2[j1]);
        }
        k += 1;
      }
      if is_string {
        result.push(Expr::List(vec![Expr::String(diff1), Expr::String(diff2)]));
      } else {
        result.push(Expr::List(vec![Expr::String(diff1), Expr::String(diff2)]));
      }
    }
  }

  Ok(Expr::List(result))
}

/// StringPart[s, n] - nth character; StringPart[s, {n1,n2,...}] - multiple characters
pub fn string_part_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringPart expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;

  fn resolve_index(n: i128, len: i128) -> Result<usize, InterpreterError> {
    let idx = if n > 0 { n - 1 } else { len + n };
    if idx < 0 || idx >= len {
      return Err(InterpreterError::EvaluationError(format!(
        "StringPart: index {} out of range for string of length {}",
        n, len
      )));
    }
    Ok(idx as usize)
  }

  match &args[1] {
    Expr::List(indices) => {
      let mut result = Vec::new();
      for idx_expr in indices {
        let n = expr_to_int(idx_expr)?;
        let idx = resolve_index(n, len)?;
        result.push(Expr::String(chars[idx].to_string()));
      }
      Ok(Expr::List(result))
    }
    _ => {
      let n = expr_to_int(&args[1])?;
      let idx = resolve_index(n, len)?;
      Ok(Expr::String(chars[idx].to_string()))
    }
  }
}

/// StringTakeDrop[s, n] - returns {StringTake[s,n], StringDrop[s,n]}
pub fn string_take_drop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringTakeDrop expects exactly 2 arguments".into(),
    ));
  }
  let taken = string_take_ast(args)?;
  let dropped = string_drop_ast(args)?;
  Ok(Expr::List(vec![taken, dropped]))
}

/// HammingDistance[s1, s2] - number of positions where characters differ
pub fn hamming_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "HammingDistance expects exactly 2 arguments".into(),
    ));
  }
  let s1 = expr_to_str(&args[0])?;
  let s2 = expr_to_str(&args[1])?;
  let chars1: Vec<char> = s1.chars().collect();
  let chars2: Vec<char> = s2.chars().collect();

  if chars1.len() != chars2.len() {
    return Err(InterpreterError::EvaluationError(
      "HammingDistance: strings must have the same length".into(),
    ));
  }

  let dist = chars1
    .iter()
    .zip(chars2.iter())
    .filter(|(a, b)| a != b)
    .count();
  Ok(Expr::Integer(dist as i128))
}

/// CharacterCounts[s] - association of character frequencies, sorted by frequency descending
pub fn character_counts_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CharacterCounts expects exactly 1 argument".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;

  // Count characters preserving first-occurrence order
  let mut counts: Vec<(char, i128)> = Vec::new();
  let mut seen: std::collections::HashMap<char, usize> =
    std::collections::HashMap::new();
  for c in s.chars() {
    if let Some(&idx) = seen.get(&c) {
      counts[idx].1 += 1;
    } else {
      seen.insert(c, counts.len());
      counts.push((c, 1));
    }
  }

  // Sort by frequency descending, then by reverse first-occurrence for ties
  // (reverse the list, then stable-sort by frequency descending)
  counts.reverse();
  counts.sort_by(|a, b| b.1.cmp(&a.1));

  let items: Vec<(Expr, Expr)> = counts
    .into_iter()
    .map(|(c, n)| (Expr::String(c.to_string()), Expr::Integer(n)))
    .collect();
  Ok(Expr::Association(items))
}

/// RemoveDiacritics[s] - remove diacritical marks from string
pub fn remove_diacritics_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RemoveDiacritics expects exactly 1 argument".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  use unicode_normalization::UnicodeNormalization;
  // NFD decompose, then filter out combining marks (Unicode category Mn)
  let result: String = s.nfd().filter(|c| !is_combining_mark(*c)).collect();
  Ok(Expr::String(result))
}

/// Check if a character is a Unicode combining mark (category Mn/Mc/Me)
fn is_combining_mark(c: char) -> bool {
  let cp = c as u32;
  // Combining Diacritical Marks: U+0300..U+036F
  // Combining Diacritical Marks Extended: U+1AB0..U+1AFF
  // Combining Diacritical Marks Supplement: U+1DC0..U+1DFF
  // Combining Diacritical Marks for Symbols: U+20D0..U+20FF
  // Combining Half Marks: U+FE20..U+FE2F
  (0x0300..=0x036F).contains(&cp)
    || (0x1AB0..=0x1AFF).contains(&cp)
    || (0x1DC0..=0x1DFF).contains(&cp)
    || (0x20D0..=0x20FF).contains(&cp)
    || (0xFE20..=0xFE2F).contains(&cp)
}

/// StringRotateLeft[s] or StringRotateLeft[s, n] - rotate characters left
pub fn string_rotate_left_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "StringRotateLeft expects 1 or 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let n = if args.len() == 2 {
    expr_to_int(&args[1])?
  } else {
    1
  };

  let chars: Vec<char> = s.chars().collect();
  if chars.is_empty() {
    return Ok(Expr::String(s));
  }

  let len = chars.len();
  let shift = ((n % len as i128) + len as i128) as usize % len;
  let mut result: Vec<char> = Vec::with_capacity(len);
  result.extend_from_slice(&chars[shift..]);
  result.extend_from_slice(&chars[..shift]);
  Ok(Expr::String(result.into_iter().collect()))
}

/// StringRotateRight[s] or StringRotateRight[s, n] - rotate characters right
pub fn string_rotate_right_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "StringRotateRight expects 1 or 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let n = if args.len() == 2 {
    expr_to_int(&args[1])?
  } else {
    1
  };

  // Rotate right by n = rotate left by -n
  let chars: Vec<char> = s.chars().collect();
  if chars.is_empty() {
    return Ok(Expr::String(s));
  }

  let len = chars.len();
  let shift = ((-(n % len as i128) % len as i128) + len as i128) as usize % len;
  let mut result: Vec<char> = Vec::with_capacity(len);
  result.extend_from_slice(&chars[shift..]);
  result.extend_from_slice(&chars[..shift]);
  Ok(Expr::String(result.into_iter().collect()))
}

/// AlphabeticSort[list] - case-insensitive alphabetic sort
pub fn alphabetic_sort_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AlphabeticSort expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      let mut sorted = items.clone();
      sorted.sort_by(|a, b| {
        let sa = crate::syntax::expr_to_string(a).to_lowercase();
        let sb = crate::syntax::expr_to_string(b).to_lowercase();
        sa.cmp(&sb)
      });
      Ok(Expr::List(sorted))
    }
    _ => Err(InterpreterError::EvaluationError(
      "AlphabeticSort expects a list argument".into(),
    )),
  }
}

/// Hash[s] or Hash[s, type] or Hash[s, type, format]
pub fn hash_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Hash expects 1, 2, or 3 arguments".into(),
    ));
  }

  let hash_type = if args.len() >= 2 {
    expr_to_str(&args[1])?
  } else {
    "Expression".to_string()
  };

  // For the default Expression hash, hash the expression's InputForm representation
  if hash_type == "Expression" {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let repr = crate::syntax::expr_to_string(&args[0]);
    let mut hasher = DefaultHasher::new();
    repr.hash(&mut hasher);
    let h = hasher.finish();
    return Ok(Expr::Integer(h as i128));
  }

  let s = expr_to_str(&args[0])?;

  let format = if args.len() == 3 {
    expr_to_str(&args[2])?
  } else {
    "Integer".to_string()
  };

  let hex_string = match hash_type.as_str() {
    "MD5" => {
      use md5::Digest;
      let result = md5::Md5::digest(s.as_bytes());
      result
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
    }
    "SHA" | "SHA1" => {
      use sha1::Digest;
      let result = sha1::Sha1::digest(s.as_bytes());
      result
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
    }
    "SHA256" => {
      use sha2::Digest;
      let result = sha2::Sha256::digest(s.as_bytes());
      result
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
    }
    "SHA384" => {
      use sha2::Digest;
      let result = sha2::Sha384::digest(s.as_bytes());
      result
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
    }
    "SHA512" => {
      use sha2::Digest;
      let result = sha2::Sha512::digest(s.as_bytes());
      result
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Hash".to_string(),
        args: args.to_vec(),
      });
    }
  };

  match format.as_str() {
    "HexString" => Ok(Expr::String(hex_string)),
    "Integer" | _ => {
      // Convert hex to big integer
      let n = num_bigint::BigInt::parse_bytes(hex_string.as_bytes(), 16)
        .unwrap_or_default();
      // Try to fit in i128
      use num_traits::ToPrimitive;
      if let Some(i) = n.to_i128() {
        Ok(Expr::Integer(i))
      } else {
        Ok(Expr::BigInteger(n))
      }
    }
  }
}

/// Compress[expr] — compresses an expression into a base64-encoded string
pub fn compress_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(format!(
      "Compress expects 1 argument, got {}",
      args.len()
    )));
  }
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let repr = crate::syntax::expr_to_string(&evaluated);

  use flate2::Compression;
  use flate2::write::ZlibEncoder;
  use std::io::Write;

  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  encoder.write_all(repr.as_bytes()).map_err(|e| {
    InterpreterError::EvaluationError(format!("Compress failed: {}", e))
  })?;
  let compressed = encoder.finish().map_err(|e| {
    InterpreterError::EvaluationError(format!("Compress failed: {}", e))
  })?;

  use base64::Engine;
  let encoded = base64::engine::general_purpose::STANDARD.encode(&compressed);
  Ok(Expr::String(format!("1:eJx{}", encoded)))
}

/// Uncompress[str] — decompresses a string produced by Compress
pub fn uncompress_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(format!(
      "Uncompress expects 1 argument, got {}",
      args.len()
    )));
  }
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let s = match &evaluated {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Uncompress".to_string(),
        args: vec![evaluated],
      });
    }
  };

  // Strip the version prefix "1:eJx"
  let data = if let Some(rest) = s.strip_prefix("1:eJx") {
    rest
  } else {
    return Err(InterpreterError::EvaluationError(
      "Uncompress: invalid compressed data".to_string(),
    ));
  };

  use base64::Engine;
  let decoded = base64::engine::general_purpose::STANDARD
    .decode(data)
    .map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Uncompress decode failed: {}",
        e
      ))
    })?;

  use flate2::read::ZlibDecoder;
  use std::io::Read;

  let mut decoder = ZlibDecoder::new(&decoded[..]);
  let mut decompressed = String::new();
  decoder.read_to_string(&mut decompressed).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Uncompress decompress failed: {}",
      e
    ))
  })?;

  // Parse the decompressed string back into an expression
  crate::syntax::string_to_expr(&decompressed).map_err(|e| {
    InterpreterError::EvaluationError(format!("Uncompress parse failed: {}", e))
  })
}

/// Extract text content for ReadList from the first argument.
/// Handles:
/// - StringToStream["text"] → returns the text
/// - "filename" → reads the file (CLI only)
fn readlist_get_text(source: &Expr) -> Result<String, InterpreterError> {
  match source {
    // StringToStream["text"] → use text directly
    Expr::FunctionCall { name, args }
      if name == "StringToStream" && args.len() == 1 =>
    {
      if let Expr::String(s) = &args[0] {
        Ok(s.clone())
      } else {
        Err(InterpreterError::EvaluationError(
          "StringToStream expects a string argument".into(),
        ))
      }
    }
    // InputStream[name, id] — look up in stream registry
    Expr::FunctionCall { name, args }
      if name == "InputStream" && args.len() == 2 =>
    {
      if let Expr::Integer(id) = &args[1] {
        let stream_id = *id as usize;
        crate::STREAM_REGISTRY.with(|reg| {
          let registry = reg.borrow();
          if let Some(stream) = registry.get(&stream_id) {
            match &stream.kind {
              crate::StreamKind::StringStream(text) => Ok(text.clone()),
              crate::StreamKind::FileStream(path) => {
                std::fs::read_to_string(path).map_err(|_| {
                  InterpreterError::EvaluationError(format!(
                    "ReadList::noopen: Cannot open {}.",
                    path
                  ))
                })
              }
            }
          } else {
            Err(InterpreterError::EvaluationError(
              "ReadList: stream is not open".into(),
            ))
          }
        })
      } else {
        Err(InterpreterError::EvaluationError(
          "ReadList: invalid stream object".into(),
        ))
      }
    }
    // String path → read file
    Expr::String(path) => std::fs::read_to_string(path).map_err(|_| {
      InterpreterError::EvaluationError(format!(
        "ReadList::noopen: Cannot open {}.",
        path
      ))
    }),
    _ => Err(InterpreterError::EvaluationError(
      "ReadList expects a filename string or StringToStream[\"text\"]".into(),
    )),
  }
}

/// ReadList[source] or ReadList[source, type] or ReadList[source, type, n]
pub fn read_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ReadList expects 1 to 3 arguments".into(),
    ));
  }

  let text = readlist_get_text(&args[0])?;

  // Determine the read type (default: Expression)
  let read_type = if args.len() >= 2 {
    &args[1]
  } else {
    &Expr::Identifier("Expression".to_string())
  };

  // Optional max count
  let max_count: Option<usize> = if args.len() == 3 {
    if let Expr::Integer(n) = &args[2] {
      Some(*n as usize)
    } else {
      None
    }
  } else {
    None
  };

  // Handle composite types like {Word, Word}
  if let Expr::List(types) = read_type {
    return read_list_record(&text, types, max_count);
  }

  let type_name = match read_type {
    Expr::Identifier(s) => s.as_str(),
    _ => "Expression",
  };

  let mut results = Vec::new();

  match type_name {
    "String" => {
      // Each line is a string
      for line in text.lines() {
        if let Some(max) = max_count
          && results.len() >= max
        {
          break;
        }
        results.push(Expr::String(line.to_string()));
      }
    }
    "Word" => {
      // Whitespace-separated words (returned as strings)
      for word in text.split_whitespace() {
        if let Some(max) = max_count
          && results.len() >= max
        {
          break;
        }
        results.push(Expr::String(word.to_string()));
      }
    }
    "Number" => {
      // Whitespace-separated numbers
      for token in text.split_whitespace() {
        if let Some(max) = max_count
          && results.len() >= max
        {
          break;
        }
        if let Ok(n) = token.parse::<i128>() {
          results.push(Expr::Integer(n));
        } else if let Ok(f) = token.parse::<f64>() {
          results.push(Expr::Real(f));
        }
        // Skip non-numeric tokens
      }
    }
    "Expression" | _ => {
      // Parse and evaluate each line as a Wolfram expression
      for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
          continue;
        }
        if let Some(max) = max_count
          && results.len() >= max
        {
          break;
        }
        match crate::syntax::string_to_expr(trimmed) {
          Ok(expr) => match crate::evaluator::evaluate_expr_to_expr(&expr) {
            Ok(val) => results.push(val),
            Err(_) => results.push(expr),
          },
          Err(_) => {
            // If parsing fails, skip
          }
        }
      }
    }
  }

  Ok(Expr::List(results))
}

/// Handle ReadList with record types like {Word, Word}
fn read_list_record(
  text: &str,
  types: &[Expr],
  max_count: Option<usize>,
) -> Result<Expr, InterpreterError> {
  let mut results = Vec::new();

  for line in text.lines() {
    let trimmed = line.trim();
    if trimmed.is_empty() {
      continue;
    }
    if let Some(max) = max_count
      && results.len() >= max
    {
      break;
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let mut record = Vec::new();

    for (i, type_spec) in types.iter().enumerate() {
      let type_name = match type_spec {
        Expr::Identifier(s) => s.as_str(),
        _ => "Expression",
      };
      let token = tokens.get(i).unwrap_or(&"");
      match type_name {
        "Word" => record.push(Expr::String(token.to_string())),
        "Number" => {
          if let Ok(n) = token.parse::<i128>() {
            record.push(Expr::Integer(n));
          } else if let Ok(f) = token.parse::<f64>() {
            record.push(Expr::Real(f));
          } else {
            record.push(Expr::Identifier(token.to_string()));
          }
        }
        "String" => record.push(Expr::String(token.to_string())),
        _ => {
          if let Ok(expr) = crate::syntax::string_to_expr(token) {
            if let Ok(val) = crate::evaluator::evaluate_expr_to_expr(&expr) {
              record.push(val);
            } else {
              record.push(expr);
            }
          }
        }
      }
    }

    results.push(Expr::List(record));
  }

  Ok(Expr::List(results))
}

/// Convert an expression to C language format
pub fn expr_to_c(expr: &Expr) -> String {
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::Real(f) => {
      let s = format!("{}", f);
      // Ensure decimal point
      if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        format!("{}.", s)
      } else {
        s
      }
    }
    Expr::String(s) => format!("\"{}\"", s),
    Expr::Identifier(name) => name.clone(),
    Expr::Constant(name) => name.clone(),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let parts: Vec<String> = args.iter().map(expr_to_c).collect();
        parts.join(" + ")
      }
      "Times" => {
        let parts: Vec<String> = args
          .iter()
          .map(|a| {
            let s = expr_to_c(a);
            if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Plus,
                  ..
                }
              )
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Minus,
                  ..
                }
              )
            {
              format!("({})", s)
            } else {
              s
            }
          })
          .collect();
        parts.join("*")
      }
      "Power" if args.len() == 2 => {
        if matches!(&args[1], Expr::Integer(-1)) {
          format!("1/{}", c_paren(&args[0]))
        } else if matches!(&args[1], Expr::FunctionCall { name, args: ra } if name == "Rational" && ra.len() == 2 && matches!(&ra[0], Expr::Integer(1)) && matches!(&ra[1], Expr::Integer(2)))
        {
          format!("Sqrt({})", expr_to_c(&args[0]))
        } else {
          format!("Power({},{})", expr_to_c(&args[0]), expr_to_c(&args[1]))
        }
      }
      "Rational" if args.len() == 2 => {
        format!("{}./{}", expr_to_c(&args[0]), c_paren(&args[1]))
      }
      _ => format!("{}({})", name, c_args(args)),
    },
    Expr::BinaryOp { op, left, right } => {
      let l = expr_to_c(left);
      let r = expr_to_c(right);
      match op {
        BinaryOperator::Plus => format!("{} + {}", l, r),
        BinaryOperator::Minus => format!("{} - {}", l, r),
        BinaryOperator::Times => format!("{}*{}", l, r),
        BinaryOperator::Divide => format!("{}/{}", l, r),
        BinaryOperator::Power => {
          if matches!(right.as_ref(), Expr::Integer(-1)) {
            format!("1/{}", c_paren(left))
          } else if matches!(right.as_ref(), Expr::FunctionCall { name, args: ra } if name == "Rational" && ra.len() == 2 && matches!(&ra[0], Expr::Integer(1)) && matches!(&ra[1], Expr::Integer(2)))
          {
            format!("Sqrt({})", expr_to_c(left))
          } else {
            format!("Power({},{})", l, r)
          }
        }
        _ => format!("{}({})", format!("{:?}", op), format!("{},{}", l, r)),
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      format!("-{}", c_paren(operand))
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_c).collect();
      format!("List({})", parts.join(","))
    }
    // Rational numbers are FunctionCall{name:"Rational", args:[num, den]}
    // but they get evaluated before reaching here, so this pattern is rare
    _ => crate::syntax::expr_to_string(expr),
  }
}

/// Add parentheses for C output if needed
fn c_paren(expr: &Expr) -> String {
  let s = expr_to_c(expr);
  match expr {
    Expr::FunctionCall { name, .. } if name == "Plus" => format!("({})", s),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      ..
    } => format!("({})", s),
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      ..
    } => format!("({})", s),
    _ => s,
  }
}

/// Format C function arguments
fn c_args(args: &[Expr]) -> String {
  args.iter().map(expr_to_c).collect::<Vec<_>>().join(",")
}

/// Convert an expression to Fortran language format
pub fn expr_to_fortran(expr: &Expr) -> String {
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::Real(f) => {
      let s = format!("{}", f);
      // Ensure decimal point
      if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        format!("{}.", s)
      } else {
        s
      }
    }
    Expr::String(s) => format!("\"{}\"", s),
    Expr::Identifier(name) => name.clone(),
    Expr::Constant(name) => name.clone(),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let parts: Vec<String> = args.iter().map(expr_to_fortran).collect();
        parts.join(" + ")
      }
      "Times" => {
        // Handle Times[-1, x] as -x
        if args.len() == 2 && matches!(&args[0], Expr::Integer(-1)) {
          return format!("-{}", fortran_paren(&args[1]));
        }
        // Handle fraction form: Times[..., Power[den, -1]]
        let (num, den) =
          crate::functions::polynomial_ast::together::extract_num_den(expr);
        if !matches!(&den, Expr::Integer(1)) {
          let num_str = expr_to_fortran(&num);
          let den_str = expr_to_fortran(&den);
          return format!("{}/{}", num_str, den_str);
        }
        let parts: Vec<String> = args
          .iter()
          .map(|a| {
            let s = expr_to_fortran(a);
            if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Plus,
                  ..
                }
              )
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Minus,
                  ..
                }
              )
            {
              format!("({})", s)
            } else {
              s
            }
          })
          .collect();
        parts.join("*")
      }
      "Power" if args.len() == 2 => {
        if matches!(&args[1], Expr::FunctionCall { name, args: ra } if name == "Rational" && ra.len() == 2 && matches!(&ra[0], Expr::Integer(1)) && matches!(&ra[1], Expr::Integer(2)))
        {
          format!("Sqrt({})", expr_to_fortran(&args[0]))
        } else {
          format!("{}**{}", fortran_paren(&args[0]), fortran_paren(&args[1]))
        }
      }
      "Rational" if args.len() == 2 => {
        // Wolfram FortranForm evaluates rationals to decimal
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          let val = *n as f64 / *d as f64;
          let s = format!("{}", val);
          // Ensure decimal point
          if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            format!("{}.", s)
          } else {
            s
          }
        } else {
          format!("{}./{}", expr_to_fortran(&args[0]), fortran_paren(&args[1]))
        }
      }
      _ => format!("{}({})", name, fortran_args(args)),
    },
    Expr::BinaryOp { op, left, right } => {
      // Power[x, Rational[1, 2]] → Sqrt(x) in Fortran
      if matches!(op, BinaryOperator::Power)
        && let Some(sqrt_arg) = crate::functions::is_sqrt(expr)
      {
        return format!("Sqrt({})", expr_to_fortran(sqrt_arg));
      }
      let l = expr_to_fortran(left);
      let r = expr_to_fortran(right);
      match op {
        BinaryOperator::Plus => format!("{} + {}", l, r),
        BinaryOperator::Minus => format!("{} - {}", l, r),
        BinaryOperator::Times => format!("{}*{}", l, r),
        BinaryOperator::Divide => format!("{}/{}", l, r),
        BinaryOperator::Power => format!("{}**{}", l, r),
        _ => format!("{}({})", format!("{:?}", op), format!("{},{}", l, r)),
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      format!("-{}", fortran_paren(operand))
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_fortran).collect();
      format!("List({})", parts.join(","))
    }
    _ => crate::syntax::expr_to_string(expr),
  }
}

/// Add parentheses for Fortran output if needed
fn fortran_paren(expr: &Expr) -> String {
  let s = expr_to_fortran(expr);
  match expr {
    Expr::FunctionCall { name, .. } if name == "Plus" => format!("({})", s),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      ..
    } => format!("({})", s),
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      ..
    } => format!("({})", s),
    Expr::Integer(n) if *n < 0 => format!("({})", s),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => format!("({})", s),
    _ => s,
  }
}

/// Format Fortran function arguments
fn fortran_args(args: &[Expr]) -> String {
  args
    .iter()
    .map(expr_to_fortran)
    .collect::<Vec<_>>()
    .join(",")
}

/// TemplateApply[template, args] - Apply arguments to a string template.
/// Replaces `n` with the nth argument (1-indexed) from a list,
/// or `key` with the value from an association.
pub fn template_apply_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "TemplateApply".to_string(),
      args: args.to_vec(),
    });
  }

  let template = match &args[0] {
    Expr::String(s) => s.clone(),
    // Non-string template: if second arg is a list, apply like TemplateApply[expr, {args}]
    other => {
      // For non-string first arg, return as-is (like wolframscript returns 42 for TemplateApply[42, {1}])
      return Ok(other.clone());
    }
  };

  // Build replacement map
  let replacements: std::collections::HashMap<String, String> = match &args[1] {
    Expr::List(items) => {
      let mut map = std::collections::HashMap::new();
      for (i, item) in items.iter().enumerate() {
        let value_str = match item {
          Expr::String(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        map.insert((i + 1).to_string(), value_str);
      }
      map
    }
    Expr::Association(pairs) => {
      let mut map = std::collections::HashMap::new();
      for (k, v) in pairs {
        let key = match k {
          Expr::String(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        let value = match v {
          Expr::String(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        map.insert(key, value);
      }
      map
    }
    Expr::FunctionCall {
      name,
      args: assoc_args,
    } if name == "Association" => {
      let mut map = std::collections::HashMap::new();
      for arg in assoc_args {
        let (key_expr, val_expr): (&Expr, &Expr) = match arg {
          Expr::Rule {
            pattern,
            replacement,
          } => (pattern.as_ref(), replacement.as_ref()),
          Expr::RuleDelayed {
            pattern,
            replacement,
          } => (pattern.as_ref(), replacement.as_ref()),
          Expr::FunctionCall {
            name: rule_name,
            args: rule_args,
          } if (rule_name == "Rule" || rule_name == "RuleDelayed")
            && rule_args.len() == 2 =>
          {
            (&rule_args[0], &rule_args[1])
          }
          _ => continue,
        };
        let key = match key_expr {
          Expr::String(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        let value = match val_expr {
          Expr::String(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        };
        map.insert(key, value);
      }
      map
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TemplateApply".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Replace `key` or `` (positional) patterns in the template
  let mut result = String::new();
  let mut chars = template.chars().peekable();
  let mut positional_index: usize = 1; // counter for `` positional slots
  while let Some(ch) = chars.next() {
    if ch == '`' {
      // Read until closing backtick
      let mut key = String::new();
      loop {
        match chars.next() {
          Some('`') => break,
          Some(c) => key.push(c),
          None => {
            // No closing backtick - include literally
            result.push('`');
            result.push_str(&key);
            return Ok(Expr::String(result));
          }
        }
      }
      // Empty key `` means positional slot (1st, 2nd, etc.)
      let lookup_key = if key.is_empty() {
        let k = positional_index.to_string();
        positional_index += 1;
        k
      } else {
        key.clone()
      };
      if let Some(replacement) = replacements.get(&lookup_key) {
        result.push_str(replacement);
      } else {
        // No replacement found - keep the slot
        result.push('`');
        result.push_str(&key);
        result.push('`');
      }
    } else {
      result.push(ch);
    }
  }

  Ok(Expr::String(result))
}

/// StringPartition[string, n] partitions string into non-overlapping substrings of length n.
/// StringPartition[string, n, d] partitions with offset d.
pub fn string_partition_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])? as usize;
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "Partition size must be positive.".to_string(),
    ));
  }
  let d = if args.len() >= 3 {
    expr_to_int(&args[2])? as usize
  } else {
    n
  };
  if d == 0 {
    return Err(InterpreterError::EvaluationError(
      "Offset must be positive.".to_string(),
    ));
  }

  let chars: Vec<char> = s.chars().collect();
  let mut parts = Vec::new();
  let mut i = 0;
  while i + n <= chars.len() {
    let part: String = chars[i..i + n].iter().collect();
    parts.push(Expr::String(part));
    i += d;
  }
  Ok(Expr::List(parts))
}

// ─── DictionaryWordQ ──────────────────────────────────────────────

use std::sync::LazyLock;

static DICTIONARY_WORDS: LazyLock<std::collections::HashSet<String>> =
  LazyLock::new(|| {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let compressed = include_bytes!("../../resources/dictionary_words.txt.gz");
    let mut decoder = GzDecoder::new(&compressed[..]);
    let mut text = String::new();
    decoder
      .read_to_string(&mut text)
      .expect("Failed to decompress dictionary");

    text.lines().map(|line| line.to_lowercase()).collect()
  });

/// DictionaryWordQ[string] - True if string is a dictionary word
pub fn dictionary_word_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DictionaryWordQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::String(s) => {
      let result = s.is_empty() || DICTIONARY_WORDS.contains(&s.to_lowercase());
      Ok(Expr::Identifier(
        if result { "True" } else { "False" }.to_string(),
      ))
    }
    _ => Ok(Expr::FunctionCall {
      name: "DictionaryWordQ".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// URLEncode[string] - percent-encode a string for use in URLs
pub fn url_encode_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  match &evaluated {
    Expr::String(s) => Ok(Expr::String(percent_encode(s))),
    Expr::Integer(n) => Ok(Expr::String(n.to_string())),
    Expr::Real(f) => Ok(Expr::String(format!("{}", f))),
    Expr::BigFloat(digits, _) => Ok(Expr::String(digits.clone())),
    Expr::Identifier(id) if id == "None" => Ok(Expr::String(String::new())),
    Expr::FunctionCall { name, .. } if name == "Missing" => {
      Ok(Expr::String(String::new()))
    }
    Expr::List(items) => {
      // Thread over lists
      let encoded: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| url_encode_ast(&[item.clone()]))
        .collect();
      Ok(Expr::List(encoded?))
    }
    _ => Ok(Expr::FunctionCall {
      name: "URLEncode".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// URLDecode[string] - decode a percent-encoded URL string
pub fn url_decode_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  match &evaluated {
    Expr::String(s) => Ok(Expr::String(percent_decode(s))),
    Expr::List(items) => {
      let decoded: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| url_decode_ast(&[item.clone()]))
        .collect();
      Ok(Expr::List(decoded?))
    }
    _ => Ok(Expr::FunctionCall {
      name: "URLDecode".to_string(),
      args: args.to_vec(),
    }),
  }
}

fn percent_encode(s: &str) -> String {
  let mut result = String::with_capacity(s.len() * 3);
  for byte in s.as_bytes() {
    match *byte {
      // Unreserved characters (RFC 3986): ALPHA / DIGIT / "-" / "." / "_" / "~"
      b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
        result.push(*byte as char);
      }
      _ => {
        result.push('%');
        result.push_str(&format!("{:02X}", byte));
      }
    }
  }
  result
}

fn percent_decode(s: &str) -> String {
  let mut result = Vec::new();
  let bytes = s.as_bytes();
  let mut i = 0;
  while i < bytes.len() {
    if bytes[i] == b'%'
      && i + 2 < bytes.len()
      && let Ok(byte) = u8::from_str_radix(&s[i + 1..i + 3], 16)
    {
      result.push(byte);
      i += 3;
      continue;
    }
    result.push(bytes[i]);
    i += 1;
  }
  String::from_utf8_lossy(&result).to_string()
}

/// StringToByteArray[string] - convert a string to a ByteArray (UTF-8 encoding)
/// StringToByteArray[string, encoding] - with specified encoding (only UTF-8 supported)
pub fn string_to_byte_array_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "StringToByteArray expects 1 or 2 arguments".into(),
    ));
  }
  let s = match &args[0] {
    Expr::String(s) => s,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StringToByteArray".to_string(),
        args: args.to_vec(),
      });
    }
  };
  // Encode as UTF-8 bytes and create ByteArray
  let bytes = s.as_bytes();
  use base64::Engine;
  let engine = base64::engine::general_purpose::STANDARD;
  let b64 = engine.encode(bytes);
  Ok(Expr::FunctionCall {
    name: "ByteArray".to_string(),
    args: vec![Expr::String(b64)],
  })
}

/// ByteArrayToString[bytearray] - convert a ByteArray to a string (UTF-8 decoding)
/// ByteArrayToString[bytearray, encoding] - with specified encoding (only UTF-8 supported)
pub fn byte_array_to_string_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ByteArrayToString expects 1 or 2 arguments".into(),
    ));
  }
  // Extract bytes from ByteArray
  if let Expr::FunctionCall {
    name,
    args: ba_args,
  } = &args[0]
    && name == "ByteArray"
    && ba_args.len() == 1
  {
    let bytes: Vec<u8> = match &ba_args[0] {
      Expr::String(b64) => {
        use base64::Engine;
        let engine = base64::engine::general_purpose::STANDARD;
        engine.decode(b64).unwrap_or_default()
      }
      Expr::List(items) => items
        .iter()
        .filter_map(|item| {
          if let Expr::Integer(n) = item {
            Some(*n as u8)
          } else {
            None
          }
        })
        .collect(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "ByteArrayToString".to_string(),
          args: args.to_vec(),
        });
      }
    };
    return Ok(Expr::String(String::from_utf8_lossy(&bytes).to_string()));
  }
  Ok(Expr::FunctionCall {
    name: "ByteArrayToString".to_string(),
    args: args.to_vec(),
  })
}
