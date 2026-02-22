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

  // Multiple arguments - join them all
  for arg in args {
    joined.push_str(&expr_to_str(arg)?);
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

  // Extract options from args[2..]
  let ignore_case = extract_ignore_case(&args[2..]);

  // Check if delimiter is a RegularExpression
  if let Some(pat) = extract_regex_pattern(&args[1]) {
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
    let parts: Vec<Expr> = re
      .split(&s)
      .filter(|p| !p.is_empty())
      .map(|p| Expr::String(p.to_string()))
      .collect();
    return Ok(Expr::List(parts));
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

  let parts: Vec<Expr> = if delims.len() == 1 && delims[0].is_empty() {
    s.chars().map(|c| Expr::String(c.to_string())).collect()
  } else if delims.len() == 1 {
    if ignore_case {
      let re = regex::RegexBuilder::new(&regex::escape(&delims[0]))
        .case_insensitive(true)
        .build()
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("Regex error: {}", e))
        })?;
      re.split(&s)
        .filter(|p| !p.is_empty())
        .map(|p| Expr::String(p.to_string()))
        .collect()
    } else {
      s.split(&delims[0])
        .filter(|p| !p.is_empty())
        .map(|p| Expr::String(p.to_string()))
        .collect()
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
  Ok(Expr::List(parts))
}

/// StringStartsQ[s, prefix] - checks if string starts with prefix
pub fn string_starts_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringStartsQ expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let prefix = expr_to_str(&args[1])?;
  Ok(Expr::Identifier(
    if s.starts_with(&prefix) {
      "True"
    } else {
      "False"
    }
    .to_string(),
  ))
}

/// StringEndsQ[s, suffix] - checks if string ends with suffix
pub fn string_ends_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringEndsQ expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let suffix = expr_to_str(&args[1])?;
  Ok(Expr::Identifier(
    if s.ends_with(&suffix) {
      "True"
    } else {
      "False"
    }
    .to_string(),
  ))
}

/// StringContainsQ[s, sub] - checks if string contains substring
pub fn string_contains_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringContainsQ expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let sub = expr_to_str(&args[1])?;
  Ok(Expr::Identifier(
    if s.contains(&sub) { "True" } else { "False" }.to_string(),
  ))
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

  fn extract_rule(expr: &Expr) -> Result<(String, String), InterpreterError> {
    match expr {
      Expr::Rule {
        pattern,
        replacement,
      } => Ok((expr_to_str(pattern)?, expr_to_str(replacement)?)),
      Expr::RuleDelayed {
        pattern,
        replacement,
      } => Ok((expr_to_str(pattern)?, expr_to_str(replacement)?)),
      _ => Err(InterpreterError::EvaluationError(
        "StringReplace: rules must be of the form pattern -> replacement"
          .into(),
      )),
    }
  }

  // Collect all rules into a vec
  let rules: Vec<(String, String)> = match &args[1] {
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
    rules: &[(String, String)],
    max: Option<usize>,
  ) -> String {
    let mut result = String::new();
    let mut count = 0usize;
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < s.len() {
      if max.is_some() && count >= max.unwrap() {
        // Limit reached, append the rest
        result.push_str(&s[i..]);
        break;
      }
      let mut matched = false;
      for (pattern, replacement) in rules {
        if pattern.is_empty() {
          continue;
        }
        if i + pattern.len() <= bytes.len()
          && &s[i..i + pattern.len()] == pattern.as_str()
        {
          result.push_str(replacement);
          i += pattern.len();
          count += 1;
          matched = true;
          break;
        }
      }
      if !matched {
        // Advance by one character
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
  let s = expr_to_str(&args[0])?;
  let chars: Vec<Expr> =
    s.chars().map(|c| Expr::String(c.to_string())).collect();
  Ok(Expr::List(chars))
}

/// StringRiffle[list] or StringRiffle[list, sep] or StringRiffle[list, {left, sep, right}]
pub fn string_riffle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "StringRiffle expects 1 or 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "StringRiffle: first argument must be a list".into(),
      ));
    }
  };

  let strings: Result<Vec<String>, _> = items.iter().map(expr_to_str).collect();
  let strings = strings?;

  if args.len() == 2 {
    // Check for {left, sep, right} form
    if let Expr::List(sep_parts) = &args[1]
      && sep_parts.len() == 3
    {
      let left = expr_to_str(&sep_parts[0])?;
      let sep = expr_to_str(&sep_parts[1])?;
      let right = expr_to_str(&sep_parts[2])?;
      return Ok(Expr::String(format!(
        "{}{}{}",
        left,
        strings.join(&sep),
        right
      )));
    }
    let sep = expr_to_str(&args[1])?;
    Ok(Expr::String(strings.join(&sep)))
  } else {
    Ok(Expr::String(strings.join(" ")))
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
pub fn string_match_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringMatchQ expects exactly 2 arguments".into(),
    ));
  }

  let s = expr_to_str(&args[0])?;

  // Try pattern-based matching (DigitCharacter, LetterCharacter, Repeated, etc.)
  if let Some(regex_str) = string_pattern_to_regex(&args[1]) {
    let full_regex = format!("^(?:{})$", regex_str);
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
    let full_regex = format!("^(?:{})$", pat);
    let re = regex::Regex::new(&full_regex).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid regex: {}", e))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "True" } else { "False" }.to_string(),
    ));
  }

  // Fall back to string-based wildcard matching
  let pattern = expr_to_str(&args[1])?;
  let matches = wildcard_match(&s, &pattern);
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
  let s = expr_to_str(&args[0])?;

  if args.len() == 1 {
    Ok(Expr::String(s.trim().to_string()))
  } else {
    let patt = expr_to_str(&args[1])?;
    let trimmed = s.trim_start_matches(&patt).trim_end_matches(&patt);
    Ok(Expr::String(trimmed.to_string()))
  }
}

/// Convert a Wolfram string pattern expression to a regex pattern string.
/// Returns None if the expression is not a recognized string pattern.
fn string_pattern_to_regex(expr: &Expr) -> Option<String> {
  match expr {
    // String literal patterns
    Expr::String(s) => Some(regex::escape(s)),

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
      parts.map(|ps| ps.join("|"))
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
      if name == "Repeated" && args.len() == 1 =>
    {
      string_pattern_to_regex(&args[0]).map(|r| format!("(?:{})+", r))
    }

    // RepeatedNull[pat] = pat... (zero or more)
    Expr::FunctionCall { name, args }
      if name == "RepeatedNull" && args.len() == 1 =>
    {
      string_pattern_to_regex(&args[0]).map(|r| format!("(?:{})*", r))
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

  // Check for form argument
  if args.len() == 2
    && let Expr::Identifier(form) = &args[1]
  {
    match form.as_str() {
      "InputForm" => {
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
  // Uses expr_to_output which renders strings without quotes and handles
  // display forms like FullForm[expr] → FullForm notation
  let s = crate::syntax::expr_to_output(&args[0]);
  Ok(Expr::String(s))
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

/// Handle multiplication in TeX (space-separated)
fn tex_times(left: &Expr, right: &Expr) -> String {
  let l = expr_to_tex(left);
  let r = expr_to_tex(right);

  // -1 * x → -x
  if matches!(left, Expr::Integer(-1)) {
    return format!("-{}", r);
  }

  format!("{} {}", l, r)
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

  let base_tex = tex_base_with_parens(base);
  let exp_tex = expr_to_tex(exp);

  format!("{}^{{{}}}", base_tex, exp_tex)
}

/// Wrap base in parens if needed for power
fn tex_base_with_parens(base: &Expr) -> String {
  match base {
    Expr::BinaryOp { .. } | Expr::UnaryOp { .. } => {
      format!("\\left({}\\right)", expr_to_tex(base))
    }
    _ => expr_to_tex(base),
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
      format!("\\left| {} \\right|", expr_to_tex(&args[0]))
    }
    // Rational
    "Rational" if args.len() == 2 => {
      format!(
        "\\frac{{{}}}{{{}}}",
        expr_to_tex(&args[0]),
        expr_to_tex(&args[1])
      )
    }
    // Plus (n-ary)
    "Plus" if !args.is_empty() => {
      let mut result = expr_to_tex(&args[0]);
      for arg in args.iter().skip(1) {
        let t = expr_to_tex(arg);
        if t.starts_with('-') {
          result.push_str(&t);
        } else {
          result.push('+');
          result.push_str(&t);
        }
      }
      result
    }
    // Times (n-ary)
    "Times" if args.len() >= 2 => {
      // Check for -1 factor
      if matches!(&args[0], Expr::Integer(-1)) {
        let rest: Vec<String> = args[1..].iter().map(expr_to_tex).collect();
        return format!("-{}", rest.join(" "));
      }
      let parts: Vec<String> = args.iter().map(expr_to_tex).collect();
      parts.join(" ")
    }
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

  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringPadLeft must be non-negative".into(),
    ));
  }

  let target_len = n as usize;
  let pad_char = if args.len() == 3 {
    let pad_str = expr_to_str(&args[2])?;
    if pad_str.is_empty() {
      ' '
    } else {
      pad_str.chars().next().unwrap()
    }
  } else {
    ' '
  };

  let char_count = s.chars().count();
  if char_count >= target_len {
    Ok(Expr::String(
      s.chars().skip(char_count - target_len).collect(),
    ))
  } else {
    let padding: String =
      std::iter::repeat_n(pad_char, target_len - char_count).collect();
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

  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringPadRight must be non-negative".into(),
    ));
  }

  let target_len = n as usize;
  let pad_char = if args.len() == 3 {
    let pad_str = expr_to_str(&args[2])?;
    if pad_str.is_empty() {
      ' '
    } else {
      pad_str.chars().next().unwrap()
    }
  } else {
    ' '
  };

  let char_count = s.chars().count();
  if char_count >= target_len {
    Ok(Expr::String(s.chars().take(target_len).collect()))
  } else {
    let padding: String =
      std::iter::repeat_n(pad_char, target_len - char_count).collect();
    Ok(Expr::String(format!("{}{}", s, padding)))
  }
}

/// StringCount[s, sub] - count occurrences of substring
pub fn string_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringCount expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let sub = expr_to_str(&args[1])?;

  if sub.is_empty() {
    return Ok(Expr::Integer(0));
  }

  Ok(Expr::Integer(s.matches(&sub).count() as i128))
}

/// StringFreeQ[s, sub] - check if string does NOT contain substring
pub fn string_free_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringFreeQ expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
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

// ─── StringDelete ──────────────────────────────────────────────────

/// StringDelete[string, sub] - deletes all occurrences of sub from string
pub fn string_delete_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringDelete expects exactly 2 arguments".into(),
    ));
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

/// LongestCommonSubsequence[s1, s2] - longest common contiguous substring
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

  let mut max_len = 0usize;
  let mut end_idx = 0usize;

  // DP table for longest common substring
  let mut dp = vec![vec![0usize; m + 1]; n + 1];
  for i in 1..=n {
    for j in 1..=m {
      if chars1[i - 1] == chars2[j - 1] {
        dp[i][j] = dp[i - 1][j - 1] + 1;
        if dp[i][j] > max_len {
          max_len = dp[i][j];
          end_idx = i;
        }
      }
    }
  }

  let result: String = chars1[end_idx - max_len..end_idx].iter().collect();
  Ok(Expr::String(result))
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

  let s = expr_to_str(&args[0])?;

  let hash_type = if args.len() >= 2 {
    expr_to_str(&args[1])?
  } else {
    // Default hash type — not replicable, return unevaluated
    return Ok(Expr::FunctionCall {
      name: "Hash".to_string(),
      args: args.to_vec(),
    });
  };

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
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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
      // Whitespace-separated words
      for word in text.split_whitespace() {
        if let Some(max) = max_count
          && results.len() >= max
        {
          break;
        }
        results.push(Expr::Identifier(word.to_string()));
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
#[cfg(not(target_arch = "wasm32"))]
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
        "Word" => record.push(Expr::Identifier(token.to_string())),
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
    Expr::Constant(name) => match name.as_str() {
      "Pi" => "M_PI".to_string(),
      "E" => "M_E".to_string(),
      _ => name.clone(),
    },
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
          format!("1./{}", c_paren(&args[0]))
        } else if matches!(&args[1], Expr::FunctionCall { name, args: ra } if name == "Rational" && ra.len() == 2 && matches!(&ra[0], Expr::Integer(1)) && matches!(&ra[1], Expr::Integer(2)))
        {
          format!("sqrt({})", expr_to_c(&args[0]))
        } else {
          format!("pow({},{})", expr_to_c(&args[0]), expr_to_c(&args[1]))
        }
      }
      "Sin" => format!("sin({})", c_args(args)),
      "Cos" => format!("cos({})", c_args(args)),
      "Tan" => format!("tan({})", c_args(args)),
      "Exp" => format!("exp({})", c_args(args)),
      "Log" if args.len() == 1 => format!("log({})", c_args(args)),
      "Log" if args.len() == 2 => {
        format!("log({})/log({})", expr_to_c(&args[1]), expr_to_c(&args[0]))
      }
      "Sqrt" => format!("sqrt({})", c_args(args)),
      "Abs" => format!("abs({})", c_args(args)),
      "Floor" => format!("floor({})", c_args(args)),
      "Ceiling" => format!("ceil({})", c_args(args)),
      "ArcSin" => format!("asin({})", c_args(args)),
      "ArcCos" => format!("acos({})", c_args(args)),
      "ArcTan" if args.len() == 1 => format!("atan({})", c_args(args)),
      "ArcTan" if args.len() == 2 => {
        format!("atan2({},{})", expr_to_c(&args[1]), expr_to_c(&args[0]))
      }
      "Sinh" => format!("sinh({})", c_args(args)),
      "Cosh" => format!("cosh({})", c_args(args)),
      "Tanh" => format!("tanh({})", c_args(args)),
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
        BinaryOperator::Power => format!("pow({},{})", l, r),
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
