//! AST-native string functions.
//!
//! These functions work directly with `Expr` AST nodes, avoiding string round-trips.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Helper to extract a string from an Expr
fn expr_to_str(expr: &Expr) -> Result<String, InterpreterError> {
  match expr {
    Expr::String(s) => Ok(s.clone()),
    Expr::Identifier(s) => Ok(s.clone()),
    Expr::Integer(n) => Ok(n.to_string()),
    Expr::BigInteger(n) => Ok(n.to_string()),
    Expr::Real(f) => Ok(crate::format_result(*f)),
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
  let s = expr_to_str(&args[0])?;
  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;

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
pub fn string_drop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringDrop expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;
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

/// StringSplit[s] - splits by whitespace; StringSplit[s, delim] - splits by delimiter
pub fn string_split_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "StringSplit expects 1 or 2 arguments".into(),
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
    s.split(&delims[0])
      .map(|p| Expr::String(p.to_string()))
      .collect()
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
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringReplace expects exactly 2 arguments".into(),
    ));
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

  match &args[1] {
    Expr::List(rules) => {
      let mut result = s;
      for rule in rules {
        let (pattern, replacement) = extract_rule(rule)?;
        result = result.replace(&pattern, &replacement);
      }
      Ok(Expr::String(result))
    }
    rule => {
      let (pattern, replacement) = extract_rule(rule)?;
      Ok(Expr::String(s.replace(&pattern, &replacement)))
    }
  }
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
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringPosition expects exactly 2 arguments".into(),
    ));
  }

  let s = expr_to_str(&args[0])?;
  let sub = expr_to_str(&args[1])?;

  if sub.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let mut positions = Vec::new();
  let s_chars: Vec<char> = s.chars().collect();
  let sub_chars: Vec<char> = sub.chars().collect();

  for i in 0..=s_chars.len().saturating_sub(sub_chars.len()) {
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
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringRepeat expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringRepeat must be non-negative".into(),
    ));
  }
  Ok(Expr::String(s.repeat(n as usize)))
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

    // Character class patterns
    Expr::Identifier(name) => match name.as_str() {
      "DigitCharacter" => Some("[0-9]".to_string()),
      "LetterCharacter" => Some("[a-zA-Z]".to_string()),
      "WhitespaceCharacter" => Some("\\s".to_string()),
      "WordCharacter" => Some("[a-zA-Z0-9]".to_string()),
      "HexadecimalCharacter" => Some("[0-9a-fA-F]".to_string()),
      _ => None,
    },

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

  // Check for InputForm as second argument
  if args.len() == 2
    && let Expr::Identifier(form) = &args[1]
    && form == "InputForm"
  {
    // InputForm: infix operators + quoted strings
    let s = crate::syntax::expr_to_input_form(&args[0]);
    return Ok(Expr::String(s));
  }
  // Other forms: fall through to default (OutputForm-like) behavior

  // Default (no form or unrecognized form): OutputForm-like
  // Uses expr_to_output which renders strings without quotes and handles
  // display forms like FullForm[expr] → FullForm notation
  let s = crate::syntax::expr_to_output(&args[0]);
  Ok(Expr::String(s))
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

  let is_negative = n < 0;
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

  if is_negative {
    result.insert(0, '-');
  }

  // If length is specified, pad with zeros on the left
  if args.len() == 3 {
    let target_len = expr_to_int(&args[2])? as usize;
    let current_len = result.len();
    if current_len < target_len {
      let padding: String =
        std::iter::repeat_n('0', target_len - current_len).collect();
      result = format!("{}{}", padding, result);
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
      let result = !s.is_empty()
        && s.chars().all(|c| c.is_alphabetic() && c.is_uppercase());
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
      let result = !s.is_empty()
        && s.chars().all(|c| c.is_alphabetic() && c.is_lowercase());
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
  let s = expr_to_str(&args[0])?;
  let insert = expr_to_str(&args[1])?;
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
