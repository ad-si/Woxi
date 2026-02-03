use pest::iterators::Pair;

use crate::{InterpreterError, Rule, evaluate_term, extract_string};

/// Handle StringLength[s] - returns the length of a string
pub fn string_length(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StringLength expects exactly 1 argument".into(),
    ));
  }
  let s = extract_string(args_pairs[0].clone())?;
  Ok(s.chars().count().to_string())
}

/// Handle StringTake[s, n] - returns the first n characters of a string
pub fn string_take(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
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
  Ok(taken)
}

/// Handle StringDrop[s, n] - returns the string with the first n characters removed
pub fn string_drop(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
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
  Ok(dropped)
}

/// Handle StringJoin[s1, s2, ...] - concatenates multiple strings
pub fn string_join(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "StringJoin expects at least 1 argument".into(),
    ));
  }
  let mut joined = String::new();
  for ap in args_pairs {
    joined.push_str(&extract_string(ap.clone())?);
  }
  Ok(joined)
}

/// Handle StringSplit[s, delim] - splits a string by a delimiter
pub fn string_split(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
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
  Ok(format!("{{{}}}", parts.join(", ")))
}

/// Handle StringStartsQ[s, prefix] - checks if a string starts with the given prefix
pub fn string_starts_q(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  // expects exactly 2 string arguments
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringStartsQ expects exactly 2 arguments".into(),
    ));
  }
  let s = extract_string(args_pairs[0].clone())?;
  let prefix = extract_string(args_pairs[1].clone())?;
  Ok(
    if s.starts_with(&prefix) {
      "True"
    } else {
      "False"
    }
    .to_string(),
  )
}

/// Handle StringEndsQ[s, suffix] - checks if a string ends with the given suffix
pub fn string_ends_q(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  // expects exactly 2 string arguments
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringEndsQ expects exactly 2 arguments".into(),
    ));
  }
  let s = extract_string(args_pairs[0].clone())?;
  let suffix = extract_string(args_pairs[1].clone())?;
  Ok(
    if s.ends_with(&suffix) {
      "True"
    } else {
      "False"
    }
    .to_string(),
  )
}

/// Handle StringReplace[s, pattern -> replacement] - replaces occurrences of pattern with replacement
pub fn string_replace(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringReplace expects exactly 2 arguments".into(),
    ));
  }

  let s = extract_string(args_pairs[0].clone())?;

  // Second argument should be a rule: pattern -> replacement
  let rule = &args_pairs[1];
  let rule_str = rule.as_str();

  // Find the -> in the rule
  if let Some(arrow_pos) = rule_str.find("->") {
    let pattern = rule_str[..arrow_pos].trim();
    let replacement = rule_str[arrow_pos + 2..].trim();

    // Remove quotes if present
    let pattern = pattern.trim_matches('"');
    let replacement = replacement.trim_matches('"');

    Ok(s.replace(pattern, replacement))
  } else {
    Err(InterpreterError::EvaluationError(
      "StringReplace: second argument must be a rule (pattern -> replacement)"
        .into(),
    ))
  }
}

/// Handle ToUpperCase[s] - converts a string to uppercase
pub fn to_upper_case(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToUpperCase expects exactly 1 argument".into(),
    ));
  }
  let s = extract_string(args_pairs[0].clone())?;
  Ok(s.to_uppercase())
}

/// Handle ToLowerCase[s] - converts a string to lowercase
pub fn to_lower_case(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToLowerCase expects exactly 1 argument".into(),
    ));
  }
  let s = extract_string(args_pairs[0].clone())?;
  Ok(s.to_lowercase())
}

/// Handle StringContainsQ[s, sub] - checks if a string contains a substring
pub fn string_contains_q(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringContainsQ expects exactly 2 arguments".into(),
    ));
  }
  let s = extract_string(args_pairs[0].clone())?;
  let sub = extract_string(args_pairs[1].clone())?;
  Ok(if s.contains(&sub) { "True" } else { "False" }.to_string())
}

/// Handle Characters[s] - converts a string to a list of characters
pub fn characters(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Characters expects exactly 1 argument".into(),
    ));
  }
  let s = extract_string(args_pairs[0].clone())?;
  let chars: Vec<String> = s.chars().map(|c| c.to_string()).collect();
  Ok(format!("{{{}}}", chars.join(", ")))
}

/// Handle StringRiffle[list, sep] - joins strings with a separator
pub fn string_riffle(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() || args_pairs.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "StringRiffle expects 1 or 2 arguments".into(),
    ));
  }

  // Get the list
  let list_str = crate::evaluate_expression(args_pairs[0].clone())?;
  if !list_str.starts_with('{') || !list_str.ends_with('}') {
    return Err(InterpreterError::EvaluationError(
      "StringRiffle: first argument must be a list".into(),
    ));
  }

  // Parse the list elements
  let inner = &list_str[1..list_str.len() - 1];
  let elements: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();

  // Get the separator (default is space)
  let sep = if args_pairs.len() == 2 {
    extract_string(args_pairs[1].clone())?
  } else {
    " ".to_string()
  };

  Ok(elements.join(&sep))
}

/// Handle StringPosition[s, sub] - Find all positions of substring in string
/// Returns a list of {start, end} pairs (1-indexed, inclusive)
pub fn string_position(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringPosition expects exactly 2 arguments".into(),
    ));
  }

  let s = extract_string(args_pairs[0].clone())?;
  let sub = extract_string(args_pairs[1].clone())?;

  if sub.is_empty() {
    return Ok("{}".to_string());
  }

  let mut positions = Vec::new();
  let s_chars: Vec<char> = s.chars().collect();
  let sub_chars: Vec<char> = sub.chars().collect();

  // Find all occurrences
  for i in 0..=s_chars.len().saturating_sub(sub_chars.len()) {
    let mut matched = true;
    for (j, &sub_char) in sub_chars.iter().enumerate() {
      if s_chars[i + j] != sub_char {
        matched = false;
        break;
      }
    }
    if matched {
      // Wolfram uses 1-based indexing
      let start = i + 1;
      let end = i + sub_chars.len();
      positions.push(format!("{{{}, {}}}", start, end));
    }
  }

  Ok(format!("{{{}}}", positions.join(", ")))
}

/// Handle StringMatchQ[s, pattern] - Test if string matches a pattern
/// Supports wildcards: * matches any sequence, @ matches one or more non-uppercase chars
pub fn string_match_q(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringMatchQ expects exactly 2 arguments".into(),
    ));
  }

  let s = extract_string(args_pairs[0].clone())?;
  let pattern = extract_string(args_pairs[1].clone())?;

  // Convert Wolfram pattern to regex-like matching
  // * matches any sequence of characters (including empty)
  // @ matches one or more characters, excluding uppercase letters
  let matches = wildcard_match(&s, &pattern);

  Ok(if matches { "True" } else { "False" }.to_string())
}

/// Simple wildcard matching function
/// * matches any sequence (including empty)
/// @ matches one or more characters, excluding uppercase letters
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
      (true, false) => {
        // Check if remaining pattern is all *
        p.iter().all(|&c| c == '*')
      }
      (false, false) => {
        match p[0] {
          '*' => {
            // * matches zero or more characters
            match_helper(s, &p[1..]) || match_helper(&s[1..], p)
          }
          '@' => {
            // @ matches one or more non-uppercase characters
            // Must match at least one character
            if !is_non_uppercase(s[0]) {
              return false;
            }
            // Try matching 1 char and continue, or match more with @
            match_helper(&s[1..], &p[1..]) || match_helper(&s[1..], p)
          }
          c => {
            // Regular character must match exactly
            s[0] == c && match_helper(&s[1..], &p[1..])
          }
        }
      }
    }
  }

  match_helper(&s_chars, &p_chars)
}
