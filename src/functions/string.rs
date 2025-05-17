use pest::iterators::Pair;

use crate::{evaluate_term, extract_string, InterpreterError, Rule};

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
