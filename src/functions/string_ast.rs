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

/// StringTake[s, n] - returns the first n characters of a string
pub fn string_take_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringTake expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringTake must be non-negative".into(),
    ));
  }
  let taken: String = s.chars().take(n as usize).collect();
  Ok(Expr::String(taken))
}

/// StringDrop[s, n] - returns the string with the first n characters removed
pub fn string_drop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringDrop expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of StringDrop must be non-negative".into(),
    ));
  }
  let dropped: String = s.chars().skip(n as usize).collect();
  Ok(Expr::String(dropped))
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

/// StringSplit[s, delim] - splits a string by a delimiter
pub fn string_split_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringSplit expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
  let delim = expr_to_str(&args[1])?;

  let parts: Vec<Expr> = if delim.is_empty() {
    s.chars().map(|c| Expr::String(c.to_string())).collect()
  } else {
    s.split(&delim)
      .map(|p| Expr::String(p.to_string()))
      .collect()
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

/// StringRiffle[list, sep] - joins strings with separator
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

  let sep = if args.len() == 2 {
    expr_to_str(&args[1])?
  } else {
    " ".to_string()
  };

  let strings: Result<Vec<String>, _> = items.iter().map(expr_to_str).collect();
  Ok(Expr::String(strings?.join(&sep)))
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

/// StringCases[s, patt] - find all substrings matching pattern
pub fn string_cases_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StringCases expects exactly 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;
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
    Expr::List(items) => {
      let mut result = String::new();
      for item in items {
        match item {
          Expr::Integer(n) => {
            let c = char::from_u32(*n as u32).ok_or_else(|| {
              InterpreterError::EvaluationError(format!(
                "Invalid character code: {}",
                n
              ))
            })?;
            result.push(c);
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "FromCharacterCode expects integer arguments".into(),
            ));
          }
        }
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
      let result = !s.is_empty() && s.chars().all(|c| c.is_alphabetic());
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
      let result = !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
      Ok(Expr::Identifier(
        if result { "True" } else { "False" }.to_string(),
      ))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}
