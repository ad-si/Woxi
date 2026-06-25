//! AST-native string functions.
//!
//! These functions work directly with `Expr` AST nodes, avoiding string round-trips.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
  // Compiled-regex cache so tight loops calling `StringCases` /
  // `StringSplit` / `StringReplace` etc. with the same pattern don't pay
  // regex compilation cost on every iteration. `regex::Regex` clones are
  // cheap (the compiled state is Arc'd internally) so the cache value type
  // is just `Regex`. Capped to bound memory; on overflow we drop the
  // entire cache rather than maintain an LRU — the workloads we care
  // about reuse a handful of patterns at a time.
  static REGEX_CACHE: RefCell<HashMap<String, regex::Regex>> =
    RefCell::new(HashMap::new());
}

/// Return a compiled regex for `pat`, reusing a cached copy when one
/// exists. Identical pattern strings always produce identical compiled
/// regexes, so caching is observationally indistinguishable from
/// recompiling — only faster.
pub(crate) fn compile_regex(pat: &str) -> Result<regex::Regex, regex::Error> {
  use regex::Regex;
  REGEX_CACHE.with(|c| {
    if let Some(re) = c.borrow().get(pat) {
      return Ok(re.clone());
    }
    let re = Regex::new(pat)?;
    let mut cache = c.borrow_mut();
    if cache.len() >= 256 {
      cache.clear();
    }
    cache.insert(pat.to_string(), re.clone());
    Ok(re)
  })
}

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
  // StringLength threads over lists of strings.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|e| string_length_ast(&[e.clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  if let Expr::String(s) = &args[0] {
    return Ok(Expr::Integer(s.chars().count() as i128));
  }
  // Non-string argument: emit message and return unevaluated (matches wolframscript).
  crate::emit_message(&format!(
    "StringLength::string: String expected at position 1 in {}.",
    crate::syntax::expr_to_string(&Expr::FunctionCall {
      name: "StringLength".to_string(),
      args: args.to_vec().into(),
    })
  ));
  Ok(Expr::FunctionCall {
    name: "StringLength".to_string(),
    args: args.to_vec().into(),
  })
}

/// StringTake[s, n] - first n chars; StringTake[s, -n] - last n chars;
/// StringTake[s, {m, n}] - chars m through n; StringTake[s, {n}] - nth char
/// Emit StringTake::take / StringDrop::drop for an invalid position range.
fn string_take_drop_message(fname: &str, from: i128, to: i128, s: &str) {
  let (tag, verb) = if fname == "StringTake" {
    ("take", "take")
  } else {
    ("drop", "drop")
  };
  crate::emit_message(&format!(
    "{}::{}: Cannot {} positions {} through {} in \"{}\".",
    fname, tag, verb, from, to, s
  ));
}

/// Emit the `<fname>::strse: A string or list of strings is expected at
/// position 1 in <call>.` message, rendering the call in OutputForm.
fn emit_strse(fname: &str, args: &[Expr]) {
  crate::emit_message(&format!(
    "{}::strse: A string or list of strings is expected at position 1 in {}.",
    fname,
    crate::syntax::format_expr(
      &Expr::FunctionCall {
        name: fname.to_string(),
        args: args.to_vec().into(),
      },
      crate::syntax::ExprForm::Output
    )
  ));
}

/// Is `e` a valid string subject: a string, or a list whose elements are all
/// strings? (An empty list counts as valid.)
fn is_string_subject(e: &Expr) -> bool {
  match e {
    Expr::String(_) => true,
    Expr::List(items) => items.iter().all(|i| matches!(i, Expr::String(_))),
    _ => false,
  }
}

/// Shared StringTake/StringDrop engine, mirroring the Take/Drop position
/// conventions: scalar n / -n, {i}, {i, j}, {i, j, step}, All, None,
/// UpTo[n]. The adjacent reversed range {i, i-1} is an empty take /
/// no-op drop; anything further reversed or out of range emits the
/// take/drop message and stays unevaluated, as do non-string arguments
/// (::strse).
fn string_take_drop(
  fname: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: fname.to_string(),
    args: args.to_vec().into(),
  };
  // Thread over a list of strings
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| string_take_drop(fname, &[item.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  let Expr::String(s) = &args[0] else {
    emit_strse(fname, args);
    return Ok(unevaluated());
  };
  let chars: Vec<char> = s.chars().collect();
  let len = chars.len() as i128;
  let is_take = fname == "StringTake";
  let take = |from: usize, to: usize, step: i128| -> Expr {
    // 1-based inclusive positions, already validated
    if is_take {
      let mut out = String::new();
      let mut i = from as i128;
      while (step > 0 && i <= to as i128) || (step < 0 && i >= to as i128) {
        out.push(chars[(i - 1) as usize]);
        i += step;
      }
      Expr::String(out)
    } else {
      let mut keep: Vec<bool> = vec![true; chars.len()];
      let mut i = from as i128;
      while (step > 0 && i <= to as i128) || (step < 0 && i >= to as i128) {
        keep[(i - 1) as usize] = false;
        i += step;
      }
      Expr::String(
        chars
          .iter()
          .zip(&keep)
          .filter(|(_, k)| **k)
          .map(|(c, _)| c)
          .collect(),
      )
    }
  };
  let empty_or_all = |empty: bool| -> Expr {
    if empty == is_take {
      // empty take or full drop -> ""
      if is_take {
        Expr::String(String::new())
      } else {
        Expr::String(String::new())
      }
    } else {
      Expr::String(s.clone())
    }
  };

  match &args[1] {
    Expr::Identifier(name) if name == "All" => {
      // take everything / drop everything
      Ok(if is_take {
        Expr::String(s.clone())
      } else {
        Expr::String(String::new())
      })
    }
    Expr::Identifier(name) if name == "None" => {
      // take nothing / drop nothing
      Ok(if is_take {
        Expr::String(String::new())
      } else {
        Expr::String(s.clone())
      })
    }
    // A list of sub-specifications returns a list
    Expr::List(elems) if elems.iter().any(|e| matches!(e, Expr::List(_))) => {
      let results: Result<Vec<Expr>, InterpreterError> = elems
        .iter()
        .map(|e| string_take_drop(fname, &[args[0].clone(), e.clone()]))
        .collect();
      Ok(Expr::List(results?.into()))
    }
    Expr::List(elems) if elems.len() == 1 => {
      let Some(i) = crate::functions::list_helpers_ast::expr_to_i128(&elems[0])
      else {
        return Ok(unevaluated());
      };
      let real = if i > 0 { i } else { len + i + 1 };
      if real >= 1 && real <= len {
        Ok(take(real as usize, real as usize, 1))
      } else {
        string_take_drop_message(fname, i, i, s);
        Ok(unevaluated())
      }
    }
    Expr::List(elems) if elems.len() == 2 || elems.len() == 3 => {
      let nums: Option<Vec<i128>> = elems
        .iter()
        .map(crate::functions::list_helpers_ast::expr_to_i128)
        .collect();
      let Some(nums) = nums else {
        return Ok(unevaluated());
      };
      let (m, n) = (nums[0], nums[1]);
      let step = if nums.len() == 3 { nums[2] } else { 1 };
      if step == 0 {
        return Ok(unevaluated());
      }
      let real_start = if m >= 0 { m } else { len + m + 1 };
      let real_end = if n >= 0 { n } else { len + n + 1 };
      if real_end == real_start - step {
        return Ok(empty_or_all(true));
      }
      let in_range = real_start >= 1
        && real_end >= 1
        && real_start <= len
        && real_end <= len;
      let proper = (step > 0 && real_end >= real_start)
        || (step < 0 && real_end <= real_start);
      if in_range && proper {
        Ok(take(real_start as usize, real_end as usize, step))
      } else {
        string_take_drop_message(fname, m, n, s);
        Ok(unevaluated())
      }
    }
    Expr::FunctionCall {
      name: up_name,
      args: up_args,
    } if up_name == "UpTo" && up_args.len() == 1 => {
      match crate::functions::list_helpers_ast::expr_to_i128(&up_args[0]) {
        Some(k) if k >= 0 => {
          let count = k.min(len);
          if count == 0 {
            Ok(empty_or_all(true))
          } else {
            Ok(take(1, count as usize, 1))
          }
        }
        _ => Ok(unevaluated()),
      }
    }
    other => {
      let Some(count) = crate::functions::list_helpers_ast::expr_to_i128(other)
      else {
        return Ok(unevaluated());
      };
      if count.unsigned_abs() > len.unsigned_abs() {
        let (from, to) = if count >= 0 { (1, count) } else { (count, -1) };
        string_take_drop_message(fname, from, to, s);
        return Ok(unevaluated());
      }
      if count == 0 {
        return Ok(empty_or_all(true));
      }
      if count > 0 {
        Ok(take(1, count as usize, 1))
      } else {
        Ok(take((len + count + 1) as usize, len as usize, 1))
      }
    }
  }
}

/// Convert a `Span[i, j]` / `Span[i, j, k]` spec into the equivalent list form
/// `{i, j}` / `{i, j, k}` that StringTake/StringDrop already understand. A
/// missing/`All` start becomes 1 and a missing/`All` end becomes -1. Returns
/// None for non-Span or non-integer (symbolic) bounds.
fn span_to_list_arg(span: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = span else {
    return None;
  };
  if name != "Span" || !(args.len() == 2 || args.len() == 3) {
    return None;
  }
  let bound = |e: &Expr, all_default: i128| -> Option<Expr> {
    match e {
      Expr::Integer(_) => Some(e.clone()),
      Expr::Identifier(s) if s == "All" => Some(Expr::Integer(all_default)),
      _ => None,
    }
  };
  let mut list = vec![bound(&args[0], 1)?, bound(&args[1], -1)?];
  if args.len() == 3 {
    match &args[2] {
      Expr::Integer(_) => list.push(args[2].clone()),
      _ => return None,
    }
  }
  Some(Expr::List(list.into()))
}

pub fn string_take_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "StringTake".to_string(),
      args: args.to_vec().into(),
    });
  }
  // StringTake[s, i;;j] — a Span spec maps to the {i, j} list form.
  if let Some(list) = span_to_list_arg(&args[1]) {
    return string_take_drop("StringTake", &[args[0].clone(), list]);
  }
  string_take_drop("StringTake", args)
}

/// StringDrop[s, n] - drop first n chars; StringDrop[s, -n] - drop last n chars
/// StringDrop[s, {n}] - drop nth character; StringDrop[s, {m, n}] - drop chars m through n
pub fn string_drop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "StringDrop".to_string(),
      args: args.to_vec().into(),
    });
  }
  // StringDrop[s, i;;j] — a Span spec maps to the {i, j} list form.
  if let Some(list) = span_to_list_arg(&args[1]) {
    return string_take_drop("StringDrop", &[args[0].clone(), list]);
  }
  string_take_drop("StringDrop", args)
}

/// StringJoin[s1, s2, ...] or StringJoin[{s1, s2, ...}] - concatenates strings
pub fn string_join_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // StringJoin[] with no arguments is the empty string (matches wolframscript);
  // this also lets `StringJoin @@ {}` fold cleanly to "".
  if args.is_empty() {
    return Ok(Expr::String(String::new()));
  }

  // Validate: every leaf must be a string; lists of strings are flattened.
  // On any non-string leaf, emit a StringJoin::string message pointing at the
  // first offending leaf's 1-based position and return unevaluated (matches
  // wolframscript).
  fn first_non_string(expr: &Expr, pos: &mut usize) -> Option<usize> {
    match expr {
      Expr::String(_) => {
        *pos += 1;
        None
      }
      Expr::List(items) => {
        for item in items {
          if let Some(p) = first_non_string(item, pos) {
            return Some(p);
          }
        }
        None
      }
      _ => {
        *pos += 1;
        Some(*pos)
      }
    }
  }
  let mut cursor = 0usize;
  let mut bad_pos: Option<usize> = None;
  for arg in args {
    if let Some(p) = first_non_string(arg, &mut cursor) {
      bad_pos = Some(p);
      break;
    }
  }
  if let Some(pos) = bad_pos {
    // Format as infix: args joined by `<>`, dropping the enclosing quotes of
    // each string argument — so StringJoin["U", 2] renders as `U<>2`.
    let infix = args
      .iter()
      .map(|a| match a {
        Expr::String(s) => s.clone(),
        _ => crate::syntax::expr_to_string(a),
      })
      .collect::<Vec<_>>()
      .join("<>");
    crate::emit_message(&format!(
      "StringJoin::string: String expected at position {} in {}.",
      pos, infix
    ));
    return Ok(Expr::FunctionCall {
      name: "StringJoin".to_string(),
      args: args.to_vec().into(),
    });
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
/// Whether a string pattern is a plain literal (a `String`, or a list whose
/// every element is a `String`). Such patterns use exact substring matching
/// rather than the regex converter, avoiding metacharacter surprises.
fn is_literal_string_pattern(expr: &Expr) -> bool {
  match expr {
    Expr::String(_) => true,
    Expr::List(items) => items.iter().all(|it| matches!(it, Expr::String(_))),
    _ => false,
  }
}

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
/// Build a regex for the LHS of a `StringSplit` rule delimiter. Returns the
/// regex source plus, for a named pattern `name : sub`, the captured name so
/// the replacement can reference the actual matched delimiter.
fn split_delim_regex(
  lhs: &Expr,
) -> Result<(String, Option<String>), InterpreterError> {
  // Named pattern: Pattern[name, sub]
  if let Expr::FunctionCall { name, args } = lhs
    && name == "Pattern"
    && args.len() == 2
    && let Expr::Identifier(var) = &args[0]
  {
    let (re, _) = split_delim_regex(&args[1])?;
    return Ok((re, Some(var.clone())));
  }
  // Literal string delimiter.
  if let Expr::String(d) = lhs {
    return Ok((regex::escape(d), None));
  }
  // List of delimiters → alternation, longest source first so that overlapping
  // literals (e.g. {"::", ":"}) prefer the longer match.
  if let Expr::List(items) = lhs {
    let mut alts: Vec<String> = Vec::new();
    for it in items {
      let (re, _) = split_delim_regex(it)?;
      alts.push(format!("(?:{})", re));
    }
    alts.sort_by(|a, b| b.len().cmp(&a.len()));
    return Ok((alts.join("|"), None));
  }
  // Character classes / string patterns (DigitCharacter, RegularExpression…).
  if let Some(p) = extract_regex_pattern(lhs) {
    return Ok((p, None));
  }
  if let Some(p) = string_pattern_to_regex(lhs) {
    return Ok((p, None));
  }
  // Fallback: treat as a literal string.
  Ok((regex::escape(&expr_to_str(lhs)?), None))
}

/// Compute the replacement expression for a matched delimiter. A plain string
/// RHS is used verbatim; otherwise the captured name (if any) is substituted
/// with the matched text and the result evaluated.
fn split_replacement_value(
  rhs: &Expr,
  cap_name: Option<&str>,
  matched: &str,
) -> Result<Expr, InterpreterError> {
  if let Expr::String(r) = rhs {
    return Ok(Expr::String(r.clone()));
  }
  if let Some(name) = cap_name {
    let substituted = crate::syntax::substitute_variable(
      rhs,
      name,
      &Expr::String(matched.to_string()),
    );
    // Identity capture (`x : patt :> x`) needs no evaluation.
    if let Expr::String(_) = substituted {
      return Ok(substituted);
    }
    return crate::evaluator::evaluate_expr_to_expr(&substituted);
  }
  crate::evaluator::evaluate_expr_to_expr(rhs)
}

/// `StringSplit[s, patt -> repl]` / `patt :> repl`: split on `patt`, replacing
/// each matched delimiter with `repl` and keeping it between the pieces. Only
/// leading and trailing empty text segments are dropped (internal empties,
/// e.g. between adjacent delimiters, are preserved — matching WL).
fn string_split_with_replacement(
  s: &str,
  lhs: &Expr,
  rhs: &Expr,
  max_parts: Option<usize>,
  ignore_case: bool,
) -> Result<Expr, InterpreterError> {
  let (pat, cap_name) = split_delim_regex(lhs)?;
  let pat = if ignore_case {
    format!("(?i){}", pat)
  } else {
    pat
  };
  let re = compile_regex(&pat).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Invalid regular expression: {}",
      e
    ))
  })?;

  let mut result: Vec<Expr> = Vec::new();
  let mut last = 0usize;
  let mut splits = 0usize;
  for m in re.find_iter(s) {
    if m.start() == m.end() {
      continue; // ignore zero-width matches
    }
    if let Some(n) = max_parts
      && splits + 1 >= n
    {
      break; // stop after n-1 splits → at most n text pieces
    }
    result.push(Expr::String(s[last..m.start()].to_string()));
    let rep = split_replacement_value(
      rhs,
      cap_name.as_deref(),
      &s[m.start()..m.end()],
    )?;
    result.push(rep);
    last = m.end();
    splits += 1;
  }
  result.push(Expr::String(s[last..].to_string()));

  // Drop a leading / trailing empty *text* segment (positions 0 and last).
  if let Some(Expr::String(t)) = result.last()
    && t.is_empty()
    && result.len() > 1
  {
    result.pop();
  }
  if let Some(Expr::String(t)) = result.first()
    && t.is_empty()
    && result.len() > 1
  {
    result.remove(0);
  }
  Ok(Expr::List(result.into()))
}

pub fn string_split_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "StringSplit expects at least 1 argument".into(),
    ));
  }
  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0]
    && items.iter().all(|it| matches!(it, Expr::String(_)))
  {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut call = vec![item.clone()];
        call.extend(args[1..].iter().cloned());
        string_split_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  let s = expr_to_str(&args[0])?;

  // 1-argument form: split by whitespace, removing empty strings
  if args.len() == 1 {
    let parts: Vec<Expr> = s
      .split_whitespace()
      .map(|p| Expr::String(p.to_string()))
      .collect();
    return Ok(Expr::List(parts.into()));
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

  // Delimiter given as a rule `patt -> repl` or `patt :> repl`: split on
  // `patt` and keep the (replaced) delimiters interleaved between the pieces.
  if let Expr::Rule {
    pattern,
    replacement,
  }
  | Expr::RuleDelayed {
    pattern,
    replacement,
  } = &args[1]
  {
    return string_split_with_replacement(
      &s,
      pattern,
      replacement,
      max_parts,
      ignore_case,
    );
  }

  // Conditional delimiter: `patt /; test`. Split on each span where the inner
  // pattern matches and the `/;` test (with captures substituted) is True,
  // dropping empty pieces like Wolfram's default StringSplit.
  if let Expr::FunctionCall { name, args: cargs } = &args[1]
    && name == "Condition"
    && cargs.len() == 2
    && let Some((regex_str, _)) = string_pattern_to_regex_with_state(&cargs[0])
  {
    let test = &cargs[1];
    let pat = if ignore_case {
      format!("(?i){}", regex_str)
    } else {
      regex_str
    };
    let re = compile_regex(&pat).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Invalid regular expression: {}",
        e
      ))
    })?;
    let mut parts: Vec<Expr> = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    while i < s.len() {
      let mut is_delim = false;
      if let Some(caps) = re.captures_at(&s, i)
        && let Some(m) = caps.get(0)
        && m.start() == i
        && !m.as_str().is_empty()
      {
        let test_sub = substitute_captures(test, &caps);
        let test_val = crate::evaluator::evaluate_expr_to_expr(&test_sub)?;
        if matches!(&test_val, Expr::Identifier(t) if t == "True") {
          if !current.is_empty() {
            parts.push(Expr::String(std::mem::take(&mut current)));
          }
          i += m.len();
          is_delim = true;
        }
      }
      if !is_delim {
        let ch = s[i..].chars().next().unwrap();
        current.push(ch);
        i += ch.len_utf8();
      }
    }
    if !current.is_empty() {
      parts.push(Expr::String(current));
    }
    return Ok(Expr::List(parts.into()));
  }

  // Check if the delimiter is a RegularExpression or string pattern. Plain
  // string literals (and lists of them) use the literal-delimiter path below;
  // any non-literal delimiter — a bare pattern, or a list that contains one
  // (e.g. {DigitCharacter}) — goes through the pattern → regex converter.
  let regex_from_pattern = if let Some(p) = extract_regex_pattern(&args[1]) {
    Some(p)
  } else if !is_literal_string_pattern(&args[1]) {
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
    let re = compile_regex(&regex_pat).map_err(|e| {
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
    // Leading/trailing empty pieces are only dropped when no explicit
    // maximum number of pieces was requested.
    let parts = if max_parts.is_none() {
      trim_empty_strings(parts)
    } else {
      parts
    };
    return Ok(Expr::List(parts.into()));
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
      if let Some(n) = max_parts {
        re.splitn(&s, n)
          .map(|p| Expr::String(p.to_string()))
          .collect()
      } else {
        re.split(&s).map(|p| Expr::String(p.to_string())).collect()
      }
    } else if let Some(n) = max_parts {
      s.splitn(n, &delims[0])
        .map(|p| Expr::String(p.to_string()))
        .collect()
    } else {
      s.split(&delims[0])
        .map(|p| Expr::String(p.to_string()))
        .collect()
    }
  } else {
    // Split by multiple delimiters: scan left to right, trying the longest
    // delimiter match first. Once `max_parts - 1` pieces have been emitted,
    // stop splitting so the final piece keeps the original remainder verbatim
    // (mirroring `splitn`), matching wolframscript.
    let mut sorted_delims = delims.clone();
    sorted_delims.sort_by(|a, b| b.len().cmp(&a.len()));
    let mut result = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
      if let Some(n) = max_parts
        && result.len() == n - 1
      {
        current.extend(&chars[i..]);
        break;
      }
      let remaining: String = chars[i..].iter().collect();
      let mut matched = false;
      for d in &sorted_delims {
        if !d.is_empty() && remaining.starts_with(d.as_str()) {
          result.push(Expr::String(current.clone()));
          current.clear();
          i += d.chars().count();
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
  // Leading/trailing empty pieces are only dropped when no explicit maximum
  // number of pieces was requested.
  let parts = if max_parts.is_none() {
    trim_empty_strings(parts)
  } else {
    parts
  };
  Ok(Expr::List(parts.into()))
}

/// StringStartsQ[s, prefix] - checks if string starts with prefix
pub fn string_starts_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringStartsQ expects 2 or 3 arguments".into(),
    ));
  }
  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut call = vec![item.clone()];
        call.extend(args[1..].iter().cloned());
        string_starts_q_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
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
    let re = compile_regex(&full_pat).map_err(|e| {
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
  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut call = vec![item.clone()];
        call.extend(args[1..].iter().cloned());
        string_ends_q_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
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
    let re = compile_regex(&full_pat).map_err(|e| {
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
  // Thread over list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut call = vec![item.clone()];
        call.extend(args[1..].iter().cloned());
        string_contains_q_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
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
    let re = compile_regex(&full_pat).map_err(|e| {
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
/// Expand regex capture-group backreferences in a RegularExpression
/// replacement string: `$0` is the whole match, `$1`…`$n` (and `${n}`) the
/// numbered groups. A `$` not followed by a digit or `{` (including `$$`) is
/// left verbatim, matching Wolfram.
fn expand_dollar_replacement(template: &str, caps: &regex::Captures) -> String {
  let mut out = String::new();
  let mut chars = template.chars().peekable();
  while let Some(c) = chars.next() {
    if c != '$' {
      out.push(c);
      continue;
    }
    match chars.peek() {
      Some('{') => {
        chars.next(); // consume '{'
        let mut num = String::new();
        let mut closed = false;
        for d in chars.by_ref() {
          if d == '}' {
            closed = true;
            break;
          }
          num.push(d);
        }
        match (closed, num.parse::<usize>()) {
          (true, Ok(n)) => {
            if let Some(g) = caps.get(n) {
              out.push_str(g.as_str());
            }
          }
          // Malformed `${…}` is emitted verbatim.
          _ => {
            out.push_str("${");
            out.push_str(&num);
            if closed {
              out.push('}');
            }
          }
        }
      }
      Some(d) if d.is_ascii_digit() => {
        let mut num = String::new();
        while let Some(&d) = chars.peek() {
          if d.is_ascii_digit() {
            num.push(d);
            chars.next();
          } else {
            break;
          }
        }
        if let Ok(n) = num.parse::<usize>()
          && let Some(g) = caps.get(n)
        {
          out.push_str(g.as_str());
        }
      }
      // A lone `$` (or `$$`, `$x`, …) is literal.
      _ => out.push('$'),
    }
  }
  out
}

/// Expand `$0`/`$1`/… regex backreferences in every string literal within an
/// expression (used for StringCases RegularExpression transforms).
fn expand_dollar_in_expr(expr: &Expr, caps: &regex::Captures) -> Expr {
  match expr {
    Expr::String(s) => Expr::String(expand_dollar_replacement(s, caps)),
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| expand_dollar_in_expr(e, caps))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| expand_dollar_in_expr(e, caps))
        .collect::<Vec<_>>()
        .into(),
    },
    other => other.clone(),
  }
}

pub fn string_replace_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(
      "StringReplace expects 2 to 4 arguments".into(),
    ));
  }

  // The optional replacement limit is the first integer after the rules; the
  // `IgnoreCase -> True` option may appear as a trailing argument.
  let max_replacements = args[2..].iter().find_map(|a| match a {
    Expr::Integer(n) => Some(*n as usize),
    _ => None,
  });
  let ignore_case = has_ignore_case_option(args);

  // The subject (position 1) must be a string or a list of strings. A
  // non-string — or a list containing a non-string — makes wolframscript emit
  // StringReplace::strse and return the call unevaluated, rather than coercing
  // the argument to its textual form (e.g. `StringReplace[xyz, "a" -> "x"]`).
  if !is_string_subject(&args[0]) {
    emit_strse("StringReplace", args);
    return Ok(Expr::FunctionCall {
      name: "StringReplace".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Handle list of strings as first arg
  if let Expr::List(strings) = &args[0] {
    let mut results = Vec::new();
    for s_expr in strings {
      let mut new_args = vec![s_expr.clone()];
      new_args.extend_from_slice(&args[1..]);
      results.push(string_replace_ast(&new_args)?);
    }
    return Ok(Expr::List(results.into()));
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
      /// Whether `$0`/`$1`/… in the replacement expand to regex capture
      /// groups. Only RegularExpression patterns do this; a plain literal
      /// pattern (e.g. compiled for IgnoreCase) keeps `$n` verbatim.
      expand_dollar: bool,
    },
    /// Regex pattern with a delayed replacement expression (RuleDelayed).
    /// Captured named groups are substituted into the expression before evaluation.
    /// `condition` is the optional `/;` test (`patt /; test`): when present,
    /// the captured names are substituted into it and it must evaluate to
    /// `True` for the match to apply.
    RegexDelayed {
      regex: regex::Regex,
      replacement_expr: Expr,
      condition: Option<Expr>,
    },
  }

  fn extract_rule(
    expr: &Expr,
    ignore_case: bool,
  ) -> Result<ReplaceRule, InterpreterError> {
    let (pattern_expr, replacement_expr, is_delayed) = match expr {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref(), false),
      Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref(), true),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "StringReplace: rules must be of the form pattern -> replacement"
            .into(),
        ));
      }
    };

    // Unwrap a `/;` condition on the pattern: `patt /; test` parses to
    // `Condition[patt, test]`. A conditional pattern must evaluate its test
    // for each candidate match, so route it through `RegexDelayed` (with the
    // test attached) regardless of the replacement's shape.
    if let Expr::FunctionCall { name, args } = pattern_expr
      && name == "Condition"
      && args.len() == 2
    {
      let inner_pattern = &args[0];
      let test = &args[1];
      let regex_str =
        string_pattern_to_regex(inner_pattern).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "StringReplace: unsupported conditional pattern".into(),
          )
        })?;
      let regex_str = if ignore_case {
        format!("(?i){}", regex_str)
      } else {
        regex_str
      };
      let re = compile_regex(&regex_str).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "StringReplace: invalid pattern regex: {}",
          e
        ))
      })?;
      return Ok(ReplaceRule::RegexDelayed {
        regex: re,
        replacement_expr: replacement_expr.clone(),
        condition: Some(test.clone()),
      });
    }

    // For simple string literals, use direct matching. A delayed rule (:>)
    // still has to evaluate its RHS for each match, so compile the escaped
    // literal as a regex and route it through RegexDelayed. With IgnoreCase,
    // compile the escaped literal as a case-insensitive regex instead.
    if let Expr::String(pat_str) = pattern_expr {
      if is_delayed && !pat_str.is_empty() {
        let prefix = if ignore_case { "(?i)" } else { "" };
        let re =
          compile_regex(&format!("{}{}", prefix, regex::escape(pat_str)))
            .map_err(|e| {
              InterpreterError::EvaluationError(format!(
                "StringReplace: invalid pattern regex: {}",
                e
              ))
            })?;
        return Ok(ReplaceRule::RegexDelayed {
          regex: re,
          replacement_expr: replacement_expr.clone(),
          condition: None,
        });
      }
      let replacement = expr_to_str(replacement_expr)?;
      if ignore_case && !pat_str.is_empty() {
        let re = compile_regex(&format!("(?i){}", regex::escape(pat_str)))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "StringReplace: invalid pattern regex: {}",
              e
            ))
          })?;
        return Ok(ReplaceRule::Regex {
          regex: re,
          replacement,
          expand_dollar: false,
        });
      }
      return Ok(ReplaceRule::Simple {
        pattern: pat_str.clone(),
        replacement,
      });
    }

    // For complex patterns (Alternatives, StringExpression, etc.), use regex
    if let Some(regex_str) = string_pattern_to_regex(pattern_expr) {
      let regex_str = if ignore_case {
        format!("(?i){}", regex_str)
      } else {
        regex_str
      };
      let re = compile_regex(&regex_str).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "StringReplace: invalid pattern regex: {}",
          e
        ))
      })?;

      // Delayed rules (:>) evaluate their RHS for each match (after
      // substituting any named captures), so route them through RegexDelayed
      // even when the pattern has no capture groups — otherwise a constant
      // RHS like `LetterCharacter :> ToUpperCase["x"]` is inserted unevaluated.
      // A *plain-string* RHS is constant, so the delayed/immediate distinction
      // is moot: route it like a Rule below so `$1`/`$2` backreferences expand
      // (RegularExpression["(a)(b)"] :> "$2$1" -> "ba", matching wolframscript).
      if is_delayed && !matches!(replacement_expr, Expr::String(_)) {
        return Ok(ReplaceRule::RegexDelayed {
          regex: re,
          replacement_expr: replacement_expr.clone(),
          condition: None,
        });
      }

      // For Rule (->), also substitute captured names into the replacement
      // as strings before returning
      if re.capture_names().flatten().next().is_some() {
        return Ok(ReplaceRule::RegexDelayed {
          regex: re,
          replacement_expr: replacement_expr.clone(),
          condition: None,
        });
      }

      let replacement = expr_to_str(replacement_expr)?;
      return Ok(ReplaceRule::Regex {
        regex: re,
        replacement,
        // Regex capture-group backreferences ($1, $2, …) are a
        // RegularExpression feature.
        expand_dollar: matches!(
          pattern_expr,
          Expr::FunctionCall { name, .. } if name == "RegularExpression"
        ),
      });
    }

    // Fallback: try expr_to_str for identifiers etc.
    if let Ok(pat_str) = expr_to_str(pattern_expr) {
      let replacement = expr_to_str(replacement_expr)?;
      if ignore_case && !pat_str.is_empty() {
        let re = compile_regex(&format!("(?i){}", regex::escape(&pat_str)))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "StringReplace: invalid pattern regex: {}",
              e
            ))
          })?;
        return Ok(ReplaceRule::Regex {
          regex: re,
          replacement,
          expand_dollar: false,
        });
      }
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
        v.push(extract_rule(rule, ignore_case)?);
      }
      v
    }
    rule => vec![extract_rule(rule, ignore_case)?],
  };

  // Scan-based replacement: scan left-to-right, at each position try each rule
  fn scan_replace(
    s: &str,
    rules: &[ReplaceRule],
    max: Option<usize>,
  ) -> Result<String, InterpreterError> {
    let mut result = String::new();
    let mut count = 0usize;
    let mut i = 0;
    // Try the zero-width (empty) regex matches anchored at position `pos`.
    // Returns Some(replacement) for the first rule that matches empty there.
    // These are assertions like WordBoundary (\b), StartOfString, etc. that
    // consume no characters but still trigger a replacement.
    fn zero_width_match<'a>(
      s: &str,
      rules: &'a [ReplaceRule],
      pos: usize,
    ) -> Option<&'a str> {
      for rule in rules {
        if let ReplaceRule::Regex {
          regex, replacement, ..
        } = rule
          && let Some(m) = regex.find_at(s, pos)
          && m.start() == pos
          && m.as_str().is_empty()
        {
          return Some(replacement.as_str());
        }
      }
      None
    }
    while i < s.len() {
      if max.is_some() && count >= max.unwrap() {
        result.push_str(&s[i..]);
        break;
      }
      // Zero-width assertion matches (e.g. WordBoundary) are emitted before
      // the character at the current position, without consuming it.
      if let Some(replacement) = zero_width_match(s, rules, i) {
        result.push_str(replacement);
        count += 1;
        if max.is_some() && count >= max.unwrap() {
          result.push_str(&s[i..]);
          break;
        }
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
          ReplaceRule::Regex {
            regex,
            replacement,
            expand_dollar,
          } => {
            // Use captures_at on the full string so anchors like ^ and \b can
            // inspect surrounding characters. Require the match to start
            // exactly at position i.
            if let Some(caps) = regex.captures_at(s, i)
              && let Some(m) = caps.get(0)
              && m.start() == i
              && !m.as_str().is_empty()
            {
              if *expand_dollar {
                result.push_str(&expand_dollar_replacement(replacement, &caps));
              } else {
                result.push_str(replacement);
              }
              i += m.len();
              count += 1;
              matched = true;
              break;
            }
          }
          ReplaceRule::RegexDelayed {
            regex,
            replacement_expr,
            condition,
          } => {
            if let Some(caps) = regex.captures_at(s, i)
              && let Some(m) = caps.get(0)
              && m.start() == i
              && !m.as_str().is_empty()
            {
              // A `/;` condition must hold for this match. Substitute the
              // captured names into it and evaluate; skip the rule otherwise.
              if let Some(cond) = condition {
                let cond_sub = substitute_captures(cond, &caps);
                let cond_val =
                  crate::evaluator::evaluate_expr_to_expr(&cond_sub)?;
                if !matches!(&cond_val, Expr::Identifier(t) if t == "True") {
                  continue;
                }
              }
              // Substitute named captures into the replacement expr
              let substituted = substitute_captures(replacement_expr, &caps);
              // Evaluate the substituted expression
              let evaluated =
                crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              // Convert result to string
              let replacement_str = match &evaluated {
                Expr::String(s) => s.clone(),
                other => crate::syntax::expr_to_string(other),
              };
              result.push_str(&replacement_str);
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
    // Zero-width assertions can also match at the very end of the string
    // (e.g. WordBoundary after the final word character). The main loop
    // stops at `i == s.len()`, so check that final position here. Honour the
    // replacement limit so we don't exceed `max`.
    if max.is_none_or(|m| count < m)
      && zero_width_match(s, rules, s.len()).is_some()
      && let Some(replacement) = zero_width_match(s, rules, s.len())
    {
      result.push_str(replacement);
    }
    Ok(result)
  }

  Ok(Expr::String(scan_replace(&s, &rules, max_replacements)?))
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
    return Ok(Expr::List(results?.into()));
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
    return Ok(Expr::List(results?.into()));
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
    return Ok(Expr::List(results?.into()));
  }
  // Only a string yields a character list. Any other expression (number,
  // symbol, compound expression) stays unevaluated — wolframscript accepts
  // only a string or a list of strings and emits no message otherwise.
  if let Expr::String(s) = &args[0] {
    let chars: Vec<Expr> =
      s.chars().map(|c| Expr::String(c.to_string())).collect();
    return Ok(Expr::List(chars.into()));
  }
  Ok(Expr::FunctionCall {
    name: "Characters".to_string(),
    args: args.to_vec().into(),
  })
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
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "StringPosition expects at least 2 arguments".into(),
    ));
  }

  // Thread over a list of strings in the first argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| {
        let mut call = vec![s.clone()];
        call.extend(args[1..].iter().cloned());
        string_position_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let s = expr_to_str(&args[0])?;
  // Trailing arguments: an optional count limit (Integer) and/or an
  // `Overlaps -> True | False | All` option. StringPosition reports
  // overlapping matches by default.
  let mut max_results: Option<usize> = None;
  let mut overlaps = true;
  for a in &args[2..] {
    match a {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Overlaps") =>
      {
        overlaps = !matches!(replacement.as_ref(),
          Expr::Identifier(v) if v == "False");
      }
      other => {
        if let Ok(n) = expr_to_int(other) {
          max_results = Some(n as usize);
        }
      }
    }
  }
  // IgnoreCase -> True makes the pattern match regardless of case: the regex
  // path gets the `(?i)` flag and the literal path compares case-folded.
  let ignore_case = has_ignore_case_option(args);

  // Resolve the pattern to a regex. Plain string literals (and lists of
  // them) use the exact literal path below; RegularExpression and every other
  // string pattern (character classes, Alternatives, `_?test`, lists
  // containing patterns, …) go through the shared pattern → regex converter.
  let regex_pat = extract_regex_pattern(&args[1]).or_else(|| {
    if is_literal_string_pattern(&args[1]) {
      None
    } else {
      string_pattern_to_regex(&args[1])
    }
  });

  let s_chars: Vec<char> = s.chars().collect();
  // For each match: (start_index_0based, length_in_chars)
  let mut raw_matches: Vec<(usize, usize)> = Vec::new();

  // Bare conditional pattern: `patt /; test`. Match the inner pattern at each
  // position and keep the span only when the `/;` test (with captures
  // substituted in) evaluates to True.
  if let Expr::FunctionCall { name, args: cargs } = &args[1]
    && name == "Condition"
    && cargs.len() == 2
    && let Some((regex_str, _)) = string_pattern_to_regex_with_state(&cargs[0])
  {
    let test = &cargs[1];
    let pat = if ignore_case {
      format!("(?i){}", regex_str)
    } else {
      regex_str
    };
    if let Ok(re) = compile_regex(&pat) {
      for start_char in 0..s_chars.len() {
        let byte_offset: usize =
          s_chars[..start_char].iter().map(|c| c.len_utf8()).sum();
        if let Some(caps) = re.captures_at(&s, byte_offset)
          && let Some(m) = caps.get(0)
          && m.start() == byte_offset
          && !m.as_str().is_empty()
        {
          let test_sub = substitute_captures(test, &caps);
          let test_val = crate::evaluator::evaluate_expr_to_expr(&test_sub)?;
          if matches!(&test_val, Expr::Identifier(t) if t == "True") {
            raw_matches.push((start_char, m.as_str().chars().count()));
          }
        }
      }
    }
  } else if let Some(pat) = regex_pat.as_ref() {
    // Regex-based matching: find all overlapping matches
    let pat = if ignore_case {
      format!("(?i){}", pat)
    } else {
      pat.clone()
    };
    if let Ok(re) = compile_regex(&pat) {
      for start_char in 0..s_chars.len() {
        // Get byte offset for this character position
        let byte_offset: usize =
          s_chars[..start_char].iter().map(|c| c.len_utf8()).sum();
        let substr = &s[byte_offset..];
        if let Some(m) = re.find(substr)
          && m.start() == 0
          && !m.is_empty()
        {
          let match_chars = m.as_str().chars().count();
          raw_matches.push((start_char, match_chars));
        }
      }
    }
  } else {
    // Literal substring matching
    let subs: Vec<String> = match &args[1] {
      Expr::List(items) => {
        let mut v = Vec::with_capacity(items.len());
        for it in items {
          v.push(expr_to_str(it)?);
        }
        v
      }
      _ => vec![expr_to_str(&args[1])?],
    };

    for sub in &subs {
      if sub.is_empty() {
        continue;
      }
      let sub_chars: Vec<char> = sub.chars().collect();
      if sub_chars.len() > s_chars.len() {
        continue;
      }
      for i in 0..=s_chars.len() - sub_chars.len() {
        let mut matched = true;
        for (j, &sub_char) in sub_chars.iter().enumerate() {
          let c = s_chars[i + j];
          let eq = c == sub_char
            || (ignore_case && c.to_lowercase().eq(sub_char.to_lowercase()));
          if !eq {
            matched = false;
            break;
          }
        }
        if matched {
          raw_matches.push((i, sub_chars.len()));
        }
      }
    }
  }

  // Sort by start position, then by length, and drop duplicates so that
  // multiple alternatives matching at the same span don't yield repeats.
  raw_matches.sort();
  raw_matches.dedup();

  // `Overlaps -> False` keeps matches greedily left-to-right, skipping any
  // that begin before the end of the previously kept match.
  if !overlaps {
    let mut filtered: Vec<(usize, usize)> = Vec::new();
    let mut next_start = 0usize;
    for (start, len) in raw_matches {
      if start >= next_start {
        filtered.push((start, len));
        next_start = start + len;
      }
    }
    raw_matches = filtered;
  }

  let mut positions = Vec::new();
  for (start, len) in raw_matches {
    if let Some(max) = max_results
      && positions.len() >= max
    {
      break;
    }
    let s1 = (start + 1) as i128;
    let e1 = (start + len) as i128;
    positions.push(Expr::List(
      vec![Expr::Integer(s1), Expr::Integer(e1)].into(),
    ));
  }

  Ok(Expr::List(positions.into()))
}

/// StringMatchQ[s, pattern] - test if string matches pattern
/// StringMatchQ[s, pattern, IgnoreCase -> True] - case-insensitive
pub fn string_match_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringMatchQ expects 2 or 3 arguments".into(),
    ));
  }

  // Thread over a list of strings: StringMatchQ[{s1, s2, ...}, pat]
  // returns {StringMatchQ[s1, pat], StringMatchQ[s2, pat], ...}.
  if let Expr::List(items) = &args[0] {
    let mut result: Vec<Expr> = Vec::with_capacity(items.len());
    for item in items {
      let mut sub_args: Vec<Expr> = Vec::with_capacity(args.len());
      sub_args.push(item.clone());
      sub_args.extend(args[1..].iter().cloned());
      result.push(string_match_q_ast(&sub_args)?);
    }
    return Ok(Expr::List(result.into()));
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
    let re = compile_regex(&full_regex).map_err(|e| {
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
    let re = compile_regex(&full_regex).map_err(|e| {
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
    return Ok(Expr::List(results?.into()));
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
    return Ok(Expr::List(results?.into()));
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
      compile_regex(&format!("^(?:{})", regex_pat)).map_err(|e| {
        InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
      })?;
    let end_re =
      compile_regex(&format!("(?:{})$", regex_pat)).map_err(|e| {
        InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
      })?;
    let trimmed = start_re.replace(&s, "");
    let trimmed = end_re.replace(&trimmed, "");
    Ok(Expr::String(trimmed.into_owned()))
  } else {
    // Unrecognized pattern: return unevaluated
    Ok(Expr::FunctionCall {
      name: "StringTrim".to_string(),
      args: args.to_vec().into(),
    })
  }
}

/// Substitute named captures into an expression, replacing pattern variable
/// identifiers with matched string values.
fn substitute_captures(expr: &Expr, captures: &regex::Captures) -> Expr {
  match expr {
    Expr::Identifier(name) => {
      if let Some(m) = captures.name(name) {
        Expr::String(m.as_str().to_string())
      } else {
        expr.clone()
      }
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> = args
        .iter()
        .map(|a| substitute_captures(a, captures))
        .collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_captures(left, captures)),
      right: Box::new(substitute_captures(right, captures)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_captures(operand, captures)),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| substitute_captures(a, captures))
        .collect(),
    ),
    // A rule (or delayed rule) RHS such as `w -> d` must have its pattern
    // variables substituted on both sides, not be left verbatim.
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_captures(pattern, captures)),
      replacement: Box::new(substitute_captures(replacement, captures)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(substitute_captures(pattern, captures)),
      replacement: Box::new(substitute_captures(replacement, captures)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|a| substitute_captures(a, captures))
        .collect(),
      operators: operators.clone(),
    },
    Expr::Association(pairs) => Expr::Association(
      pairs
        .iter()
        .map(|(k, v)| {
          (
            substitute_captures(k, captures),
            substitute_captures(v, captures),
          )
        })
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// Convert a bare (top-level) string pattern using Wolfram's metacharacter
/// shorthand: `*` matches any sequence and `@` matches one-or-more
/// non-uppercase characters. Every other character is matched literally.
fn bare_string_metachars_to_regex(s: &str) -> String {
  let mut result = String::new();
  for ch in s.chars() {
    match ch {
      '*' => result.push_str(".*"),
      '@' => result.push_str("[^A-Z]+"),
      _ => result.push_str(&regex::escape(&ch.to_string())),
    }
  }
  result
}

/// Append `?` to every greedy quantifier in `regex` so repetition matches
/// the shortest substring. Skips quantifiers that are already lazy, leaves a
/// standalone `?` (optional) untouched, and never touches escaped literals
/// (e.g. `\*` / `\+`, which are literal characters, not quantifiers).
fn make_non_greedy(regex: &str) -> String {
  let bytes = regex.as_bytes();
  let mut out = String::with_capacity(bytes.len() + 4);
  let mut i = 0;
  while i < bytes.len() {
    let b = bytes[i];
    if b == b'\\' && i + 1 < bytes.len() {
      // Escaped character: copy the backslash and its target verbatim; the
      // escaped char is a literal, never a quantifier.
      out.push('\\');
      out.push(bytes[i + 1] as char);
      i += 2;
      continue;
    }
    out.push(b as char);
    let next = bytes.get(i + 1).copied();
    let make_lazy = match b {
      b'+' | b'*' => next != Some(b'?'),
      b'}' => {
        // Only quantifier `{..}` (not a literal brace) should be lazy.
        // Determine if we're closing a quantifier by scanning backward to
        // the matching `{` and ensuring the content is digits/commas.
        let mut j = i;
        let mut found_open = false;
        while j > 0 {
          j -= 1;
          let c = bytes[j];
          if c == b'{' {
            found_open = true;
            break;
          }
          if !(c.is_ascii_digit() || c == b',') {
            break;
          }
        }
        found_open && next != Some(b'?')
      }
      _ => false,
    };
    if make_lazy {
      out.push('?');
    }
    i += 1;
  }
  out
}

/// Compute the set-difference of two regex character classes (`base \ excluded`)
/// when both are simple bracket forms like `[a-zA-Z0-9\p{L}]`. Returns a
/// regex string representing only the atoms present in `base` but not in
/// `excluded`. Returns None if either side is not a single bracket class.
fn char_class_difference(excluded: &str, base: &str) -> Option<String> {
  let parse = |s: &str| -> Option<Vec<String>> {
    let body = s.strip_prefix('[').and_then(|s| s.strip_suffix(']'))?;
    if body.starts_with('^') {
      // Negated classes are not currently supported here.
      return None;
    }
    let mut atoms: Vec<String> = Vec::new();
    let chars: Vec<char> = body.chars().collect();
    let mut i = 0;
    while i < chars.len() {
      let c = chars[i];
      if c == '\\' && i + 1 < chars.len() {
        // Escape sequence: \p{L}, \w, \d, \s, \\, etc.
        let next = chars[i + 1];
        if next == 'p' || next == 'P' {
          // \p{...} — capture through closing brace.
          let mut j = i + 2;
          if j < chars.len() && chars[j] == '{' {
            while j < chars.len() && chars[j] != '}' {
              j += 1;
            }
            if j < chars.len() {
              j += 1; // include }
              atoms.push(chars[i..j].iter().collect());
              i = j;
              continue;
            }
          }
          atoms.push(format!("\\{}", next));
          i += 2;
          continue;
        }
        atoms.push(format!("\\{}", next));
        i += 2;
        continue;
      }
      // Range like a-z?
      if i + 2 < chars.len() && chars[i + 1] == '-' && chars[i + 2] != ']' {
        atoms.push(format!("{}-{}", c, chars[i + 2]));
        i += 3;
        continue;
      }
      atoms.push(c.to_string());
      i += 1;
    }
    Some(atoms)
  };

  let excluded_atoms = parse(excluded)?;
  let base_atoms = parse(base)?;
  let kept: Vec<String> = base_atoms
    .into_iter()
    .filter(|a| !excluded_atoms.contains(a))
    .collect();
  if kept.is_empty() {
    return Some(r"(?!.)".to_string());
  }
  Some(format!("[{}]", kept.join("")))
}

/// True if `name` is usable as a Rust regex named-capture identifier
/// (`[A-Za-z_][A-Za-z0-9_]*`). Wolfram pattern variables that contain
/// special characters can't round-trip as regex group names — fall back
/// to a non-capturing group in that case.
fn is_valid_regex_capture_name(name: &str) -> bool {
  let mut chars = name.chars();
  match chars.next() {
    Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
    _ => return false,
  }
  chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Tracks pattern-variable bindings while translating a Wolfram string
/// pattern to a regex. Repeated pattern names are realised as Wolfram-style
/// back-references: the second occurrence reuses the first occurrence's
/// regex body (so e.g. `x : WordCharacter .. ~~ "-" ~~ x_` matches the same
/// run on both sides of the dash), and the resulting (orig, dup) capture-name
/// pair is recorded so callers can post-filter matches whose captures don't
/// actually agree.
struct PatternState {
  /// Maps a pattern name to the regex body of its first occurrence.
  first_body: std::collections::HashMap<String, String>,
  /// Counter used to name duplicate captures uniquely (`name__dupN`).
  dup_counter: usize,
  /// Pairs `(original_capture_name, duplicate_capture_name)` whose captures
  /// must compare equal for the overall match to be accepted.
  constraints: Vec<(String, String)>,
}

impl PatternState {
  fn new() -> Self {
    Self {
      first_body: std::collections::HashMap::new(),
      dup_counter: 0,
      constraints: Vec::new(),
    }
  }
}

/// Public entry point: builds the regex and discards any back-reference
/// constraints (callers that need them should use
/// `string_pattern_to_regex_with_state` instead).
fn string_pattern_to_regex(expr: &Expr) -> Option<String> {
  string_pattern_to_regex_with_state(expr).map(|(s, _)| s)
}

/// Like `string_pattern_to_regex` but also returns the list of `(orig, dup)`
/// capture-name pairs that must agree for a regex match to be considered a
/// real Wolfram pattern match.
fn string_pattern_to_regex_with_state(
  expr: &Expr,
) -> Option<(String, Vec<(String, String)>)> {
  let mut state = PatternState::new();
  // A bare string pattern uses Wolfram's metacharacter shorthand (`*` matches
  // any run, `@` matches one-or-more non-uppercase). Strings nested inside a
  // compound pattern (StringExpression via `~~`, Shortest, …) match literally,
  // so this only fires when the whole pattern is a single string.
  let regex = if let Expr::String(s) = expr {
    bare_string_metachars_to_regex(s)
  } else {
    string_pattern_to_regex_inner(expr, &mut state)?
  };
  // Enable the "dot matches newline" flag so blanks (`.`, `.+`, `.*`) and `*`
  // span newlines, matching Wolfram string patterns — e.g. Shortest[___]
  // stripping a multi-line block comment.
  Some((format!("(?s){}", regex), state.constraints))
}

/// Wrap a regex body in a named capture group. On a repeated name, reuse the
/// first occurrence's regex body (so the duplicate matches the *same* shape
/// as the original) and record a constraint that the captured substrings
/// must compare equal — emulating Wolfram's pattern back-references on top
/// of the Rust `regex` crate, which has no native backreference support.
fn maybe_named_group(
  name: &str,
  body: &str,
  state: &mut PatternState,
) -> String {
  if !is_valid_regex_capture_name(name) {
    return format!("(?:{})", body);
  }
  if let Some(first_body) = state.first_body.get(name).cloned() {
    let dup_name = format!("{}__dup{}", name, state.dup_counter);
    state.dup_counter += 1;
    state.constraints.push((name.to_string(), dup_name.clone()));
    format!("(?P<{}>{})", dup_name, first_body)
  } else {
    state.first_body.insert(name.to_string(), body.to_string());
    format!("(?P<{}>{})", name, body)
  }
}

/// True if `re` is a regex body that matches exactly one character — either
/// a single non-metacharacter, or a backslash escape of one character. Used
/// by `Except[c]` to lift the inner pattern into a negated character class
/// (`[^c]`), which composes with `..` quantifiers in the Rust `regex` crate
/// without requiring look-around.
fn is_single_char_atom(re: &str) -> bool {
  let mut chars = re.chars();
  match (chars.next(), chars.next(), chars.next()) {
    (Some(c), None, _) => !"\\^$.|?*+()[]{}".contains(c),
    (Some('\\'), Some(_), None) => true,
    _ => false,
  }
}

/// Convert a Wolfram string pattern expression to a regex pattern string.
/// Returns None if the expression is not a recognized string pattern.
fn string_pattern_to_regex_inner(
  expr: &Expr,
  seen: &mut PatternState,
) -> Option<String> {
  match expr {
    // A string literal nested inside a compound pattern matches literally —
    // the `*`/`@` metacharacter shorthand only applies to a bare top-level
    // string (handled in `string_pattern_to_regex_with_state`). So e.g.
    // `"/*" ~~ Shortest[___] ~~ "*/"` matches a literal `/*…*/`.
    Expr::String(s) => Some(regex::escape(s)),

    // Character class patterns and blank patterns
    Expr::Identifier(name) => match name.as_str() {
      "DigitCharacter" => Some("[0-9]".to_string()),
      "LetterCharacter" => Some("[a-zA-Z\\p{L}]".to_string()),
      "WhitespaceCharacter" => Some("\\s".to_string()),
      "Whitespace" => Some("\\s+".to_string()),
      // Wolfram's `WordCharacter` matches only ASCII letters and digits
      // (verified by `StringMatchQ["ö", WordCharacter]` → False), so don't
      // include the broader `\p{L}` Unicode-letter class here.
      "WordCharacter" => Some("[a-zA-Z0-9]".to_string()),
      // PunctuationCharacter matches any Unicode punctuation character
      // (general category P) plus the ASCII symbol characters
      // `$ + < = > ^ ` | ~` (which Unicode classifies as symbols, not
      // punctuation, but Wolfram treats as punctuation). Verified against
      // wolframscript across the BMP.
      "PunctuationCharacter" => Some("(?:\\p{P}|[$+<=>^`|~])".to_string()),
      "HexadecimalCharacter" => Some("[0-9a-fA-F]".to_string()),
      // NumberString matches an optional leading sign followed by digits
      // with an optional decimal point, or a leading-decimal form like `.5`
      // (no exponent — Wolfram treats `1e5` as two number strings).
      "NumberString" => Some("[+-]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)".to_string()),
      "_" => Some(".".to_string()), // Blank: any single character
      "__" => Some(".+".to_string()), // BlankSequence: one or more characters
      "___" => Some(".*".to_string()), // BlankNullSequence: zero or more characters
      // Position anchors — StringMatchQ already anchors at both ends, so these
      // collapse to empty matches there, but inside patterns used with
      // StringCases/StringReplace they bind to regex anchors.
      "StartOfString" => Some("\\A".to_string()),
      "EndOfString" => Some("\\z".to_string()),
      "StartOfLine" => Some("(?m:^)".to_string()),
      "EndOfLine" => Some("(?m:$)".to_string()),
      "WordBoundary" => Some("\\b".to_string()),
      _ => None,
    },

    // Blank/BlankSequence/BlankNullSequence as Pattern AST nodes
    Expr::Pattern {
      name,
      head: None,
      blank_type,
    } => {
      let inner = match blank_type {
        1 => ".",  // Blank: any single character
        2 => ".+", // BlankSequence: one or more characters
        3 => ".*", // BlankNullSequence: zero or more characters
        _ => return None,
      };
      if !name.is_empty() {
        Some(maybe_named_group(name, inner, seen))
      } else {
        Some(inner.to_string())
      }
    }

    // A blank with a character-class predicate test, e.g. `_?LetterQ`,
    // `x_?DigitQ`, `__?UpperCaseQ`. Map the known single-character predicates
    // to the matching regex character class; the blank type controls the
    // repetition (Blank → one char, BlankSequence → one or more, etc.).
    Expr::PatternTest {
      name,
      head: None,
      blank_type,
      test,
    } => {
      let class = match test.as_ref() {
        Expr::Identifier(t) => match t.as_str() {
          "LetterQ" => "[a-zA-Z\\p{L}]",
          "DigitQ" => "[0-9]",
          "UpperCaseQ" => "[A-Z\\p{Lu}]",
          "LowerCaseQ" => "[a-z\\p{Ll}]",
          _ => return None,
        },
        _ => return None,
      };
      let inner = match blank_type {
        1 => class.to_string(),
        2 => format!("{}+", class),
        3 => format!("{}*", class),
        _ => return None,
      };
      if name.is_empty() {
        Some(inner)
      } else {
        Some(maybe_named_group(name, &inner, seen))
      }
    }

    // `name : pattern` parses as `Pattern[name, pattern]` — wrap the
    // inner regex in a named capture group so a downstream rule's RHS
    // can refer back to the matched substring.
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      let inner = string_pattern_to_regex_inner(&args[1], seen)?;
      if let Expr::Identifier(var) = &args[0] {
        Some(maybe_named_group(var, &inner, seen))
      } else {
        Some(format!("(?:{})", inner))
      }
    }

    // Alternatives as BinaryOp (e.g. pat1 | pat2)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left,
      right,
    } => {
      let l = string_pattern_to_regex_inner(left, seen)?;
      let r = string_pattern_to_regex_inner(right, seen)?;
      Some(format!("(?:{}|{})", l, r))
    }

    // Alternatives[pat1, pat2, ...] = pat1 | pat2 | ...
    Expr::FunctionCall { name, args }
      if name == "Alternatives" && !args.is_empty() =>
    {
      let parts: Option<Vec<String>> = args
        .iter()
        .map(|a| string_pattern_to_regex_inner(a, seen))
        .collect();
      parts.map(|ps| format!("(?:{})", ps.join("|")))
    }

    // A list {pat1, pat2, ...} in a string-pattern context is treated as
    // Alternatives[pat1, pat2, ...].
    Expr::List(items) if !items.is_empty() => {
      let parts: Option<Vec<String>> = items
        .iter()
        .map(|a| string_pattern_to_regex_inner(a, seen))
        .collect();
      parts.map(|ps| format!("(?:{})", ps.join("|")))
    }

    // StringExpression[pat1, pat2, ...] = pat1 ~~ pat2 ~~ ...
    Expr::FunctionCall { name, args }
      if name == "StringExpression" && !args.is_empty() =>
    {
      let parts: Option<Vec<String>> = args
        .iter()
        .map(|a| string_pattern_to_regex_inner(a, seen))
        .collect();
      parts.map(|ps| ps.join(""))
    }

    // Repeated[pat] = pat.. (one or more)
    Expr::FunctionCall { name, args }
      if name == "Repeated" && (args.len() == 1 || args.len() == 2) =>
    {
      // Wolfram parses `x : pat..` as `Pattern[x, Repeated[pat]]` — `..`
      // binds tighter than `:`. Our parser produces `Repeated[Pattern[x,
      // pat]]` instead. The two are equivalent under matching, but only
      // the outer-Pattern shape captures the *whole* run as `x` rather
      // than just the last single-char match. Rewrite the inner pattern
      // by lifting the named Pattern out so the named capture spans all
      // repetitions.
      let (capture_name, inner_pat) = match &args[0] {
        Expr::FunctionCall {
          name: pn,
          args: pargs,
        } if pn == "Pattern" && pargs.len() == 2 => {
          if let Expr::Identifier(var) = &pargs[0] {
            (Some(var.clone()), &pargs[1])
          } else {
            (None, &args[0])
          }
        }
        _ => (None, &args[0]),
      };
      let base = string_pattern_to_regex_inner(inner_pat, seen)?;
      let body = if args.len() == 1 {
        format!("(?:{})+", base)
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
        format!("(?:{}){}", base, quantifier)
      };
      Some(match capture_name {
        Some(var) => maybe_named_group(&var, &body, seen),
        None => body,
      })
    }

    // RepeatedNull[pat] = pat... (zero or more)
    Expr::FunctionCall { name, args }
      if name == "RepeatedNull" && args.len() == 1 =>
    {
      let (capture_name, inner_pat) = match &args[0] {
        Expr::FunctionCall {
          name: pn,
          args: pargs,
        } if pn == "Pattern" && pargs.len() == 2 => {
          if let Expr::Identifier(var) = &pargs[0] {
            (Some(var.clone()), &pargs[1])
          } else {
            (None, &args[0])
          }
        }
        _ => (None, &args[0]),
      };
      let base = string_pattern_to_regex_inner(inner_pat, seen)?;
      let body = format!("(?:{})*", base);
      Some(match capture_name {
        Some(var) => maybe_named_group(&var, &body, seen),
        None => body,
      })
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

    // Shortest[pat] — match the shortest substring (non-greedy).
    Expr::FunctionCall { name, args }
      if name == "Shortest" && args.len() == 1 =>
    {
      let constraints_before = seen.constraints.len();
      let inner = string_pattern_to_regex_inner(&args[0], seen)?;
      if seen.constraints.len() > constraints_before {
        // A duplicate-name back-reference inside Shortest would conflict
        // with non-greedy semantics: the engine would commit to a shortest
        // match that fails the equality post-filter and never try the
        // longer match where both captures agree. Skip the non-greedy
        // rewrite here — the equality constraint already pins down the
        // unique valid match length.
        Some(inner)
      } else {
        Some(make_non_greedy(&inner))
      }
    }

    // Longest[pat] — default greedy match; equivalent to pat itself.
    Expr::FunctionCall { name, args }
      if name == "Longest" && args.len() == 1 =>
    {
      string_pattern_to_regex_inner(&args[0], seen)
    }

    // Except[pattern] — negate a single-character class. Only meaningful
    // when the inner pattern translates to a character-class regex like
    // `[...]` (Except matches exactly one non-matching character).
    Expr::FunctionCall { name, args }
      if name == "Except" && args.len() == 1 =>
    {
      let inner = string_pattern_to_regex_inner(&args[0], seen)?;
      // If the inner regex is already a character class, negate it.
      if let Some(stripped) = inner.strip_prefix('[')
        && let Some(body) = stripped.strip_suffix(']')
      {
        if let Some(neg_body) = body.strip_prefix('^') {
          // Double negation — just drop the ^.
          return Some(format!("[{}]", neg_body));
        }
        return Some(format!("[^{}]", body));
      }
      // Single-character literal (e.g. `]` → `\]`, `a` → `a`): lift into
      // a negated character class so that `Except[c]..` quantifies as
      // `[^c]+` instead of `(?:(?!c).)+`. The Rust `regex` crate has no
      // look-around, so this is the only shape that composes with `..`
      // / `...` quantifiers without errors.
      if is_single_char_atom(&inner) {
        return Some(format!("[^{}]", inner));
      }
      // Otherwise fall back to a regex negative lookahead + any char.
      Some(format!("(?:(?!{}).)", inner))
    }

    // Except[c, base] — match `base` but not `c`. The Rust `regex` crate
    // does not support look-around, so realize the set difference at the
    // character-class level: parse both regex character classes into atoms
    // (single chars, ranges, escape classes) and drop any atom from `base`
    // that also appears in `excluded`.
    Expr::FunctionCall { name, args }
      if name == "Except" && args.len() == 2 =>
    {
      let excluded = string_pattern_to_regex_inner(&args[0], seen)?;
      let base = string_pattern_to_regex_inner(&args[1], seen)?;
      char_class_difference(&excluded, &base)
    }

    _ => None,
  }
}

/// StringCases[s, patt] - find all substrings matching pattern
pub fn string_cases_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "StringCases expects at least 2 arguments".into(),
    ));
  }
  let s = expr_to_str(&args[0])?;

  // Parse the optional trailing arguments: a max-count (Integer/Infinity)
  // and/or an `Overlaps -> True | All` option (default: non-overlapping).
  let mut max_count: usize = usize::MAX;
  let mut overlaps = false;
  for a in &args[2..] {
    match a {
      Expr::Integer(n) if *n >= 0 => max_count = *n as usize,
      Expr::Identifier(id) if id == "Infinity" => max_count = usize::MAX,
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Overlaps") =>
      {
        overlaps = matches!(replacement.as_ref(),
          Expr::Identifier(v) if v == "True" || v == "All");
      }
      _ => {}
    }
  }
  // IgnoreCase -> True makes every pattern match case-insensitively; it is
  // applied by prefixing the compiled regexes with the `(?i)` flag.
  let ignore_case = has_ignore_case_option(args);
  let with_ci = |pat: &str| -> String {
    if ignore_case {
      format!("(?i){}", pat)
    } else {
      pat.to_string()
    }
  };

  // Rule or list of rules: at each position, try each rule's LHS pattern;
  // on a match, emit the (capture-substituted) RHS and advance past it.
  if let Some(rules) = extract_cases_rules(&args[1]) {
    let mut compiled: Vec<(regex::Regex, Vec<(String, String)>, Expr, bool)> =
      Vec::with_capacity(rules.len());
    for (pat, constraints, rhs, expand_dollar) in &rules {
      let re = compile_regex(&with_ci(pat)).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "Invalid string pattern: {}",
          e
        ))
      })?;
      compiled.push((re, constraints.clone(), rhs.clone(), *expand_dollar));
    }

    let mut result: Vec<Expr> = Vec::new();
    let mut i = 0;
    while i < s.len() && result.len() < max_count {
      let mut matched = false;
      for (re, constraints, rhs, expand_dollar) in &compiled {
        if let Some(caps) = re.captures_at(&s, i)
          && let Some(m) = caps.get(0)
          && m.start() == i
          && !m.as_str().is_empty()
          && constraints.iter().all(|(orig, dup)| {
            match (caps.name(orig), caps.name(dup)) {
              (Some(a), Some(b)) => a.as_str() == b.as_str(),
              _ => true,
            }
          })
        {
          let mut substituted = substitute_captures(rhs, &caps);
          if *expand_dollar {
            substituted = expand_dollar_in_expr(&substituted, &caps);
          }
          let evaluated =
            crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          result.push(evaluated);
          // Overlapping matches advance one character past the start; the
          // default skips the whole matched substring.
          if overlaps {
            let ch = s[i..].chars().next().unwrap();
            i += ch.len_utf8();
          } else {
            i += m.len();
          }
          matched = true;
          break;
        }
      }
      if !matched {
        let ch = s[i..].chars().next().unwrap();
        i += ch.len_utf8();
      }
    }
    return Ok(Expr::List(result.into()));
  }

  // Bare conditional pattern: `patt /; test`. Match the inner pattern, then
  // require the `/;` test (with the captured names substituted in) to evaluate
  // to True before emitting the matched substring.
  if let Expr::FunctionCall { name, args: cargs } = &args[1]
    && name == "Condition"
    && cargs.len() == 2
    && let Some((regex_str, constraints)) =
      string_pattern_to_regex_with_state(&cargs[0])
  {
    let test = &cargs[1];
    let re = compile_regex(&with_ci(&regex_str)).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Invalid string pattern: {}",
        e
      ))
    })?;
    let mut out: Vec<Expr> = Vec::new();
    let mut i = 0;
    while i < s.len() && out.len() < max_count {
      let mut matched = false;
      if let Some(caps) = re.captures_at(&s, i)
        && let Some(m) = caps.get(0)
        && m.start() == i
        && !m.as_str().is_empty()
        && constraints.iter().all(|(orig, dup)| {
          match (caps.name(orig), caps.name(dup)) {
            (Some(a), Some(b)) => a.as_str() == b.as_str(),
            _ => true,
          }
        })
      {
        let test_sub = substitute_captures(test, &caps);
        let test_val = crate::evaluator::evaluate_expr_to_expr(&test_sub)?;
        if matches!(&test_val, Expr::Identifier(t) if t == "True") {
          out.push(Expr::String(m.as_str().to_string()));
          if overlaps {
            let ch = s[i..].chars().next().unwrap();
            i += ch.len_utf8();
          } else {
            i += m.len();
          }
          matched = true;
        }
      }
      if !matched {
        let ch = s[i..].chars().next().unwrap();
        i += ch.len_utf8();
      }
    }
    return Ok(Expr::List(out.into()));
  }

  // Try pattern-based matching first
  if let Some((regex_str, constraints)) =
    string_pattern_to_regex_with_state(&args[1])
  {
    let re = compile_regex(&with_ci(&regex_str)).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "Invalid string pattern: {}",
        e
      ))
    })?;
    let matches: Vec<Expr> = if overlaps {
      // Overlapping: emit a match for every char start position where the
      // (anchored) pattern matches, satisfying any back-reference constraints.
      let mut out: Vec<Expr> = Vec::new();
      for (idx, _) in s.char_indices() {
        if out.len() >= max_count {
          break;
        }
        if let Some(caps) = re.captures_at(&s, idx)
          && let Some(m) = caps.get(0)
          && m.start() == idx
          && !m.as_str().is_empty()
          && constraints.iter().all(|(orig, dup)| {
            match (caps.name(orig), caps.name(dup)) {
              (Some(a), Some(b)) => a.as_str() == b.as_str(),
              _ => true,
            }
          })
        {
          out.push(Expr::String(m.as_str().to_string()));
        }
      }
      out
    } else if constraints.is_empty() {
      re.find_iter(&s)
        .take(max_count)
        .map(|m| Expr::String(m.as_str().to_string()))
        .collect()
    } else {
      // Non-overlapping match with back-reference constraints. Scan position
      // by position (like the rules branch above) so that when the match at
      // one start fails its constraints, the next start is still tried —
      // `captures_iter` would instead greedily consume those characters and
      // skip a valid later match (e.g. miss the "ff" in "abcdeff").
      let mut out: Vec<Expr> = Vec::new();
      let mut i = 0;
      while i < s.len() && out.len() < max_count {
        if let Some(caps) = re.captures_at(&s, i)
          && let Some(m) = caps.get(0)
          && m.start() == i
          && !m.as_str().is_empty()
          && constraints.iter().all(|(orig, dup)| {
            match (caps.name(orig), caps.name(dup)) {
              (Some(a), Some(b)) => a.as_str() == b.as_str(),
              _ => true,
            }
          })
        {
          out.push(Expr::String(m.as_str().to_string()));
          i += m.len();
        } else {
          let ch = s[i..].chars().next().unwrap();
          i += ch.len_utf8();
        }
      }
      out
    };
    return Ok(Expr::List(matches.into()));
  }

  // Fall back to literal string matching. Under IgnoreCase, search case-
  // insensitively and emit the actual matched substring (which may differ
  // from the pattern in case).
  let patt = expr_to_str(&args[1])?;
  if ignore_case && !patt.is_empty() {
    let re =
      compile_regex(&format!("(?i){}", regex::escape(&patt))).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "Invalid string pattern: {}",
          e
        ))
      })?;
    let matches: Vec<Expr> = re
      .find_iter(&s)
      .take(max_count)
      .map(|m| Expr::String(m.as_str().to_string()))
      .collect();
    return Ok(Expr::List(matches.into()));
  }

  let mut matches = Vec::new();
  let mut start = 0;
  while let Some(pos) = s[start..].find(&patt) {
    matches.push(Expr::String(patt.clone()));
    if matches.len() >= max_count {
      break;
    }
    // Overlapping advances one character past the match start.
    start = if overlaps {
      let match_start = start + pos;
      match_start + s[match_start..].chars().next().map_or(1, char::len_utf8)
    } else {
      start + pos + patt.len()
    };
  }

  Ok(Expr::List(matches.into()))
}

/// Extract a list of (pattern_regex, back_reference_constraints,
/// replacement_expr) tuples from a rule or list of rules. Returns None if
/// the expression is not a rule or rule list, or if any pattern cannot be
/// converted to a regex.
fn extract_cases_rules(
  expr: &Expr,
) -> Option<Vec<(String, Vec<(String, String)>, Expr, bool)>> {
  fn one(e: &Expr) -> Option<(String, Vec<(String, String)>, Expr, bool)> {
    let (pat, rep) = match e {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => return None,
    };
    let (regex_str, constraints) = string_pattern_to_regex_with_state(pat)?;
    // RegularExpression patterns expand $0/$1/… in the replacement.
    let expand_dollar = matches!(
      pat,
      Expr::FunctionCall { name, .. } if name == "RegularExpression"
    );
    Some((regex_str, constraints, rep.clone(), expand_dollar))
  }
  match expr {
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => Some(vec![one(expr)?]),
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|i| {
          matches!(i, Expr::Rule { .. } | Expr::RuleDelayed { .. })
        }) =>
    {
      let mut v = Vec::with_capacity(items.len());
      for it in items {
        v.push(one(it)?);
      }
      Some(v)
    }
    _ => None,
  }
}

/// ToString[expr] - convert expression to string
/// Render `NumberForm[x, n]` to a string: a real `x` is rounded to `n`
/// significant figures and shown with a trailing decimal point and no padding
/// zeros (`3.14`, `3.`, `1200.`); an integer `x` is shown unchanged.
fn number_form_to_string(x: &Expr, n: i64) -> Option<String> {
  // Integer argument: shown unchanged, ignoring the precision.
  if let Expr::Integer(i) = x {
    return Some(i.to_string());
  }
  let f = match x {
    Expr::Real(f) => *f,
    _ => return None,
  };
  if n < 1 {
    return None;
  }
  if f == 0.0 {
    return Some("0.".to_string());
  }
  let neg = f < 0.0;
  let ax = f.abs();
  let m = ax.log10().floor() as i64;
  // Round to n significant figures.
  let factor = 10f64.powi((n - 1 - m) as i32);
  let rounded = (ax * factor).round() / factor;
  // Rounding can bump the magnitude (e.g. 9.99 -> 10), so recompute it.
  let m2 = if rounded == 0.0 {
    0
  } else {
    rounded.log10().floor() as i64
  };
  let decimals = (n - 1 - m2).max(0) as usize;
  let mut s = format!("{:.*}", decimals, rounded);
  if s.contains('.') {
    // Drop padding zeros but keep the trailing point (3.00 -> 3.).
    while s.ends_with('0') {
      s.pop();
    }
  } else {
    s.push('.');
  }
  Some(if neg { format!("-{s}") } else { s })
}

/// Render the `n`-significant-figure form of `NumberForm[x, n]` (and the
/// single-argument `NumberForm[x]`, where `n = 6`). A real whose decimal
/// exponent is `>= 6` or `<= -6` (|x| >= 10^6 or |x| < 10^-5) switches to 2D
/// scientific notation, matching wolframscript's default ExponentFunction;
/// integers and in-range reals use the fixed-notation renderer.
fn number_form_render(x: &Expr, n: i64) -> Option<String> {
  let (mantissa, exp) = number_form_parts(x, n)?;
  Some(sci_parts_to_2d_string(&mantissa, exp))
}

/// Decompose `NumberForm[x, n]` (and single-argument `NumberForm[x]`, where
/// `n = 6`) into a mantissa string and optional base-10 exponent. A real whose
/// decimal exponent (of the value rounded to `n` significant figures) is `>= 6`
/// or `<= -6` switches to scientific notation (matching wolframscript's default
/// ExponentFunction) and yields a `Some(exp)`; integers and in-range reals
/// render in fixed notation with a `None` exponent.
pub(crate) fn number_form_parts(
  x: &Expr,
  n: i64,
) -> Option<(String, Option<i64>)> {
  if let Expr::Real(f) = x
    && *f != 0.0
    && n >= 1
  {
    let ax = f.abs();
    let m0 = ax.log10().floor() as i64;
    // Use the exponent of the value rounded to n significant figures, so
    // e.g. 999999.9 rounding up to 10^6 is detected as out of range.
    let factor = 10f64.powi((n - 1 - m0) as i32);
    let rounded = (ax * factor).round() / factor;
    let m = if rounded == 0.0 {
      0
    } else {
      rounded.log10().floor() as i64
    };
    if m >= 6 || m <= -6 {
      return scientific_form_parts(x, n);
    }
  }
  Some((number_form_to_string(x, n)?, None))
}

/// Render `NumberForm[x, {n, f}]` as a `(mantissa, exponent)` pair. The fixed
/// `f`-decimal-places renderer never produces a `× 10^e` factor, so the
/// exponent is always `None`.
pub(crate) fn number_form_fixed_parts(
  x: &Expr,
  f: i64,
) -> Option<(String, Option<i64>)> {
  Some((number_form_fixed_to_string(x, f)?, None))
}

/// Group a digit string into blocks of `block`, counting from the RIGHT (used
/// for the integer part), joined by `sep`. E.g. ("1234567", 3, ",") -> "1,234,567".
fn group_digits_from_right(digits: &str, block: usize, sep: &str) -> String {
  let chars: Vec<char> = digits.chars().collect();
  if chars.is_empty() {
    return String::new();
  }
  let mut groups: Vec<String> = Vec::new();
  let mut i = chars.len();
  while i > 0 {
    let start = i.saturating_sub(block);
    groups.push(chars[start..i].iter().collect());
    i = start;
  }
  groups.reverse();
  groups.join(sep)
}

/// Group a digit string into blocks of `block`, counting from the LEFT (used
/// for the fractional part), joined by `sep`. E.g. ("23457", 3, " ") -> "234 57".
fn group_digits_from_left(digits: &str, block: usize, sep: &str) -> String {
  let chars: Vec<char> = digits.chars().collect();
  if chars.is_empty() {
    return String::new();
  }
  let mut groups: Vec<String> = Vec::new();
  let mut i = 0;
  while i < chars.len() {
    let end = (i + block).min(chars.len());
    groups.push(chars[i..end].iter().collect());
    i = end;
  }
  groups.join(sep)
}

/// Render `NumberForm[x, DigitBlock -> n]` (with optional positional precision
/// and `NumberSeparator`): group the integer-part digits into blocks of `n`
/// from the right and the fractional-part digits into blocks of `n` from the
/// left, with the given separators (default `,` integer side, ` ` fractional
/// side). Returns None for cases wolframscript renders in scientific notation
/// (integer part wider than the precision), so the caller keeps the symbolic
/// form.
fn number_form_digit_block_to_string(
  x: &Expr,
  prec: i64,
  block: i64,
  int_sep: &str,
  frac_sep: &str,
) -> Option<String> {
  if block < 1 {
    return None;
  }
  let block = block as usize;

  // A real whose decimal exponent (of the value rounded to `prec` significant
  // figures) is >= 6 or <= -6 switches to 2D scientific notation, with the
  // mantissa digit-blocked (e.g. NumberForm[1234567., DigitBlock -> 3] ->
  // "1.234 57 × 10^6"). This matches the non-DigitBlock NumberForm threshold.
  if let Expr::Real(f) = x
    && *f != 0.0
    && prec >= 1
  {
    let ax = f.abs();
    let m0 = ax.log10().floor() as i64;
    let factor = 10f64.powi((prec - 1 - m0) as i32);
    let rounded = (ax * factor).round() / factor;
    let m = if rounded == 0.0 {
      0
    } else {
      rounded.log10().floor() as i64
    };
    if m >= 6 || m <= -6 {
      let prec_digits = (prec - 1).max(0) as usize;
      let formatted = format!("{:.*e}", prec_digits, ax);
      let (mantissa_raw, exp_raw) = formatted.split_once('e')?;
      let exp: i64 = exp_raw.parse().ok()?;
      let (int_part, frac_part) = match mantissa_raw.split_once('.') {
        Some((ip, fp)) => {
          (ip.to_string(), fp.trim_end_matches('0').to_string())
        }
        None => (mantissa_raw.to_string(), String::new()),
      };
      let mut mantissa = group_digits_from_right(&int_part, block, int_sep);
      mantissa.push('.');
      if !frac_part.is_empty() {
        mantissa.push_str(&group_digits_from_left(&frac_part, block, frac_sep));
      }
      if *f < 0.0 {
        mantissa = format!("-{mantissa}");
      }
      if exp == 0 {
        return Some(mantissa);
      }
      let line2 = format!("{mantissa} \u{00d7} 10");
      let indent = " ".repeat(line2.chars().count());
      return Some(format!("{indent}{exp}\n{line2}"));
    }
  }

  let (neg, int_digits, frac_digits, real_no_frac) = match x {
    Expr::Integer(i) => {
      (*i < 0, i.unsigned_abs().to_string(), String::new(), false)
    }
    Expr::Real(f) => {
      let f = *f;
      let s = number_form_to_string(&Expr::Real(f.abs()), prec)?;
      match s.split_once('.') {
        Some((ip, fp)) => {
          (f < 0.0, ip.to_string(), fp.to_string(), fp.is_empty())
        }
        None => (f < 0.0, s, String::new(), true),
      }
    }
    _ => return None,
  };
  let mut out = String::new();
  if neg {
    out.push('-');
  }
  out.push_str(&group_digits_from_right(&int_digits, block, int_sep));
  if !frac_digits.is_empty() {
    out.push('.');
    out.push_str(&group_digits_from_left(&frac_digits, block, frac_sep));
  } else if real_no_frac {
    // A real with no fractional digits keeps its trailing point ("123,457.").
    out.push('.');
  }
  Some(out)
}

/// Render `ScientificForm[x]` / `ScientificForm[x, n]` to a string: a real `x`
/// is shown as `mantissa × 10` with the exponent placed as a superscript on the
/// line above (e.g. `1.23457 × 10` over `4`). Default `n` is 6 significant
/// figures. An exponent of 0 collapses to just the mantissa (`5.`). An integer
/// `x` is shown unchanged.
fn scientific_form_to_string(x: &Expr, n: i64) -> Option<String> {
  let (mantissa, exp) = scientific_form_parts(x, n)?;
  Some(sci_parts_to_2d_string(&mantissa, exp))
}

/// Decompose `ScientificForm[x]` / `ScientificForm[x, n]` into its mantissa
/// string and (optional) base-10 exponent. The exponent is `None` when the
/// value renders without a `× 10^e` factor (integer argument, or an exponent of
/// zero). This drives both the `ToString` 2D text renderer and the box-form
/// renderer used by the Playground/Studio SVG output.
pub(crate) fn scientific_form_parts(
  x: &Expr,
  n: i64,
) -> Option<(String, Option<i64>)> {
  // Integer argument: shown unchanged, ignoring the precision.
  if let Expr::Integer(i) = x {
    return Some((i.to_string(), None));
  }
  let v = match x {
    Expr::Real(f) => *f,
    _ => return None,
  };
  let n = if n < 1 { 1 } else { n };
  let prec = (n - 1) as usize;
  // Rust's `{:e}` formatter produces a normalized mantissa (one nonzero digit
  // before the point) and renormalizes after rounding (9.9995e3 -> 1.00000e4).
  let formatted = format!("{:.*e}", prec, v);
  let (mantissa_raw, exp_raw) = formatted.split_once('e')?;
  let exp: i64 = exp_raw.parse().ok()?;
  // Drop padding zeros from the fractional part but keep the trailing point.
  let mantissa = match mantissa_raw.split_once('.') {
    Some((int_part, frac_part)) => {
      format!("{}.{}", int_part, frac_part.trim_end_matches('0'))
    }
    None => format!("{mantissa_raw}."),
  };
  if exp == 0 {
    return Some((mantissa, None));
  }
  Some((mantissa, Some(exp)))
}

/// Render a `(mantissa, exponent)` decomposition as the 2D text form used under
/// `ToString`: the exponent sits as a superscript on the line above the
/// `mantissa × 10` line. A `None` exponent collapses to just the mantissa.
fn sci_parts_to_2d_string(mantissa: &str, exp: Option<i64>) -> String {
  match exp {
    None => mantissa.to_string(),
    Some(e) => {
      let line2 = format!("{mantissa} \u{00d7} 10");
      let indent = " ".repeat(line2.chars().count());
      format!("{indent}{e}\n{line2}")
    }
  }
}

/// Horizontally concatenate multi-line text blocks, bottom-aligned: their last
/// lines share a baseline and shorter blocks are padded with blank lines on top.
/// Each block's lines are right-padded to the block's own max width so columns
/// stay aligned (e.g. an exponent sitting above its mantissa).
fn hcat_2d_blocks(blocks: &[String]) -> String {
  let split: Vec<Vec<&str>> =
    blocks.iter().map(|b| b.lines().collect()).collect();
  let height = split.iter().map(|l| l.len()).max().unwrap_or(0);
  let widths: Vec<usize> = split
    .iter()
    .map(|lines| lines.iter().map(|l| l.chars().count()).max().unwrap_or(0))
    .collect();
  let mut out_lines: Vec<String> = Vec::with_capacity(height);
  for row in 0..height {
    let mut line = String::new();
    for (bi, lines) in split.iter().enumerate() {
      let top_pad = height - lines.len();
      let cell = if row >= top_pad {
        lines[row - top_pad]
      } else {
        ""
      };
      line.push_str(cell);
      line
        .push_str(&" ".repeat(widths[bi].saturating_sub(cell.chars().count())));
    }
    out_lines.push(line.trim_end().to_string());
  }
  out_lines.join("\n")
}

/// Build the 2D text rendering of a list of already-rendered (possibly
/// multi-line) element strings: `{elem, elem, ...}` with the braces and commas
/// on the baseline, matching wolframscript's `ScientificForm[{...}]` output.
fn list_form_2d_string(elems: &[String]) -> String {
  let mut blocks: Vec<String> = Vec::with_capacity(elems.len() * 2 + 2);
  blocks.push("{".to_string());
  for (i, e) in elems.iter().enumerate() {
    if i > 0 {
      blocks.push(", ".to_string());
    }
    blocks.push(e.clone());
  }
  blocks.push("}".to_string());
  hcat_2d_blocks(&blocks)
}

/// Apply a single-value 2D form renderer to an argument, threading over a list
/// (these display forms are Listable). Returns `None` if any element is one the
/// renderer leaves unhandled.
fn render_form_threaded<F>(arg: &Expr, render: F) -> Option<String>
where
  F: Fn(&Expr) -> Option<String>,
{
  if let Expr::List(items) = arg {
    let parts: Option<Vec<String>> = items.iter().map(&render).collect();
    Some(list_form_2d_string(&parts?))
  } else {
    render(arg)
  }
}

/// Render `EngineeringForm[x]` / `EngineeringForm[x, n]` to a string: like
/// `ScientificForm` but the exponent is forced to a multiple of 3, so the
/// mantissa lies in `[1, 1000)` (e.g. `12.3457 × 10` over `3`). Default `n` is 6
/// significant figures. An exponent of 0 collapses to just the mantissa. An
/// integer `x` is shown unchanged.
fn engineering_form_to_string(x: &Expr, n: i64) -> Option<String> {
  let (mantissa, exp) = engineering_form_parts(x, n)?;
  Some(sci_parts_to_2d_string(&mantissa, exp))
}

/// Decompose `EngineeringForm[x]` / `EngineeringForm[x, n]` into its mantissa
/// string and (optional) base-10 exponent — like [`scientific_form_parts`] but
/// the exponent is forced to a multiple of 3 (mantissa in `[1, 1000)`).
pub(crate) fn engineering_form_parts(
  x: &Expr,
  n: i64,
) -> Option<(String, Option<i64>)> {
  // Integer argument: shown unchanged, ignoring the precision.
  if let Expr::Integer(i) = x {
    return Some((i.to_string(), None));
  }
  let v = match x {
    Expr::Real(f) => *f,
    _ => return None,
  };
  let n = if n < 1 { 1 } else { n };
  let prec = (n - 1) as usize;
  // Scientific decomposition (normalized mantissa, renormalized after rounding).
  let formatted = format!("{:.*e}", prec, v);
  let (mantissa_raw, exp_raw) = formatted.split_once('e')?;
  let e: i64 = exp_raw.parse().ok()?;
  // Engineering exponent is the largest multiple of 3 not exceeding `e`; the
  // mantissa's decimal point shifts right by `shift` places accordingly.
  let shift = e.rem_euclid(3);
  let eng_exp = e - shift;
  let neg = mantissa_raw.starts_with('-');
  let body = mantissa_raw.trim_start_matches('-');
  let mut digits: String = body.chars().filter(|c| *c != '.').collect();
  let int_len = (shift + 1) as usize;
  while digits.len() < int_len {
    digits.push('0');
  }
  let (ip, fp) = digits.split_at(int_len);
  let fp = fp.trim_end_matches('0');
  let mantissa = if fp.is_empty() {
    format!("{ip}.")
  } else {
    format!("{ip}.{fp}")
  };
  let mantissa = if neg {
    format!("-{mantissa}")
  } else {
    mantissa
  };
  if eng_exp == 0 {
    return Some((mantissa, None));
  }
  Some((mantissa, Some(eng_exp)))
}

/// Render `NumberForm[x, {n, f}]`: a real `x` shown with exactly `f` digits
/// after the decimal point (zero-padded), e.g. `3.0 -> 3.00`. An integer `x`
/// is shown unchanged.
fn number_form_fixed_to_string(x: &Expr, f: i64) -> Option<String> {
  if let Expr::Integer(i) = x {
    return Some(i.to_string());
  }
  let val = match x {
    Expr::Real(v) => *v,
    _ => return None,
  };
  if f < 0 {
    return None;
  }
  let f = f as usize;
  let mut s = format!("{:.*}", f, val);
  // `{:.0}` omits the decimal point; NumberForm keeps a trailing one (3.).
  if f == 0 {
    s.push('.');
  }
  Some(s)
}

/// Default (one-argument) `DecimalForm` rendering of a real: always decimal
/// notation, keeping the full integer part and rounding the fraction to machine
/// precision (~6 significant figures). `1234567.89 -> 1234568.`, `0.0001 ->
/// 0.0001`.
/// Number of digits to the left of the decimal point in the magnitude of a
/// real or integer value (always at least 1, so e.g. `0.5` counts as one).
/// Returns `None` for non-numeric expressions.
fn integer_digit_count(magnitude: &Expr) -> Option<i64> {
  let f = match magnitude {
    Expr::Integer(i) => i.unsigned_abs() as f64,
    Expr::Real(f) => f.abs(),
    _ => return None,
  };
  if f < 1.0 {
    Some(1)
  } else {
    Some(f.log10().floor() as i64 + 1)
  }
}

fn decimal_form_default(f: f64) -> String {
  if f == 0.0 {
    return "0.".to_string();
  }
  let neg = f < 0.0;
  let ax = f.abs();
  let m = ax.log10().floor() as i64;
  // Decimal places implied by 6 significant figures; never negative, so the
  // integer part is always shown in full.
  let decimals = (6 - 1 - m).max(0) as usize;
  let factor = 10f64.powi(decimals as i32);
  let rounded = (ax * factor).round() / factor;
  let mut s = format!("{:.*}", decimals, rounded);
  if s.contains('.') {
    while s.ends_with('0') {
      s.pop();
    }
  } else {
    s.push('.');
  }
  if neg { format!("-{s}") } else { s }
}

/// Render an integer in the given base (2..=36) using lowercase digits, keeping
/// a leading `-` for negatives. `255` in base 16 → `ff`, `-255` → `-ff`.
fn integer_to_base_string(n: i128, base: i128) -> String {
  if n == 0 {
    return "0".to_string();
  }
  const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
  let neg = n < 0;
  let mut value = n.unsigned_abs();
  let base = base as u128;
  let mut out: Vec<u8> = Vec::new();
  while value > 0 {
    out.push(DIGITS[(value % base) as usize]);
    value /= base;
  }
  if neg {
    out.push(b'-');
  }
  out.reverse();
  String::from_utf8(out).unwrap()
}

/// Render `TableForm[matrix]` (or a 1D vector) as the aligned text grid that
/// Wolfram produces under `ToString`. Columns are left-aligned and padded to the
/// widest cell in the column, joined by three spaces; rows are separated by a
/// blank line (`\n\n`); trailing whitespace is trimmed from every line.
/// Returns `None` for arguments that are not lists.
fn table_form_to_string(arg: &Expr) -> Option<String> {
  let Expr::List(items) = arg else {
    return None;
  };
  // A matrix is a non-empty list whose every element is itself a list.
  let is_matrix =
    !items.is_empty() && items.iter().all(|it| matches!(it, Expr::List(_)));
  if is_matrix {
    let rows: Vec<Vec<String>> = items
      .iter()
      .map(|row| match row {
        Expr::List(cells) => {
          cells.iter().map(crate::syntax::expr_to_output).collect()
        }
        _ => vec![],
      })
      .collect();
    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut col_w = vec![0usize; ncols];
    for row in &rows {
      for (j, cell) in row.iter().enumerate() {
        col_w[j] = col_w[j].max(cell.chars().count());
      }
    }
    let lines: Vec<String> = rows
      .iter()
      .map(|row| {
        let padded: Vec<String> = row
          .iter()
          .enumerate()
          .map(|(j, cell)| {
            let pad = col_w[j].saturating_sub(cell.chars().count());
            format!("{}{}", cell, " ".repeat(pad))
          })
          .collect();
        padded.join("   ").trim_end().to_string()
      })
      .collect();
    Some(lines.join("\n\n"))
  } else {
    // A flat vector renders one element per row.
    let lines: Vec<String> =
      items.iter().map(crate::syntax::expr_to_output).collect();
    Some(lines.join("\n\n"))
  }
}

/// Render `MatrixForm[matrix]` to a text grid. Unlike `TableForm` (which sizes
/// each column independently), every cell is padded to a single uniform width —
/// the widest cell anywhere in the matrix — left-aligned, with three-space
/// separators and a blank line between rows. A flat vector renders as a single
/// column, one element per row.
fn matrix_form_to_string(arg: &Expr) -> Option<String> {
  let Expr::List(items) = arg else {
    return None;
  };
  let is_matrix =
    !items.is_empty() && items.iter().all(|it| matches!(it, Expr::List(_)));
  if is_matrix {
    let rows: Vec<Vec<String>> = items
      .iter()
      .map(|row| match row {
        Expr::List(cells) => {
          cells.iter().map(crate::syntax::expr_to_output).collect()
        }
        _ => vec![],
      })
      .collect();
    // Single uniform column width: the widest cell anywhere in the matrix.
    let width = rows
      .iter()
      .flat_map(|r| r.iter())
      .map(|c| c.chars().count())
      .max()
      .unwrap_or(0);
    let lines: Vec<String> = rows
      .iter()
      .map(|row| {
        let padded: Vec<String> = row
          .iter()
          .map(|cell| {
            let pad = width.saturating_sub(cell.chars().count());
            format!("{}{}", cell, " ".repeat(pad))
          })
          .collect();
        padded.join("   ").trim_end().to_string()
      })
      .collect();
    Some(lines.join("\n\n"))
  } else {
    // A flat vector renders one element per row (single column).
    let lines: Vec<String> =
      items.iter().map(crate::syntax::expr_to_output).collect();
    Some(lines.join("\n\n"))
  }
}

/// Recursively remove `HoldForm[x]` wrappers, replacing each with its content.
/// HoldForm is transparent in OutputForm (so `ToString[HoldForm[1 + 1]]` is
/// `1 + 1` and `ToString[f[HoldForm[a + b]]]` is `f[a + b]`), but it is kept in
/// InputForm and the bare REPL echo — hence the stripping happens only in
/// ToString's OutputForm paths, not in the shared formatters. The content is
/// structurally re-emitted (never re-evaluated), so the held form is preserved.
fn strip_hold_form(e: &Expr) -> Expr {
  match e {
    Expr::FunctionCall { name, args }
      if name == "HoldForm" && args.len() == 1 =>
    {
      strip_hold_form(&args[0])
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(strip_hold_form).collect::<Vec<_>>().into(),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(strip_hold_form).collect::<Vec<_>>().into())
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(strip_hold_form(left)),
      right: Box::new(strip_hold_form(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(strip_hold_form(operand)),
    },
    other => other.clone(),
  }
}

pub fn to_string_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ToString expects 1 or 2 arguments".into(),
    ));
  }

  // The optional second argument must be a format symbol (InputForm,
  // OutputForm, TeXForm, …), never a number. wolframscript rejects e.g.
  // ToString[255, 2] with ToString::fmtval and returns the call unevaluated,
  // rather than silently ignoring the bogus format and stringifying the value.
  if args.len() == 2 {
    let is_numeric_form = matches!(
      &args[1],
      Expr::Integer(_)
        | Expr::Real(_)
        | Expr::BigInteger(_)
        | Expr::BigFloat(_, _)
    ) || matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Rational");
    if is_numeric_form {
      crate::emit_message(&format!(
        "ToString::fmtval: {} is not a valid format type.",
        crate::syntax::expr_to_output(&args[1])
      ));
      return Ok(Expr::FunctionCall {
        name: "ToString".to_string(),
        args: args.to_vec().into(),
      });
    }
  }

  // When the requested form is structural InputForm, display-directive
  // wrappers (TableForm, Column, BaseForm, NumberForm, AccountingForm,
  // DecimalForm, PaddedForm) are NOT rendered to their 2D text form — they
  // keep the symbolic head, matching wolframscript:
  // `ToString[Column[{1,2,3}], InputForm]` -> "Column[{1, 2, 3}]". The
  // InputForm branch further down handles this via expr_to_input_form.
  let is_input_form = args.len() == 2
    && matches!(&args[1], Expr::Identifier(f) if f == "InputForm");
  // TeXForm must not take the 2D-OutputForm intercepts below (e.g. Subscript):
  // it is rendered by the dedicated TeX path further down.
  let is_tex_form = args.len() == 2
    && matches!(&args[1], Expr::Identifier(f) if f == "TeXForm");

  // TableForm[matrix] — render as an aligned text grid (left-aligned columns
  // padded to the widest cell, three-space separators, blank line between rows).
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "TableForm"
    && !is_input_form
    && inner_args.len() == 1
    && let Some(rendered) = table_form_to_string(&inner_args[0])
  {
    return Ok(Expr::String(rendered));
  }

  // Grid[matrix] — under ToString, wolframscript renders Grid exactly like
  // TableForm (left-aligned columns padded to the widest cell, three-space
  // separators, blank line between rows). Only the bare one-argument form is
  // handled; option arguments (Frame, alignment, ...) fall through.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "Grid"
    && !is_input_form
    && inner_args.len() == 1
    && let Some(rendered) = table_form_to_string(&inner_args[0])
  {
    return Ok(Expr::String(rendered));
  }

  // MatrixForm[matrix] — like TableForm but with a single uniform column width
  // (the widest cell anywhere in the matrix) rather than per-column widths.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "MatrixForm"
    && !is_input_form
    && inner_args.len() == 1
    && let Some(rendered) = matrix_form_to_string(&inner_args[0])
  {
    return Ok(Expr::String(rendered));
  }

  // Column[{e1, e2, ...}] / ColumnForm[{e1, e2, ...}] — stack the elements one
  // per line (single newline). Trailing alignment/option arguments do not
  // affect the plain-text form. Column keeps its symbolic head under InputForm;
  // ColumnForm (the legacy directive) renders to text under every form.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && ((name == "Column" && !is_input_form) || name == "ColumnForm")
    && !inner_args.is_empty()
    && let Expr::List(items) = &inner_args[0]
  {
    let lines: Vec<String> =
      items.iter().map(crate::syntax::expr_to_output).collect();
    return Ok(Expr::String(lines.join("\n")));
  }

  // BaseForm[n, b] — render the integer n in base b with the base shown as a
  // subscript on the line below (e.g. `ff` over `  16`). Base 10 shows no
  // subscript. Negative numbers keep their sign on the digit line.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "BaseForm"
    && !is_input_form
    && inner_args.len() == 2
    && let (Expr::Integer(n), Expr::Integer(b)) =
      (&inner_args[0], &inner_args[1])
    && (2..=36).contains(b)
  {
    let digits = integer_to_base_string(*n, *b);
    let rendered = if *b == 10 {
      digits
    } else {
      let indent = " ".repeat(digits.chars().count());
      format!("{}\n{}{}", digits, indent, b)
    };
    return Ok(Expr::String(rendered));
  }

  // Subscript[base, sub] / Superscript[base, sup] — under ToString these
  // render in 2D OutputForm: the script sits on a separate line, indented by
  // the width of the base. `ToString[Subscript[x, 2]]` → "x\n 2";
  // `ToString[Superscript[x, 2]]` → " 2\nx" (script above the base). The
  // InputForm target keeps the head literal (`"Subscript[x, 2]"`) and falls
  // through to the generic path below.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && (name == "Subscript" || name == "Superscript")
    && !is_input_form
    && !is_tex_form
    && inner_args.len() == 2
  {
    let base_str = to_string_default_form(&inner_args[0]);
    let script_str = to_string_default_form(&inner_args[1]);
    let indent = " ".repeat(base_str.chars().count());
    let rendered = if name == "Subscript" {
      format!("{}\n{}{}", base_str, indent, script_str)
    } else {
      format!("{}{}\n{}", indent, script_str, base_str)
    };
    return Ok(Expr::String(rendered));
  }

  // NumberForm[x, DigitBlock -> n, ...] — group digits into blocks. Detected by
  // the presence of a `DigitBlock` option among the arguments.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "NumberForm"
    && !is_input_form && inner_args.iter().any(|a| {
    matches!(
      a,
      Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "DigitBlock")
    )
  }) {
    // Defaults: comma between integer groups, space between fractional groups.
    let mut block: Option<i64> = None;
    let mut int_sep = ",".to_string();
    let mut frac_sep = " ".to_string();
    for a in inner_args.iter() {
      if let Expr::Rule {
        pattern,
        replacement,
      } = a
        && let Expr::Identifier(opt) = pattern.as_ref()
      {
        match opt.as_str() {
          "DigitBlock" => {
            if let Expr::Integer(n) = replacement.as_ref() {
              block = Some(*n as i64);
            }
          }
          "NumberSeparator" => match replacement.as_ref() {
            Expr::String(s) => {
              int_sep = s.clone();
              frac_sep = s.clone();
            }
            Expr::List(items) if items.len() == 2 => {
              if let Expr::String(s) = &items[0] {
                int_sep = s.clone();
              }
              if let Expr::String(s) = &items[1] {
                frac_sep = s.clone();
              }
            }
            _ => {}
          },
          _ => {}
        }
      }
    }
    let positional: Vec<&Expr> = inner_args
      .iter()
      .filter(|a| !matches!(a, Expr::Rule { .. }))
      .collect();
    let prec = match positional.get(1) {
      Some(Expr::Integer(n)) => *n as i64,
      _ => 6,
    };
    if let (Some(x), Some(block)) = (positional.first(), block)
      && let Some(rendered) =
        number_form_digit_block_to_string(x, prec, block, &int_sep, &frac_sep)
    {
      return Ok(Expr::String(rendered));
    }
  }

  // NumberForm[x, n, NumberPadding -> {p1, p2}] — pad the rendered number to a
  // fixed field width. For the integer-precision form the whole rendering
  // (sign and decimal point included) is left-padded with p1 to a total of
  // n + 1 digit positions, e.g. NumberForm[1.5, 3, {"0", " "}] -> "001.5",
  // NumberForm[42, 3, …] -> "0042", NumberForm[-1.5, 3, …] -> "0-1.5".
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "NumberForm"
    && !is_input_form
    && let Some(Expr::List(pad)) = inner_args.iter().find_map(|a| match a {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "NumberPadding") => {
        Some(replacement.as_ref())
      }
      _ => None,
    })
    && pad.len() == 2
  {
    let p1 = match &pad[0] {
      Expr::String(s) => s.clone(),
      _ => " ".to_string(),
    };
    let positional: Vec<&Expr> = inner_args
      .iter()
      .filter(|a| !matches!(a, Expr::Rule { .. }))
      .collect();
    if let (Some(x), Some(Expr::Integer(n))) =
      (positional.first(), positional.get(1))
      && let Some(rendered) = number_form_to_string(x, *n as i64)
    {
      let has_dot = rendered.contains('.');
      let width = (*n as usize + 1) + usize::from(has_dot);
      let pad_count = width.saturating_sub(rendered.chars().count());
      return Ok(Expr::String(format!("{}{}", p1.repeat(pad_count), rendered)));
    }
  }

  // NumberForm[x, n] — render x to n significant figures.
  // NumberForm[x, {n, f}] — render x with exactly f digits after the decimal
  // point (zero-padded).
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "NumberForm"
    && !is_input_form
    && inner_args.len() == 2
  {
    let rendered = match &inner_args[1] {
      Expr::Integer(n) => render_form_threaded(&inner_args[0], |x| {
        number_form_render(x, *n as i64)
      }),
      Expr::List(spec) if spec.len() == 2 => match &spec[1] {
        Expr::Integer(f) => render_form_threaded(&inner_args[0], |x| {
          number_form_fixed_to_string(x, *f as i64)
        }),
        _ => None,
      },
      _ => None,
    };
    if let Some(rendered) = rendered {
      return Ok(Expr::String(rendered));
    }
  }

  // NumberForm[x] — no precision spec. wolframscript renders this exactly like
  // NumberForm[x, 6] (the machine-precision display default: 6 significant
  // figures), e.g. NumberForm[12345.678] -> "12345.7", NumberForm[-3.5] ->
  // "-3.5", NumberForm[42] -> "42".
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "NumberForm"
    && !is_input_form
    && inner_args.len() == 1
    && let Some(rendered) =
      render_form_threaded(&inner_args[0], |x| number_form_render(x, 6))
  {
    return Ok(Expr::String(rendered));
  }

  // ScientificForm[x] / ScientificForm[x, n] — scientific (mantissa × 10^exp)
  // rendering. Default 6 significant figures; the exponent is shown as a
  // superscript on the line above. Exponent 0 collapses to just the mantissa.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "ScientificForm"
    && !is_input_form
    && (inner_args.len() == 1 || inner_args.len() == 2)
  {
    let n = match inner_args.get(1) {
      None => Some(6),
      Some(Expr::Integer(n)) => Some(*n as i64),
      _ => None,
    };
    if let Some(n) = n
      && let Some(rendered) = render_form_threaded(&inner_args[0], |x| {
        scientific_form_to_string(x, n)
      })
    {
      return Ok(Expr::String(rendered));
    }
  }

  // EngineeringForm[x] / EngineeringForm[x, n] — like ScientificForm but the
  // exponent is forced to a multiple of 3 (mantissa in [1, 1000)).
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "EngineeringForm"
    && !is_input_form
    && (inner_args.len() == 1 || inner_args.len() == 2)
  {
    let n = match inner_args.get(1) {
      None => Some(6),
      Some(Expr::Integer(n)) => Some(*n as i64),
      _ => None,
    };
    if let Some(n) = n
      && let Some(rendered) = render_form_threaded(&inner_args[0], |x| {
        engineering_form_to_string(x, n)
      })
    {
      return Ok(Expr::String(rendered));
    }
  }

  // AccountingForm[x] / AccountingForm[x, n] — like NumberForm but negatives
  // are shown in parentheses, e.g. -1234.5 -> (1234.5). A `DigitBlock` option
  // groups the digits (e.g. AccountingForm[-1234.5, DigitBlock->3] -> (1,234.5));
  // unlike NumberForm, AccountingForm always renders in full decimal.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "AccountingForm"
    && !is_input_form
    && !inner_args.is_empty()
  {
    let positional: Vec<&Expr> = inner_args
      .iter()
      .filter(|a| !matches!(a, Expr::Rule { .. }))
      .collect();
    // Parse DigitBlock / NumberSeparator options.
    let mut block: Option<usize> = None;
    let mut int_sep = ",".to_string();
    let mut frac_sep = " ".to_string();
    for a in inner_args.iter() {
      if let Expr::Rule {
        pattern,
        replacement,
      } = a
        && let Expr::Identifier(opt) = pattern.as_ref()
      {
        match opt.as_str() {
          "DigitBlock" => {
            if let Expr::Integer(n) = replacement.as_ref()
              && *n >= 1
            {
              block = Some(*n as usize);
            }
          }
          "NumberSeparator" => match replacement.as_ref() {
            Expr::String(s) => {
              int_sep = s.clone();
              frac_sep = s.clone();
            }
            Expr::List(items) if items.len() == 2 => {
              if let Expr::String(s) = &items[0] {
                int_sep = s.clone();
              }
              if let Expr::String(s) = &items[1] {
                frac_sep = s.clone();
              }
            }
            _ => {}
          },
          _ => {}
        }
      }
    }
    let n = match positional.get(1) {
      None => None,
      Some(Expr::Integer(n)) => Some(*n as i64),
      _ => None,
    };
    // Reject a non-integer second positional argument by falling through.
    if let Some(x0) = positional.first()
      && (positional.len() == 1 || n.is_some())
    {
      let (is_neg, magnitude) = match x0 {
        Expr::Integer(i) => (*i < 0, Expr::Integer(i.abs())),
        Expr::Real(f) => (*f < 0.0, Expr::Real(f.abs())),
        _ => (false, (*x0).clone()),
      };
      // When the requested number of significant figures is lower than the
      // count of digits to the left of the decimal point, the trailing
      // integer digits are padded with zeros (e.g. 1234.5678 -> 1230. at
      // n = 3). wolframscript warns about this with `<Head>::reqsigz`.
      if let Some(n) = n
        && let Some(int_digits) = integer_digit_count(&magnitude)
        && n < int_digits
      {
        crate::emit_message(
          "AccountingForm::reqsigz: Requested number precision is lower \
           than number of digits shown; padding with zeros.",
        );
      }
      let mag_str = match n {
        Some(n) => number_form_to_string(&magnitude, n),
        // AccountingForm never uses scientific notation, so a large/small
        // Real is rendered in full decimal (like DecimalForm) rather than via
        // the default OutputForm `*^` notation.
        None => match &magnitude {
          Expr::Integer(i) => Some(i.to_string()),
          Expr::Real(f) => Some(decimal_form_default(*f)),
          _ => match to_string_ast(std::slice::from_ref(&magnitude)) {
            Ok(Expr::String(ref s)) => Some(s.clone()),
            _ => None,
          },
        },
      };
      if let Some(mut mag_str) = mag_str {
        if let Some(block) = block {
          mag_str = match mag_str.split_once('.') {
            Some((ip, fp)) => {
              let grouped_int = group_digits_from_right(ip, block, &int_sep);
              if fp.is_empty() {
                format!("{grouped_int}.")
              } else {
                format!(
                  "{}.{}",
                  grouped_int,
                  group_digits_from_left(fp, block, &frac_sep)
                )
              }
            }
            None => group_digits_from_right(&mag_str, block, &int_sep),
          };
        }
        return Ok(Expr::String(if is_neg {
          format!("({mag_str})")
        } else {
          mag_str
        }));
      }
    }
  }

  // DecimalForm[x] / DecimalForm[x, n] — decimal (non-scientific) rendering.
  // The one-argument form keeps the integer part exact and rounds the fraction
  // to machine precision; the two-argument form rounds to n significant
  // figures. Both always use decimal notation and keep the minus sign.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "DecimalForm"
    && !is_input_form
    && (inner_args.len() == 1 || inner_args.len() == 2)
  {
    let rendered = match inner_args.get(1) {
      None => match &inner_args[0] {
        Expr::Integer(i) => Some(i.to_string()),
        Expr::Real(f) => Some(decimal_form_default(*f)),
        _ => None,
      },
      Some(Expr::Integer(n)) => {
        number_form_to_string(&inner_args[0], *n as i64)
      }
      _ => None,
    };
    if let Some(rendered) = rendered {
      return Ok(Expr::String(rendered));
    }
  }

  // PaddedForm[expr, n] / PaddedForm[expr, {n, f}] — right-aligned
  // number rendering (width n+1 for the integer spec, n+2 for {n, f},
  // reserving sign/decimal-point columns like wolframscript)
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "PaddedForm"
    && !is_input_form
    && inner_args.len() == 2
    && let Some(rendered) =
      padded_form_to_string(&inner_args[0], &inner_args[1])
  {
    return Ok(Expr::String(rendered));
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

  // If the expression is TeXForm[inner], produce TeX representation.
  // When the inner expression's head has user-defined MakeBoxes or
  // Format rules, route through the box AST (so user delimiters like
  // `<~` / `~>` surface in the rendered TeX), otherwise use the
  // direct expression-to-TeX path.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "TeXForm"
    && inner_args.len() == 1
  {
    let inner = &inner_args[0];
    // `TeXForm[InputForm[expr]]` renders the InputForm string of `expr`
    // (with user `Format[…, InputForm]` rules applied) inside `\text{…}`,
    // with `{` / `}` escaped using TeX's `$\{$` / `$\}$` braces.
    if let Expr::FunctionCall {
      name: ifname,
      args: ifargs,
    } = inner
      && ifname == "InputForm"
      && ifargs.len() == 1
    {
      let formatted = crate::evaluator::dispatch::complex_and_special::apply_format_recursively(
        &ifargs[0],
        "InputForm",
      );
      // Use OutputForm-style rendering (strings without surrounding
      // quotes), then escape `{` / `}` using TeX's `$\{$` / `$\}$`.
      let input_text = crate::syntax::expr_to_output(&formatted);
      let escaped = input_text.replace('{', "$\\{$").replace('}', "$\\}$");
      return Ok(Expr::String(format!("\\text{{{}}}", escaped)));
    }
    if let Expr::FunctionCall { name: head, .. } = inner {
      let has_format = crate::evaluator::assignment::FORMAT_VALUES
        .with(|m| m.borrow().contains_key(head));
      let has_user_mb = crate::FUNC_DEFS.with(|m| {
        m.borrow().get("MakeBoxes").is_some_and(|entries| {
          entries.iter().any(|(_, conditions, _, _, _, _)| {
            // User MakeBoxes patterns like `MakeBoxes[F[x__], fmt_]`
            // are stored as `__StructuralPattern__[__sp0, F[x__]]` in
            // the conditions of the first slot. Check whether any
            // entry's structural pattern matches the inner head.
            conditions.first().is_some_and(|cond| {
              matches!(
                cond,
                Some(Expr::FunctionCall { name: cn, args: ca })
                  if cn == "__StructuralPattern__"
                  && ca.len() == 2
                  && matches!(
                    &ca[1],
                    Expr::FunctionCall { name: pn, .. } if pn == head
                  )
              )
            })
          })
        })
      });
      if has_format || has_user_mb {
        // Pass `TeXForm` as the target so the Format pre-check in
        // MakeBoxes only fires on `Format[…, TeXForm]` rules. User
        // MakeBoxes patterns use `fmt_` and match any form, so they
        // still apply.
        let mb_call = Expr::FunctionCall {
          name: "MakeBoxes".to_string(),
          args: vec![inner.clone(), Expr::Identifier("TeXForm".to_string())]
            .into(),
        };
        if let Ok(box_ast) = crate::evaluator::evaluate_expr_to_expr(&mb_call) {
          return Ok(Expr::String(box_ast_to_tex(&box_ast)));
        }
      }
    }
    return Ok(Expr::String(expr_to_tex(inner)));
  }

  // If the expression is OutputForm[inner], produce the OutputForm-rendered
  // text (e.g. `f'[x]` for derivative prime notation). Mirrors what
  // wolframscript prints when ToString unwraps the form wrapper.
  if let Expr::FunctionCall {
    name,
    args: inner_args,
  } = &args[0]
    && name == "OutputForm"
    && inner_args.len() == 1
  {
    return Ok(Expr::String(crate::syntax::expr_to_output_form_2d(
      &inner_args[0],
    )));
  }

  // `ToString[InputForm[expr]]` (single arg, default OutputForm rendering)
  // unwraps the InputForm head and renders the inner expression in
  // InputForm — wolframscript prints `a + b`, not `InputForm[a + b]`,
  // because the outer form is OutputForm and `InputForm[x]`'s OutputForm
  // is the InputForm of x.
  //
  // For the 2-arg form `ToString[InputForm[expr], InputForm]` the target
  // is structural InputForm, which keeps the `InputForm[…]` head visible
  // (`"InputForm[a + b]"`). Skip this unwrap path in that case; the
  // 2-arg branch below renders the wrapped expression via
  // `expr_to_input_form`, which preserves the wrapper.
  if args.len() == 1
    && let Expr::FunctionCall {
      name,
      args: inner_args,
    } = &args[0]
    && name == "InputForm"
    && inner_args.len() == 1
  {
    let formatted =
      crate::evaluator::dispatch::complex_and_special::apply_format_recursively(
        &inner_args[0],
        "InputForm",
      );
    return Ok(Expr::String(crate::syntax::expr_to_input_form(&formatted)));
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
        // InputForm: infix operators + quoted strings. Apply
        // user-defined `Format[head[…], InputForm]` rules bottom-up
        // first so e.g. `Format[F[x_, y_], InputForm] := {F[x], "In"}`
        // surfaces in the printed text.
        let formatted =
          crate::evaluator::dispatch::complex_and_special::apply_format_recursively(
            &args[0],
            "InputForm",
          );
        let s = crate::syntax::expr_to_input_form(&formatted);
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
        // Apply user-defined `Format[head[…], OutputForm]` rules
        // bottom-up before rendering, matching wolframscript.
        let formatted = strip_hold_form(
          &crate::evaluator::dispatch::complex_and_special::apply_format_recursively(
            &args[0],
            "OutputForm",
          ),
        );
        let s = crate::syntax::expr_to_output_form_2d(&formatted);
        return Ok(Expr::String(s));
      }
      "StandardForm" => {
        // Build the box AST via MakeBoxes (which dispatches user-defined
        // Format / MakeBoxes rules), then encode it using Wolfram's
        // box-syntax escape characters (`\!\(\*…\)`). The resulting
        // String displays as `DisplayForm[…]` in OutputForm — matching
        // what wolframscript prints for `ToString[expr, StandardForm]`.
        let make_boxes_call = Expr::FunctionCall {
          name: "MakeBoxes".to_string(),
          args: vec![
            args[0].clone(),
            Expr::Identifier("StandardForm".to_string()),
          ]
          .into(),
        };
        let box_ast = crate::evaluator::evaluate_expr_to_expr(&make_boxes_call)
          .unwrap_or_else(|_| args[0].clone());
        let box_inner_text = crate::syntax::expr_to_input_form(&box_ast);
        return Ok(Expr::String(format!(
          "{}{}{}{}{}",
          BOX_START, BOX_OPEN, BOX_SEP, box_inner_text, BOX_CLOSE
        )));
      }
      _ => {}
    }
  }
  // Other forms: fall through to default (OutputForm-like) behavior

  // ToString[FullForm[expr]] returns the structural FullForm string of `expr`,
  // matching wolframscript: `ToString[FullForm[(a*Sqrt[x])^-1]]` →
  // `Times[Power[a, -1], Power[x, Rational[-1, 2]]]`.
  if args.len() == 1
    && let Expr::FunctionCall {
      name,
      args: inner_args,
    } = &args[0]
    && name == "FullForm"
    && inner_args.len() == 1
  {
    return Ok(Expr::String(crate::functions::expr_form::render_full_form(
      &inner_args[0],
    )));
  }

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

  // Arbitrary-precision numbers: ToString drops the `\`p` precision
  // marker and returns just the truncated decimal expansion (matching
  // wolframscript: `ToString[N[Pi, 100]]` is the bare 101-char string).
  if let Expr::BigFloat(digits, prec) = &args[0] {
    let trimmed = digits.trim();
    let (sign, body) = if let Some(rest) = trimmed.strip_prefix('-') {
      ("-", rest)
    } else {
      ("", trimmed)
    };
    let dot = body.find('.');
    let int_part = match dot {
      Some(i) => &body[..i],
      None => body,
    };
    let frac_part = match dot {
      Some(i) => &body[i + 1..],
      None => "",
    };
    let int_zero = int_part.is_empty() || int_part.chars().all(|c| c == '0');
    // Number of "significant" digits already in the integer part.
    let int_sig = if int_zero {
      0
    } else {
      int_part.trim_start_matches('0').len()
    };
    let prec_digits = prec.round() as i64;
    let frac_keep = if prec_digits > 0 {
      let want = prec_digits - int_sig as i64;
      if want < 0 { 0 } else { want as usize }
    } else {
      frac_part.len()
    };
    let frac_truncated: String = frac_part.chars().take(frac_keep).collect();
    let int_display = if int_part.is_empty() { "0" } else { int_part };
    let s = if frac_truncated.is_empty() && dot.is_some() {
      format!("{}{}.", sign, int_display)
    } else if frac_truncated.is_empty() {
      format!("{}{}", sign, int_display)
    } else {
      format!("{}{}.{}", sign, int_display, frac_truncated)
    };
    return Ok(Expr::String(s));
  }

  // Default (no form or unrecognized form): OutputForm-like
  // Resolve any nested form wrappers (TeXForm, CForm, FortranForm) first,
  // matching wolframscript behavior where ToString extracts form conversions.
  // Then truncate machine-precision Reals to 6 significant digits — Wolfram's
  // ToString default for `Real` values, which is *not* the same as the REPL
  // /Print display (those keep full f64 precision).
  let resolved = strip_hold_form(&resolve_form_wrappers(&args[0]));
  let truncated = truncate_machine_reals_for_to_string(&resolved);
  let s = crate::syntax::expr_to_output(&truncated);
  Ok(Expr::String(s))
}

/// Format `expr` the way `ToString[expr]` (default OutputForm) does:
/// machine-precision Reals are shown at 6 significant digits (so the f64
/// noise in e.g. `0.47000000000000003` renders as `0.47`). Used for chart
/// labels, which Wolfram typesets in OutputForm rather than full precision.
pub fn to_string_default_form(expr: &Expr) -> String {
  crate::syntax::expr_to_output(&truncate_machine_reals_for_to_string(expr))
}

/// Round a machine-precision Real to 6 significant decimal digits — the
/// width wolframscript uses for `ToString[real]` on a default
/// (MachinePrecision) Real. Returns NaN/inf unchanged; 0.0 stays 0.0.
fn round_real_to_6_sig_digits(f: f64) -> f64 {
  if f == 0.0 || !f.is_finite() {
    return f;
  }
  // Round to 6 significant digits via scientific formatting. Multiplying by a
  // 10^k factor (which is not exactly representable in f64) loses precision —
  // e.g. 1.5e10 would round-trip to 1.4999999999999998e10. Formatting to 5
  // digits after the point in `e` notation rounds correctly.
  format!("{:.5e}", f).parse().unwrap_or(f)
}

/// Recursively replace each `Expr::Real(f)` in `expr` with its
/// 6-sig-digit-rounded counterpart, so the default ToString path
/// emits e.g. `15.8406` for the f64 value `15.840646417884168`.
/// Leaves BigFloat, Integer, and symbolic Expr nodes intact.
fn truncate_machine_reals_for_to_string(expr: &Expr) -> Expr {
  match expr {
    Expr::Real(f) => Expr::Real(round_real_to_6_sig_digits(*f)),
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(truncate_machine_reals_for_to_string)
        .collect(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(truncate_machine_reals_for_to_string)
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(truncate_machine_reals_for_to_string(left)),
      right: Box::new(truncate_machine_reals_for_to_string(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(truncate_machine_reals_for_to_string(operand)),
    },
    _ => expr.clone(),
  }
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
/// Render a box AST (typically the result of `MakeBoxes[expr,
/// StandardForm]`) as TeX. Used by `ToString[TeXForm[expr]]` when the
/// expression's head has user-defined MakeBoxes or Format rules so
/// custom box delimiters (`<~`, `~>`, etc.) surface in the rendered
/// TeX.
fn box_ast_to_tex(expr: &Expr) -> String {
  fn tex_special_chars(s: &str) -> String {
    s.replace('~', "\\sim ")
  }
  match expr {
    Expr::String(s) => {
      // Box-element strings whose contents are wrapped in literal `"`
      // characters represent user-supplied Wolfram strings — wrap
      // their unquoted contents in `\text{…}`. Strings ending in a
      // backtick (the precision/accuracy marker on Real / BigFloat
      // box elements) drop the trailing `\``. Everything else gets
      // TeX special-character translation (notably `~` → `\sim `).
      if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        format!("\\text{{{}}}", &s[1..s.len() - 1])
      } else if let Some(stripped) = s.strip_suffix('`') {
        tex_special_chars(stripped)
      } else {
        tex_special_chars(s)
      }
    }
    Expr::FunctionCall { name, args }
      if name == "RowBox" && args.len() == 1 =>
    {
      if let Expr::List(items) = &args[0] {
        items
          .iter()
          .map(box_ast_to_tex)
          .collect::<Vec<_>>()
          .join("")
      } else {
        box_ast_to_tex(&args[0])
      }
    }
    Expr::FunctionCall { name, args }
      if name == "FractionBox" && args.len() == 2 =>
    {
      format!(
        "\\frac{{{}}}{{{}}}",
        box_ast_to_tex(&args[0]),
        box_ast_to_tex(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "SqrtBox" && !args.is_empty() =>
    {
      format!("\\sqrt{{{}}}", box_ast_to_tex(&args[0]))
    }
    Expr::FunctionCall { name, args }
      if name == "SuperscriptBox" && args.len() == 2 =>
    {
      format!(
        "{}^{{{}}}",
        box_ast_to_tex(&args[0]),
        box_ast_to_tex(&args[1])
      )
    }
    Expr::FunctionCall { name, args }
      if name == "SubscriptBox" && args.len() == 2 =>
    {
      format!(
        "{}_{{{}}}",
        box_ast_to_tex(&args[0]),
        box_ast_to_tex(&args[1])
      )
    }
    _ => expr_to_tex(expr),
  }
}

/// Pad a BigFloat digit string with trailing zeros so the total
/// number of significant digits equals `prec`. Used by TeX/C/
/// Fortran rendering where the precision-tag suffix isn't shown
/// but the digit count should still reflect the stored precision.
/// `digits` is e.g. `-14.` or `3.14`; `prec` is the significant-
/// digit count. Returns `-14.0` for `("-14.", 3.0)`.
fn pad_bigfloat_to_precision(digits: &str, prec: f64) -> String {
  let prec_target = prec.round().max(0.0) as usize;
  let (sign, rest) = if let Some(r) = digits.strip_prefix('-') {
    ("-", r)
  } else {
    ("", digits)
  };
  // Split into integer and fractional parts around the decimal
  // point (if any).
  let (int_part, frac_part) = match rest.find('.') {
    Some(dp) => (&rest[..dp], &rest[dp + 1..]),
    None => (rest, ""),
  };
  // Significant digits in the integer part: skip leading zeros
  // unless the integer is just "0".
  let int_sig = if int_part == "0" {
    0
  } else {
    int_part.trim_start_matches('0').len()
  };
  let current_sig = int_sig + frac_part.len();
  if current_sig >= prec_target {
    return digits.to_string();
  }
  let pad = prec_target - current_sig;
  if rest.contains('.') {
    format!("{}{}.{}{}", sign, int_part, frac_part, "0".repeat(pad))
  } else {
    format!("{}{}.{}", sign, int_part, "0".repeat(pad))
  }
}

pub fn expr_to_tex(expr: &Expr) -> String {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  // HoldForm[x] is a display wrapper; render its content transparently.
  if let Expr::FunctionCall { name, args } = expr
    && name == "HoldForm"
    && args.len() == 1
  {
    return expr_to_tex(&args[0]);
  }
  // OutputForm[x] renders the content to its OutputForm text first, then
  // TeXForm wraps that in `\text{…}`. Wolfram prints
  // `b // OutputForm // TeXForm` as `\text{b}`.
  if let Expr::FunctionCall { name, args } = expr
    && name == "OutputForm"
    && args.len() == 1
  {
    let rendered = crate::syntax::expr_to_output_form_2d(&args[0]);
    return format!("\\text{{{}}}", rendered);
  }
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::Real(f) => crate::syntax::format_real(*f),
    Expr::BigFloat(digits, prec) => {
      // TeX rendering of a precision-tagged BigFloat uses the
      // plain decimal digits (no `…` precision marker), padded
      // with trailing zeros to show all `prec` significant
      // digits. wolframscript's `-14.`3 // TeXForm` → `-14.0`
      // (3 sig digits = "14" + "0").
      pad_bigfloat_to_precision(digits, *prec)
    }
    Expr::String(s) => format!("\\text{{{}}}", s),
    // Raw is the rendered output of OutputForm / 2d boxes — treat as text.
    Expr::Raw(s) => format!("\\text{{{}}}", s),
    Expr::Identifier(name) | Expr::Constant(name) => tex_identifier(name),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => format!("-{}", expr_to_tex(operand)),
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      operand,
    } => format!("\\neg {}", tex_logic_operand(operand, "Not", 40)),
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
    } => format!(
      "{}\\land {}",
      tex_logic_operand(left, "And", 20),
      tex_logic_operand(right, "And", 20)
    ),
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      left,
      right,
    } => format!(
      "{}\\lor {}",
      tex_logic_operand(left, "Or", 20),
      tex_logic_operand(right, "Or", 20)
    ),
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
  let mut numer_args: Vec<&Expr> = Vec::new();
  let mut denom: Vec<String> = Vec::new();
  for arg in factors {
    if let Some((base, pos_exp)) = as_neg_int_power(arg) {
      denom.push(tex_denom_factor(base, pos_exp));
    } else {
      numer_args.push(arg);
    }
  }

  // A `Plus`/`Minus` factor in a multi-factor product needs parentheses, e.g.
  // `a (b+c)`. A lone numerator factor (the whole numerator of a fraction, or
  // a single product term) is already grouped by its brace, so it must not.
  let multi = numer_args.len() > 1;
  let tex_needs_product_parens = |arg: &Expr| -> bool {
    use crate::syntax::BinaryOperator::{Minus, Plus};
    matches!(
      arg,
      Expr::BinaryOp {
        op: Plus | Minus,
        ..
      }
    ) || matches!(arg, Expr::FunctionCall { name, args }
        if name == "Plus" && args.len() >= 2)
  };
  let numer: Vec<String> = numer_args
    .iter()
    .map(|arg| {
      let tex = expr_to_tex(arg);
      if multi && tex_needs_product_parens(arg) {
        format!("({})", tex)
      } else {
        tex
      }
    })
    .collect();

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

  // Powers of trig/hyperbolic/log functions move the exponent onto the
  // function name: Sin[x]^2 -> \sin ^2(x), Sech[x]^2 -> \text{sech}^2(x).
  // (Inverse trig like ArcSin is excluded — it keeps the trailing ^2.)
  if let Expr::FunctionCall { name, args } = base
    && args.len() == 1
  {
    let exp_tex = expr_to_tex(exp);
    let exp_grp = if exp_tex.chars().count() == 1 {
      exp_tex
    } else {
      format!("{{{}}}", exp_tex)
    };
    let arg = expr_to_tex(&args[0]);
    if matches!(
      name.as_str(),
      "Sin"
        | "Cos"
        | "Tan"
        | "Cot"
        | "Sec"
        | "Csc"
        | "Sinh"
        | "Cosh"
        | "Tanh"
        | "Coth"
        | "Log"
    ) {
      return format!("\\{} ^{}({})", name.to_lowercase(), exp_grp, arg);
    }
    if matches!(name.as_str(), "Sech" | "Csch") {
      return format!("\\text{{{}}}^{}({})", name.to_lowercase(), exp_grp, arg);
    }
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

/// Classify a logical operator for TeX parenthesization, returning
/// `(kind, precedence)` for And/Or/Xor/Implies/Not (higher binds tighter).
fn tex_logic_op(expr: &Expr) -> Option<(&'static str, u8)> {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      "And" if args.len() >= 2 => Some(("And", 20)),
      "Or" if args.len() >= 2 => Some(("Or", 20)),
      "Xor" if args.len() >= 2 => Some(("Xor", 20)),
      "Implies" if args.len() == 2 => Some(("Implies", 10)),
      "Not" if args.len() == 1 => Some(("Not", 40)),
      _ => None,
    },
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      ..
    } => Some(("Not", 40)),
    Expr::BinaryOp {
      op: BinaryOperator::And,
      ..
    } => Some(("And", 20)),
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      ..
    } => Some(("Or", 20)),
    _ => None,
  }
}

/// Render a logical operand, parenthesizing it per Wolfram's TeXForm rules:
/// a child of strictly lower precedence is wrapped, as is an equal-precedence
/// child of a different kind (mixing And/Or/Xor) or any Implies under Implies.
fn tex_logic_operand(
  child: &Expr,
  parent_kind: &str,
  parent_prec: u8,
) -> String {
  let inner = expr_to_tex(child);
  if let Some((ck, cp)) = tex_logic_op(child) {
    let paren = cp < parent_prec
      || (cp == parent_prec && (ck != parent_kind || parent_kind == "Implies"));
    if paren {
      return format!("({})", inner);
    }
  }
  inner
}

/// Render an operand atomically, wrapping compound expressions (sums,
/// products, …) in parentheses. Atoms and genuine function calls render
/// bare; arithmetic-operator heads (Plus/Times/…) are parenthesized.
fn tex_atom_or_paren(expr: &Expr) -> String {
  let inner = expr_to_tex(expr);
  let atomic = match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::Identifier(_)
    | Expr::Constant(_)
    | Expr::String(_) => true,
    Expr::FunctionCall { name, .. } => !matches!(
      name.as_str(),
      "Plus" | "Times" | "Subtract" | "Divide" | "Power"
    ),
    _ => false,
  };
  if atomic {
    inner
  } else {
    format!("({})", inner)
  }
}

/// Map a domain symbol to its blackboard-bold TeX set, for Element[_, dom].
fn tex_set_domain(expr: &Expr) -> Option<&'static str> {
  if let Expr::Identifier(d) = expr {
    return match d.as_str() {
      "Reals" => Some("\\mathbb{R}"),
      "Integers" => Some("\\mathbb{Z}"),
      "Complexes" => Some("\\mathbb{C}"),
      "Rationals" => Some("\\mathbb{Q}"),
      "Primes" => Some("\\mathbb{P}"),
      _ => None,
    };
  }
  None
}

/// Handle function calls in TeX
fn tex_function_call(name: &str, args: &[Expr]) -> String {
  match name {
    // Logical operators (infix) with precedence-aware parenthesization.
    "And" if args.len() >= 2 => args
      .iter()
      .map(|a| tex_logic_operand(a, "And", 20))
      .collect::<Vec<_>>()
      .join("\\land "),
    "Or" if args.len() >= 2 => args
      .iter()
      .map(|a| tex_logic_operand(a, "Or", 20))
      .collect::<Vec<_>>()
      .join("\\lor "),
    "Xor" if args.len() >= 2 => args
      .iter()
      .map(|a| tex_logic_operand(a, "Xor", 20))
      .collect::<Vec<_>>()
      .join("\\veebar "),
    "Implies" if args.len() == 2 => format!(
      "{}\\Rightarrow {}",
      tex_logic_operand(&args[0], "Implies", 10),
      tex_logic_operand(&args[1], "Implies", 10)
    ),
    // Set membership: Element[x, Reals] -> x\in \mathbb{R}.
    "Element" if args.len() == 2 && tex_set_domain(&args[1]).is_some() => {
      let dom = tex_set_domain(&args[1]).unwrap();
      let lhs = if matches!(
        &args[0],
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Alternatives,
          ..
        }
      ) {
        format!("({})", expr_to_tex(&args[0]))
      } else {
        expr_to_tex(&args[0])
      };
      format!("{}\\in {}", lhs, dom)
    }
    // Max/Min/GCD use LaTeX operator names; LCM is plain text.
    "Max" if !args.is_empty() => format!(
      "\\max ({})",
      args.iter().map(expr_to_tex).collect::<Vec<_>>().join(",")
    ),
    "Min" if !args.is_empty() => format!(
      "\\min ({})",
      args.iter().map(expr_to_tex).collect::<Vec<_>>().join(",")
    ),
    "GCD" if !args.is_empty() => format!(
      "\\gcd ({})",
      args.iter().map(expr_to_tex).collect::<Vec<_>>().join(",")
    ),
    "LCM" if !args.is_empty() => format!(
      "\\text{{lcm}}({})",
      args.iter().map(expr_to_tex).collect::<Vec<_>>().join(",")
    ),
    // Mod[a, b] -> (a \bmod b), parenthesizing compound operands.
    "Mod" if args.len() == 2 => format!(
      "({} \\bmod {})",
      tex_atom_or_paren(&args[0]),
      tex_atom_or_paren(&args[1])
    ),
    // Norm[v] -> \| v\| (vector bars).
    "Norm" if args.len() == 1 => format!("\\| {}\\|", expr_to_tex(&args[0])),
    // Factorial2[n] -> n!! (typeset as \text{!!}).
    "Factorial2" if args.len() == 1 => {
      format!("{}\\text{{!!}}", tex_atom_or_paren(&args[0]))
    }
    // KroneckerDelta[i, j, ...] -> \delta _{i,j,...}.
    "KroneckerDelta" if !args.is_empty() => format!(
      "\\delta _{{{}}}",
      args.iter().map(expr_to_tex).collect::<Vec<_>>().join(",")
    ),
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
    // Remaining inverse trig (the primitives have a LaTeX command name).
    "ArcCot" | "ArcSec" | "ArcCsc" if args.len() == 1 => {
      format!(
        "\\{} ^{{-1}}({})",
        name[3..].to_lowercase(),
        expr_to_tex(&args[0])
      )
    }
    // Hyperbolic functions: sinh/cosh/tanh/coth are LaTeX primitives,
    // sech/csch are rendered with \text{}.
    "Sinh" | "Cosh" | "Tanh" | "Coth" if args.len() == 1 => {
      format!("\\{} ({})", name.to_lowercase(), expr_to_tex(&args[0]))
    }
    "Sech" | "Csch" if args.len() == 1 => {
      format!(
        "\\text{{{}}}({})",
        name.to_lowercase(),
        expr_to_tex(&args[0])
      )
    }
    // Inverse hyperbolic functions.
    "ArcSinh" | "ArcCosh" | "ArcTanh" | "ArcCoth" if args.len() == 1 => {
      format!(
        "\\{} ^{{-1}}({})",
        name[3..].to_lowercase(),
        expr_to_tex(&args[0])
      )
    }
    "ArcSech" | "ArcCsch" if args.len() == 1 => {
      format!(
        "\\text{{{}}}^{{-1}}({})",
        name[3..].to_lowercase(),
        expr_to_tex(&args[0])
      )
    }
    // Special functions with dedicated LaTeX notation.
    "Sign" if args.len() == 1 => {
      format!("\\text{{sgn}}({})", expr_to_tex(&args[0]))
    }
    "Gamma" if !args.is_empty() => {
      let inner: Vec<String> = args.iter().map(expr_to_tex).collect();
      format!("\\Gamma ({})", inner.join(","))
    }
    "Zeta" if args.len() == 1 => {
      format!("\\zeta ({})", expr_to_tex(&args[0]))
    }
    "Re" if args.len() == 1 => {
      format!("\\Re({})", expr_to_tex(&args[0]))
    }
    "Im" if args.len() == 1 => {
      format!("\\Im({})", expr_to_tex(&args[0]))
    }
    "Erf" if args.len() == 1 => {
      format!("\\text{{erf}}({})", expr_to_tex(&args[0]))
    }
    "Erfc" if args.len() == 1 => {
      format!("\\text{{erfc}}({})", expr_to_tex(&args[0]))
    }
    "Beta" if args.len() == 2 => {
      format!("B({},{})", expr_to_tex(&args[0]), expr_to_tex(&args[1]))
    }
    // Conjugate: postfix star, parenthesizing compound arguments.
    "Conjugate" if args.len() == 1 => {
      format!("{}^*", tex_atom_or_paren(&args[0]))
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
      // Pull a leading minus out of the fraction: -3/4 -> -\frac{3}{4}.
      let num = expr_to_tex(&args[0]);
      let den = expr_to_tex(&args[1]);
      if let Some(rest) = num.strip_prefix('-') {
        format!("-\\frac{{{}}}{{{}}}", rest, den)
      } else {
        format!("\\frac{{{}}}{{{}}}", num, den)
      }
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
      // TeX convention: a single-character subscript/superscript needs no
      // braces (e.g. `\int_a^b` vs `\int_{ab}^{...}`), matching wolframscript.
      fn brace_if_needed(s: &str) -> String {
        if s.chars().count() == 1 {
          s.to_string()
        } else {
          format!("{{{}}}", s)
        }
      }
      if let Expr::List(bounds) = &args[1]
        && bounds.len() == 3
      {
        return format!(
          "\\int_{}^{} {} \\, d{}",
          brace_if_needed(&expr_to_tex(&bounds[1])),
          brace_if_needed(&expr_to_tex(&bounds[2])),
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
    // Subscript[x, i, j, ...] -> x_{i, j, ...}
    "Subscript" if args.len() >= 2 => {
      let base = expr_to_tex(&args[0]);
      let idx_tex: Vec<String> = args[1..].iter().map(expr_to_tex).collect();
      if args.len() == 2 {
        let s = &idx_tex[0];
        if s.chars().count() == 1 {
          format!("{}_{}", base, s)
        } else {
          format!("{}_{{{}}}", base, s)
        }
      } else {
        format!("{}_{{{}}}", base, idx_tex.join(", "))
      }
    }
    // Superscript[x, n] -> x^n or x^{n}
    "Superscript" if args.len() == 2 => {
      let base = expr_to_tex(&args[0]);
      let exp = expr_to_tex(&args[1]);
      if exp.chars().count() == 1 {
        format!("{}^{}", base, exp)
      } else {
        format!("{}^{{{}}}", base, exp)
      }
    }
    // Subsuperscript[x, b, c] -> x_b^c (with brace wrapping for multi-char parts)
    "Subsuperscript" if args.len() == 3 => {
      let base = expr_to_tex(&args[0]);
      let sub = expr_to_tex(&args[1]);
      let sup = expr_to_tex(&args[2]);
      let sub_part = if sub.chars().count() == 1 {
        format!("_{}", sub)
      } else {
        format!("_{{{}}}", sub)
      };
      let sup_part = if sup.chars().count() == 1 {
        format!("^{}", sup)
      } else {
        format!("^{{{}}}", sup)
      };
      format!("{}{}{}", base, sub_part, sup_part)
    }
    // Floor[x] -> \lfloor x\rfloor, Ceiling[x] -> \lceil x\rceil.
    "Floor" if args.len() == 1 => {
      format!("\\lfloor {}\\rfloor", expr_to_tex(&args[0]))
    }
    "Ceiling" if args.len() == 1 => {
      format!("\\lceil {}\\rceil", expr_to_tex(&args[0]))
    }
    // Binomial[n, k] -> \binom{n}{k}.
    "Binomial" if args.len() == 2 => {
      format!(
        "\\binom{{{}}}{{{}}}",
        expr_to_tex(&args[0]),
        expr_to_tex(&args[1])
      )
    }
    // Default: render as function name with parenthesized args. Single-letter
    // names render bare (matches wolframscript: f[x] -> f(x)); multi-letter
    // names use \text{} to distinguish them from implicit products.
    _ => {
      let args_tex: Vec<String> = args.iter().map(expr_to_tex).collect();
      let head = if name.chars().count() == 1 {
        name.to_string()
      } else {
        format!("\\text{{{}}}", name)
      };
      format!("{}({})", head, args_tex.join(","))
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
        args: args.to_vec().into(),
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
/// Out-of-range indices leave the placeholder literal in the output and
/// emit a StringForm::sfr warning, matching wolframscript.
pub fn format_string_form(template: &str, values: &[Expr]) -> String {
  let mut result = String::new();
  let chars: Vec<char> = template.chars().collect();
  let len = chars.len();
  let mut i = 0;
  // Last slot consumed by any placeholder (`` or `n`). The next `` picks
  // up at last_index + 1. Starts at 0 so the first `` pulls arg 1.
  let mut last_index: i64 = 0;

  while i < len {
    if chars[i] == '`' {
      // Check for `` (sequential placeholder)
      if i + 1 < len && chars[i + 1] == '`' {
        let idx = last_index + 1;
        if idx >= 1 && (idx as usize) <= values.len() {
          result.push_str(&crate::syntax::expr_to_output(
            &values[(idx - 1) as usize],
          ));
        } else {
          // Out of range — keep the `` literal and warn.
          result.push('`');
          result.push('`');
          crate::emit_message(&format!(
            "StringForm::sfr: Item {} requested in \"{}\" out of \
             range; {} items available.",
            idx,
            template,
            values.len()
          ));
        }
        last_index = idx;
        i += 2;
        continue;
      }
      // Check for `n` or `-n` (indexed placeholder; negative indices are
      // always out-of-range and emit StringForm::sfr).
      let num_start = i + 1;
      let mut end = num_start;
      if end < len && chars[end] == '-' {
        end += 1;
      }
      let digits_start = end;
      while end < len && chars[end].is_ascii_digit() {
        end += 1;
      }
      if end > digits_start && end < len && chars[end] == '`' {
        let signed: i64 = chars[num_start..end]
          .iter()
          .collect::<String>()
          .parse()
          .unwrap_or(0);
        if signed >= 1 && (signed as usize) <= values.len() {
          let idx = signed as usize;
          result.push_str(&crate::syntax::expr_to_output(&values[idx - 1]));
        } else {
          // Out of range — keep the `n` placeholder literal and warn.
          result.push('`');
          result.extend(chars[num_start..end].iter());
          result.push('`');
          crate::emit_message(&format!(
            "StringForm::sfr: Item {} requested in \"{}\" out of \
             range; {} items available.",
            signed,
            template,
            values.len()
          ));
        }
        // Numbered placeholders also update last_index so the next `` is
        // relative to the most recent numbered reference.
        last_index = signed;
        i = end + 1;
        continue;
      }
    }
    result.push(chars[i]);
    i += 1;
  }
  result
}

/// Apply a `StringTemplate[template]` to arguments, filling its slots:
///   `` `` ``        sequential positional argument
///   `` `n` ``       the n-th positional argument
///   `` `name` ``    the value for key `name` (from a single association arg)
/// Positional arguments come from the call args; named slots are filled from a
/// single association argument. An unfilled slot renders as the empty string,
/// matching wolframscript.
pub fn apply_string_template(
  template: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let tmpl = match template {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "StringTemplate".to_string(),
          args: vec![template.clone()].into(),
        }),
        args: args.to_vec(),
      });
    }
  };

  // Named bindings from a single association argument.
  let mut named: std::collections::HashMap<String, Expr> =
    std::collections::HashMap::new();
  if args.len() == 1
    && let Expr::Association(pairs) = &args[0]
  {
    for (k, v) in pairs {
      let key = match k {
        Expr::String(s) => s.clone(),
        other => crate::syntax::expr_to_string(other),
      };
      named.insert(key, v.clone());
    }
  }

  let chars: Vec<char> = tmpl.chars().collect();
  let mut result = String::new();
  let mut i = 0;
  let mut seq: usize = 0;
  while i < chars.len() {
    if chars[i] == '`'
      && let Some(rel) = chars[i + 1..].iter().position(|&c| c == '`')
    {
      let close = i + 1 + rel;
      let content: String = chars[i + 1..close].iter().collect();
      let value: Option<&Expr> = if content.is_empty() {
        seq += 1;
        args.get(seq - 1)
      } else if content.chars().all(|c| c.is_ascii_digit()) {
        let n: usize = content.parse().unwrap_or(0);
        seq = n;
        n.checked_sub(1).and_then(|j| args.get(j))
      } else {
        named.get(&content)
      };
      if let Some(v) = value {
        result.push_str(&crate::syntax::expr_to_output(v));
      }
      // Unfilled slots contribute nothing.
      i = close + 1;
      continue;
    }
    result.push(chars[i]);
    i += 1;
  }
  Ok(Expr::String(result))
}

/// ToExpression[s] / ToExpression[s, form] / ToExpression[s, form, h]
/// - Parse and evaluate `s` as code. `form` (InputForm, StandardForm, ...)
///   is accepted but the parser is the same regardless.
/// - With `h`, apply `h` to the evaluated expression before returning.
pub fn to_expression_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ToExpression expects 1 to 3 arguments".into(),
    ));
  }
  // ToExpression[InterpretationBox[boxes, expr]] returns the interpreted
  // expression directly (the second argument), evaluated. This matches
  // wolframscript: `ToExpression[InterpretationBox["Four", 4]]` → 4.
  if let crate::syntax::Expr::FunctionCall {
    name,
    args: ib_args,
  } = &args[0]
    && name == "InterpretationBox"
    && ib_args.len() == 2
  {
    let interpreted = crate::evaluator::evaluate_expr_to_expr(&ib_args[1])?;
    if args.len() == 3 {
      let wrapped = crate::syntax::Expr::FunctionCall {
        name: crate::syntax::expr_to_string(&args[2]),
        args: vec![interpreted].into(),
      };
      return crate::evaluator::evaluate_expr_to_expr(&wrapped);
    }
    return Ok(interpreted);
  }
  let s = expr_to_str(&args[0])?;
  // Empty (or whitespace-only) input is Null, not a value.
  if s.trim().is_empty() {
    return Ok(Expr::Identifier("Null".to_string()));
  }
  // Syntactically invalid input yields $Failed with a Wolfram syntax message
  // (sntxi/sntx), rather than leaking the internal parser error.
  if let Some(msg) = to_expression_syntax_error(&s) {
    crate::emit_message(&msg);
    return Ok(Expr::Identifier("$Failed".to_string()));
  }
  // Three-argument form ToExpression[str, form, h]: wrap the *parsed but
  // unevaluated* expression with head `h`, then evaluate `h[parsed]`. This
  // lets a holding head keep its argument unevaluated — e.g.
  // ToExpression["1+1", InputForm, Hold] is Hold[1 + 1], not Hold[2].
  if args.len() == 3 {
    let parsed = parse_program_to_expr(&s)?;
    let wrapped = crate::syntax::Expr::FunctionCall {
      name: crate::syntax::expr_to_string(&args[2]),
      args: vec![parsed].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&wrapped);
  }
  // Multi-statement input (e.g. "2\n3" or "2; 3") evaluates each statement
  // in order and returns the last result, matching Wolfram semantics.
  parse_and_evaluate_program(&s)
}

/// Parse `src` as a Wolfram program and return the (unevaluated) expression
/// for its single top-level statement. Multiple `;`-separated statements
/// become a `CompoundExpression`. Used by the three-argument ToExpression
/// form so a holding head receives the unevaluated parse.
fn parse_program_to_expr(src: &str) -> Result<Expr, InterpreterError> {
  use crate::Rule;
  use crate::syntax::pair_to_expr;
  let normalized = if src.contains('\r') {
    src.replace("\r\n", "\n").replace('\r', "\n")
  } else {
    src.to_string()
  };
  let preprocessed = crate::insert_statement_separators(normalized.trim());
  if let Ok(mut pairs) = crate::parse(&preprocessed)
    && let Some(program) = pairs.next()
    && program.as_rule() == Rule::Program
  {
    let mut exprs: Vec<Expr> = Vec::new();
    for node in program.into_inner() {
      if matches!(node.as_rule(), Rule::Expression | Rule::TopLevelSpan) {
        exprs.push(pair_to_expr(node));
      }
    }
    match exprs.len() {
      0 => {}
      1 => return Ok(exprs.into_iter().next().unwrap()),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "CompoundExpression".to_string(),
          args: exprs.into(),
        });
      }
    }
  }
  crate::syntax::string_to_expr(&normalized)
}

/// If `src` is not syntactically valid Wolfram input, return the message
/// `ToExpression` emits for it: `sntxi` for an incomplete expression (the
/// parse failed at end of input) or `sntx` for invalid syntax elsewhere.
/// Returns `None` when the input parses — it may still fail at evaluation
/// time, which is handled by the normal evaluation path.
fn to_expression_syntax_error(src: &str) -> Option<String> {
  use crate::Rule;
  let normalized = if src.contains('\r') {
    src.replace("\r\n", "\n").replace('\r', "\n")
  } else {
    src.to_string()
  };
  let preprocessed = crate::insert_statement_separators(normalized.trim());
  // The program grammar accepts it -> syntactically valid.
  if let Ok(mut pairs) = crate::parse(&preprocessed)
    && pairs.next().is_some_and(|p| p.as_rule() == Rule::Program)
  {
    return None;
  }
  // The single-expression fallback accepts it -> valid.
  if crate::syntax::string_to_expr(&normalized).is_ok() {
    return None;
  }
  // Both paths failed: classify the parse error by where it occurred.
  let offset = match crate::parse(&preprocessed) {
    Err(e) => {
      use pest::error::InputLocation;
      match e.location {
        InputLocation::Pos(p) => p,
        InputLocation::Span((s, _)) => s,
      }
    }
    Ok(_) => return None,
  };
  if offset >= preprocessed.trim_end().len() {
    Some(
      "ToExpression::sntxi: Incomplete expression; more input is needed."
        .to_string(),
    )
  } else {
    let snippet = preprocessed[offset..].trim();
    Some(format!(
      "ToExpression::sntx: Invalid syntax in or before \"{}\".",
      snippet
    ))
  }
}

/// Parse `src` as a Wolfram program and evaluate every top-level
/// Expression/TopLevelSpan statement in order, returning the last result.
/// Falls back to `string_to_expr` + evaluate for inputs the parser treats as
/// a single expression.
fn parse_and_evaluate_program(src: &str) -> Result<Expr, InterpreterError> {
  use crate::Rule;
  use crate::syntax::pair_to_expr;
  let normalized = if src.contains('\r') {
    src.replace("\r\n", "\n").replace('\r', "\n")
  } else {
    src.to_string()
  };
  // Insert `;` at top-level newline boundaries so the grammar treats
  // lines as distinct statements instead of implicit multiplication.
  let preprocessed = crate::insert_statement_separators(normalized.trim());
  if let Ok(mut pairs) = crate::parse(&preprocessed)
    && let Some(program) = pairs.next()
    && program.as_rule() == Rule::Program
  {
    let mut last: Option<Expr> = None;
    for node in program.into_inner() {
      if matches!(node.as_rule(), Rule::Expression | Rule::TopLevelSpan) {
        let expr = pair_to_expr(node);
        last = Some(crate::evaluator::evaluate_expr_to_expr(&expr)?);
      }
    }
    if let Some(v) = last {
      return Ok(v);
    }
  }
  // Fallback path for inputs the program grammar rejected.
  let parsed = crate::syntax::string_to_expr(&normalized)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

/// StringPadLeft[s, n] or StringPadLeft[s, n, pad] - pad string on left
/// Outcome of the one-argument `StringPad{Left,Right}[{s1, …}]` form.
enum PadOneArg {
  /// Pad every string to the longest one's length; delegate to the 2-arg form
  /// with these arguments.
  Rewrite(Vec<Expr>),
  /// An empty list pads to itself.
  Empty,
  /// The argument is not a list of strings (`::strlist`).
  Invalid,
}

/// Classify the single argument of `StringPad{Left,Right}[arg]`.
fn string_pad_one_arg(args: &[Expr]) -> PadOneArg {
  let [Expr::List(items)] = args else {
    return PadOneArg::Invalid;
  };
  if items.is_empty() {
    return PadOneArg::Empty;
  }
  if !items.iter().all(|e| matches!(e, Expr::String(_))) {
    return PadOneArg::Invalid;
  }
  let max_len = items
    .iter()
    .filter_map(|e| match e {
      Expr::String(s) => Some(s.chars().count()),
      _ => None,
    })
    .max()
    .unwrap_or(0);
  PadOneArg::Rewrite(vec![args[0].clone(), Expr::Integer(max_len as i128)])
}

/// Emit `<name>::strlist` for a one-argument call whose argument is not a list
/// of strings, and return the unevaluated call.
fn string_pad_strlist(name: &str, args: &[Expr]) -> Expr {
  let call = Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
  crate::emit_message(&format!(
    "{}::strlist: List of strings expected at position 1 in {}.",
    name,
    crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
  ));
  call
}

pub fn string_pad_left_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // StringPadLeft[{s1, …}] pads each string to the longest one's length.
  if args.len() == 1 {
    return match string_pad_one_arg(args) {
      PadOneArg::Rewrite(rewritten) => string_pad_left_ast(&rewritten),
      PadOneArg::Empty => Ok(Expr::List(vec![].into())),
      PadOneArg::Invalid => Ok(string_pad_strlist("StringPadLeft", args)),
    };
  }
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringPadLeft expects 2 or 3 arguments".into(),
    ));
  }

  // The padding (3rd argument) must be a non-empty string. Wolfram emits
  // StringPadLeft::stringnz and returns the call unevaluated for anything
  // else (empty string, a number, a list), even when the result would be a
  // pure truncation, so validate before threading or padding.
  if args.len() == 3 && !matches!(&args[2], Expr::String(p) if !p.is_empty()) {
    let call = Expr::FunctionCall {
      name: "StringPadLeft".to_string(),
      args: args.to_vec().into(),
    };
    crate::emit_message(&format!(
      "StringPadLeft::stringnz: String of non-zero length expected at \
       position 3 in {}.",
      crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
    ));
    return Ok(call);
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
    return Ok(Expr::List(results?.into()));
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
    expr_to_str(&args[2])?
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
  // StringPadRight[{s1, …}] pads each string to the longest one's length.
  if args.len() == 1 {
    return match string_pad_one_arg(args) {
      PadOneArg::Rewrite(rewritten) => string_pad_right_ast(&rewritten),
      PadOneArg::Empty => Ok(Expr::List(vec![].into())),
      PadOneArg::Invalid => Ok(string_pad_strlist("StringPadRight", args)),
    };
  }
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringPadRight expects 2 or 3 arguments".into(),
    ));
  }

  // The padding (3rd argument) must be a non-empty string. Wolfram emits
  // StringPadRight::stringnz and returns the call unevaluated for anything
  // else (empty string, a number, a list), even when the result would be a
  // pure truncation, so validate before threading or padding.
  if args.len() == 3 && !matches!(&args[2], Expr::String(p) if !p.is_empty()) {
    let call = Expr::FunctionCall {
      name: "StringPadRight".to_string(),
      args: args.to_vec().into(),
    };
    crate::emit_message(&format!(
      "StringPadRight::stringnz: String of non-zero length expected at \
       position 3 in {}.",
      crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
    ));
    return Ok(call);
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
    return Ok(Expr::List(results?.into()));
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
    expr_to_str(&args[2])?
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

/// InsertLinebreaks[s] / InsertLinebreaks[s, n] — wrap `s` so each line is at
/// most `n` characters (default 78). Greedy word wrap: words are kept whole
/// where they fit, packed onto a line separated by single spaces; a word longer
/// than `n` is hard-broken every `n` characters. The number of characters is
/// counted by Unicode code point.
pub fn insert_linebreaks_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "InsertLinebreaks".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }
  let s = match &args[0] {
    Expr::String(s) => s.clone(),
    _ => return unevaluated(),
  };
  // The width must be a positive integer; wolframscript leaves the call
  // unevaluated (with an ::intp message) otherwise.
  let n = match args.get(1) {
    None => 78usize,
    Some(Expr::Integer(n)) if *n > 0 => *n as usize,
    Some(_) => return unevaluated(),
  };

  let mut lines: Vec<String> = Vec::new();
  let mut current = String::new();
  let mut line_has_word = false;

  // Place a single (space-delimited) word, hard-breaking it when it does not
  // fit on an empty line.
  fn place_word(
    word: &str,
    n: usize,
    lines: &mut Vec<String>,
    current: &mut String,
    line_has_word: &mut bool,
  ) {
    if *line_has_word {
      let need = current.chars().count() + 1 + word.chars().count();
      if need <= n {
        current.push(' ');
        current.push_str(word);
        return;
      }
      // Does not fit: finish the current line and retry on a fresh one.
      lines.push(std::mem::take(current));
      *line_has_word = false;
    }
    // Empty line: hard-break any prefix that exceeds the width.
    let mut rest: String = word.to_string();
    while rest.chars().count() > n {
      let head: String = rest.chars().take(n).collect();
      lines.push(head);
      rest = rest.chars().skip(n).collect();
    }
    *current = rest;
    *line_has_word = true;
  }

  for word in s.split(' ') {
    place_word(word, n, &mut lines, &mut current, &mut line_has_word);
  }
  lines.push(current);
  Ok(Expr::String(lines.join("\n")))
}

/// StringCount[s, sub] - count occurrences of substring or pattern
pub fn string_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "StringCount expects at least 2 arguments".into(),
    ));
  }

  // Thread over list of strings in the first argument, preserving any options.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| {
        let mut call = vec![s.clone()];
        call.extend(args[1..].iter().cloned());
        string_count_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let s = expr_to_str(&args[0])?;
  let ignore_case = has_ignore_case_option(args);
  let overlaps = has_overlaps_option(args);

  // A conditional pattern (`patt /; test`) needs per-match evaluation of the
  // test; delegate to StringCases (which handles it) and count the matches.
  if let Expr::FunctionCall { name, .. } = &args[1]
    && name == "Condition"
  {
    let cases = string_cases_ast(args)?;
    if let Expr::List(items) = &cases {
      return Ok(Expr::Integer(items.len() as i128));
    }
  }

  // Try regex-based pattern first (handles RegularExpression, string patterns,
  // lists-of-patterns as alternatives, etc.)
  if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    let count = if overlaps {
      // Overlapping matches: count every start position where the pattern
      // matches, anchored at that position.
      let anchored_pat = if ignore_case {
        format!("^(?i:{})", regex_pat)
      } else {
        format!("^(?:{})", regex_pat)
      };
      let anchored = compile_regex(&anchored_pat).map_err(|e| {
        InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
      })?;
      count_overlapping_regex(&anchored, &s)
    } else {
      let full = if ignore_case {
        format!("(?i:{})", regex_pat)
      } else {
        regex_pat
      };
      let re = compile_regex(&full).map_err(|e| {
        InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
      })?;
      re.find_iter(&s).count()
    };
    return Ok(Expr::Integer(count as i128));
  }

  // Fallback to plain string matching.
  let sub = expr_to_str(&args[1])?;
  if sub.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let (hay, needle) = if ignore_case {
    (s.to_lowercase(), sub.to_lowercase())
  } else {
    (s.clone(), sub.clone())
  };
  let count = if overlaps {
    count_overlapping_substring(&hay, &needle)
  } else {
    hay.matches(&needle).count()
  };
  Ok(Expr::Integer(count as i128))
}

/// Whether `Overlaps -> True` appears in the option arguments.
fn has_overlaps_option(args: &[Expr]) -> bool {
  for arg in args.iter().skip(2) {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && crate::syntax::expr_to_string(pattern) == "Overlaps"
      && crate::syntax::expr_to_string(replacement) == "True"
    {
      return true;
    }
  }
  false
}

/// Count overlapping occurrences of `needle` in `hay` — one per start position
/// (in chars) where `needle` occurs.
fn count_overlapping_substring(hay: &str, needle: &str) -> usize {
  if needle.is_empty() {
    return 0;
  }
  let mut count = 0;
  for (i, _) in hay.char_indices() {
    if hay[i..].starts_with(needle) {
      count += 1;
    }
  }
  count
}

/// Count overlapping regex matches — one per char start position where the
/// (start-anchored) regex matches.
fn count_overlapping_regex(anchored: &regex::Regex, hay: &str) -> usize {
  let mut count = 0;
  for (i, _) in hay.char_indices() {
    if anchored.is_match(&hay[i..]) {
      count += 1;
    }
  }
  count
}

/// StringFreeQ[s, sub] - check if string does NOT contain substring or pattern
/// StringFreeQ[s, sub, IgnoreCase -> True] - case-insensitive
pub fn string_free_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "StringFreeQ expects 2 or 3 arguments".into(),
    ));
  }

  // Thread over a list of strings in the first argument (matches wolframscript).
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| {
        let mut call = vec![s.clone()];
        call.extend(args[1..].iter().cloned());
        string_free_q_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
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
    let re = compile_regex(&full_pat).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
    })?;
    return Ok(Expr::Identifier(
      if re.is_match(&s) { "False" } else { "True" }.to_string(),
    ));
  }

  let sub = expr_to_str(&args[1])?;
  let contains = if ignore_case {
    s.to_lowercase().contains(&sub.to_lowercase())
  } else {
    s.contains(&sub)
  };
  Ok(Expr::Identifier(
    if contains { "False" } else { "True" }.to_string(),
  ))
}

/// ToCharacterCode[s] - converts a string to a list of character codes
pub fn to_character_code_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ToCharacterCode expects 1 or 2 arguments".into(),
    ));
  }

  // Encoding selector: "UTF8" / "UTF-8" returns the byte sequence for each
  // character; any ASCII-compatible encoding like "ISO8859-1" returns the
  // codepoints directly.
  let encoding = args.get(1).map(|e| match e {
    Expr::String(s) => s.clone(),
    _ => crate::syntax::expr_to_string(e),
  });
  let is_utf8 = encoding
    .as_deref()
    .map(|e| {
      let e = e.replace('-', "").to_ascii_lowercase();
      e == "utf8"
    })
    .unwrap_or(false);

  let codes_for = |s: &str| -> Vec<Expr> {
    if is_utf8 {
      s.as_bytes()
        .iter()
        .map(|b| Expr::Integer(*b as i128))
        .collect()
    } else {
      s.chars().map(|c| Expr::Integer(c as i128)).collect()
    }
  };

  // Handle list of strings — all elements must be strings (or lists of
  // strings, handled recursively). Non-string atoms trigger a
  // ToCharacterCode::strse warning and the call returns unevaluated.
  if let Expr::List(items) = &args[0] {
    if !items.iter().all(|i| matches!(i, Expr::String(_))) {
      crate::emit_message(&format!(
        "ToCharacterCode::strse: A string or list of strings is \
         expected at position 1 in {}.",
        crate::syntax::expr_to_string(&Expr::FunctionCall {
          name: "ToCharacterCode".to_string(),
          args: args.to_vec().into(),
        })
      ));
      return Ok(Expr::FunctionCall {
        name: "ToCharacterCode".to_string(),
        args: args.to_vec().into(),
      });
    }
    let mut results = Vec::new();
    for item in items {
      let s = expr_to_str(item)?;
      results.push(Expr::List(codes_for(&s).into()));
    }
    return Ok(Expr::List(results.into()));
  }
  // Single-argument form: only accept actual strings; everything else
  // returns unevaluated with the strse warning.
  if !matches!(&args[0], Expr::String(_)) {
    crate::emit_message(&format!(
      "ToCharacterCode::strse: A string or list of strings is \
       expected at position 1 in {}.",
      crate::syntax::expr_to_string(&Expr::FunctionCall {
        name: "ToCharacterCode".to_string(),
        args: args.to_vec().into(),
      })
    ));
    return Ok(Expr::FunctionCall {
      name: "ToCharacterCode".to_string(),
      args: args.to_vec().into(),
    });
  }
  let s = expr_to_str(&args[0])?;
  Ok(Expr::List(codes_for(&s).into()))
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
        return Ok(Expr::List(results.into()));
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

  // Each endpoint is either a character code (non-negative integer) or a
  // single-character string. CharacterRange[97, 99] -> {"a", "b", "c"};
  // CharacterRange["a","c"] is the same. (Previously integers were stringified,
  // so 97 became the digit '9' and CharacterRange[97, 99] wrongly produced
  // {"9"}.) wolframscript requires BOTH endpoints to be the same kind — a
  // mixed call like CharacterRange["a", 99] is left unevaluated.
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "CharacterRange".to_string(),
      args: args.to_vec().into(),
    })
  };
  let a0_int = matches!(&args[0], Expr::Integer(n) if *n >= 0);
  let a1_int = matches!(&args[1], Expr::Integer(n) if *n >= 0);
  if a0_int != a1_int {
    return unevaluated();
  }
  let endpoint_code = |e: &Expr| -> Option<u32> {
    if let Expr::Integer(n) = e {
      return (*n >= 0).then_some(*n as u32);
    }
    expr_to_str(e).ok()?.chars().next().map(|c| c as u32)
  };
  let (Some(start), Some(end)) =
    (endpoint_code(&args[0]), endpoint_code(&args[1]))
  else {
    return unevaluated();
  };

  if start > end {
    return Ok(Expr::List(vec![].into()));
  }

  let chars: Vec<Expr> = (start..=end)
    .filter_map(char::from_u32)
    .map(|c| Expr::String(c.to_string()))
    .collect();
  Ok(Expr::List(chars.into()))
}

/// Encode the absolute value of an i128/BigInt-backed integer as a
/// Wolfram-compatible Base64 string. The alphabet (in digit-value order
/// 0..63) is `A-Z`, `a-z`, `0-9`, `+`, `/`; the sign of the input is
/// dropped, matching Wolfram's `IntegerString[n, "Base64"]`.
fn integer_to_base64_string(n: &Expr) -> Result<String, InterpreterError> {
  const ALPHABET: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let abs_big = match n {
    Expr::BigInteger(b) => {
      use num_bigint::Sign;
      match b.sign() {
        Sign::Minus => -b,
        _ => b.clone(),
      }
    }
    _ => num_bigint::BigInt::from(expr_to_int(n)?.unsigned_abs()),
  };
  if abs_big == num_bigint::BigInt::from(0) {
    return Ok("A".to_string());
  }
  let mut val = abs_big;
  let base = num_bigint::BigInt::from(64);
  let mut digits: Vec<u8> = Vec::new();
  while val > num_bigint::BigInt::from(0) {
    let rem = &val % &base;
    let d = rem
      .to_string()
      .parse::<usize>()
      .expect("remainder fits in usize");
    digits.push(ALPHABET[d]);
    val /= &base;
  }
  digits.reverse();
  Ok(String::from_utf8(digits).expect("ASCII alphabet"))
}

/// Encode a positive integer as a Roman numeral string. Wolfram's
/// `IntegerString[n, "Roman"]` uses the standard subtractive form for
/// 1..3999 and an OverscriptBox-based "vinculum" form for larger values;
/// we implement the plain subtractive form (sufficient for the
/// 1..3999 range), and fall back to the unevaluated head for inputs
/// outside that range.
fn integer_to_roman_string(n: i128) -> Option<String> {
  if !(1..=3999).contains(&n) {
    return None;
  }
  let mut n = n as u32;
  let table: &[(u32, &str)] = &[
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
  ];
  let mut out = String::new();
  for (v, s) in table {
    while n >= *v {
      out.push_str(s);
      n -= *v;
    }
  }
  Some(out)
}

/// IntegerString[n] or IntegerString[n, base] or IntegerString[n, base, length] - convert integer to string
pub fn integer_string_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "IntegerString expects 1, 2, or 3 arguments".into(),
    ));
  }

  // Wolfram accepts a named "numeric system" as the base, e.g.
  // `IntegerString[n, "Base64"]`. Handle the named forms before the
  // generic integer-base path so the string isn't rejected by
  // `expr_to_int`.
  if args.len() >= 2
    && let Expr::String(name) = &args[1]
  {
    let result = match name.as_str() {
      "Base64" => integer_to_base64_string(&args[0])?,
      "Roman" => {
        let Some(s) =
          expr_to_int(&args[0]).ok().and_then(integer_to_roman_string)
        else {
          // Unsupported range / non-integer — leave unevaluated rather
          // than guess at the vinculum/OverscriptBox encoding.
          return Ok(Expr::FunctionCall {
            name: "IntegerString".to_string(),
            args: args.to_vec().into(),
          });
        };
        s
      }
      _ => {
        crate::emit_message(&format!(
          "IntegerString::numsys: Invalid numeric system {name}."
        ));
        return Ok(Expr::FunctionCall {
          name: "IntegerString".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    return Ok(Expr::String(result));
  }

  let unevaluated = || Expr::FunctionCall {
    name: "IntegerString".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  // Position 1 must be an integer; explicit non-integer numbers (even
  // integral reals like 2.) emit ::int, symbols stay silent.
  if !matches!(&args[0], Expr::Integer(_) | Expr::BigInteger(_)) {
    let is_explicit_non_integer =
      matches!(&args[0], Expr::Real(_) | Expr::BigFloat(..))
        || matches!(&args[0], Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2);
    if is_explicit_non_integer {
      crate::emit_message(&format!(
        "IntegerString::int: Integer expected at position 1 in {}.",
        show(&unevaluated())
      ));
    }
    return Ok(unevaluated());
  }

  // Position 2: an explicit base outside 2..36 emits ::basf; symbolic
  // bases stay silently unevaluated.
  let base: u32 = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(b) if (2..=36).contains(b) => *b as u32,
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(..) => {
        crate::emit_message(&format!(
          "IntegerString::basf: Requested base {} should be an integer between 2 and 36.",
          show(&args[1])
        ));
        return Ok(unevaluated());
      }
      _ => return Ok(unevaluated()),
    }
  } else {
    10
  };

  // IntegerString uses absolute value (drops sign). Handle BigInteger
  // directly via num-bigint's to_str_radix; fall back to i128 for
  // smaller integers.
  let mut result = match &args[0] {
    Expr::BigInteger(n) => {
      use num_bigint::Sign;
      match n.sign() {
        Sign::Minus => (-n).to_str_radix(base),
        _ => n.to_str_radix(base),
      }
    }
    _ => {
      let n = expr_to_int(&args[0])?;
      let abs_n = n.unsigned_abs();
      if abs_n == 0 {
        "0".to_string()
      } else {
        let mut val = abs_n;
        let mut digits = String::new();
        while val > 0 {
          let digit = (val % base as u128) as u32;
          digits.push(char::from_digit(digit, base).unwrap());
          val /= base as u128;
        }
        digits.chars().rev().collect()
      }
    }
  };

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

/// Alphabet[] / Alphabet[language] - Returns the list of lowercase letters
/// for the named language, defaulting to English.
pub fn alphabet_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let lang: Option<&str> = match args.first() {
    None => None,
    Some(Expr::String(s)) => Some(s.as_str()),
    Some(_) => {
      return Ok(Expr::FunctionCall {
        name: "Alphabet".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let ascii: Vec<Expr> =
    ('a'..='z').map(|c| Expr::String(c.to_string())).collect();

  let chars_to_list = |s: &str| -> Vec<Expr> {
    s.chars().map(|c| Expr::String(c.to_string())).collect()
  };

  let letters: Vec<Expr> = match lang {
    None | Some("English") | Some("French") | Some("German")
    | Some("Italian") | Some("Dutch") | Some("Portuguese") | Some("Latin") => {
      ascii
    }
    Some("Spanish") => chars_to_list("abcdefghijklmnñopqrstuvwxyz"),
    Some("Swedish") | Some("Finnish") => {
      chars_to_list("abcdefghijklmnopqrstuvwxyzåäö")
    }
    Some("Norwegian") | Some("Danish") => {
      chars_to_list("abcdefghijklmnopqrstuvwxyzæøå")
    }
    Some("Polish") => [
      "a", "ą", "b", "c", "ć", "d", "e", "ę", "f", "g", "h", "i", "j", "k",
      "l", "ł", "m", "n", "ń", "o", "ó", "p", "r", "s", "ś", "t", "u", "w",
      "y", "z", "ź", "ż",
    ]
    .iter()
    .map(|s| Expr::String((*s).to_string()))
    .collect(),
    Some("Russian") => chars_to_list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    // Pan-Cyrillic: superset covering several national alphabets (Russian,
    // Ukrainian, Serbian, …) including combined graphemes like з́ and с́.
    // Matches wolframscript's list, so Alphabet["Russian"] ≠ Alphabet["Cyrillic"].
    Some("Cyrillic") => [
      "а", "б", "в", "г", "ґ", "д", "ђ", "ѓ", "е", "ё", "є", "ж", "з", "з́",
      "ѕ", "и", "і", "ї", "й", "ј", "к", "л", "љ", "м", "н", "њ", "о", "п",
      "р", "с", "с́", "т", "ћ", "ќ", "у", "ў", "ф", "х", "ц", "ч", "џ", "ш",
      "щ", "ъ", "ы", "ь", "э", "ю", "я",
    ]
    .iter()
    .map(|s| Expr::String((*s).to_string()))
    .collect(),
    Some("Greek") => chars_to_list("αβγδεζηθικλμνξοπρστυφχψω"),
    Some(_) => {
      return Ok(Expr::FunctionCall {
        name: "Alphabet".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  Ok(Expr::List(letters.into()))
}

// ─── FromLetterNumber / LetterNumber ──────────────────────────────

/// FromLetterNumber[n] - give the nth letter of the English alphabet.
/// Out-of-range or 0 returns a space character.
/// Negative numbers wrap cyclically (e.g., -1 -> z).
/// Works with lists of integers too.
pub fn from_letter_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Two-argument form: FromLetterNumber[n, alphabet] indexes into the named
  // alphabet (1-based, negative counts from the end), with out-of-range or 0
  // giving a space — the same convention as the English default.
  if args.len() == 2 {
    let unevaluated = || {
      Ok(Expr::FunctionCall {
        name: "FromLetterNumber".to_string(),
        args: args.to_vec().into(),
      })
    };
    let alphabet = alphabet_ast(std::slice::from_ref(&args[1]))?;
    let letters = match &alphabet {
      Expr::List(items) => items,
      _ => return unevaluated(),
    };
    let len = letters.len() as i128;
    let from_index = |n: i128| -> Expr {
      let pos = if (1..=len).contains(&n) {
        n
      } else if (-len..=-1).contains(&n) {
        len + n + 1
      } else {
        return Expr::String(" ".to_string());
      };
      letters[(pos - 1) as usize].clone()
    };
    return match &args[0] {
      Expr::Integer(n) => Ok(from_index(*n)),
      Expr::List(items) => {
        let results: Vec<Expr> = items
          .iter()
          .map(|item| match item {
            Expr::Integer(n) => from_index(*n),
            other => other.clone(),
          })
          .collect();
        Ok(Expr::List(results.into()))
      }
      _ => unevaluated(),
    };
  }

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
            args: vec![item.clone()].into(),
          }),
        })
        .collect();
      Ok(Expr::List(results?.into()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "FromLetterNumber".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Position of a character in the requested alphabet (1-indexed, 0 if not
/// present). Supports English (default), Greek, Cyrillic, Russian.
fn alphabet_position(ch: char, alphabet: &str) -> i128 {
  match alphabet {
    "English" | "" => {
      let lower = ch.to_ascii_lowercase();
      if lower.is_ascii_lowercase() {
        (lower as i128) - ('a' as i128) + 1
      } else {
        0
      }
    }
    "Greek" => {
      // Lowercase Greek letters α(U+03B1)..ω(U+03C9) with final sigma ς
      // mapped to the same position as σ. Uppercase is normalized to lower.
      let c = ch as u32;
      // Uppercase Α..Ω (U+0391..U+03A9); convert to lowercase equivalent.
      let lower_code = if (0x0391..=0x03A9).contains(&c) {
        c + 0x20 // Α -> α etc.
      } else {
        c
      };
      // Standard 24-letter ordering:
      // α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ(ς) τ υ φ χ ψ ω
      match lower_code {
        0x03B1 => 1,           // α
        0x03B2 => 2,           // β
        0x03B3 => 3,           // γ
        0x03B4 => 4,           // δ
        0x03B5 => 5,           // ε
        0x03B6 => 6,           // ζ
        0x03B7 => 7,           // η
        0x03B8 => 8,           // θ
        0x03B9 => 9,           // ι
        0x03BA => 10,          // κ
        0x03BB => 11,          // λ
        0x03BC => 12,          // μ
        0x03BD => 13,          // ν
        0x03BE => 14,          // ξ
        0x03BF => 15,          // ο
        0x03C0 => 16,          // π
        0x03C1 => 17,          // ρ
        0x03C2 | 0x03C3 => 18, // ς, σ
        0x03C4 => 19,          // τ
        0x03C5 => 20,          // υ
        0x03C6 => 21,          // φ
        0x03C7 => 22,          // χ
        0x03C8 => 23,          // ψ
        0x03C9 => 24,          // ω
        _ => 0,
      }
    }
    _ => 0,
  }
}

/// LetterNumber["c"] - give the position of a letter in the English alphabet.
/// LetterNumber["c", "Greek"] - position in the Greek alphabet.
/// Returns 0 for non-letter characters.
pub fn letter_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let alphabet = match args.get(1) {
    Some(Expr::String(s)) => s.clone(),
    Some(_) => {
      return Ok(Expr::FunctionCall {
        name: "LetterNumber".to_string(),
        args: args.to_vec().into(),
      });
    }
    None => "English".to_string(),
  };
  match &args[0] {
    Expr::String(s) => {
      let chars: Vec<char> = s.chars().collect();
      if chars.len() == 1 {
        Ok(Expr::Integer(alphabet_position(chars[0], &alphabet)))
      } else {
        // For multi-character strings, return a list
        let results: Vec<Expr> = chars
          .iter()
          .map(|ch| Expr::Integer(alphabet_position(*ch, &alphabet)))
          .collect();
        Ok(Expr::List(results.into()))
      }
    }
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|item| {
          let mut call = vec![item.clone()];
          if args.len() == 2 {
            call.push(args[1].clone());
          }
          letter_number_ast(&call)
        })
        .collect();
      Ok(Expr::List(results?.into()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "LetterNumber".to_string(),
      args: args.to_vec().into(),
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

// ─── PrintableASCIIQ ───────────────────────────────────────────────

/// PrintableASCIIQ[string] - True if every character is printable ASCII
/// (character codes 32 through 126 inclusive; the empty string is True).
/// Threads over a flat list of strings, returning a list of booleans.
/// A non-string argument — or a list that is not entirely strings — emits
/// the `::strse` message and stays unevaluated.
pub fn printable_ascii_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "PrintableASCIIQ".to_string(),
    args: args.to_vec().into(),
  };
  let strse = || {
    crate::emit_message(&format!(
      "PrintableASCIIQ::strse: A string or list of strings is expected at position 1 in {}.",
      crate::syntax::format_expr(
        &unevaluated(),
        crate::syntax::ExprForm::Output
      )
    ));
  };
  let bool_expr =
    |b: bool| Expr::Identifier(if b { "True" } else { "False" }.to_string());
  let is_printable =
    |s: &str| s.chars().all(|c| (32..=126).contains(&(c as u32)));

  match &args[0] {
    Expr::String(s) => Ok(bool_expr(is_printable(s))),
    Expr::List(items) if items.iter().all(|e| matches!(e, Expr::String(_))) => {
      let results: Vec<Expr> = items
        .iter()
        .map(|e| match e {
          Expr::String(s) => bool_expr(is_printable(s)),
          _ => unreachable!(),
        })
        .collect();
      Ok(Expr::List(results.into()))
    }
    _ => {
      strse();
      Ok(unevaluated())
    }
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

  // The inserted text (position 2) must be a single string. A list or any
  // other expression there emits StringInsert::string and stays unevaluated,
  // matching wolframscript (checked before the list-of-strings first-argument
  // form, so the message reports the whole original call).
  if !matches!(&args[1], Expr::String(_)) {
    let call = Expr::FunctionCall {
      name: "StringInsert".to_string(),
      args: args.to_vec().into(),
    };
    crate::emit_message(&format!(
      "StringInsert::string: String expected at position 2 in {}.",
      crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
    ));
    return Ok(call);
  }

  // Handle list of strings as first argument
  if let Expr::List(strings) = &args[0] {
    let mut results = Vec::new();
    for s_expr in strings {
      let new_args = vec![s_expr.clone(), args[1].clone(), args[2].clone()];
      results.push(string_insert_ast(&new_args)?);
    }
    return Ok(Expr::List(results.into()));
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
        args: args.to_vec().into(),
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
      args: args.to_vec().into(),
    }),
  }
}

// ─── StringDelete ──────────────────────────────────────────────────

/// StringDelete[string, sub] - deletes all occurrences of sub from string
pub fn string_delete_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "StringDelete expects at least 2 arguments".into(),
    ));
  }
  // Thread over list of strings in the first argument, preserving options.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut call = vec![item.clone()];
        call.extend(args[1..].iter().cloned());
        string_delete_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  let s = expr_to_str(&args[0])?;
  let ignore_case = has_ignore_case_option(args);

  // Deleting is replacing every match of the pattern with the empty string.
  // Reuse the shared string-pattern → regex conversion so character classes
  // (DigitCharacter, …), blanks, alternation lists, and string literals all
  // work, matching StringReplace[s, patt -> ""].
  if let Some(regex_pat) = string_pattern_to_regex(&args[1]) {
    let full = if ignore_case {
      format!("(?i:{})", regex_pat)
    } else {
      regex_pat
    };
    let re = compile_regex(&full).map_err(|e| {
      InterpreterError::EvaluationError(format!("Invalid pattern: {}", e))
    })?;
    return Ok(Expr::String(re.replace_all(&s, "").into_owned()));
  }

  // Fallback: plain substring deletion (single pattern or list of literals).
  if let Expr::List(items) = &args[1] {
    let mut result = s;
    for item in items {
      let sub = expr_to_str(item)?;
      result = result.replace(&sub, "");
    }
    return Ok(Expr::String(result));
  }
  let sub = expr_to_str(&args[1])?;
  Ok(Expr::String(s.replace(&sub, "")))
}

// ─── Capitalize ────────────────────────────────────────────────────

/// Capitalize[string] - capitalizes the first letter of the string
pub fn capitalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Capitalize expects 1 or 2 arguments".into(),
    ));
  }
  // Thread over a list of strings, preserving the optional type argument.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut call = vec![item.clone()];
        call.extend(args[1..].iter().cloned());
        capitalize_ast(&call)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  let s = expr_to_str(&args[0])?;

  // Capitalize the first character of a whitespace-delimited word.
  let cap_word = |w: &str| -> String {
    let mut chars = w.chars();
    match chars.next() {
      Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
      None => String::new(),
    }
  };

  // The two-argument form selects which words to capitalize.
  if args.len() == 2 {
    let kind = match &args[1] {
      Expr::String(k) => k.clone(),
      _ => crate::syntax::expr_to_string(&args[1]),
    };
    if !matches!(kind.as_str(), "AllWords" | "FirstWord" | "LongWords") {
      // "TitleCase" depends on part-of-speech data we do not model; any
      // other value is invalid (Capitalize::nform). Leave unevaluated.
      if kind != "TitleCase" {
        crate::emit_message(&format!(
          "Capitalize::nform: Argument {} is not \"AllWords\", \"FirstWord\", \"LongWords\" or \"TitleCase\".",
          crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output)
        ));
      }
      return Ok(Expr::FunctionCall {
        name: "Capitalize".to_string(),
        args: args.to_vec().into(),
      });
    }
    // Rebuild the string, capitalizing the selected words while preserving
    // the original whitespace runs.
    let mut result = String::new();
    let mut word_idx = 0usize;
    for (is_ws, seg) in split_keep_whitespace(&s) {
      if is_ws {
        result.push_str(&seg);
      } else {
        let cap = match kind.as_str() {
          "AllWords" => true,
          "FirstWord" => word_idx == 0,
          "LongWords" => seg.chars().count() > 3,
          _ => unreachable!(),
        };
        result.push_str(&if cap { cap_word(&seg) } else { seg.clone() });
        word_idx += 1;
      }
    }
    return Ok(Expr::String(result));
  }

  // One-argument form: capitalize only the first character of the string.
  Ok(Expr::String(cap_word(&s)))
}

/// Split a string into alternating (is_whitespace, segment) runs, preserving
/// the original characters so they can be rejoined unchanged.
fn split_keep_whitespace(s: &str) -> Vec<(bool, String)> {
  let mut out: Vec<(bool, String)> = Vec::new();
  for c in s.chars() {
    let is_ws = c.is_whitespace();
    match out.last_mut() {
      Some((last_ws, run)) if *last_ws == is_ws => run.push(c),
      _ => out.push((is_ws, c.to_string())),
    }
  }
  out
}

// ─── Decapitalize ──────────────────────────────────────────────────

/// Decapitalize[string] - lowercases the first letter of the string
pub fn decapitalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Decapitalize expects exactly 1 argument".into(),
    ));
  }
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| decapitalize_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
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

/// EditDistance[s1, s2] - Levenshtein distance between two strings.
/// Also accepts lists of items, and an optional IgnoreCase -> True rule.
pub fn edit_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "EditDistance expects 2 or 3 arguments".into(),
    ));
  }
  let ignore_case = args.len() == 3 && extract_ignore_case(&args[2..]);

  fn to_tokens(
    expr: &Expr,
    lower: bool,
  ) -> Result<Vec<String>, InterpreterError> {
    match expr {
      Expr::String(s) => {
        let s = if lower { s.to_lowercase() } else { s.clone() };
        Ok(s.chars().map(|c| c.to_string()).collect())
      }
      Expr::List(items) => {
        Ok(items.iter().map(crate::syntax::expr_to_output).collect())
      }
      _ => {
        let s = expr_to_str(expr)?;
        let s = if lower { s.to_lowercase() } else { s };
        Ok(s.chars().map(|c| c.to_string()).collect())
      }
    }
  }

  let a = to_tokens(&args[0], ignore_case)?;
  let b = to_tokens(&args[1], ignore_case)?;
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

/// Tokenize an alignment argument: a string into its characters, a list into
/// its (output-formatted) elements.
fn alignment_tokens(expr: &Expr) -> Result<Vec<String>, InterpreterError> {
  match expr {
    Expr::String(s) => Ok(s.chars().map(|c| c.to_string()).collect()),
    Expr::List(items) => {
      Ok(items.iter().map(crate::syntax::expr_to_output).collect())
    }
    _ => {
      let s = expr_to_str(expr)?;
      Ok(s.chars().map(|c| c.to_string()).collect())
    }
  }
}

/// NeedlemanWunschSimilarity[s1, s2] - global sequence-alignment score with the
/// default scoring (match +1, mismatch -1, gap penalty 1). When one input is
/// empty, wolframscript returns the length of the other.
pub fn needleman_wunsch_similarity_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "NeedlemanWunschSimilarity".to_string(),
      args: args.to_vec().into(),
    });
  }
  let a = alignment_tokens(&args[0])?;
  let b = alignment_tokens(&args[1])?;
  let (n, m) = (a.len(), b.len());
  if n == 0 || m == 0 {
    return Ok(Expr::Real(n.max(m) as f64));
  }
  let mut dp = vec![vec![0i64; m + 1]; n + 1];
  for (i, row) in dp.iter_mut().enumerate() {
    row[0] = -(i as i64);
  }
  for j in 0..=m {
    dp[0][j] = -(j as i64);
  }
  for i in 1..=n {
    for j in 1..=m {
      let s = if a[i - 1] == b[j - 1] { 1 } else { -1 };
      dp[i][j] = (dp[i - 1][j - 1] + s)
        .max(dp[i - 1][j] - 1)
        .max(dp[i][j - 1] - 1);
    }
  }
  Ok(Expr::Real(dp[n][m] as f64))
}

/// SmithWatermanSimilarity[s1, s2] - local sequence-alignment score with the
/// default scoring (match +1, mismatch -1, gap penalty 1). Cell scores are
/// floored at 0 and the result is the maximum cell value.
pub fn smith_waterman_similarity_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "SmithWatermanSimilarity".to_string(),
      args: args.to_vec().into(),
    });
  }
  let a = alignment_tokens(&args[0])?;
  let b = alignment_tokens(&args[1])?;
  let (n, m) = (a.len(), b.len());
  let mut dp = vec![vec![0i64; m + 1]; n + 1];
  let mut best = 0i64;
  for i in 1..=n {
    for j in 1..=m {
      let s = if a[i - 1] == b[j - 1] { 1 } else { -1 };
      let v = (dp[i - 1][j - 1] + s)
        .max(dp[i - 1][j] - 1)
        .max(dp[i][j - 1] - 1)
        .max(0);
      dp[i][j] = v;
      best = best.max(v);
    }
  }
  Ok(Expr::Real(best as f64))
}

/// DamerauLevenshteinDistance[s1, s2] - like Levenshtein distance but also
/// allows a single transposition of two adjacent characters as a unit cost.
/// Also accepts lists of items (compared by equality).
pub fn damerau_levenshtein_distance_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "DamerauLevenshteinDistance expects 2 or 3 arguments".into(),
    ));
  }
  let ignore_case = args.len() == 3 && extract_ignore_case(&args[2..]);

  fn to_tokens(
    expr: &Expr,
    lower: bool,
  ) -> Result<Vec<String>, InterpreterError> {
    match expr {
      Expr::String(s) => {
        let s = if lower { s.to_lowercase() } else { s.clone() };
        Ok(s.chars().map(|c| c.to_string()).collect())
      }
      Expr::List(items) => {
        Ok(items.iter().map(crate::syntax::expr_to_output).collect())
      }
      _ => {
        let s = expr_to_str(expr)?;
        let s = if lower { s.to_lowercase() } else { s };
        Ok(s.chars().map(|c| c.to_string()).collect())
      }
    }
  }

  let a = to_tokens(&args[0], ignore_case)?;
  let b = to_tokens(&args[1], ignore_case)?;
  let n = a.len();
  let m = b.len();

  // True (unrestricted) Damerau-Levenshtein distance, matching Wolfram. Unlike
  // the Optimal String Alignment variant — which forbids editing any substring
  // more than once and so over-counts cases like "ca" -> "abc" (3 vs 2) — this
  // uses a last-occurrence table so a transposition can interleave with other
  // edits. The matrix is offset by +1 so the algorithm's logical index -1 maps
  // to row/column 0, which holds the "infinity" border (max_dist).
  let max_dist = n + m;
  let mut d = vec![vec![0usize; m + 2]; n + 2];
  d[0][0] = max_dist;
  for i in 0..=n {
    d[i + 1][0] = max_dist; // logical d[i][-1]
    d[i + 1][1] = i; // logical d[i][0]
  }
  for j in 0..=m {
    d[0][j + 1] = max_dist; // logical d[-1][j]
    d[1][j + 1] = j; // logical d[0][j]
  }
  // Last row (1-based) at which each token last appeared in `a`; 0 = never.
  let mut last_seen: std::collections::HashMap<String, usize> =
    std::collections::HashMap::new();
  for i in 1..=n {
    // Column (1-based) of the last token in `b` matched so far in this row.
    let mut db = 0usize;
    for j in 1..=m {
      let k = *last_seen.get(&b[j - 1]).unwrap_or(&0);
      let l = db;
      let cost = if a[i - 1] == b[j - 1] {
        db = j;
        0
      } else {
        1
      };
      // Index mapping: logical d[x][y] is stored at d[x+1][y+1].
      let transposition = d[k][l] + (i - k - 1) + 1 + (j - l - 1);
      d[i + 1][j + 1] = (d[i][j] + cost) // substitution
        .min(d[i + 1][j] + 1) // insertion
        .min(d[i][j + 1] + 1) // deletion
        .min(transposition);
    }
    last_seen.insert(a[i - 1].clone(), i);
  }
  Ok(Expr::Integer(d[n + 1][m + 1] as i128))
}

/// The (start index in `a`, start index in `b`, length) of the longest
/// contiguous run of tokens common to `a` and `b`. Ties resolve to the
/// earliest run. Returns (0, 0, 0) when there is no common token.
fn longest_common_run(a: &[String], b: &[String]) -> (usize, usize, usize) {
  let n = a.len();
  let m = b.len();
  let mut dp = vec![vec![0usize; m + 1]; n + 1];
  let mut max_len = 0usize;
  let mut end_i = 0usize; // end position in `a`
  let mut end_j = 0usize; // end position in `b`
  for i in 1..=n {
    for j in 1..=m {
      if a[i - 1] == b[j - 1] {
        dp[i][j] = dp[i - 1][j - 1] + 1;
        if dp[i][j] > max_len {
          max_len = dp[i][j];
          end_i = i;
          end_j = j;
        }
      }
    }
  }
  (end_i - max_len, end_j - max_len, max_len)
}

/// Token sequences for LongestCommonSubsequence-family functions: a string's
/// characters or a list's elements (compared by their output form).
fn lcs_tokens(expr: &Expr) -> Option<Vec<String>> {
  match expr {
    Expr::List(items) => {
      Some(items.iter().map(crate::syntax::expr_to_output).collect())
    }
    Expr::String(s) => Some(s.chars().map(|c| c.to_string()).collect()),
    _ => None,
  }
}

/// LongestCommonSubsequence[s1, s2] — Wolfram's LongestCommonSubsequence
/// returns the longest *contiguous* match. Strings yield the matching
/// substring; lists yield the matching sublist of elements.
pub fn longest_common_subsequence_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LongestCommonSubsequence expects exactly 2 arguments".into(),
    ));
  }

  // List inputs compare whole elements and return the matching sublist.
  if let (Expr::List(l1), Expr::List(l2)) = (&args[0], &args[1]) {
    let t1: Vec<String> =
      l1.iter().map(crate::syntax::expr_to_output).collect();
    let t2: Vec<String> =
      l2.iter().map(crate::syntax::expr_to_output).collect();
    let (start, _, len) = longest_common_run(&t1, &t2);
    let sub: Vec<Expr> = l1[start..start + len].to_vec();
    return Ok(Expr::List(sub.into()));
  }

  let s1 = expr_to_str(&args[0])?;
  let s2 = expr_to_str(&args[1])?;
  let chars1: Vec<char> = s1.chars().collect();
  let chars2: Vec<char> = s2.chars().collect();
  let t1: Vec<String> = chars1.iter().map(|c| c.to_string()).collect();
  let t2: Vec<String> = chars2.iter().map(|c| c.to_string()).collect();
  let (start, _, len) = longest_common_run(&t1, &t2);
  let result: String = chars1[start..start + len].iter().collect();
  Ok(Expr::String(result))
}

/// LongestCommonSubsequencePositions[s1, s2] — the 1-indexed inclusive
/// {start, end} spans of the longest common contiguous run within each
/// argument: `{{start1, end1}, {start2, end2}}`. Returns `{}` when there is
/// no common element.
pub fn longest_common_subsequence_positions_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LongestCommonSubsequencePositions expects exactly 2 arguments".into(),
    ));
  }
  let (t1, t2) = match (lcs_tokens(&args[0]), lcs_tokens(&args[1])) {
    (Some(t1), Some(t2)) => (t1, t2),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LongestCommonSubsequencePositions".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let (start1, start2, len) = longest_common_run(&t1, &t2);
  if len == 0 {
    return Ok(Expr::List(vec![].into()));
  }
  let span = |start: usize| {
    Expr::List(
      vec![
        Expr::Integer((start + 1) as i128),
        Expr::Integer((start + len) as i128),
      ]
      .into(),
    )
  };
  Ok(Expr::List(vec![span(start1), span(start2)].into()))
}

/// LongestCommonSequence[s1, s2] — the longest *noncontiguous* common
/// subsequence (classic LCS dynamic programming), distinct from Wolfram's
/// contiguous LongestCommonSubsequence. Strings yield the matching characters
/// as a string; lists yield the matching sublist of elements. Returns an empty
/// string/list when there is no common element.
pub fn longest_common_sequence_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LongestCommonSequence expects exactly 2 arguments".into(),
    ));
  }

  // List inputs compare whole elements and return the matching sublist.
  if let (Expr::List(l1), Expr::List(l2)) = (&args[0], &args[1]) {
    let t1: Vec<String> =
      l1.iter().map(crate::syntax::expr_to_output).collect();
    let t2: Vec<String> =
      l2.iter().map(crate::syntax::expr_to_output).collect();
    let sub: Vec<Expr> = lcs_indices(&t1, &t2)
      .iter()
      .map(|&i| l1[i].clone())
      .collect();
    return Ok(Expr::List(sub.into()));
  }

  let s1 = expr_to_str(&args[0])?;
  let s2 = expr_to_str(&args[1])?;
  let chars1: Vec<char> = s1.chars().collect();
  let chars2: Vec<char> = s2.chars().collect();
  let t1: Vec<String> = chars1.iter().map(|c| c.to_string()).collect();
  let t2: Vec<String> = chars2.iter().map(|c| c.to_string()).collect();
  let result: String =
    lcs_indices(&t1, &t2).iter().map(|&i| chars1[i]).collect();
  Ok(Expr::String(result))
}

/// Indices into the first sequence forming the longest common subsequence with
/// the second. Backtracking prefers moving up (decreasing the first index) on
/// ties, matching Wolfram's choice among equal-length results.
fn lcs_indices(t1: &[String], t2: &[String]) -> Vec<usize> {
  let m = t1.len();
  let n = t2.len();
  let mut l = vec![vec![0usize; n + 1]; m + 1];
  for i in 1..=m {
    for j in 1..=n {
      l[i][j] = if t1[i - 1] == t2[j - 1] {
        l[i - 1][j - 1] + 1
      } else {
        l[i - 1][j].max(l[i][j - 1])
      };
    }
  }
  let mut idx = Vec::new();
  let (mut i, mut j) = (m, n);
  while i > 0 && j > 0 {
    if t1[i - 1] == t2[j - 1] {
      idx.push(i - 1);
      i -= 1;
      j -= 1;
    } else if l[i - 1][j] >= l[i][j - 1] {
      i -= 1;
    } else {
      j -= 1;
    }
  }
  idx.reverse();
  idx
}

/// SequenceAlignment[s1, s2] — aligns two strings using Needleman-Wunsch
/// Returns a list of matching segments and {diff1, diff2} pairs.
pub fn sequence_alignment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SequenceAlignment expects exactly 2 arguments".into(),
    ));
  }

  // Handle both strings and lists. For lists we keep the original elements
  // (`orig1`/`orig2`) so segments can be emitted as element sublists rather
  // than concatenated strings; `chars1`/`chars2` are only used for comparison.
  let (is_string, chars1, chars2, orig1, orig2) = match (&args[0], &args[1]) {
    (Expr::String(s1), Expr::String(s2)) => {
      let c1: Vec<String> = s1.chars().map(|c| c.to_string()).collect();
      let c2: Vec<String> = s2.chars().map(|c| c.to_string()).collect();
      (true, c1, c2, Vec::<Expr>::new(), Vec::<Expr>::new())
    }
    (Expr::List(l1), Expr::List(l2)) => {
      let c1: Vec<String> =
        l1.iter().map(crate::syntax::expr_to_output).collect();
      let c2: Vec<String> =
        l2.iter().map(crate::syntax::expr_to_output).collect();
      let o1: Vec<Expr> = l1.iter().cloned().collect();
      let o2: Vec<Expr> = l2.iter().cloned().collect();
      (false, c1, c2, o1, o2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SequenceAlignment".to_string(),
        args: args.to_vec().into(),
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

  // Emit a segment from indices into the first (`from_first`) or second
  // sequence: a concatenated string for string input, a sublist of the
  // original elements for list input.
  let make_seg = |idxs: &[usize], from_first: bool| -> Expr {
    if is_string {
      let src = if from_first { &chars1 } else { &chars2 };
      Expr::String(idxs.iter().map(|&x| src[x].clone()).collect::<String>())
    } else {
      let src = if from_first { &orig1 } else { &orig2 };
      Expr::List(
        idxs
          .iter()
          .map(|&x| src[x].clone())
          .collect::<Vec<_>>()
          .into(),
      )
    }
  };

  // Build result segments
  let mut result: Vec<Expr> = Vec::new();
  let mut k = 0;
  let len = aligned1.len();

  while k < len {
    if aligned1[k].is_some() && aligned2[k].is_some() {
      let i1 = aligned1[k].unwrap();
      let j1 = aligned2[k].unwrap();
      if chars1[i1] == chars2[j1] {
        // Matching segment (both sides identical, so emit one piece).
        let mut idxs = vec![i1];
        k += 1;
        while k < len
          && aligned1[k].is_some()
          && aligned2[k].is_some()
          && chars1[aligned1[k].unwrap()] == chars2[aligned2[k].unwrap()]
        {
          idxs.push(aligned1[k].unwrap());
          k += 1;
        }
        result.push(make_seg(&idxs, true));
      } else {
        // Mismatch: a {removed, inserted} pair.
        let mut idx1 = vec![i1];
        let mut idx2 = vec![j1];
        k += 1;
        while k < len
          && aligned1[k].is_some()
          && aligned2[k].is_some()
          && chars1[aligned1[k].unwrap()] != chars2[aligned2[k].unwrap()]
        {
          idx1.push(aligned1[k].unwrap());
          idx2.push(aligned2[k].unwrap());
          k += 1;
        }
        result.push(Expr::List(
          vec![make_seg(&idx1, true), make_seg(&idx2, false)].into(),
        ));
      }
    } else {
      // Gap on one side: a {removed, inserted} pair where one side is empty.
      let mut idx1 = Vec::new();
      let mut idx2 = Vec::new();
      while k < len && (aligned1[k].is_none() || aligned2[k].is_none()) {
        if let Some(i1) = aligned1[k] {
          idx1.push(i1);
        }
        if let Some(j1) = aligned2[k] {
          idx2.push(j1);
        }
        k += 1;
      }
      result.push(Expr::List(
        vec![make_seg(&idx1, true), make_seg(&idx2, false)].into(),
      ));
    }
  }

  Ok(Expr::List(result.into()))
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
      Ok(Expr::List(result.into()))
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
  Ok(Expr::List(vec![taken, dropped].into()))
}

/// HammingDistance[s1, s2] - number of positions where characters differ.
/// Also accepts an IgnoreCase -> True option.
pub fn hamming_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "HammingDistance expects 2 or 3 arguments".into(),
    ));
  }
  let ignore_case = args.len() == 3 && extract_ignore_case(&args[2..]);
  let s1 = expr_to_str(&args[0])?;
  let s2 = expr_to_str(&args[1])?;
  let s1 = if ignore_case { s1.to_lowercase() } else { s1 };
  let s2 = if ignore_case { s2.to_lowercase() } else { s2 };
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

/// Tokenize a string into words the way TextWords/WordCounts do: split on
/// whitespace, then trim leading and trailing non-alphanumeric characters from
/// each token (internal hyphens/apostrophes are kept, e.g. "YT-1300",
/// "don't"). Empty results (pure punctuation) are dropped.
pub fn text_word_tokens(s: &str) -> Vec<String> {
  s.split_whitespace()
    .map(|tok| tok.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
    .filter(|w| !w.is_empty())
    .collect()
}

/// WordCounts[s] - counts of words; WordCounts[s, n] - counts of word n-grams
/// (keyed by lists of n words). Sorted by count descending, then by last
/// occurrence descending, matching wolframscript.
/// WordFrequency[text, word] - the fraction of words in `text` equal to
/// `word` (a Real). `word` may be a list, giving an association of fractions.
/// Case-sensitive by default; `IgnoreCase -> True` folds case.
pub fn word_frequency_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "WordFrequency".to_string(),
      args: args.to_vec().into(),
    })
  };
  let s = expr_to_str(&args[0])?;
  let ignore_case = extract_ignore_case(&args[2..]);
  let words = text_word_tokens(&s);
  let total = words.len();

  let frequency = |target: &str| -> f64 {
    if total == 0 {
      return 0.0;
    }
    let count = words
      .iter()
      .filter(|w| {
        if ignore_case {
          w.to_lowercase() == target.to_lowercase()
        } else {
          w.as_str() == target
        }
      })
      .count();
    count as f64 / total as f64
  };

  match &args[1] {
    Expr::String(word) => Ok(Expr::Real(frequency(word))),
    Expr::List(items) => {
      let mut pairs = Vec::with_capacity(items.len());
      for item in items.iter() {
        match item {
          Expr::String(w) => {
            pairs.push((Expr::String(w.clone()), Expr::Real(frequency(w))));
          }
          _ => return unevaluated(),
        }
      }
      Ok(Expr::Association(pairs))
    }
    _ => unevaluated(),
  }
}

pub fn word_counts_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let s = expr_to_str(&args[0])?;
  let n = if args.len() == 2 {
    let n = expr_to_int(&args[1])?;
    if n < 1 {
      return Ok(Expr::Association(Vec::new()));
    }
    Some(n as usize)
  } else {
    None
  };

  let words = text_word_tokens(&s);

  // Build the units to count: single words, or n-grams as word lists.
  // Each unit carries a dedup key (the joined words) and its display Expr.
  let units: Vec<(String, Expr)> = match n {
    None => words
      .iter()
      .map(|w| (w.clone(), Expr::String(w.clone())))
      .collect(),
    Some(n) => {
      if words.len() < n {
        Vec::new()
      } else {
        (0..=words.len() - n)
          .map(|i| {
            let gram = &words[i..i + n];
            let key = gram.join("\u{0}");
            let list = Expr::List(
              gram.iter().map(|w| Expr::String(w.clone())).collect(),
            );
            (key, list)
          })
          .collect()
      }
    }
  };

  // Count, tracking count and last position, preserving first display Expr.
  let mut counts: Vec<(Expr, i128, usize)> = Vec::new();
  let mut seen: std::collections::HashMap<String, usize> =
    std::collections::HashMap::new();
  for (pos, (key, disp)) in units.into_iter().enumerate() {
    if let Some(&idx) = seen.get(&key) {
      counts[idx].1 += 1;
      counts[idx].2 = pos;
    } else {
      seen.insert(key, counts.len());
      counts.push((disp, 1, pos));
    }
  }
  // Sort by count descending, then by last position descending.
  counts.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
  Ok(Expr::Association(
    counts
      .into_iter()
      .map(|(disp, count, _)| (disp, Expr::Integer(count)))
      .collect(),
  ))
}

/// Build an association of n-gram counts in first-occurrence order (the order
/// Counts uses), as the two-argument CharacterCounts/LetterCounts forms do.
pub fn ngram_counts(grams: Vec<String>) -> Expr {
  let mut counts: Vec<(String, i128)> = Vec::new();
  let mut seen: std::collections::HashMap<String, usize> =
    std::collections::HashMap::new();
  for g in grams {
    if let Some(&idx) = seen.get(&g) {
      counts[idx].1 += 1;
    } else {
      seen.insert(g.clone(), counts.len());
      counts.push((g, 1));
    }
  }
  Expr::Association(
    counts
      .into_iter()
      .map(|(g, n)| (Expr::String(g), Expr::Integer(n)))
      .collect(),
  )
}

/// Sliding window n-grams (length `n`) over a slice of characters.
fn char_ngrams(chars: &[char], n: usize) -> Vec<String> {
  if n == 0 || chars.len() < n {
    return Vec::new();
  }
  (0..=chars.len() - n)
    .map(|i| chars[i..i + n].iter().collect())
    .collect()
}

/// CharacterCounts[s, n] - counts of length-n character n-grams (sliding window
/// over all characters), in first-occurrence order.
pub fn character_counts_ngram_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Thread over a list of strings: each string gets its own n-gram counts.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| character_counts_ngram_ast(&[s.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n == 1 {
    return character_counts_ast(&args[0..1]);
  }
  if n < 1 {
    return Ok(Expr::Association(Vec::new()));
  }
  let chars: Vec<char> = s.chars().collect();
  Ok(ngram_counts(char_ngrams(&chars, n as usize)))
}

/// LetterCounts[s] - association of letter (alphabetic) frequencies, sorted by
/// count descending then by last position descending.
pub fn letter_counts_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Thread over a list of strings: LetterCounts[{s1, …}] = {LetterCounts[s1], …}.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| letter_counts_ast(std::slice::from_ref(s)))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  let s = expr_to_str(&args[0])?;
  // Count letters preserving first-occurrence order.
  let mut counts: Vec<(char, i128)> = Vec::new();
  let mut seen: std::collections::HashMap<char, usize> =
    std::collections::HashMap::new();
  for ch in s.chars() {
    if ch.is_alphabetic() {
      if let Some(&idx) = seen.get(&ch) {
        counts[idx].1 += 1;
      } else {
        seen.insert(ch, counts.len());
        counts.push((ch, 1));
      }
    }
  }
  // Sort by frequency descending, breaking ties by reverse first-occurrence
  // (reverse, then stable-sort), matching CharacterCounts and wolframscript.
  counts.reverse();
  counts.sort_by(|a, b| b.1.cmp(&a.1));
  Ok(Expr::Association(
    counts
      .into_iter()
      .map(|(ch, count)| (Expr::String(ch.to_string()), Expr::Integer(count)))
      .collect(),
  ))
}

/// LetterCounts[s, n] - counts of length-n letter n-grams. Non-letter
/// characters break the sliding window (n-grams never span them), in
/// first-occurrence order.
pub fn letter_counts_ngram_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Thread over a list of strings: each string gets its own n-gram counts.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| letter_counts_ngram_ast(&[s.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  let s = expr_to_str(&args[0])?;
  let n = expr_to_int(&args[1])?;
  if n == 1 {
    return letter_counts_ast(&args[0..1]);
  }
  if n < 1 {
    return Ok(Expr::Association(Vec::new()));
  }
  let n = n as usize;
  let mut grams: Vec<String> = Vec::new();
  // Split into maximal runs of alphabetic characters and take n-grams in each.
  let mut run: Vec<char> = Vec::new();
  let flush = |run: &mut Vec<char>, grams: &mut Vec<String>| {
    if !run.is_empty() {
      grams.extend(char_ngrams(run, n));
      run.clear();
    }
  };
  for c in s.chars() {
    if c.is_alphabetic() {
      run.push(c);
    } else {
      flush(&mut run, &mut grams);
    }
  }
  flush(&mut run, &mut grams);
  Ok(ngram_counts(grams))
}

/// CharacterCounts[s] - association of character frequencies, sorted by frequency descending
pub fn character_counts_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CharacterCounts expects exactly 1 argument".into(),
    ));
  }
  // Thread over a list of strings: one association per string.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|s| character_counts_ast(std::slice::from_ref(s)))
      .collect();
    return Ok(Expr::List(results?.into()));
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
/// Standard IEEE 802.3 CRC-32 (the zlib/PNG variant, polynomial 0xEDB88320).
fn crc32(data: &[u8]) -> u32 {
  let mut crc: u32 = 0xFFFF_FFFF;
  for &byte in data {
    crc ^= byte as u32;
    for _ in 0..8 {
      crc = if crc & 1 != 0 {
        (crc >> 1) ^ 0xEDB8_8320
      } else {
        crc >> 1
      };
    }
  }
  !crc
}

/// Adler-32 checksum (RFC 1950).
fn adler32(data: &[u8]) -> u32 {
  const MOD: u32 = 65521;
  let mut a: u32 = 1;
  let mut b: u32 = 0;
  for &byte in data {
    a = (a + byte as u32) % MOD;
    b = (b + a) % MOD;
  }
  (b << 16) | a
}

/// MD2 message digest (RFC 1319). Returns the 16-byte digest.
fn md2(data: &[u8]) -> [u8; 16] {
  // S-table: a fixed permutation of 0..255 derived from the digits of pi.
  const S: [u8; 256] = [
    41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6, 19, 98,
    167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188, 76, 130, 202, 30,
    155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24, 138, 23, 229, 18, 190,
    78, 196, 214, 218, 158, 222, 73, 160, 251, 245, 142, 187, 47, 238, 122,
    169, 104, 121, 145, 21, 178, 7, 63, 148, 194, 16, 137, 11, 34, 95, 33, 128,
    127, 93, 154, 90, 144, 50, 39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25,
    48, 179, 72, 165, 181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184,
    56, 210, 150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
    112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27, 96, 37,
    173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15, 85, 71, 163, 35,
    221, 81, 175, 58, 195, 92, 249, 206, 186, 197, 234, 38, 44, 83, 13, 110,
    133, 40, 132, 9, 211, 223, 205, 244, 65, 129, 77, 82, 106, 220, 55, 200,
    108, 193, 171, 250, 36, 225, 123, 8, 12, 189, 177, 74, 120, 136, 149, 139,
    227, 99, 232, 109, 233, 203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14,
    102, 88, 208, 228, 166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180,
    143, 237, 31, 26, 219, 153, 141, 51, 159, 17, 131, 20,
  ];

  // Pad so the length is a multiple of 16; the pad byte equals the pad length.
  let mut msg = data.to_vec();
  let pad = 16 - (msg.len() % 16);
  msg.extend(std::iter::repeat_n(pad as u8, pad));

  // Append the 16-byte checksum.
  let mut checksum = [0u8; 16];
  let mut l = 0u8;
  for block in msg.chunks(16) {
    for j in 0..16 {
      let c = block[j];
      checksum[j] ^= S[(c ^ l) as usize];
      l = checksum[j];
    }
  }
  msg.extend_from_slice(&checksum);

  // Compute the digest from the 48-byte state X.
  let mut x = [0u8; 48];
  for block in msg.chunks(16) {
    for j in 0..16 {
      x[16 + j] = block[j];
      x[32 + j] = x[16 + j] ^ x[j];
    }
    let mut t = 0u8;
    for j in 0..18 {
      for k in 0..48 {
        x[k] ^= S[t as usize];
        t = x[k];
      }
      t = t.wrapping_add(j as u8);
    }
  }

  let mut out = [0u8; 16];
  out.copy_from_slice(&x[0..16]);
  out
}

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
    "CRC32" => format!("{:08x}", crc32(s.as_bytes())),
    "Adler32" => format!("{:08x}", adler32(s.as_bytes())),
    "MD2" => md2(s.as_bytes())
      .iter()
      .map(|b| format!("{:02x}", b))
      .collect::<String>(),
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
        args: args.to_vec().into(),
      });
    }
  };

  // Raw digest bytes, recovered from the hex string.
  let bytes: Vec<u8> = (0..hex_string.len())
    .step_by(2)
    .filter_map(|i| u8::from_str_radix(&hex_string[i..i + 2], 16).ok())
    .collect();

  match format.as_str() {
    "HexString" => Ok(Expr::String(hex_string)),
    "Base64Encoding" => {
      use base64::Engine;
      let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
      Ok(Expr::String(b64))
    }
    // A Woxi ByteArray stores its bytes as an internal base64 string.
    "ByteArray" => {
      use base64::Engine;
      let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
      Ok(Expr::FunctionCall {
        name: "ByteArray".to_string(),
        args: vec![Expr::String(b64)].into(),
      })
    }
    // The integer, zero-padded to the fixed width of the digest size — the
    // number of decimal digits of 256^nbytes (matching wolframscript).
    "DecimalString" => {
      let n = num_bigint::BigInt::parse_bytes(hex_string.as_bytes(), 16)
        .unwrap_or_default();
      let width = num_bigint::BigInt::from(256u32)
        .pow(bytes.len() as u32)
        .to_string()
        .len();
      Ok(Expr::String(format!("{:0>width$}", n, width = width)))
    }
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
        args: vec![evaluated].into(),
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

  // Optional max count: must be a non-negative machine integer when
  // given. Match wolframscript on rejection: emit `ReadList::intnm`
  // and leave the call unevaluated.
  let max_count: Option<usize> = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(n) if *n >= 0 => Some(*n as usize),
      _ => {
        let formatted_args = args
          .iter()
          .map(crate::syntax::expr_to_string)
          .collect::<Vec<_>>()
          .join(", ");
        crate::emit_message(&format!(
          "ReadList::intnm: Non-negative machine-sized integer expected at position 3 in ReadList[{}].",
          formatted_args
        ));
        return Ok(Expr::FunctionCall {
          name: "ReadList".to_string(),
          args: args.to_vec().into(),
        });
      }
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

  Ok(Expr::List(results.into()))
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

    results.push(Expr::List(record.into()));
  }

  Ok(Expr::List(results.into()))
}

/// Convert an expression to C language format
/// Flatten a (possibly nested) Times tree — represented either as
/// FunctionCall{Times} or chained BinaryOp::Times — into a flat factor list.
fn flatten_times(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args.iter() {
        flatten_times(a, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      flatten_times(left, out);
      flatten_times(right, out);
    }
    _ => out.push(expr.clone()),
  }
}

/// Render a single Times factor for C/Fortran output, parenthesizing
/// additive (Plus/Minus) factors so the surrounding product stays correct.
fn c_like_factor(expr: &Expr, fortran: bool) -> String {
  let s = if fortran {
    expr_to_fortran(expr)
  } else {
    expr_to_c(expr)
  };
  if matches!(expr, Expr::FunctionCall { name, .. } if name == "Plus")
    || matches!(
      expr,
      Expr::BinaryOp {
        op: BinaryOperator::Plus | BinaryOperator::Minus,
        ..
      }
    )
  {
    format!("({})", s)
  } else {
    s
  }
}

/// Render a Times expression as a C/Fortran fraction with correct
/// numerator/denominator grouping. Factors with a negative integer power and
/// the denominator of a Rational coefficient move below the division bar;
/// a leading -1 coefficient becomes a unary minus. `fortran` selects backend.
fn c_like_times(args: &[Expr], fortran: bool) -> String {
  let render = |e: &Expr| {
    if fortran {
      expr_to_fortran(e)
    } else {
      expr_to_c(e)
    }
  };
  // A leading -1 coefficient renders as a unary minus over the rest.
  let (neg, factors): (bool, &[Expr]) =
    if args.len() > 1 && matches!(&args[0], Expr::Integer(-1)) {
      (true, &args[1..])
    } else {
      (false, args)
    };

  let mut num: Vec<String> = Vec::new();
  let mut den: Vec<String> = Vec::new();
  for a in factors {
    // Negative integer power -> denominator factor with positive exponent
    // (as_neg_int_power already returns the positive exponent).
    if let Some((base, e)) = as_neg_int_power(a) {
      if e == 1 {
        den.push(c_like_factor(base, fortran));
      } else {
        let pos = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![base.clone(), Expr::Integer(e)].into(),
        };
        den.push(render(&pos));
      }
      continue;
    }
    // Rational coefficient p/q -> p above (unless 1), q. below the bar.
    if let Expr::FunctionCall { name, args: ra } = a
      && name == "Rational"
      && ra.len() == 2
      && let (Expr::Integer(p), Expr::Integer(q)) = (&ra[0], &ra[1])
    {
      if *p != 1 {
        num.push(p.to_string());
      }
      den.push(format!("{}.", q));
      continue;
    }
    num.push(c_like_factor(a, fortran));
  }

  let group = |v: &[String]| -> String {
    if v.len() > 1 {
      format!("({})", v.join("*"))
    } else {
      v.join("*")
    }
  };
  let compound = !den.is_empty() || num.len() > 1;
  let body = if den.is_empty() {
    if num.is_empty() {
      "1".to_string()
    } else {
      num.join("*")
    }
  } else {
    let num_s = if num.is_empty() {
      "1".to_string()
    } else {
      group(&num)
    };
    format!("{}/{}", num_s, group(&den))
  };
  if neg && compound {
    format!("-({})", body)
  } else if neg {
    format!("-{}", body)
  } else {
    body
  }
}

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
        let mut factors = Vec::new();
        flatten_times(expr, &mut factors);
        c_like_times(&factors, false)
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
        // CForm evaluates exact rationals to machine-precision decimals.
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          let s = format!("{}", *n as f64 / *d as f64);
          if s.contains('.') || s.contains('e') || s.contains('E') {
            s
          } else {
            format!("{}.", s)
          }
        } else {
          format!("{}/{}", expr_to_c(&args[0]), c_paren(&args[1]))
        }
      }
      _ => format!("{}({})", name, c_args(args)),
    },
    Expr::BinaryOp { op, left, right } => {
      let l = expr_to_c(left);
      let r = expr_to_c(right);
      match op {
        BinaryOperator::Plus => format!("{} + {}", l, r),
        BinaryOperator::Minus => format!("{} - {}", l, r),
        BinaryOperator::Times => {
          let mut factors = Vec::new();
          flatten_times(expr, &mut factors);
          c_like_times(&factors, false)
        }
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
        let mut factors = Vec::new();
        flatten_times(expr, &mut factors);
        c_like_times(&factors, true)
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
        BinaryOperator::Times => {
          let mut factors = Vec::new();
          flatten_times(expr, &mut factors);
          c_like_times(&factors, true)
        }
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
      args: args.to_vec().into(),
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
        args: args.to_vec().into(),
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
  Ok(Expr::List(parts.into()))
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
      args: args.to_vec().into(),
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
      Ok(Expr::List(encoded?.into()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "URLEncode".to_string(),
      args: args.to_vec().into(),
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
      Ok(Expr::List(decoded?.into()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "URLDecode".to_string(),
      args: args.to_vec().into(),
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
        args: args.to_vec().into(),
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
    args: vec![Expr::String(b64)].into(),
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
          args: args.to_vec().into(),
        });
      }
    };
    return Ok(Expr::String(String::from_utf8_lossy(&bytes).to_string()));
  }
  crate::emit_message(&format!(
    "ByteArrayToString::barray: {} is not a ByteArray object or {{}}.",
    crate::syntax::expr_to_string(&args[0])
  ));
  Ok(Expr::FunctionCall {
    name: "ByteArrayToString".to_string(),
    args: args.to_vec().into(),
  })
}

/// Extract the bytes of a ByteArray, which stores its data either as a base64
/// string or as a list of integer byte values.
fn byte_array_bytes(expr: &Expr) -> Option<Vec<u8>> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "ByteArray"
    && args.len() == 1
  {
    use base64::Engine;
    let engine = base64::engine::general_purpose::STANDARD;
    return match &args[0] {
      Expr::String(b64) => engine.decode(b64).ok(),
      Expr::List(items) => items
        .iter()
        .map(|e| match e {
          Expr::Integer(n) if (0..=255).contains(n) => Some(*n as u8),
          _ => None,
        })
        .collect(),
      _ => None,
    };
  }
  None
}

/// BaseEncode[bytearray] — Base64-encode a ByteArray to a string.
pub fn base_encode_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "BaseEncode".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 1 {
    return unevaluated();
  }
  match byte_array_bytes(&args[0]) {
    Some(bytes) => {
      use base64::Engine;
      let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
      Ok(Expr::String(b64))
    }
    None => {
      crate::emit_message(&format!(
        "BaseEncode::barray: {} is not a ByteArray object.",
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
      unevaluated()
    }
  }
}

/// BaseDecode["string"] — Base64-decode a string to a ByteArray.
pub fn base_decode_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "BaseDecode".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 1 {
    return unevaluated();
  }
  match &args[0] {
    Expr::String(s) => {
      use base64::Engine;
      let engine = base64::engine::general_purpose::STANDARD;
      match engine.decode(s) {
        // Store the (canonicalised) base64 — Woxi's ByteArray representation.
        Ok(bytes) => Ok(Expr::FunctionCall {
          name: "ByteArray".to_string(),
          args: vec![Expr::String(engine.encode(&bytes))].into(),
        }),
        Err(_) => unevaluated(),
      }
    }
    other => {
      crate::emit_message(&format!(
        "BaseDecode::strx: String expected instead of {}.",
        crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
      ));
      unevaluated()
    }
  }
}

/// TextSentences["string"] — split a string into sentences.
/// TextSentences["string", n] — first n sentences.
///
/// Rule-based segmentation approximating Wolfram's NLP behaviour:
/// a sentence ends at a run of `.`/`!`/`?` (plus any closing quotes or
/// brackets) followed by whitespace or end of input — unless the run is
/// an ellipsis or the preceding token is an abbreviation.
pub fn text_sentences_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "TextSentences".to_string(),
    args: args.to_vec().into(),
  };

  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated(args));
  }
  let text = match &args[0] {
    Expr::String(s) => s,
    _ => {
      crate::emit_message(
        "TextSentences::arg1: String or ContentObject expected at position 1.",
      );
      return Ok(unevaluated(args));
    }
  };
  let limit = match args.get(1) {
    None => None,
    Some(Expr::Integer(n)) if *n >= 1 => Some(*n as usize),
    Some(_) => {
      crate::emit_message(
        "TextSentences::arg2: Positive integer expected at position 2.",
      );
      return Ok(unevaluated(args));
    }
  };

  let mut sentences: Vec<Expr> = split_sentences(text)
    .into_iter()
    .map(Expr::String)
    .collect();
  if let Some(n) = limit {
    sentences.truncate(n);
  }
  Ok(Expr::List(sentences.into()))
}

fn split_sentences(text: &str) -> Vec<String> {
  let chars: Vec<char> = text.chars().collect();
  let mut sentences = Vec::new();
  let mut start = 0usize;
  let mut i = 0usize;

  while i < chars.len() {
    if !matches!(chars[i], '.' | '!' | '?') {
      i += 1;
      continue;
    }

    // Collect the terminator run (e.g. ".", "?!", "...")
    let run_start = i;
    while i < chars.len() && matches!(chars[i], '.' | '!' | '?') {
      i += 1;
    }
    let run_len = i - run_start;
    let all_dots = chars[run_start..i].iter().all(|&c| c == '.');

    // Include closing quotes/brackets in the sentence
    while i < chars.len()
      && matches!(
        chars[i],
        '"' | '\'' | '\u{201d}' | '\u{2019}' | ')' | ']' | '}'
      )
    {
      i += 1;
    }
    let boundary_end = i;

    // A boundary needs trailing whitespace or end of input
    // (so decimals like "5.50" never split)
    if boundary_end < chars.len() && !chars[boundary_end].is_whitespace() {
      continue;
    }
    // An ellipsis does not end a sentence
    if all_dots && run_len >= 3 {
      continue;
    }
    // Abbreviations (only relevant for a single ".") do not end a sentence
    if all_dots
      && run_len == 1
      && is_abbreviation(&chars[..run_start], &chars[boundary_end..])
    {
      continue;
    }

    let sentence: String = chars[start..boundary_end].iter().collect();
    let sentence = sentence.trim();
    if !sentence.is_empty() {
      sentences.push(sentence.to_string());
    }
    start = boundary_end;
  }

  // Trailing text without a terminator is its own sentence
  let rest: String = chars[start..].iter().collect();
  let rest = rest.trim();
  if !rest.is_empty() {
    sentences.push(rest.to_string());
  }

  sentences
}

/// Decide whether the token directly before a `.` is an abbreviation.
/// `before` is everything up to (not including) the period;
/// `after` is everything after the boundary (starting at the whitespace).
fn is_abbreviation(before: &[char], after: &[char]) -> bool {
  // Token immediately preceding the period (letters and internal periods)
  let mut tok_start = before.len();
  while tok_start > 0 && !before[tok_start - 1].is_whitespace() {
    tok_start -= 1;
  }
  let token: String = before[tok_start..].iter().collect();
  // Strip leading quotes/brackets from the token
  let token = token
    .trim_start_matches(['"', '\'', '\u{201c}', '\u{2018}', '(', '[', '{']);

  if token.is_empty() {
    return false;
  }

  // Tokens with internal periods: U.S.A, p.m, a.m, e.g, i.e, ...
  if token.contains('.') {
    return true;
  }
  // Single-letter initials: "J. Smith"
  if token.chars().count() == 1 && token.chars().all(|c| c.is_alphabetic()) {
    return true;
  }
  // Titles and common abbreviations that never end a sentence
  const ALWAYS: &[&str] = &[
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Rev", "Gen", "Sen", "Rep", "Sgt", "Col",
    "Capt", "Lt", "Mt", "St", "vs", "cf", "al",
  ];
  if ALWAYS.contains(&token) {
    return true;
  }
  // Abbreviations that only continue when followed by a number
  // ("No. 5", "Fig. 3", "Eq. 2")
  const BEFORE_NUMBER: &[&str] = &["No", "Fig", "Eq", "Sec", "Ch", "pp", "p"];
  if BEFORE_NUMBER.contains(&token) {
    let next = after.iter().find(|c| !c.is_whitespace());
    return matches!(next, Some(c) if c.is_ascii_digit());
  }

  false
}

/// Render PaddedForm[value, spec] as wolframscript's padded string.
/// Integer spec n: right-aligned to width n + 1. List spec {n, f}:
/// rounded to f decimals (with trailing zeros) and right-aligned to
/// width n + 2. None for non-numeric values or malformed specs.
/// Right-align an integer's digit string into a padded field. A column is always
/// reserved for the sign, and the digit field widens past `n` when the number
/// has more digits, so e.g. `123` with n=2 renders as ` 123` (width 4).
fn pad_integer_body(body: &str, n: usize) -> String {
  let num_digits = body.trim_start_matches('-').chars().count();
  let width = n.max(num_digits) + 1;
  format!("{body:>width$}")
}

fn padded_form_to_string(value: &Expr, spec: &Expr) -> Option<String> {
  // A list pads each element with the same spec and renders as `{e1, e2, ...}`.
  if let Expr::List(items) = value {
    let parts: Option<Vec<String>> = items
      .iter()
      .map(|e| padded_form_to_string(e, spec))
      .collect();
    return Some(format!("{{{}}}", parts?.join(", ")));
  }
  match spec {
    Expr::Integer(n) if *n >= 0 => match value {
      Expr::Integer(i) => Some(pad_integer_body(&i.to_string(), *n as usize)),
      Expr::BigInteger(i) => {
        Some(pad_integer_body(&i.to_string(), *n as usize))
      }
      _ => {
        // Reals/rationals: render via output form, padded to width n+1.
        crate::functions::math_ast::expr_to_num(value)?;
        let body = crate::syntax::expr_to_output(value);
        let width = (*n as usize) + 1;
        Some(format!("{body:>width$}"))
      }
    },
    Expr::List(parts) if parts.len() == 2 => {
      if let (Expr::Integer(n), Expr::Integer(f)) = (&parts[0], &parts[1])
        && *n >= 0
        && *f >= 0
      {
        match value {
          // An integer ignores the fractional spec: it is shown as an integer
          // padded to width n+1, never with a spurious decimal part.
          Expr::Integer(i) => {
            Some(pad_integer_body(&i.to_string(), *n as usize))
          }
          Expr::BigInteger(i) => {
            Some(pad_integer_body(&i.to_string(), *n as usize))
          }
          _ => {
            let v = crate::functions::math_ast::expr_to_num(value)?;
            let body = format!("{v:.prec$}", prec = *f as usize);
            let width = (*n as usize) + 2;
            Some(format!("{body:>width$}"))
          }
        }
      } else {
        None
      }
    }
    _ => None,
  }
}

/// StringExtract[s, n], StringExtract[s, {n1, ...}],
/// StringExtract[s, "delim" -> n, ...] — extract whitespace-separated
/// fields, or fields split by the given delimiters with chained rules
/// drilling into nested fields. Out-of-range positions give
/// Missing[PartAbsent, n]; invalid specifications emit ::patt.
enum Pos {
  One(i64),
  Many(Vec<i64>),
  /// A span `i ;; j` of block positions (end `None` means `All`/open).
  Range(i64, Option<i64>),
}

pub fn string_extract_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "StringExtract".to_string(),
    args: args.to_vec().into(),
  };
  // A list of strings maps the extraction over its elements
  if let Expr::List(items) = &args[0] {
    let mut out = Vec::with_capacity(items.len());
    for item in items.iter() {
      let mut sub_args = args.to_vec();
      sub_args[0] = item.clone();
      out.push(string_extract_ast(&sub_args)?);
    }
    return Ok(Expr::List(out.into()));
  }
  let s = match &args[0] {
    Expr::String(s) => s.clone(),
    _ => {
      crate::emit_message(&format!(
        "StringExtract::strse: A string or list of strings is expected at position 1 in {}.",
        crate::syntax::format_expr(
          &Expr::FunctionCall {
            name: "StringExtract".to_string(),
            args: args.to_vec().into(),
          },
          crate::syntax::ExprForm::Output
        )
      ));
      return Ok(unevaluated());
    }
  };
  // Parse the spec list: each step is (delimiter, positions)
  let parse_pos = |e: &Expr| -> Option<Pos> {
    match e {
      Expr::Integer(n) if *n != 0 => Some(Pos::One(*n as i64)),
      Expr::List(items) => {
        let mut v = Vec::with_capacity(items.len());
        for i in items.iter() {
          match i {
            Expr::Integer(n) if *n != 0 => v.push(*n as i64),
            _ => return None,
          }
        }
        Some(Pos::Many(v))
      }
      // A span `i ;; j` selects a contiguous range of blocks. Only the
      // two-argument form is supported; a stepped span `i ;; j ;; k` is
      // rejected (matching Wolfram's ::patt behaviour).
      Expr::FunctionCall { name, args }
        if name == "Span" && args.len() == 2 =>
      {
        let start = match &args[0] {
          Expr::Integer(n) => *n as i64,
          Expr::Identifier(a) if a == "All" => 1,
          _ => return None,
        };
        let end = match &args[1] {
          Expr::Integer(n) => Some(*n as i64),
          Expr::Identifier(a) if a == "All" => None,
          _ => return None,
        };
        Some(Pos::Range(start, end))
      }
      _ => None,
    }
  };
  let mut steps: Vec<(Option<String>, Pos)> = Vec::new();
  for spec in &args[1..] {
    match spec {
      Expr::Rule {
        pattern,
        replacement,
      } => {
        let delim = match pattern.as_ref() {
          Expr::String(d) => d.clone(),
          _ => {
            crate::emit_message(&format!(
              "StringExtract::patt: {} is not a valid extraction specification.",
              crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output)
            ));
            return Ok(unevaluated());
          }
        };
        match parse_pos(replacement) {
          Some(p) => steps.push((Some(delim), p)),
          None => {
            crate::emit_message(&format!(
              "StringExtract::patt: {} is not a valid extraction specification.",
              crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output)
            ));
            return Ok(unevaluated());
          }
        }
      }
      other => match parse_pos(other) {
        Some(p) => steps.push((None, p)),
        None => {
          crate::emit_message(&format!(
            "StringExtract::patt: {} is not a valid extraction specification.",
            crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
          ));
          return Ok(unevaluated());
        }
      },
    }
  }

  // Split into fields (default: whitespace runs; explicit delimiter:
  // literal split with empty fields dropped, like StringSplit)
  let split = |text: &str, delim: &Option<String>| -> Vec<String> {
    match delim {
      None => text.split_whitespace().map(str::to_string).collect(),
      Some(d) => text
        .split(d.as_str())
        .filter(|f| !f.is_empty())
        .map(str::to_string)
        .collect(),
    }
  };
  let pick = |fields: &[String], n: i64| -> Result<String, Expr> {
    let len = fields.len() as i64;
    let idx = if n > 0 { n - 1 } else { len + n };
    if idx >= 0 && idx < len {
      Ok(fields[idx as usize].clone())
    } else {
      Err(Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![
          Expr::String("PartAbsent".to_string()),
          Expr::Integer(n as i128),
        ]
        .into(),
      })
    }
  };

  // Apply the steps recursively; a list of positions yields a list at
  // that level, and later steps map over it.
  fn apply_steps(
    item: &Expr,
    steps: &[(Option<String>, Pos)],
    split: &dyn Fn(&str, &Option<String>) -> Vec<String>,
    pick: &dyn Fn(&[String], i64) -> Result<String, Expr>,
  ) -> Expr {
    let Some((delim, pos)) = steps.first() else {
      return item.clone();
    };
    match item {
      Expr::List(items) => Expr::List(
        items
          .iter()
          .map(|i| apply_steps(i, steps, split, pick))
          .collect::<Vec<_>>()
          .into(),
      ),
      Expr::String(text) => {
        let fields = split(text, delim);
        let one = |n: i64| match pick(&fields, n) {
          Ok(f) => apply_steps(&Expr::String(f), &steps[1..], split, pick),
          Err(m) => m,
        };
        match pos {
          Pos::One(n) => one(*n),
          Pos::Many(ns) => {
            Expr::List(ns.iter().map(|&n| one(n)).collect::<Vec<_>>().into())
          }
          Pos::Range(start, end) => {
            // Resolve each endpoint (negatives count from the end) and
            // clamp both into [1, len]; an empty/reversed range yields {}.
            let len = fields.len() as i64;
            let resolve = |n: i64| -> i64 {
              let r = if n > 0 {
                n
              } else if n < 0 {
                len + n + 1
              } else {
                0
              };
              r.max(1).min(len)
            };
            let lo = resolve(*start);
            let hi = match end {
              Some(e) => resolve(*e),
              None => len,
            };
            let mut out = Vec::new();
            if len > 0 && lo <= hi {
              for idx in lo..=hi {
                out.push(one(idx));
              }
            }
            Expr::List(out.into())
          }
        }
      }
      other => other.clone(),
    }
  }
  Ok(apply_steps(&Expr::String(s), &steps, &split, &pick))
}
