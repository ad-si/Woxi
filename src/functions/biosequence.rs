//! String functions applied to `BioSequence` objects.
//!
//! In Wolfram Language the String* family accepts `BioSequence[type, seq]`
//! in place of a string: the operation runs on the underlying letters and
//! string results come back wrapped as a `BioSequence` of the same type.
//! In string patterns, degenerate letters (IUPAC codes such as `N`, `Y`,
//! `R`) match every letter they stand for, unless wrapped in `Verbatim`.
//!
//! This module is the single adapter for all of that: it unwraps the
//! sequence, translates the pattern argument once, delegates to the
//! canonical string implementation, and rewraps the result.

use crate::InterpreterError;
use crate::syntax::Expr;

/// The letter alphabet of a linear sequence type, as
/// `(letter, set of base letters it stands for)` pairs.
fn alphabet(kind: &str) -> Option<&'static [(char, &'static str)]> {
  const DNA: &[(char, &str)] = &[
    ('A', "A"),
    ('C', "C"),
    ('G', "G"),
    ('T', "T"),
    ('R', "AG"),
    ('Y', "CT"),
    ('S', "CG"),
    ('W', "AT"),
    ('K', "GT"),
    ('M', "AC"),
    ('B', "CGT"),
    ('D', "AGT"),
    ('H', "ACT"),
    ('V', "ACG"),
    ('N', "ACGT"),
  ];
  const RNA: &[(char, &str)] = &[
    ('A', "A"),
    ('C', "C"),
    ('G', "G"),
    ('U', "U"),
    ('R', "AG"),
    ('Y', "CU"),
    ('S', "CG"),
    ('W', "AU"),
    ('K', "GU"),
    ('M', "AC"),
    ('B', "CGU"),
    ('D', "AGU"),
    ('H', "ACU"),
    ('V', "ACG"),
    ('N', "ACGU"),
  ];
  match kind {
    "DNA" => Some(DNA),
    "RNA" => Some(RNA),
    _ => None,
  }
}

/// A `BioSequence[type, "letters", extra...]` subject. Trailing
/// arguments (such as an empty annotation list) are carried over to
/// results unchanged.
struct Seq {
  kind: String,
  letters: String,
  extra: Vec<Expr>,
}

impl Seq {
  fn from_expr(expr: &Expr) -> Option<Seq> {
    let Expr::FunctionCall { name, args } = expr else {
      return None;
    };
    if name != "BioSequence" || args.len() < 2 || args.len() > 3 {
      return None;
    }
    match (&args[0], &args[1]) {
      (Expr::String(t), Expr::String(s)) => Some(Seq {
        kind: t.clone(),
        letters: s.clone(),
        extra: args.iter().skip(2).cloned().collect(),
      }),
      _ => None,
    }
  }

  /// A sequence of the same type (and annotations) with other letters.
  fn with_letters(&self, letters: String) -> Expr {
    let mut args = vec![Expr::String(self.kind.clone()), Expr::String(letters)];
    args.extend(self.extra.iter().cloned());
    Expr::FunctionCall {
      name: "BioSequence".to_string(),
      args: args.into(),
    }
  }

  /// Wrap every string of a (possibly nested list) result as a sequence
  /// of this type.
  fn rewrap(&self, expr: &Expr) -> Expr {
    match expr {
      Expr::String(s) => self.with_letters(s.clone()),
      Expr::List(items) => {
        Expr::List(items.iter().map(|e| self.rewrap(e)).collect())
      }
      other => other.clone(),
    }
  }
}

/// The string-pattern form of one pattern letter: the letter itself, or
/// for a degenerate letter, the alternatives of every letter whose bases
/// it covers (including itself).
fn letter_pattern(c: char, table: &[(char, &str)]) -> Expr {
  let Some((_, bases)) = table.iter().find(|(l, _)| *l == c) else {
    return Expr::String(c.to_string());
  };
  let covered: Vec<Expr> = table
    .iter()
    .filter(|(_, b)| b.chars().all(|x| bases.contains(x)))
    .map(|(l, _)| Expr::String(l.to_string()))
    .collect();
  if covered.len() == 1 {
    covered.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Alternatives".to_string(),
      args: covered.into(),
    }
  }
}

/// Translate a string pattern so that degenerate letters in literal
/// strings match the letters they represent. `Verbatim[s]` keeps `s`
/// literal; only the left-hand side of rules is a pattern.
fn translate_pattern(expr: &Expr, table: &[(char, &str)]) -> Expr {
  match expr {
    Expr::String(s) => {
      let degenerate =
        |c: char| table.iter().any(|(l, bases)| *l == c && bases.len() > 1);
      if !s.chars().any(degenerate) {
        return expr.clone();
      }
      // Merge runs of plain letters into one literal.
      let mut parts: Vec<Expr> = Vec::new();
      let mut run = String::new();
      for c in s.chars() {
        if degenerate(c) {
          if !run.is_empty() {
            parts.push(Expr::String(std::mem::take(&mut run)));
          }
          parts.push(letter_pattern(c, table));
        } else {
          run.push(c);
        }
      }
      if !run.is_empty() {
        parts.push(Expr::String(run));
      }
      if parts.len() == 1 {
        parts.pop().unwrap()
      } else {
        Expr::FunctionCall {
          name: "StringExpression".to_string(),
          args: parts.into(),
        }
      }
    }
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| translate_pattern(e, table)).collect())
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(translate_pattern(pattern, table)),
      replacement: replacement.clone(),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(translate_pattern(pattern, table)),
      replacement: replacement.clone(),
    },
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Verbatim" if args.len() == 1 => args[0].clone(),
      "StringExpression" | "Alternatives" | "Repeated" | "RepeatedNull"
      | "Shortest" | "Longest" | "Except" => Expr::FunctionCall {
        name: name.clone(),
        args: args.iter().map(|e| translate_pattern(e, table)).collect(),
      },
      "Rule" | "RuleDelayed" | "Condition" | "PatternTest"
        if !args.is_empty() =>
      {
        let mut new_args: Vec<Expr> = args.iter().cloned().collect();
        new_args[0] = translate_pattern(&args[0], table);
        Expr::FunctionCall {
          name: name.clone(),
          args: new_args.into(),
        }
      }
      "Pattern" if args.len() == 2 => Expr::FunctionCall {
        name: name.clone(),
        args: vec![args[0].clone(), translate_pattern(&args[1], table)].into(),
      },
      _ => expr.clone(),
    },
    _ => expr.clone(),
  }
}

/// String functions whose second argument is a string pattern.
fn takes_pattern(name: &str) -> bool {
  matches!(
    name,
    "StringReplace"
      | "StringMatchQ"
      | "StringSplit"
      | "StringCases"
      | "StringFreeQ"
      | "StringPosition"
      | "StringCount"
      | "StringContainsQ"
      | "StringDelete"
      | "StringStartsQ"
      | "StringEndsQ"
  )
}

/// String functions that take a position, count or replacement rather
/// than a pattern; the sequence letters are used as plain characters.
fn takes_plain(name: &str, argc: usize) -> bool {
  match name {
    "StringLength" | "StringReverse" => argc == 1,
    "StringTake" | "StringDrop" | "StringTakeDrop" => argc == 2,
    "StringRotateLeft" | "StringRotateRight" => argc == 2,
    "StringRepeat" => argc == 2 || argc == 3,
    "StringInsert" | "StringReplacePart" => argc == 3,
    "StringPadLeft" | "StringPadRight" => argc == 2 || argc == 3,
    "StringPart" => argc == 2,
    _ => false,
  }
}

/// Evaluate `name[args]` with the given (already converted) arguments
/// through the canonical string implementation. `None` when it stays
/// unevaluated.
fn delegate(
  name: &str,
  args: Vec<Expr>,
) -> Option<Result<Expr, InterpreterError>> {
  let call = Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  };
  match crate::evaluator::evaluate_expr_to_expr(&call) {
    Ok(Expr::FunctionCall { name: ref n, .. }) if n == name => None,
    other => Some(other),
  }
}

/// Evaluate `name[args]` for a String* function whose subject is a
/// `BioSequence`. Returns `None` when the call does not involve a
/// supported BioSequence, so the ordinary string dispatch applies.
pub fn dispatch_biosequence_string(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if name == "StringJoin" {
    return string_join(args);
  }
  let seq = Seq::from_expr(args.first()?)?;
  if let Some(linear_type) = seq.kind.strip_prefix("Circular") {
    let table = alphabet(linear_type)?;
    return circular(name, args, &seq, table);
  }
  let table = alphabet(&seq.kind)?;
  let mut new_args: Vec<Expr> = args.to_vec();
  new_args[0] = Expr::String(seq.letters.clone());
  if takes_pattern(name) && args.len() >= 2 {
    new_args[1] = translate_pattern(&args[1], table);
  } else if !takes_plain(name, args.len()) {
    return None;
  }
  if matches!(name, "StringPadLeft" | "StringPadRight") && args.len() == 2 {
    // Sequences are padded with the fully degenerate letter, not spaces.
    new_args.push(Expr::String("N".to_string()));
  }
  Some(delegate(name, new_args)?.map(|r| seq.rewrap(&r)))
}

/// `StringJoin` of BioSequences of one linear type.
fn string_join(args: &[Expr]) -> Option<Result<Expr, InterpreterError>> {
  let mut first: Option<Seq> = None;
  let mut joined = String::new();
  for arg in args {
    let seq = Seq::from_expr(arg)?;
    alphabet(&seq.kind)?;
    if !seq.extra.is_empty()
      || first.as_ref().is_some_and(|f| f.kind != seq.kind)
    {
      return None;
    }
    joined.push_str(&seq.letters);
    first.get_or_insert(seq);
  }
  Some(Ok(first?.with_letters(joined)))
}

/// A reversed range `{i, j}` with `j < i` on a circular sequence of
/// length `n`, which wraps around from position `n` to position 1.
fn wrapping_range(spec: &Expr, n: usize) -> Option<(usize, usize)> {
  let Expr::List(items) = spec else {
    return None;
  };
  match items.as_slice() {
    [Expr::Integer(i), Expr::Integer(j)]
      if 1 <= *j && j < i && *i as usize <= n =>
    {
      Some((*i as usize, *j as usize))
    }
    _ => None,
  }
}

/// The `Overlaps` setting of trailing option arguments, or `None` when
/// they hold anything else.
fn overlaps_option(options: &[Expr], default: bool) -> Option<bool> {
  let mut overlaps = default;
  for opt in options {
    let (lhs, rhs) = match opt {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => return None,
    };
    match (lhs, rhs) {
      (Expr::Identifier(o), Expr::Identifier(v)) if o == "Overlaps" => {
        overlaps = match v.as_str() {
          "True" => true,
          "False" => false,
          _ => return None,
        }
      }
      _ => return None,
    }
  }
  Some(overlaps)
}

fn contains_rule(expr: &Expr) -> bool {
  match expr {
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => true,
    Expr::List(items) => items.iter().any(contains_rule),
    _ => false,
  }
}

/// The patterns of a rule or list of rules, as one alternative.
fn rule_lhs_pattern(rules: &Expr) -> Option<Expr> {
  let lhs = |r: &Expr| match r {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
      Some(pattern.as_ref().clone())
    }
    _ => None,
  };
  match rules {
    Expr::List(items) => Some(Expr::FunctionCall {
      name: "Alternatives".to_string(),
      args: items.iter().map(lhs).collect::<Option<Vec<_>>>()?.into(),
    }),
    other => lhs(other),
  }
}

/// Matches of `pattern` on a circular sequence, as 1-based
/// `(start, end)` spans where `end` may exceed `n` for a match that wraps
/// past the last letter. Without overlaps the leftmost matches win and a
/// wrapping match may not reach into the first match.
fn circular_spans(
  letters: &[char],
  pattern: &Expr,
  overlaps: bool,
) -> Option<Result<Vec<(usize, usize)>, InterpreterError>> {
  let n = letters.len();
  let extended: String = letters
    .iter()
    .chain(letters.iter().take(n.saturating_sub(1)))
    .collect();
  let positions = match delegate(
    "StringPosition",
    vec![Expr::String(extended), pattern.clone()],
  )? {
    Ok(p) => p,
    Err(e) => return Some(Err(e)),
  };
  let Expr::List(items) = &positions else {
    return None;
  };
  let mut spans = Vec::new();
  for item in items {
    let Expr::List(pair) = item else {
      return None;
    };
    let [Expr::Integer(a), Expr::Integer(b)] = pair.as_slice() else {
      return None;
    };
    let (a, b) = (*a as usize, *b as usize);
    if a <= n && b >= a && b - a < n {
      spans.push((a, b));
    }
  }
  if !overlaps {
    let mut chosen: Vec<(usize, usize)> = Vec::new();
    for (a, b) in spans {
      if chosen.last().is_some_and(|&(_, e)| a <= e) {
        continue;
      }
      if b > n && chosen.first().is_some_and(|&(s, _)| b - n >= s) {
        continue;
      }
      chosen.push((a, b));
    }
    spans = chosen;
  }
  Some(Ok(spans))
}

/// String functions on a circular sequence, which wraps around from its
/// last letter to its first for matching and for reversed ranges.
fn circular(
  name: &str,
  args: &[Expr],
  seq: &Seq,
  table: &[(char, &str)],
) -> Option<Result<Expr, InterpreterError>> {
  let letters: Vec<char> = seq.letters.chars().collect();
  let n = letters.len();
  let slice = |from: usize, to: usize| -> String {
    // 1-based inclusive, wrapping past n.
    (from..=to).map(|k| letters[(k - 1) % n]).collect()
  };
  match name {
    "StringLength" if args.len() == 1 => {
      return Some(Ok(Expr::Integer(n as i128)));
    }
    "StringTake" | "StringDrop" | "StringTakeDrop" | "StringReplacePart"
      if args.len() == if name == "StringReplacePart" { 3 } else { 2 } =>
    {
      let spec = &args[args.len() - 1];
      if let Some((i, j)) = wrapping_range(spec, n) {
        let taken = slice(i, j + n);
        let rest = slice(j + 1, i - 1);
        let result = match name {
          "StringTake" => seq.with_letters(taken),
          "StringDrop" => seq.with_letters(rest),
          "StringTakeDrop" => Expr::List(
            vec![seq.with_letters(taken), seq.with_letters(rest)].into(),
          ),
          _ => {
            let Expr::String(r) = &args[1] else {
              return None;
            };
            seq.with_letters(rest + r)
          }
        };
        return Some(Ok(result));
      }
      let mut new_args = args.to_vec();
      new_args[0] = Expr::String(seq.letters.clone());
      return Some(delegate(name, new_args)?.map(|r| seq.rewrap(&r)));
    }
    _ => {}
  }
  if args.len() < 2 || n == 0 {
    return None;
  }
  let pattern = translate_pattern(&args[1], table);
  match name {
    "StringPosition" | "StringCount" | "StringCases" | "StringContainsQ"
    | "StringFreeQ" => {
      if contains_rule(&pattern) {
        return None;
      }
      let overlaps = overlaps_option(&args[2..], name == "StringPosition")?;
      let spans = match circular_spans(&letters, &pattern, overlaps)? {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
      };
      let result = match name {
        "StringPosition" => Expr::List(
          spans
            .iter()
            .map(|&(a, b)| {
              Expr::List(
                vec![
                  Expr::Integer(a as i128),
                  Expr::Integer(((b - 1) % n + 1) as i128),
                ]
                .into(),
              )
            })
            .collect(),
        ),
        "StringCount" => Expr::Integer(spans.len() as i128),
        "StringCases" => Expr::List(
          spans
            .iter()
            .map(|&(a, b)| seq.with_letters(slice(a, b)))
            .collect(),
        ),
        "StringContainsQ" => bool_expr(!spans.is_empty()),
        _ => bool_expr(spans.is_empty()),
      };
      Some(Ok(result))
    }
    // A circular sequence matches when some rotation of it does.
    "StringMatchQ" if args.len() == 2 => {
      for k in 0..n {
        let rotated = Expr::String(slice(k + 1, k + n));
        match delegate(name, vec![rotated, pattern.clone()])? {
          Ok(Expr::Identifier(ref t)) if t == "True" => {
            return Some(Ok(bool_expr(true)));
          }
          Ok(_) => {}
          Err(e) => return Some(Err(e)),
        }
      }
      Some(Ok(bool_expr(false)))
    }
    // How Wolfram orders the letters of a circular result after removing a
    // match that wraps around is not settled, so only calls without such a
    // match are handled (they agree with the linear operation).
    "StringReplace" | "StringDelete" if args.len() == 2 => {
      let lhs = if name == "StringReplace" {
        rule_lhs_pattern(&pattern)?
      } else {
        pattern.clone()
      };
      let spans = match circular_spans(&letters, &lhs, false)? {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
      };
      if spans.iter().any(|&(_, b)| b > n) {
        return None;
      }
      let new_args = vec![Expr::String(seq.letters.clone()), pattern];
      Some(delegate(name, new_args)?.map(|r| seq.rewrap(&r)))
    }
    _ => None,
  }
}

fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}
