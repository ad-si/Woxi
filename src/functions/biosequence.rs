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
fn alphabet(seq_type: &str) -> Option<&'static [(char, &'static str)]> {
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
  match seq_type {
    "DNA" => Some(DNA),
    "RNA" => Some(RNA),
    _ => None,
  }
}

/// `BioSequence[type, "letters"]` → `(type, letters)`.
fn as_biosequence(expr: &Expr) -> Option<(&str, &str)> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "BioSequence" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::String(t), Expr::String(s)) => Some((t.as_str(), s.as_str())),
        _ => None,
      }
    }
    _ => None,
  }
}

fn biosequence(seq_type: &str, letters: String) -> Expr {
  Expr::FunctionCall {
    name: "BioSequence".to_string(),
    args: vec![Expr::String(seq_type.to_string()), Expr::String(letters)]
      .into(),
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

/// Wrap every string of a (possibly nested list) result as a BioSequence.
fn rewrap(expr: &Expr, seq_type: &str) -> Expr {
  match expr {
    Expr::String(s) => biosequence(seq_type, s.clone()),
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| rewrap(e, seq_type)).collect())
    }
    other => other.clone(),
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
    // Without an explicit padding string Wolfram pads with a letter of
    // the sequence alphabet, not a space; only the explicit form is
    // handled.
    "StringPadLeft" | "StringPadRight" => argc == 3,
    _ => false,
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
  let (seq_type, letters) = as_biosequence(args.first()?)?;
  // Circular sequences wrap around for matching and for positions; only
  // the wrap-independent StringLength is supported for them.
  if seq_type.starts_with("Circular") {
    if name == "StringLength"
      && args.len() == 1
      && alphabet(&seq_type[8..]).is_some()
    {
      return Some(Ok(Expr::Integer(letters.chars().count() as i128)));
    }
    return None;
  }
  let table = alphabet(seq_type)?;
  let mut new_args: Vec<Expr> = args.to_vec();
  new_args[0] = Expr::String(letters.to_string());
  if takes_pattern(name) && args.len() >= 2 {
    new_args[1] = translate_pattern(&args[1], table);
  } else if !takes_plain(name, args.len()) {
    return None;
  }
  let call = Expr::FunctionCall {
    name: name.to_string(),
    args: new_args.into(),
  };
  let result = match crate::evaluator::evaluate_expr_to_expr(&call) {
    Ok(r) => r,
    Err(e) => return Some(Err(e)),
  };
  if let Expr::FunctionCall { name: n, .. } = &result
    && n == name
  {
    // The string implementation refused the arguments.
    return None;
  }
  Some(Ok(rewrap(&result, seq_type)))
}

/// `StringJoin` of BioSequences of one linear type.
fn string_join(args: &[Expr]) -> Option<Result<Expr, InterpreterError>> {
  let mut seq_type: Option<&str> = None;
  let mut joined = String::new();
  for arg in args {
    let (t, s) = as_biosequence(arg)?;
    alphabet(t)?;
    if seq_type.is_some_and(|prev| prev != t) {
      return None;
    }
    seq_type = Some(t);
    joined.push_str(s);
  }
  Some(Ok(biosequence(seq_type?, joined)))
}
