//! `QuestionObject`, `AssessmentFunction`, and `AssessmentResultObject`.
//!
//! These implement the Wolfram Language quiz/assessment primitives.
//!
//! * `AssessmentFunction[spec]` is a symbolic object that, when applied to a
//!   candidate answer, grades it and returns an `AssessmentResultObject`.
//! * `QuestionObject[q, assess]` (or `QuestionObject[assess]`) pairs a prompt
//!   with an assessment; applying it to an answer grades the answer through the
//!   embedded assessment.
//! * `AssessmentResultObject[<|…|>]` holds the grade of a single answer and
//!   supports property access such as `result["Score"]` /
//!   `result["AnswerCorrect"]`.
//!
//! The grading follows the documented convention: *True and positive numeric
//! scores denote correct answers, while False, zero and negative scores are
//! incorrect.* A spec entry given as a plain value (not a rule) denotes a
//! correct answer with a score of `1`.

use crate::InterpreterError;
use crate::evaluator::pattern_matching::expr_equal;
use crate::syntax::Expr;

/// Turn a numeric-or-boolean grade expression into a `(score, correct)` pair.
/// Returns `None` when the expression is not a recognized grade value.
fn grade_value(v: &Expr) -> Option<(f64, bool)> {
  match v {
    Expr::Identifier(s) if s == "True" => Some((1.0, true)),
    Expr::Identifier(s) if s == "False" => Some((0.0, false)),
    Expr::Integer(n) => Some((*n as f64, *n > 0)),
    Expr::Real(r) => Some((*r, *r > 0.0)),
    // Full specification `<|"Score" -> s, …|>`: read the "Score" key.
    Expr::Association(pairs) => {
      for (k, val) in pairs {
        if matches!(k, Expr::String(s) if s == "Score") {
          return grade_value(val);
        }
      }
      // An association without an explicit score counts as correct.
      Some((1.0, true))
    }
    _ => None,
  }
}

/// Normalize an `AssessmentFunction` specification into `(key, score, correct)`
/// entries. A bare (non-list) spec is treated as a single-entry answer key.
fn spec_entries(spec: &Expr) -> Vec<(Expr, f64, bool)> {
  let elems: Vec<Expr> = match spec {
    Expr::List(items) => items.iter().cloned().collect(),
    other => vec![other.clone()],
  };
  let mut entries = Vec::with_capacity(elems.len());
  for e in elems {
    match &e {
      // `answer -> grade`: the answer is correct according to the grade value.
      Expr::Rule {
        pattern,
        replacement,
      } => {
        if let Some((score, correct)) = grade_value(replacement) {
          entries.push(((**pattern).clone(), score, correct));
        }
      }
      // A bare value is itself a correct answer (score 1).
      _ => entries.push((e.clone(), 1.0, true)),
    }
  }
  entries
}

/// Grade `answer` against a normalized set of spec entries, returning
/// `(score, correct)`. An answer matching no entry scores `0` (incorrect).
fn grade(spec: &Expr, answer: &Expr) -> (f64, bool) {
  for (key, score, correct) in spec_entries(spec) {
    if expr_equal(&key, answer) {
      return (score, correct);
    }
  }
  (0.0, false)
}

/// Render a score as an integer expression when it is whole, else as a real.
fn score_expr(score: f64) -> Expr {
  if score.fract() == 0.0 && score.abs() < 1e15 {
    Expr::Integer(score as i128)
  } else {
    Expr::Real(score)
  }
}

fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

/// Build an `AssessmentResultObject[<|"AnswerCorrect" -> …, "Score" -> …|>]`.
fn result_object(score: f64, correct: bool) -> Expr {
  let assoc = Expr::Association(vec![
    (
      Expr::String("AnswerCorrect".to_string()),
      bool_expr(correct),
    ),
    (Expr::String("Score".to_string()), score_expr(score)),
  ]);
  Expr::FunctionCall {
    name: "AssessmentResultObject".to_string(),
    args: vec![assoc].into(),
  }
}

/// Extract the assessment specification embedded in an `AssessmentFunction`.
/// Returns `None` when `expr` is not an `AssessmentFunction[spec]`.
fn assessment_function_spec(expr: &Expr) -> Option<&Expr> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "AssessmentFunction" && args.len() == 1 =>
    {
      Some(&args[0])
    }
    _ => None,
  }
}

/// Apply `AssessmentFunction[spec]` to `answer`, producing an
/// `AssessmentResultObject`.
pub fn apply_assessment_function(
  func_args: &[Expr],
  answer: &Expr,
) -> Result<Expr, InterpreterError> {
  if func_args.len() != 1 {
    return Ok(keep_curried("AssessmentFunction", func_args, answer));
  }
  let (score, correct) = grade(&func_args[0], answer);
  Ok(result_object(score, correct))
}

/// Apply `QuestionObject[q, assess]` (or `QuestionObject[assess]`) to `answer`.
/// The embedded assessment does the grading; a `QuestionObject` whose
/// assessment is an `AssessmentFunction` is graded through it.
pub fn apply_question_object(
  func_args: &[Expr],
  answer: &Expr,
) -> Result<Expr, InterpreterError> {
  // The assessment is the last argument (`QuestionObject[q, assess]`) or the
  // sole argument (`QuestionObject[assess]`).
  let assess = match func_args.len() {
    1 => &func_args[0],
    2 => &func_args[1],
    _ => return Ok(keep_curried("QuestionObject", func_args, answer)),
  };
  if let Some(spec) = assessment_function_spec(assess) {
    let (score, correct) = grade(spec, answer);
    return Ok(result_object(score, correct));
  }
  // A bare assessment specification (list/rule/value) is graded directly.
  let (score, correct) = grade(assess, answer);
  Ok(result_object(score, correct))
}

/// Access a property of an `AssessmentResultObject[<|…|>]`, e.g.
/// `result["Score"]` or `result["AnswerCorrect"]`. Returns the stored
/// association for `result[All]`. Leaves the call unevaluated (curried) when
/// the property is unknown.
pub fn apply_result_object(
  func_args: &[Expr],
  key: &Expr,
) -> Result<Expr, InterpreterError> {
  if let Some(Expr::Association(pairs)) = func_args.first() {
    // `result[All]` returns the whole association.
    if matches!(key, Expr::Identifier(s) if s == "All") {
      return Ok(Expr::Association(pairs.clone()));
    }
    if let Expr::String(wanted) = key {
      for (k, v) in pairs {
        if matches!(k, Expr::String(s) if s == wanted) {
          return Ok(v.clone());
        }
      }
    }
  }
  Ok(keep_curried("AssessmentResultObject", func_args, key))
}

/// Reconstruct an inert curried call `head[func_args][arg]` for cases that
/// cannot be reduced.
fn keep_curried(head: &str, func_args: &[Expr], arg: &Expr) -> Expr {
  Expr::CurriedCall {
    func: Box::new(Expr::FunctionCall {
      name: head.to_string(),
      args: func_args.to_vec().into(),
    }),
    args: vec![arg.clone()],
  }
}
