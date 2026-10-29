use super::*;

/// Convert an Expr to a boolean value.
/// Returns Some(true) for Identifier("True"), Some(false) for Identifier("False").
pub fn expr_to_bool(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  }
}

/// Apply a function/predicate to an argument and return the resulting Expr.
/// Uses the existing apply_function_to_arg from evaluator.
pub fn apply_func_ast(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::apply_function_to_arg(func, arg)
}

/// Apply `func` to `arg` for its value, at a boundary a `Return` does not
/// cross. `Map` keeps whatever its function produced, `Return[…]` included,
/// where `Scan` — which uses [`apply_func_ast`] — lets it out.
pub fn apply_func_value(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  match apply_func_ast(func, arg) {
    Err(InterpreterError::ReturnValue(val)) => Ok(call1("Return", *val)),
    other => other,
  }
}

/// Apply a binary function to two arguments.
pub fn apply_func_to_two_args(
  func: &Expr,
  arg1: &Expr,
  arg2: &Expr,
) -> Result<Expr, InterpreterError> {
  apply_func_to_args(func, &[arg1.clone(), arg2.clone()])
}

/// Apply a function to an arbitrary number of arguments (generalizes
/// `apply_func_to_two_args` to n-ary application).
pub fn apply_func_to_args(
  func: &Expr,
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, call_args)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, call_args);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(call_args.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    // A compound head is applied to the arguments, never extended by them:
    // g[h] applied to (a, b) is g[h][a, b], not g[h, a, b].
    Expr::FunctionCall { .. } => {
      crate::evaluator::apply_curried_call(func, call_args)
    }
    _ => {
      let func_str = expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(&func_str, call_args)
    }
  }
}

/// Get the head of an expression as a string. Operator nodes
/// (`Expr::BinaryOp`/`UnaryOp`/`Comparison`) resolve to their Wolfram
/// canonical head — `x^2` is `Power`, `a*b`/`a/b`/`-x` are `Times`,
/// `a-b` is `Plus`, `a==b` is `Equal` — so head-constrained patterns
/// (`_Power`, `_Plus`, …) match them correctly. Mirrors `Head[]`.
pub fn get_expr_head_str(expr: &Expr) -> &str {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer",
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real",
    Expr::String(_) => "String",
    Expr::List(_) => "List",
    Expr::FunctionCall { name, .. } => name,
    Expr::Association(_) => "Association",
    Expr::Rule { .. } => "Rule",
    Expr::RuleDelayed { .. } => "RuleDelayed",
    Expr::BinaryOp { op, left, .. } => match op {
      // a - b is stored as a Minus node but is Plus[a, Times[-1, b]] in WL.
      BinaryOperator::Plus | BinaryOperator::Minus => "Plus",
      BinaryOperator::Times => "Times",
      // a/b is Times[a, Power[b, -1]]; 1/b is just Power[b, -1].
      BinaryOperator::Divide => {
        if matches!(left.as_ref(), Expr::Integer(1)) {
          "Power"
        } else {
          "Times"
        }
      }
      BinaryOperator::Power => "Power",
      BinaryOperator::And => "And",
      BinaryOperator::Or => "Or",
      BinaryOperator::StringJoin => "StringJoin",
      BinaryOperator::Alternatives => "Alternatives",
    },
    Expr::UnaryOp { op, .. } => match op {
      UnaryOperator::Minus => "Times",
      UnaryOperator::Not => "Not",
    },
    Expr::Comparison { operators, .. } => {
      let first = operators.first().copied();
      if operators.iter().all(|o| Some(*o) == first) {
        match first {
          Some(ComparisonOp::Equal) | None => "Equal",
          Some(ComparisonOp::NotEqual) => "Unequal",
          Some(ComparisonOp::Less) => "Less",
          Some(ComparisonOp::LessEqual) => "LessEqual",
          Some(ComparisonOp::Greater) => "Greater",
          Some(ComparisonOp::GreaterEqual) => "GreaterEqual",
          Some(ComparisonOp::SameQ) => "SameQ",
          Some(ComparisonOp::UnsameQ) => "UnsameQ",
        }
      } else {
        "Inequality"
      }
    }
    Expr::Image { .. } | Expr::Graphics { .. } => {
      crate::functions::predicate_ast::rendered_object_head(expr)
        .unwrap_or("Symbol")
    }
    _ => "Symbol",
  }
}

/// Helper to extract i128 from Expr
pub(crate) fn expr_to_i128(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i128()
    }
    Expr::Real(f) if f.fract() == 0.0 => Some(*f as i128),
    _ => None,
  }
}

/// Like [`expr_to_i128`] but only for exact integers: a whole-valued Real
/// such as `3.` stays a Real, so an iterator over it yields `3.`, `4.`, …
pub(crate) fn expr_to_exact_i128(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Real(_) => None,
    _ => expr_to_i128(expr),
  }
}

/// One resolved `Do`/`Table`-style `{min, max, step}` numeric iterator spec.
pub(crate) enum NumericSteps {
  /// All three bounds are exact integers — iterate `i128` directly.
  Int { min: i128, max: i128, step: i128 },
  /// Anything else (a Real/Rational bound, or a fractional step). Each
  /// term is computed as `min + k*step` (`k` an exact integer counter)
  /// rather than by repeatedly adding `step` — repeated addition would
  /// accumulate float rounding, e.g. eighteen additions of `0.1` land a
  /// bit off exact `1.8`, breaking a later `f[1.8] = …` lookup keyed on
  /// the loop variable. `min`/`step` staying exact (e.g. `Integer`) is
  /// also what makes `{i, 2, 3.5}` still visit the *exact* integers
  /// `2, 3` instead of `2., 3.`. `min_num`/`max_num`/`step_num` decide,
  /// via `f64`, when to stop.
  Real {
    min_expr: Expr,
    step_expr: Expr,
    min_num: f64,
    max_num: f64,
    step_num: f64,
  },
}

/// Resolves a `Do`/`Table` iterator's `{min, max, step}` triple (already
/// evaluated) into a [`NumericSteps`] plan. Shared by `Do`'s single- and
/// multi-iterator forms so both walk the exact same sequence of values —
/// notably, so a fractional `step` (e.g. `{condInit, 0.1, 3.1, 0.1}`) is
/// iterated as real numbers instead of being floored to zero.
pub(crate) fn resolve_numeric_steps(
  min_expr: Expr,
  max_expr: &Expr,
  step_expr: Expr,
) -> Result<NumericSteps, InterpreterError> {
  if let (Some(min_val), Some(max_val), Some(step_val)) = (
    expr_to_exact_i128(&min_expr),
    expr_to_exact_i128(max_expr),
    expr_to_exact_i128(&step_expr),
  ) {
    if step_val == 0 {
      return Err(InterpreterError::EvaluationError(
        "Do: step cannot be zero".into(),
      ));
    }
    return Ok(NumericSteps::Int {
      min: min_val,
      max: max_val,
      step: step_val,
    });
  }

  let bound_error = || {
    InterpreterError::EvaluationError(
      "Do: iterator bound must be an integer".into(),
    )
  };
  let step_num =
    crate::functions::math_ast::try_eval_to_f64_with_infinity(&step_expr);
  let min_num = crate::functions::math_ast::try_eval_to_f64(&min_expr);
  let max_num =
    crate::functions::math_ast::try_eval_to_f64_with_infinity(max_expr);
  let (min_num, max_num, step_num) = match (min_num, max_num, step_num) {
    (Some(lo), Some(hi), Some(step)) => (lo, hi, step),
    (Some(_), Some(_), None) => {
      return Err(InterpreterError::EvaluationError(
        "Do: step must be an integer".into(),
      ));
    }
    // Symbolic bounds iterate as long as their span is a number of steps:
    // {n, x, x + 3 y, y} visits x, x + y, x + 2 y, x + 3 y.
    _ => {
      let span = crate::evaluator::evaluate_function_call_ast(
        "Together",
        &[crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Minus,
            left: Box::new(max_expr.clone()),
            right: Box::new(min_expr.clone()),
          }),
          right: Box::new(step_expr.clone()),
        })?],
      )?;
      let count = crate::functions::math_ast::try_eval_to_f64(&span)
        .ok_or_else(bound_error)?;
      (0.0, count, 1.0)
    }
  };
  if step_num == 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Do: step cannot be zero".into(),
    ));
  }

  Ok(NumericSteps::Real {
    min_expr,
    step_expr,
    min_num,
    max_num,
    step_num,
  })
}

/// Walks the numeric sequence implied by a `{min, max, step}` spec,
/// calling `visit` once per value in order. `visit` returns `Ok(true)` to
/// keep going or `Ok(false)` to stop early (e.g. on `Break[]`); a genuine
/// evaluation error propagates as `Err`.
pub(crate) fn for_each_numeric_step(
  min_expr: Expr,
  max_expr: &Expr,
  step_expr: Expr,
  mut visit: impl FnMut(Expr) -> Result<bool, InterpreterError>,
) -> Result<(), InterpreterError> {
  match resolve_numeric_steps(min_expr, max_expr, step_expr)? {
    NumericSteps::Int { min, max, step } => {
      let mut i = min;
      if step > 0 {
        while i <= max {
          if !visit(Expr::Integer(i))? {
            return Ok(());
          }
          i += step;
        }
      } else {
        while i >= max {
          if !visit(Expr::Integer(i))? {
            return Ok(());
          }
          i += step;
        }
      }
    }
    NumericSteps::Real {
      min_expr,
      step_expr,
      min_num,
      max_num,
      step_num,
    } => {
      // As many values as Range gives: rounding in (max - min)/step must
      // neither add nor drop the last one, at any magnitude (a slack of
      // machine epsilon ran a step of 10^-20 a million times).
      // An infinite bound ({k, Infinity}) runs until Break or Return.
      let ratio = (max_num - min_num) / step_num;
      let unbounded = ratio == f64::INFINITY;
      let count = if unbounded {
        i128::MAX
      } else {
        super::construction::range_step_count(ratio) + 1
      };
      let mut k: i128 = 0;
      loop {
        if k >= count {
          break;
        }
        // `min + k*step` for every k, the first included: a machine-real
        // step makes `{i, 1, 2, 1.}` start at `1.`.
        let current_expr = super::construction::nth_iterator_value(
          &min_expr, &step_expr, k as usize,
        )?;
        if !visit(current_expr)? {
          return Ok(());
        }
        k += 1;
        if !unbounded && k > 1_000_000 {
          return Err(InterpreterError::EvaluationError(
            "Do: iterator exceeded maximum iterations".into(),
          ));
        }
      }
    }
  }
  Ok(())
}

/// Apply a function to n arguments.
pub fn apply_func_to_n_args(
  func: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  apply_func_to_args(func, args)
}

/// Helper to extract f64 from Expr
pub(crate) fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_f64()
    }
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

/// `expr_to_f64` extended with exact rationals (`Rational[n, d]`), for the
/// histogram/bin-count family whose data wolframscript machine-numericizes.
pub(crate) fn numeric_expr_to_f64(expr: &Expr) -> Option<f64> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
  {
    return match (expr_to_f64(&args[0]), expr_to_f64(&args[1])) {
      (Some(n), Some(d)) if d != 0.0 => Some(n / d),
      _ => None,
    };
  }
  expr_to_f64(expr)
}

/// Helper to convert f64 to appropriate Expr
pub fn f64_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

/// Identity[x] - returns x unchanged
pub fn identity_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(arg.clone())
}

/// Check if an expression is the symbol Nothing
pub fn is_nothing(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(s) if s == "Nothing")
}

/// Check if an expression is an atomic (non-compound) expression
pub fn is_atom_expr(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(_) | Expr::Constant(_) | Expr::String(_))
}
