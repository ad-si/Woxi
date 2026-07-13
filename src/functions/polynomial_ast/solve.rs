use super::together::negate_expr;
#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
};

use crate::functions::calculus_ast::{is_constant_wrt, simplify};
use crate::functions::math_ast::{is_sqrt, make_sqrt};

/// In Solve context, simplify Sqrt[expr^(2n)] → expr^n since ± handles the sign.
/// Also simplifies products containing such terms.
fn strip_sqrt_square(expr: Expr) -> Expr {
  match &expr {
    // Sqrt[base^(2n)] → base^n
    e if is_sqrt(e).is_some() => {
      let sqrt_arg = is_sqrt(e).unwrap();
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = sqrt_arg
        && let Expr::Integer(n) = exp.as_ref()
        && *n > 0
        && n % 2 == 0
      {
        let half = n / 2;
        if half == 1 {
          return *base.clone();
        } else {
          return Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: base.clone(),
            right: Box::new(Expr::Integer(half)),
          };
        }
      }
      expr
    }
    // c * Sqrt[base^(2n)] → c * base^n
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let new_left = strip_sqrt_square(*left.clone());
      let new_right = strip_sqrt_square(*right.clone());
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(new_left),
        right: Box::new(new_right),
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let new_args: Vec<Expr> =
        args.iter().map(|a| strip_sqrt_square(a.clone())).collect();
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: new_args.into(),
      }
    }
    _ => expr,
  }
}

// ─── NSolve ─────────────────────────────────────────────────────────

/// NSolve[equation, var] — solve an equation numerically.
///
/// For quadratic polynomials, uses Kahan's numerically stable formula to
/// match Wolfram's machine-precision output. For all other equations,
/// solves symbolically first, then converts to numerical form via N[].
pub fn nsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Try numerically stable quadratic formula for degree-2 polynomials
  if let Some(result) = try_nsolve_quadratic(args) {
    return result;
  }
  // Fall back to symbolic solve + numerize
  let symbolic = solve_ast(args)?;
  let numerized = nsolve_numerize(&symbolic)?;
  Ok(sort_nsolve_solutions(numerized))
}

/// wolframscript lists NSolve roots ordered by ascending real part, breaking
/// ties by ascending imaginary part (the symbolic Solve order they inherit is
/// not numerically sorted). Only reorder when every solution is a single
/// numeric `var -> value` rule, so multi-variable systems and any
/// non-numericised solutions are left untouched.
fn sort_nsolve_solutions(expr: Expr) -> Expr {
  let Expr::List(ref items) = expr else {
    return expr;
  };
  let key = |item: &Expr| -> Option<(f64, f64)> {
    if let Expr::List(rules) = item
      && rules.len() == 1
      && let Expr::Rule { replacement, .. } = &rules[0]
    {
      if let Some(v) = crate::functions::math_ast::try_eval_to_f64(replacement)
      {
        return Some((v, 0.0));
      }
      if let Some((re, im)) =
        crate::functions::math_ast::try_extract_complex_float(replacement)
      {
        return Some((re, im));
      }
    }
    None
  };
  let mut items_vec: Vec<Expr> = items.iter().cloned().collect();
  if !items_vec.is_empty() && items_vec.iter().all(|it| key(it).is_some()) {
    items_vec.sort_by(|a, b| {
      let (ar, ai) = key(a).unwrap();
      let (br, bi) = key(b).unwrap();
      ar.partial_cmp(&br)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(ai.partial_cmp(&bi).unwrap_or(std::cmp::Ordering::Equal))
    });
  }
  Expr::List(items_vec.into())
}

/// Try to solve a quadratic equation using Kahan's numerically stable formula.
/// Returns None if the equation is not a degree-2 polynomial with numeric coefficients.
fn try_nsolve_quadratic(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() != 2 {
    return None;
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => return None,
  };

  // Extract equation: lhs == rhs → lhs - rhs
  let poly = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(fargs[0].clone()),
        right: Box::new(fargs[1].clone()),
      }
    }
    _ => return None,
  };

  // Expand and collect polynomial coefficients
  let expanded_raw = expand_and_combine(&poly);
  let expanded = {
    let together = together_expr(&expanded_raw);
    match &together {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: numerator,
        right: _,
      } => expand_and_combine(numerator),
      _ => expanded_raw,
    }
  };
  let terms = collect_additive_terms(&expanded);
  let degree = max_power_int(&expanded, &var)? as usize;

  // Only handle quadratics
  if degree != 2 {
    return None;
  }

  // Extract f64 coefficients
  let mut coeffs_f64 = [0.0f64; 3];
  for d in 0..=2 {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, &var, d as i128) {
        let val = crate::functions::math_ast::try_eval_to_f64(&simplify(c))?;
        coeffs_f64[d] += val;
      }
    }
  }

  let a = coeffs_f64[2];
  let b = coeffs_f64[1];
  let c = coeffs_f64[0];
  let disc = b * b - 4.0 * a * c;

  let make_rule = |val: Expr| -> Expr {
    Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::Identifier(var.clone())),
        replacement: Box::new(val),
      }]
      .into(),
    )
  };

  if disc >= 0.0 {
    let sqrt_disc = disc.sqrt();
    // Kahan's method: compute the well-conditioned root first,
    // then use Vieta's formula (c/q) for the other root.
    // This avoids cancellation when -b and sqrt(disc) nearly cancel.
    let q = if b >= 0.0 {
      -0.5 * (b + sqrt_disc)
    } else {
      -0.5 * (b - sqrt_disc)
    };
    let r1 = q / a;
    let r2 = if q.abs() > 0.0 { c / q } else { r1 };
    let mut roots = [r1, r2];
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(Ok(Expr::List(
      vec![
        make_rule(Expr::Real(roots[0])),
        make_rule(Expr::Real(roots[1])),
      ]
      .into(),
    )))
  } else {
    let sqrt_neg_disc = (-disc).sqrt();
    let re = -b / (2.0 * a);
    let im = sqrt_neg_disc / (2.0 * a);
    let c1 = crate::evaluator::evaluate_function_call_ast(
      "Complex",
      &[Expr::Real(re), Expr::Real(-im.abs())],
    )
    .unwrap_or(Expr::Real(re));
    let c2 = crate::evaluator::evaluate_function_call_ast(
      "Complex",
      &[Expr::Real(re), Expr::Real(im.abs())],
    )
    .unwrap_or(Expr::Real(re));
    Some(Ok(Expr::List(vec![make_rule(c1), make_rule(c2)].into())))
  }
}

/// Principal-branch complex power: (a+bi)^(c+di) = exp((c+di) * Log[a+bi]).
fn complex_pow(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
  let abs_z = (a * a + b * b).sqrt();
  if abs_z == 0.0 {
    return (0.0, 0.0);
  }
  let ln_abs = abs_z.ln();
  let arg_z = b.atan2(a);
  let re_exp = c * ln_abs - d * arg_z;
  let im_exp = d * ln_abs + c * arg_z;
  let mag = re_exp.exp();
  (mag * im_exp.cos(), mag * im_exp.sin())
}

/// Numerically evaluate an exact algebraic expression to a complex `(re, im)`.
/// Extends `try_extract_complex_float` with a `Power` rule (principal branch),
/// so radical roots such as `-(-1)^(1/3)` — which Solve returns as
/// `Times[-1, Power[-1, 1/3]]` — fully numericize instead of leaking a
/// symbolic `Power` into NSolve's output.
fn eval_complex_full(expr: &Expr) -> Option<(f64, f64)> {
  // Reuse the existing extractor for everything but Power.
  if let Some(v) = crate::functions::math_ast::try_extract_complex_float(expr) {
    return Some(v);
  }
  let pow_parts = |base: &Expr, exp: &Expr| -> Option<(f64, f64)> {
    let (a, b) = eval_complex_full(base)?;
    let (c, d) = eval_complex_full(exp)?;
    Some(complex_pow(a, b, c, d))
  };
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      pow_parts(&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => pow_parts(left, right),
    // Re-handle products/sums/negation here too, since a Power factor would
    // have made try_extract_complex_float bail on the whole expression.
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let mut res = eval_complex_full(&args[0])?;
      for arg in &args[1..] {
        let (c, d) = eval_complex_full(arg)?;
        res = (res.0 * c - res.1 * d, res.0 * d + res.1 * c);
      }
      Some(res)
    }
    Expr::FunctionCall { name, args } if name == "Plus" && !args.is_empty() => {
      let mut res = (0.0, 0.0);
      for arg in args.iter() {
        let (c, d) = eval_complex_full(arg)?;
        res = (res.0 + c, res.1 + d);
      }
      Some(res)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (a, b) = eval_complex_full(left)?;
      let (c, d) = eval_complex_full(right)?;
      Some((a * c - b * d, a * d + b * c))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let (a, b) = eval_complex_full(left)?;
      let (c, d) = eval_complex_full(right)?;
      Some((a + c, b + d))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let (a, b) = eval_complex_full(left)?;
      let (c, d) = eval_complex_full(right)?;
      Some((a - c, b - d))
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (a, b) = eval_complex_full(operand)?;
      Some((-a, -b))
    }
    _ => None,
  }
}

/// Recursively convert a Solve result to numerical form.
/// Handles nested lists and rules, converting replacement values to floats.
fn nsolve_numerize(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> =
        items.iter().map(nsolve_numerize).collect();
      Ok(Expr::List(results?.into()))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(nsolve_numerize(replacement)?),
    }),
    _ => {
      // Try pure real first
      if let Some(v) = crate::functions::math_ast::try_eval_to_f64(expr) {
        return Ok(Expr::Real(v));
      }
      // Try complex (handles I, -I, a + b*I, and radical Power roots like
      // -(-1)^(1/3) that Solve returns as Times[-1, Power[-1, 1/3]]).
      if let Some((re, im)) = eval_complex_full(expr) {
        if im == 0.0 {
          return Ok(Expr::Real(re));
        }
        return Ok(
          crate::evaluator::evaluate_function_call_ast(
            "Complex",
            &[Expr::Real(re), Expr::Real(im)],
          )
          .unwrap_or_else(|_| expr.clone()),
        );
      }
      // Fall back to N[]
      crate::functions::math_ast::n_eval(expr)
    }
  }
}

// ─── Solve ──────────────────────────────────────────────────────────

/// Roots[equation, var] — find roots of a polynomial equation.
///
/// Returns solutions as `x == val1 || x == val2 || ...`
pub fn roots_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Roots expects exactly 2 arguments".into(),
    ));
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Roots".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Use Solve to find solutions
  let solutions = solve_ast(args)?;

  // Convert {{var -> val1}, {var -> val2}, ...} to x == val1 || x == val2 || ...
  match &solutions {
    Expr::List(outer) => {
      let mut conditions: Vec<Expr> = Vec::new();
      for item in outer {
        if let Expr::List(inner) = item {
          if inner.is_empty() {
            // {{}} means all values (identity)
            return Ok(Expr::Identifier("True".to_string()));
          }
          for rule in inner {
            if let Expr::Rule { replacement, .. } = rule {
              conditions.push(Expr::Comparison {
                operands: vec![
                  Expr::Identifier(var.clone()),
                  *replacement.clone(),
                ],
                operators: vec![ComparisonOp::Equal],
              });
            }
          }
        }
      }
      // Roots lists the solutions in Solve's order (ascending by value), which
      // matches wolframscript for general polynomials. The one exception is a
      // pure quadratic x^2 == c (the two roots sum to zero, e.g. ±3 or ±I):
      // wolframscript lists the principal `+` root first (3 || -3, I || -I,
      // Sqrt[2] || -Sqrt[2]), so reverse Solve's `-r, +r` ordering there.
      if conditions.len() == 2
        && let (
          Expr::Comparison { operands: o1, .. },
          Expr::Comparison { operands: o2, .. },
        ) = (&conditions[0], &conditions[1])
      {
        let sum = crate::evaluator::evaluate_function_call_ast(
          "Plus",
          &[o1[1].clone(), o2[1].clone()],
        );
        if matches!(sum, Ok(Expr::Integer(0))) {
          conditions.reverse();
        }
      }

      if conditions.is_empty() {
        Ok(Expr::Identifier("False".to_string()))
      } else if conditions.len() == 1 {
        Ok(conditions.into_iter().next().unwrap())
      } else {
        Ok(Expr::FunctionCall {
          name: "Or".to_string(),
          args: conditions.into(),
        })
      }
    }
    // Solve returned unevaluated
    _ => Ok(Expr::FunctionCall {
      name: "Roots".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// NRoots[equation, var] — numerical roots of a polynomial equation.
///
/// Returns `x == r1 || x == r2 || ...` with all roots (real and complex)
/// computed numerically via Durand-Kerner iteration on the companion form.
pub fn nroots_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NRoots expects exactly 2 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NRoots".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Extract `lhs - rhs`
  let poly = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(fargs[0].clone()),
        right: Box::new(fargs[1].clone()),
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NRoots".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let unevaluated = || Expr::FunctionCall {
    name: "NRoots".to_string(),
    args: args.to_vec().into(),
  };

  // Expand and pull out polynomial coefficients in x.
  let expanded_raw = expand_and_combine(&poly);
  let expanded = {
    let together = together_expr(&expanded_raw);
    match &together {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: numerator,
        right: _,
      } => expand_and_combine(numerator),
      _ => expanded_raw,
    }
  };
  let terms = collect_additive_terms(&expanded);
  let degree = match max_power_int(&expanded, &var) {
    Some(d) if d >= 1 => d as usize,
    _ => return Ok(unevaluated()),
  };

  // Numeric f64 coefficients (index = degree).
  let mut coeffs = vec![0.0f64; degree + 1];
  for d in 0..=degree {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, &var, d as i128) {
        let val = crate::functions::math_ast::try_eval_to_f64(&simplify(c));
        match val {
          Some(v) => coeffs[d] += v,
          None => return Ok(unevaluated()),
        }
      }
    }
  }
  if coeffs[degree].abs() < 1e-300 {
    return Ok(unevaluated());
  }

  // Durand-Kerner finds all roots simultaneously.
  let roots = durand_kerner_roots(&coeffs);

  // Sort by (Re, Im) ascending.
  let mut roots = roots;
  roots.sort_by(|a, b| {
    a.0
      .partial_cmp(&b.0)
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
  });

  // Build x == root_i expressions.
  let mut conds: Vec<Expr> = Vec::with_capacity(roots.len());
  for (re, im) in roots {
    let rhs = if im == 0.0 {
      Expr::Real(re)
    } else {
      crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
    };
    conds.push(Expr::Comparison {
      operands: vec![Expr::Identifier(var.clone()), rhs],
      operators: vec![ComparisonOp::Equal],
    });
  }
  if conds.len() == 1 {
    return Ok(conds.into_iter().next().unwrap());
  }
  Ok(Expr::FunctionCall {
    name: "Or".to_string(),
    args: conds.into(),
  })
}

/// Find all roots of polynomial (coeffs[i] is coefficient of x^i) using
/// the Durand-Kerner (Weierstrass) method on complex doubles.
fn durand_kerner_roots(coeffs: &[f64]) -> Vec<(f64, f64)> {
  let n = coeffs.len() - 1;
  if n == 0 {
    return vec![];
  }
  if n == 1 {
    // a0 + a1*x = 0 → x = -a0/a1
    return vec![(-coeffs[0] / coeffs[1], 0.0)];
  }

  // Monic form: divide by leading coefficient.
  let lc = coeffs[n];
  let mut monic = vec![0.0f64; n + 1];
  for i in 0..=n {
    monic[i] = coeffs[i] / lc;
  }
  // Evaluate p(z) for complex z using Horner's scheme with FMA.
  // The FMA-based form is more accurate near a root and helps the polish
  // step distinguish the correctly-rounded f64 from its neighbour.
  let eval_p = |zr: f64, zi: f64| -> (f64, f64) {
    let mut re = monic[n];
    let mut im = 0.0;
    for k in (0..n).rev() {
      // (re + im*i) * (zr + zi*i) + monic[k]
      // Real part: re*zr - im*zi + monic[k]
      // Imag part: re*zi + im*zr
      let neg_im: f64 = -im;
      let nr = re.mul_add(zr, neg_im.mul_add(zi, monic[k]));
      let ni = re.mul_add(zi, im * zr);
      re = nr;
      im = ni;
    }
    (re, im)
  };

  // Evaluate derivative p'(z) for complex z.
  let eval_dp = |zr: f64, zi: f64| -> (f64, f64) {
    if n == 0 {
      return (0.0, 0.0);
    }
    let mut re = (n as f64) * monic[n];
    let mut im = 0.0;
    for k in (1..n).rev() {
      let nr = re * zr - im * zi + (k as f64) * monic[k];
      let ni = re * zi + im * zr;
      re = nr;
      im = ni;
    }
    (re, im)
  };

  // Initialise n distinct roots on a circle.
  // Use complex base 0.4 + 0.9i as in classic Durand-Kerner.
  let base_r = 0.4_f64;
  let base_i = 0.9_f64;
  let mut roots: Vec<(f64, f64)> = (0..n)
    .map(|k| {
      // (base_r + base_i*i)^k via repeated multiplication
      let mut r = 1.0;
      let mut i = 0.0;
      for _ in 0..k {
        let nr = r * base_r - i * base_i;
        let ni = r * base_i + i * base_r;
        r = nr;
        i = ni;
      }
      (r, i)
    })
    .collect();

  // Iterate.
  for _ in 0..2000 {
    let prev = roots.clone();
    let mut max_delta: f64 = 0.0;
    for k in 0..n {
      let (zr, zi) = prev[k];
      let (mut dr, mut di) = (1.0, 0.0);
      for (j, &(jr, ji)) in prev.iter().enumerate() {
        if j == k {
          continue;
        }
        // (dr + di*i) * (zr - jr + (zi - ji)*i)
        let ar = zr - jr;
        let ai = zi - ji;
        let nr = dr * ar - di * ai;
        let ni = dr * ai + di * ar;
        dr = nr;
        di = ni;
      }
      // p(z) / Π
      let (pr, pi) = eval_p(zr, zi);
      let denom = dr * dr + di * di;
      if denom == 0.0 {
        continue;
      }
      let qr = (pr * dr + pi * di) / denom;
      let qi = (pi * dr - pr * di) / denom;
      let new_r = zr - qr;
      let new_i = zi - qi;
      let delta = (new_r - zr).hypot(new_i - zi);
      if delta > max_delta {
        max_delta = delta;
      }
      roots[k] = (new_r, new_i);
    }
    let scale = 1.0_f64
      + roots
        .iter()
        .map(|&(r, i)| r.hypot(i))
        .fold(0.0_f64, f64::max);
    if max_delta < 1e-15 * scale {
      break;
    }
  }

  // Polish each root with Newton steps. Track the iterate with the smallest
  // |p(z)| since near the root Newton may oscillate between two adjacent
  // f64 values (both equally good). Pick the best one.
  for k in 0..roots.len() {
    let (mut zr, mut zi) = roots[k];
    let (mut best_zr, mut best_zi) = (zr, zi);
    let (pr0, pi0) = eval_p(zr, zi);
    let mut best_err = pr0.hypot(pi0);
    for _ in 0..30 {
      let (pr, pi) = eval_p(zr, zi);
      let (dr, di) = eval_dp(zr, zi);
      let denom = dr * dr + di * di;
      if denom < 1e-300 {
        break;
      }
      let qr = (pr * dr + pi * di) / denom;
      let qi = (pi * dr - pr * di) / denom;
      let new_r = zr - qr;
      let new_i = zi - qi;
      zr = new_r;
      zi = new_i;
      let (npr, npi) = eval_p(zr, zi);
      let err = npr.hypot(npi);
      if err < best_err {
        best_err = err;
        best_zr = zr;
        best_zi = zi;
      }
      if err == 0.0 {
        break;
      }
    }
    roots[k] = (best_zr, best_zi);
  }

  // Snap to real / round trailing fp noise.
  for (r, i) in roots.iter_mut() {
    let mag = r.hypot(*i).max(1.0);
    if i.abs() < 1e-12 * mag {
      *i = 0.0;
    }
    if r.abs() < 1e-12 * mag {
      *r = 0.0;
    }
  }
  // Match conjugate pairs: when two roots have the same real part (within
  // tolerance) and opposite signs of imag part, make their imag exactly +/-.
  let snap_eps = 1e-9;
  for k in 0..roots.len() {
    for j in (k + 1)..roots.len() {
      if (roots[k].0 - roots[j].0).abs() < snap_eps
        && (roots[k].1 + roots[j].1).abs() < snap_eps
      {
        let avg = ((roots[k].0) + (roots[j].0)) / 2.0;
        let mag = (roots[k].1.abs() + roots[j].1.abs()) / 2.0;
        roots[k].0 = avg;
        roots[j].0 = avg;
        if roots[k].1 < 0.0 {
          roots[k].1 = -mag;
          roots[j].1 = mag;
        } else {
          roots[k].1 = mag;
          roots[j].1 = -mag;
        }
      }
    }
  }
  roots
}

/// ToRules[eqns] — converts logical combinations of equations to lists of rules.
/// Takes output from Roots/Reduce (Or/And of equations) and converts to Solve-style rules.
/// Discards inequalities (!=).
pub fn to_rules_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ToRules expects exactly 1 argument".into(),
    ));
  }

  fn eq_to_rule(expr: &Expr) -> Option<Expr> {
    // Convert x == val to {x -> val}
    if let Expr::Comparison {
      operands,
      operators,
    } = expr
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal
      && operands.len() == 2
    {
      return Some(Expr::Rule {
        pattern: Box::new(operands[0].clone()),
        replacement: Box::new(operands[1].clone()),
      });
    }
    None
  }

  fn collect_and_rules(expr: &Expr) -> Vec<Expr> {
    // Collect all rules from And (conjunction) of equations
    match expr {
      Expr::FunctionCall { name, args } if name == "And" => {
        let mut rules = Vec::new();
        for arg in args {
          if let Some(rule) = eq_to_rule(arg) {
            rules.push(rule);
          }
          // Discard non-equations (inequalities, etc.)
        }
        rules
      }
      // BinaryOp::And tree (used by reduce_multi_var_and)
      Expr::BinaryOp {
        op: BinaryOperator::And,
        left,
        right,
      } => {
        let mut rules = collect_and_rules(left);
        rules.extend(collect_and_rules(right));
        rules
      }
      _ => {
        if let Some(rule) = eq_to_rule(expr) {
          vec![rule]
        } else {
          vec![]
        }
      }
    }
  }

  fn collect_or_terms(expr: &Expr) -> Vec<Expr> {
    match expr {
      Expr::BinaryOp {
        op: BinaryOperator::Or,
        left,
        right,
      } => {
        let mut terms = collect_or_terms(left);
        terms.extend(collect_or_terms(right));
        terms
      }
      Expr::FunctionCall { name, args } if name == "Or" => {
        args.iter().flat_map(collect_or_terms).collect()
      }
      _ => vec![expr.clone()],
    }
  }

  let input = &args[0];
  match input {
    // Or[x == a, x == b, ...] → Sequence[{x -> a}, {x -> b}, ...]
    // Wolfram's ToRules returns a Sequence of rule-lists for Or input,
    // which displays as {x -> a}{x -> b} (elements joined without separator)
    Expr::FunctionCall { name, args } if name == "Or" => {
      let result: Vec<Expr> = args
        .iter()
        .map(|arg| Expr::List(collect_and_rules(arg).into()))
        .filter(|list| {
          if let Expr::List(items) = list {
            !items.is_empty()
          } else {
            false
          }
        })
        .collect();
      Ok(Expr::FunctionCall {
        name: "Sequence".to_string(),
        args: result.into(),
      })
    }
    // BinaryOp::Or tree (used by reduce_multi_var_and for multiple solutions)
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      ..
    } => {
      let or_terms = collect_or_terms(input);
      let result: Vec<Expr> = or_terms
        .iter()
        .map(|arg| Expr::List(collect_and_rules(arg).into()))
        .filter(|list| {
          if let Expr::List(items) = list {
            !items.is_empty()
          } else {
            false
          }
        })
        .collect();
      Ok(Expr::FunctionCall {
        name: "Sequence".to_string(),
        args: result.into(),
      })
    }
    // And[x == a, y == b] → {x -> a, y -> b}
    Expr::FunctionCall { name, .. } if name == "And" => {
      let rules = collect_and_rules(input);
      Ok(Expr::List(rules.into()))
    }
    // BinaryOp::And tree (used by reduce_multi_var_and for single solution)
    Expr::BinaryOp {
      op: BinaryOperator::And,
      ..
    } => {
      let rules = collect_and_rules(input);
      Ok(Expr::List(rules.into()))
    }
    // Single equation: x == a → {x -> a}
    Expr::Comparison { .. } => {
      let rules = collect_and_rules(input);
      Ok(Expr::List(rules.into()))
    }
    // True → {} (trivially satisfied, no constraints)
    Expr::Identifier(s) if s == "True" => Ok(Expr::List(vec![].into())),
    // False → Sequence[] (no solutions, matches Wolfram: splices to nothing in context)
    Expr::Identifier(s) if s == "False" => Ok(Expr::FunctionCall {
      name: "Sequence".to_string(),
      args: vec![].into(),
    }),
    // Anything else: return unevaluated
    _ => Ok(Expr::FunctionCall {
      name: "ToRules".to_string(),
      args: vec![input.clone()].into(),
    }),
  }
}

/// Solve[equation, var] — solve a polynomial equation for a variable.
///
/// Supports linear (degree 1) and quadratic (degree 2) equations.
/// Also handles systems: Solve[{eq1, eq2, ...}, {x1, x2, ...}]
/// And inequality constraints: Solve[eq && ineq, var]
/// `SolveValues[eqn, var]` returns the values directly (not as rules).
/// It's `Solve[eqn, var]` flattened to just the right-hand sides.
pub fn solve_values_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let solutions = solve_ast(args)?;
  // `solve_ast` returns either a list of rule-lists like {{var->val}, {var->val2}}
  // or stays unevaluated. If it stayed symbolic, mirror that for SolveValues.
  let Expr::List(solution_sets) = &solutions else {
    return Ok(Expr::FunctionCall {
      name: "SolveValues".to_string(),
      args: args.to_vec().into(),
    });
  };

  // The rules inside Solve's output use the dedicated `Expr::Rule` variant
  // (not a generic `Rule[lhs, rhs]` function call), so we destructure it
  // directly. Any unrecognised branch causes us to fall back to symbolic.
  let mut values = Vec::with_capacity(solution_sets.len());
  for branch in solution_sets.iter() {
    let Expr::List(rules) = branch else {
      return Ok(Expr::FunctionCall {
        name: "SolveValues".to_string(),
        args: args.to_vec().into(),
      });
    };
    if rules.len() != 1 {
      // Multi-variable Solve: take the values in order so SolveValues mirrors
      // wolframscript's `{{val_x, val_y}, …}` output. But the actuarial
      // example uses only single-variable Solve, so fall back symbolically
      // if we see something more elaborate.
      return Ok(Expr::FunctionCall {
        name: "SolveValues".to_string(),
        args: args.to_vec().into(),
      });
    }
    let Expr::Rule { replacement, .. } = &rules[0] else {
      return Ok(Expr::FunctionCall {
        name: "SolveValues".to_string(),
        args: args.to_vec().into(),
      });
    };
    values.push((**replacement).clone());
  }
  Ok(Expr::List(values.into()))
}

/// Whether `s` names a built-in constant rather than a solve variable.
fn is_solve_constant(s: &str) -> bool {
  matches!(
    s,
    "Pi"
      | "E"
      | "I"
      | "Infinity"
      | "ComplexInfinity"
      | "Indeterminate"
      | "GoldenRatio"
      | "EulerGamma"
      | "Catalan"
      | "Degree"
      | "Glaisher"
      | "Khinchin"
      | "True"
      | "False"
      | "Null"
  )
}

/// Collect the free variable symbols of an equation (or list/And of
/// equations), in first-appearance order, descending through comparisons,
/// arithmetic and function arguments. Used by the one-argument Solve form.
fn collect_solve_vars(expr: &Expr, out: &mut Vec<String>) {
  match expr {
    Expr::Identifier(s) if !is_solve_constant(s) && !out.contains(s) => {
      out.push(s.clone());
    }
    Expr::Comparison { operands, .. } => {
      for e in operands {
        collect_solve_vars(e, out);
      }
    }
    Expr::List(items) => {
      for e in items {
        collect_solve_vars(e, out);
      }
    }
    Expr::FunctionCall { args, .. } => {
      for e in args {
        collect_solve_vars(e, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_solve_vars(left, out);
      collect_solve_vars(right, out);
    }
    Expr::UnaryOp { operand, .. } => collect_solve_vars(operand, out),
    _ => {}
  }
}

pub fn solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // One-argument form Solve[eqns]: auto-detect the variables and delegate to
  // the two-argument form. Only the unambiguous cases are handled — a single
  // variable, or a determined/overdetermined system (variables <= equations).
  // An underdetermined system (which wolframscript solves with a non-obvious
  // variable-selection heuristic) is left unevaluated rather than guessed.
  if args.len() == 1 {
    // A trivially true/false condition (e.g. Solve[x == x] after x == x
    // evaluated to True) needs no variables: True -> {{}}, False -> {}.
    if let Expr::Identifier(s) = &args[0] {
      if s == "True" {
        return Ok(Expr::List(vec![Expr::List(vec![].into())].into()));
      }
      if s == "False" {
        return Ok(Expr::List(vec![].into()));
      }
    }
    let mut vars = Vec::new();
    collect_solve_vars(&args[0], &mut vars);
    let n_eqns = match &args[0] {
      Expr::List(items) => items.len(),
      _ => 1,
    };
    let var_arg = if vars.len() == 1 {
      Some(Expr::Identifier(vars.remove(0)))
    } else if vars.len() >= 2 && vars.len() <= n_eqns {
      Some(Expr::List(vars.into_iter().map(Expr::Identifier).collect()))
    } else {
      None
    };
    return match var_arg {
      Some(va) => solve_ast(&[args[0].clone(), va]),
      None => Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec().into(),
      }),
    };
  }
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Solve expects 2 or 3 arguments".into(),
    ));
  }

  // Pre-pass: generalized variables. Solve[eqns, f[x]] (or a variable list
  // containing such applications) treats the whole application `f[x]` as the
  // unknown, matching wolframscript. Replace every function-application target
  // with a fresh bare symbol, solve, then map the fresh symbols back to the
  // original applications in the result. Only triggers when a target is a
  // FunctionCall, so ordinary bare-symbol solves are unaffected.
  {
    let targets: Vec<Expr> = match &args[1] {
      Expr::List(items) => items.to_vec(),
      other => vec![other.clone()],
    };
    if targets
      .iter()
      .any(|t| matches!(t, Expr::FunctionCall { .. }))
    {
      let ctx =
        format!("{}{}", expr_to_string(&args[0]), expr_to_string(&args[1]));
      let mut subs: Vec<(Expr, Expr)> = Vec::new();
      let mut fresh_targets: Vec<Expr> = Vec::new();
      for (i, t) in targets.iter().enumerate() {
        if matches!(t, Expr::FunctionCall { .. }) {
          let mut k = i;
          let mut name = format!("WoxiSolveVar{k}");
          while ctx.contains(&name) {
            k += 1000;
            name = format!("WoxiSolveVar{k}");
          }
          let sym = Expr::Identifier(name);
          subs.push((t.clone(), sym.clone()));
          fresh_targets.push(sym);
        } else {
          fresh_targets.push(t.clone());
        }
      }
      let mut new_eqns = args[0].clone();
      for (from, to) in &subs {
        new_eqns = substitute_expr(&new_eqns, from, to);
      }
      let new_var_arg = match &args[1] {
        Expr::List(_) => Expr::List(fresh_targets.into()),
        _ => fresh_targets.into_iter().next().unwrap(),
      };
      let mut new_args = vec![new_eqns, new_var_arg];
      if args.len() == 3 {
        new_args.push(args[2].clone());
      }
      let result = solve_ast(&new_args)?;
      // Map the fresh symbols back to the original applications.
      let mut mapped = result;
      for (from, to) in &subs {
        mapped = substitute_expr(&mapped, to, from);
      }
      return Ok(mapped);
    }
  }

  // Pre-pass: turn an And-of-equations into a List of equations so the
  // multi-equation path picks them up. Wolfram lets users write
  // Solve[a == b && c == d, ...] interchangeably with the list form.
  // && parses to a BinaryOp::And tree (left-associative), so flatten
  // the chain. The FunctionCall("And", …) variant covers the other
  // path through the parser.
  fn flatten_and(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
      Expr::BinaryOp {
        op: BinaryOperator::And,
        left,
        right,
      } => {
        flatten_and(left, out);
        flatten_and(right, out);
      }
      Expr::FunctionCall { name, args: aargs } if name == "And" => {
        for a in aargs.iter() {
          flatten_and(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  // Only flatten when every conjunct is an equality (Comparison with
  // Equal). Inequalities stay inside the And so the existing Reduce
  // path (which understands constraints) handles cases like
  // `m^2 == 4 && m > 0`.
  fn all_equalities(items: &[Expr]) -> bool {
    items.iter().all(|e| {
      matches!(
        e,
        Expr::Comparison { operators, .. }
          if operators.iter().all(|o| matches!(o, ComparisonOp::Equal))
      ) || matches!(
        e,
        Expr::FunctionCall { name, args }
          if name == "Equal" && args.len() == 2
      )
    })
  }
  let args_owned: Vec<Expr>;
  let args = match &args[0] {
    Expr::FunctionCall { name, .. } if name == "And" => {
      let mut conjuncts = Vec::new();
      flatten_and(&args[0], &mut conjuncts);
      if all_equalities(&conjuncts) {
        let mut new_args = args.to_vec();
        new_args[0] = Expr::List(conjuncts.into());
        args_owned = new_args;
        args_owned.as_slice()
      } else {
        args
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::And,
      ..
    } => {
      let mut conjuncts = Vec::new();
      flatten_and(&args[0], &mut conjuncts);
      if all_equalities(&conjuncts) {
        let mut new_args = args.to_vec();
        new_args[0] = Expr::List(conjuncts.into());
        args_owned = new_args;
        args_owned.as_slice()
      } else {
        args
      }
    }
    _ => args,
  };

  // Pre-pass: normalize a bare-symbol variable to a single-element list when
  // the first argument is a list of equations. Solve[{x == 1, x == 2}, x]
  // behaves like Solve[{x == 1, x == 2}, {x}], which the system path handles;
  // without this it fell through to the single-equation path and wrongly
  // emitted Solve::naqs.
  let barevar_args_owned: Vec<Expr>;
  let args = if matches!(&args[0], Expr::List(_))
    && matches!(&args[1], Expr::Identifier(_))
  {
    let mut new_args = args.to_vec();
    new_args[1] = Expr::List(vec![args[1].clone()].into());
    barevar_args_owned = new_args;
    barevar_args_owned.as_slice()
  } else {
    args
  };

  // Pre-pass: drop variables from the var list that don't appear in any
  // equation. Wolfram emits Solve::svars and continues with the
  // remaining variables. Without this, an extra var like `y` in
  // `Solve[x^2 == 1 && z^2 == -1, {x, y, z}]` would block the solver.
  // We can't reuse calculus_ast::is_constant_wrt because it doesn't
  // recurse into Comparison nodes — the equations here are
  // x^2 == 1 etc.
  fn expr_uses_var(expr: &Expr, var: &str) -> bool {
    match expr {
      Expr::Identifier(s) => s == var,
      Expr::List(items) => items.iter().any(|e| expr_uses_var(e, var)),
      Expr::BinaryOp { left, right, .. } => {
        expr_uses_var(left, var) || expr_uses_var(right, var)
      }
      Expr::UnaryOp { operand, .. } => expr_uses_var(operand, var),
      Expr::Comparison { operands, .. } => {
        operands.iter().any(|e| expr_uses_var(e, var))
      }
      Expr::FunctionCall { args, .. } => {
        args.iter().any(|e| expr_uses_var(e, var))
      }
      Expr::CurriedCall { func, args } => {
        expr_uses_var(func, var) || args.iter().any(|e| expr_uses_var(e, var))
      }
      _ => false,
    }
  }
  let svars_args_owned: Vec<Expr>;
  let args = if let (Expr::List(eqs), Expr::List(vars)) = (&args[0], &args[1]) {
    let used: Vec<usize> = vars
      .iter()
      .enumerate()
      .filter_map(|(i, v)| {
        if let Expr::Identifier(name) = v
          && eqs.iter().any(|e| expr_uses_var(e, name))
        {
          Some(i)
        } else if !matches!(v, Expr::Identifier(_)) {
          // Non-identifier var: keep as-is (existing handling will deal).
          Some(i)
        } else {
          None
        }
      })
      .collect();
    if used.len() < vars.len() {
      crate::emit_message(
        "Solve::svars: Equations may not give solutions for all \"solve\" variables.",
      );
      let kept: Vec<Expr> = used.into_iter().map(|i| vars[i].clone()).collect();
      let mut new_args = args.to_vec();
      new_args[1] = Expr::List(kept.into());
      svars_args_owned = new_args;
      svars_args_owned.as_slice()
    } else {
      args
    }
  } else {
    args
  };

  // Parse domain from optional 3rd argument (Reals, Integers, Complexes, etc.)
  let domain = if args.len() == 3 {
    match &args[2] {
      Expr::Identifier(s) => Some(s.clone()),
      _ => None,
    }
  } else {
    None
  };

  // If domain is specified, solve without domain first, then filter
  if let Some(ref dom) = domain {
    let base_solutions = solve_ast(&args[..2])?;
    if dom == "Reals" {
      // Filter out complex solutions
      if let Expr::List(solutions) = &base_solutions {
        let filtered: Vec<Expr> = solutions
          .iter()
          .filter(|sol| !contains_complex(sol))
          .cloned()
          .collect();
        return Ok(Expr::List(filtered.into()));
      }
    }
    if dom == "Integers" {
      // Bounded linear systems: enumerate integer solutions directly.
      // Handles cases like Solve[{15n+17m==200, n>=0, m>=0}, {n,m}, Integers]
      // where filtering real solutions wouldn't terminate or would lose
      // discrete answers buried in a parametric form.
      if let Some(result) = try_solve_integer_bounded(&args[0], &args[1]) {
        return Ok(result);
      }
      // Filter to only integer solutions
      if let Expr::List(solutions) = &base_solutions {
        let filtered: Vec<Expr> = solutions
          .iter()
          .filter(|sol| {
            if let Expr::List(rules) = sol {
              rules.iter().all(|rule| {
                if let Expr::Rule { replacement, .. } = rule {
                  is_integer_expr(replacement)
                } else {
                  false
                }
              })
            } else {
              false
            }
          })
          .cloned()
          .collect();
        return Ok(Expr::List(filtered.into()));
      }
    }
    // For other domains, just return the base solutions
    return Ok(base_solutions);
  }

  // Handle system of equations: Solve[{eq1,...}, {var1,...}]
  if let (Expr::List(eqs_raw), Expr::List(vars_exprs)) = (&args[0], &args[1]) {
    // Flatten any And conjunctions inside the list, so that
    // Solve[{a == b && c == d}, {x, y}] behaves like Solve[{a == b, c == d}, {x, y}].
    let eqs: Vec<Expr> = flatten_and_constraints(eqs_raw);
    let eqs = &eqs;
    let var_names: Vec<String> = vars_exprs
      .iter()
      .filter_map(|v| {
        if let Expr::Identifier(name) = v {
          Some(name.clone())
        } else {
          None
        }
      })
      .collect();
    if var_names.len() == vars_exprs.len() && !var_names.is_empty() {
      // Try symbolic Gaussian elimination for linear systems (handles underdetermined case)
      if let Some(result) = solve_linear_symbolic(eqs, &var_names) {
        return Ok(result);
      }
      // High-degree coupled systems (e.g. a quintic in x plus a quintic
      // in y with x-dependent coefficients) blow up in the
      // Reduce-based path because Woxi has no multivariate Root form.
      // Detect them and bail out unevaluated rather than hang.
      if eqs.len() >= 2
        && var_names.len() >= 2
        && eqs.iter().any(|e| {
          var_names
            .iter()
            .any(|v| max_degree_of_var(e, v).unwrap_or(0) >= 3)
        })
      {
        return Ok(Expr::FunctionCall {
          name: "Solve".to_string(),
          args: args.to_vec().into(),
        });
      }
      // Fall back to Reduce's multi-variable elimination for nonlinear systems
      let constraints: Vec<Expr> = eqs.clone();
      let reduce_result =
        crate::functions::polynomial_ast::reduce::reduce_multi_var_and(
          &constraints,
          &var_names,
          None,
        )?;
      // to_rules_ast returns Sequence for Or (multi-solution) or List for single solution
      // Solve always wraps into {{...}, {...}, ...} format
      let rules = to_rules_ast(&[reduce_result])?;
      let mut wrapped = match &rules {
        // Sequence of rule-lists → wrap in outer List
        Expr::FunctionCall {
          name,
          args: seq_args,
        } if name == "Sequence" => Expr::List(seq_args.clone()),
        // Single solution as flat rules → wrap in double list
        Expr::List(items)
          if items.iter().all(|i| matches!(i, Expr::Rule { .. })) =>
        {
          Expr::List(vec![rules].into())
        }
        _ => rules,
      };
      // Sort rules within each solution to match variable order, and sort solutions
      if let Expr::List(ref mut solutions) = wrapped {
        for sol in solutions.iter_mut() {
          if let Expr::List(rules) = sol {
            rules.sort_by_key(|rule| {
              if let Expr::Rule { pattern, .. } = rule {
                if let Expr::Identifier(name) = pattern.as_ref() {
                  var_names
                    .iter()
                    .position(|v| v == name)
                    .unwrap_or(usize::MAX)
                } else {
                  usize::MAX
                }
              } else {
                usize::MAX
              }
            });
          }
        }
        // Sort solutions: real solutions first, then complex
        solutions.sort_by_key(|sol| if contains_complex(sol) { 1 } else { 0 });
      }
      return Ok(wrapped);
    }
  }

  // Handle single equation with list of variables: Solve[eq, {var1, var2, ...}]
  if let Expr::List(vars_exprs) = &args[1] {
    if vars_exprs.len() == 1 {
      return solve_ast(&[args[0].clone(), vars_exprs[0].clone()]);
    }
    // Multiple variables with a single equation: solve for the variable
    // with the lowest degree (matching Wolfram's behavior).
    if !matches!(&args[0], Expr::List(_)) && vars_exprs.len() > 1 {
      // Determine degree of each variable in the equation
      let eq_expr = &args[0];
      let (lhs, rhs) = if let Some((l, r, _)) =
        crate::functions::polynomial_ast::reduce::extract_comparison(eq_expr)
      {
        (l, r)
      } else {
        (eq_expr.clone(), Expr::Integer(0))
      };
      let poly = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(lhs),
        right: Box::new(rhs),
      };
      let expanded =
        crate::functions::polynomial_ast::expand_and_combine(&poly);

      // Sort variables by degree (ascending), keeping original order for ties
      let mut var_degrees: Vec<(usize, i128)> = vars_exprs
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| {
          if let Expr::Identifier(name) = v {
            crate::functions::polynomial_ast::max_power_int(&expanded, name)
              .map(|deg| (idx, deg))
          } else {
            None
          }
        })
        .collect();
      var_degrees.sort_by_key(|&(idx, deg)| (deg, idx));

      // Try solving for each variable in degree order
      for (idx, _deg) in &var_degrees {
        let var_expr = &vars_exprs[*idx];
        let result = solve_ast(&[args[0].clone(), var_expr.clone()])?;
        if let Expr::List(ref solutions) = result
          && !solutions.is_empty()
        {
          return Ok(result);
        }
        // If solve returned unevaluated, try next variable
        if !matches!(&result, Expr::FunctionCall { name, .. } if name == "Solve")
        {
          return Ok(result);
        }
      }
      // None succeeded — return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec().into(),
      });
    }
  }

  // Handle equation + inequality: Solve[eq && ineq, var]
  // Extract the equation and inequality parts from an And expression
  if let Expr::Identifier(var_name) = &args[1] {
    let var_name = var_name.clone();
    let (eq_part_opt, ineqs) = extract_eq_and_ineq_parts(&args[0]);
    if let Some(eq_part) = eq_part_opt
      && !ineqs.is_empty()
    {
      // Solve the equation part, then filter by inequalities. A periodic
      // solution `var -> ConditionalExpression[a + b C, C ∈ Integers]` is
      // specialized to the concrete values satisfying the bounds.
      let eq_solutions = solve_ast(&[eq_part, args[1].clone()])?;
      if let Expr::List(solutions) = &eq_solutions {
        let mut out: Vec<Expr> = Vec::new();
        let mut seen: std::collections::HashSet<String> =
          std::collections::HashSet::new();
        let mut specialized = false;
        for sol in solutions.iter() {
          // A single-rule solution `{var -> ConditionalExpression[...]}` may be
          // specialized into several concrete rules.
          if let Expr::List(rules) = sol
            && rules.len() == 1
            && let Expr::Rule { replacement, .. } = &rules[0]
            && let Some(concrete) =
              specialize_periodic_solution(&var_name, replacement, &ineqs)
          {
            specialized = true;
            for c in concrete {
              let key = crate::syntax::expr_to_string(&c);
              if seen.insert(key) {
                out.push(c);
              }
            }
            continue;
          }
          // Otherwise keep the solution unless an inequality is definitely
          // violated.
          let ineq_false = |ineq: &Expr, replacement: &Expr| -> bool {
            let subst =
              crate::syntax::substitute_variable(ineq, &var_name, replacement);
            matches!(
              crate::evaluator::evaluate_expr_to_expr(&subst),
              Ok(Expr::Identifier(ref s)) if s == "False"
            )
          };
          let violated = matches!(sol, Expr::List(rules) if rules.iter().any(|rule| {
            matches!(rule, Expr::Rule { replacement, .. }
              if ineqs.iter().any(|ineq| ineq_false(ineq, replacement)))
          }));
          if !violated {
            let key = crate::syntax::expr_to_string(sol);
            if seen.insert(key) {
              out.push(sol.clone());
            }
          }
        }
        // Specialized periodic solutions are returned in ascending value order,
        // matching wolframscript.
        if specialized {
          let key = |sol: &Expr| -> f64 {
            if let Expr::List(rules) = sol
              && let Some(Expr::Rule { replacement, .. }) = rules.first()
            {
              crate::functions::math_ast::try_eval_to_f64(replacement)
                .unwrap_or(f64::INFINITY)
            } else {
              f64::INFINITY
            }
          };
          out.sort_by(|a, b| {
            key(a)
              .partial_cmp(&key(b))
              .unwrap_or(std::cmp::Ordering::Equal)
          });
        }
        return Ok(Expr::List(out.into()));
      }
      return Ok(eq_solutions);
    }
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    // Constants (E, Pi, Degree) are not valid variables
    Expr::Constant(name) => {
      crate::emit_message(&format!(
        "Solve::ivar: {} is not a valid variable.",
        name
      ));
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec().into(),
      });
    }
    target_expr => {
      // Non-identifier solve target (e.g., f[x + y])
      // Handle Solve[target == value, target] → {{target -> value}}
      let (lhs, rhs, is_eq) = match &args[0] {
        Expr::Comparison {
          operands,
          operators,
        } if operands.len() == 2
          && operators.len() == 1
          && operators[0] == ComparisonOp::Equal =>
        {
          (operands[0].clone(), operands[1].clone(), true)
        }
        Expr::FunctionCall {
          name: fname,
          args: fargs,
        } if fname == "Equal" && fargs.len() == 2 => {
          (fargs[0].clone(), fargs[1].clone(), true)
        }
        _ => (Expr::Integer(0), Expr::Integer(0), false),
      };
      if is_eq {
        // Check if lhs matches target → solve for target
        let target_str = crate::syntax::expr_to_string(target_expr);
        let lhs_str = crate::syntax::expr_to_string(&lhs);
        let rhs_str = crate::syntax::expr_to_string(&rhs);
        if lhs_str == target_str {
          return Ok(Expr::List(
            vec![Expr::List(
              vec![Expr::Rule {
                pattern: Box::new(target_expr.clone()),
                replacement: Box::new(rhs),
              }]
              .into(),
            )]
            .into(),
          ));
        }
        if rhs_str == target_str {
          return Ok(Expr::List(
            vec![Expr::List(
              vec![Expr::Rule {
                pattern: Box::new(target_expr.clone()),
                replacement: Box::new(lhs),
              }]
              .into(),
            )]
            .into(),
          ));
        }
      }
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Check if variable has Constant attribute (user-defined constants)
  let is_constant = crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(var)
      .is_some_and(|attrs| attrs.contains(&"Constant".to_string()))
  });
  if is_constant {
    crate::emit_message(&format!(
      "Solve::ivar: {} is not a valid variable.",
      var
    ));
    return Ok(Expr::FunctionCall {
      name: "Solve".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Extract equation: lhs == rhs → lhs - rhs
  let poly = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      // lhs - rhs
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(fargs[0].clone()),
        right: Box::new(fargs[1].clone()),
      }
    }
    Expr::Identifier(s) if s == "True" => {
      // x == x → True → all solutions
      return Ok(Expr::List(vec![Expr::List(vec![].into())].into()));
    }
    Expr::Identifier(s) if s == "False" => {
      // contradiction → no solutions
      return Ok(Expr::List(vec![].into()));
    }
    _ => {
      // Solve::naqs: expr is not a quantified system of equations and inequalities.
      let expr_str = crate::syntax::expr_to_string(&args[0]);
      crate::emit_message(&format!(
        "Solve::naqs: {} is not a quantified system of equations and inequalities.",
        expr_str
      ));
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Absolute-value equations: Abs[f(x)] == c → f == c ∪ f == -c.
  if let Some(result) = try_solve_abs_eq(&args[0], var) {
    return result;
  }

  // Try to solve equations with invertible functions:
  // Log[expr] == a → expr == E^a, Sqrt[expr] == a → expr == a^2, etc.
  if let Some(result) = try_solve_inverse_function(&args[0], var) {
    return result;
  }

  // Trigonometric equations of the form `Trig[var] == const` where the trig
  // head is Sin/Cos/Tan/Cot. Wolframscript prints the periodic solution set
  // as `{{var -> ConditionalExpression[base + 2*Pi*C[1], Element[C[1],
  // Integers]]}, …}` for Sin/Cos and `{var -> ConditionalExpression[base +
  // Pi*C[1], Element[C[1], Integers]]}` for Tan/Cot.
  if let Some(result) = try_solve_trig_eq(&args[0], var) {
    return Ok(result);
  }

  // Expand and collect polynomial coefficients
  // Clear denominators: f(x)/g(x) == 0 ↔ f(x) == 0
  let expanded_raw = expand_and_combine(&poly);
  let expanded = {
    let together = together_expr(&expanded_raw);
    match &together {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: numerator,
        right: _denominator,
      } => expand_and_combine(numerator),
      _ => expanded_raw,
    }
  };
  // Factor out constant factors (w.r.t. the solve variable) so the
  // quadratic formula sees integer leading coefficients when possible.
  // E.g. 2*a^2*k*q - 4*k*q*x^2  →  a^2 - 2*x^2
  let expanded = factor_out_constant_factors(&expanded, var);
  let terms = collect_additive_terms(&expanded);

  // Find maximum degree
  let degree = match max_power_int(&expanded, var) {
    Some(d) => d,
    None => {
      // Non-polynomial: try factoring out common fractional-power sub-expressions
      if let Some(result) = try_solve_factoring_powers(&expanded, var, args) {
        return result;
      }
      return Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // A negative maximum power means a Laurent/rational expression (e.g. only
  // x^-2 terms). The coefficient extraction below builds an empty `0..=degree`
  // range, and the later `degree as usize` cast would index out of bounds, so
  // bail out with the unevaluated Solve.
  if degree < 0 {
    return Ok(Expr::FunctionCall {
      name: "Solve".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Extract coefficients for each power of var
  let mut coeffs: Vec<Expr> = Vec::new();
  for d in 0..=degree {
    let mut coeff_sum: Vec<Expr> = Vec::new();
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d) {
        coeff_sum.push(c);
      }
    }
    if coeff_sum.is_empty() {
      coeffs.push(Expr::Integer(0));
    } else if coeff_sum.len() == 1 {
      coeffs.push(simplify(coeff_sum.remove(0)));
    } else {
      let mut result = coeff_sum.remove(0);
      for c in coeff_sum {
        result = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(result),
          right: Box::new(c),
        };
      }
      coeffs.push(simplify(result));
    }
  }

  let make_rule = |solution: Expr| -> Expr {
    // A raw `UnaryOp[Minus, …]` negation wrapper prints its operand in
    // isolation (`-2^(-1/2)`); evaluating it collapses to the canonical
    // `Times[-1, …]` form that wolframscript shows (`-(1/Sqrt[2])`). Only the
    // negation wrapper needs this — already-canonical roots are left as built.
    let solution = if matches!(solution, Expr::UnaryOp { .. }) {
      crate::evaluator::evaluate_expr_to_expr(&solution)
        .unwrap_or_else(|_| solution.clone())
    } else {
      solution
    };
    Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::Identifier(var.to_string())),
        replacement: Box::new(solution),
      }]
      .into(),
    )
  };

  // Factor out x^k when the k lowest-degree coefficients are zero.
  // E.g., a - a^3/6 = 0  → coeffs = [0, 1, 0, -1/6]
  // → x=0 is a root, reduced polynomial: 1 - a^2/6 = 0
  if degree > 1 && matches!(&coeffs[0], Expr::Integer(0)) {
    let zero_count = coeffs
      .iter()
      .take_while(|c| matches!(c, Expr::Integer(0)))
      .count();
    if zero_count > 0 && (zero_count as i128) < degree {
      let reduced_eq = build_eq_from_coeffs(&coeffs[zero_count..], var);
      let reduced_solutions = solve_ast(&[reduced_eq, args[1].clone()])?;
      if let Expr::List(ref reduced_sols) = reduced_solutions {
        let x_zero = make_rule(Expr::Integer(0));
        let mut all_solutions = vec![x_zero];
        all_solutions.extend(reduced_sols.iter().cloned());
        sort_solutions(&mut all_solutions);
        return Ok(Expr::List(all_solutions.into()));
      }
    }
  }

  match degree {
    0 => {
      // No variable present — check if constant is zero
      let c0 = &coeffs[0];
      if matches!(c0, Expr::Integer(0)) {
        Ok(Expr::List(vec![Expr::List(vec![].into())].into()))
      } else {
        Ok(Expr::List(vec![].into()))
      }
    }
    1 => {
      // Linear: a*x + b = 0  → x = -b/a
      let b = &coeffs[0]; // constant term
      let a = &coeffs[1]; // coefficient of x
      let neg_b = negate_expr(b);
      let solution = simplify(solve_divide(&neg_b, a));
      // Run the user-level Simplify so forms like -((1 - z)/2) collapse
      // to (-1 + z)/2, matching wolframscript's canonical output.
      let solution =
        crate::functions::polynomial_ast::simplify_ast(&[solution.clone()])
          .unwrap_or(solution);
      Ok(Expr::List(vec![make_rule(solution)].into()))
    }
    2 => {
      // Quadratic: a*x^2 + b*x + c = 0
      let c = &coeffs[0]; // constant term
      let b = &coeffs[1]; // coefficient of x
      let a = &coeffs[2]; // coefficient of x^2

      // Discriminant: b^2 - 4*a*c
      let b_sq = multiply_exprs(b, b);
      let four_ac = multiply_exprs(&Expr::Integer(4), &multiply_exprs(a, c));
      let discriminant = simplify(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(b_sq),
        right: Box::new(four_ac),
      });

      let neg_b = negate_expr(b);
      let two_a = multiply_exprs(&Expr::Integer(2), a);

      // For integer coefficients, use exact arithmetic with simplified Sqrt
      if let (Expr::Integer(ai), Expr::Integer(bi), Expr::Integer(ci)) =
        (a, b, c)
      {
        let ai = *ai;
        let bi = *bi;
        let ci = *ci;
        let disc_int = bi * bi - 4 * ai * ci;

        if disc_int >= 0 {
          let (sqrt_out, sqrt_in) = simplify_sqrt_parts(disc_int);
          // roots = (-bi ± sqrt_out * Sqrt[sqrt_in]) / (2*ai)
          if sqrt_in == 1 {
            // Perfect square discriminant: exact integer/rational roots.
            let sol1 = solve_divide(
              &Expr::Integer(-bi - sqrt_out),
              &Expr::Integer(2 * ai),
            );
            let sol2 = solve_divide(
              &Expr::Integer(-bi + sqrt_out),
              &Expr::Integer(2 * ai),
            );
            // Dividing by 2a flips the root order when a < 0, so emit the
            // smaller (more negative) root first to match Wolfram.
            return Ok(if ai < 0 {
              Expr::List(vec![make_rule(sol2), make_rule(sol1)].into())
            } else {
              Expr::List(vec![make_rule(sol1), make_rule(sol2)].into())
            });
          } else {
            // Irrational roots: (-bi ± sqrt_out*Sqrt[sqrt_in]) / (2*ai)
            // Simplify by dividing common factors
            let g =
              gcd_i128(gcd_i128(-bi, sqrt_out).abs(), (2 * ai).abs()).abs();
            let nb = -bi / g;
            let so = sqrt_out / g;
            let den = 2 * ai / g;
            // Normalize the denominator to be positive. Only the numerator's
            // additive term (`nb`) and the denominator flip sign; the radical
            // coefficient `so` must stay non-negative. (Negating `so` here made
            // `sqrt_part` negative, so the minus root came out as the
            // unsimplified `-(-Sqrt[..])` instead of `Sqrt[..]`.) Because both
            // ± roots are emitted, keeping `so > 0` with `den > 0` still yields
            // the smaller (more negative) root from `make_sol(true)`, matching
            // Wolfram's negative-root-first ordering.
            let (nb, so, den) = if den < 0 {
              (-nb, so, -den)
            } else {
              (nb, so, den)
            };
            let sqrt_part = if so == 1 {
              make_sqrt(Expr::Integer(sqrt_in))
            } else {
              multiply_exprs(
                &Expr::Integer(so),
                &make_sqrt(Expr::Integer(sqrt_in)),
              )
            };
            let make_sol = |sign_minus: bool| -> Expr {
              // Special case: when nb == 0 and so == 1, absorb denominator into Sqrt
              // E.g. Sqrt[6]/2 → Sqrt[3/2] to match Wolfram's canonical form
              if nb == 0 && den != 1 && so == 1 {
                let rational_arg =
                  crate::functions::math_ast::make_rational(sqrt_in, den * den);
                if let Ok(simplified) =
                  crate::functions::math_ast::sqrt_ast(&[rational_arg])
                {
                  return if sign_minus {
                    negate_expr(&simplified)
                  } else {
                    simplified
                  };
                }
              }
              let num = if nb == 0 {
                if sign_minus {
                  negate_expr(&sqrt_part)
                } else {
                  sqrt_part.clone()
                }
              } else {
                let nb_expr = Expr::Integer(nb);
                Expr::BinaryOp {
                  op: if sign_minus {
                    BinaryOperator::Minus
                  } else {
                    BinaryOperator::Plus
                  },
                  left: Box::new(nb_expr),
                  right: Box::new(sqrt_part.clone()),
                }
              };
              if den == 1 {
                num
              } else {
                Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(num),
                  right: Box::new(Expr::Integer(den)),
                }
              }
            };
            let sol1 = make_sol(true);
            let sol2 = make_sol(false);
            return Ok(Expr::List(
              vec![make_rule(sol1), make_rule(sol2)].into(),
            ));
          }
        } else {
          // Check for cyclotomic polynomials before using quadratic formula
          // x^2 + x + 1 = 0 (Φ₃): roots are (-1)^(2/3) and -(-1)^(1/3)
          // x^2 - x + 1 = 0 (Φ₆): roots are (-1)^(1/3) and -(-1)^(2/3)
          // Multiplying the polynomial through by -1 doesn't change the
          // root set, so accept `(ai, ci) ∈ {(1, 1), (-1, -1)}` and pick
          // the cyclotomic branch by `Sign[bi*ai]` rather than `bi` alone.
          let cyclo_match = (ai == 1 && ci == 1) || (ai == -1 && ci == -1);
          if cyclo_match && bi.abs() == ai.abs() {
            let make_neg1_pow = |p: i128, q: i128| -> Expr {
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![
                  Expr::Integer(-1),
                  crate::functions::math_ast::make_rational(p, q),
                ]
                .into(),
              }
            };
            // After multiplying by -1, the b/a sign flips along with a's
            // sign — so `bi*ai > 0` corresponds to Φ₃ (`x^2 + x + 1`) and
            // `bi*ai < 0` to Φ₆ (`x^2 - x + 1`).
            if bi * ai > 0 {
              // Φ₃: x^2 + x + 1 → roots: -(-1)^(1/3), (-1)^(2/3)
              let sol1 = negate_expr(&make_neg1_pow(1, 3));
              let sol2 = make_neg1_pow(2, 3);
              return Ok(Expr::List(
                vec![make_rule(sol1), make_rule(sol2)].into(),
              ));
            } else {
              // Φ₆: x^2 - x + 1 → roots: (-1)^(1/3), -(-1)^(2/3)
              let sol1 = make_neg1_pow(1, 3);
              let sol2 = negate_expr(&make_neg1_pow(2, 3));
              return Ok(Expr::List(
                vec![make_rule(sol1), make_rule(sol2)].into(),
              ));
            }
          }

          // Complex roots: (-bi ± I*Sqrt[-disc]) / (2*ai)
          let neg_disc = -disc_int;
          let (sqrt_out, sqrt_in) = simplify_sqrt_parts(neg_disc);
          if sqrt_in == 1 {
            // Gaussian integer/rational roots
            let real_part =
              solve_divide(&Expr::Integer(-bi), &Expr::Integer(2 * ai));
            let imag_part =
              solve_divide(&Expr::Integer(sqrt_out), &Expr::Integer(2 * ai));
            let make_sol = |sign_minus: bool| -> Expr {
              let i_part =
                multiply_exprs(&Expr::Identifier("I".to_string()), &imag_part);
              simplify(Expr::BinaryOp {
                op: if sign_minus {
                  BinaryOperator::Minus
                } else {
                  BinaryOperator::Plus
                },
                left: Box::new(real_part.clone()),
                right: Box::new(i_part),
              })
            };
            let sol1 = make_sol(true);
            let sol2 = make_sol(false);
            return Ok(Expr::List(
              vec![make_rule(sol1), make_rule(sol2)].into(),
            ));
          } else {
            // Complex roots with irrational imaginary part
            let g =
              gcd_i128(gcd_i128(-bi, sqrt_out).abs(), (2 * ai).abs()).abs();
            let nb = -bi / g;
            let so = sqrt_out / g;
            let den = 2 * ai / g;
            // Keep the radical coefficient `so` non-negative; only `nb` and
            // `den` flip when the denominator is negative (see the real-root
            // branch above — negating `so` produced an unsimplified
            // `-(I*(-Sqrt[..]))`).
            let (nb, so, den) = if den < 0 {
              (-nb, so, -den)
            } else {
              (nb, so, den)
            };
            let sqrt_part = multiply_exprs(
              &Expr::Identifier("I".to_string()),
              &if so == 1 {
                make_sqrt(Expr::Integer(sqrt_in))
              } else {
                multiply_exprs(
                  &Expr::Integer(so),
                  &make_sqrt(Expr::Integer(sqrt_in)),
                )
              },
            );
            let make_sol = |sign_minus: bool| -> Expr {
              let num = if nb == 0 {
                if sign_minus {
                  negate_expr(&sqrt_part)
                } else {
                  sqrt_part.clone()
                }
              } else {
                Expr::BinaryOp {
                  op: if sign_minus {
                    BinaryOperator::Minus
                  } else {
                    BinaryOperator::Plus
                  },
                  left: Box::new(Expr::Integer(nb)),
                  right: Box::new(sqrt_part.clone()),
                }
              };
              if den == 1 {
                num
              } else {
                Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(num),
                  right: Box::new(Expr::Integer(den)),
                }
              }
            };
            // Re-evaluate so a raw negation collapses (e.g. -(I*Sqrt[2]) →
            // -I*Sqrt[2]), matching wolframscript's complex-root form.
            let finish = |e: Expr| {
              crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
            };
            let sol1 = finish(make_sol(true));
            let sol2 = finish(make_sol(false));
            return Ok(Expr::List(
              vec![make_rule(sol1), make_rule(sol2)].into(),
            ));
          }
        }
      }

      // Non-integer coefficients: use general symbolic formula

      // Special case: when b=0 and a is integer, solutions are x = ±Sqrt[c_expr] / Sqrt[|a_int|]
      // This produces cleaner output matching Wolfram (e.g., a/Sqrt[2] instead of 1/2*Sqrt[2]*a)
      if matches!(b, Expr::Integer(0))
        && let Expr::Integer(a_int) = a
      {
        // x^2 = -c/a, and we want to present as Sqrt[c_expr] / Sqrt[neg_a]
        // For a<0: x = ±Sqrt[c] / Sqrt[-a]
        // For a>0: x = ±Sqrt[-c] / Sqrt[a]  (requires -c >= 0 somehow)
        let (numer_under_sqrt, denom_under_sqrt) = if *a_int < 0 {
          (c.clone(), Expr::Integer(-a_int))
        } else {
          (negate_expr(c), Expr::Integer(*a_int))
        };
        let sqrt_numer = {
          let raw = crate::functions::sqrt_ast(&[numer_under_sqrt.clone()])
            .unwrap_or_else(|_| make_sqrt(numer_under_sqrt));
          let evaled =
            crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw);
          // In Solve context, Sqrt[expr^2] → expr because ± handles sign
          let evaled = strip_sqrt_square(evaled);
          simplify(evaled)
        };
        let sqrt_denom =
          crate::functions::sqrt_ast(&[denom_under_sqrt.clone()])
            .unwrap_or_else(|_| make_sqrt(denom_under_sqrt));
        let sol_pos = if matches!(&sqrt_denom, Expr::Integer(1)) {
          sqrt_numer.clone()
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(sqrt_numer.clone()),
            right: Box::new(sqrt_denom),
          }
        };
        let sol_neg = negate_expr(&sol_pos);
        return Ok(Expr::List(
          vec![make_rule(sol_neg), make_rule(sol_pos)].into(),
        ));
      }

      // First evaluate the discriminant to simplify complex arithmetic (e.g., (3+I)^2 - 4*(2+2I) → -2I)
      let disc_eval = crate::evaluator::evaluate_expr_to_expr(&discriminant)
        .unwrap_or(discriminant.clone());
      // Try to evaluate Sqrt of the discriminant symbolically
      let sqrt_disc_raw = crate::functions::sqrt_ast(&[disc_eval])
        .unwrap_or_else(|_| make_sqrt(discriminant.clone()));
      let sqrt_disc = crate::evaluator::evaluate_expr_to_expr(&sqrt_disc_raw)
        .unwrap_or(sqrt_disc_raw);
      // Evaluate numerators first so complex arithmetic simplifies before dividing
      let eval_expr = |e: Expr| -> Expr {
        let evaled =
          crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e.clone());
        // Re-evaluate if a further reduction is possible
        let evaled2 = crate::evaluator::evaluate_expr_to_expr(&evaled)
          .unwrap_or(evaled.clone());
        simplify(evaled2)
      };
      let num1 = eval_expr(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(neg_b.clone()),
        right: Box::new(sqrt_disc.clone()),
      });
      let sol1 = eval_expr(solve_divide(&num1, &two_a));
      let num2 = eval_expr(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(neg_b),
        right: Box::new(sqrt_disc),
      });
      let sol2 = eval_expr(solve_divide(&num2, &two_a));
      // Wolfram convention: negative root first. When the leading coefficient
      // is negative, dividing by 2a flips the root order, so swap.
      let leading_negative = match a {
        Expr::Integer(n) => *n < 0,
        Expr::FunctionCall { name, args: ta }
          if name == "Times" && !ta.is_empty() =>
        {
          matches!(&ta[0], Expr::Integer(n) if *n < 0)
        }
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left,
          ..
        } => matches!(left.as_ref(), Expr::Integer(n) if *n < 0),
        _ => false,
      };
      if leading_negative {
        Ok(Expr::List(vec![make_rule(sol2), make_rule(sol1)].into()))
      } else {
        Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)].into()))
      }
    }
    _ => {
      // Pure power equation: a*x^n + c = 0 (all middle coefficients zero)
      // Solve as x = (-c/a)^(1/n) * root_of_unity for each nth root of unity
      let is_pure_power =
        (1..degree as usize).all(|i| matches!(&coeffs[i], Expr::Integer(0)));

      if is_pure_power {
        let c_coeff = &coeffs[0];
        let a_coeff = &coeffs[degree as usize];
        let neg_c = negate_expr(c_coeff);
        let val = simplify(solve_divide(&neg_c, a_coeff));
        let val = crate::evaluator::evaluate_expr_to_expr(&val).unwrap_or(val);

        // Only use nth-root approach for symbolic values;
        // integer values are handled better by the factoring path below
        if !matches!(&val, Expr::Integer(_))
          && !matches!(&val, Expr::FunctionCall { name, .. } if name == "Rational")
        {
          let n = degree;
          let mut roots = Vec::new();

          // Build val^(1/n)
          let val_root = {
            let raw = Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![
                val.clone(),
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(n)].into(),
                },
              ]
              .into(),
            };
            crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw)
          };

          // Generate n roots ordered by fractional exponent j/n
          // For odd n: j=0 → positive, j=1 → negative, j=2 → positive, ...
          // For even n: pairs of (negative, positive) per distinct frac
          if n % 2 == 1 {
            // Odd n
            for j in 0..n {
              let root = if j == 0 {
                val_root.clone()
              } else {
                let g = super::factor::gcd_i128(j, n);
                let p = j / g;
                let q = n / g;
                let multiplier = Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![
                    Expr::Integer(-1),
                    Expr::FunctionCall {
                      name: "Rational".to_string(),
                      args: vec![Expr::Integer(p), Expr::Integer(q)].into(),
                    },
                  ]
                  .into(),
                };
                let product = Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(multiplier),
                  right: Box::new(val_root.clone()),
                };
                if j % 2 == 1 {
                  // Negative: -((-1)^(j/n) * val^(1/n))
                  negate_expr(&product)
                } else {
                  // Positive: (-1)^(j/n) * val^(1/n)
                  product
                }
              };
              roots.push(make_rule(root));
            }
          } else {
            // Even n: pairs (negative, positive) for each fractional exponent
            let half_n = n / 2;
            for j in 0..half_n {
              let frac_num = 2 * j;
              let frac_den = n;
              let g = super::factor::gcd_i128(frac_num, frac_den);

              if frac_num == 0 {
                // frac = 0: roots are -val^(1/n) and val^(1/n)
                roots.push(make_rule(negate_expr(&val_root)));
                roots.push(make_rule(val_root.clone()));
              } else {
                let p = frac_num / g;
                let q = frac_den / g;
                let multiplier = if p == 1 && q == 2 {
                  Expr::Identifier("I".to_string())
                } else {
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![
                      Expr::Integer(-1),
                      Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(p), Expr::Integer(q)].into(),
                      },
                    ]
                    .into(),
                  }
                };
                let product = Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(multiplier),
                  right: Box::new(val_root.clone()),
                };
                roots.push(make_rule(negate_expr(&product)));
                roots.push(make_rule(product));
              }
            }
          }

          sort_solutions(&mut roots);
          return Ok(Expr::List(roots.into()));
        }
      }

      // Higher degree: try Factor-based solving
      if let Ok(factored) =
        crate::functions::polynomial_ast::factor_ast(&[expanded.clone()])
      {
        let factors = extract_times_factors(&factored);
        if factors.len() > 1 {
          // Solve each factor separately
          let mut all_solutions: Vec<Expr> = Vec::new();
          for factor in &factors {
            if is_constant_wrt(factor, var) {
              continue; // Skip constant factors
            }
            let factor_eq = Expr::Comparison {
              operands: vec![factor.clone(), Expr::Integer(0)],
              operators: vec![ComparisonOp::Equal],
            };
            if let Ok(Expr::List(ref sols)) =
              solve_ast(&[factor_eq, args[1].clone()])
            {
              all_solutions.extend(sols.iter().cloned());
            }
          }
          if !all_solutions.is_empty() {
            sort_solutions(&mut all_solutions);
            return Ok(Expr::List(all_solutions.into()));
          }
        }
      }
      // Last resort for irreducible polynomials of degree ≥ 3 with
      // integer/rational coefficients: emit the wolframscript-style
      // list of Root expressions (`Root[poly &, k, 0]` for k = 1..deg).
      if let Some(rs) = make_root_solutions(&coeffs, var) {
        return Ok(rs);
      }
      Ok(Expr::FunctionCall {
        name: "Solve".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// Highest power of `var` in `eq` (treated as `lhs - rhs`). Returns
/// `None` for non-polynomial equations or when `var` does not appear.
fn max_degree_of_var(eq: &Expr, var: &str) -> Option<i128> {
  let lhs_minus_rhs = match eq {
    Expr::Comparison { operands, .. } if operands.len() == 2 => {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(args[0].clone()),
        right: Box::new(args[1].clone()),
      }
    }
    other => other.clone(),
  };
  let expanded =
    crate::evaluator::evaluate_expr_to_expr(&lhs_minus_rhs).ok()?;
  crate::functions::polynomial_ast::max_power_int(&expanded, var)
}

/// Build `{{var -> Root[poly &, 1, 0]}, …, {var -> Root[poly &, deg, 0]}}`
/// for a polynomial whose coefficients are exact integers or rationals.
/// Returns None for non-rational coefficients (so the caller falls back
/// to leaving the call unevaluated).
fn make_root_solutions(coeffs: &[Expr], var: &str) -> Option<Expr> {
  if coeffs.len() < 4 {
    return None;
  }
  // Require every coefficient to be an exact rational (integer or
  // Rational[]). Floats here would mean numerical roots are expected
  // instead — Root[…] only represents algebraic roots.
  let is_rational = |c: &Expr| -> bool {
    match c {
      Expr::Integer(_) => true,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        matches!(args[0], Expr::Integer(_))
          && matches!(args[1], Expr::Integer(_))
      }
      _ => false,
    }
  };
  if !coeffs.iter().all(is_rational) {
    return None;
  }
  let degree = coeffs.len() - 1;
  // Build polynomial body in Slot[1], ascending in powers, skipping
  // zero coefficients. The result is what wolframscript prints inside
  // Root, e.g. `1 + 2*#1 + #1^5`.
  let slot = Expr::Slot(1);
  let mut terms: Vec<Expr> = Vec::new();
  for (i, c) in coeffs.iter().enumerate() {
    if matches!(c, Expr::Integer(0)) {
      continue;
    }
    let var_pow = match i {
      0 => None,
      1 => Some(slot.clone()),
      _ => Some(Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![slot.clone(), Expr::Integer(i as i128)].into(),
      }),
    };
    let term = match (var_pow, c) {
      (None, c) => c.clone(),
      (Some(p), Expr::Integer(1)) => p,
      (Some(p), c) => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![c.clone(), p].into(),
      },
    };
    terms.push(term);
  }
  let body = match terms.len() {
    0 => return None,
    1 => terms.remove(0),
    _ => Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    },
  };
  let body = crate::evaluator::evaluate_expr_to_expr(&body).ok()?;
  let func = Expr::Function {
    body: Box::new(body),
  };
  let mut solutions = Vec::with_capacity(degree);
  for k in 1..=degree {
    let root = Expr::FunctionCall {
      name: "Root".to_string(),
      args: vec![func.clone(), Expr::Integer(k as i128), Expr::Integer(0)]
        .into(),
    };
    solutions.push(Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::Identifier(var.to_string())),
        replacement: Box::new(root),
      }]
      .into(),
    ));
  }
  Some(Expr::List(solutions.into()))
}

/// Sort a list of Solve solutions (each is `{var -> val}`) by root value.
/// Uses `solve_order` so complex roots interleave with reals by real
/// part, matching wolframscript's `Solve[x^5 == x, x]` output.
fn sort_solutions(solutions: &mut Vec<Expr>) {
  solutions.sort_by(|a, b| {
    let val_a = match a {
      Expr::List(rules) if !rules.is_empty() => match &rules[0] {
        Expr::Rule { replacement, .. } => replacement.as_ref(),
        _ => a,
      },
      _ => a,
    };
    let val_b = match b {
      Expr::List(rules) if !rules.is_empty() => match &rules[0] {
        Expr::Rule { replacement, .. } => replacement.as_ref(),
        _ => b,
      },
      _ => b,
    };
    solve_order(val_a, val_b)
  });
}

/// Check if an expression contains complex elements (I, (-1)^(p/q) with q>1, etc.)
fn contains_complex(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) if s == "I" => true,
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      // (-1)^(p/q) where q > 1 is complex
      if matches!(&args[0], Expr::Integer(n) if *n < 0)
        && let Expr::FunctionCall { name: rn, args: ra } = &args[1]
        && rn == "Rational"
        && ra.len() == 2
      {
        return true;
      }
      args.iter().any(contains_complex)
    }
    Expr::FunctionCall { args, .. } => args.iter().any(contains_complex),
    Expr::List(items) => items.iter().any(contains_complex),
    Expr::Rule { replacement, .. } => contains_complex(replacement),
    Expr::BinaryOp { left, right, .. } => {
      contains_complex(left) || contains_complex(right)
    }
    Expr::UnaryOp { operand, .. } => contains_complex(operand),
    _ => false,
  }
}

/// Extract multiplicative factors from a Times expression (FunctionCall or BinaryOp).
fn extract_times_factors(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut factors = extract_times_factors(left);
      factors.extend(extract_times_factors(right));
      factors
    }
    _ => vec![expr.clone()],
  }
}

/// Factor out multiplicative factors that are constant w.r.t. the solve variable.
/// For example: `2*a^2*k*q - 4*k*q*x^2` → `a^2 - 2*x^2` (factoring out `2*k*q`)
fn factor_out_constant_factors(expr: &Expr, var: &str) -> Expr {
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return expr.clone();
  }

  // For each term, extract (integer_coeff, non_integer_const_factors, var_factors)
  struct TermParts {
    int_coeff: i128,
    const_factors: Vec<String>, // non-integer constant factor strings
    all_factors: Vec<Expr>,     // all multiplicative factors
  }

  fn decompose_term(term: &Expr, var: &str) -> TermParts {
    let factors = collect_multiplicative_factors(term);
    let mut int_coeff: i128 = 1;
    let mut sign = 1i128;
    let mut expanded_factors: Vec<Expr> = Vec::new();

    // Flatten negation
    for f in &factors {
      match f {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } => {
          sign *= -1;
          let inner_factors = collect_multiplicative_factors(operand);
          expanded_factors.extend(inner_factors);
        }
        _ => expanded_factors.push(f.clone()),
      }
    }

    // Extract integer coefficients
    let mut remaining: Vec<Expr> = Vec::new();
    for f in &expanded_factors {
      if let Expr::Integer(n) = f {
        int_coeff *= n;
      } else {
        remaining.push(f.clone());
      }
    }
    int_coeff *= sign;

    let const_strs: Vec<String> = remaining
      .iter()
      .filter(|f| is_constant_wrt(f, var))
      .map(expr_to_string)
      .collect();

    TermParts {
      int_coeff,
      const_factors: const_strs,
      all_factors: remaining,
    }
  }

  let parts: Vec<TermParts> =
    terms.iter().map(|t| decompose_term(t, var)).collect();

  // Compute numeric GCD of all integer coefficients
  let num_gcd = parts
    .iter()
    .map(|p| p.int_coeff)
    .filter(|&n| n != 0)
    .fold(0i128, gcd_i128)
    .abs();

  // Find symbolic constant factors common to ALL terms
  let mut common_symbolic: Vec<String> = Vec::new();
  if !parts.is_empty() && !parts[0].const_factors.is_empty() {
    for candidate in &parts[0].const_factors {
      if parts[1..]
        .iter()
        .all(|p| p.const_factors.iter().any(|s| s == candidate))
      {
        common_symbolic.push(candidate.clone());
      }
    }
  }

  if num_gcd <= 1 && common_symbolic.is_empty() {
    return expr.clone();
  }

  // Rebuild terms with common factors removed
  let mut new_terms: Vec<Expr> = Vec::new();
  for part in &parts {
    let new_coeff = if num_gcd > 1 {
      part.int_coeff / num_gcd
    } else {
      part.int_coeff
    };

    let mut remaining: Vec<Expr> = Vec::new();
    let mut used_common: Vec<bool> = vec![false; common_symbolic.len()];

    for f in &part.all_factors {
      if is_constant_wrt(f, var) {
        // Check if this is a common symbolic factor
        let f_str = expr_to_string(f);
        let mut is_common = false;
        for (ci, cs) in common_symbolic.iter().enumerate() {
          if !used_common[ci] && f_str == *cs {
            used_common[ci] = true;
            is_common = true;
            break;
          }
        }
        if !is_common {
          remaining.push(f.clone());
        }
      } else {
        remaining.push(f.clone());
      }
    }

    // Build the term: new_coeff * remaining_factors
    let var_part = if remaining.is_empty() {
      None
    } else {
      Some(build_product(remaining))
    };

    let term = match (new_coeff, var_part) {
      (0, _) => Expr::Integer(0),
      (1, Some(v)) => v,
      (-1, Some(v)) => negate_term(&v),
      (c, Some(v)) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(c)),
        right: Box::new(v),
      },
      (c, None) => Expr::Integer(c),
    };
    new_terms.push(term);
  }

  expand_and_combine(&build_sum(new_terms))
}

/// Try to solve equations by applying inverse functions.
///
/// Handles: Log[expr] == a → expr == E^a,
///          Sqrt[expr] == a → expr == a^2,
///          Exp[expr] == a → expr == Log[a],
///          Sin/Cos/Tan/ArcSin/ArcCos/ArcTan[expr] == a → inverse function
/// Solve `Trig[var] == const` symbolically with the wolframscript
/// `ConditionalExpression[base + period*C[1], Element[C[1], Integers]]`
/// shape. Currently handles `Sin/Cos/Tan/Cot` against constants in
/// `{-1, 0, 1}` (the cases where the inverse trig has a closed-form
/// rational multiple of `Pi`). Returns `None` for everything else so the
/// caller falls through to the generic polynomial path.
/// True when `e` is a concrete real literal strictly greater than 1 (positive
/// integer ≥ 2, rational > 1, or real > 1). Used to decide whether `b^x == val`
/// gets the full 2*Pi*I/Log[b] periodic branches: for such a base Log[b] > 0 so
/// no sign canonicalization is needed to match wolframscript.
fn base_is_real_gt_one(e: &Expr) -> bool {
  match e {
    Expr::Integer(n) => *n > 1,
    Expr::Real(r) => *r > 1.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!((&args[0], &args[1]),
        (Expr::Integer(p), Expr::Integer(q)) if *q > 0 && *p > *q)
    }
    _ => false,
  }
}

/// Extract `(coeff, inner)` from `coeff * Abs[inner]` where `coeff` is free of
/// `var`. Returns `None` if the expression isn't a constant multiple of a
/// single `Abs[...]`.
fn extract_abs_factor(e: &Expr, var: &str) -> Option<(Expr, Expr)> {
  let factors: Vec<Expr> = match e {
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    _ => vec![e.clone()],
  };
  let mut inner: Option<Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Expr::FunctionCall { name, args } = f
      && name == "Abs"
      && args.len() == 1
    {
      if inner.is_some() {
        return None; // product of two Abs — not handled
      }
      inner = Some(args[0].clone());
    } else if is_constant_wrt(f, var) {
      coeff_factors.push(f.clone());
    } else {
      return None; // a non-constant factor that isn't the Abs
    }
  }
  let inner = inner?;
  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.pop().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, inner))
}

/// Solve `Abs[f(x)] == c` (optionally `k*Abs[f(x)] == c`). With `d = c/k`:
/// `d < 0` → no solution, `d == 0` → `f == 0`, otherwise `f == d` ∪ `f == -d`.
fn try_solve_abs_eq(
  eq: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  let (lhs, rhs) = match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      (operands[0].clone(), operands[1].clone())
    }
    _ => return None,
  };
  // Orient so the Abs (var-dependent) side is `abs_side`.
  let (abs_side, val_side) =
    if !is_constant_wrt(&lhs, var) && is_constant_wrt(&rhs, var) {
      (lhs, rhs)
    } else if !is_constant_wrt(&rhs, var) && is_constant_wrt(&lhs, var) {
      (rhs, lhs)
    } else {
      return None;
    };

  let (coeff, inner) = extract_abs_factor(&abs_side, var)?;
  let eff = if matches!(&coeff, Expr::Integer(1)) {
    val_side
  } else {
    crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(val_side),
      right: Box::new(coeff),
    })
    .ok()?
  };

  // Solve `inner == value`, returning the outer list's solution entries.
  let solve_branch = |value: Expr| -> Option<Vec<Expr>> {
    let branch_eq = Expr::Comparison {
      operands: vec![inner.clone(), value],
      operators: vec![ComparisonOp::Equal],
    };
    let r = solve_ast(&[branch_eq, Expr::Identifier(var.to_string())]).ok()?;
    match r {
      Expr::List(ref items) => Some(items.to_vec()),
      _ => None,
    }
  };

  let mut solutions: Vec<Expr> = Vec::new();
  match crate::functions::math_ast::try_eval_to_f64(&eff) {
    Some(v) if v < 0.0 => {} // no real solution → {}
    Some(v) if v == 0.0 => {
      solutions.extend(solve_branch(Expr::Integer(0))?);
    }
    _ => {
      // Positive numeric or symbolic value: both signs. The negative branch
      // is added first so the symbolic case keeps wolframscript's order
      // ({x -> -a} before {x -> a}); numeric cases are reordered by
      // sort_solutions anyway.
      let neg = crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(eff.clone()),
      })
      .ok()?;
      solutions.extend(solve_branch(neg)?);
      solutions.extend(solve_branch(eff)?);
    }
  }
  sort_solutions(&mut solutions);
  Some(Ok(Expr::List(solutions.into())))
}

fn try_solve_trig_eq(eq: &Expr, var: &str) -> Option<Expr> {
  // Extract lhs == rhs.
  let (lhs, rhs) = match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      (&operands[0], &operands[1])
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return None,
  };
  // lhs must be `Trig[var]` for some trig head.
  let (trig_name, trig_arg) = match lhs {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      (name.as_str(), &args[0])
    }
    _ => return None,
  };
  if !matches!(trig_name, "Sin" | "Cos" | "Tan" | "Cot") {
    return None;
  }
  // Inner argument has to be the bare solve variable.
  if !matches!(trig_arg, Expr::Identifier(s) if s == var) {
    return None;
  }
  // Constant rhs. The simplified special forms below apply to Sin/Cos at
  // {-1, 0, 1} and Tan/Cot at 0; every other numeric constant (including
  // Tan/Cot at ±1) uses the general inverse-trig family.
  let rhs_special = match (trig_name, rhs) {
    ("Sin" | "Cos", Expr::Integer(n)) if matches!(*n, -1..=1) => Some(*n),
    ("Tan" | "Cot", Expr::Integer(0)) => Some(0),
    _ => None,
  };
  if rhs_special.is_none() {
    // General case: rhs must be a numeric constant. A magnitude > 1 for
    // Sin/Cos is still solved symbolically via ArcSin/ArcCos (the inverse is
    // complex-valued), matching wolframscript's ConditionalExpression form —
    // e.g. Solve[Cos[x] == 2, x] -> ±ArcCos[2] + 2*Pi*C[1].
    let _c = crate::functions::math_ast::try_eval_to_f64(rhs)?;
  }

  let var_expr = Expr::Identifier(var.to_string());
  let pi = Expr::Constant("Pi".to_string());
  let c1 = Expr::FunctionCall {
    name: "C".to_string(),
    args: vec![Expr::Integer(1)].into(),
  };
  let element_c1_integers = Expr::FunctionCall {
    name: "Element".to_string(),
    args: vec![c1.clone(), Expr::Identifier("Integers".to_string())].into(),
  };
  let two_pi_c1 = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::Integer(2)),
    right: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(pi.clone()),
      right: Box::new(c1.clone()),
    }),
  };
  let pi_c1 = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(pi.clone()),
    right: Box::new(c1.clone()),
  };
  let neg_half_pi = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(crate::functions::math_ast::make_rational(-1, 2)),
    right: Box::new(pi.clone()),
  };
  let half_pi = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(pi.clone()),
    right: Box::new(Expr::Integer(2)),
  };
  // Helper: build `ConditionalExpression[expr, Element[C[1], Integers]]`.
  let cond = |body: Expr| Expr::FunctionCall {
    name: "ConditionalExpression".to_string(),
    args: vec![body, element_c1_integers.clone()].into(),
  };
  let make_rule_list = |bodies: Vec<Expr>| -> Expr {
    Expr::List(
      bodies
        .into_iter()
        .map(|body| {
          Expr::List(
            vec![Expr::Rule {
              pattern: Box::new(var_expr.clone()),
              replacement: Box::new(cond(body)),
            }]
            .into(),
          )
        })
        .collect(),
    )
  };

  // Build "base + 2*Pi*C[1]" / "base + Pi*C[1]" expressions.
  let plus = |a: Expr, b: Expr| Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(a),
    right: Box::new(b),
  };

  // Evaluate an expression (simplifies e.g. ArcSin[1/2] → Pi/6).
  let eval = |e: Expr| {
    crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or_else(|_| e.clone())
  };
  let inverse = |head: &str| {
    eval(Expr::FunctionCall {
      name: head.to_string(),
      args: vec![rhs.clone()].into(),
    })
  };
  let negate = |e: Expr| {
    eval(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(e),
    })
  };

  let solutions: Vec<Expr> = match (trig_name, rhs_special) {
    ("Sin", Some(0)) => {
      vec![two_pi_c1.clone(), plus(pi.clone(), two_pi_c1.clone())]
    }
    ("Sin", Some(1)) => vec![plus(half_pi.clone(), two_pi_c1.clone())],
    ("Sin", Some(-1)) => vec![plus(neg_half_pi.clone(), two_pi_c1.clone())],
    ("Cos", Some(0)) => vec![
      plus(neg_half_pi.clone(), two_pi_c1.clone()),
      plus(half_pi.clone(), two_pi_c1.clone()),
    ],
    ("Cos", Some(1)) => vec![two_pi_c1.clone()],
    ("Cos", Some(-1)) => {
      // Wolframscript returns the two-solution list `{x -> -Pi + 2*Pi*C[1],
      // x -> Pi + 2*Pi*C[1]}` even though they coincide modulo 2*Pi.
      let neg_pi = Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(pi.clone()),
      };
      vec![
        plus(neg_pi, two_pi_c1.clone()),
        plus(pi.clone(), two_pi_c1.clone()),
      ]
    }
    ("Tan", Some(0)) => vec![pi_c1.clone()],
    ("Cot", Some(0)) => vec![plus(half_pi.clone(), pi_c1.clone())],
    // General numeric rhs c: x = ArcSin[c] + 2πC, Pi - ArcSin[c] + 2πC for
    // Sin; -ArcCos[c] + 2πC, ArcCos[c] + 2πC for Cos; ArcTan[c] + πC for Tan.
    ("Sin", None) => {
      let a = inverse("ArcSin");
      let arcsin_sol = plus(a.clone(), two_pi_c1.clone());
      let pi_minus_sol =
        plus(eval(plus(pi.clone(), negate(a.clone()))), two_pi_c1.clone());
      // wolframscript orders the two branches by canonical form: when
      // `ArcSin[c]` stays symbolic (|c| is not a special value) it lists
      // `Pi - ArcSin[c]` first; when it simplifies to a concrete multiple of Pi
      // the pair is in ascending value order (`ArcSin[c]` first, since
      // `ArcSin[c] < Pi - ArcSin[c]` for every real c).
      if matches!(&a, Expr::FunctionCall { name, .. } if name == "ArcSin") {
        vec![pi_minus_sol, arcsin_sol]
      } else {
        vec![arcsin_sol, pi_minus_sol]
      }
    }
    ("Cos", None) => {
      let a = inverse("ArcCos");
      vec![
        plus(negate(a.clone()), two_pi_c1.clone()),
        plus(a, two_pi_c1.clone()),
      ]
    }
    ("Tan", None) => vec![plus(inverse("ArcTan"), pi_c1.clone())],
    _ => return None,
  };

  Some(make_rule_list(solutions))
}

fn try_solve_inverse_function(
  eq: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  // Extract lhs and rhs from the equation
  let (lhs, rhs) = match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      (operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      (args[0].clone(), args[1].clone())
    }
    _ => return None,
  };

  // Check if an expression is a function call or power (invertible form)
  let is_invertible_form = |e: &Expr| -> bool {
    matches!(
      e,
      Expr::FunctionCall { .. }
        | Expr::BinaryOp {
          op: BinaryOperator::Power,
          ..
        }
    )
  };

  // Try both orientations: f[expr] == val and val == f[expr]
  let (func_call, val) = if is_invertible_form(&lhs)
    && is_constant_wrt(&rhs, var)
    && !is_constant_wrt(&lhs, var)
  {
    (&lhs, &rhs)
  } else if is_invertible_form(&rhs)
    && is_constant_wrt(&lhs, var)
    && !is_constant_wrt(&rhs, var)
  {
    (&rhs, &lhs)
  } else {
    return None;
  };

  // Handle Power expressions: Power[base, exp] == val
  // Sqrt[x] is Power[x, 1/2], Exp[x] is Power[E, x]
  let power_parts = match func_call {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => Some((left.as_ref().clone(), right.as_ref().clone())),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  };
  if let Some((base, exp)) = power_parts {
    if is_constant_wrt(&exp, var) && !is_constant_wrt(&base, var) {
      // Skip if exponent is a positive integer — the polynomial solver
      // handles those and gives all roots (not just the principal root).
      if let Expr::Integer(n) = &exp
        && *n > 0
      {
        // Let polynomial solver handle x^n == a
        return None;
      }
      // base^exp == val where exp is constant (non-integer), base contains var
      // → base == val^(1/exp)
      let inverse_exp = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(exp),
      };
      let inverse_rhs = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(val.clone()),
        right: Box::new(inverse_exp),
      };
      let simplified_rhs =
        crate::evaluator::evaluate_expr_to_expr(&inverse_rhs).ok()?;
      let new_eq = Expr::Comparison {
        operands: vec![base, simplified_rhs],
        operators: vec![ComparisonOp::Equal],
      };
      return Some(solve_ast(&[new_eq, Expr::Identifier(var.to_string())]));
    }
    if !is_constant_wrt(&exp, var) && is_constant_wrt(&base, var) {
      // b^x == val (bare-var exponent, constant base): the full complex
      // solution has 2*Pi*I/Log[b] periodicity, which wolframscript reports as
      //   ConditionalExpression[Log[val]/Log[b] + (2*I*Pi*C[1])/Log[b], C ∈ Z]
      //   Solve[E^x == 5, x]  → Log[5] + 2*I*Pi*C[1]          (Log[E] = 1)
      //   Solve[2^x == 8, x]  → Log[8]/Log[2] + (2*I*Pi*C[1])/Log[2]
      // The periodic branches are only emitted for a concrete base E or > 1; a
      // symbolic base (or 0 < base < 1) falls through to the principal value,
      // matching wolframscript.
      let base_gets_period = matches!(&base, Expr::Constant(n) if n == "E")
        || base_is_real_gt_one(&base);
      if matches!(&exp, Expr::Identifier(n) if n == var) && base_gets_period {
        // Log[base] (evaluates to 1 for E).
        let log_base =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![base.clone()].into(),
          })
          .ok()?;
        // Principal part: Log[val] / Log[base].
        let principal =
          crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![val.clone()].into(),
            }),
            right: Box::new(log_base.clone()),
          })
          .ok()?;
        // Periodic part: (2*Pi*I*C[1]) / Log[base].
        let c1 = Expr::FunctionCall {
          name: "C".to_string(),
          args: vec![Expr::Integer(1)].into(),
        };
        let periodic =
          crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(2),
                Expr::Identifier("I".to_string()),
                Expr::Identifier("Pi".to_string()),
                c1.clone(),
              ]
              .into(),
            }),
            right: Box::new(log_base),
          })
          .ok()?;
        // Keep the periodic term first without re-canonicalizing the sum:
        // wolframscript lists `(2*I*Pi*C[1])/Log[b] + Log[val]/Log[b]` in that
        // order, whereas Woxi's Plus ordering would otherwise float the
        // principal Log term to the front. A zero principal (val == 1) drops
        // out entirely, matching `Solve[E^x == 1, x] -> 2*I*Pi*C[1]`.
        let principal_is_zero = matches!(&principal, Expr::Integer(0))
          || matches!(&principal, Expr::Real(r) if *r == 0.0);
        let general = if principal_is_zero {
          periodic
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![periodic, principal].into(),
          }
        };
        let cond = Expr::FunctionCall {
          name: "ConditionalExpression".to_string(),
          args: vec![
            general,
            Expr::FunctionCall {
              name: "Element".to_string(),
              args: vec![c1, Expr::Identifier("Integers".to_string())].into(),
            },
          ]
          .into(),
        };
        return Some(Ok(Expr::List(
          vec![Expr::List(
            vec![Expr::Rule {
              pattern: Box::new(Expr::Identifier(var.to_string())),
              replacement: Box::new(cond),
            }]
            .into(),
          )]
          .into(),
        )));
      }
      // base^exp == val where base is constant, exp contains var
      // → exp == Log[val] / Log[base]
      let inverse_rhs = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![val.clone()].into(),
        }),
        right: Box::new(Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![base].into(),
        }),
      };
      let simplified_rhs =
        crate::evaluator::evaluate_expr_to_expr(&inverse_rhs).ok()?;
      let new_eq = Expr::Comparison {
        operands: vec![exp, simplified_rhs],
        operators: vec![ComparisonOp::Equal],
      };
      return Some(solve_ast(&[new_eq, Expr::Identifier(var.to_string())]));
    }
  }

  if let Expr::FunctionCall { name, args } = func_call {
    if args.len() != 1 {
      return None;
    }
    let inner = &args[0];
    // Build the inverse equation: inner == inverse(val)
    let inverse_rhs = match name.as_str() {
      "Log" => {
        // Log[inner] == val → inner == E^val
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Constant("E".to_string())),
          right: Box::new(val.clone()),
        }
      }
      "Sqrt" => {
        // Sqrt[inner] == val → inner == val^2
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(val.clone()),
          right: Box::new(Expr::Integer(2)),
        }
      }
      "Exp" => {
        // Exp[inner] == val → inner == Log[val]
        Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![val.clone()].into(),
        }
      }
      "ArcSin" => {
        // ArcSin[inner] == val → inner == Sin[val]
        Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![val.clone()].into(),
        }
      }
      "ArcCos" => {
        // ArcCos[inner] == val → inner == Cos[val]
        Expr::FunctionCall {
          name: "Cos".to_string(),
          args: vec![val.clone()].into(),
        }
      }
      "ArcTan" => {
        // ArcTan[inner] == val → inner == Tan[val]
        Expr::FunctionCall {
          name: "Tan".to_string(),
          args: vec![val.clone()].into(),
        }
      }
      // ArcXxxDegrees[inner] == val → inner == Xxx[val Degree]. The
      // degree-flavoured arc functions invert to ordinary trig functions
      // applied to `val * Degree`.
      "ArcSinDegrees" | "ArcCosDegrees" | "ArcTanDegrees" | "ArcCotDegrees"
      | "ArcSecDegrees" | "ArcCscDegrees" => {
        let inverse_name = match name.as_str() {
          "ArcSinDegrees" => "Sin",
          "ArcCosDegrees" => "Cos",
          "ArcTanDegrees" => "Tan",
          "ArcCotDegrees" => "Cot",
          "ArcSecDegrees" => "Sec",
          "ArcCscDegrees" => "Csc",
          _ => unreachable!(),
        };
        let val_deg = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(val.clone()),
          right: Box::new(Expr::Constant("Degree".to_string())),
        };
        Expr::FunctionCall {
          name: inverse_name.to_string(),
          args: vec![val_deg].into(),
        }
      }
      "Log10" => {
        // Log10[inner] == val → inner == 10^val
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Integer(10)),
          right: Box::new(val.clone()),
        }
      }
      "Log2" => {
        // Log2[inner] == val → inner == 2^val
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(val.clone()),
        }
      }
      _ => return None,
    };

    // Simplify the inverse value
    let simplified_rhs =
      crate::evaluator::evaluate_expr_to_expr(&inverse_rhs).ok()?;

    // Build the new equation: inner == simplified_rhs
    let new_eq = Expr::Comparison {
      operands: vec![inner.clone(), simplified_rhs],
      operators: vec![ComparisonOp::Equal],
    };

    // Recursively solve the resulting equation
    Some(solve_ast(&[new_eq, Expr::Identifier(var.to_string())]))
  } else {
    None
  }
}

/// Try to solve a non-polynomial equation by factoring out common
/// sub-expressions with fractional exponents.
///
/// For example: `2*k*q*(a²+x²)^(3/2) - 6*k*q*x²*(a²+x²)^(1/2) == 0`
/// - Common base: `(a²+x²)`, min exponent: `1/2`
/// - After factoring out `(a²+x²)^(1/2)`: `2*k*q*(a²+x²) - 6*k*q*x²`
/// - Solve the remaining polynomial: `x = ±a/Sqrt[2]`
fn try_solve_factoring_powers(
  expanded: &Expr,
  var: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let terms = collect_additive_terms(expanded);
  if terms.is_empty() {
    return None;
  }

  // For each term, collect multiplicative factors and find factors of
  // the form base^(p/q) where base contains the solve variable.
  // We represent exponents as (numerator, denominator) rationals.
  struct PowerFactor {
    base_str: String,
    exp_num: i128, // exponent numerator
    exp_den: i128, // exponent denominator
  }

  fn extract_power_factors(term: &Expr, var: &str) -> Vec<PowerFactor> {
    let factors = collect_multiplicative_factors(term);
    let mut result = Vec::new();
    for f in &factors {
      // Handle Sqrt[expr] as expr^(1/2)
      if let Some(sqrt_arg) = is_sqrt(f)
        && !is_constant_wrt(sqrt_arg, var)
      {
        result.push(PowerFactor {
          base_str: expr_to_string(sqrt_arg),
          exp_num: 1,
          exp_den: 2,
        });
        continue;
      }
      let (base, exp) = extract_base_and_exp(f);
      if is_constant_wrt(&base, var) {
        continue;
      }
      let (num, den) = match &exp {
        Expr::Integer(n) => (*n, 1i128),
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
            (*n, *d)
          } else {
            continue;
          }
        }
        _ => continue,
      };
      result.push(PowerFactor {
        base_str: expr_to_string(&base),
        exp_num: num,
        exp_den: den,
      });
    }
    result
  }

  // Collect power factors for each term
  let all_power_factors: Vec<Vec<PowerFactor>> = terms
    .iter()
    .map(|t| extract_power_factors(t, var))
    .collect();

  // Find bases common to ALL terms
  if all_power_factors.iter().any(|pf| pf.is_empty()) {
    return None;
  }

  // Collect candidate base strings from the first term
  let candidate_bases: Vec<String> = all_power_factors[0]
    .iter()
    .map(|pf| pf.base_str.clone())
    .collect();

  for candidate_base in &candidate_bases {
    // Check if this base appears in ALL terms
    let mut min_exp: Option<(i128, i128)> = None;
    let mut all_have = true;
    for term_factors in &all_power_factors {
      let mut found = false;
      for pf in term_factors {
        if &pf.base_str == candidate_base {
          // Compute min exponent (as rational)
          let exp = (pf.exp_num, pf.exp_den);
          min_exp = Some(match min_exp {
            None => exp,
            Some((mn, md)) => {
              // Compare mn/md vs exp_num/exp_den
              if mn * exp.1 <= exp.0 * md {
                (mn, md)
              } else {
                exp
              }
            }
          });
          found = true;
          break;
        }
      }
      if !found {
        all_have = false;
        break;
      }
    }

    if !all_have || min_exp.is_none() {
      continue;
    }
    let (min_n, min_d) = min_exp.unwrap();
    if min_n == 0 {
      continue;
    }

    // Factor out base^(min_n/min_d) from each term
    let mut new_terms: Vec<Expr> = Vec::new();
    for (term, term_factors) in terms.iter().zip(all_power_factors.iter()) {
      // Find the matching factor and subtract exponent
      let mut remaining_factors: Vec<Expr> =
        collect_multiplicative_factors(term);
      let mut factored = false;

      // Helper: get the base string from a factor (handles Sqrt and Power)
      let factor_base_str = |f: &Expr| -> Option<String> {
        if let Some(sqrt_arg) = is_sqrt(f) {
          return Some(expr_to_string(sqrt_arg));
        }
        let (base, _) = extract_base_and_exp(f);
        if expr_to_string(&base) != expr_to_string(f)
          || matches!(
            f,
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              ..
            }
          )
        {
          Some(expr_to_string(&base))
        } else {
          // It's just a plain identifier (exponent = 1), which we also need
          if matches!(f, Expr::Identifier(_)) {
            Some(expr_to_string(f))
          } else {
            None
          }
        }
      };

      for idx in 0..remaining_factors.len() {
        let f = &remaining_factors[idx];
        if let Some(base_s) = factor_base_str(f)
          && base_s == *candidate_base
        {
          // Find exponent for this factor
          for pf in term_factors {
            if pf.base_str == *candidate_base {
              // New exponent = (pf.exp_num/pf.exp_den) - (min_n/min_d)
              let new_num = pf.exp_num * min_d - min_n * pf.exp_den;
              let new_den = pf.exp_den * min_d;
              let g = gcd_i128(new_num.abs(), new_den.abs());
              let new_num = new_num / g;
              let new_den = new_den / g;

              // Get the base expression
              let base_expr =
                if let Some(sqrt_arg) = is_sqrt(&remaining_factors[idx]) {
                  sqrt_arg.clone()
                } else {
                  extract_base_and_exp(&remaining_factors[idx]).0
                };

              if new_num == 0 {
                // Remove this factor entirely
                remaining_factors.remove(idx);
              } else {
                // Replace with base^(new_num/new_den)
                let new_exp = if new_den == 1 {
                  Expr::Integer(new_num)
                } else {
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(new_num), Expr::Integer(new_den)]
                      .into(),
                  }
                };
                remaining_factors[idx] = Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(base_expr),
                  right: Box::new(new_exp),
                };
              }
              factored = true;
              break;
            }
          }
          break;
        }
      }
      if !factored {
        return None;
      }
      if remaining_factors.is_empty() {
        new_terms.push(Expr::Integer(1));
      } else {
        new_terms.push(build_product(remaining_factors));
      }
    }

    // Build the remaining expression and try to solve it
    let remaining = expand_and_combine(&build_sum(new_terms));
    // Factor out common terms that are constant w.r.t. the solve variable
    // e.g. 2*a^2*k*q - 4*k*q*x^2 → factor out 2*k*q → a^2 - 2*x^2
    let remaining = factor_out_constant_factors(&remaining, var);
    if max_power_int(&remaining, var).is_some() {
      // Recursively solve
      let new_eq = Expr::Comparison {
        operands: vec![remaining, Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      };
      return Some(solve_ast(&[new_eq, args[1].clone()]));
    }
  }

  None
}

/// Divide two expressions symbolically, simplifying integer cases.
pub fn solve_divide(num: &Expr, den: &Expr) -> Expr {
  match (num, den) {
    (Expr::Integer(0), _) => Expr::Integer(0),
    (_, Expr::Integer(1)) => num.clone(),
    (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => {
      let g = gcd_i128(*n, *d).abs();
      let mut rn = n / g;
      let mut rd = d / g;
      // Normalize sign: denominator always positive
      if rd < 0 {
        rn = -rn;
        rd = -rd;
      }
      if rd == 1 {
        Expr::Integer(rn)
      } else {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(rn), Expr::Integer(rd)].into(),
        }
      }
    }
    // Non-integer denominator (a rational such as -1/2, or a symbolic
    // expression): evaluate the quotient so it is fully simplified
    // (e.g. -x / (-1/2) -> 2*x) rather than left as a nested fraction.
    _ => {
      let div = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(num.clone()),
        right: Box::new(den.clone()),
      };
      crate::evaluator::evaluate_expr_to_expr(&div).unwrap_or(div)
    }
  }
}

/// Build an equation `p(var) == 0` from a coefficient array where
/// `coeffs[i]` is the coefficient of `var^i`.
/// Used to construct the reduced polynomial after factoring out a zero root.
fn build_eq_from_coeffs(coeffs: &[Expr], var: &str) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();
  for (i, c) in coeffs.iter().enumerate() {
    if matches!(c, Expr::Integer(0)) {
      continue;
    }
    let term = if i == 0 {
      c.clone()
    } else if i == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(c.clone()),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(c.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Identifier(var.to_string())),
          right: Box::new(Expr::Integer(i as i128)),
        }),
      }
    };
    terms.push(term);
  }
  let poly_expr = if terms.is_empty() {
    Expr::Integer(0)
  } else {
    let mut result = terms.remove(0);
    for t in terms {
      result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(result),
        right: Box::new(t),
      };
    }
    result
  };
  Expr::Comparison {
    operands: vec![poly_expr, Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  }
}

/// Simplify Sqrt for integer arguments.
/// Returns (outside, inside) where Sqrt[n] = outside * Sqrt[inside].
/// E.g. Sqrt[20] = 2*Sqrt[5] → (2, 5), Sqrt[4] = 2 → (2, 1).
pub fn simplify_sqrt_parts(n: i128) -> (i128, i128) {
  if n == 0 {
    return (0, 1); // Sqrt[0] = 0 → (0, 1) so 0 * Sqrt[1] = 0
  }
  if n < 0 {
    return (1, n);
  }
  let mut outside = 1i128;
  let mut inside = n;
  // Extract perfect square factors
  let mut factor = 2i128;
  while factor * factor <= inside {
    while inside % (factor * factor) == 0 {
      inside /= factor * factor;
      outside *= factor;
    }
    factor += 1;
  }
  (outside, inside)
}

// ─── Root ─────────────────────────────────────────────────────────────

/// Root[f, k] — the k-th root of the polynomial defined by pure function f.
/// f is a pure function like `#^2 - 2 &`, and k is a positive integer.
/// Roots are ordered: real roots first (ascending), then complex roots
/// (by imaginary part, negative before positive).
/// Replace `Slot(k)` with a named identifier inside a (pure-function) body.
fn rs_subst_slot(expr: &Expr, k: usize, name: &str) -> Expr {
  match expr {
    Expr::Slot(n) if *n == k => Expr::Identifier(name.to_string()),
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| rs_subst_slot(e, k, name)).collect())
    }
    Expr::FunctionCall { name: fname, args } => Expr::FunctionCall {
      name: fname.clone(),
      args: args.iter().map(|e| rs_subst_slot(e, k, name)).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(rs_subst_slot(left, k, name)),
      right: Box::new(rs_subst_slot(right, k, name)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(rs_subst_slot(operand, k, name)),
    },
    _ => expr.clone(),
  }
}

/// `RootSum[f, form]` — the sum of `form[r]` over the roots `r` of the
/// polynomial equation `f[#] == 0`.
///
/// When `f` is a polynomial with exact numeric coefficients and `form` is a
/// polynomial, the sum is a symmetric function of the roots and equals a
/// power-sum combination obtained from Newton's identities — an exact rational
/// that matches wolframscript without finding the roots explicitly (e.g.
/// `RootSum[#^3 - # - 1 &, #^2 &]` → `2`). Other shapes (symbolic coefficients,
/// non-polynomial `form`) — for which wolframscript substitutes explicit
/// radical roots — are left unevaluated.
pub fn root_sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RootSum".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let var = "__rootsum_x__";
  let var_sym = Expr::Identifier(var.to_string());

  // Apply a pure-function argument `#... &` to the variable and expand.
  let apply = |f: &Expr| -> Option<Expr> {
    let body = match f {
      Expr::Function { body } => body.as_ref().clone(),
      _ => return None,
    };
    let b = crate::syntax::substitute_variable(&body, "#1", &var_sym);
    let b = rs_subst_slot(&b, 1, var);
    Some(crate::functions::polynomial_ast::expand_and_combine(&b))
  };
  let (Some(poly_f), Some(poly_form)) = (apply(&args[0]), apply(&args[1]))
  else {
    return unevaluated();
  };

  let is_poly = |e: &Expr| -> bool {
    matches!(
      crate::evaluator::evaluate_function_call_ast(
        "PolynomialQ",
        &[e.clone(), var_sym.clone()],
      ),
      Ok(Expr::Identifier(ref s)) if s == "True"
    )
  };
  if !is_poly(&poly_f) || !is_poly(&poly_form) {
    return unevaluated();
  }

  // Coefficient lists in ascending powers of `var`.
  let coeff_list = |e: &Expr| -> Option<Vec<Expr>> {
    match crate::evaluator::evaluate_function_call_ast(
      "CoefficientList",
      &[e.clone(), var_sym.clone()],
    ) {
      Ok(Expr::List(ref items)) => Some(items.iter().cloned().collect()),
      _ => None,
    }
  };
  let (Some(cf), Some(cform)) = (coeff_list(&poly_f), coeff_list(&poly_form))
  else {
    return unevaluated();
  };

  let d = cf.len().saturating_sub(1);
  if d < 1 {
    return unevaluated();
  }
  // Exact numeric coefficients only; symbolic ones make wolframscript expand
  // the explicit radical roots instead (a form we do not reproduce here).
  let is_number = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  if !cf.iter().all(&is_number) || !cform.iter().all(&is_number) {
    return unevaluated();
  }

  let mul = |a: &Expr, b: &Expr| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[a.clone(), b.clone()],
    )
  };
  let add = |a: &Expr, b: &Expr| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[a.clone(), b.clone()],
    )
  };
  let div = |a: &Expr, b: &Expr| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_function_call_ast(
      "Divide",
      &[a.clone(), b.clone()],
    )
  };

  // Monic coefficients a_i = cf[i] / cf[d] (a_d = 1).
  let lead = &cf[d];
  let mut a = Vec::with_capacity(d + 1);
  for c in &cf {
    a.push(div(c, lead)?);
  }
  // Elementary symmetric functions e_j = (-1)^j a_{d-j}, j = 1..=d.
  // e[0] is unused.
  let mut e = vec![Expr::Integer(0); d + 1];
  for j in 1..=d {
    let sign = if j % 2 == 0 { 1 } else { -1 };
    e[j] = mul(&Expr::Integer(sign), &a[d - j])?;
  }

  // Power sums p_0..p_m via Newton's identities. p_0 = d (root count).
  let m = cform.len().saturating_sub(1);
  let mut p = vec![Expr::Integer(0); m + 1];
  p[0] = Expr::Integer(d as i128);
  for k in 1..=m {
    let mut acc = Expr::Integer(0);
    let lim = if k <= d { k - 1 } else { d };
    for i in 1..=lim {
      let sign = if (i - 1) % 2 == 0 { 1 } else { -1 };
      let term = mul(&e[i], &p[k - i])?;
      let term = mul(&Expr::Integer(sign), &term)?;
      acc = add(&acc, &term)?;
    }
    if k <= d {
      // Special diagonal term (-1)^(k-1) * k * e_k.
      let sign = if (k - 1) % 2 == 0 { 1 } else { -1 };
      let term = mul(&Expr::Integer(sign * k as i128), &e[k])?;
      acc = add(&acc, &term)?;
    }
    p[k] = acc;
  }

  // Sum_{j=0}^m cform[j] * p_j.
  let mut result = Expr::Integer(0);
  for j in 0..=m {
    let term = mul(&cform[j], &p[j])?;
    result = add(&result, &term)?;
  }
  Ok(result)
}

pub fn root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "Root".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Root[f, k, 0] is the exact form (same as Root[f, k]). Root[f, k, 1]
  // requests fast numerical evaluation — we leave it symbolic for now,
  // matching wolframscript's behaviour when it cannot simplify to a
  // closed form.
  if args.len() == 3 && !matches!(&args[2], Expr::Integer(0) | Expr::Integer(1))
  {
    return Ok(Expr::FunctionCall {
      name: "Root".to_string(),
      args: args.to_vec().into(),
    });
  }

  let k = match &args[1] {
    Expr::Integer(n) => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Root".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if k < 1 {
    return Err(InterpreterError::EvaluationError(
      "Root: index k must be a positive integer".into(),
    ));
  }

  // Extract pure function body
  let body = match &args[0] {
    Expr::Function { body } => body,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Root".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Substitute Slot(1) with a temporary variable
  let var_name = "\u{2620}root\u{2620}"; // unique internal variable
  let poly =
    crate::syntax::substitute_slots(body, &[Expr::Identifier(var_name.into())]);

  // Solve the polynomial equation poly == 0
  let eq = Expr::Comparison {
    operands: vec![poly, Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };

  let solutions = solve_ast(&[eq, Expr::Identifier(var_name.into())])?;

  // Extract root values from {{var -> val1}, {var -> val2}, ...}
  let mut roots: Vec<Expr> = Vec::new();
  if let Expr::List(outer) = &solutions {
    for item in outer {
      if let Expr::List(inner) = item {
        for rule in inner {
          if let Expr::Rule { replacement, .. } = rule {
            roots.push(*replacement.clone());
          }
        }
      }
    }
  }

  if roots.is_empty() {
    // No closed-form roots: return the canonical Root[poly &, k, 0] form
    // with the polynomial body re-evaluated so its terms sort ascending in
    // Slot[1] (matching wolframscript: `Root[#1^5+2#1+1&, 2]` →
    // `Root[1 + 2*#1 + #1^5 &, 2, 0]`).
    let canonical_body = match &args[0] {
      Expr::Function { body } => {
        let normalized = crate::evaluator::evaluate_expr_to_expr(body)?;
        Expr::Function {
          body: Box::new(normalized),
        }
      }
      _ => args[0].clone(),
    };
    return Ok(Expr::FunctionCall {
      name: "Root".to_string(),
      args: vec![canonical_body, args[1].clone(), Expr::Integer(0)].into(),
    });
  }

  // Sort roots: real roots first (ascending), then complex roots
  roots.sort_by(root_order);

  let idx = (k as usize) - 1;
  if idx >= roots.len() {
    return Err(InterpreterError::EvaluationError(format!(
      "Root: index {} is out of range; polynomial has only {} roots",
      k,
      roots.len()
    )));
  }

  // Simplify the result
  crate::evaluator::evaluate_expr_to_expr(&roots[idx])
}

/// Order roots the way Wolfram's `Root` does: real roots first, sorted
/// ascending, then complex roots sorted by (real, imag).
pub fn root_order(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  use crate::functions::list_helpers_ast::expr_to_complex_parts;
  let pa = expr_to_complex_parts(a);
  let pb = expr_to_complex_parts(b);

  match (pa, pb) {
    (Some((a_re, a_im)), Some((b_re, b_im))) => {
      let a_real = a_im.abs() < 1e-15;
      let b_real = b_im.abs() < 1e-15;
      match (a_real, b_real) {
        (true, true) => {
          // Both real: sort ascending
          a_re.partial_cmp(&b_re).unwrap_or(std::cmp::Ordering::Equal)
        }
        (true, false) => std::cmp::Ordering::Less, // real before complex
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => {
          // Both complex: sort by real part, then imaginary part
          match a_re.partial_cmp(&b_re).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => {
              a_im.partial_cmp(&b_im).unwrap_or(std::cmp::Ordering::Equal)
            }
            other => other,
          }
        }
      }
    }
    (Some(_), None) => std::cmp::Ordering::Less,
    (None, Some(_)) => std::cmp::Ordering::Greater,
    (None, None) => std::cmp::Ordering::Equal,
  }
}

/// Order solutions the way Wolfram's `Solve` does: lexicographic by
/// (real, imag) with real (imag = 0) tied to the front of any complex
/// group sharing the same real part. `{-1, 0, 1, -I, I}` sorts as
/// `{-1, 0, -I, I, 1}` — `-I` and `I` slot between `0` and `1` because
/// they share real part 0. (`Root` uses a different rule that floats
/// every real to the head; both functions are intentionally distinct.)
fn solve_order(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  use crate::functions::list_helpers_ast::expr_to_complex_parts;
  let pa = expr_to_complex_parts(a);
  let pb = expr_to_complex_parts(b);

  match (pa, pb) {
    (Some((a_re, a_im)), Some((b_re, b_im))) => {
      let by_re = a_re.partial_cmp(&b_re).unwrap_or(std::cmp::Ordering::Equal);
      if by_re != std::cmp::Ordering::Equal {
        return by_re;
      }
      let a_real = a_im.abs() < 1e-15;
      let b_real = b_im.abs() < 1e-15;
      match (a_real, b_real) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a_im.partial_cmp(&b_im).unwrap_or(std::cmp::Ordering::Equal),
      }
    }
    (Some(_), None) => std::cmp::Ordering::Less,
    (None, Some(_)) => std::cmp::Ordering::Greater,
    (None, None) => std::cmp::Ordering::Equal,
  }
}

// ─── FindRoot ────────────────────────────────────────────────────────

/// FindRoot[expr, {var, x0}] — numerically find a root using Newton's method.
///
/// `expr` can be an expression (finds where it equals 0) or an equation `lhs == rhs`.
/// Returns `{var -> root_value}`.
pub fn find_root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "FindRoot expects at least 2 arguments".into(),
    ));
  }

  // Multivariate form: FindRoot[{eqns}, {{x, x0}, {y, y0}, ...}] — every
  // variable spec is itself a {var, start} list. Solved with multidimensional
  // Newton iteration.
  if let Expr::List(specs) = &args[1]
    && !specs.is_empty()
    && specs.iter().all(|s| {
      matches!(s, Expr::List(p)
      if p.len() == 2 && matches!(&p[0], Expr::Identifier(_)))
    })
  {
    return find_root_multivariate(&args[0], specs);
  }

  // Parse options from additional arguments (Method, etc.) — currently ignored
  let mut use_secant = false;
  for opt in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(s) = pattern.as_ref()
      && s == "Method"
      && let Expr::String(m) = replacement.as_ref()
      && m == "Secant"
    {
      use_secant = true;
    }
  }

  // Parse second argument: {var, x0} or {var, x0, x1}
  // First peek at the variable name and start point; if the start point
  // evaluates to a complex number we route to a complex Newton iteration
  // before the real-only path below.
  let (var_name, x_start_expr) = match &args[1] {
    Expr::List(items) if items.len() == 2 || items.len() == 3 => {
      let name = match &items[0] {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: variable must be a symbol".into(),
          ));
        }
      };
      (name, items[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "FindRoot: second argument must be {var, x0} or {var, x0, x1}".into(),
      ));
    }
  };
  // Try to detect a complex start (e.g. -I, 1 + 2 I). If the start
  // evaluates to a non-real numeric value, fall through to a complex
  // Newton iteration.
  if let Some((re0, im0)) = try_extract_complex_f64(&x_start_expr)
    && im0 != 0.0
  {
    let func = build_find_root_func(&args[0]);
    let deriv =
      crate::functions::calculus_ast::differentiate_expr(&func, &var_name)
        .ok()
        .map(simplify);
    return find_root_complex_newton(
      &func,
      deriv.as_ref(),
      &var_name,
      re0,
      im0,
    );
  }
  let (var, x0, x1_opt) = match &args[1] {
    Expr::List(items) if items.len() == 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: variable must be a symbol".into(),
          ));
        }
      };
      let x0 = find_root_eval_number(&items[1])?;
      (var_name, x0, None)
    }
    Expr::List(items) if items.len() == 3 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: variable must be a symbol".into(),
          ));
        }
      };
      let x0 = find_root_eval_number(&items[1])?;
      let x1 = find_root_eval_number(&items[2])?;
      (var_name, x0, Some(x1))
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "FindRoot: second argument must be {var, x0} or {var, x0, x1}".into(),
      ));
    }
  };
  // Use secant method if x1 is provided or Method -> "Secant"
  let use_secant = use_secant || x1_opt.is_some();

  // Extract the function to find root of: expr or lhs - rhs for equations
  let func = build_find_root_func(&args[0]);

  // Secant method when requested
  if use_secant {
    let max_iter = 100;
    let tol = 1e-15;
    let mut x_prev = x0;
    let mut x_curr = x1_opt.unwrap_or(x0 + 0.1);
    let mut f_prev = find_root_eval_at(&func, &var, x_prev)?;

    for _ in 0..max_iter {
      let f_curr = find_root_eval_at(&func, &var, x_curr)?;
      if f_curr.abs() < tol {
        break;
      }
      let denom = f_curr - f_prev;
      if denom.abs() < 1e-30 {
        break;
      }
      let x_next = x_curr - f_curr * (x_curr - x_prev) / denom;
      x_prev = x_curr;
      f_prev = f_curr;
      x_curr = x_next;
    }

    let result_val =
      if x_curr == 0.0 || (x_curr.abs() > 1e-15 && x_curr.abs() < 1e15) {
        Expr::Real(x_curr)
      } else {
        Expr::Real(x_curr)
      };
    return Ok(Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::Identifier(var)),
        replacement: Box::new(result_val),
      }]
      .into(),
    ));
  }

  // Try symbolic derivative; fall back to numerical if unavailable
  let deriv_expr =
    crate::functions::calculus_ast::differentiate_expr(&func, &var)
      .ok()
      .map(simplify)
      .filter(|d| !contains_unevaluated_d(d));

  // Newton's method
  let max_iter = 100;
  let tol = 1e-15;
  let mut x = x0;

  for _ in 0..max_iter {
    let fx = find_root_eval_at(&func, &var, x)?;
    if fx.abs() < tol {
      break;
    }
    // Compute derivative: symbolic if available, else numerical
    let fpx = if let Some(ref d) = deriv_expr {
      find_root_eval_at(d, &var, x)?
    } else {
      // 4th-order central difference for high-precision derivative
      let h = x.abs().max(1.0) * 1e-4;
      let fp1 = find_root_eval_at(&func, &var, x + h)?;
      let fm1 = find_root_eval_at(&func, &var, x - h)?;
      let fp2 = find_root_eval_at(&func, &var, x + 2.0 * h)?;
      let fm2 = find_root_eval_at(&func, &var, x - 2.0 * h)?;
      (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
    };
    if fpx.abs() < 1e-30 {
      // Derivative too small — try secant method step
      let h = 1e-8;
      let fx_plus = find_root_eval_at(&func, &var, x + h)?;
      let fpx_approx = (fx_plus - fx) / h;
      if fpx_approx.abs() < 1e-30 {
        return Err(InterpreterError::EvaluationError(
          "FindRoot: derivative is zero, cannot converge".into(),
        ));
      }
      x -= fx / fpx_approx;
    } else {
      x -= fx / fpx;
    }
  }

  // Format the result
  let result_val = if x == 0.0 || (x.abs() > 1e-15 && x.abs() < 1e15) {
    Expr::Real(x)
  } else {
    Expr::Real(x)
  };

  // Clean up -0.0
  let result_val = if x == 0.0 {
    Expr::Real(0.0)
  } else {
    result_val
  };

  // Re-evaluate the LHS so a held variable name with an OwnValue (e.g.
  // `x = "I am the result!"; FindRoot[…, {x, 1}]`) gets surfaced as that
  // bound value in the returned rule, matching wolframscript's
  // `{I am the result! -> 1.149…}` behaviour.
  let lhs_ident = Expr::Identifier(var);
  let lhs =
    crate::evaluator::evaluate_expr_to_expr(&lhs_ident).unwrap_or(lhs_ident);
  Ok(Expr::List(
    vec![Expr::Rule {
      pattern: Box::new(lhs),
      replacement: Box::new(result_val),
    }]
    .into(),
  ))
}

/// Convert FindRoot's first argument into the function whose root we
/// seek. For equations `lhs == rhs` this is `lhs - rhs`; otherwise the
/// expression is used directly.
fn build_find_root_func(arg: &Expr) -> Expr {
  match arg {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(fargs[0].clone()),
        right: Box::new(fargs[1].clone()),
      }
    }
    other => other.clone(),
  }
}

/// Try to evaluate `expr` to a complex `(re, im)` pair using f64
/// arithmetic. Returns None when the expression isn't fully numeric.
fn try_extract_complex_f64(expr: &Expr) -> Option<(f64, f64)> {
  let n_result = crate::functions::math_ast::n_ast(&[expr.clone()]).ok()?;
  expr_to_complex_f64(&n_result)
}

/// Decompose an evaluated expression into a `(re, im)` pair.
fn expr_to_complex_f64(expr: &Expr) -> Option<(f64, f64)> {
  match expr {
    Expr::Integer(n) => Some((*n as f64, 0.0)),
    Expr::Real(r) => Some((*r, 0.0)),
    Expr::Constant(s) | Expr::Identifier(s) if s == "I" => Some((0.0, 1.0)),
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      let re = expr_to_real_f64(&args[0])?;
      let im = expr_to_real_f64(&args[1])?;
      Some((re, im))
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n as f64 / *d as f64, 0.0))
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (re, im) = expr_to_complex_f64(operand)?;
      Some((-re, -im))
    }
    // Plus form: a + b — sum of complex parts.
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let (lr, li) = expr_to_complex_f64(left)?;
      let (rr, ri) = expr_to_complex_f64(right)?;
      Some((lr + rr, li + ri))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let (lr, li) = expr_to_complex_f64(left)?;
      let (rr, ri) = expr_to_complex_f64(right)?;
      Some((lr - rr, li - ri))
    }
    // Times form: a * b — complex multiplication.
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (ar, ai) = expr_to_complex_f64(left)?;
      let (br, bi) = expr_to_complex_f64(right)?;
      Some((ar * br - ai * bi, ar * bi + ai * br))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let (ar, ai) = expr_to_complex_f64(left)?;
      let (br, bi) = expr_to_complex_f64(right)?;
      let denom = br * br + bi * bi;
      if denom < 1e-300 {
        return None;
      }
      Some(((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom))
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut re = 0.0;
      let mut im = 0.0;
      for a in args.iter() {
        let (r, i) = expr_to_complex_f64(a)?;
        re += r;
        im += i;
      }
      Some((re, im))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut re = 1.0;
      let mut im = 0.0;
      for a in args.iter() {
        let (br, bi) = expr_to_complex_f64(a)?;
        let nr = re * br - im * bi;
        let ni = re * bi + im * br;
        re = nr;
        im = ni;
      }
      Some((re, im))
    }
    _ => None,
  }
}

fn expr_to_real_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(*n as f64 / *d as f64)
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(-expr_to_real_f64(operand)?),
    _ => None,
  }
}

/// Substitute `var` with the complex value `re + im*I` in `expr`,
/// evaluate, and return the result as `(re, im)`. Falls back to None
/// when the result isn't reducible to a complex number.
fn find_root_eval_complex_at(
  expr: &Expr,
  var: &str,
  re: f64,
  im: f64,
) -> Option<(f64, f64)> {
  let value = if im == 0.0 {
    Expr::Real(re)
  } else {
    Expr::FunctionCall {
      name: "Complex".to_string(),
      args: vec![Expr::Real(re), Expr::Real(im)].into(),
    }
  };
  let substituted = crate::syntax::substitute_variable(expr, var, &value);
  let evaled = crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
  // First try the natural form; if not yet a Complex/Real, push through
  // N[] which collapses things like Sin[Complex[…]] into a numeric form.
  if let Some(c) = expr_to_complex_f64(&evaled) {
    return Some(c);
  }
  let n_result = crate::functions::math_ast::n_ast(&[evaled]).ok()?;
  expr_to_complex_f64(&n_result)
}

/// Newton's method on the complex plane. Mirrors the real-only path
/// but works in (re, im) pairs throughout. Uses a numerical derivative
/// when the symbolic derivative isn't available.
fn find_root_complex_newton(
  func: &Expr,
  deriv: Option<&Expr>,
  var: &str,
  re0: f64,
  im0: f64,
) -> Result<Expr, InterpreterError> {
  let max_iter = 100;
  let tol = 1e-15;
  let (mut re, mut im) = (re0, im0);
  // Complex helpers
  let cabs = |a: f64, b: f64| (a * a + b * b).sqrt();
  let cdiv = |ar: f64, ai: f64, br: f64, bi: f64| -> Option<(f64, f64)> {
    let denom = br * br + bi * bi;
    if denom < 1e-300 {
      return None;
    }
    Some(((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom))
  };
  for _ in 0..max_iter {
    let (fr, fi) = match find_root_eval_complex_at(func, var, re, im) {
      Some(v) => v,
      None => {
        return Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression at complex point".into(),
        ));
      }
    };
    if cabs(fr, fi) < tol {
      break;
    }
    let (dr, di) = if let Some(d) = deriv {
      match find_root_eval_complex_at(d, var, re, im) {
        Some(v) => v,
        None => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: cannot evaluate derivative at complex point".into(),
          ));
        }
      }
    } else {
      // Numerical derivative via complex finite difference along the
      // real axis. f(z+h) − f(z) over h, with small real h.
      let h = re.abs().max(1.0) * 1e-6;
      let (fr_p, fi_p) = match find_root_eval_complex_at(func, var, re + h, im)
      {
        Some(v) => v,
        None => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: cannot evaluate expression for derivative".into(),
          ));
        }
      };
      ((fr_p - fr) / h, (fi_p - fi) / h)
    };
    let Some((sr, si)) = cdiv(fr, fi, dr, di) else {
      return Err(InterpreterError::EvaluationError(
        "FindRoot: derivative is zero, cannot converge".into(),
      ));
    };
    re -= sr;
    im -= si;
  }
  // Build the complex result. Drop the imaginary part if it collapsed
  // to zero so the rule reads like a real solution. Otherwise build
  // `re + im*I` (or `re - |im|*I`) so it formats like wolframscript.
  let value = if im.abs() < 1e-14 {
    Expr::Real(re)
  } else {
    let im_term = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Real(im.abs())),
      right: Box::new(Expr::Identifier("I".to_string())),
    };
    let combined = if im >= 0.0 {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Real(re)),
        right: Box::new(im_term),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Real(re)),
        right: Box::new(im_term),
      }
    };
    crate::evaluator::evaluate_expr_to_expr(&combined).unwrap_or(combined)
  };
  let lhs_ident = Expr::Identifier(var.to_string());
  let lhs =
    crate::evaluator::evaluate_expr_to_expr(&lhs_ident).unwrap_or(lhs_ident);
  Ok(Expr::List(
    vec![Expr::Rule {
      pattern: Box::new(lhs),
      replacement: Box::new(value),
    }]
    .into(),
  ))
}

/// Evaluate an expression numerically at a specific value of var.
fn find_root_eval_at(
  expr: &Expr,
  var: &str,
  x: f64,
) -> Result<f64, InterpreterError> {
  let substituted =
    crate::syntax::substitute_variable(expr, var, &Expr::Real(x));
  let evaled = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
  match &evaled {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Ok(*n as f64 / *d as f64)
      } else {
        Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression numerically".into(),
        ))
      }
    }
    _ => {
      // Try N[] evaluation
      let n_result = crate::functions::math_ast::n_ast(&[evaled])?;
      match &n_result {
        Expr::Real(r) => Ok(*r),
        Expr::Integer(n) => Ok(*n as f64),
        _ => Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression numerically".into(),
        )),
      }
    }
  }
}

/// Parse a number from an expression for FindRoot starting point.
fn find_root_eval_number(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Ok(-find_root_eval_number(operand)?),
    _ => {
      // Try evaluating
      let evaled = crate::evaluator::evaluate_expr_to_expr(expr)?;
      match &evaled {
        Expr::Integer(n) => Ok(*n as f64),
        Expr::Real(r) => Ok(*r),
        _ => {
          // Try N[] to convert symbolic expressions (e.g. Pi/4) to numeric
          let n_expr = Expr::FunctionCall {
            name: "N".to_string(),
            args: vec![evaled.clone()].into(),
          };
          let n_result = crate::evaluator::evaluate_expr_to_expr(&n_expr)?;
          match &n_result {
            Expr::Real(r) => Ok(*r),
            Expr::Integer(n) => Ok(*n as f64),
            _ => Err(InterpreterError::EvaluationError(
              "FindRoot: starting point must be numeric".into(),
            )),
          }
        }
      }
    }
  }
}

/// Evaluate `expr` to an f64 with every variable in `vars` bound to the
/// corresponding value in `vals`.
fn find_root_eval_multivar(
  expr: &Expr,
  vars: &[String],
  vals: &[f64],
) -> Result<f64, InterpreterError> {
  let mut e = expr.clone();
  for (v, &x) in vars.iter().zip(vals) {
    e = crate::syntax::substitute_variable(&e, v, &Expr::Real(x));
  }
  let evaled = crate::evaluator::evaluate_expr_to_expr(&e)?;
  match &evaled {
    Expr::Integer(k) => Ok(*k as f64),
    Expr::Real(r) => Ok(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Ok(*n as f64 / *d as f64)
      } else {
        Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression numerically".into(),
        ))
      }
    }
    _ => {
      let n_result = crate::functions::math_ast::n_ast(&[evaled])?;
      match &n_result {
        Expr::Real(r) => Ok(*r),
        Expr::Integer(k) => Ok(*k as f64),
        _ => Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression numerically".into(),
        )),
      }
    }
  }
}

/// Solve the square f64 linear system `A x = b` by Gaussian elimination with
/// partial pivoting. None if the matrix is (near-)singular.
fn find_root_solve_linear(
  mut a: Vec<Vec<f64>>,
  mut b: Vec<f64>,
) -> Option<Vec<f64>> {
  let n = a.len();
  for col in 0..n {
    let piv = (col..n).max_by(|&r1, &r2| {
      a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap()
    })?;
    if a[piv][col].abs() < 1e-14 {
      return None;
    }
    a.swap(col, piv);
    b.swap(col, piv);
    for r in 0..n {
      if r == col {
        continue;
      }
      let factor = a[r][col] / a[col][col];
      for c in col..n {
        a[r][c] -= factor * a[col][c];
      }
      b[r] -= factor * b[col];
    }
  }
  Some((0..n).map(|i| b[i] / a[i][i]).collect())
}

/// Multivariate FindRoot via Newton's method:
/// FindRoot[{f1==g1, ...}, {{x, x0}, {y, y0}, ...}] -> {x -> .., y -> ..}.
fn find_root_multivariate(
  eqns_arg: &Expr,
  specs: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Variables and starting points.
  let mut vars: Vec<String> = Vec::new();
  let mut x: Vec<f64> = Vec::new();
  for spec in specs {
    if let Expr::List(p) = spec {
      let v = match &p[0] {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: variable must be a symbol".into(),
          ));
        }
      };
      vars.push(v);
      x.push(find_root_eval_number(&p[1])?);
    }
  }
  let n = vars.len();

  // Equations as f_i = lhs - rhs (or bare expression).
  let eqns: Vec<Expr> = match eqns_arg {
    Expr::List(es) => es.iter().map(build_find_root_func).collect(),
    other => vec![build_find_root_func(other)],
  };
  if eqns.len() != n {
    return Err(InterpreterError::EvaluationError(
      "FindRoot: number of equations must match number of variables".into(),
    ));
  }

  // Jacobian J[i][j] = d f_i / d x_j (symbolic).
  let mut jac: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for f in &eqns {
    let mut row = Vec::with_capacity(n);
    for v in &vars {
      let d = crate::functions::calculus_ast::differentiate_expr(f, v)
        .map(simplify)
        .map_err(|_| {
          InterpreterError::EvaluationError(
            "FindRoot: cannot differentiate equation".into(),
          )
        })?;
      row.push(d);
    }
    jac.push(row);
  }

  // Newton iteration.
  let max_iter = 100;
  let tol = 1e-13;
  for _ in 0..max_iter {
    let mut fv = vec![0.0; n];
    for (i, f) in eqns.iter().enumerate() {
      fv[i] = find_root_eval_multivar(f, &vars, &x)?;
    }
    if fv.iter().fold(0.0f64, |a, &b| a.max(b.abs())) < tol {
      break;
    }
    let mut jm = vec![vec![0.0; n]; n];
    for (i, row) in jac.iter().enumerate() {
      for (j, dij) in row.iter().enumerate() {
        jm[i][j] = find_root_eval_multivar(dij, &vars, &x)?;
      }
    }
    let neg_f: Vec<f64> = fv.iter().map(|v| -v).collect();
    let delta = match find_root_solve_linear(jm, neg_f) {
      Some(d) => d,
      None => break,
    };
    let mut max_d = 0.0f64;
    for (j, &dj) in delta.iter().enumerate() {
      x[j] += dj;
      max_d = max_d.max(dj.abs());
    }
    if max_d < tol {
      break;
    }
  }

  let rules: Vec<Expr> = vars
    .iter()
    .zip(&x)
    .map(|(v, &xv)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(Expr::Real(xv)),
    })
    .collect();
  Ok(Expr::List(rules.into()))
}

/// Check if an expression contains an unevaluated D[...] or Dt[...] call.
fn contains_unevaluated_d(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, .. } if name == "D" || name == "Dt" => true,
    Expr::FunctionCall { args, .. } => args.iter().any(contains_unevaluated_d),
    Expr::BinaryOp { left, right, .. } => {
      contains_unevaluated_d(left) || contains_unevaluated_d(right)
    }
    Expr::UnaryOp { operand, .. } => contains_unevaluated_d(operand),
    _ => false,
  }
}

// ─── Minimize / Maximize ─────────────────────────────────────────────

/// Minimize[f, x] or Minimize[f, {x, y, ...}] — find the global minimum.
/// Minimize[{f, cons1, cons2, ...}, vars] — constrained minimization.
/// Returns {min_val, {x -> x_min, ...}} with exact results when possible.
///
/// Maximize[f, vars] is the dual (negates objective and result).
pub fn minimize_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize { "Maximize" } else { "Minimize" };
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(format!(
      "{func_name} expects 2 or 3 arguments"
    )));
  }

  // A non-symbol in the variable slot (a constraint, equation, or literal —
  // e.g. Minimize[x^2, x >= 1]) is not a valid variable: wolframscript emits
  // <func>::ivar and returns the call unevaluated rather than raising an
  // error.
  if let Some(bad) = minimize_first_invalid_var(&args[1]) {
    crate::emit_message(&format!(
      "{func_name}::ivar: {} is not a valid variable.",
      expr_to_string(bad)
    ));
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: args.to_vec().into(),
    });
  }

  // Parse variable list: x, {x}, {x, y}, or {n[1], n[2], ...}
  let (_var_strings, var_exprs) =
    minimize_parse_vars_full(&args[1], func_name)?;

  // Detect if any variable is a FunctionCall (e.g. n[1]).
  // If so, rename them to fresh identifiers so the solver can treat them as plain symbols.
  let has_funccall_vars = var_exprs
    .iter()
    .any(|e| matches!(e, Expr::FunctionCall { .. }));

  // Fresh names for FunctionCall vars: __ilp_0, __ilp_1, ...
  let fresh_names: Vec<String> =
    (0..var_exprs.len()).map(|i| format!("__ilp_{i}")).collect();

  // Rename FunctionCall vars in an expression
  let rename_forward = |mut e: Expr| -> Expr {
    if has_funccall_vars {
      for (orig, fresh) in var_exprs.iter().zip(fresh_names.iter()) {
        if matches!(orig, Expr::FunctionCall { .. }) {
          e = substitute_expr(&e, orig, &Expr::Identifier(fresh.clone()));
        }
      }
    }
    e
  };

  // The effective var names used by the solver
  let vars: Vec<String> = if has_funccall_vars {
    fresh_names.clone()
  } else {
    var_exprs
      .iter()
      .map(|e| {
        if let Expr::Identifier(n) = e {
          n.clone()
        } else {
          expr_to_string(e)
        }
      })
      .collect()
  };

  // Parse objective and constraints: f or {f, cons1, cons2, ...}
  let (raw_objective, raw_constraints) = minimize_parse_objective(&args[0]);
  let objective = rename_forward(raw_objective);
  let mut constraints: Vec<Expr> =
    raw_constraints.into_iter().map(rename_forward).collect();

  // Handle 3-argument form: Minimize[{obj, cons}, vars, Domain]
  // If the third argument is Integers, inject Element[var, Integers] for each var.
  if args.len() == 3 {
    let domain_is_integers =
      matches!(&args[2], Expr::Identifier(d) if d == "Integers");
    if domain_is_integers {
      for var in &vars {
        constraints.push(Expr::FunctionCall {
          name: "Element".to_string(),
          args: vec![
            Expr::Identifier(var.clone()),
            Expr::Identifier("Integers".to_string()),
          ]
          .into(),
        });
      }
    }
  }

  let result = if constraints.is_empty() {
    // Unconstrained
    if vars.len() == 1 {
      minimize_single_var(&objective, &vars[0], maximize, func_name)
    } else {
      minimize_multi_var(&objective, &vars, maximize, func_name)
    }
  } else {
    minimize_constrained(&objective, &constraints, &vars, maximize, func_name)
  }?;

  // Reverse renaming: replace fresh identifiers back with original FunctionCall exprs
  // in Rule patterns so the result uses the original variable names.
  if has_funccall_vars {
    let result = fresh_names
      .iter()
      .zip(var_exprs.iter())
      .filter(|(_, orig)| matches!(orig, Expr::FunctionCall { .. }))
      .fold(result, |acc, (fresh, orig)| {
        substitute_expr(&acc, &Expr::Identifier(fresh.clone()), orig)
      });
    return Ok(result);
  }
  Ok(result)
}

/// Return the first entry in the variable specification that is not a valid
/// variable (a plain symbol or an indexed symbol like `n[1]`). A constraint,
/// equation, or literal in the variable slot is invalid. Returns `None` when
/// every entry is a valid variable (or the spec is an empty list, which is
/// handled separately).
fn minimize_first_invalid_var(expr: &Expr) -> Option<&Expr> {
  fn is_valid_var(item: &Expr) -> bool {
    matches!(item, Expr::Identifier(_) | Expr::FunctionCall { .. })
  }
  match expr {
    Expr::List(items) => items.iter().find(|it| !is_valid_var(it)),
    _ if !is_valid_var(expr) => Some(expr),
    _ => None,
  }
}

/// Parse var list returning (string_names, original_exprs).
/// Accepts plain identifiers AND FunctionCall expressions like n[1].
fn minimize_parse_vars_full(
  expr: &Expr,
  func_name: &str,
) -> Result<(Vec<String>, Vec<Expr>), InterpreterError> {
  fn parse_one(
    item: &Expr,
    func_name: &str,
  ) -> Result<(String, Expr), InterpreterError> {
    match item {
      Expr::Identifier(name) => Ok((name.clone(), item.clone())),
      Expr::FunctionCall { .. } => Ok((expr_to_string(item), item.clone())),
      _ => Err(InterpreterError::EvaluationError(format!(
        "{func_name}: variables must be symbols"
      ))),
    }
  }
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(format!(
          "{func_name}: variable list cannot be empty"
        )));
      }
      let mut names = Vec::new();
      let mut exprs = Vec::new();
      for item in items {
        let (n, e) = parse_one(item, func_name)?;
        names.push(n);
        exprs.push(e);
      }
      Ok((names, exprs))
    }
    _ => {
      let (n, e) = parse_one(expr, func_name)?;
      Ok((vec![n], vec![e]))
    }
  }
}

/// Recursively replace every occurrence of `from` with `to` in `expr`.
fn substitute_expr(expr: &Expr, from: &Expr, to: &Expr) -> Expr {
  if expr_to_string(expr) == expr_to_string(from) {
    return to.clone();
  }
  match expr {
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| substitute_expr(e, from, to)).collect())
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(|e| substitute_expr(e, from, to)).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_expr(left, from, to)),
      right: Box::new(substitute_expr(right, from, to)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_expr(operand, from, to)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| substitute_expr(e, from, to))
        .collect(),
      operators: operators.clone(),
    },
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_expr(pattern, from, to)),
      replacement: Box::new(substitute_expr(replacement, from, to)),
    },
    _ => expr.clone(),
  }
}

fn minimize_parse_objective(expr: &Expr) -> (Expr, Vec<Expr>) {
  if let Expr::List(items) = expr
    && !items.is_empty()
  {
    return (items[0].clone(), items[1..].to_vec());
  }
  (expr.clone(), vec![])
}

/// Evaluate f at a specific value of var, returning an exact expression when possible.
/// Falls back to numerical evaluation and recognizes simple integers/rationals.
fn minimize_eval_exact(
  f: &Expr,
  var: &str,
  val: &Expr,
) -> Result<Expr, InterpreterError> {
  let substituted = crate::syntax::substitute_variable(f, var, val);
  let evaled = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
  let simplified = simplify(evaled);

  // If already an exact integer, return it.
  if matches!(&simplified, Expr::Integer(_)) {
    return Ok(simplified);
  }
  // A Real value at the extremum is usually the machine form of an exact
  // integer/rational (e.g. Sin[x]/3 at its minimum is -0.3333… = -1/3). Try to
  // recover that exact value, falling back to the Real when none is close.
  if let Expr::Real(v) = &simplified {
    return Ok(minimize_recognize_exact(*v));
  }
  if let Expr::FunctionCall { name, .. } = &simplified
    && name == "Rational"
  {
    return Ok(simplified);
  }
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = &simplified
  {
    if matches!(operand.as_ref(), Expr::Integer(_)) {
      return Ok(simplified);
    }
    if let Expr::FunctionCall { name, .. } = operand.as_ref()
      && name == "Rational"
    {
      return Ok(simplified);
    }
  }

  // Try numerical evaluation to recognize exact integer/rational value
  if let Some(num_val) = minimize_try_f64(&simplified) {
    return Ok(minimize_recognize_exact(num_val));
  }

  Ok(simplified)
}

/// Try to recognize a float as an exact integer or rational.
fn minimize_recognize_exact(v: f64) -> Expr {
  if !v.is_finite() {
    return Expr::Real(v);
  }
  let rounded = v.round();
  if (rounded - v).abs() < 1e-8 {
    return Expr::Integer(rounded as i128);
  }
  for q in 2i128..=20 {
    let p = (v * q as f64).round() as i128;
    if ((p as f64 / q as f64) - v).abs() < 1e-8 {
      let (rn, rd) = reduce_fraction(p, q);
      return if rd == 1 {
        Expr::Integer(rn)
      } else {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(rn), Expr::Integer(rd)].into(),
        }
      };
    }
  }
  Expr::Real(v)
}

/// Evaluate f at multiple variables, returning exact result when possible.
fn minimize_eval_exact_multi(
  f: &Expr,
  vars: &[String],
  vals: &[Expr],
) -> Result<Expr, InterpreterError> {
  let mut expr = f.clone();
  for (var, val) in vars.iter().zip(vals.iter()) {
    expr = crate::syntax::substitute_variable(&expr, var, val);
  }
  let evaled = crate::evaluator::evaluate_expr_to_expr(&expr)?;
  let simplified = simplify(evaled);

  // If already an exact integer, return it.
  if matches!(&simplified, Expr::Integer(_)) {
    return Ok(simplified);
  }
  // A Real value at the extremum is usually the machine form of an exact
  // integer/rational (e.g. Sin[x]/3 at its minimum is -0.3333… = -1/3). Try to
  // recover that exact value, falling back to the Real when none is close.
  if let Expr::Real(v) = &simplified {
    return Ok(minimize_recognize_exact(*v));
  }
  if let Expr::FunctionCall { name, .. } = &simplified
    && name == "Rational"
  {
    return Ok(simplified);
  }

  // Try numerical evaluation to recognize exact integer/rational value
  if let Some(num_val) = minimize_try_f64(&simplified) {
    return Ok(minimize_recognize_exact(num_val));
  }

  Ok(simplified)
}

/// Try to get f64 from an Expr (for comparison).
fn minimize_try_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        if *d != 0 {
          Some(*n as f64 / *d as f64)
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => {
      if let Ok(n_result) = crate::functions::math_ast::n_ast(&[expr.clone()]) {
        match n_result {
          Expr::Real(r) => Some(r),
          Expr::Integer(n) => Some(n as f64),
          _ => None,
        }
      } else {
        None
      }
    }
  }
}

/// Find roots of a univariate polynomial given its integer coefficients.
/// coeffs[i] = coefficient of x^i.
/// Returns real roots as exact Expr values.
fn minimize_poly_roots_int(coeffs: &[i128], var: &str) -> Vec<Expr> {
  let _ = var; // variable name not needed for roots computation
  let degree = coeffs.len().saturating_sub(1);
  let mut roots = Vec::new();

  match degree {
    0 => {
      // Constant: no roots (if constant != 0) or all x (if 0)
    }
    1 => {
      // a*x + b = 0 → x = -b/a
      let a = coeffs[1];
      let b = coeffs[0];
      if a != 0 {
        let (num, den) = reduce_fraction(-b, a);
        let root = if den == 1 {
          Expr::Integer(num)
        } else {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num), Expr::Integer(den)].into(),
          }
        };
        roots.push(root);
      }
    }
    2 => {
      // a*x^2 + b*x + c = 0
      let a = coeffs[2];
      let b = coeffs[1];
      let c = coeffs[0];
      if a != 0 {
        let disc = b * b - 4 * a * c;
        if disc >= 0 {
          let (sqrt_out, sqrt_in) = simplify_sqrt_parts(disc);
          if sqrt_in == 1 {
            // Perfect square
            let (n1, d1) = reduce_fraction(-b - sqrt_out, 2 * a);
            let (n2, d2) = reduce_fraction(-b + sqrt_out, 2 * a);
            roots.push(if d1 == 1 {
              Expr::Integer(n1)
            } else {
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(n1), Expr::Integer(d1)].into(),
              }
            });
            if n1 != n2 || d1 != d2 {
              roots.push(if d2 == 1 {
                Expr::Integer(n2)
              } else {
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(n2), Expr::Integer(d2)].into(),
                }
              });
            }
          } else {
            // Irrational roots: (-b ± sqrt_out * √sqrt_in) / (2a)
            let g =
              gcd_i128(gcd_i128((-b).abs(), sqrt_out.abs()), (2 * a).abs())
                .abs();
            let nb = -b / g;
            let so = sqrt_out / g;
            let den = 2 * a / g;
            let (nb, so, den) = if den < 0 {
              (-nb, -so, -den)
            } else {
              (nb, so, den)
            };
            let sqrt_part = if so == 1 {
              make_sqrt(Expr::Integer(sqrt_in))
            } else {
              multiply_exprs(
                &Expr::Integer(so),
                &make_sqrt(Expr::Integer(sqrt_in)),
              )
            };
            for sign_minus in [true, false] {
              // When nb == 0 and den != 1 and so == 1, use Sqrt[sqrt_in/den^2]
              // to produce canonical form like Sqrt[3/2] instead of Sqrt[6]/2
              if nb == 0 && den != 1 && so == 1 {
                let rational_arg =
                  crate::functions::math_ast::make_rational(sqrt_in, den * den);
                if let Ok(simplified) =
                  crate::functions::math_ast::sqrt_ast(&[rational_arg])
                {
                  let root = if sign_minus {
                    negate_expr(&simplified)
                  } else {
                    simplified
                  };
                  roots.push(root);
                  continue;
                }
              }
              let num = if nb == 0 {
                if sign_minus {
                  negate_expr(&sqrt_part)
                } else {
                  sqrt_part.clone()
                }
              } else {
                let nb_expr = Expr::Integer(nb);
                Expr::BinaryOp {
                  op: if sign_minus {
                    BinaryOperator::Minus
                  } else {
                    BinaryOperator::Plus
                  },
                  left: Box::new(nb_expr),
                  right: Box::new(sqrt_part.clone()),
                }
              };
              let root = if den == 1 {
                num
              } else {
                Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(num),
                  right: Box::new(Expr::Integer(den)),
                }
              };
              roots.push(simplify(root));
            }
          }
        }
        // disc < 0: complex roots, no real roots
      }
    }
    3 => {
      // Try constant term = 0 (x is a factor)
      if coeffs[0] == 0 {
        roots.push(Expr::Integer(0));
        // Remaining: coeffs[3]*x^2 + coeffs[2]*x + coeffs[1]
        let sub_roots =
          minimize_poly_roots_int(&[coeffs[1], coeffs[2], coeffs[3]], var);
        for r in sub_roots {
          if !roots.iter().any(|existing| {
            minimize_try_f64(&r)
              .zip(minimize_try_f64(existing))
              .is_some_and(|(a, b)| (a - b).abs() < 1e-12)
          }) {
            roots.push(r);
          }
        }
        return roots;
      }

      // Rational root theorem: try ±(factors of coeffs[0]) / (factors of coeffs[3])
      let a = coeffs[3];
      let d = coeffs[0];
      // Collect actual divisors
      let mut divs_d: Vec<i128> = Vec::new();
      for i in 1i128..=(d.abs()) {
        if d % i == 0 {
          divs_d.push(i);
        }
      }
      let mut divs_a: Vec<i128> = Vec::new();
      for i in 1i128..=(a.abs()) {
        if a % i == 0 {
          divs_a.push(i);
        }
      }
      'outer: for &p in &divs_d {
        for &q in &divs_a {
          for &sign in &[1i128, -1i128] {
            let r = sign * p;
            let q_val = q;
            // Test if r/q is a root: a*(r/q)^3 + b*(r/q)^2 + c*(r/q) + d == 0
            // Multiply through by q^3: a*r^3 + b*r^2*q + c*r*q^2 + d*q^3 == 0
            let val = a * r * r * r
              + coeffs[2] * r * r * q_val
              + coeffs[1] * r * q_val * q_val
              + d * q_val * q_val * q_val;
            if val == 0 {
              let (rn, rd) = reduce_fraction(r, q_val);
              let root = if rd == 1 {
                Expr::Integer(rn)
              } else {
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(rn), Expr::Integer(rd)].into(),
                }
              };
              roots.push(root);

              // Polynomial division to get quadratic
              // Divide a*x^3 + b*x^2 + c*x + d by (q*x - r)
              // Synthetic division with root = r/q
              // q1 = a
              // q2 = a*(r/q) + b = (a*r + b*q)/q
              // q3 = q2*(r/q) + c = ...
              // Multiply through: coefficients of (a*x^2 + (a*r/q + b)*x + ...) * q
              // Use polynomial long division:
              // (a*x^3 + b*x^2 + c*x + d) / (x - r/q)
              // = a*x^2 + (a*r/q + b)*x + (a*(r/q)^2 + b*(r/q) + c)
              // Multiply by q^2 to get integer coefficients:
              // a*q^2 * x^2 + (a*r*q + b*q^2)*x + (a*r^2 + b*r*q + c*q^2)
              // But we want integer coefficients, divide by gcd
              let qa = a;
              let qb = a * r / q_val + coeffs[2];
              let qc =
                a * r * r / (q_val * q_val) + coeffs[2] * r / q_val + coeffs[1];
              // Only proceed if exact (no fractions)
              if a * r % q_val == 0 && a * r * r % (q_val * q_val) == 0 {
                let sub_roots = minimize_poly_roots_int(&[qc, qb, qa], var);
                for sr in sub_roots {
                  if !roots.iter().any(|existing| {
                    minimize_try_f64(&sr)
                      .zip(minimize_try_f64(existing))
                      .is_some_and(|(a, b)| (a - b).abs() < 1e-12)
                  }) {
                    roots.push(sr);
                  }
                }
              }
              break 'outer;
            }
          }
        }
      }
    }
    _ => {
      // Higher degree: try numerical root finding with multiple starting points
      // We'll handle this in the caller via numerical fallback
    }
  }
  roots
}

/// Reduce fraction n/d to lowest terms with positive denominator.
fn reduce_fraction(n: i128, d: i128) -> (i128, i128) {
  if d == 0 {
    return (n, d);
  }
  let g = gcd_i128(n.abs(), d.abs()).abs();
  let mut rn = n / g;
  let mut rd = d / g;
  if rd < 0 {
    rn = -rn;
    rd = -rd;
  }
  (rn, rd)
}

/// Extract integer polynomial coefficients of `poly` in `var`.
/// Returns Some(coeffs) where coeffs[i] = coefficient of var^i.
/// Returns None if not a polynomial with integer coefficients.
fn minimize_extract_int_coeffs(poly: &Expr, var: &str) -> Option<Vec<i128>> {
  let expanded = expand_and_combine(poly);
  // A negative max power means a Laurent/rational expression (e.g. 1/x^2), not
  // a plain polynomial; bail out (also avoids `degree + 1` overflowing when the
  // sentinel -1 is cast to usize).
  let degree_raw = max_power_int(&expanded, var)?;
  if degree_raw < 0 {
    return None;
  }
  let degree = degree_raw as usize;
  let terms = collect_additive_terms(&expanded);
  // Pre-check: ensure all terms are polynomial in var. A negative power is
  // either the sentinel -1 (var appears non-polynomially, e.g. E^x, Sin[x]) or
  // a genuine negative exponent (x^-2); neither is a plain polynomial.
  for term in &terms {
    let (power, _) = term_var_power_and_coeff(term, var);
    if power < 0 {
      return None;
    }
  }
  let mut coeffs = vec![0i128; degree + 1];
  for d in 0..=degree {
    let mut sum = 0i128;
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d as i128) {
        match c {
          Expr::Integer(n) => sum += n,
          _ => return None, // non-integer coefficient
        }
      }
    }
    coeffs[d] = sum;
  }
  Some(coeffs)
}

/// Check if a polynomial f in var is bounded below.
/// Returns Some(true) if bounded below, Some(false) if not, None if unknown.
/// Only handles true polynomials with integer coefficients.
fn minimize_poly_bounded_below(f: &Expr, var: &str) -> Option<bool> {
  // Only use polynomial analysis for verified polynomials with integer coefficients
  let expanded = expand_and_combine(f);
  let degree = max_power_int(&expanded, var)?;
  if degree == 0 {
    // Might be a constant OR a non-polynomial term like E^x with "degree 0"
    // We can't distinguish here, return None to use numerical check
    return None;
  }
  // Verify the function is truly a polynomial by checking that all
  // integer polynomial coefficients can be extracted
  let coeffs = minimize_extract_int_coeffs(&expanded, var)?;
  if coeffs.len() < 2 {
    return None;
  }
  let d = coeffs.len() - 1;
  let lead_coeff = coeffs[d];

  if d % 2 == 1 {
    // Odd degree: always unbounded in both directions
    Some(false)
  } else if lead_coeff > 0 {
    Some(true)
  } else if lead_coeff < 0 {
    Some(false)
  } else {
    None
  }
}

/// Check if f is bounded below by evaluating numerically at large values.
///
/// A single large test point is not enough: a linearly unbounded objective such
/// as `-Abs[x]` reaches only `-1e6` at `x = 1e6`, so a fixed `-1e8` threshold
/// would wrongly call it bounded. Instead probe a sequence of increasing
/// magnitudes in each direction; if the objective is still strictly decreasing
/// and already deeply negative at the largest magnitude, it runs off to
/// -Infinity (e.g. `Minimize[-Abs[x], x]` -> {-Infinity, {x -> -Infinity}}).
fn minimize_bounded_below_numerical(f: &Expr, var: &str) -> bool {
  let mags: &[f64] = &[1e2, 1e4, 1e6, 1e8];
  for &sign in &[-1.0_f64, 1.0] {
    let mut vals = Vec::with_capacity(mags.len());
    for &m in mags {
      let substituted =
        crate::syntax::substitute_variable(f, var, &Expr::Real(sign * m));
      match crate::evaluator::evaluate_expr_to_expr(&substituted)
        .ok()
        .and_then(|e| minimize_try_f64(&e))
      {
        Some(val) => vals.push(val),
        None => {
          vals.clear();
          break;
        }
      }
    }
    if vals.len() == mags.len() {
      let last = vals[vals.len() - 1];
      let prev = vals[vals.len() - 2];
      if last < prev && last < -1e6 {
        return false;
      }
    }
  }
  true
}

/// Build the -Infinity result for minimize (no minimum exists).
fn minimize_neg_infinity_result(vars: &[String], maximize: bool) -> Expr {
  let inf_val = if maximize {
    // Maximize returns {Infinity, {x -> Infinity}}
    Expr::Identifier("Infinity".to_string())
  } else {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  };
  let x_val = if maximize {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  } else {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  };
  let rules: Vec<Expr> = vars
    .iter()
    .map(|v| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(x_val.clone()),
    })
    .collect();
  Expr::List(vec![inf_val, Expr::List(rules.into())].into())
}

/// Single-variable unconstrained minimize.
/// True if `e` is a genuinely complex value: it contains the imaginary unit
/// (via `contains_complex`) or a `Complex[re, im]` node with a nonzero imaginary
/// part (which `contains_complex` alone does not detect).
fn minimize_cp_is_complex(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      let im_zero = matches!(&args[1], Expr::Integer(0))
        || matches!(&args[1], Expr::Real(r) if *r == 0.0);
      !im_zero || args.iter().any(minimize_cp_is_complex)
    }
    Expr::FunctionCall { args, .. } => args.iter().any(minimize_cp_is_complex),
    Expr::List(items) => items.iter().any(minimize_cp_is_complex),
    Expr::BinaryOp { left, right, .. } => {
      minimize_cp_is_complex(left) || minimize_cp_is_complex(right)
    }
    Expr::UnaryOp { operand, .. } => minimize_cp_is_complex(operand),
    _ => contains_complex(e),
  }
}

/// Collect every `Abs[g]` subexpression argument `g` that depends on `var`.
fn collect_abs_args(e: &Expr, var: &str, out: &mut Vec<Expr>) {
  match e {
    Expr::FunctionCall { name, args } => {
      if name == "Abs"
        && args.len() == 1
        && !crate::functions::calculus_ast::is_constant_wrt(&args[0], var)
      {
        out.push(args[0].clone());
      }
      for a in args.iter() {
        collect_abs_args(a, var, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_abs_args(left, var, out);
      collect_abs_args(right, var, out);
    }
    Expr::UnaryOp { operand, .. } => collect_abs_args(operand, var, out),
    Expr::List(items) => {
      for it in items.iter() {
        collect_abs_args(it, var, out);
      }
    }
    _ => {}
  }
}

/// The `var` values where an `Abs` argument in `f` vanishes — the kink points of
/// a piecewise-linear/convex objective, which are candidate (non-smooth) minima.
fn minimize_abs_breakpoints(f: &Expr, var: &str) -> Vec<Expr> {
  let mut abs_args = Vec::new();
  collect_abs_args(f, var, &mut abs_args);
  let mut points = Vec::new();
  for g in abs_args {
    let eq = Expr::Comparison {
      operands: vec![g, Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    };
    let solved = solve_ast(&[eq, Expr::Identifier(var.to_string())]);
    if let Ok(Expr::List(sol_sets)) = &solved {
      for sol_set in sol_sets.iter() {
        if let Expr::List(rules) = sol_set {
          for rule in rules.iter() {
            if let Expr::Rule { replacement, .. } = rule {
              points.push((**replacement).clone());
            }
          }
        }
      }
    }
  }
  points
}

fn minimize_single_var(
  f: &Expr,
  var: &str,
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // For maximize, negate f and negate the result at the end
  let f_inner = if maximize {
    simplify(negate_expr(f))
  } else {
    f.clone()
  };

  // Compute symbolic derivative
  let df = simplify(crate::functions::calculus_ast::differentiate_expr(
    &f_inner, var,
  )?);

  // Check if f is bounded below (polynomial check first)
  let bounded = if let Some(b) = minimize_poly_bounded_below(&f_inner, var) {
    b
  } else {
    // Non-polynomial: try numerical check
    minimize_bounded_below_numerical(&f_inner, var)
  };

  if !bounded {
    let head = if maximize { "Maximize" } else { "Minimize" };
    let kind = if maximize { "maximum" } else { "minimum" };
    crate::emit_message(&format!(
      "{}::natt: The {} is not attained at any point satisfying the given constraints.",
      head, kind
    ));
    return Ok(minimize_neg_infinity_result(&[var.to_string()], maximize));
  }

  // Find critical points: solve df == 0
  let mut critical_points =
    minimize_find_critical_points_1d(&df, var, &f_inner)?;

  // Abs[g(x)] has a non-smooth kink where g(x) == 0; the derivative-based
  // search never sees it (d/dx Abs = Sign is never zero). Add those breakpoints
  // as candidate minimizers so e.g. Minimize[Abs[x - 3], x] -> {0, {x -> 3}}.
  // Only keep a breakpoint that is a genuine local minimum (the objective does
  // not dip below it in a small neighbourhood); this rejects concave kinks such
  // as the maximum of -Abs[x], where the true minimum is unbounded.
  let eval_num = |xval: &Expr| -> Option<f64> {
    minimize_eval_exact(&f_inner, var, xval)
      .ok()
      .and_then(|e| minimize_try_f64(&e))
  };
  for bp in minimize_abs_breakpoints(&f_inner, var) {
    let bp_str = expr_to_string(&bp);
    if critical_points.iter().any(|c| expr_to_string(c) == bp_str) {
      continue;
    }
    let (Some(x0), Some(v0)) = (minimize_try_f64(&bp), eval_num(&bp)) else {
      continue;
    };
    let left = eval_num(&Expr::Real(x0 - 1e-4));
    let right = eval_num(&Expr::Real(x0 + 1e-4));
    let is_local_min = left.is_some_and(|l| l >= v0 - 1e-9)
      && right.is_some_and(|r| r >= v0 - 1e-9);
    if is_local_min {
      critical_points.push(bp);
    }
  }

  if critical_points.is_empty() {
    // Bounded function with no critical points: return unevaluated
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![f.clone(), Expr::Identifier(var.to_string())].into(),
    });
  }

  // Evaluate f at each critical point, find the minimum
  let mut best_val: Option<f64> = None;
  let mut best_exact: Option<Expr> = None;
  let mut best_x: Option<Expr> = None;

  for cp in &critical_points {
    // Real-variable optimization ignores complex critical points (e.g. x = ±I
    // among the roots of x^4 == 1 for Minimize[x^2 + 1/x^2, x]).
    if minimize_cp_is_complex(cp) {
      continue;
    }
    let fval_exact = minimize_eval_exact(&f_inner, var, cp)?;
    let fval_num = minimize_try_f64(&fval_exact);

    if let Some(fv) = fval_num {
      let is_better = match best_val {
        None => true,
        Some(bv) => fv < bv,
      };
      if is_better {
        best_val = Some(fv);
        best_exact = Some(fval_exact);
        best_x = Some(cp.clone());
      }
    }
  }

  let (min_val, min_x) = match (best_exact, best_x) {
    (Some(v), Some(x)) => (v, x),
    _ => {
      return Ok(Expr::FunctionCall {
        name: func_name.to_string(),
        args: vec![f.clone(), Expr::Identifier(var.to_string())].into(),
      });
    }
  };

  // For maximize, negate the value back
  let result_val = if maximize {
    simplify(negate_expr(&min_val))
  } else {
    min_val
  };

  let rule = Expr::Rule {
    pattern: Box::new(Expr::Identifier(var.to_string())),
    replacement: Box::new(min_x),
  };
  Ok(Expr::List(
    vec![result_val, Expr::List(vec![rule].into())].into(),
  ))
}

/// Find critical points of f' = 0 in one variable.
fn minimize_find_critical_points_1d(
  df: &Expr,
  var: &str,
  f: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  // Try polynomial root finding
  let expanded_df = expand_and_combine(df);

  if let Some(coeffs) = minimize_extract_int_coeffs(&expanded_df, var) {
    let roots = minimize_poly_roots_int(&coeffs, var);
    if !roots.is_empty() || matches!(coeffs.len(), 0 | 1) {
      return Ok(roots);
    }
  }

  // Fallback: try Solve[df == 0, var]
  let df_eq = Expr::Comparison {
    operands: vec![df.clone(), Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };
  match solve_ast(&[df_eq, Expr::Identifier(var.to_string())]) {
    Ok(solutions) => {
      if let Expr::List(sol_sets) = &solutions {
        // If Solve returned unevaluated pieces or empty list, try numerical
        if sol_sets.iter().any(|s| {
          !matches!(s, Expr::List(_))
            || matches!(s, Expr::FunctionCall { name, .. } if name == "Solve")
        }) {
          return minimize_find_critical_points_numerical(f, var);
        }
        // If Solve returned empty (no solutions found), also try numerical
        // as it might have incorrectly classified the equation
        if sol_sets.is_empty() {
          return minimize_find_critical_points_numerical(f, var);
        }
        let mut roots = Vec::new();
        for sol_set in sol_sets {
          if let Expr::List(rules) = sol_set {
            for rule in rules {
              if let Expr::Rule { replacement, .. } = rule {
                roots.push(*replacement.clone());
              }
            }
          }
        }
        // If Solve found no actual roots (all empty rule sets), try numerical
        if roots.is_empty() {
          return minimize_find_critical_points_numerical(f, var);
        }
        return Ok(roots);
      }
      // Unevaluated Solve result - try numerical
      minimize_find_critical_points_numerical(f, var)
    }
    Err(_) => minimize_find_critical_points_numerical(f, var),
  }
}

/// Numerically find critical points of f using Newton's method with multiple starts.
fn minimize_find_critical_points_numerical(
  f: &Expr,
  var: &str,
) -> Result<Vec<Expr>, InterpreterError> {
  let df =
    simplify(crate::functions::calculus_ast::differentiate_expr(f, var)?);

  let starts: &[f64] = &[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0];
  let mut roots: Vec<f64> = Vec::new();
  let tol = 1e-10;

  for &x0 in starts {
    let mut x = x0;
    for _ in 0..100 {
      let gval = find_root_eval_at(&df, var, x).unwrap_or(f64::NAN);
      if gval.is_nan() || gval.is_infinite() {
        break;
      }
      if gval.abs() < tol {
        break;
      }
      let hval = {
        let h = 1e-7;
        let g1 = find_root_eval_at(&df, var, x + h).unwrap_or(f64::NAN);
        if g1.is_nan() {
          break;
        }
        (g1 - gval) / h
      };
      if hval.abs() < 1e-30 {
        break;
      }
      x -= gval / hval;
      if !x.is_finite() {
        break;
      }
    }
    if x.is_finite() {
      let gval = find_root_eval_at(&df, var, x).unwrap_or(f64::INFINITY);
      if gval.abs() < 1e-6 {
        // Check if this root is already found
        if !roots.iter().any(|&r| (r - x).abs() < 1e-6) {
          roots.push(x);
        }
      }
    }
  }

  Ok(roots.into_iter().map(minimize_recognize_exact).collect())
}

/// Multi-variable unconstrained minimize.
fn minimize_multi_var(
  f: &Expr,
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  let f_inner = if maximize {
    simplify(negate_expr(f))
  } else {
    f.clone()
  };

  let n = vars.len();

  // Compute symbolic gradient
  let mut grad: Vec<Expr> = Vec::new();
  for var in vars {
    let dfi =
      crate::functions::calculus_ast::differentiate_expr(&f_inner, var)?;
    grad.push(simplify(dfi));
  }

  // Try to solve the gradient system symbolically
  // For independent linear equations in each variable, solve separately
  let mut solutions: Vec<Option<Expr>> = vec![None; n];
  let mut all_solved = true;

  for (i, var) in vars.iter().enumerate() {
    let grad_eq = Expr::Comparison {
      operands: vec![grad[i].clone(), Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    };
    match solve_ast(&[grad_eq, Expr::Identifier(var.clone())]) {
      Ok(sol) => {
        if let Expr::List(sol_sets) = &sol
          && sol_sets.len() == 1
          && let Some(Expr::List(rules)) = sol_sets.first()
          && rules.len() == 1
          && let Some(Expr::Rule { replacement, .. }) = rules.first()
        {
          solutions[i] = Some(*replacement.clone());
          continue;
        }
        all_solved = false;
        break;
      }
      Err(_) => {
        all_solved = false;
        break;
      }
    }
  }

  if all_solved {
    let vals: Vec<Expr> = solutions.into_iter().flatten().collect();
    if vals.len() == n {
      // Evaluate f at the critical point
      let fval = minimize_eval_exact_multi(&f_inner, vars, &vals)?;
      let result_val = if maximize {
        simplify(negate_expr(&fval))
      } else {
        fval
      };
      let rules: Vec<Expr> = vars
        .iter()
        .zip(vals.iter())
        .map(|(v, val)| Expr::Rule {
          pattern: Box::new(Expr::Identifier(v.clone())),
          replacement: Box::new(val.clone()),
        })
        .collect();
      return Ok(Expr::List(
        vec![result_val, Expr::List(rules.into())].into(),
      ));
    }
  }

  // Fallback: numerical multi-variable minimize (gradient descent from origin)
  let mut x: Vec<f64> = vec![0.0; n];
  let tol = 1e-12;
  let max_iter = 500;

  for _ in 0..max_iter {
    let mut grad_vals = vec![0.0f64; n];
    let mut grad_norm = 0.0f64;
    for i in 0..n {
      // For multi-var we need proper substitution, use eval_at_multi
      let mut gexpr = grad[i].clone();
      for (j, vj) in vars.iter().enumerate() {
        gexpr =
          crate::syntax::substitute_variable(&gexpr, vj, &Expr::Real(x[j]));
      }
      let gval = crate::evaluator::evaluate_expr_to_expr(&gexpr)
        .ok()
        .and_then(|e| minimize_try_f64(&e))
        .unwrap_or(0.0);
      grad_vals[i] = gval;
      grad_norm += gval * gval;
    }
    grad_norm = grad_norm.sqrt();
    if grad_norm < tol {
      break;
    }

    // Gradient descent step
    let alpha = 0.01 / (1.0 + grad_norm);
    for i in 0..n {
      x[i] -= alpha * grad_vals[i];
    }
  }

  // Evaluate f at the numerical minimum
  let mut fexpr = f_inner.clone();
  for (i, var) in vars.iter().enumerate() {
    fexpr = crate::syntax::substitute_variable(&fexpr, var, &Expr::Real(x[i]));
  }
  let fval = crate::evaluator::evaluate_expr_to_expr(&fexpr)
    .ok()
    .and_then(|e| minimize_try_f64(&e))
    .unwrap_or(f64::NAN);

  if !fval.is_finite() {
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![
        f.clone(),
        Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
      ]
      .into(),
    });
  }

  let result_val = if maximize {
    Expr::Real(-fval)
  } else {
    Expr::Real(fval)
  };
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(v, &val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(Expr::Real(val)),
    })
    .collect();
  Ok(Expr::List(
    vec![result_val, Expr::List(rules.into())].into(),
  ))
}

/// Enumerate integer solutions of a bounded linear system for Solve[..., Integers].
///
/// Returns Some(list-of-solutions) when the system reduces to a finite integer
/// box (after deriving implicit upper bounds from equalities with non-negative
/// coefficients). Returns None when the structure isn't supported, letting the
/// caller fall back to filter-by-integer-replacement.
fn try_solve_integer_bounded(
  constraints_arg: &Expr,
  vars_arg: &Expr,
) -> Option<Expr> {
  let vars_exprs = if let Expr::List(v) = vars_arg {
    v
  } else {
    return None;
  };
  let vars: Vec<String> = vars_exprs
    .iter()
    .filter_map(|v| {
      if let Expr::Identifier(name) = v {
        Some(name.clone())
      } else {
        None
      }
    })
    .collect();
  if vars.len() != vars_exprs.len() || vars.len() < 2 {
    return None;
  }

  let raw = match constraints_arg {
    Expr::List(items) => items.iter().cloned().collect::<Vec<_>>(),
    _ => vec![constraints_arg.clone()],
  };
  let constraints = flatten_and_constraints(&raw);

  let mut equalities: Vec<(Vec<f64>, f64)> = Vec::new();
  let mut other_ineqs: Vec<(Vec<f64>, f64, i32, bool)> = Vec::new();
  let mut lb: Vec<f64> = vec![f64::NEG_INFINITY; vars.len()];
  let mut ub: Vec<f64> = vec![f64::INFINITY; vars.len()];

  for con in &constraints {
    let (coeffs, rhs, sense) = minimize_extract_linear_constraint(con, &vars)?;
    // A strict `>`/`<` excludes the boundary. For integer enumeration that
    // matters when the boundary is itself an integer (e.g. x > 0 means x >= 1),
    // so nudge a strict bound by a tiny epsilon: a lower bound up, an upper
    // bound down. The nudge only crosses an integer when the bound IS that
    // integer, leaving non-integer bounds unaffected. (Previously strict and
    // non-strict were treated identically, so Solve[x+y==5 && x>0 && y>0, ...,
    // Integers] wrongly included x=0 and y=0.)
    let strict = constraint_is_strict(con);
    let eps = 1e-9;
    let nonzero: Vec<usize> = coeffs
      .iter()
      .enumerate()
      .filter(|(_, c)| c.abs() > 1e-12)
      .map(|(i, _)| i)
      .collect();
    match sense {
      0 => equalities.push((coeffs, rhs)),
      1 if nonzero.len() == 1 => {
        let i = nonzero[0];
        let bound = rhs / coeffs[i];
        if coeffs[i] > 0.0 {
          // lower bound
          let bound = if strict { bound + eps } else { bound };
          if bound > lb[i] {
            lb[i] = bound;
          }
        } else {
          // upper bound
          let bound = if strict { bound - eps } else { bound };
          if bound < ub[i] {
            ub[i] = bound;
          }
        }
      }
      -1 if nonzero.len() == 1 => {
        let i = nonzero[0];
        let bound = rhs / coeffs[i];
        if coeffs[i] > 0.0 {
          // upper bound
          let bound = if strict { bound - eps } else { bound };
          if bound < ub[i] {
            ub[i] = bound;
          }
        } else {
          // lower bound
          let bound = if strict { bound + eps } else { bound };
          if bound > lb[i] {
            lb[i] = bound;
          }
        }
      }
      _ => other_ineqs.push((coeffs, rhs, sense, strict)),
    }
  }

  if equalities.is_empty() {
    return None;
  }

  // Derive implicit upper bounds: for an equality sum(c_i * x_i) == T with all
  // c_i >= 0, all lb_i >= 0, and T >= 0, each x_i with c_i > 0 satisfies
  // x_i <= T / c_i.
  for (coeffs, rhs) in &equalities {
    if coeffs.iter().all(|&c| c >= 0.0)
      && lb.iter().all(|&b| b >= 0.0)
      && *rhs >= 0.0
    {
      for (i, &c) in coeffs.iter().enumerate() {
        if c > 0.0 {
          let bound = rhs / c;
          if bound < ub[i] {
            ub[i] = bound;
          }
        }
      }
    }
  }

  if lb.iter().any(|&b| b.is_infinite()) || ub.iter().any(|&b| b.is_infinite())
  {
    return None;
  }

  let lb_int: Vec<i64> = lb.iter().map(|&b| b.ceil() as i64).collect();
  let ub_int: Vec<i64> = ub.iter().map(|&b| b.floor() as i64).collect();

  // Bound the enumeration size to avoid pathological inputs.
  let mut total: i128 = 1;
  for i in 0..vars.len() {
    let range = (ub_int[i] - lb_int[i] + 1) as i128;
    if range <= 0 {
      return Some(Expr::List(Vec::new().into()));
    }
    total = total.saturating_mul(range);
    if total > 1_000_000 {
      return None;
    }
  }

  let satisfies = |x: &[i64]| -> bool {
    for (coeffs, rhs) in &equalities {
      let sum: f64 = coeffs
        .iter()
        .zip(x.iter())
        .map(|(&c, &xi)| c * xi as f64)
        .sum();
      if (sum - rhs).abs() > 1e-6 {
        return false;
      }
    }
    for (coeffs, rhs, sense, strict) in &other_ineqs {
      let sum: f64 = coeffs
        .iter()
        .zip(x.iter())
        .map(|(&c, &xi)| c * xi as f64)
        .sum();
      let ok = match (sense, strict) {
        (1, false) => sum >= *rhs - 1e-6,
        (1, true) => sum > *rhs + 1e-6,
        (-1, false) => sum <= *rhs + 1e-6,
        (-1, true) => sum < *rhs - 1e-6,
        _ => true,
      };
      if !ok {
        return false;
      }
    }
    true
  };

  // Lexicographic enumeration with the first variable as the slowest index,
  // matching Wolfram's solution ordering.
  let n = vars.len();
  let mut current = lb_int.clone();
  let mut solutions: Vec<Vec<i64>> = Vec::new();
  loop {
    if satisfies(&current) {
      solutions.push(current.clone());
    }
    let mut i = n;
    let mut carried_out = true;
    while i > 0 {
      i -= 1;
      current[i] += 1;
      if current[i] <= ub_int[i] {
        carried_out = false;
        break;
      }
      current[i] = lb_int[i];
    }
    if carried_out {
      break;
    }
  }

  let sol_exprs: Vec<Expr> = solutions
    .into_iter()
    .map(|sol| {
      let rules: Vec<Expr> = vars
        .iter()
        .zip(sol.iter())
        .map(|(v, &val)| Expr::Rule {
          pattern: Box::new(Expr::Identifier(v.clone())),
          replacement: Box::new(Expr::Integer(val as i128)),
        })
        .collect();
      Expr::List(rules.into())
    })
    .collect();
  Some(Expr::List(sol_exprs.into()))
}

/// Flatten And[a, b, c, ...] recursively into a flat list of constraints.
fn flatten_and_constraints(constraints: &[Expr]) -> Vec<Expr> {
  let mut result = Vec::new();
  for c in constraints {
    flatten_and_expr(c, &mut result);
  }
  result
}

fn flatten_and_expr(expr: &Expr, result: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        flatten_and_expr(arg, result);
      }
    }
    Expr::List(items) => {
      for item in items {
        flatten_and_expr(item, result);
      }
    }
    // Split chained comparisons like 0 <= x <= 30 into pairwise:
    // 0 <= x and x <= 30
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() > 2 => {
      for i in 0..operators.len() {
        result.push(Expr::Comparison {
          operands: vec![operands[i].clone(), operands[i + 1].clone()],
          operators: vec![operators[i]],
        });
      }
    }
    _ => result.push(expr.clone()),
  }
}

/// Constrained minimization.
fn minimize_constrained(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // Flatten And[...] chains into individual constraints
  let constraints = flatten_and_constraints(constraints);

  let f_inner = if maximize {
    simplify(negate_expr(f))
  } else {
    f.clone()
  };

  // When Maximize can't solve the problem the sub-solvers echo the call
  // unevaluated, but embed the internally-negated objective `f_inner`. Restore
  // the user's original objective so the echo matches wolframscript
  // (Maximize[{x*y, …}] rather than Maximize[{-(x*y), …}]).
  let restore = |result: Expr| -> Expr {
    if maximize
      && let Expr::FunctionCall { name, .. } = &result
      && name == func_name
    {
      return substitute_expr(&result, &f_inner, f);
    }
    result
  };

  // Try ILP if any Element[x, Integers] constraint is present
  if constraints
    .iter()
    .any(|c| matches!(c, Expr::FunctionCall { name, .. } if name == "Element"))
    && let Some(result) =
      minimize_try_ilp(&f_inner, &constraints, vars, maximize, func_name)?
  {
    return Ok(restore(result));
  }

  // Single variable with simple bound constraints
  if vars.len() == 1 {
    let var = &vars[0];
    return Ok(restore(minimize_constrained_1d(
      &f_inner,
      &constraints,
      var,
      maximize,
      func_name,
    )?));
  }

  // Multi-variable: try linear programming for linear constraints + linear/quadratic objective
  Ok(restore(minimize_constrained_nd(
    &f_inner,
    &constraints,
    vars,
    maximize,
    func_name,
  )?))
}

/// Try Integer Linear Programming. Returns Some(result) if ILP was solved, None if unsupported.
fn minimize_try_ilp(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Option<Expr>, InterpreterError> {
  use std::collections::HashSet;

  // Walk an `Element[…, Integers]` subject and collect identifier leaves.
  // `Element` may carry the symbols as: a single Identifier, a List, an
  // `Alternatives` FunctionCall, or the BinaryOp Alternatives chain that
  // `x | y | z` parses to.
  fn collect_element_symbols(e: &Expr, out: &mut HashSet<String>) {
    match e {
      Expr::Identifier(var) => {
        out.insert(var.clone());
      }
      Expr::List(items) => {
        for a in items.iter() {
          collect_element_symbols(a, out);
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Alternatives" || name == "List" =>
      {
        for a in args.iter() {
          collect_element_symbols(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Alternatives,
        left,
        right,
      } => {
        collect_element_symbols(left, out);
        collect_element_symbols(right, out);
      }
      _ => {}
    }
  }

  // Separate Element[x, Integers] from actual constraints.
  let mut integer_vars: HashSet<String> = HashSet::new();
  let mut actual_constraints: Vec<&Expr> = Vec::new();
  for c in constraints {
    match c {
      Expr::FunctionCall { name, args }
        if name == "Element"
          && args.len() == 2
          && matches!(&args[1], Expr::Identifier(d) if d == "Integers") =>
      {
        collect_element_symbols(&args[0], &mut integer_vars);
        // Don't add to actual_constraints
      }
      _ => actual_constraints.push(c),
    }
  }

  // All problem variables must be integer-constrained
  if !vars.iter().all(|v| integer_vars.contains(v)) {
    return Ok(None);
  }

  // Extract linear objective coefficients
  let obj_coeffs = match minimize_extract_linear_expr(f, vars) {
    Some((c, _)) => c,
    None => return Ok(None), // non-linear objective
  };

  // Extract linear constraints: one equality + bound inequalities
  let mut equalities: Vec<(Vec<f64>, f64)> = Vec::new(); // (coeffs, rhs)
  let mut lb: Vec<f64> = vec![f64::NEG_INFINITY; vars.len()]; // lower bounds
  let mut ub: Vec<f64> = vec![f64::INFINITY; vars.len()]; // upper bounds

  for con in &actual_constraints {
    if let Some((coeffs, rhs, sense)) =
      minimize_extract_linear_constraint(con, vars)
    {
      // Check if it's a simple bound on a single variable
      let nonzero: Vec<usize> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| c.abs() > 1e-12)
        .map(|(i, _)| i)
        .collect();
      match sense {
        0 => equalities.push((coeffs, rhs)), // ==
        1
          // coeffs · x >= rhs
          if nonzero.len() == 1 => {
            let i = nonzero[0];
            let bound = rhs / coeffs[i];
            if coeffs[i] > 0.0 {
              // x_i >= bound
              if bound > lb[i] {
                lb[i] = bound;
              }
            } else {
              // x_i <= -bound (flipped sign)
              let upper = rhs / coeffs[i];
              if upper < ub[i] {
                ub[i] = upper;
              }
            }
          }
        -1
          // coeffs · x <= rhs  (sense -1 means LessEqual)
          // This means: lhs - rhs <= 0, so lhs <= rhs
          // From the extraction: diff = lhs - rhs, coeffs · x + constant >= 0 for sense 1
          // For sense -1: diff = lhs - rhs, coeffs · x + constant <= 0
          // i.e., sum(coeffs[i] * x[i]) <= -constant = rhs
          if nonzero.len() == 1 => {
            let i = nonzero[0];
            if coeffs[i] > 0.0 {
              // x_i <= rhs / coeffs[i]
              let bound = rhs / coeffs[i];
              if bound < ub[i] {
                ub[i] = bound;
              }
            } else {
              // x_i >= rhs / coeffs[i] (flipped sign)
              let bound = rhs / coeffs[i];
              if bound > lb[i] {
                lb[i] = bound;
              }
            }
          }
        _ => {}
      }
    } else {
      return Ok(None); // non-linear constraint
    }
  }

  // Only support single equality constraint for DP
  if equalities.len() != 1 {
    return Ok(None);
  }
  let (eq_coeffs, eq_rhs) = &equalities[0];

  // Default lower bounds to 0 for non-negative variables
  for i in 0..vars.len() {
    if lb[i] == f64::NEG_INFINITY {
      lb[i] = 0.0;
    }
  }

  // All lower bounds must be non-negative integers for DP
  let lb_int: Vec<i64> = lb.iter().map(|&b| b.ceil() as i64).collect();
  if lb_int.iter().any(|&b| b < 0) {
    return Ok(None);
  }

  // Upper bounds (default to a large number if infinite)
  let ub_int: Vec<i64> = ub
    .iter()
    .map(|&b| {
      if b.is_infinite() {
        i64::MAX
      } else {
        b.floor() as i64
      }
    })
    .collect();

  // Scale decimal coefficients to integers.
  // Find a common scale factor that makes all coefficients integers.
  let mut all_values: Vec<f64> = eq_coeffs.to_vec();
  all_values.push(*eq_rhs);
  let scale = find_integer_scale(&all_values);
  if scale == 0 {
    return Ok(None);
  }

  let mut weights: Vec<i64> = Vec::with_capacity(vars.len());
  for &c in eq_coeffs {
    let scaled = c * scale as f64;
    let ci = scaled.round() as i64;
    if (scaled - ci as f64).abs() > 1e-4 || ci <= 0 {
      return Ok(None);
    }
    weights.push(ci);
  }
  let target_f = *eq_rhs * scale as f64;
  let target_i = target_f.round() as i64;
  if (target_f - target_i as f64).abs() > 1e-4 || target_i < 0 {
    return Ok(None);
  }
  let target = target_i as usize;

  // Shift variables by lower bounds: x_i' = x_i - lb_i
  // New target = target - sum(weights[i] * lb_i)
  let mut shifted_target = target as i64;
  for i in 0..vars.len() {
    shifted_target -= weights[i] * lb_int[i];
  }
  if shifted_target < 0 {
    // Infeasible
    return Ok(Some(minimize_neg_infinity_result(vars, maximize)));
  }
  let shifted_target = shifted_target as usize;

  // Shifted upper bounds
  let shifted_ub: Vec<i64> = ub_int
    .iter()
    .zip(lb_int.iter())
    .map(|(&u, &l)| if u == i64::MAX { i64::MAX } else { u - l })
    .collect();

  // Verify objective coefficients are non-negative integers
  let mut obj_int: Vec<i64> = Vec::with_capacity(vars.len());
  for &c in &obj_coeffs {
    let ci = c.round() as i64;
    if (c - ci as f64).abs() > 1e-8 || ci < 0 {
      return Ok(None);
    }
    obj_int.push(ci);
  }

  // Guard: if the target is too large for DP (> 10M), bail out
  if shifted_target > 10_000_000 {
    return Ok(None);
  }

  // Bounded DP: dp[t] = minimum objective to achieve shifted weight t
  // Each variable i can be used at most shifted_ub[i] times
  let n = vars.len();
  const INF: i64 = i64::MAX / 2;
  let mut dp = vec![INF; shifted_target + 1];
  // Store full assignment at each DP state for bounded tracking
  let mut dp_assign: Vec<Vec<i64>> = vec![vec![0; n]; shifted_target + 1];
  dp[0] = 0;

  for t in 1..=shifted_target {
    for i in 0..n {
      let wi = weights[i] as usize;
      if wi <= t && dp[t - wi] != INF {
        // Check upper bound: can we use one more of item i?
        if dp_assign[t - wi][i] < shifted_ub[i] {
          let new_val = dp[t - wi] + obj_int[i];
          if new_val < dp[t] {
            dp[t] = new_val;
            dp_assign[t] = dp_assign[t - wi].clone();
            dp_assign[t][i] += 1;
          }
        }
      }
    }
  }

  if dp[shifted_target] == INF {
    // Infeasible
    return Ok(Some(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![
        f.clone(),
        Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
      ]
      .into(),
    }));
  }

  // Recover variable assignments (add back lower bounds)
  let mut x = dp_assign[shifted_target].clone();
  for i in 0..n {
    x[i] += lb_int[i];
  }

  // Compute the actual objective value using the original coefficients
  let obj_val: f64 = obj_coeffs
    .iter()
    .zip(x.iter())
    .map(|(&c, &xi)| c * xi as f64)
    .sum();
  let obj_val = if maximize { -obj_val } else { obj_val };
  // If the constraint coefficients were non-integer (scale > 1),
  // Wolfram returns a Real result
  let result_val = if scale > 1 {
    Expr::Real(obj_val)
  } else {
    minimize_recognize_exact(obj_val)
  };
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(v, &val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(Expr::Integer(val as i128)),
    })
    .collect();
  Ok(Some(Expr::List(
    vec![result_val, Expr::List(rules.into())].into(),
  )))
}

/// Find a scale factor that makes all values close to integers.
/// Tries powers of 10 up to 10^6.
fn find_integer_scale(values: &[f64]) -> i64 {
  for &scale in &[1i64, 10, 100, 1_000, 10_000, 100_000, 1_000_000] {
    let all_int = values.iter().all(|&v| {
      let scaled = v * scale as f64;
      (scaled - scaled.round()).abs() < 1e-6
    });
    if all_int {
      return scale;
    }
  }
  0 // could not find a suitable scale
}

/// Extract linear expression coefficients: f = sum(coeffs[i] * vars[i]) + constant.
/// Returns None if f is not linear in vars.
fn minimize_extract_linear_expr(
  f: &Expr,
  vars: &[String],
) -> Option<(Vec<f64>, f64)> {
  let expanded = expand_and_combine(f);
  let mut coeffs = vec![0.0f64; vars.len()];

  // Check degree <= 1 in each variable
  for var in vars {
    let deg = max_power_int(&expanded, var);
    if matches!(deg, Some(d) if d > 1) {
      return None;
    }
  }

  let terms = collect_additive_terms(&expanded);
  for (i, var) in vars.iter().enumerate() {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, 1) {
        if let Some(cv) = minimize_try_f64(&c) {
          coeffs[i] += cv;
        } else {
          return None;
        }
      }
    }
  }

  // Constant term: set all vars to 0
  let mut const_expr = expanded.clone();
  for var in vars {
    const_expr =
      crate::syntax::substitute_variable(&const_expr, var, &Expr::Integer(0));
  }
  let constant = crate::evaluator::evaluate_expr_to_expr(&const_expr)
    .ok()
    .and_then(|e| minimize_try_f64(&e))
    .unwrap_or(0.0);

  Some((coeffs, constant))
}

/// Single-variable constrained minimize.
fn minimize_constrained_1d(
  f: &Expr,
  constraints: &[Expr],
  var: &str,
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // Collect boundary points from constraints: x >= a, x <= b, x == c
  let mut lb: Option<f64> = None; // lower bound
  let mut ub: Option<f64> = None; // upper bound
  let mut eq_constraints: Vec<Expr> = Vec::new();
  let mut other_constraints = false;

  for con in constraints {
    match con {
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2 && operators.len() == 1 => {
        let lhs = &operands[0];
        let rhs = &operands[1];
        match &operators[0] {
          ComparisonOp::GreaterEqual => {
            // lhs >= rhs
            // Check if it's var >= const or const <= var
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                lb = Some(lb.map_or(v, |cur: f64| cur.max(v)));
              } else {
                other_constraints = true;
              }
            } else if matches!(rhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(lhs) {
                ub = Some(ub.map_or(v, |cur: f64| cur.min(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          ComparisonOp::LessEqual => {
            // lhs <= rhs
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                ub = Some(ub.map_or(v, |cur: f64| cur.min(v)));
              } else {
                other_constraints = true;
              }
            } else if matches!(rhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(lhs) {
                lb = Some(lb.map_or(v, |cur: f64| cur.max(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          ComparisonOp::Equal => {
            eq_constraints.push(con.clone());
          }
          ComparisonOp::Greater => {
            // Strict inequality: treat as >= for boundary
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                lb = Some(lb.map_or(v, |cur: f64| cur.max(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          ComparisonOp::Less => {
            if matches!(lhs, Expr::Identifier(n) if n == var) {
              if let Some(v) = minimize_try_f64(rhs) {
                ub = Some(ub.map_or(v, |cur: f64| cur.min(v)));
              } else {
                other_constraints = true;
              }
            } else {
              other_constraints = true;
            }
          }
          _ => other_constraints = true,
        }
      }
      _ => other_constraints = true,
    }
  }

  if other_constraints {
    // Cannot handle, return unevaluated
    let obj_with_cons = Expr::List(
      std::iter::once(f.clone())
        .chain(constraints.iter().cloned())
        .collect(),
    );
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![obj_with_cons, Expr::Identifier(var.to_string())].into(),
    });
  }

  // Collect candidate x values: bounds + unconstrained critical points
  let mut candidates: Vec<f64> = Vec::new();

  // Add boundary points
  if let Some(l) = lb {
    candidates.push(l);
  }
  if let Some(u) = ub {
    candidates.push(u);
  }

  // Find unconstrained critical points and filter to feasible region
  let df =
    simplify(crate::functions::calculus_ast::differentiate_expr(f, var)?);
  let cps = minimize_find_critical_points_1d(&df, var, f)?;
  for cp in &cps {
    if let Some(v) = minimize_try_f64(cp) {
      let feasible =
        lb.is_none_or(|l| v >= l - 1e-10) && ub.is_none_or(|u| v <= u + 1e-10);
      if feasible {
        candidates.push(v);
      }
    }
  }

  if candidates.is_empty() {
    let obj_with_cons = Expr::List(
      std::iter::once(f.clone())
        .chain(constraints.iter().cloned())
        .collect(),
    );
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![obj_with_cons, Expr::Identifier(var.to_string())].into(),
    });
  }

  // Find the minimum among candidates
  let mut best_f = f64::INFINITY;
  let mut best_x_f64 = candidates[0];
  for &cx in &candidates {
    let fx = find_root_eval_at(f, var, cx).unwrap_or(f64::INFINITY);
    if fx < best_f {
      best_f = fx;
      best_x_f64 = cx;
    }
  }

  // Try to find exact expression for best_x from critical points
  let best_x_exact = cps.iter().find(|cp| {
    minimize_try_f64(cp).is_some_and(|v| (v - best_x_f64).abs() < 1e-8)
  });

  let (result_val, result_x) = if let Some(exact_cp) = best_x_exact {
    let fval = minimize_eval_exact(f, var, exact_cp)?;
    let rv = if maximize {
      simplify(negate_expr(&fval))
    } else {
      fval
    };
    (rv, exact_cp.clone())
  } else {
    // Check if best_x_f64 is a boundary (integer or simple rational)
    let bx_rounded = best_x_f64.round();
    let result_x_expr = if (bx_rounded - best_x_f64).abs() < 1e-10 {
      Expr::Integer(bx_rounded as i128)
    } else {
      Expr::Real(best_x_f64)
    };
    let fval = minimize_eval_exact(f, var, &result_x_expr)?;
    let rv = if maximize {
      simplify(negate_expr(&fval))
    } else {
      fval
    };
    (rv, result_x_expr)
  };

  let rule = Expr::Rule {
    pattern: Box::new(Expr::Identifier(var.to_string())),
    replacement: Box::new(result_x),
  };
  Ok(Expr::List(
    vec![result_val, Expr::List(vec![rule].into())].into(),
  ))
}

/// Multi-variable constrained minimize.
/// Handles: LP (linear objective + linear constraints), and
/// non-linear objectives with linear equality/inequality constraints.
fn minimize_constrained_nd(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  // First try pure LP (linear objective + linear constraints)
  if vars.len() >= 2
    && let Some(result) = minimize_lp_2d(f, constraints, vars, maximize)
  {
    return Ok(result);
  }

  // For any dimension, try boundary reduction for linear constraints
  if let Some(result) =
    minimize_constrained_boundary(f, constraints, vars, maximize)?
  {
    return Ok(result);
  }

  // Return unevaluated
  let obj_with_cons = Expr::List(
    std::iter::once(f.clone())
      .chain(constraints.iter().cloned())
      .collect(),
  );
  Ok(Expr::FunctionCall {
    name: func_name.to_string(),
    args: vec![
      obj_with_cons,
      Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
    ]
    .into(),
  })
}

/// Check if a point (given as var→val map) satisfies all constraints numerically.
fn minimize_satisfies_constraints(
  constraints: &[Expr],
  vars: &[String],
  vals: &[f64],
) -> bool {
  for con in constraints {
    if let Expr::Comparison {
      operands,
      operators,
    } = con
      && operands.len() == 2
      && operators.len() == 1
    {
      let mut lhs_expr = operands[0].clone();
      let mut rhs_expr = operands[1].clone();
      for (var, &val) in vars.iter().zip(vals.iter()) {
        lhs_expr =
          crate::syntax::substitute_variable(&lhs_expr, var, &Expr::Real(val));
        rhs_expr =
          crate::syntax::substitute_variable(&rhs_expr, var, &Expr::Real(val));
      }
      let lhs_val = crate::evaluator::evaluate_expr_to_expr(&lhs_expr)
        .ok()
        .and_then(|e| minimize_try_f64(&e));
      let rhs_val = crate::evaluator::evaluate_expr_to_expr(&rhs_expr)
        .ok()
        .and_then(|e| minimize_try_f64(&e));
      if let (Some(l), Some(r)) = (lhs_val, rhs_val) {
        let ok = match &operators[0] {
          ComparisonOp::GreaterEqual => l >= r - 1e-8,
          ComparisonOp::LessEqual => l <= r + 1e-8,
          ComparisonOp::Greater => l > r - 1e-8,
          ComparisonOp::Less => l < r + 1e-8,
          ComparisonOp::Equal => (l - r).abs() <= 1e-8,
          _ => true,
        };
        if !ok {
          return false;
        }
      }
    }
  }
  true
}

/// Extract linear constraint coefficients for the form: a*x + b*y + ... >= c.
/// Returns None if constraint is not linear.
/// True if `con` is a strict comparison (`<` or `>`), as opposed to `<=`/`>=`
/// or `==`. Used by integer enumeration to exclude boundary integers.
fn constraint_is_strict(con: &Expr) -> bool {
  matches!(con, Expr::Comparison { operators, .. }
    if operators.len() == 1
      && matches!(operators[0], ComparisonOp::Greater | ComparisonOp::Less))
}

fn minimize_extract_linear_constraint(
  con: &Expr,
  vars: &[String],
) -> Option<(Vec<f64>, f64, i32)> {
  let (operands, operators) = match con {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => (operands, operators),
    _ => return None,
  };
  let sense = match &operators[0] {
    ComparisonOp::GreaterEqual | ComparisonOp::Greater => 1,
    ComparisonOp::LessEqual | ComparisonOp::Less => -1,
    ComparisonOp::Equal => 0,
    _ => return None,
  };

  // diff = lhs - rhs, should be linear
  let diff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(operands[0].clone()),
    right: Box::new(operands[1].clone()),
  };
  let expanded = expand_and_combine(&diff);

  let mut coeffs = vec![0.0f64; vars.len()];
  let mut constant = 0.0f64;

  // Check this is a polynomial of degree <= 1 in all vars
  for (i, var) in vars.iter().enumerate() {
    let deg = max_power_int(&expanded, var);
    match deg {
      Some(d) if d > 1 => return None, // non-linear
      _ => {}
    }
    let terms = collect_additive_terms(&expanded);
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, 1) {
        if let Some(cv) = minimize_try_f64(&c) {
          coeffs[i] += cv;
        } else {
          return None; // non-constant coefficient
        }
      }
    }
  }
  // Constant term: evaluate with all vars set to 0
  let mut const_expr = expanded.clone();
  for var in vars {
    const_expr =
      crate::syntax::substitute_variable(&const_expr, var, &Expr::Integer(0));
  }
  if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(&const_expr) {
    constant = minimize_try_f64(&evaled).unwrap_or(0.0);
  }
  // diff = sum(coeffs[i] * vars[i]) + constant >= 0 (for sense 1)
  // So sum(coeffs[i] * vars[i]) >= -constant
  Some((coeffs, -constant, sense))
}

/// Minimize with linear constraints by trying constraint boundaries.
/// For each linear constraint, substitute it as equality into f and minimize the
/// resulting lower-dimensional problem.
fn minimize_constrained_boundary(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let n = vars.len();

  // Collect all linear constraints
  let mut lin_cons = Vec::new();
  for con in constraints {
    if let Some(lc) = minimize_extract_linear_constraint(con, vars) {
      lin_cons.push((con.clone(), lc));
    }
  }

  let mut candidates: Vec<(f64, Vec<f64>)> = Vec::new();

  // First try the unconstrained minimum
  if n == 1 {
    if let Ok(result) = minimize_single_var(f, &vars[0], false, "Minimize")
      && let Expr::List(items) = &result
      && items.len() == 2
      && let Some(fval) = minimize_try_f64(&items[0])
      && let Expr::List(rules) = &items[1]
      && let Some(Expr::Rule { replacement, .. }) = rules.first()
      && let Some(xval) = minimize_try_f64(replacement)
    {
      let feasible = minimize_satisfies_constraints(constraints, vars, &[xval]);
      if feasible {
        candidates.push((fval, vec![xval]));
      }
    }
  } else if n == 2
    && let Ok(result) = minimize_multi_var(f, vars, false, "Minimize")
    && let Expr::List(items) = &result
    && items.len() == 2
    && let Some(fval) = minimize_try_f64(&items[0])
    && let Expr::List(rules) = &items[1]
  {
    let mut vals = vec![0.0f64; n];
    let mut all_ok = true;
    for rule in rules {
      if let Expr::Rule {
        pattern,
        replacement,
      } = rule
        && let Expr::Identifier(vname) = pattern.as_ref()
        && let Some(pos) = vars.iter().position(|v| v == vname)
      {
        if let Some(val) = minimize_try_f64(replacement) {
          vals[pos] = val;
        } else {
          all_ok = false;
        }
      }
    }
    if all_ok {
      let feasible = minimize_satisfies_constraints(constraints, vars, &vals);
      if feasible {
        candidates.push((fval, vals));
      }
    }
  }

  // Try each linear constraint as equality boundary
  for (_, (coeffs, rhs, _)) in &lin_cons {
    // Find a variable with non-zero coefficient to eliminate
    let Some(elim_idx) = coeffs.iter().position(|&c| c.abs() > 1e-12) else {
      continue;
    };
    let elim_var = &vars[elim_idx];
    let elim_coeff = coeffs[elim_idx];

    // Solve: coeffs[elim_idx] * elim_var + sum(others) = rhs
    // elim_var = (rhs - sum(others)) / elim_coeff
    // Build expression: (rhs - sum(coeff_j * var_j for j != elim_idx)) / elim_coeff
    let mut elim_expr: Expr = Expr::Real(*rhs);
    for (j, var_j) in vars.iter().enumerate() {
      if j != elim_idx && coeffs[j].abs() > 1e-12 {
        let term = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Real(coeffs[j])),
          right: Box::new(Expr::Identifier(var_j.clone())),
        };
        elim_expr = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(elim_expr),
          right: Box::new(term),
        };
      }
    }
    if (elim_coeff - 1.0).abs() > 1e-12 {
      elim_expr = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(elim_expr),
        right: Box::new(Expr::Real(elim_coeff)),
      };
    }

    // Substitute elim_var = elim_expr in f
    let f_reduced = crate::syntax::substitute_variable(f, elim_var, &elim_expr);
    let f_reduced = simplify(f_reduced);

    // Get remaining variables (all except elim_var)
    let remaining_vars: Vec<String> =
      vars.iter().filter(|v| *v != elim_var).cloned().collect();

    if remaining_vars.is_empty() {
      // All variables eliminated - evaluate f
      if let Ok(fval_expr) = crate::evaluator::evaluate_expr_to_expr(&f_reduced)
        && let Some(fval) = minimize_try_f64(&fval_expr)
      {
        // Get the eliminated variable's value
        let elim_val_expr = crate::evaluator::evaluate_expr_to_expr(&elim_expr)
          .unwrap_or(elim_expr.clone());
        if let Some(elim_val) = minimize_try_f64(&elim_val_expr) {
          let mut vals = vec![0.0f64; n];
          vals[elim_idx] = elim_val;
          if minimize_satisfies_constraints(constraints, vars, &vals) {
            candidates.push((fval, vals));
          }
        }
      }
    } else if remaining_vars.len() == 1 {
      // 1D reduced problem
      let rem_var = &remaining_vars[0];
      let rem_idx = vars.iter().position(|v| v == rem_var).unwrap();

      if let Ok(result) =
        minimize_single_var(&f_reduced, rem_var, false, "Minimize")
        && let Expr::List(items) = &result
        && items.len() == 2
        && let Some(fval) = minimize_try_f64(&items[0])
        && let Expr::List(rules) = &items[1]
        && let Some(Expr::Rule { replacement, .. }) = rules.first()
        && let Some(rem_val) = minimize_try_f64(replacement)
      {
        // Compute elim_var value
        let elim_val_expr = crate::syntax::substitute_variable(
          &elim_expr,
          rem_var,
          &Expr::Real(rem_val),
        );
        if let Ok(evaled) =
          crate::evaluator::evaluate_expr_to_expr(&elim_val_expr)
          && let Some(elim_val) = minimize_try_f64(&evaled)
        {
          let mut vals = vec![0.0f64; n];
          vals[elim_idx] = elim_val;
          vals[rem_idx] = rem_val;
          if minimize_satisfies_constraints(constraints, vars, &vals) {
            candidates.push((fval, vals));
          }
        }
      }
    }
  }

  if candidates.is_empty() {
    return Ok(None);
  }

  // Find minimum candidate
  let best = candidates
    .iter()
    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
    .unwrap();

  let result_fval = if maximize {
    minimize_recognize_exact(-best.0)
  } else {
    minimize_recognize_exact(best.0)
  };

  let rules: Vec<Expr> = vars
    .iter()
    .zip(best.1.iter())
    .map(|(v, &val)| {
      let exact_val = minimize_recognize_exact(val);
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(v.clone())),
        replacement: Box::new(exact_val),
      }
    })
    .collect();

  Ok(Some(Expr::List(
    vec![result_fval, Expr::List(rules.into())].into(),
  )))
}

/// Try to solve a 2D linear program by enumerating vertices.
/// Returns None if the problem is not a linear program.
fn minimize_lp_2d(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
) -> Option<Expr> {
  let (x_name, y_name) = (&vars[0], &vars[1]);

  // Each constraint ax + by >= c or ax + by <= c or ax + by == c
  // We store as (a, b, c, sense) where sense: 1 = >=, -1 = <=, 0 = ==
  let mut linear_cons: Vec<(f64, f64, f64, i32)> = Vec::new();

  for con in constraints {
    let Expr::Comparison {
      operands,
      operators,
    } = con
    else {
      return None;
    };
    if operands.len() != 2 || operators.len() != 1 {
      return None;
    }
    let lhs = &operands[0];
    let rhs = &operands[1];

    let sense = match &operators[0] {
      ComparisonOp::GreaterEqual => 1,
      ComparisonOp::LessEqual => -1,
      ComparisonOp::Greater => 1,
      ComparisonOp::Less => -1,
      ComparisonOp::Equal => 0,
      _ => return None,
    };

    // Extract coefficients from lhs - rhs as linear function of vars
    let diff = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(lhs.clone()),
      right: Box::new(rhs.clone()),
    };
    let expanded = expand_and_combine(&diff);
    let terms = collect_additive_terms(&expanded);

    let mut ax = 0.0f64;
    let mut ay = 0.0f64;
    let mut ac = 0.0f64;

    for term in &terms {
      let cx = extract_coefficient_of_power(term, x_name, 1);
      let cy = extract_coefficient_of_power(term, y_name, 1);

      if let Some(ref c) = cx {
        ax += minimize_try_f64(c)?;
      }
      if let Some(ref c) = cy {
        ay += minimize_try_f64(c)?;
      }
      // Constant term: coefficient of x^0 that doesn't contain y
      if cx.is_none() && cy.is_none() {
        ac += minimize_try_f64(term)?;
      }
    }

    linear_cons.push((ax, ay, -ac, sense));
  }

  // Extract linear objective coefficients
  let expanded_f = expand_and_combine(f);
  let terms_f = collect_additive_terms(&expanded_f);
  let mut fx = 0.0f64;
  let mut fy = 0.0f64;
  let mut fc = 0.0f64;
  for term in &terms_f {
    let cx = extract_coefficient_of_power(term, x_name, 1);
    let cy = extract_coefficient_of_power(term, y_name, 1);
    if let Some(ref c) = cx {
      fx += minimize_try_f64(c)?;
    }
    if let Some(ref c) = cy {
      fy += minimize_try_f64(c)?;
    }
    if cx.is_none() && cy.is_none() {
      fc += minimize_try_f64(term)?;
    }
  }

  // Enumerate vertices: intersections of all pairs of constraint lines
  let mut vertices: Vec<(f64, f64)> = Vec::new();

  // Add intersections of pairs of constraints (treating each as equality)
  for i in 0..linear_cons.len() {
    for j in (i + 1)..linear_cons.len() {
      let (a1, b1, c1, _) = linear_cons[i];
      let (a2, b2, c2, _) = linear_cons[j];
      let det = a1 * b2 - a2 * b1;
      if det.abs() < 1e-12 {
        continue;
      }
      let xv = (c1 * b2 - c2 * b1) / det;
      let yv = (a1 * c2 - a2 * c1) / det;
      // Check feasibility
      let feasible = linear_cons.iter().all(|&(a, b, c, sense)| {
        let val = a * xv + b * yv - c;
        match sense {
          1 => val >= -1e-8,
          -1 => val <= 1e-8,
          0 => val.abs() <= 1e-8,
          _ => true,
        }
      });
      if feasible {
        vertices.push((xv, yv));
      }
    }
  }

  if vertices.is_empty() {
    return None;
  }

  // Find the vertex that minimizes the objective
  let mut best_val = f64::INFINITY;
  let mut best_vertex = vertices[0];

  for &(xv, yv) in &vertices {
    let val = fx * xv + fy * yv + fc;
    if val < best_val {
      best_val = val;
      best_vertex = (xv, yv);
    }
  }

  // Try to make exact values from approximate
  let make_exact = |v: f64| -> Expr {
    let rounded = v.round();
    if (rounded - v).abs() < 1e-8 {
      Expr::Integer(rounded as i128)
    } else {
      // Check if v = p/q for small q
      for q in 1i128..=10 {
        let p = (v * q as f64).round() as i128;
        if ((p as f64 / q as f64) - v).abs() < 1e-8 {
          let (rn, rd) = reduce_fraction(p, q);
          return if rd == 1 {
            Expr::Integer(rn)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(rn), Expr::Integer(rd)].into(),
            }
          };
        }
      }
      Expr::Real(v)
    }
  };

  // Also try to make the objective value exact
  let result_val = {
    let v = if maximize { -best_val } else { best_val };
    let rounded = v.round();
    if (rounded - v).abs() < 1e-8 {
      Expr::Integer(rounded as i128)
    } else {
      for q in 1i128..=10 {
        let p = (v * q as f64).round() as i128;
        if ((p as f64 / q as f64) - v).abs() < 1e-8 {
          let (rn, rd) = reduce_fraction(p, q);
          let e = if rd == 1 {
            Expr::Integer(rn)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(rn), Expr::Integer(rd)].into(),
            }
          };
          return Some(Expr::List(
            vec![
              e,
              Expr::List(
                vec![
                  Expr::Rule {
                    pattern: Box::new(Expr::Identifier(x_name.clone())),
                    replacement: Box::new(make_exact(best_vertex.0)),
                  },
                  Expr::Rule {
                    pattern: Box::new(Expr::Identifier(y_name.clone())),
                    replacement: Box::new(make_exact(best_vertex.1)),
                  },
                ]
                .into(),
              ),
            ]
            .into(),
          ));
        }
      }
      Expr::Real(v)
    }
  };

  Some(Expr::List(
    vec![
      result_val,
      Expr::List(
        vec![
          Expr::Rule {
            pattern: Box::new(Expr::Identifier(x_name.clone())),
            replacement: Box::new(make_exact(best_vertex.0)),
          },
          Expr::Rule {
            pattern: Box::new(Expr::Identifier(y_name.clone())),
            replacement: Box::new(make_exact(best_vertex.1)),
          },
        ]
        .into(),
      ),
    ]
    .into(),
  ))
}

// ─── FindMinimum / FindMaximum ───────────────────────────────────────

/// Solve the dense linear system `a * x = b` with Gaussian elimination
/// and partial pivoting. Returns None when the matrix is singular.
fn solve_dense_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
  let n = b.len();
  let mut aug: Vec<Vec<f64>> = a
    .iter()
    .zip(b.iter())
    .map(|(row, &bi)| {
      let mut r = row.clone();
      r.push(bi);
      r
    })
    .collect();

  for col in 0..n {
    let pivot_row = (col..n).max_by(|&r1, &r2| {
      aug[r1][col]
        .abs()
        .partial_cmp(&aug[r2][col].abs())
        .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    if aug[pivot_row][col].abs() < 1e-12 {
      return None;
    }
    aug.swap(col, pivot_row);
    for row in (col + 1)..n {
      let factor = aug[row][col] / aug[col][col];
      for j in col..=n {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  let mut x = vec![0.0; n];
  for i in (0..n).rev() {
    let mut s = aug[i][n];
    for j in (i + 1)..n {
      s -= aug[i][j] * x[j];
    }
    x[i] = s / aug[i][i];
  }
  Some(x)
}

/// FindMinValue / FindMaxValue — like FindMinimum / FindMaximum, but
/// return only the extremum value (first element of the result pair).
pub fn find_min_value_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  match find_minimum_ast(args, maximize)? {
    Expr::List(ref items) if items.len() == 2 => Ok(items[0].clone()),
    _ => Ok(Expr::FunctionCall {
      name: if maximize {
        "FindMaxValue"
      } else {
        "FindMinValue"
      }
      .to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// FindMinimum[f, {x, x0}] — find a local minimum of f starting at x0
/// FindMinimum[f, {{x, x0}, {y, y0}}] — multivariable
/// Returns {min_value, {x -> x_min, ...}}
///
/// FindMaximum is implemented by negating f and negating the result.
pub fn find_minimum_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize {
    "FindMaximum"
  } else {
    "FindMinimum"
  };
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{func_name} expects at least 2 arguments"
    )));
  }
  // Trailing arguments are options (e.g. MaxIterations -> 2,
  // Method -> "Newton"). They aren't honoured yet, but we accept them
  // silently rather than aborting so call shapes match Wolfram.
  // Only the first two positional arguments drive the optimisation.
  let f = &args[0];

  // Parse variables and starting points: x, {x, y}, {x, x0} or
  // {{x, x0}, {y, y0}}. Bare symbols get Wolfram's automatic starting
  // point of 1 (FindMinimum[f, x] == FindMinimum[f, {x, 1}]).
  let var_specs = match &args[1] {
    Expr::Identifier(name) => vec![(name.clone(), 1.0)],
    Expr::List(items)
      if !items.is_empty() && matches!(&items[0], Expr::List(_)) =>
    {
      // Multivariable: {{x, x0}, {y, y0}, ...}
      let mut specs = Vec::new();
      for item in items {
        if let Expr::List(pair) = item
          && pair.len() == 2
          && let Expr::Identifier(name) = &pair[0]
        {
          let x0 = find_root_eval_number(&pair[1])?;
          specs.push((name.clone(), x0));
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "{func_name}: variable spec must be {{var, start}}"
          )));
        }
      }
      specs
    }
    // List of bare symbols: {x, y, ...} with automatic starting points.
    // A two-element list is only a variable list when the second element
    // is not a numeric starting point (so {x, Pi} stays {var, start}).
    Expr::List(items)
      if items.len() >= 2
        && items.iter().all(|i| matches!(i, Expr::Identifier(_)))
        && (items.len() != 2 || find_root_eval_number(&items[1]).is_err()) =>
    {
      items
        .iter()
        .map(|i| match i {
          Expr::Identifier(name) => (name.clone(), 1.0),
          _ => unreachable!(),
        })
        .collect()
    }
    Expr::List(items) if items.len() == 2 => {
      // Single variable: {x, x0}
      if let Expr::Identifier(name) = &items[0] {
        let x0 = find_root_eval_number(&items[1])?;
        vec![(name.clone(), x0)]
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{func_name}: variable spec must be {{var, start}}"
        )));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "{func_name}: second argument must be {{var, start}} or {{{{x, x0}}, {{y, y0}}}}"
      )));
    }
  };

  let vars: Vec<String> = var_specs.iter().map(|(v, _)| v.clone()).collect();
  let mut x: Vec<f64> = var_specs.iter().map(|(_, x0)| *x0).collect();
  let n = vars.len();

  // Compute symbolic gradients (partial derivatives)
  let mut grad_exprs: Vec<Expr> = Vec::with_capacity(n);
  for var in &vars {
    let deriv = crate::functions::calculus_ast::differentiate_expr(f, var)?;
    grad_exprs.push(simplify(deriv));
  }

  // Compute symbolic Hessian (for Newton's method in 1D, second derivative)
  let mut hess_exprs: Vec<Vec<Expr>> = Vec::new();
  for i in 0..n {
    let mut row = Vec::new();
    for j in 0..n {
      let h = crate::functions::calculus_ast::differentiate_expr(
        &grad_exprs[i],
        &vars[j],
      )?;
      row.push(simplify(h));
    }
    hess_exprs.push(row);
  }

  // Evaluate expression at point
  let eval_at = |expr: &Expr, point: &[f64]| -> Result<f64, InterpreterError> {
    let mut e = expr.clone();
    for (i, var) in vars.iter().enumerate() {
      e = crate::syntax::substitute_variable(&e, var, &Expr::Real(point[i]));
    }
    let evaled = crate::evaluator::evaluate_expr_to_expr(&e)?;
    expr_to_f64(&evaled)
  };

  // Pre-flight: if f doesn't reduce to a real number at the starting
  // point (e.g. contains an unbound symbol like phi[x]), emit
  // {func_name}::nrnum and return the call unevaluated. Matches
  // wolframscript's behaviour and lets script chains keep flowing
  // instead of aborting with a hard "Cannot evaluate numerically".
  if eval_at(f, &x).is_err() {
    let mut substituted = f.clone();
    for (i, var) in vars.iter().enumerate() {
      substituted = crate::syntax::substitute_variable(
        &substituted,
        var,
        &Expr::Real(x[i]),
      );
    }
    let value_str = crate::syntax::expr_to_output(&substituted);
    let var_str: String = if vars.len() == 1 {
      format!("{{{}}} = {{{}}}", vars[0], x[0])
    } else {
      let names = vars.join(", ");
      let vals = x
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
      format!("{{{}}} = {{{}}}", names, vals)
    };
    crate::emit_message(&format!(
      "{func_name}::nrnum: The function value {} is not a real number at {}.",
      value_str, var_str,
    ));
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: args.to_vec().into(),
    });
  }

  let sign = if maximize { -1.0 } else { 1.0 };
  let max_iter = 200;
  let tol = 1e-15;

  if n == 1 {
    // Single variable: damped Newton's method on the derivative
    // Uses line search to ensure we actually decrease/increase the function
    for _ in 0..max_iter {
      let gval = eval_at(&grad_exprs[0], &x)?;
      let hval = eval_at(&hess_exprs[0][0], &x)?;

      // Compute Newton direction. For a quadratic this gives the exact
      // minimum in one step regardless of overall scale, so we converge
      // even when both gval and hval are tiny (e.g. 10^-30 · (x-3)^2).
      let step = if hval.abs() < 1e-30 {
        // Hessian too small — use gradient descent step
        sign * gval * 0.1
      } else if (maximize && hval > 0.0) || (!maximize && hval < 0.0) {
        // Hessian has wrong sign for our goal (saddle point or max when seeking min)
        // Use gradient descent instead
        sign * gval * 0.1
      } else {
        gval / hval
      };

      // Convergence: the Newton step itself is small. Using gval alone
      // would terminate too eagerly when the function is scaled by a
      // tiny constant (gradient is O(scale), but the step is O(1)).
      if step.abs() < tol {
        break;
      }

      // Line search along Newton direction to ensure improvement.
      // Use `<=`: if the function is flat to machine precision (e.g. a
      // quadratic scaled by 10^-30 added to 2., which evaluates to 2.0
      // identically in f64), we still want to take the Newton step rather
      // than backtracking to a near-zero alpha and freezing in place.
      let current_f = eval_at(f, &x)? * sign;
      let mut alpha = 1.0;
      let mut best_x = x[0] - step;
      let mut best_f = eval_at(f, &[best_x])? * sign;

      // Backtracking: reduce step only if it strictly worsens the value.
      for _ in 0..30 {
        if best_f <= current_f {
          break;
        }
        alpha *= 0.5;
        best_x = x[0] - alpha * step;
        best_f = eval_at(f, &[best_x])? * sign;
      }
      x[0] = best_x;
    }
  } else {
    // Multivariable: damped Newton on sign*f using the symbolic Hessian.
    // Falls back to steepest descent when the Newton system is singular
    // or its direction is not a descent direction (e.g. near a saddle).
    // Plain gradient descent only converges linearly and used to stall
    // short of the optimum within the iteration budget.
    for _ in 0..max_iter {
      // Signed gradient (of sign*f, so "descent" always means improvement)
      let mut grad = vec![0.0; n];
      for i in 0..n {
        grad[i] = eval_at(&grad_exprs[i], &x)? * sign;
      }

      let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
      if grad_norm < tol {
        break;
      }

      // Signed Hessian
      let mut hess = vec![vec![0.0; n]; n];
      for (i, row) in hess.iter_mut().enumerate() {
        for (j, h) in row.iter_mut().enumerate() {
          *h = eval_at(&hess_exprs[i][j], &x)? * sign;
        }
      }

      // Newton direction: hess * d = -grad
      let neg_grad: Vec<f64> = grad.iter().map(|g| -g).collect();
      let dir = match solve_dense_linear_system(&hess, &neg_grad) {
        Some(d)
          if grad.iter().zip(d.iter()).map(|(g, di)| g * di).sum::<f64>()
            < 0.0 =>
        {
          d
        }
        _ => neg_grad,
      };

      let step_norm: f64 = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
      if step_norm < tol {
        break;
      }

      // Backtracking line search (Armijo condition on sign*f)
      let c = 1e-4;
      let current_f = eval_at(f, &x)? * sign;
      let decrease: f64 =
        grad.iter().zip(dir.iter()).map(|(g, d)| g * d).sum::<f64>();
      let mut alpha = 1.0;

      for _ in 0..50 {
        let x_new: Vec<f64> = x
          .iter()
          .zip(dir.iter())
          .map(|(xi, di)| xi + alpha * di)
          .collect();
        let new_f = eval_at(f, &x_new)? * sign;
        if new_f <= current_f + c * alpha * decrease || alpha < 1e-15 {
          x = x_new;
          break;
        }
        alpha *= 0.5;
      }
    }
  }

  // Compute final function value
  let min_val = eval_at(f, &x)?;
  let min_val_expr = Expr::Real(min_val);

  // Build result: {min_val, {x -> x_min, y -> y_min, ...}}
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(var, val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(var.clone())),
      replacement: Box::new(Expr::Real(*val)),
    })
    .collect();

  Ok(Expr::List(
    vec![min_val_expr, Expr::List(rules.into())].into(),
  ))
}

/// Split an And expression into its equality part and inequality parts.
///
/// For `eq && ineq1 && ineq2`, returns `(Some(eq), [ineq1, ineq2])`.
/// Equalities are `Comparison { op: Equal }`, inequalities are everything else.
pub fn extract_eq_and_ineq_parts(expr: &Expr) -> (Option<Expr>, Vec<Expr>) {
  let mut constraints = Vec::new();
  collect_and_constraints(expr, &mut constraints);
  let mut eq_part: Option<Expr> = None;
  let mut ineqs: Vec<Expr> = Vec::new();
  for c in constraints {
    let is_eq = matches!(
      &c,
      Expr::Comparison { operators, .. }
        if operators.len() == 1
          && operators[0] == ComparisonOp::Equal
    );
    if is_eq && eq_part.is_none() {
      eq_part = Some(c);
    } else {
      ineqs.push(c);
    }
  }
  (eq_part, ineqs)
}

/// Given a solution value `var -> ConditionalExpression[a + b·C, C ∈ Integers]`
/// (a periodic family) and bounding inequalities, return the concrete
/// `var -> value` rules satisfying every inequality. Returns `None` when the
/// value is not such a family, the body is not linear in the parameter, or the
/// constraints do not bound the parameter to a finite range.
fn specialize_periodic_solution(
  var_name: &str,
  replacement: &Expr,
  ineqs: &[Expr],
) -> Option<Vec<Expr>> {
  // Unwrap ConditionalExpression[body, Element[param, Integers]].
  let Expr::FunctionCall { name, args } = replacement else {
    return None;
  };
  if name != "ConditionalExpression" || args.len() != 2 {
    return None;
  }
  let body = &args[0];
  let Expr::FunctionCall { name: en, args: ea } = &args[1] else {
    return None;
  };
  if en != "Element"
    || ea.len() != 2
    || !matches!(&ea[1], Expr::Identifier(s) if s == "Integers")
  {
    return None;
  }
  let param = &ea[0];

  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let subst_param = |value: Expr| -> Option<Expr> {
    eval(Expr::FunctionCall {
      name: "ReplaceAll".to_string(),
      args: vec![
        body.clone(),
        Expr::Rule {
          pattern: Box::new(param.clone()),
          replacement: Box::new(value),
        },
      ]
      .into(),
    })
  };
  // Linear coefficients: a = body | C=0, b = Coefficient[body, C, 1].
  let a = crate::functions::math_ast::try_eval_to_f64(&subst_param(
    Expr::Integer(0),
  )?)?;
  let b_expr = eval(Expr::FunctionCall {
    name: "Coefficient".to_string(),
    args: vec![body.clone(), param.clone(), Expr::Integer(1)].into(),
  })?;
  let b = crate::functions::math_ast::try_eval_to_f64(&b_expr)?;
  if b.abs() < 1e-12 {
    return None;
  }

  // Finite numeric bounds on `var` taken from the inequality operands.
  let mut x_bounds: Vec<f64> = Vec::new();
  for ineq in ineqs {
    if let Expr::Comparison { operands, .. } = ineq {
      for op in operands {
        if crate::syntax::expr_to_string(op) == var_name {
          continue;
        }
        if let Some(v) = crate::functions::math_ast::try_eval_to_f64(op) {
          x_bounds.push(v);
        }
      }
    }
  }
  if x_bounds.len() < 2 {
    return None; // not bounded on both sides
  }
  let x_lo = x_bounds.iter().cloned().fold(f64::INFINITY, f64::min);
  let x_hi = x_bounds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  // Parameter range from x = a + b·C ∈ [x_lo, x_hi], with a margin so that
  // boundary integers are tested by the exact inequality check below.
  let c1 = (x_lo - a) / b;
  let c2 = (x_hi - a) / b;
  let (c_lo, c_hi) = (c1.min(c2), c1.max(c2));
  let k_lo = (c_lo.floor() as i64) - 1;
  let k_hi = (c_hi.ceil() as i64) + 1;
  if k_hi - k_lo > 100_000 {
    return None; // runaway guard
  }

  let mut result: Vec<Expr> = Vec::new();
  for k in k_lo..=k_hi {
    let value = subst_param(Expr::Integer(k as i128))?;
    // Keep k only if every inequality holds for var = value.
    let ok = ineqs.iter().all(|ineq| {
      let subst = crate::syntax::substitute_variable(ineq, var_name, &value);
      matches!(crate::evaluator::evaluate_expr_to_expr(&subst),
        Ok(Expr::Identifier(ref s)) if s == "True")
    });
    if ok {
      result.push(Expr::List(
        vec![Expr::Rule {
          pattern: Box::new(Expr::Identifier(var_name.to_string())),
          replacement: Box::new(value),
        }]
        .into(),
      ));
    }
  }
  Some(result)
}

/// Check if an expression is zero.
fn is_expr_zero(e: &Expr) -> bool {
  matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(x) if *x == 0.0)
}

/// Evaluate and simplify an expression (double pass for compound simplifications).
fn eval_entry(e: Expr) -> Expr {
  let r = crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e);
  let r2 = crate::evaluator::evaluate_expr_to_expr(&r).unwrap_or(r);
  simplify(r2)
}

/// Solve a system of linear equations using symbolic Gaussian elimination.
///
/// Returns `Some(Expr::List([Expr::List([rules...])]))` if the system is linear and consistent,
/// `Some(Expr::List([]))` if inconsistent, or `None` if the system is not linear.
fn solve_linear_symbolic(eqs: &[Expr], var_names: &[String]) -> Option<Expr> {
  let n = var_names.len();
  let mut matrix: Vec<Vec<Expr>> = Vec::new();

  for eq in eqs {
    let (lhs, rhs) = match eq {
      Expr::Comparison {
        operands,
        operators,
      } if operators.len() == 1
        && operators[0] == ComparisonOp::Equal
        && operands.len() == 2 =>
      {
        (&operands[0], &operands[1])
      }
      _ => return None,
    };
    // poly = lhs - rhs; find coefficients of poly == 0
    let poly_raw = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(lhs.clone()),
      right: Box::new(rhs.clone()),
    };
    let poly = expand_and_combine(&poly_raw);
    let terms = collect_additive_terms(&poly);

    let mut coeffs: Vec<Expr> = vec![Expr::Integer(0); n];
    let mut constant = Expr::Integer(0);

    for term in &terms {
      let all_const = var_names.iter().all(|v| is_constant_wrt(term, v));
      if all_const {
        constant = add_exprs(&constant, term);
        continue;
      }
      let mut found_var: Option<(usize, Expr)> = None;
      let mut valid = true;
      for (j, var) in var_names.iter().enumerate() {
        let (power, coeff) = term_var_power_and_coeff(term, var);
        if power == 1 {
          if found_var.is_some() {
            valid = false; // product of two variables → nonlinear
            break;
          }
          found_var = Some((j, coeff));
        } else if power != 0 {
          valid = false; // higher power or sentinel
          break;
        }
      }
      if !valid {
        return None; // non-linear system — fall back to reduce
      }
      if let Some((j, coeff)) = found_var {
        coeffs[j] = eval_entry(add_exprs(&coeffs[j], &coeff));
      } else {
        constant = add_exprs(&constant, term);
      }
    }
    // Augmented row: [a0, ..., a_{n-1}, b] where A*x = b (b = -constant)
    let mut row: Vec<Expr> =
      coeffs.iter().map(|c| eval_entry(c.clone())).collect();
    row.push(eval_entry(negate_expr(&eval_entry(constant))));
    matrix.push(row);
  }

  let nrows = matrix.len();
  let ncols = n + 1;
  let mut pivot_row = 0;
  let mut pivot_cols: Vec<(usize, usize)> = Vec::new();

  for col in 0..n {
    if pivot_row >= nrows {
      break;
    }
    let found = (pivot_row..nrows).find(|&r| !is_expr_zero(&matrix[r][col]));
    let Some(swap_row) = found else { continue };
    if swap_row != pivot_row {
      matrix.swap(pivot_row, swap_row);
    }
    pivot_cols.push((pivot_row, col));
    let pivot = matrix[pivot_row][col].clone();

    for row in 0..nrows {
      if row == pivot_row {
        continue;
      }
      let factor = matrix[row][col].clone();
      if !is_expr_zero(&factor) {
        for j in 0..ncols {
          let t1 = eval_entry(multiply_exprs(&pivot, &matrix[row][j]));
          let t2 = eval_entry(multiply_exprs(&factor, &matrix[pivot_row][j]));
          matrix[row][j] = eval_entry(Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(t1),
            right: Box::new(t2),
          });
        }
      }
    }
    pivot_row += 1;
  }

  // Normalize each pivot row by dividing by its pivot element
  for &(row, col) in &pivot_cols {
    let pivot = matrix[row][col].clone();
    if !is_expr_zero(&pivot) {
      for j in 0..ncols {
        let entry = matrix[row][j].clone();
        if !is_expr_zero(&entry) {
          matrix[row][j] = eval_entry(solve_divide(&entry, &pivot));
        }
      }
    }
  }

  // Check for inconsistency
  for row in 0..nrows {
    if (0..n).all(|j| is_expr_zero(&matrix[row][j]))
      && !is_expr_zero(&matrix[row][n])
    {
      return Some(Expr::List(vec![].into())); // no solution
    }
  }

  let pivot_var_cols: Vec<usize> = pivot_cols.iter().map(|(_, c)| *c).collect();
  let free_var_cols: Vec<usize> =
    (0..n).filter(|j| !pivot_var_cols.contains(j)).collect();

  // Build solution expression for one parameterization:
  // vars[pivot_col] = rhs - sum_fc(coeff_fc * vars[fc]) where fc are free cols.
  // Rules are sorted by variable index to match Wolfram's output order.
  let build_rules = |pivot_cols: &[(usize, usize)],
                     free_var_cols: &[usize],
                     matrix: &[Vec<Expr>]|
   -> Vec<Expr> {
    let mut rules = Vec::new();
    // Sort pivot_cols by column index so rules appear in variable order
    let mut sorted_pivots = pivot_cols.to_vec();
    sorted_pivots.sort_by_key(|&(_, c)| c);
    for &(row, col) in &sorted_pivots {
      let mut rhs_expr = matrix[row][n].clone();
      for &fc in free_var_cols {
        let coeff = matrix[row][fc].clone();
        if !is_expr_zero(&coeff) {
          let term =
            multiply_exprs(&coeff, &Expr::Identifier(var_names[fc].clone()));
          let neg_term = negate_expr(&eval_entry(term));
          rhs_expr = eval_entry(add_exprs(&rhs_expr, &neg_term));
        }
      }
      // Run the user-level Simplify so the RHS collapses forms like
      // -1*(1 - E^3)/2 into (-1 + E^3)/2 (matching wolframscript).
      let intermediate = eval_entry(rhs_expr);
      let simplified_rhs =
        crate::functions::polynomial_ast::simplify_ast(&[intermediate.clone()])
          .unwrap_or(intermediate);
      rules.push(Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_names[col].clone())),
        replacement: Box::new(simplified_rhs),
      });
    }
    rules
  };

  // Check if an expression contains rational (fractional) coefficients
  fn has_fraction(e: &Expr) -> bool {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        !matches!(&args[1], Expr::Integer(1))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        right,
        ..
      } => !matches!(right.as_ref(), Expr::Integer(1)),
      Expr::BinaryOp { left, right, .. } => {
        has_fraction(left) || has_fraction(right)
      }
      Expr::FunctionCall { args, .. } => args.iter().any(has_fraction),
      Expr::UnaryOp { operand, .. } => has_fraction(operand),
      _ => false,
    }
  }

  let rules = build_rules(&pivot_cols, &free_var_cols, &matrix);

  // If any rule has fractional coefficients, try column swaps to eliminate fractions.
  // This matches Wolfram's convention of preferring integer-coefficient parameterizations.
  let rules = if free_var_cols.is_empty()
    || !rules.iter().any(|r| {
      if let Expr::Rule { replacement, .. } = r {
        has_fraction(replacement)
      } else {
        false
      }
    }) {
    rules
  } else {
    // Try each (free_col, pivot_row) swap.
    // A swap of free column fc with pivot at (row r, col pc_r) is "integer-clean" if:
    //   for all other pivot rows r', rref[r'][fc] / rref[r][fc] is integer.
    let mut best_rules = rules;
    'swap_search: for fi in 0..free_var_cols.len() {
      let fc = free_var_cols[fi];
      for pi in 0..pivot_cols.len() {
        let (pivot_r, pivot_c) = pivot_cols[pi];
        let swap_coeff = matrix[pivot_r][fc].clone();
        if is_expr_zero(&swap_coeff) {
          continue;
        }
        // Check that for all other pivot rows, the ratio is integer
        let mut all_ratios_integer = true;
        for (pi2, &(r2, _)) in pivot_cols.iter().enumerate() {
          if pi2 == pi {
            continue;
          }
          let other_coeff = &matrix[r2][fc];
          if is_expr_zero(other_coeff) {
            continue;
          }
          // Check if other_coeff / swap_coeff is integer
          let ratio = eval_entry(solve_divide(other_coeff, &swap_coeff));
          if has_fraction(&ratio) {
            all_ratios_integer = false;
            break;
          }
        }
        if !all_ratios_integer {
          continue;
        }
        // Perform the column swap: fc becomes a pivot, pivot_c becomes free.
        // New pivot rows = same as before but row pi now solves for vars[fc] instead of vars[pivot_c].
        // New free cols = (free_var_cols with fc replaced by pivot_c).
        let new_pivot_cols: Vec<(usize, usize)> = pivot_cols
          .iter()
          .enumerate()
          .map(|(i, &(r, c))| if i == pi { (r, fc) } else { (r, c) })
          .collect();
        let new_free_var_cols: Vec<usize> = free_var_cols
          .iter()
          .map(|&f| if f == fc { pivot_c } else { f })
          .collect();
        // Rebuild the RREF for the new pivot structure.
        // We need to "pivot" column fc out of row pi:
        // For row pi: new_matrix[pi][fc] = 1, others in col fc = 0, vars[pivot_c] is free.
        // Re-express: row pi → divide by swap_coeff, then eliminate fc from all other rows.
        let mut new_matrix = matrix.clone();
        // Normalize row pi: divide by swap_coeff
        {
          let sc = new_matrix[pivot_r][fc].clone();
          for j in 0..ncols {
            let v = new_matrix[pivot_r][j].clone();
            if !is_expr_zero(&v) {
              new_matrix[pivot_r][j] = eval_entry(solve_divide(&v, &sc));
            }
          }
          // After dividing, old pivot col entry: divide pivot_c col
          // (was 1, now 1/swap_coeff * 1 = 1/swap_coeff... wait)
          // Actually the matrix had rref[pi][pivot_c] = 1 (since it was normalized after GE)
          // and rref[pi][fc] = swap_coeff.
          // After dividing row pi by swap_coeff: rref[pi][fc] = 1, rref[pi][pivot_c] = 1/swap_coeff.
        }
        // Eliminate fc from all other pivot rows
        for (pi2, &(r2, _)) in pivot_cols.iter().enumerate() {
          if pi2 == pi {
            continue;
          }
          let factor = new_matrix[r2][fc].clone();
          if is_expr_zero(&factor) {
            continue;
          }
          for j in 0..ncols {
            let t1 = new_matrix[r2][j].clone();
            let t2 =
              eval_entry(multiply_exprs(&factor, &new_matrix[pivot_r][j]));
            new_matrix[r2][j] = eval_entry(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(t1),
              right: Box::new(t2),
            });
          }
        }
        let new_rules =
          build_rules(&new_pivot_cols, &new_free_var_cols, &new_matrix);
        let any_fraction = new_rules.iter().any(|r| {
          if let Expr::Rule { replacement, .. } = r {
            has_fraction(replacement)
          } else {
            false
          }
        });
        if !any_fraction {
          best_rules = new_rules;
          break 'swap_search;
        }
      }
    }
    best_rules
  };

  Some(Expr::List(vec![Expr::List(rules.into())].into()))
}

/// Convert an evaluated expression to f64
fn expr_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Ok(*n as f64 / *d as f64)
      } else {
        Err(InterpreterError::EvaluationError(
          "Cannot evaluate expression numerically".into(),
        ))
      }
    }
    _ => {
      let n_result = crate::functions::math_ast::n_ast(&[expr.clone()])?;
      match &n_result {
        Expr::Real(r) => Ok(*r),
        Expr::Integer(n) => Ok(*n as f64),
        _ => Err(InterpreterError::EvaluationError(
          "Cannot evaluate expression numerically".into(),
        )),
      }
    }
  }
}

// ─── NMinimize / NMaximize ──────────────────────────────────────────

/// NMinimize[{f, constraints...}, vars] / NMaximize[{f, constraints...}, vars]
/// Numerical global optimization using sampling + local refinement.
/// Returns {opt_value, {var -> val, ...}}.
pub fn nminimize_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize { "NMaximize" } else { "NMinimize" };
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{func_name} expects 2 arguments"
    )));
  }

  // Parse objective and constraints from first argument
  let (objective, constraints) = minimize_parse_objective(&args[0]);

  // Parse variable(s) from second argument
  let vars: Vec<String> = match &args[1] {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::List(items) => {
      let mut v = Vec::new();
      for item in items {
        if let Expr::Identifier(name) = item {
          v.push(name.clone());
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "{func_name}: variables must be symbols"
          )));
        }
      }
      v
    }
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "{func_name}: second argument must be a variable or list of variables"
      )));
    }
  };

  // A constraint that reduced to literal `False` (e.g. the chained
  // `5 <= x <= 1` evaluates to False) makes the feasible set empty.
  let mut flat_constraints: Vec<&Expr> = Vec::new();
  for c in &constraints {
    flatten_and_constraints_ref(c, &mut flat_constraints);
  }
  if flat_constraints
    .iter()
    .any(|c| matches!(c, Expr::Identifier(s) if s == "False"))
  {
    return Ok(nminimize_infeasible_result(&constraints, &vars, maximize));
  }

  // A constraint coupling two or more of the optimization variables (e.g.
  // x + y == 1 or x^2 + y^2 <= 1) can't be reduced to per-variable bounds, so
  // the numeric grid sampler below would silently ignore it. Delegate such
  // cases to the symbolic Minimize/Maximize solver (which respects the
  // constraints) and numericize its result. Single-variable box bounds still
  // go through the grid sampler.
  // Check each *atomic* constraint (the flattened conjuncts), not the whole
  // `And`: a conjunction like `x >= 5 && x <= 2 && y >= 0` mentions two
  // variables overall but couples none of them, so it must still go to the
  // per-variable grid sampler (which detects the empty x-range as infeasible)
  // rather than the symbolic Minimize path.
  let has_coupling_constraint = flat_constraints.iter().any(|c| {
    vars
      .iter()
      .filter(|v| crate::functions::polynomial_ast::contains_var(c, v))
      .count()
      >= 2
  });
  if has_coupling_constraint {
    // The grid sampler below only understands per-variable box bounds, so a
    // coupling constraint would be silently ignored. Always run the numeric
    // penalty-method optimizer, which honours arbitrary constraints.
    let numeric =
      nminimize_penalty(&objective, &constraints, &vars, maximize).ok();

    // Also try the full symbolic Minimize/Maximize dispatch (it exercises
    // specialized closed-form handlers). The symbolic solver is not always
    // correct for constrained quadratics, so don't trust it blindly: keep
    // whichever feasible candidate has the better objective value.
    let sym_name = if maximize { "Maximize" } else { "Minimize" };
    let symbolic =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: sym_name.to_string(),
        args: args.to_vec().into(),
      })
      .ok()
      .filter(|sym| matches!(sym, Expr::List(items) if items.len() == 2))
      .and_then(|sym| {
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "N".to_string(),
          args: vec![sym].into(),
        })
        .ok()
      });

    // Symbolic first: when it produces an exact, optimal answer we want to
    // keep it rather than overwrite it with float noise from the numeric
    // optimizer (which is only adopted when meaningfully better).
    let candidates: Vec<Expr> =
      [symbolic, numeric].into_iter().flatten().collect();
    return pick_best_optimum(
      candidates,
      &objective,
      &constraints,
      &vars,
      maximize,
    );
  }

  // Extract bounds from constraints (e.g. 0 < x < Pi/2)
  let bounds = extract_bounds(&constraints, &vars)?;

  // An empty box (lower bound above upper bound) means the per-variable
  // constraints are unsatisfiable. Mirror wolframscript's infeasible result
  // instead of feeding an inverted interval to the sampler (which would
  // panic in `clamp`).
  if bounds.iter().any(|&(lo, hi)| lo > hi) {
    return Ok(nminimize_infeasible_result(&constraints, &vars, maximize));
  }

  // Evaluate expression at a given point. The objective and its gradients are
  // each evaluated thousands of times during sampling and refinement, so the
  // first time a given expression is seen we compile it into a fast numeric
  // closure (keyed by its address, which is stable for the borrowed
  // `objective`/`grads[i]`) and reuse that. Expressions the compiler can't
  // handle — or points where the compiled form is non-finite (e.g. a fractional
  // power of a negative base) — fall back to the full AST evaluator.
  let compiled_cache: std::cell::RefCell<
    std::collections::HashMap<*const Expr, Option<NumNode>>,
  > = std::cell::RefCell::new(std::collections::HashMap::new());
  let eval_at = |expr: &Expr, point: &[f64]| -> Result<f64, InterpreterError> {
    let key = expr as *const Expr;
    if !compiled_cache.borrow().contains_key(&key) {
      let compiled = compile_numeric(expr, &vars);
      compiled_cache.borrow_mut().insert(key, compiled);
    }
    if let Some(node) = compiled_cache.borrow().get(&key).unwrap() {
      let v = node.eval(point);
      if v.is_finite() {
        return Ok(v);
      }
    }
    let mut e = expr.clone();
    for (i, var) in vars.iter().enumerate() {
      e = crate::syntax::substitute_variable(&e, var, &Expr::Real(point[i]));
    }
    let evaled = crate::evaluator::evaluate_expr_to_expr(&e)?;
    expr_to_f64(&evaled)
  };

  let n = vars.len();

  // Phase 1: Multi-scale grid sampling to find best starting point.
  // Use multiple scales to avoid missing optima near the origin when bounds
  // are very wide (e.g. the default -1e6 to 1e6).
  let samples_per_dim = 50;
  let mut best_x = vec![0.0; n];
  let mut best_f = if maximize {
    f64::NEG_INFINITY
  } else {
    f64::INFINITY
  };

  let update_best =
    |pt: &[f64],
     best_x: &mut Vec<f64>,
     best_f: &mut f64,
     eval_at: &dyn Fn(&Expr, &[f64]) -> Result<f64, InterpreterError>| {
      if let Ok(fval) = eval_at(&objective, pt)
        && fval.is_finite()
        && ((maximize && fval > *best_f) || (!maximize && fval < *best_f))
      {
        *best_f = fval;
        *best_x = pt.to_vec();
      }
    };

  // Determine which scale ranges to sample. Always include the full bounds,
  // plus tighter ranges when the bounds are wide.
  let mut scale_bounds: Vec<Vec<(f64, f64)>> = Vec::new();

  // Full bounds
  let full: Vec<(f64, f64)> = bounds.clone();
  scale_bounds.push(full);

  // Add tighter ranges when default bounds are wide
  for &scale in &[10.0, 100.0, 1000.0] {
    let tight: Vec<(f64, f64)> = bounds
      .iter()
      .map(|&(lo, hi)| {
        let range = hi - lo;
        if range > scale * 4.0 {
          let mid = (lo + hi) / 2.0;
          ((mid - scale).max(lo), (mid + scale).min(hi))
        } else {
          (lo, hi)
        }
      })
      .collect();
    if tight != *scale_bounds.last().unwrap() {
      scale_bounds.push(tight);
    }
  }

  for sb in &scale_bounds {
    let mut sample_points: Vec<Vec<f64>> = vec![vec![]];
    for i in 0..n {
      let (lo, hi) = sb[i];
      let mut new_points = Vec::new();
      for pt in &sample_points {
        for j in 0..=samples_per_dim {
          let t = j as f64 / samples_per_dim as f64;
          let val = lo + t * (hi - lo);
          let mut new_pt = pt.clone();
          new_pt.push(val);
          new_points.push(new_pt);
        }
      }
      sample_points = new_points;
    }

    for pt in &sample_points {
      update_best(pt, &mut best_x, &mut best_f, &eval_at);
    }
  }

  // Phase 2: Local refinement using golden section / gradient-free search
  // For each variable, refine using Brent-like narrowing
  let sign = if maximize { -1.0 } else { 1.0 };
  let max_iter = 200;
  let tol = 1e-12;

  // Try to compute symbolic gradients for gradient-based refinement
  let grad_exprs: Option<Vec<Expr>> = {
    let mut grads = Vec::new();
    let mut ok = true;
    for var in &vars {
      match crate::functions::calculus_ast::differentiate_expr(&objective, var)
      {
        Ok(d) => {
          let d = simplify(d);
          // Check for unevaluated D
          if contains_unevaluated_d(&d) {
            ok = false;
            break;
          }
          grads.push(d);
        }
        Err(_) => {
          ok = false;
          break;
        }
      }
    }
    if ok { Some(grads) } else { None }
  };

  let mut x = best_x;

  // Run gradient descent from a starting point, returning the optimized point and value.
  let run_gradient_descent =
    |start: Vec<f64>, grads: &[Expr]| -> (Vec<f64>, f64) {
      let mut x = start;
      for _ in 0..max_iter {
        let mut grad = vec![0.0; n];
        let mut grad_ok = true;
        for i in 0..n {
          match eval_at(&grads[i], &x) {
            Ok(g) if g.is_finite() => grad[i] = g,
            _ => {
              grad_ok = false;
              break;
            }
          }
        }
        if !grad_ok {
          break;
        }

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
          break;
        }

        let mut alpha = 0.1 / grad_norm.max(1.0);
        let current_f = eval_at(&objective, &x).unwrap_or(f64::INFINITY) * sign;

        for _ in 0..30 {
          let x_new: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, xi)| {
              let raw = xi - sign * alpha * grad[i];
              raw.clamp(bounds[i].0, bounds[i].1)
            })
            .collect();
          if let Ok(new_f) = eval_at(&objective, &x_new)
            && new_f.is_finite()
            && new_f * sign < current_f
          {
            x = x_new;
            break;
          }
          alpha *= 0.5;
          if alpha < 1e-20 {
            break;
          }
        }
      }
      let fval = eval_at(&objective, &x).unwrap_or(f64::INFINITY);
      (x, fval)
    };

  if let Some(ref grads) = grad_exprs {
    // Run gradient descent from the best sampled point
    let (x_opt, f_opt) = run_gradient_descent(x.clone(), grads);
    x = x_opt;

    // Check for saddle points by perturbing and re-running from nearby points.
    // This avoids getting stuck at local maxima or saddle points where
    // gradient is zero.
    let perturbations = [0.1, 1.0, 10.0];
    for eps in &perturbations {
      for i in 0..n {
        for &dir in &[-1.0, 1.0] {
          let mut x_perturbed = x.clone();
          x_perturbed[i] =
            (x_perturbed[i] + dir * eps).clamp(bounds[i].0, bounds[i].1);
          let (x_new, f_new) = run_gradient_descent(x_perturbed, grads);
          if f_new.is_finite()
            && ((maximize && f_new > f_opt) || (!maximize && f_new < f_opt))
          {
            let better = (maximize
              && f_new > eval_at(&objective, &x).unwrap_or(f64::NEG_INFINITY))
              || (!maximize
                && f_new < eval_at(&objective, &x).unwrap_or(f64::INFINITY));
            if better {
              x = x_new;
            }
          }
        }
      }
    }
  } else {
    // Gradient-free refinement: coordinate-wise golden section
    let golden_ratio = 0.6180339887498949;
    for _ in 0..max_iter {
      let mut improved = false;
      for i in 0..n {
        let (mut lo, mut hi) = bounds[i];
        // Narrow around current best
        let range = (hi - lo) * 0.1;
        lo = (x[i] - range).max(bounds[i].0);
        hi = (x[i] + range).min(bounds[i].1);

        let mut a = lo;
        let mut b = hi;
        let mut c = b - golden_ratio * (b - a);
        let mut d = a + golden_ratio * (b - a);

        for _ in 0..100 {
          if (b - a).abs() < tol {
            break;
          }
          let mut xc = x.clone();
          xc[i] = c;
          let mut xd = x.clone();
          xd[i] = d;

          let fc = eval_at(&objective, &xc).unwrap_or(f64::INFINITY) * sign;
          let fd = eval_at(&objective, &xd).unwrap_or(f64::INFINITY) * sign;

          if fc < fd {
            b = d;
            d = c;
            c = b - golden_ratio * (b - a);
          } else {
            a = c;
            c = d;
            d = a + golden_ratio * (b - a);
          }
        }

        let new_val = (a + b) / 2.0;
        if (new_val - x[i]).abs() > tol {
          x[i] = new_val;
          improved = true;
        }
      }
      if !improved {
        break;
      }
    }
  }

  // Nelder–Mead polish. Coordinate-wise gradient descent / golden section
  // stalls in curved valleys (e.g. Rosenbrock), where coordinated multi-
  // dimensional steps are needed. A simplex polish over the box follows such
  // valleys to the true optimum. Evaluations are compiled, so this is cheap.
  if n >= 2 {
    let sign_p = if maximize { -1.0 } else { 1.0 };
    let polish_obj = |p: &[f64]| -> f64 {
      // Reject points outside the box so the simplex respects the bounds.
      for i in 0..n {
        if p[i] < bounds[i].0 || p[i] > bounds[i].1 {
          return f64::INFINITY;
        }
      }
      match eval_at(&objective, p) {
        Ok(v) if v.is_finite() => sign_p * v,
        _ => f64::INFINITY,
      }
    };
    let cur = eval_at(&objective, &x)
      .map(|v| sign_p * v)
      .unwrap_or(f64::INFINITY);
    let polished = nelder_mead_min(&polish_obj, &x, 0.05, n);
    if polish_obj(&polished) < cur {
      x = polished;
    }
  }

  // Compute final value
  let opt_val = eval_at(&objective, &x)?;

  // Unboundedness probe: if the optimum landed on the artificial default
  // outer bound (±1e6) of a variable the user left unconstrained on that
  // side, push the variable much further out. If the objective keeps
  // improving by a real margin, the problem has no finite optimum.
  //
  // Restricted to affine objectives (constant gradient): that's the case
  // wolframscript reliably flags as unbounded. For nonlinear objectives it
  // instead returns a large finite boundary value, so don't probe those.
  let objective_is_affine = grad_exprs.as_ref().is_some_and(|g| {
    g.iter().all(|gi| {
      vars
        .iter()
        .all(|v| crate::functions::calculus_ast::is_constant_wrt(gi, v))
    })
  });
  for i in 0..n {
    if !objective_is_affine {
      break;
    }
    let at_hi = bounds[i].1 >= 1e6 - 1.0 && (x[i] - bounds[i].1).abs() < 1.0;
    let at_lo = bounds[i].0 <= -1e6 + 1.0 && (x[i] - bounds[i].0).abs() < 1.0;
    if !(at_hi || at_lo) {
      continue;
    }
    let mut probe = x.clone();
    probe[i] = if at_hi { 1e12 } else { -1e12 };
    if let Ok(pf) = eval_at(&objective, &probe)
      && pf.is_finite()
    {
      let margin = 1e-3 * (1.0 + opt_val.abs());
      let improves = if maximize {
        pf > opt_val + margin
      } else {
        pf < opt_val - margin
      };
      if improves {
        return Ok(nminimize_unbounded_result(&vars, maximize));
      }
    }
  }

  // Build the numeric result: {opt_val, {var -> val, ...}}
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(var, val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(var.clone())),
      replacement: Box::new(Expr::Real(*val)),
    })
    .collect();
  let numeric =
    Expr::List(vec![Expr::Real(opt_val), Expr::List(rules.into())].into());

  // The local optimizer converges only to within tolerance, so an exact
  // optimum like `(x-1)^2` at x->1 comes back as float noise
  // (`2.1*^-25` at `x->0.9999999999995`). wolframscript reports the clean
  // `{0., {x -> 1.}}`. Consult the symbolic Minimize/Maximize solver, which
  // closes such cases exactly, and numericize its answer. `pick_best_optimum`
  // keeps the symbolic candidate when it's at least as good (it's listed
  // first and a later candidate must improve by a real margin to displace it),
  // so the exact result wins over the numeric noise while genuinely better
  // numeric optima are still preferred.
  let sym_name = if maximize { "Maximize" } else { "Minimize" };
  let symbolic = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: sym_name.to_string(),
    args: args.to_vec().into(),
  })
  .ok()
  .filter(|sym| matches!(sym, Expr::List(items) if items.len() == 2))
  .and_then(|sym| {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![sym].into(),
    })
    .ok()
  });

  let candidates: Vec<Expr> =
    [symbolic, Some(numeric)].into_iter().flatten().collect();
  pick_best_optimum(candidates, &objective, &constraints, &vars, maximize)
}

/// A compiled numeric expression tree over the optimization variables.
///
/// Optimizers evaluate the objective and constraints thousands of times per
/// run. Re-cloning and re-evaluating the full `Expr` AST at every point is
/// orders of magnitude too slow to afford a thorough search. Compiling the
/// arithmetic once into this closed enum makes each evaluation a few hundred
/// floating-point ops, so many restarts / iterations stay cheap. Anything the
/// compiler doesn't recognise yields `None`, and the caller falls back to the
/// slow AST path.
enum NumNode {
  Const(f64),
  Var(usize),
  Neg(Box<NumNode>),
  Add(Vec<NumNode>),
  Mul(Vec<NumNode>),
  Sub(Box<NumNode>, Box<NumNode>),
  Div(Box<NumNode>, Box<NumNode>),
  Pow(Box<NumNode>, Box<NumNode>),
  Unary(fn(f64) -> f64, Box<NumNode>),
  Binary(fn(f64, f64) -> f64, Box<NumNode>, Box<NumNode>),
}

impl NumNode {
  fn eval(&self, point: &[f64]) -> f64 {
    match self {
      NumNode::Const(c) => *c,
      NumNode::Var(i) => point[*i],
      NumNode::Neg(a) => -a.eval(point),
      NumNode::Add(xs) => xs.iter().map(|x| x.eval(point)).sum(),
      NumNode::Mul(xs) => xs.iter().map(|x| x.eval(point)).product(),
      NumNode::Sub(a, b) => a.eval(point) - b.eval(point),
      NumNode::Div(a, b) => a.eval(point) / b.eval(point),
      NumNode::Pow(a, b) => {
        let base = a.eval(point);
        let exp = b.eval(point);
        // Integer exponents via powi keep `(-x)^2` real instead of NaN.
        if exp.fract() == 0.0 && exp.abs() < 1e9 {
          base.powi(exp as i32)
        } else {
          base.powf(exp)
        }
      }
      NumNode::Unary(f, a) => f(a.eval(point)),
      NumNode::Binary(f, a, b) => f(a.eval(point), b.eval(point)),
    }
  }
}

/// Compile an `Expr` over `vars` into a fast numeric closure tree, or `None`
/// if it contains anything not handled here (forcing the slow AST fallback).
fn compile_numeric(expr: &Expr, vars: &[String]) -> Option<NumNode> {
  let c = |e: &Expr| compile_numeric(e, vars);
  match expr {
    Expr::Integer(n) => Some(NumNode::Const(*n as f64)),
    Expr::BigInteger(n) => Some(NumNode::Const(n.to_string().parse().ok()?)),
    Expr::Real(r) => Some(NumNode::Const(*r)),
    Expr::Identifier(name) => vars
      .iter()
      .position(|v| v == name)
      .map(NumNode::Var)
      .or_else(|| named_constant_value(name).map(NumNode::Const)),
    Expr::Constant(name) => named_constant_value(name).map(NumNode::Const),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(NumNode::Neg(Box::new(c(operand)?))),
    Expr::BinaryOp { op, left, right } => {
      use BinaryOperator::*;
      let l = Box::new(c(left)?);
      let r = Box::new(c(right)?);
      match op {
        Plus => Some(NumNode::Add(vec![*l, *r])),
        Minus => Some(NumNode::Sub(l, r)),
        Times => Some(NumNode::Mul(vec![*l, *r])),
        Divide => Some(NumNode::Div(l, r)),
        Power => Some(NumNode::Pow(l, r)),
        _ => None,
      }
    }
    Expr::FunctionCall { name, args } => {
      let unary =
        |f: fn(f64) -> f64, args: &crate::ExprList| -> Option<NumNode> {
          if args.len() != 1 {
            return None;
          }
          Some(NumNode::Unary(
            f,
            Box::new(compile_numeric(&args[0], vars)?),
          ))
        };
      match name.as_str() {
        "Plus" => {
          Some(NumNode::Add(args.iter().map(&c).collect::<Option<_>>()?))
        }
        "Times" => {
          Some(NumNode::Mul(args.iter().map(&c).collect::<Option<_>>()?))
        }
        "Subtract" if args.len() == 2 => {
          Some(NumNode::Sub(Box::new(c(&args[0])?), Box::new(c(&args[1])?)))
        }
        "Divide" | "Rational" if args.len() == 2 => {
          Some(NumNode::Div(Box::new(c(&args[0])?), Box::new(c(&args[1])?)))
        }
        "Power" if args.len() == 2 => {
          Some(NumNode::Pow(Box::new(c(&args[0])?), Box::new(c(&args[1])?)))
        }
        "Minus" if args.len() == 1 => {
          Some(NumNode::Neg(Box::new(c(&args[0])?)))
        }
        "Sqrt" => unary(f64::sqrt, args),
        "Exp" => unary(f64::exp, args),
        "Sin" => unary(f64::sin, args),
        "Cos" => unary(f64::cos, args),
        "Tan" => unary(f64::tan, args),
        "Cot" => unary(|x| 1.0 / x.tan(), args),
        "Sec" => unary(|x| 1.0 / x.cos(), args),
        "Csc" => unary(|x| 1.0 / x.sin(), args),
        "ArcSin" => unary(f64::asin, args),
        "ArcCos" => unary(f64::acos, args),
        "ArcTan" if args.len() == 2 => Some(NumNode::Binary(
          |y, x| y.atan2(x),
          Box::new(c(&args[0])?),
          Box::new(c(&args[1])?),
        )),
        "ArcTan" => unary(f64::atan, args),
        "Sinh" => unary(f64::sinh, args),
        "Cosh" => unary(f64::cosh, args),
        "Tanh" => unary(f64::tanh, args),
        "Abs" => unary(f64::abs, args),
        "Sign" => unary(f64::signum, args),
        "Log" if args.len() == 2 => Some(NumNode::Binary(
          |b, x| x.ln() / b.ln(),
          Box::new(c(&args[0])?),
          Box::new(c(&args[1])?),
        )),
        "Log" => unary(f64::ln, args),
        "Log2" => unary(f64::log2, args),
        "Log10" => unary(f64::log10, args),
        "Floor" => unary(f64::floor, args),
        "Ceiling" => unary(f64::ceil, args),
        "Round" => unary(f64::round, args),
        "Min" => args
          .iter()
          .map(&c)
          .collect::<Option<Vec<_>>>()
          .filter(|nodes| !nodes.is_empty())
          .map(|nodes| fold_binary(nodes, f64::min)),
        "Max" => args
          .iter()
          .map(c)
          .collect::<Option<Vec<_>>>()
          .filter(|nodes| !nodes.is_empty())
          .map(|nodes| fold_binary(nodes, f64::max)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Reduce a non-empty list of nodes with an associative binary float op.
fn fold_binary(mut nodes: Vec<NumNode>, f: fn(f64, f64) -> f64) -> NumNode {
  let mut acc = nodes.remove(0);
  for n in nodes {
    acc = NumNode::Binary(f, Box::new(acc), Box::new(n));
  }
  acc
}

/// Numeric value of a named mathematical constant, if known.
fn named_constant_value(name: &str) -> Option<f64> {
  Some(match name {
    "Pi" => std::f64::consts::PI,
    "E" => std::f64::consts::E,
    "Degree" => std::f64::consts::PI / 180.0,
    "GoldenRatio" => (1.0 + 5.0_f64.sqrt()) / 2.0,
    "EulerGamma" => 0.577_215_664_901_532_9,
    "Catalan" => 0.915_965_594_177_219,
    _ => return None,
  })
}

/// An atomic comparison split out of a (possibly chained / And-joined)
/// constraint expression, e.g. `x*y >= 1` or `-3 <= x`.
struct AtomicComparison {
  left: Expr,
  op: ComparisonOp,
  right: Expr,
}

/// Flatten constraint expressions into a list of atomic comparisons so each
/// one can be scored individually for the penalty function.
fn collect_atomic_comparisons(constraints: &[Expr]) -> Vec<AtomicComparison> {
  let mut flat: Vec<&Expr> = Vec::new();
  for c in constraints {
    flatten_and_constraints_ref(c, &mut flat);
  }
  let mut out = Vec::new();
  for c in flat {
    if let Expr::Comparison {
      operands,
      operators,
    } = c
    {
      for i in 0..operators.len() {
        out.push(AtomicComparison {
          left: operands[i].clone(),
          op: operators[i],
          right: operands[i + 1].clone(),
        });
      }
    }
  }
  out
}

/// Build the result wolframscript returns for an infeasible problem:
/// emits an `NMinimize::nsol` / `NMaximize::nsol` message listing the
/// constraints and returns `{±Infinity, {v -> Indeterminate, ...}}`.
fn nminimize_infeasible_result(
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
) -> Expr {
  let func_name = if maximize { "NMaximize" } else { "NMinimize" };
  // List each constraint as wolframscript does: flatten `&&`, split chained
  // comparisons into atomics, and render anything else (e.g. `False`) as-is.
  let mut flat: Vec<&Expr> = Vec::new();
  for c in constraints {
    flatten_and_constraints_ref(c, &mut flat);
  }
  let mut constraint_strs: Vec<String> = Vec::new();
  for term in flat {
    if let Expr::Comparison {
      operands,
      operators,
    } = term
    {
      for i in 0..operators.len() {
        constraint_strs.push(crate::syntax::expr_to_output(
          &Expr::Comparison {
            operands: vec![operands[i].clone(), operands[i + 1].clone()],
            operators: vec![operators[i]],
          },
        ));
      }
    } else {
      constraint_strs.push(crate::syntax::expr_to_output(term));
    }
  }
  crate::emit_message(&format!(
    "{func_name}::nsol: There are no points that satisfy the constraints {{{}}}.",
    constraint_strs.join(", ")
  ));

  let inf = if maximize {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  } else {
    Expr::Identifier("Infinity".to_string())
  };
  let rules: Vec<Expr> = vars
    .iter()
    .map(|var| Expr::Rule {
      pattern: Box::new(Expr::Identifier(var.clone())),
      replacement: Box::new(Expr::Identifier("Indeterminate".to_string())),
    })
    .collect();
  Expr::List(vec![inf, Expr::List(rules.into())].into())
}

/// Build the result wolframscript returns for an unbounded problem:
/// emits an `ubnd` message and returns `{∓Infinity, {v -> Indeterminate}}`
/// (−Infinity when minimizing, +Infinity when maximizing).
fn nminimize_unbounded_result(vars: &[String], maximize: bool) -> Expr {
  let func_name = if maximize { "NMaximize" } else { "NMinimize" };
  crate::emit_message(&format!("{func_name}::ubnd: The problem is unbounded."));
  let inf = if maximize {
    Expr::Identifier("Infinity".to_string())
  } else {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    }
  };
  let rules: Vec<Expr> = vars
    .iter()
    .map(|var| Expr::Rule {
      pattern: Box::new(Expr::Identifier(var.clone())),
      replacement: Box::new(Expr::Identifier("Indeterminate".to_string())),
    })
    .collect();
  Expr::List(vec![inf, Expr::List(rules.into())].into())
}

/// Evaluate an expression numerically with `vars` bound to `point`,
/// returning NaN on any failure.
fn eval_expr_at_point(expr: &Expr, vars: &[String], point: &[f64]) -> f64 {
  let mut e = expr.clone();
  for (i, var) in vars.iter().enumerate() {
    e = crate::syntax::substitute_variable(&e, var, &Expr::Real(point[i]));
  }
  match crate::evaluator::evaluate_expr_to_expr(&e) {
    Ok(evaled) => expr_to_f64(&evaled).unwrap_or(f64::NAN),
    Err(_) => f64::NAN,
  }
}

/// Total constraint violation at a point (0 when feasible, +∞ if any
/// constraint can't be evaluated numerically).
fn constraint_violation(
  comparisons: &[AtomicComparison],
  vars: &[String],
  point: &[f64],
) -> f64 {
  use ComparisonOp::*;
  let mut total = 0.0;
  for c in comparisons {
    let l = eval_expr_at_point(&c.left, vars, point);
    let r = eval_expr_at_point(&c.right, vars, point);
    if !l.is_finite() || !r.is_finite() {
      return f64::INFINITY;
    }
    total += match c.op {
      Less | LessEqual => (l - r).max(0.0),
      Greater | GreaterEqual => (r - l).max(0.0),
      Equal => (l - r).abs(),
      _ => 0.0,
    };
  }
  total
}

/// Choose the best feasible result among optimizer candidates. Each candidate
/// is a `{value, {var -> val, ...}}` list. Prefers feasible candidates, then
/// the best objective value for the optimization direction.
fn pick_best_optimum(
  candidates: Vec<Expr>,
  objective: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let comparisons = collect_atomic_comparisons(constraints);
  let mut best: Option<(Expr, f64, bool)> = None;

  for cand in candidates {
    // Extract the point from the rule list.
    let Expr::List(items) = &cand else { continue };
    if items.len() != 2 {
      continue;
    }
    let Expr::List(rules) = &items[1] else {
      continue;
    };
    let mut point = vec![0.0; vars.len()];
    let mut ok = true;
    for (vi, var) in vars.iter().enumerate() {
      let mut found = None;
      for r in rules.iter() {
        let (pat, rep) = match r {
          Expr::Rule {
            pattern,
            replacement,
          } => (pattern.as_ref(), replacement.as_ref()),
          Expr::FunctionCall { name, args }
            if name == "Rule" && args.len() == 2 =>
          {
            (&args[0], &args[1])
          }
          _ => continue,
        };
        if matches!(pat, Expr::Identifier(n) if n == var) {
          found = expr_to_f64(rep).ok();
        }
      }
      match found {
        Some(v) => point[vi] = v,
        None => {
          ok = false;
          break;
        }
      }
    }
    if !ok {
      continue;
    }

    let obj = eval_expr_at_point(objective, vars, &point);
    if !obj.is_finite() {
      continue;
    }
    let feasible = constraint_violation(&comparisons, vars, &point) < 1e-6;

    let better = match &best {
      None => true,
      Some((_, best_obj, best_feasible)) => match (feasible, *best_feasible) {
        (true, false) => true,
        (false, true) => false,
        _ => {
          // Require a meaningful improvement so a later candidate's float
          // noise can't displace an equally-good (often exact) earlier one.
          let margin = 1e-6 * (1.0 + best_obj.abs());
          if maximize {
            obj > *best_obj + margin
          } else {
            obj < *best_obj - margin
          }
        }
      },
    };
    if better {
      best = Some((cand, obj, feasible));
    }
  }

  // No feasible candidate from any optimizer ⇒ the constraints are
  // unsatisfiable; mirror wolframscript's infeasible result.
  match best {
    Some((c, _, true)) => Ok(c),
    _ => Ok(nminimize_infeasible_result(constraints, vars, maximize)),
  }
}

/// Numeric penalty-method optimizer for problems whose constraints couple
/// several variables (e.g. `x*y >= 1`, `x^2 + y^2 <= 4`). Uses multi-start
/// Nelder–Mead simplex minimization of an arbitrary `n`-dimensional closure.
/// Builds an initial simplex of size `step` around `start` and returns the best
/// vertex found. Used both as the penalty-method inner solver and as a local
/// polish for the grid-sampler path.
fn nelder_mead_min(
  f: &dyn Fn(&[f64]) -> f64,
  start: &[f64],
  step: f64,
  n: usize,
) -> Vec<f64> {
  let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
  simplex.push(start.to_vec());
  for i in 0..n {
    let mut p = start.to_vec();
    let s = if step.abs() > 1e-12 { step } else { 0.1 };
    p[i] += if p[i].abs() > 1e-9 {
      p[i] * 0.05 + s
    } else {
      s
    };
    simplex.push(p);
  }
  let mut fvals: Vec<f64> = simplex.iter().map(|p| f(p)).collect();

  for _ in 0..600 {
    // Order vertices by value.
    let mut idx: Vec<usize> = (0..=n).collect();
    idx.sort_by(|&a, &b| {
      fvals[a]
        .partial_cmp(&fvals[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = idx[0];
    let worst = idx[n];
    let second_worst = idx[n - 1];

    // Convergence: simplex collapsed.
    let spread = (fvals[worst] - fvals[best]).abs();
    if spread < 1e-14 {
      break;
    }

    // Centroid of all but worst.
    let mut centroid = vec![0.0; n];
    for (k, vert) in simplex.iter().enumerate() {
      if k == worst {
        continue;
      }
      for d in 0..n {
        centroid[d] += vert[d] / n as f64;
      }
    }

    let reflect: Vec<f64> = (0..n)
      .map(|d| centroid[d] + (centroid[d] - simplex[worst][d]))
      .collect();
    let fr = f(&reflect);

    if fr < fvals[best] {
      // Expand.
      let expand: Vec<f64> = (0..n)
        .map(|d| centroid[d] + 2.0 * (centroid[d] - simplex[worst][d]))
        .collect();
      let fe = f(&expand);
      if fe < fr {
        simplex[worst] = expand;
        fvals[worst] = fe;
      } else {
        simplex[worst] = reflect;
        fvals[worst] = fr;
      }
    } else if fr < fvals[second_worst] {
      simplex[worst] = reflect;
      fvals[worst] = fr;
    } else {
      // Contract.
      let contract: Vec<f64> = (0..n)
        .map(|d| centroid[d] + 0.5 * (simplex[worst][d] - centroid[d]))
        .collect();
      let fc = f(&contract);
      if fc < fvals[worst] {
        simplex[worst] = contract;
        fvals[worst] = fc;
      } else {
        // Shrink toward best.
        for k in 0..=n {
          if k == best {
            continue;
          }
          for d in 0..n {
            simplex[k][d] =
              simplex[best][d] + 0.5 * (simplex[k][d] - simplex[best][d]);
          }
          fvals[k] = f(&simplex[k]);
        }
      }
    }
  }

  // Return current best vertex.
  let mut best = 0;
  for k in 1..=n {
    if fvals[k] < fvals[best] {
      best = k;
    }
  }
  simplex[best].clone()
}

/// Nelder–Mead on `objective + mu * violation` with penalty continuation.
fn nminimize_penalty(
  objective: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let n = vars.len();
  let sign = if maximize { -1.0 } else { 1.0 };
  let comparisons = collect_atomic_comparisons(constraints);
  let bounds = extract_bounds(constraints, vars)?;

  // Compile the objective and each constraint side into fast numeric closures
  // once. The optimizer evaluates these tens of thousands of times; the
  // compiled form is orders of magnitude faster than re-evaluating the AST and
  // lets the search run enough restarts to converge. Falls back to the AST
  // evaluator for anything the compiler can't handle.
  let obj_compiled = compile_numeric(objective, vars);
  let cmp_compiled: Vec<Option<(NumNode, NumNode)>> = comparisons
    .iter()
    .map(|c| {
      Some((
        compile_numeric(&c.left, vars)?,
        compile_numeric(&c.right, vars)?,
      ))
    })
    .collect();

  let eval_num = |expr: &Expr, point: &[f64]| -> f64 {
    eval_expr_at_point(expr, vars, point)
  };

  // Total constraint violation at a point (0 when feasible).
  let violation = |point: &[f64]| -> f64 {
    use ComparisonOp::*;
    let mut total = 0.0;
    for (c, compiled) in comparisons.iter().zip(cmp_compiled.iter()) {
      let (l, r) = match compiled {
        Some((lc, rc)) => (lc.eval(point), rc.eval(point)),
        None => (
          eval_expr_at_point(&c.left, vars, point),
          eval_expr_at_point(&c.right, vars, point),
        ),
      };
      if !l.is_finite() || !r.is_finite() {
        return f64::INFINITY;
      }
      total += match c.op {
        Less | LessEqual => (l - r).max(0.0),
        Greater | GreaterEqual => (r - l).max(0.0),
        Equal => (l - r).abs(),
        _ => 0.0,
      };
    }
    total
  };

  let obj_signed = |point: &[f64]| -> f64 {
    let f = match &obj_compiled {
      Some(node) => node.eval(point),
      None => eval_num(objective, point),
    };
    if f.is_finite() {
      sign * f
    } else {
      f64::INFINITY
    }
  };

  // Build a set of starting points by sampling a grid over a bounded region.
  let start_region: Vec<(f64, f64)> = bounds
    .iter()
    .map(|&(lo, hi)| {
      let lo = if lo <= -1e5 { -10.0 } else { lo };
      let hi = if hi >= 1e5 { 10.0 } else { hi };
      (lo, hi)
    })
    .collect();

  let per_dim = match n {
    1 => 41,
    2 => 13,
    3 => 7,
    _ => 4,
  };
  let mut starts: Vec<Vec<f64>> = vec![vec![]];
  for (lo, hi) in &start_region {
    let mut next = Vec::new();
    for pt in &starts {
      for j in 0..per_dim {
        let t = if per_dim == 1 {
          0.5
        } else {
          j as f64 / (per_dim - 1) as f64
        };
        let mut np = pt.clone();
        np.push(lo + t * (hi - lo));
        next.push(np);
      }
    }
    starts = next;
  }

  // Rank starts by penalized value (high penalty) and refine the best few.
  let rank = |p: &[f64]| -> f64 {
    let o = obj_signed(p);
    if o.is_finite() {
      o + 1e6 * violation(p)
    } else {
      f64::INFINITY
    }
  };
  starts.sort_by(|a, b| {
    rank(a)
      .partial_cmp(&rank(b))
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  starts.truncate(8);

  let mut best_point: Option<Vec<f64>> = None;
  let mut best_obj = f64::INFINITY;
  let mut best_viol = f64::INFINITY;

  let mus = [1e2_f64, 1e4, 1e6, 1e8, 1e10];
  for start in &starts {
    let mut cur = start.clone();
    for (k, &mu) in mus.iter().enumerate() {
      let penalized = |p: &[f64]| -> f64 {
        let o = obj_signed(p);
        if !o.is_finite() {
          return f64::INFINITY;
        }
        o + mu * violation(p)
      };
      let step = 0.5 / (k as f64 + 1.0);
      // A single Nelder–Mead pass often stalls with a collapsed simplex that
      // can't traverse along an active constraint (e.g. moving toward the
      // balanced point on an equality sphere). Restart from the converged
      // vertex with a freshly inflated simplex a few times to escape such
      // stalls; this is cheap and substantially improves convergence for
      // equality/coupling-constrained problems.
      let mut prev = f64::INFINITY;
      for _ in 0..12 {
        cur = nelder_mead_min(&penalized, &cur, step, n);
        let fv = penalized(&cur);
        if (prev - fv).abs() <= 1e-12 * (1.0 + fv.abs()) {
          break;
        }
        prev = fv;
      }
    }

    let o = obj_signed(&cur);
    let v = violation(&cur);
    if !o.is_finite() {
      continue;
    }
    // Prefer feasible points; among feasible, lowest objective. Among
    // infeasible only, lowest violation.
    let feasible = v < 1e-6;
    let best_feasible = best_viol < 1e-6;
    let better = match (feasible, best_feasible) {
      (true, false) => true,
      (false, true) => false,
      (true, true) => o < best_obj,
      (false, false) => v < best_viol || (v == best_viol && o < best_obj),
    };
    if best_point.is_none() || better {
      best_point = Some(cur);
      best_obj = o;
      best_viol = v;
    }
  }

  let point = best_point.ok_or_else(|| {
    InterpreterError::EvaluationError(
      "NMinimize: numeric optimization failed".into(),
    )
  })?;

  let opt_val = sign * best_obj;
  let rules: Vec<Expr> = vars
    .iter()
    .zip(point.iter())
    .map(|(var, val)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(var.clone())),
      replacement: Box::new(Expr::Real(*val)),
    })
    .collect();

  Ok(Expr::List(
    vec![Expr::Real(opt_val), Expr::List(rules.into())].into(),
  ))
}

/// Extract variable bounds from constraints.
/// Handles patterns like: 0 < x < Pi/2, x > 0, x < 1, etc.
/// Returns (lower_bound, upper_bound) for each variable.
fn extract_bounds(
  constraints: &[Expr],
  vars: &[String],
) -> Result<Vec<(f64, f64)>, InterpreterError> {
  let mut bounds: Vec<(f64, f64)> = vars.iter().map(|_| (-1e6, 1e6)).collect();

  for constraint in constraints {
    // Flatten And expressions
    let mut flat = Vec::new();
    flatten_and_constraints_ref(constraint, &mut flat);
    for c in flat {
      extract_bound_from_comparison(c, vars, &mut bounds)?;
    }
  }

  Ok(bounds)
}

fn flatten_and_constraints_ref<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => {
      flatten_and_constraints_ref(left, out);
      flatten_and_constraints_ref(right, out);
    }
    // `a && b && c` is parsed/evaluated as a nested `And[...]` FunctionCall,
    // so flatten that form too.
    Expr::FunctionCall { name, args } if name == "And" => {
      for a in args.iter() {
        flatten_and_constraints_ref(a, out);
      }
    }
    _ => out.push(expr),
  }
}

fn extract_bound_from_comparison(
  expr: &Expr,
  vars: &[String],
  bounds: &mut [(f64, f64)],
) -> Result<(), InterpreterError> {
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
  {
    // Handle chained comparisons like 0 < x < Pi/2
    for i in 0..operators.len() {
      let left = &operands[i];
      let right = &operands[i + 1];
      let op = &operators[i];

      // Try to identify if left or right is a variable and the other is a number
      for (vi, var) in vars.iter().enumerate() {
        let left_is_var = matches!(left, Expr::Identifier(n) if n == var);
        let right_is_var = matches!(right, Expr::Identifier(n) if n == var);

        if left_is_var {
          // var < value or var <= value
          if let Ok(val) = eval_to_f64(right) {
            match op {
              ComparisonOp::Less | ComparisonOp::LessEqual => {
                bounds[vi].1 = bounds[vi].1.min(val);
              }
              ComparisonOp::Greater | ComparisonOp::GreaterEqual => {
                bounds[vi].0 = bounds[vi].0.max(val);
              }
              _ => {}
            }
          }
        } else if right_is_var {
          // value < var or value <= var
          if let Ok(val) = eval_to_f64(left) {
            match op {
              ComparisonOp::Less | ComparisonOp::LessEqual => {
                bounds[vi].0 = bounds[vi].0.max(val);
              }
              ComparisonOp::Greater | ComparisonOp::GreaterEqual => {
                bounds[vi].1 = bounds[vi].1.min(val);
              }
              _ => {}
            }
          }
        }
      }
    }
  }
  Ok(())
}

/// Try to evaluate an expression to f64.
fn eval_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    _ => {
      // Try evaluating via N[]
      let n_expr = Expr::FunctionCall {
        name: "N".to_string(),
        args: vec![expr.clone()].into(),
      };
      let evaled = crate::evaluator::evaluate_expr_to_expr(&n_expr)?;
      match &evaled {
        Expr::Real(r) => Ok(*r),
        Expr::Integer(n) => Ok(*n as f64),
        _ => {
          let evaled2 = crate::evaluator::evaluate_expr_to_expr(expr)?;
          expr_to_f64(&evaled2)
        }
      }
    }
  }
}

// ── FindInstance implementation ──────────────────────────────────────

/// FindInstance[cond, vars] — find 1 instance satisfying condition
/// FindInstance[cond, vars, n] — find n instances
/// FindInstance[cond, vars, domain] — find in domain (Integers, Reals, etc.)
/// FindInstance[cond, vars, domain, n] — find n instances in domain
pub fn find_instance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(
      "FindInstance requires 2 to 4 arguments".into(),
    ));
  }

  let cond = &args[0];
  let vars = &args[1];

  // Parse optional n and domain from args[2..]
  let mut n: usize = 1;
  let mut domain: Option<String> = None;

  for arg in &args[2..] {
    match arg {
      Expr::Integer(k) if *k >= 0 => {
        n = *k as usize;
      }
      Expr::Identifier(name)
        if matches!(
          name.as_str(),
          "Integers" | "Reals" | "Complexes" | "Rationals" | "Booleans"
        ) =>
      {
        domain = Some(name.clone());
      }
      _ => {}
    }
  }

  // Extract variable names
  let var_names: Vec<String> = match vars {
    Expr::List(items) => items
      .iter()
      .filter_map(|v| {
        if let Expr::Identifier(name) = v {
          Some(name.clone())
        } else {
          None
        }
      })
      .collect(),
    Expr::Identifier(name) => vec![name.clone()],
    _ => vec![],
  };

  // Try Solve first (suppress warnings from Solve)
  let mut solutions: Vec<Expr> = {
    crate::push_quiet();
    let solve_result = solve_ast(&[cond.clone(), vars.clone()]);
    crate::pop_quiet();
    match &solve_result {
      Ok(Expr::List(sols)) if !sols.is_empty() => {
        // Filter out parametric solutions (solutions with free variables)
        // FindInstance needs concrete values, not expressions in terms of
        // other variables.
        sols
          .iter()
          .filter(|sol| {
            if let Expr::List(rules) = sol {
              // Check that all rules map to concrete values (no free vars)
              let solved: Vec<&str> = rules
                .iter()
                .filter_map(|r| {
                  if let Expr::Rule { pattern, .. } = r
                    && let Expr::Identifier(name) = pattern.as_ref()
                  {
                    return Some(name.as_str());
                  }
                  None
                })
                .collect();
              // A solution is concrete if all requested vars are solved
              // and the replacements don't contain unsolved vars
              let all_vars_solved =
                var_names.iter().all(|v| solved.contains(&v.as_str()));
              if !all_vars_solved {
                return false;
              }
              // Check replacements don't contain other requested vars
              rules.iter().all(|r| {
                if let Expr::Rule { replacement, .. } = r {
                  !var_names.iter().any(|v| !is_constant_wrt(replacement, v))
                } else {
                  true
                }
              })
            } else {
              true
            }
          })
          .cloned()
          .collect()
      }
      _ => Vec::new(),
    }
  };

  // If Solve failed or returned no solutions, try numerical search
  if solutions.is_empty() && !var_names.is_empty() {
    solutions = find_instance_numerical(cond, &var_names, n, domain.as_deref());
  }

  if solutions.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Filter by domain if specified
  let filtered = if let Some(ref dom) = domain {
    match dom.as_str() {
      "Integers" => solutions
        .into_iter()
        .filter(solution_is_integer)
        .collect::<Vec<_>>(),
      "Reals" => solutions
        .into_iter()
        .filter(solution_is_real)
        .collect::<Vec<_>>(),
      _ => solutions,
    }
  } else {
    solutions
  };

  if filtered.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Take at most n solutions from the end (Wolfram picks the largest solutions first)
  let len = filtered.len();
  let result: Vec<Expr> = if len <= n {
    filtered
  } else {
    filtered.into_iter().skip(len - n).collect()
  };
  Ok(Expr::List(result.into()))
}

/// Try to find instances numerically by evaluating the condition at sample points.
fn find_instance_numerical(
  cond: &Expr,
  var_names: &[String],
  n: usize,
  domain: Option<&str>,
) -> Vec<Expr> {
  use crate::evaluator::evaluate_expr_to_expr;
  use crate::functions::plot::substitute_var;

  let is_integer_domain = domain == Some("Integers");

  // Sample range and step
  let (range_lo, range_hi, step) = if is_integer_domain {
    (-100i64, 100i64, 1i64)
  } else {
    // Use integer grid for simplicity, then convert
    (-100i64, 100i64, 1i64)
  };

  let mut results: Vec<Expr> = Vec::new();

  // For single variable, simple scan
  if var_names.len() == 1 {
    let var = &var_names[0];
    let mut val = range_lo;
    while val <= range_hi && results.len() < n {
      let test_val: Expr = if is_integer_domain {
        Expr::Integer(val as i128)
      } else {
        Expr::Integer(val as i128)
      };
      let subst = substitute_var(cond, var, &test_val);
      if let Ok(evaled) = evaluate_expr_to_expr(&subst)
        && matches!(evaled, Expr::Identifier(ref s) if s == "True")
      {
        results.push(Expr::List(
          vec![Expr::Rule {
            pattern: Box::new(Expr::Identifier(var.clone())),
            replacement: Box::new(test_val),
          }]
          .into(),
        ));
      }
      val += step;
    }
  } else if var_names.len() == 2 {
    // For two variables, scan a grid
    let var1 = &var_names[0];
    let var2 = &var_names[1];
    let step2 = if is_integer_domain { 1i64 } else { 1i64 };
    let mut val1 = range_lo;
    'outer: while val1 <= range_hi && results.len() < n {
      let test1 = Expr::Integer(val1 as i128);
      let mut val2 = range_lo;
      while val2 <= range_hi && results.len() < n {
        let test2 = Expr::Integer(val2 as i128);
        let subst =
          substitute_var(&substitute_var(cond, var1, &test1), var2, &test2);
        if let Ok(evaled) = evaluate_expr_to_expr(&subst)
          && matches!(evaled, Expr::Identifier(ref s) if s == "True")
        {
          results.push(Expr::List(
            vec![
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(var1.clone())),
                replacement: Box::new(test1.clone()),
              },
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(var2.clone())),
                replacement: Box::new(test2),
              },
            ]
            .into(),
          ));
          if results.len() >= n {
            break 'outer;
          }
        }
        val2 += step2;
      }
      val1 += step;
    }
  }
  // For 3+ variables, could extend but keep simple for now

  results
}

/// Check if all values in a solution are integers
fn solution_is_integer(sol: &Expr) -> bool {
  if let Expr::List(rules) = sol {
    rules.iter().all(|rule| {
      if let Expr::Rule { replacement, .. } = rule {
        is_integer_expr(replacement)
      } else {
        false
      }
    })
  } else {
    false
  }
}

/// Check if an expression is an integer value
fn is_integer_expr(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) => true,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_integer_expr(operand),
    _ => {
      // Try evaluating to see if it's an integer
      if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(expr) {
        matches!(evaled, Expr::Integer(_))
      } else {
        false
      }
    }
  }
}

/// Check if all values in a solution are real (not complex)
fn solution_is_real(sol: &Expr) -> bool {
  if let Expr::List(rules) = sol {
    rules.iter().all(|rule| {
      if let Expr::Rule { replacement, .. } = rule {
        !contains_complex(replacement)
      } else {
        true
      }
    })
  } else {
    true
  }
}
