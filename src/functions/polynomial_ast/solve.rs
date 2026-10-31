use super::together::negate_expr;
use super::*;
use crate::functions::calculus_ast::{
  differentiate_expr, is_constant_wrt, simplify,
};
use crate::functions::math_ast::{
  make_rational, n_ast, try_eval_to_f64, try_extract_complex_float,
};
use crate::{ENV, StoredValue};

/// In Solve context, simplify Sqrt[expr^(2n)] → expr^n since ± handles the sign.
/// Also simplifies products containing such terms.
/// `Sqrt[c s1^e1 s2^e2 …]` of a monomial taken apart the way Solve
/// reports roots: the even powers of the symbols come out
/// (`Sqrt[a^2 b] → a Sqrt[b]`) and a negative coefficient becomes a factor
/// `I` (`Sqrt[-a] → I Sqrt[a]`). `None` for anything but such a monomial.
fn split_monomial_sqrt(arg: &Expr) -> Option<Expr> {
  split_monomial_root(arg, 2)
}

/// The `n`th-root counterpart of [`split_monomial_sqrt`]: whole `n`th
/// powers of the symbols come out (`(a^3 b)^(1/3) → a b^(1/3)`). Only a
/// square root turns a negative coefficient into `I`; for a higher root
/// that is `None`.
fn split_monomial_root(arg: &Expr, n: i128) -> Option<Expr> {
  fn factors(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Times" => {
        args.iter().for_each(|a| factors(a, out));
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        factors(left, out);
        factors(right, out);
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        out.push(Expr::Integer(-1));
        factors(operand, out);
      }
      other => out.push(other.clone()),
    }
  }
  let mut parts = Vec::new();
  factors(arg, &mut parts);
  let mut coefficient = Expr::Integer(1);
  let mut outside = Vec::new();
  let mut inside = Vec::new();
  let mut inside_denominator = Vec::new();
  for part in parts {
    if crate::functions::predicate_ast::is_numeric_q(&part)
      && try_eval_to_f64(&part).is_some()
    {
      coefficient = times2(coefficient, part);
      continue;
    }
    let (base, exponent) = match &part {
      Expr::Identifier(_) => (part.clone(), 1),
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(left.as_ref(), Expr::Identifier(_)) => match right.as_ref()
      {
        Expr::Integer(n) if *n != 0 => (*left.clone(), *n),
        _ => return None,
      },
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(args[0], Expr::Identifier(_)) =>
      {
        match &args[1] {
          Expr::Integer(n) if *n != 0 => (args[0].clone(), *n),
          _ => return None,
        }
      }
      _ => return None,
    };
    // Whole multiples of n come out; the rest stays under the root, in the
    // denominator for a negative power (`Sqrt[a/b]` is `Sqrt[a]/Sqrt[b]`).
    let whole = exponent / n;
    if whole != 0 {
      outside.push(pow2(base.clone(), Expr::Integer(whole)));
    }
    let rest = exponent % n;
    if rest > 0 {
      inside.push(pow2(base, Expr::Integer(rest)));
    } else if rest < 0 {
      inside_denominator.push(pow2(base, Expr::Integer(-rest)));
    }
  }
  if outside.is_empty() && inside.is_empty() && inside_denominator.is_empty() {
    return None;
  }
  let c = try_eval_to_f64(
    &crate::evaluator::evaluate_expr_to_expr(&coefficient).ok()?,
  )?;
  let mut result = outside;
  if c < 0.0 {
    if n != 2 {
      return None;
    }
    result.push(Expr::Identifier("I".to_string()));
    coefficient = negate_expr(&coefficient);
  }
  let root = |e: Expr| {
    if n == 2 {
      make_sqrt(e)
    } else {
      pow2(e, make_rational(1, n))
    }
  };
  result.push(root(times2(coefficient, times(inside))));
  if !inside_denominator.is_empty() {
    result.push(pow2(root(times(inside_denominator)), Expr::Integer(-1)));
  }
  crate::evaluator::evaluate_expr_to_expr(&times(result)).ok()
}

fn strip_sqrt_square(expr: Expr) -> Expr {
  match &expr {
    // Sqrt[c s^e …] of a monomial: even powers out, a negative sign as I.
    e if is_sqrt(e).is_some_and(|arg| split_monomial_sqrt(arg).is_some()) => {
      split_monomial_sqrt(is_sqrt(e).unwrap()).unwrap()
    }
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
        }
        return pow2(*base.clone(), Expr::Integer(half));
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
      times2(new_left, new_right)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let new_args: Vec<Expr> =
        args.iter().map(|a| strip_sqrt_square(a.clone())).collect();
      times(new_args)
    }
    _ => expr,
  }
}

// ─── NSolve ─────────────────────────────────────────────────────────

/// True when `expr` already states a (possibly quantified) condition, i.e. it
/// is an equation, an inequality, a logical combination of those, or a literal
/// truth value. Anything else is a bare expression that `NSolve` reads as
/// `expr == 0`.
fn is_relational_expr(expr: &Expr) -> bool {
  match expr {
    Expr::Comparison { .. } => true,
    Expr::BinaryOp {
      op: BinaryOperator::And | BinaryOperator::Or,
      ..
    } => true,
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      ..
    } => true,
    Expr::Identifier(s) | Expr::Constant(s) => s == "True" || s == "False",
    Expr::FunctionCall { name, .. } => matches!(
      name.as_str(),
      "Equal"
        | "Unequal"
        | "Less"
        | "LessEqual"
        | "Greater"
        | "GreaterEqual"
        | "Inequality"
        | "And"
        | "Or"
        | "Not"
        | "Xor"
        | "Nand"
        | "Nor"
        | "Implies"
        | "Element"
        | "ForAll"
        | "Exists"
    ),
    _ => false,
  }
}

/// `NSolve` accepts a bare polynomial where an equation is expected:
/// `NSolve[poly, x]` means `NSolve[poly == 0, x]`. A list threads
/// element-wise, so `NSolve[{p, q}, {x, y}]` is `NSolve[{p == 0, q == 0}, …]`
/// and a mixed list keeps the parts that already are equations.
///
/// Returns `None` when nothing needs rewriting, so the common path stays
/// allocation-free.
fn equations_from_bare_polynomials(expr: &Expr) -> Option<Expr> {
  let eq_zero = |e: &Expr| Expr::Comparison {
    operands: vec![e.clone(), Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };
  // `{p, q} == 0` (a list against a scalar) is the system `{p == 0, q == 0}`.
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && let [Expr::List(items), rhs] = operands.as_slice()
    && !matches!(rhs, Expr::List(_))
    && matches!(operators.as_slice(), [ComparisonOp::Equal])
  {
    return Some(Expr::List(
      items
        .iter()
        .map(|it| Expr::Comparison {
          operands: vec![it.clone(), rhs.clone()],
          operators: vec![ComparisonOp::Equal],
        })
        .collect(),
    ));
  }
  match expr {
    Expr::List(items) => {
      if items.iter().all(is_relational_expr) {
        return None;
      }
      Some(Expr::List(
        items
          .iter()
          .map(|it| {
            if is_relational_expr(it) {
              it.clone()
            } else {
              eq_zero(it)
            }
          })
          .collect(),
      ))
    }
    _ if is_relational_expr(expr) => None,
    _ => Some(eq_zero(expr)),
  }
}

/// True when the optional third argument of `NSolve` is a working-precision
/// specification (`NSolve[eqns, vars, prec]`) rather than a solution domain
/// like `Reals`. The precision only controls how many digits the roots carry,
/// so the solving path itself ignores it.
fn is_precision_spec(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_) => true,
    Expr::Identifier(s) | Expr::Constant(s) => s == "MachinePrecision",
    _ => false,
  }
}

/// Whether `expr` holds a square root or any other non-integer power. The
/// exact elimination of such equations is slow and squares radicals into
/// spurious roots, so inexact radical systems skip it.
fn has_radical(expr: &Expr) -> bool {
  is_sqrt(expr).is_some()
    || match expr {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        right,
        ..
      } => !matches!(right.as_ref(), Expr::Integer(_)),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        !matches!(args[1], Expr::Integer(_))
      }
      _ => false,
    }
    || crate::syntax::expr_children(expr)
      .into_iter()
      .any(has_radical)
}

/// NSolve[equation, var] — solve an equation numerically.
///
/// For quadratic polynomials, uses Kahan's numerically stable formula to
/// match Wolfram's machine-precision output. For all other equations,
/// solves symbolically first, then converts to numerical form via N[].
///
/// `NSolve[equation]` with the variable left out solves for whatever the
/// equation contains, the way `Solve[equation]` does.
/// The polynomial `Σ coeffs[k] var^k`.
fn build_poly_from_coeffs(coeffs: &[Expr], var: &str) -> Expr {
  plus(
    coeffs
      .iter()
      .enumerate()
      .filter(|(_, c)| !matches!(c, Expr::Integer(0)))
      .map(|(k, c)| {
        times(vec![c.clone(), pow(id_expr(var), Expr::Integer(k as i128))])
      })
      .collect(),
  )
}

/// Solve `poly == 0` for `var` factor by factor when `FactorList` splits it
/// into two or more factors in `var`; `None` when it does not split.
fn solve_by_factor_list(
  poly: &Expr,
  var: &str,
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let factors = crate::evaluator::evaluate_function_call_ast(
    "FactorList",
    &[crate::evaluator::evaluate_expr_to_expr(poly)?],
  )?;
  let Expr::List(factors) = &factors else {
    return Ok(None);
  };
  let mut in_var: Vec<(Expr, i128)> = Vec::new();
  for f in factors {
    let Expr::List(pair) = f else { return Ok(None) };
    let [factor, Expr::Integer(m)] = pair.as_slice() else {
      return Ok(None);
    };
    if !is_constant_wrt(factor, var) {
      in_var.push((factor.clone(), *m));
    }
  }
  if in_var.len() < 2 && in_var.iter().all(|(_, m)| *m == 1) {
    return Ok(None);
  }
  let mut all = Vec::new();
  for (factor, multiplicity) in in_var {
    let eq = Expr::Comparison {
      operands: vec![factor, Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    };
    let solved = solve_ast(&[eq, args[1].clone()])?;
    let Expr::List(sols) = &solved else {
      return Ok(None);
    };
    for _ in 0..multiplicity {
      all.extend(sols.iter().cloned());
    }
  }
  sort_solutions(&mut all);
  Ok(Some(Expr::List(all.into())))
}

/// `Solve`/`NSolve[lhs == rhs, t]` with an InterpolatingFunction applied to
/// `t`: Wolfram inverts the interpolating function, reporting `head::ifun`,
/// and gives the first root in its domain. The root is located by a scan of the
/// domain for a sign change and refined by bisection.
fn solve_interpolating_function(head: &str, args: &[Expr]) -> Option<Expr> {
  fn domain_of(e: &Expr) -> Option<(f64, f64)> {
    if let Expr::CurriedCall { func, .. } = e
      && let Expr::FunctionCall { name, args } = func.as_ref()
      && name == "InterpolatingFunction"
      && let Some(Expr::List(dom)) = args.first()
      && let Some(Expr::List(ab)) = dom.first()
      && let [a, b] = ab.as_slice()
    {
      return Some((try_eval_to_f64(a)?, try_eval_to_f64(b)?));
    }
    crate::syntax::expr_children(e)
      .into_iter()
      .find_map(domain_of)
  }
  let [eq, var] = args else {
    return None;
  };
  let Expr::Identifier(var_name) = var else {
    return None;
  };
  let Expr::Comparison {
    operands,
    operators,
  } = eq
  else {
    return None;
  };
  if operands.len() != 2 || operators[0] != ComparisonOp::Equal {
    return None;
  }
  let (lo, hi) = domain_of(eq)?;
  if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
    return None;
  }
  let difference = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(operands[0].clone()),
    right: Box::new(operands[1].clone()),
  };
  let g = |t: f64| -> Option<f64> {
    let at =
      crate::syntax::substitute_variable(&difference, var_name, &Expr::Real(t));
    try_eval_to_f64(&crate::evaluator::evaluate_expr_to_expr(&at).ok()?)
  };
  crate::emit_message(&format!(
    "{head}::ifun: Inverse functions are being used by {head}, so some solutions may not be found; use Reduce for complete solution information."
  ));
  const SAMPLES: usize = 2000;
  let mut prev_t = lo;
  let mut prev = g(lo)?;
  let mut root = (prev == 0.0).then_some(lo);
  for i in 1..=SAMPLES {
    if root.is_some() {
      break;
    }
    let t = lo + (hi - lo) * i as f64 / SAMPLES as f64;
    let v = g(t)?;
    if v == 0.0 {
      root = Some(t);
    } else if v.signum() != prev.signum() {
      let (mut a, mut b, mut fa) = (prev_t, t, prev);
      for _ in 0..200 {
        let m = f64::midpoint(a, b);
        let fm = g(m)?;
        if fm == 0.0 || (b - a) <= f64::EPSILON * m.abs().max(1.0) {
          a = m;
          b = m;
          break;
        }
        if fm.signum() == fa.signum() {
          a = m;
          fa = fm;
        } else {
          b = m;
        }
      }
      root = Some(f64::midpoint(a, b));
    }
    prev_t = t;
    prev = v;
  }
  Some(Expr::List(
    root
      .map(|t| {
        Expr::List(
          vec![Expr::Rule {
            pattern: Box::new(var.clone()),
            replacement: Box::new(Expr::Real(t)),
          }]
          .into(),
        )
      })
      .into_iter()
      .collect(),
  ))
}

pub fn nsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Some(first) = args.first()
    && contains_inequality_call(first)
  {
    let mut normalized = args.to_vec();
    normalized[0] = super::reduce::inequalities_to_comparisons(first);
    return nsolve_ast(&normalized);
  }
  let _head = IfunHead::new("NSolve");
  let _nsolve = NSolveScope::new();
  // `NSolve[poly, x]` is `NSolve[poly == 0, x]`, and a trailing precision
  // argument is not a domain — normalize both away before solving.
  let normalized_owned: Vec<Expr>;
  let args = {
    let rewritten_first =
      args.first().and_then(equations_from_bare_polynomials);
    // `NSolve[eqns, n]` with a number in the variable slot is the legacy
    // precision form, solving for the variables the equations contain.
    let precision_at = match args.len() {
      4 if is_precision_spec(&args[3]) => Some(3),
      3 if is_precision_spec(&args[2]) => Some(2),
      2 if is_precision_spec(&args[1]) => Some(1),
      _ => None,
    };
    if rewritten_first.is_some() || precision_at.is_some() {
      let mut new_args = args.to_vec();
      if let Some(first) = rewritten_first {
        new_args[0] = first;
      }
      if let Some(at) = precision_at {
        new_args.truncate(at);
      }
      normalized_owned = new_args;
      normalized_owned.as_slice()
    } else {
      args
    }
  };
  if let Some(result) = solve_interpolating_function("NSolve", args) {
    return Ok(result);
  }
  // `NSolve[eqns]` solves for the variables `Solve[eqns]` would pick.
  if args.len() == 1
    && !matches!(&args[0], Expr::Identifier(s) if s == "True" || s == "False")
    && let Some(vars) = auto_detect_solve_vars(&args[0])
  {
    return nsolve_ast(&[args[0].clone(), vars]);
  }
  // An unknown that is a call (`f''[0]`, `a[1]`) is solved for as a fresh
  // symbol: numericizing the equations would turn its `0` into `0.`.
  if let Some(result) = with_generalized_variables(args, nsolve_ast) {
    return result;
  }
  // A system of several equations with machine coefficients is solved the
  // way wolframscript does it: rationalize, solve exactly, numericize. The
  // inexact elimination drops branches — a factor `x1` that a machine
  // coefficient keeps from cancelling cleanly loses the `x1 -> 0` family
  // (PhasePlaneAnalysisOfABiochemicalReactor). Only coefficients that are
  // close to simple rationals are taken this way: a machine number such as a
  // computed coordinate rationalizes to a 53-bit fraction whose radicals
  // made a line-circle intersection cost seconds.
  if let Expr::List(eqs) = &args[0]
    && eqs.len() >= 2
    && crate::functions::math_ast::contains_inexact_real(&args[0])
    && !has_radical(&args[0])
    && let Ok(exact) = crate::evaluator::evaluate_function_call_ast(
      "Rationalize",
      &[args[0].clone()],
    )
    && !crate::functions::math_ast::contains_inexact_real(&exact)
  {
    let mut exact_args = args.to_vec();
    exact_args[0] = exact;
    let solved = solve_ast(&exact_args)?;
    if let Expr::List(sols) = &solved
      && !sols.is_empty()
    {
      let numerized = nsolve_numerize(&solved)?;
      let filtered = if args.len() >= 2
        && matches!(args.last(), Some(Expr::Identifier(d) | Expr::Constant(d)) if d == "Reals")
      {
        filter_real_nsolve_solutions(numerized)
      } else {
        numerized
      };
      return Ok(sort_nsolve_solutions(filtered));
    }
  }
  // Equations that already carry a machine number are solved with every
  // coefficient numericized: exact irrational coefficients (Sin[Pi/24],
  // Sqrt[3]) mixed with machine numbers otherwise reach the symbolic
  // elimination, which loses the roots (2FamiliesOfFourPointTouchingSquares).
  let numericized_owned: Vec<Expr>;
  let args = if crate::functions::math_ast::contains_inexact_real(&args[0])
    && let Ok(numeric) =
      crate::evaluator::evaluate_function_call_ast("N", &args[..1])
    && crate::syntax::expr_to_string(&numeric)
      != crate::syntax::expr_to_string(&args[0])
  {
    let mut new_args = args.to_vec();
    new_args[0] = numeric;
    numericized_owned = new_args;
    numericized_owned.as_slice()
  } else {
    args
  };
  // Try numerically stable quadratic formula for degree-2 polynomials
  if let Some(result) = try_nsolve_quadratic(args) {
    return result;
  }
  // Roots of a x^n + c on their circle, computed once from the modulus and
  // the angles rather than from the individual symbolic radicals.
  if let Some(result) = try_nsolve_pure_power(args) {
    return result;
  }
  // Polynomials with non-real numeric coefficients have no real-coefficient
  // structure to exploit; solve them with a complex-coefficient iteration.
  if let Some(result) = try_nsolve_complex_coefficients(args) {
    return result;
  }
  // Fall back to symbolic solve + numerize. An infinity the exact
  // elimination runs into is its own business (the solutions it spoils are
  // dropped below); NSolve reports none.
  crate::push_quiet_filter(
    Some(vec![
      "Power::infy".to_string(),
      "Infinity::indet".to_string(),
    ]),
    vec![],
  );
  let symbolic = solve_ast(args);
  crate::pop_quiet_filter();
  let symbolic = symbolic?;

  // An elimination whose numeric roots verify against no equation (their
  // rounding leaves residuals the exact check rejects) finds nothing; the
  // numeric pair solver below polishes such roots instead.
  if matches!(&symbolic, Expr::List(sols) if sols.is_empty())
    && let Some(result) = try_nsolve_polynomial_pair(args)
    && matches!(&result, Expr::List(sols) if !sols.is_empty())
  {
    return Ok(result);
  }
  // A square system whose elimination leaves some unknown unbound (a branch
  // like `y -> 0` that never pins down `x`, as radical equations give) is
  // only partly solved; the numeric system solver then finds the complete
  // solutions.
  if let Expr::List(sols) = &symbolic
    && !sols.is_empty()
    && let Expr::List(vars) = &args[1]
    && vars.len() >= 2
    && sols
      .iter()
      .any(|s| matches!(s, Expr::List(rules) if rules.len() < vars.len()))
    && let Some(found) = try_nsolve_numeric_system(args)
  {
    return Ok(sort_nsolve_solutions(found));
  }
  // What Solve leaves unevaluated, NSolve leaves unevaluated too — under
  // its own head — unless a numeric multi-seed search below finds real
  // roots that no symbolic technique reached in closed form.
  if matches!(&symbolic, Expr::FunctionCall { name, .. } if name == "Solve") {
    if let Some(result) = try_nsolve_bracketed(args) {
      return Ok(result);
    }
    if let Some(result) = try_nsolve_numeric_search(args) {
      return Ok(sort_nsolve_solutions(result));
    }
    // A two-variable polynomial system whose exact elimination ends in roots
    // with no radical form: eliminate numerically.
    if let Some(result) = try_nsolve_polynomial_pair(args) {
      return Ok(result);
    }
    return Ok(unevaluated("NSolve", args));
  }
  // The exact-residual checks of the symbolic elimination can discard every
  // numeric candidate of a system; the numeric elimination then still finds
  // its solutions.
  if matches!(&symbolic, Expr::List(sols) if sols.is_empty())
    && let Some(found) = try_nsolve_polynomial_pair(args)
    && matches!(&found, Expr::List(sols) if !sols.is_empty())
  {
    return Ok(found);
  }
  let numerized = nsolve_numerize(&symbolic)?;
  // An elimination that ran into an infinity (radicals of machine numbers
  // squared into huge cancelling terms) proves nothing: drop such
  // solutions, and with none left search numerically.
  let numerized = match numerized {
    Expr::List(ref sols) if sols.iter().any(has_non_finite_value) => {
      let finite: Vec<Expr> = sols
        .iter()
        .filter(|s| !has_non_finite_value(s))
        .cloned()
        .collect();
      if finite.is_empty()
        && let Some(found) = try_nsolve_numeric_system(args)
      {
        return Ok(sort_nsolve_solutions(found));
      }
      Expr::List(finite.into())
    }
    other => other,
  };
  // A solution an elimination picked up by squaring a radical away (the
  // sign of a `Sqrt` flipped) does not satisfy the equations as written:
  // drop it, and with none left search numerically.
  let numerized = match numerized {
    Expr::List(ref sols) => {
      let holding: Vec<Expr> = sols
        .iter()
        .filter(|s| !numerically_violates(&args[0], s))
        .cloned()
        .collect();
      if holding.is_empty()
        && !sols.is_empty()
        && let Some(found) = try_nsolve_numeric_system(args)
      {
        return Ok(sort_nsolve_solutions(found));
      }
      Expr::List(holding.into())
    }
    other => other,
  };
  // A `Reals` domain keeps only the real solutions. It is the optional
  // third argument in the ordinary `NSolve[eqns, vars, Reals]` form, but
  // the *last* argument either way — the domain-only two-argument form
  // `NSolve[eqns, Reals]` (no explicit variable list) carries it in
  // position 1 instead, so check `args.last()` rather than a fixed index.
  // solve_ast already filters the symbolically solvable cases, but a
  // polynomial with no radical form falls back to numeric roots that arrive
  // unfiltered — drop the complex ones here so NSolve[quintic, x, Reals]
  // matches wolframscript.
  let filtered = if args.len() >= 2
    && matches!(args.last(), Some(Expr::Identifier(d) | Expr::Constant(d)) if d == "Reals")
  {
    filter_real_nsolve_solutions(numerized)
  } else {
    numerized
  };
  Ok(sort_nsolve_solutions(filtered))
}

/// Whether the numeric solution `{v -> value, …}` clearly fails one of the
/// equations of `eqns` (a list or a single equation): both sides evaluate
/// to numbers further apart than rounding explains. Anything that does not
/// evaluate to numbers (a symbolic value, an inequality) does not count.
fn numerically_violates(eqns: &Expr, solution: &Expr) -> bool {
  let Expr::List(rules) = solution else {
    return false;
  };
  let mut point = Vec::new();
  for rule in rules {
    let Expr::Rule {
      pattern,
      replacement,
    } = rule
    else {
      return false;
    };
    if try_extract_complex_f64(replacement).is_none() {
      return false;
    }
    point.push((pattern.as_ref().clone(), replacement.as_ref().clone()));
  }
  let equations: Vec<Expr> = match eqns {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  equations.iter().any(|eq| {
    let Some((lhs, rhs, CompOp::Equal)) =
      crate::functions::polynomial_ast::reduce::extract_comparison(eq)
    else {
      return false;
    };
    let value = |side: &Expr| {
      let substituted = point
        .iter()
        .fold(side.clone(), |e, (var, v)| substitute_expr(&e, var, v));
      quietly(|| {
        crate::evaluator::evaluate_function_call_ast("N", &[substituted])
      })
      .ok()
      .and_then(|v| try_extract_complex_f64(&v))
    };
    // The residual is judged against the size of the terms that cancel in
    // it: a root of a polynomial with large coefficients leaves a rounding
    // residual far above 1 that is still a root.
    let difference = expand_and_combine(&minus2(lhs.clone(), rhs.clone()));
    let mut residual = (0.0, 0.0);
    let mut scale = 1.0;
    for term in collect_additive_terms(&difference) {
      let Some(t) = value(&term) else {
        return false;
      };
      residual = (residual.0 + t.0, residual.1 + t.1);
      scale += t.0.hypot(t.1);
    }
    residual.0.hypot(residual.1) > 1e-6 * scale
  })
}

/// Complex coefficients of `poly` in `var`, ascending, once every other
/// symbol has been replaced by a number.
fn numeric_complex_coefficients(
  poly: &Expr,
  var: &str,
) -> Option<Vec<(f64, f64)>> {
  let expanded = expand_and_combine(poly);
  let degree = max_power_int(&expanded, var)?.max(0) as usize;
  let terms = collect_additive_terms(&expanded);
  let mut coeffs = vec![(0.0f64, 0.0f64); degree + 1];
  for (d, slot) in coeffs.iter_mut().enumerate() {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d as i128) {
        let c = simplify(c);
        let (re, im) = try_eval_to_f64(&c)
          .map(|v| (v, 0.0))
          .or_else(|| try_extract_complex_float(&c))?;
        slot.0 += re;
        slot.1 += im;
      }
    }
  }
  Some(coeffs)
}

fn complex_number_expr(re: f64, im: f64) -> Expr {
  if im == 0.0 {
    Expr::Real(re)
  } else {
    crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
  }
}

/// Newton iterations on `{p, q} == 0` from an approximate common root, in
/// complex arithmetic. Repeated roots of the elimination polynomial only
/// give a few digits; the system itself is usually regular there. `None`
/// when the Jacobian is singular or the iteration leaves the neighbourhood.
fn polish_pair_root(
  polys: &[Expr],
  vars: &[String],
  x0: (f64, f64),
  y0: (f64, f64),
) -> Option<((f64, f64), (f64, f64))> {
  type C = (f64, f64);
  let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
  let sub = |a: C, b: C| (a.0 - b.0, a.1 - b.1);
  let div = |a: C, b: C| {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
  };
  let mut derivatives = Vec::new();
  for poly in polys {
    derivatives.push([
      differentiate_expr(poly, &vars[0]).ok()?,
      differentiate_expr(poly, &vars[1]).ok()?,
    ]);
  }
  let eval_at = |e: &Expr, p: (C, C)| -> Option<C> {
    let value = substitute_values(
      e,
      vars,
      &[
        complex_number_expr(p.0.0, p.0.1),
        complex_number_expr(p.1.0, p.1.1),
      ],
    );
    try_extract_complex_f64(&value)
  };
  let mut point = (x0, y0);
  for _ in 0..20 {
    let f = [eval_at(&polys[0], point)?, eval_at(&polys[1], point)?];
    let j = [
      [
        eval_at(&derivatives[0][0], point)?,
        eval_at(&derivatives[0][1], point)?,
      ],
      [
        eval_at(&derivatives[1][0], point)?,
        eval_at(&derivatives[1][1], point)?,
      ],
    ];
    let det = sub(mul(j[0][0], j[1][1]), mul(j[0][1], j[1][0]));
    let norm = |c: C| c.0.hypot(c.1);
    let bound =
      (norm(j[0][0]) + norm(j[0][1])) * (norm(j[1][0]) + norm(j[1][1]));
    if norm(det) <= 1e-10 * bound {
      return None;
    }
    let dx = div(sub(mul(f[0], j[1][1]), mul(f[1], j[0][1])), det);
    let dy = div(sub(mul(j[0][0], f[1]), mul(j[1][0], f[0])), det);
    point = (sub(point.0, dx), sub(point.1, dy));
    if norm(dx) + norm(dy) <= 1e-15 * (1.0 + norm(point.0) + norm(point.1)) {
      break;
    }
  }
  if (point.0.0 - x0.0).hypot(point.0.1 - x0.1) > 1e-3
    || (point.1.0 - y0.0).hypot(point.1.1 - y0.1) > 1e-3
  {
    return None;
  }
  // Components that are rounding noise next to the point's size go.
  let size = 1.0 + point.0.0.hypot(point.0.1) + point.1.0.hypot(point.1.1);
  let snap = |c: C| {
    (
      if c.0.abs() <= 1e-14 * size { 0.0 } else { c.0 },
      if c.1.abs() <= 1e-12 * size { 0.0 } else { c.1 },
    )
  };
  Some((snap(point.0), snap(point.1)))
}

/// `NSolve[{p == 0, q == 0}, {x, y}]` for two polynomials with exact
/// coefficients whose elimination polynomial has no closed-form roots.
///
/// `y` is eliminated with a resultant, the roots of that polynomial in `x`
/// come from Durand-Kerner, and each (clustered) `x` is paired with the `y`
/// roots of the lower-degree equation that also satisfy the other one.
fn try_nsolve_polynomial_pair(args: &[Expr]) -> Option<Expr> {
  if args.len() != 2 {
    return None;
  }
  let (Expr::List(eqs), Expr::List(var_list)) = (&args[0], &args[1]) else {
    return None;
  };
  let (x, y) = match var_list.as_slice() {
    [Expr::Identifier(x), Expr::Identifier(y)] => (x.as_str(), y.as_str()),
    _ => return None,
  };
  if eqs.len() != 2 {
    return None;
  }
  let vars = [x.to_string(), y.to_string()];
  let mut polys = Vec::with_capacity(2);
  for eq in eqs {
    let (lhs, rhs, op) =
      crate::functions::polynomial_ast::reduce::extract_comparison(eq)?;
    if op != crate::functions::polynomial_ast::reduce::CompOp::Equal {
      return None;
    }
    let poly = expand_and_combine(&minus2(lhs, rhs));
    let mut free = Vec::new();
    collect_solve_vars(&poly, &mut free);
    if free.iter().any(|f| !vars.contains(f))
      || !vars
        .iter()
        .all(|v| crate::functions::polynomial_ast::is_polynomial(&poly, v))
    {
      return None;
    }
    polys.push(poly);
  }
  let resultant = crate::functions::polynomial_ast::resultant_ast(&[
    polys[0].clone(),
    polys[1].clone(),
    id_expr(y),
  ])
  .ok()?;
  let resultant = expand_and_combine(&resultant);
  if is_zero_expr(&resultant) {
    return None;
  }
  let x_coeffs = numeric_complex_coefficients(&resultant, x)?;
  if x_coeffs.len() < 2 || x_coeffs.iter().any(|c| c.1 != 0.0) {
    return None;
  }
  let real_coeffs: Vec<f64> = x_coeffs.iter().map(|c| c.0).collect();
  let x_roots = durand_kerner_roots(&real_coeffs);

  // Roots of a repeated factor come out as a loose cluster; their mean is
  // the better estimate.
  let mut clusters: Vec<((f64, f64), usize)> = Vec::new();
  for (re, im) in x_roots {
    match clusters.iter_mut().find(|((cre, cim), _)| {
      (cre - re).hypot(cim - im) <= 1e-5 * (1.0 + re.hypot(im))
    }) {
      Some(((cre, cim), n)) => {
        *cre = (*cre * *n as f64 + re) / (*n as f64 + 1.0);
        *cim = (*cim * *n as f64 + im) / (*n as f64 + 1.0);
        *n += 1;
      }
      None => clusters.push(((re, im), 1)),
    }
  }

  let residual = |poly: &Expr, xv: (f64, f64), yv: (f64, f64)| -> Option<f64> {
    let value = substitute_values(
      poly,
      &vars,
      &[
        complex_number_expr(xv.0, xv.1),
        complex_number_expr(yv.0, yv.1),
      ],
    );
    let (re, im) = try_extract_complex_f64(&value)?;
    Some(re.hypot(im))
  };

  let mut points: Vec<((f64, f64), (f64, f64))> = Vec::new();
  for ((xre, xim), _) in clusters {
    let xv = complex_number_expr(xre, xim);
    let mut y_polys = Vec::new();
    for poly in &polys {
      let at_x =
        substitute_values(poly, &[x.to_string()], std::slice::from_ref(&xv));
      let mut coeffs = numeric_complex_coefficients(&at_x, y)?;
      let scale = coeffs
        .iter()
        .map(|c| c.0.hypot(c.1))
        .fold(0.0_f64, f64::max);
      while coeffs.len() > 1
        && coeffs
          .last()
          .is_some_and(|c| c.0.hypot(c.1) <= 1e-9 * scale)
      {
        coeffs.pop();
      }
      if coeffs.len() > 1 {
        y_polys.push(coeffs);
      }
    }
    y_polys.sort_by_key(Vec::len);
    let Some(pivot) = y_polys.first() else {
      continue;
    };
    for yv in durand_kerner_complex_roots(pivot) {
      let magnitude = 1.0 + xre.hypot(xim) + yv.0.hypot(yv.1);
      let fits = polys.iter().all(|p| {
        let degree =
          max_power_int(p, x).unwrap_or(0) + max_power_int(p, y).unwrap_or(0);
        residual(p, (xre, xim), yv)
          .is_some_and(|r| r <= 1e-6 * magnitude.powi(degree as i32))
      });
      if fits {
        let polished = polish_pair_root(&polys, &vars, (xre, xim), yv);
        let (px, py) = polished.unwrap_or(((xre, xim), yv));
        if !points.iter().any(|(ox, oy)| {
          (ox.0 - px.0).hypot(ox.1 - px.1) < 1e-8
            && (oy.0 - py.0).hypot(oy.1 - py.1) < 1e-8
        }) {
          points.push((px, py));
        }
      }
    }
  }
  if points.is_empty() {
    return None;
  }
  points.sort_by(|a, b| {
    b.0
      .0
      .partial_cmp(&a.0.0)
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(
        b.0
          .1
          .partial_cmp(&a.0.1)
          .unwrap_or(std::cmp::Ordering::Equal),
      )
  });
  let solutions = points
    .into_iter()
    .map(|((xre, xim), (yre, yim))| {
      Expr::List(
        vec![
          rule_expr(id_expr(x), complex_number_expr(xre, xim)),
          rule_expr(id_expr(y), complex_number_expr(yre, yim)),
        ]
        .into(),
      )
    })
    .collect::<Vec<_>>();
  Some(Expr::List(solutions.into()))
}

/// Keep only the solutions whose every `var -> value` replacement is real, for
/// an NSolve with a `Reals` domain. A value counts as real when it evaluates to
/// a machine real, or when it is a machine complex with a negligible imaginary
/// part (numerical noise on a genuinely real root).
fn filter_real_nsolve_solutions(expr: Expr) -> Expr {
  let Expr::List(ref items) = expr else {
    return expr;
  };
  let is_real = |item: &Expr| -> bool {
    let Expr::List(rules) = item else {
      return true;
    };
    rules.iter().all(|r| {
      let Expr::Rule { replacement, .. } = r else {
        return true;
      };
      if try_eval_to_f64(replacement).is_some() {
        return true;
      }
      match try_extract_complex_float(replacement) {
        Some((_re, im)) => im.abs() < 1e-8,
        // A non-numeric (still symbolic) replacement is left in place.
        None => true,
      }
    })
  };
  Expr::List(items.iter().filter(|it| is_real(it)).cloned().collect())
}

/// wolframscript lists NSolve roots ordered by ascending real part, breaking
/// ties by ascending imaginary part (the symbolic Solve order they inherit is
/// not numerically sorted). Only reorder when every solution is a single
/// numeric `var -> value` rule; non-numericised solutions are left untouched.
///
/// Multi-variable systems come out of wolframscript's numerical
/// polynomial-system path (Gröbner elimination plus eigenvalue root-finding),
/// which lists the eliminated variable's roots *descending* — e.g.
/// `NSolve[y == c && (x - a)^2 + (y - b)^2 == r^2, {x, y}]` puts the larger
/// intersection point first. (Ground truth: the kernel-saved definitions in
/// Demonstration notebooks, where `sol[[2]]` selects the smaller root.) Sort
/// those descending by the first variable's value.
fn sort_nsolve_solutions(expr: Expr) -> Expr {
  let Expr::List(ref items) = expr else {
    return expr;
  };
  let value_of = |replacement: &Expr| -> Option<(f64, f64)> {
    if let Some(v) = try_eval_to_f64(replacement) {
      return Some((v, 0.0));
    }
    try_extract_complex_float(replacement)
  };
  let key = |item: &Expr| -> Option<(f64, f64)> {
    if let Expr::List(rules) = item
      && rules.len() == 1
      && let Expr::Rule { replacement, .. } = &rules[0]
    {
      return value_of(replacement);
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
    return Expr::List(items_vec.into());
  }
  // Multi-variable system: every solution is a list of two or more numeric
  // rules → descending by the first rule's value.
  let multi_key = |item: &Expr| -> Option<(f64, f64)> {
    if let Expr::List(rules) = item
      && rules.len() >= 2
      && let Expr::Rule { replacement, .. } = &rules[0]
    {
      return value_of(replacement);
    }
    None
  };
  if !items_vec.is_empty() && items_vec.iter().all(|it| multi_key(it).is_some())
  {
    items_vec.sort_by(|a, b| {
      let (ar, ai) = multi_key(a).unwrap();
      let (br, bi) = multi_key(b).unwrap();
      br.partial_cmp(&ar)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(bi.partial_cmp(&ai).unwrap_or(std::cmp::Ordering::Equal))
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
      minus2(operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      minus2(fargs[0].clone(), fargs[1].clone())
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
        let val = try_eval_to_f64(&simplify(c))?;
        coeffs_f64[d] += val;
      }
    }
  }

  let a = coeffs_f64[2];
  let b = coeffs_f64[1];
  let c = coeffs_f64[0];
  let disc = b * b - 4.0 * a * c;

  let make_rule = |val: Expr| -> Expr {
    Expr::List(vec![rule_expr(Expr::Identifier(var.clone()), val)].into())
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

/// `NSolve[a x^n + c == 0, x]` for n >= 3 — the roots on a circle.
///
/// Numericizing the symbolic radicals instead would leave a conjugate pair
/// differing in its last bits (`(-1)^(2/3) 2^(1/3)` and `-(-2)^(1/3)` round
/// independently), so the roots come from the same Durand-Kerner iteration
/// `NRoots` uses, which converges on wolframscript's values.
fn try_nsolve_pure_power(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() != 2 {
    return None;
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => return None,
  };
  let poly = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      minus2(operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      minus2(fargs[0].clone(), fargs[1].clone())
    }
    _ => return None,
  };
  let expanded = expand_and_combine(&poly);
  let terms = collect_additive_terms(&expanded);
  let degree = max_power_int(&expanded, &var)? as usize;
  if degree < 3 {
    return None;
  }
  let mut coeffs_f64 = vec![0.0f64; degree + 1];
  for (d, slot) in coeffs_f64.iter_mut().enumerate() {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, &var, d as i128) {
        *slot += try_eval_to_f64(&simplify(c))?;
      }
    }
  }
  if coeffs_f64[1..degree].iter().any(|c| *c != 0.0) {
    return None;
  }
  if coeffs_f64[degree] == 0.0 || coeffs_f64[0] == 0.0 {
    return None;
  }
  let mut roots = durand_kerner_roots(&coeffs_f64);
  roots.sort_by(|a, b| {
    a.0
      .partial_cmp(&b.0)
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
  });
  let rules = roots
    .into_iter()
    .map(|(re, im)| {
      let value = if im == 0.0 {
        Expr::Real(re)
      } else {
        crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
      };
      Expr::List(vec![rule_expr(Expr::Identifier(var.clone()), value)].into())
    })
    .collect::<Vec<_>>();
  Some(Ok(Expr::List(rules.into())))
}

/// `NSolve[p == 0, x]` for a polynomial of degree >= 3 whose numeric
/// coefficients are not all real (e.g. `x^n + (1 + 2 I) x + 1 == 0`).
/// The symbolic solver cannot reach these in closed form, so the roots come
/// from Durand-Kerner on complex coefficients, sorted like every NSolve.
fn try_nsolve_complex_coefficients(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() != 2 {
    return None;
  }
  let Expr::Identifier(var) = &args[1] else {
    return None;
  };
  let poly = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      minus2(operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      minus2(fargs[0].clone(), fargs[1].clone())
    }
    _ => return None,
  };
  let expanded = expand_and_combine(&poly);
  let terms = collect_additive_terms(&expanded);
  let degree = max_power_int(&expanded, var)? as usize;
  if degree < 3 {
    return None;
  }
  let mut coeffs = vec![(0.0f64, 0.0f64); degree + 1];
  // A coefficient such as `-4. + 0.*I` is a complex number with an inexact
  // zero imaginary part: the real-coefficient solvers cannot read it, so it
  // is solved here even though every imaginary part is zero.
  let mut has_complex_coefficient = false;
  for (d, slot) in coeffs.iter_mut().enumerate() {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d as i128) {
        let c = simplify(c);
        let (re, im) = if let Some(v) = try_eval_to_f64(&c) {
          (v, 0.0)
        } else {
          has_complex_coefficient = true;
          try_extract_complex_float(&c)?
        };
        slot.0 += re;
        slot.1 += im;
      }
    }
  }
  if (!has_complex_coefficient && coeffs.iter().all(|c| c.1 == 0.0))
    || coeffs[degree] == (0.0, 0.0)
  {
    return None;
  }
  let mut roots = durand_kerner_complex_roots(&coeffs);
  roots.sort_by(|a, b| {
    a.0
      .partial_cmp(&b.0)
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
  });
  let rules = roots
    .into_iter()
    .map(|(re, im)| {
      let value = if im == 0.0 {
        Expr::Real(re)
      } else {
        crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
      };
      Expr::List(vec![rule_expr(Expr::Identifier(var.clone()), value)].into())
    })
    .collect::<Vec<_>>();
  Some(Ok(Expr::List(rules.into())))
}

/// All roots of a polynomial with complex coefficients (`coeffs[i]` is the
/// `(re, im)` coefficient of x^i, degree >= 1), by Durand-Kerner iteration
/// followed by a few Newton steps.
fn durand_kerner_complex_roots(coeffs: &[(f64, f64)]) -> Vec<(f64, f64)> {
  let n = coeffs.len() - 1;
  let mul = |a: (f64, f64), b: (f64, f64)| {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
  };
  let div = |a: (f64, f64), b: (f64, f64)| {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
  };
  let lc = coeffs[n];
  let monic: Vec<(f64, f64)> = coeffs.iter().map(|&c| div(c, lc)).collect();
  let eval = |z: (f64, f64)| {
    let mut acc = monic[n];
    for k in (0..n).rev() {
      let m = mul(acc, z);
      acc = (m.0 + monic[k].0, m.1 + monic[k].1);
    }
    acc
  };
  // Start on a circle bounding the roots (Cauchy bound), off the axes.
  let radius = 1.0
    + monic[..n]
      .iter()
      .map(|c| c.0.hypot(c.1))
      .fold(0.0_f64, f64::max);
  let mut roots: Vec<(f64, f64)> = (0..n)
    .map(|k| {
      let ang = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64) + 0.4;
      (radius * ang.cos(), radius * ang.sin())
    })
    .collect();
  for _ in 0..2000 {
    let prev = roots.clone();
    let mut max_delta = 0.0_f64;
    for k in 0..n {
      let z = prev[k];
      let mut q = eval(z);
      for (j, &w) in prev.iter().enumerate() {
        if j == k {
          continue;
        }
        let diff = (z.0 - w.0, z.1 - w.1);
        if diff != (0.0, 0.0) {
          q = div(q, diff);
        }
      }
      roots[k] = (z.0 - q.0, z.1 - q.1);
      max_delta = max_delta.max(q.0.hypot(q.1));
    }
    let scale = 1.0 + roots.iter().map(|r| r.0.hypot(r.1)).fold(0.0, f64::max);
    if max_delta < 1e-15 * scale {
      break;
    }
  }
  // Newton polish.
  for r in &mut roots {
    for _ in 0..3 {
      let mut dp = (0.0, 0.0);
      let mut p = monic[n];
      for k in (0..n).rev() {
        let m = mul(dp, *r);
        dp = (m.0 + p.0, m.1 + p.1);
        let m = mul(p, *r);
        p = (m.0 + monic[k].0, m.1 + monic[k].1);
      }
      if dp == (0.0, 0.0) {
        break;
      }
      let step = div(p, dp);
      *r = (r.0 - step.0, r.1 - step.1);
    }
  }
  roots
}

/// Principal-branch complex power: (a+bi)^(c+di) = exp((c+di) * Log[a+bi]).
pub(crate) fn complex_pow(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
  // A real power of a real base stays on the real line: going through
  // exp(c log a) costs an ulp or two (`Sqrt[2]` came out as
  // 1.414213562373095 instead of 1.4142135623730951).
  if b == 0.0 && d == 0.0 {
    if a > 0.0 {
      return (if c == 0.5 { a.sqrt() } else { a.powf(c) }, 0.0);
    }
    if a < 0.0 && c.fract() == 0.0 && c.abs() < i32::MAX as f64 {
      return (a.powi(c as i32), 0.0);
    }
  }
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
  if let Some(v) = try_extract_complex_float(expr) {
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
      for arg in args {
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
    } => Ok(rule_expr(*pattern.clone(), nsolve_numerize(replacement)?)),
    _ => {
      // Try pure real first
      if let Some(v) = try_eval_to_f64(expr) {
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
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Roots expects at least 2 arguments".into(),
    ));
  }
  if args.len() > 2 {
    return roots_with_options(args);
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("Roots", args));
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
            return Ok(bool_expr(true));
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
        Ok(bool_expr(false))
      } else if conditions.len() == 1 {
        Ok(conditions.into_iter().next().unwrap())
      } else {
        Ok(call("Or", conditions))
      }
    }
    // Solve returned unevaluated
    _ => Ok(unevaluated("Roots", args)),
  }
}

/// `Roots[eqn, var, opts]` with the options `Modulus`, `Multiplicity`,
/// `Cubics` and `Quartics`. Any other option leaves the call unevaluated.
fn roots_with_options(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::functions::expr_form::rule_parts;
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
  let mut modulus: i128 = 0;
  let mut multiplicity: Option<usize> = None;
  let mut radical_degrees: Vec<i128> = Vec::new();
  for opt in &args[2..] {
    let Some((Expr::Identifier(name), value, _)) = rule_parts(opt) else {
      return Ok(unevaluated("Roots", args));
    };
    match (name.as_str(), value) {
      ("Modulus", Expr::Integer(n)) if *n >= 0 => modulus = *n,
      ("Multiplicity", Expr::Integer(k)) if *k >= 1 => {
        multiplicity = Some(*k as usize);
      }
      ("Multiplicity", Expr::Identifier(i)) if i == "Infinity" => {}
      ("Cubics" | "Quartics", Expr::Identifier(b))
        if b == "True" || b == "False" =>
      {
        if b == "True" {
          radical_degrees.push(if name == "Cubics" { 3 } else { 4 });
        }
      }
      _ => return Ok(unevaluated("Roots", args)),
    }
  }

  if modulus > 0 {
    return eval(call(
      "Reduce",
      vec![
        args[0].clone(),
        args[1].clone(),
        rule_expr(Expr::Identifier("Modulus".into()), Expr::Integer(modulus)),
      ],
    ));
  }

  let result = roots_ast(&args[..2])?;
  let mut conditions: Vec<Expr> = match &result {
    Expr::FunctionCall { name, args: cs } if name == "Or" => cs.to_vec(),
    Expr::Comparison { .. } => vec![result.clone()],
    _ => return Ok(result),
  };

  // Root objects of the requested degrees are written out in radicals.
  if !radical_degrees.is_empty() {
    for c in &mut conditions {
      if let Expr::Comparison { operands, .. } = c
        && let Some(value) = operands.get_mut(1)
        && let Expr::FunctionCall { name, args: rargs } = &*value
        && name == "Root"
        && let Some(Expr::Function { body }) = rargs.first()
        && let Ok(Expr::Integer(deg)) =
          eval(call("Exponent", vec![*body.clone(), Expr::Slot(1)]))
        && radical_degrees.contains(&deg)
      {
        *value = eval(call1("ToRadicals", value.clone()))?;
      }
    }
  }

  // At most `k` copies of each repeated root.
  if let Some(k) = multiplicity {
    let mut kept: Vec<(Expr, usize)> = Vec::new();
    conditions.retain(|c| {
      match kept
        .iter_mut()
        .find(|(e, _)| crate::evaluator::pattern_matching::expr_equal(e, c))
      {
        Some((_, n)) if *n >= k => false,
        Some((_, n)) => {
          *n += 1;
          true
        }
        None => {
          kept.push((c.clone(), 1));
          true
        }
      }
    });
  }

  Ok(if conditions.len() == 1 {
    conditions.into_iter().next().unwrap()
  } else {
    call("Or", conditions)
  })
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
      return Ok(unevaluated("NRoots", args));
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
      minus2(operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      minus2(fargs[0].clone(), fargs[1].clone())
    }
    // A bare polynomial stands for `poly == 0`, as in wolframscript.
    Expr::Comparison { .. } => {
      return Ok(unevaluated("NRoots", args));
    }
    poly => poly.clone(),
  };

  let unevaluated = || unevaluated("NRoots", args);

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
        let val = try_eval_to_f64(&simplify(c));
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

  // A repeated root stalls Durand-Kerner a few digits short. With exact
  // coefficients the square-free factors are solved one by one instead,
  // each root repeated by the multiplicity of its factor.
  let square_free =
    (!crate::functions::math_ast::contains_inexact_real(&expanded))
      .then(|| {
        crate::evaluator::evaluate_function_call_ast(
          "FactorSquareFreeList",
          std::slice::from_ref(&expanded),
        )
        .ok()
      })
      .flatten();
  let mut factored: Option<Vec<(f64, f64)>> = None;
  if let Some(Expr::List(ref factors)) = square_free
    && factors.iter().any(|f| {
      matches!(f, Expr::List(pair) if pair.len() == 2
        && matches!(&pair[1], Expr::Integer(m) if *m > 1)
        && !is_constant_wrt(&pair[0], &var))
    })
  {
    let mut all = Vec::new();
    for f in factors {
      let Expr::List(pair) = f else {
        all.clear();
        break;
      };
      let (Some(factor), Some(Expr::Integer(m))) = (pair.first(), pair.get(1))
      else {
        all.clear();
        break;
      };
      if is_constant_wrt(factor, &var) {
        continue;
      }
      let Some(c) = polynomial_f64_coefficients(factor, &var) else {
        all.clear();
        break;
      };
      let roots = durand_kerner_roots(&c);
      for _ in 0..*m {
        all.extend(roots.iter().copied());
      }
    }
    if all.len() == degree {
      factored = Some(all);
    }
  }
  // Durand-Kerner finds all roots simultaneously.
  let roots = factored.unwrap_or_else(|| durand_kerner_roots(&coeffs));

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
  Ok(call("Or", conds))
}

/// The coefficients of a polynomial in `var` as f64 (index = degree), or
/// None when one is not a number.
fn polynomial_f64_coefficients(poly: &Expr, var: &str) -> Option<Vec<f64>> {
  let expanded = expand_and_combine(poly);
  let degree = max_power_int(&expanded, var).filter(|d| *d >= 1)? as usize;
  let terms = collect_additive_terms(&expanded);
  let mut coeffs = vec![0.0f64; degree + 1];
  for (d, slot) in coeffs.iter_mut().enumerate() {
    for term in &terms {
      if let Some(c) = extract_coefficient_of_power(term, var, d as i128) {
        *slot += try_eval_to_f64(&simplify(c))?;
      }
    }
  }
  Some(coeffs)
}

/// Find all roots of polynomial (coeffs[i] is coefficient of x^i) using
/// the Durand-Kerner (Weierstrass) method on complex doubles.
pub(crate) fn durand_kerner_roots(coeffs: &[f64]) -> Vec<(f64, f64)> {
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
      // q = p(z) / Π_{j≠k}(z - z_j). Divide factor by factor instead of
      // forming the product first: for high degrees the product overflows
      // f64 and the whole iteration collapses to NaN.
      let (mut qr, mut qi) = eval_p(zr, zi);
      for (j, &(jr, ji)) in prev.iter().enumerate() {
        if j == k {
          continue;
        }
        let ar = zr - jr;
        let ai = zi - ji;
        let denom = ar * ar + ai * ai;
        if denom == 0.0 {
          continue;
        }
        let nr = (qr * ar + qi * ai) / denom;
        let ni = (qi * ar - qr * ai) / denom;
        qr = nr;
        qi = ni;
      }
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

  crate::functions::math_ast::polish_and_pair_roots(coeffs, &mut roots);
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
      return Some(rule_expr(operands[0].clone(), operands[1].clone()));
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
      Ok(call("Sequence", result))
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
      Ok(call("Sequence", result))
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
    Expr::Identifier(s) if s == "False" => Ok(call0("Sequence")),
    // Anything else: return unevaluated
    _ => Ok(call1("ToRules", input.clone())),
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
  let _head = IfunHead::new("SolveValues");
  let solutions = solve_ast(args)?;
  Ok(
    solution_values(&solutions, &args[1])
      .unwrap_or_else(|| unevaluated("SolveValues", args)),
  )
}

/// NSolveValues[eqns, vars] / NSolveValues[eqns, vars, domain] — the numeric
/// analogue of SolveValues: the variable VALUES from NSolve rather than the
/// `{var -> value}` rules. A single variable yields a flat list of values; a
/// list of variables yields a list of value-lists (one per solution), in the
/// order the variables are given.
pub fn nsolve_values_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let _head = IfunHead::new("NSolveValues");
  let solutions = nsolve_ast(args)?;
  Ok(
    solution_values(&solutions, &args[1])
      .unwrap_or_else(|| unevaluated("NSolveValues", args)),
  )
}

/// Strip the `{var -> value}` rules off a Solve/NSolve result, keeping just
/// the values. The result's shape follows the variable specification: a bare
/// symbol gives a flat list of values, while a list of variables gives one
/// value-list per solution, in the order the variables were named. Returns
/// `None` when the solutions are not in that rule form, so the caller can
/// stay unevaluated.
fn solution_values(solutions: &Expr, var_spec: &Expr) -> Option<Expr> {
  let Expr::List(solution_sets) = solutions else {
    return None;
  };

  // A list of variables ({x, y}) means each solution contributes a value-list;
  // a single variable contributes one value.
  let vars: Option<Vec<String>> = match var_spec {
    Expr::List(vs) => vs
      .iter()
      .map(|v| match v {
        Expr::Identifier(name) => Some(name.clone()),
        _ => None,
      })
      .collect(),
    _ => None,
  };
  // A variable list that isn't all symbols is not something we can order by.
  if matches!(var_spec, Expr::List(_)) && vars.is_none() {
    return None;
  }

  let mut out = Vec::with_capacity(solution_sets.len());
  for branch in solution_sets {
    let Expr::List(rules) = branch else {
      return None;
    };
    if let Some(names) = &vars {
      let mut branch_vals = Vec::with_capacity(names.len());
      for name in names {
        let value = rules.iter().find_map(|r| match r {
          Expr::Rule {
            pattern,
            replacement,
          } if matches!(pattern.as_ref(), Expr::Identifier(p) if p == name) => {
            Some((**replacement).clone())
          }
          _ => None,
        });
        branch_vals
          .push(value.unwrap_or_else(|| Expr::Identifier(name.clone())));
      }
      out.push(Expr::List(branch_vals.into()));
    } else {
      if rules.len() != 1 {
        return None;
      }
      let Expr::Rule { replacement, .. } = &rules[0] else {
        return None;
      };
      out.push((**replacement).clone());
    }
  }
  Some(Expr::List(out.into()))
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
      // Domain names: a protected Wolfram symbol appearing inside an
      // `Element[x, Reals]` domain hint (a valid part of `eqns`) must
      // never be picked up as a solve variable itself.
      | "Reals"
      | "Integers"
      | "Complexes"
      | "Rationals"
      | "Algebraics"
      | "Booleans"
  )
}

/// Auto-detect the variable argument for a Solve/NSolve call that gave no
/// explicit variable list — the one-argument form `Solve[eqns]` and the
/// two-argument domain-only form `Solve[eqns, dom]` both resolve to this.
/// The variables come back in canonical order, as Wolfram lists them; an
/// underdetermined system gets all of them too and is solved for the ones
/// `select_underdetermined_vars` / the linear solver pick.
fn auto_detect_solve_vars(eqns: &Expr) -> Option<Expr> {
  let mut vars = Vec::new();
  collect_solve_var_exprs(eqns, &mut vars);
  if solve_equalities(eqns).is_empty() {
    return None;
  }
  vars.sort_by(crate::functions::list_helpers_ast::sorting::canonical_cmp);
  match vars.len() {
    0 => None,
    1 => Some(vars.remove(0)),
    _ => Some(Expr::List(vars.into())),
  }
}

/// The unknowns of `Solve[eqns]`: the free symbols, except that a
/// `Subscript[β, 0]` or a call of a head with no built-in meaning (`a[1]`,
/// `f[x]`) is one unknown of its own, as in Wolfram
/// (`Solve[1 - Subscript[β, 0] == 0]` solves for `Subscript[β, 0]`).
fn collect_solve_var_exprs(expr: &Expr, out: &mut Vec<Expr>) {
  let push = |e: Expr, out: &mut Vec<Expr>| {
    if !out
      .iter()
      .any(|v| crate::evaluator::pattern_matching::expr_equal(v, &e))
    {
      out.push(e);
    }
  };
  match expr {
    Expr::Identifier(s) if !is_solve_constant(s) => {
      push(expr.clone(), out);
    }
    Expr::FunctionCall { name, args }
      if !args.is_empty()
        && (name == "Subscript" && matches!(&args[0], Expr::Identifier(_))
          || !crate::evaluator::functions::is_builtin_symbol(name)) =>
    {
      push(expr.clone(), out);
    }
    // A derivative value (`f''[0]`, held as `Derivative[2, f, 0]` once
    // evaluated) is one unknown, as an undefined function's value is.
    Expr::FunctionCall { name, args }
      if name == "Derivative" && args.len() >= 3 =>
    {
      push(expr.clone(), out);
    }
    Expr::CurriedCall { func, .. }
      if matches!(func.as_ref(), Expr::CurriedCall { func: inner, .. }
        if matches!(inner.as_ref(), Expr::FunctionCall { name, .. } if name == "Derivative")) =>
    {
      push(expr.clone(), out);
    }
    Expr::Comparison { operands, .. } => {
      for e in operands {
        collect_solve_var_exprs(e, out);
      }
    }
    // `Inequality[a, Less, x, LessEqual, b]`: the relation symbols between
    // the operands are no unknowns.
    Expr::FunctionCall { name, args } if name == "Inequality" => {
      for e in args.iter().step_by(2) {
        collect_solve_var_exprs(e, out);
      }
    }
    Expr::List(items) | Expr::FunctionCall { args: items, .. } => {
      for e in items {
        collect_solve_var_exprs(e, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_solve_var_exprs(left, out);
      collect_solve_var_exprs(right, out);
    }
    Expr::UnaryOp { operand, .. } => collect_solve_var_exprs(operand, out),
    _ => {}
  }
}

/// The plain equations among `eqns` (a list, a conjunction, or a single
/// equation) — the constraints that each pin down one unknown.
fn solve_equalities(eqns: &Expr) -> Vec<Expr> {
  flatten_and_constraints(std::slice::from_ref(eqns))
    .into_iter()
    .flat_map(|e| thread_list_equation(&e).unwrap_or_else(|| vec![e]))
    .filter(|e| {
      matches!(
        e,
        Expr::Comparison { operators, .. }
          if operators.len() == 1 && operators[0] == ComparisonOp::Equal
      )
    })
    .collect()
}

/// Which variables an underdetermined nonlinear system is solved for: one
/// per equation, the rest staying free parameters. Like Wolfram, the choice
/// prefers a variable every equation is linear in with a coefficient free
/// of the other unknowns (`x + y^2 == 2` is solved for `x`, `a x + b == 0`
/// over `{a, b, x}` for `b`), then the variable of lowest degree
/// (`x^2 - y^3 == 1` for `x`), and among equals the later variable
/// (`x y == 1` for `y`). The chosen variables keep their order in `vars`.
fn select_underdetermined_vars(eqs: &[Expr], vars: &[String]) -> Vec<String> {
  let wanted = eqs.len().min(vars.len());
  let polys: Vec<Expr> = eqs
    .iter()
    .filter_map(|eq| match eq {
      Expr::Comparison { operands, .. } if operands.len() == 2 => Some(
        expand_and_combine(&minus2(operands[0].clone(), operands[1].clone())),
      ),
      _ => None,
    })
    .collect();
  // (not a clean linear unknown, degree), lower is preferred.
  let rank = |v: &String| -> (bool, i128) {
    let mut degree = 0;
    let mut clean = true;
    for poly in &polys {
      match crate::functions::polynomial_ast::max_power_int(poly, v) {
        Some(d) if d >= 0 => {
          degree = degree.max(d);
          if d == 1 {
            let coeff = crate::evaluator::evaluate_expr_to_expr(&call(
              "Coefficient",
              vec![poly.clone(), Expr::Identifier(v.clone()), Expr::Integer(1)],
            ));
            clean &= matches!(coeff, Ok(c) if !is_expr_zero(&c)
              && vars.iter().all(|w| is_constant_wrt(&c, w)));
          } else if d > 1 {
            clean = false;
          }
        }
        _ => {
          degree = i128::MAX;
          clean = false;
        }
      }
    }
    // A variable no equation mentions cannot be solved for.
    if degree == 0 {
      return (true, i128::MAX);
    }
    (!(clean && degree == 1), degree)
  };
  let mut ranked: Vec<(usize, (bool, i128))> =
    vars.iter().enumerate().map(|(i, v)| (i, rank(v))).collect();
  // Stable sort on the rank with the later variable first among equals.
  ranked.reverse();
  ranked.sort_by_key(|&(_, r)| r);
  let chosen: Vec<usize> =
    ranked.iter().take(wanted).map(|&(i, _)| i).collect();
  vars
    .iter()
    .enumerate()
    .filter(|(i, _)| chosen.contains(i))
    .map(|(_, v)| v.clone())
    .collect()
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

thread_local! {
  /// Nesting depth of a solve that must not report `ifun`. The solvers reach
  /// the inverted-function step through several layers of recursion, and by
  /// then the arguments no longer say what asked for it.
  static IFUN_SUPPRESS: std::cell::Cell<usize> = const {
    std::cell::Cell::new(0)
  };
  /// The head the `ifun` message is reported against — wolframscript tags it
  /// with the function the user called, not the `Solve` doing the work.
  static IFUN_HEAD: std::cell::RefCell<&'static str> = const {
    std::cell::RefCell::new("Solve")
  };
  /// Set when a report was swallowed by `IFUN_SUPPRESS`, so a solver that
  /// suppressed it can still report it once on its own behalf.
  static IFUN_SWALLOWED: std::cell::Cell<bool> = const {
    std::cell::Cell::new(false)
  };
}

/// Runs `f` with `ifun` held back, then reports it at most once if any part
/// of `f` asked for it — several sub-solves (one per root of a polynomial
/// in `E^x`) give one message, as in wolframscript.
fn with_single_ifun<T>(f: impl FnOnce() -> T) -> T {
  let outer = IFUN_SWALLOWED.with(|w| w.replace(false));
  let result = {
    let _quiet = SuppressIfun::new(true);
    f()
  };
  let asked = IFUN_SWALLOWED.with(|w| w.replace(outer));
  if asked {
    report_inverse_function_use();
  }
  result
}

/// Suppresses the `ifun` report for as long as it is alive.
struct SuppressIfun(bool);

impl SuppressIfun {
  fn new(active: bool) -> Self {
    if active {
      IFUN_SUPPRESS.with(|d| d.set(d.get() + 1));
    }
    Self(active)
  }
}

impl Drop for SuppressIfun {
  fn drop(&mut self) {
    if self.0 {
      IFUN_SUPPRESS.with(|d| d.set(d.get() - 1));
    }
  }
}

/// Reports the message against `head` for as long as it is alive.
struct IfunHead(&'static str);

impl IfunHead {
  /// The outermost wrapper names the message, so an inner solver that also
  /// sets a head (`NSolveValues` delegating to `NSolve`) leaves it alone.
  fn new(head: &'static str) -> Self {
    Self(IFUN_HEAD.with(|h| {
      let mut current = h.borrow_mut();
      if *current == "Solve" {
        std::mem::replace(&mut *current, head)
      } else {
        *current
      }
    }))
  }
}

impl Drop for IfunHead {
  fn drop(&mut self) {
    IFUN_HEAD.with(|h| *h.borrow_mut() = self.0);
  }
}

/// Whether `expr` constrains the solution with anything beyond equations.
/// An inequality alongside an equation narrows the answer to a set the solver
/// can report in full, which is what takes an inverted function off the hook.
fn has_inequality(expr: &Expr) -> bool {
  const INEQUALITY_HEADS: [&str; 6] = [
    "Less",
    "Greater",
    "LessEqual",
    "GreaterEqual",
    "Unequal",
    "Inequality",
  ];
  match expr {
    Expr::Comparison { operators, .. } => {
      operators.iter().any(|o| !matches!(o, ComparisonOp::Equal))
    }
    Expr::List(items) => items.iter().any(has_inequality),
    Expr::FunctionCall { name, args } => {
      INEQUALITY_HEADS.contains(&name.as_str())
        || args.iter().any(has_inequality)
    }
    Expr::BinaryOp { left, right, .. } => {
      has_inequality(left) || has_inequality(right)
    }
    Expr::UnaryOp { operand, .. } => has_inequality(operand),
    _ => false,
  }
}

/// Report that an inverted function may have cost some solutions.
fn report_inverse_function_use() {
  if IFUN_SUPPRESS.with(std::cell::Cell::get) > 0 {
    IFUN_SWALLOWED.with(|w| w.set(true));
    return;
  }
  let head = IFUN_HEAD.with(|h| *h.borrow());
  crate::emit_message(&format!(
    "{head}::ifun: Inverse functions are being used by {head}, so some \
     solutions may not be found; use Reduce for complete solution information."
  ));
}

/// Solve[eqns, vars] and its option forms.
///
/// `Modulus -> n` solves over the integers modulo `n`, which the modular
/// `Reduce` already knows how to do — this delegates to it and turns the
/// `x == a || x == b` it reports back into Solve's list of rules. `MaxRoots`
/// caps how many solutions come back. Both are peeled off here so the solver
/// proper never sees them.
///
/// The rules inside one multivariate modular solution come out in the order the
/// variables were given; wolframscript orders them by its own elimination path,
/// which reverses them for a linear system but not otherwise.
/// `Max[…]` / `Min[…]` / `Piecewise[…]` calls in `e` that involve one of
/// `vars`, each listed once.
fn collect_variable_max_min(e: &Expr, vars: &[String], out: &mut Vec<Expr>) {
  let is_split_node = |name: &str, args: &[Expr]| {
    ((name == "Max" || name == "Min") && args.len() >= 2)
      || (name == "Piecewise"
        && (1..=2).contains(&args.len())
        && matches!(&args[0], Expr::List(pieces) if !pieces.is_empty()
          && pieces.iter().all(|p| matches!(p, Expr::List(vc) if vc.len() == 2))))
  };
  if let Expr::FunctionCall { name, args } = e
    && is_split_node(name, args)
    && vars.iter().any(|v| contains_var(e, v))
  {
    let key = crate::syntax::expr_to_string(e);
    if !out.iter().any(|o| crate::syntax::expr_to_string(o) == key) {
      out.push(e.clone());
    }
    return;
  }
  match e {
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      for a in args {
        collect_variable_max_min(a, vars, out);
      }
    }
    Expr::Comparison { operands, .. } => {
      for a in operands {
        collect_variable_max_min(a, vars, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_variable_max_min(left, vars, out);
      collect_variable_max_min(right, vars, out);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_variable_max_min(operand, vars, out);
    }
    _ => {}
  }
}

/// Solve a system whose equations contain `Max`/`Min` of the unknowns by
/// splitting on which argument attains each extremum: every choice turns the
/// system into an ordinary one, whose solutions are kept when the chosen
/// argument really is the largest (smallest) there.
fn solve_piecewise_max_min(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let vars: Vec<String> = match args.get(1)? {
    Expr::Identifier(v) => vec![v.clone()],
    Expr::List(vs) => vs
      .iter()
      .map(|v| match v {
        Expr::Identifier(s) => Some(s.clone()),
        _ => None,
      })
      .collect::<Option<_>>()?,
    _ => return None,
  };
  let mut extrema = Vec::new();
  collect_variable_max_min(&args[0], &vars, &mut extrema);
  if extrema.is_empty() {
    return None;
  }
  // Each node's alternatives: the value it takes and the conditions under
  // which it takes it. `Max`/`Min` takes argument k when that one is the
  // largest (smallest); `Piecewise` takes piece k when its condition holds
  // and no earlier one does, and its default when none holds.
  let alternatives: Vec<Vec<(Expr, Vec<Expr>)>> = extrema
    .iter()
    .map(|e| {
      let Expr::FunctionCall { name, args: xs } = e else {
        return Vec::new();
      };
      if name == "Piecewise" {
        let Expr::List(pieces) = &xs[0] else {
          return Vec::new();
        };
        let not = |c: &Expr| call1("Not", c.clone());
        let mut out = Vec::new();
        let mut earlier: Vec<Expr> = Vec::new();
        for piece in pieces {
          let Expr::List(vc) = piece else { continue };
          let mut conds: Vec<Expr> = earlier.iter().map(not).collect();
          conds.push(vc[1].clone());
          out.push((vc[0].clone(), conds));
          earlier.push(vc[1].clone());
        }
        let default = xs.get(1).cloned().unwrap_or(Expr::Integer(0));
        out.push((default, earlier.iter().map(not).collect()));
        return out;
      }
      let op = if name == "Max" {
        ComparisonOp::GreaterEqual
      } else {
        ComparisonOp::LessEqual
      };
      (0..xs.len())
        .map(|k| {
          let conds = xs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != k)
            .map(|(_, other)| Expr::Comparison {
              operands: vec![xs[k].clone(), other.clone()],
              operators: vec![op],
            })
            .collect();
          (xs[k].clone(), conds)
        })
        .collect()
    })
    .collect();
  // Like Wolfram, an inexact system with `Max`/`Min` is solved exactly (its
  // machine numbers rationalized) and the solutions are numericized
  // afterwards; one with only `Piecewise` is solved as it is.
  let has_max_min = extrema.iter().any(|e| {
    matches!(e, Expr::FunctionCall { name, .. } if name == "Max" || name == "Min")
  });
  let inexact = has_max_min
    && crate::functions::predicate_ast::contains_real_literal(&args[0]);
  if inexact {
    let exact = crate::evaluator::evaluate_expr_to_expr(&call(
      "Rationalize",
      vec![args[0].clone(), Expr::Integer(0)],
    ))
    .ok()?;
    if crate::functions::predicate_ast::contains_real_literal(&exact) {
      return None;
    }
    let mut exact_args = args.to_vec();
    exact_args[0] = exact;
    let result = solve_piecewise_max_min(&exact_args)?;
    return Some(result.and_then(|sols| {
      if !in_nsolve() {
        crate::emit_message(
          "Solve::ratnz: Solve was unable to solve the system with inexact \
           coefficients. The answer was obtained by solving a corresponding \
           exact system and numericizing the result.",
        );
      }
      crate::evaluator::evaluate_expr_to_expr(&call1("N", sols))
    }));
  }
  let choices: Vec<usize> = alternatives.iter().map(Vec::len).collect();
  if choices.contains(&0) {
    return None;
  }
  if choices.iter().product::<usize>() > 512 {
    return None;
  }
  let mut pick = vec![0usize; extrema.len()];
  let mut solutions: Vec<Expr> = Vec::new();
  loop {
    let mut system = args[0].clone();
    let mut conditions = Vec::new();
    for ((e, alts), &k) in extrema.iter().zip(&alternatives).zip(&pick) {
      let (value, conds) = &alts[k];
      system = substitute_expr(&system, e, value);
      conditions.extend(conds.iter().cloned());
    }
    let mut call_args = args.to_vec();
    call_args[0] = system;
    let solved = {
      let _quiet = QuietGuard::new();
      solve_ast(&call_args)
    };
    let solved = match solved {
      Ok(e) => e,
      Err(e) => return Some(Err(e)),
    };
    match &solved {
      Expr::List(sols) => {
        for sol in sols {
          let holds = conditions.iter().all(|c| {
            let test = call("ReplaceAll", vec![c.clone(), sol.clone()]);
            matches!(
              crate::evaluator::evaluate_expr_to_expr(&test),
              Ok(Expr::Identifier(ref t)) if t == "True"
            )
          });
          let key = crate::syntax::expr_to_string(sol);
          if holds
            && !solutions
              .iter()
              .any(|o| crate::syntax::expr_to_string(o) == key)
          {
            solutions.push(sol.clone());
          }
        }
      }
      _ => return None,
    }
    // Next combination.
    let mut i = 0;
    loop {
      if i == pick.len() {
        return Some(Ok(Expr::List(solutions.into())));
      }
      pick[i] += 1;
      if pick[i] < choices[i] {
        break;
      }
      pick[i] = 0;
      i += 1;
    }
  }
}

/// Generalized variables: `Solve[eqns, f[x]]` (or a variable list containing
/// such applications) treats the whole application `f[x]` as the unknown,
/// matching wolframscript. Every function-application target is replaced
/// with a fresh bare symbol, `solver` runs on the result, and the fresh
/// symbols are mapped back to the original applications. `None` when no
/// target is an application, so bare-symbol calls are unaffected. Shared by
/// `Solve` and `Reduce`.
pub(crate) fn with_generalized_variables(
  args: &[Expr],
  solver: fn(&[Expr]) -> Result<Expr, InterpreterError>,
) -> Option<Result<Expr, InterpreterError>> {
  let targets: Vec<Expr> = match args.get(1)? {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  if !targets
    .iter()
    .any(|t| matches!(t, Expr::FunctionCall { .. }))
  {
    return None;
  }
  let ctx = format!("{}{}", expr_to_string(&args[0]), expr_to_string(&args[1]));
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
  new_args.extend(args[2..].iter().cloned());
  Some(solver(&new_args).map(|result| {
    subs.iter().fold(result, |mapped, (from, to)| {
      substitute_expr(&mapped, to, from)
    })
  }))
}

/// Whether `Inequality[…]` (the FullForm of a mixed chained comparison)
/// occurs among the constraints.
fn contains_inequality_call(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, .. } if name == "Inequality" => true,
    Expr::List(items) => items.iter().any(contains_inequality_call),
    Expr::FunctionCall { name, args } if name == "And" => {
      args.iter().any(contains_inequality_call)
    }
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => contains_inequality_call(left) || contains_inequality_call(right),
    _ => false,
  }
}

pub fn solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Some(first) = args.first()
    && contains_inequality_call(first)
  {
    let mut normalized = args.to_vec();
    normalized[0] = super::reduce::inequalities_to_comparisons(first);
    return solve_ast(&normalized);
  }
  if let Some(result) = solve_interpolating_function("Solve", args) {
    return Ok(result);
  }
  let solved = solve_ast_exact_parts(args)?;
  if matches!(&solved, Expr::FunctionCall { name, .. } if name == "Solve")
    && let Some(roots) = solve_exponential_polynomial_reals(args)
  {
    return Ok(roots);
  }
  Ok(numericize_inexact_components(args, solved))
}

/// One term `c x^k E^(a x)` of an exponential polynomial in `x`.
struct ExpTerm {
  c: f64,
  k: i32,
  a: f64,
}

/// The terms of `f` as an exponential polynomial `Σ c x^k E^(a x)` in
/// `var` (`3^x` is `E^(Log[3] x)`), or None when `f` has another shape or
/// no exponential at all.
fn exponential_polynomial_terms(f: &Expr, var: &str) -> Option<Vec<ExpTerm>> {
  let to_f64 = crate::functions::math_ast::try_eval_to_f64;
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let mut terms = Vec::new();
  for term in collect_additive_terms(&expand_and_combine(f)) {
    let (sign, body) = match &term {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => (-1.0, operand.as_ref().clone()),
      _ => (1.0, term.clone()),
    };
    let mut t = ExpTerm {
      c: sign,
      k: 0,
      a: 0.0,
    };
    for factor in extract_times_factors(&body) {
      if is_constant_wrt(&factor, var) {
        t.c *= to_f64(&factor)?;
        continue;
      }
      if matches!(&factor, Expr::Identifier(n) if n == var) {
        t.k += 1;
        continue;
      }
      let (b, e) = match &factor {
        Expr::FunctionCall { name, args }
          if name == "Power" && args.len() == 2 =>
        {
          (args[0].clone(), args[1].clone())
        }
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => (left.as_ref().clone(), right.as_ref().clone()),
        _ => return None,
      };
      if matches!(&b, Expr::Identifier(n) if n == var) {
        match e {
          Expr::Integer(n) if (1..=64).contains(&n) => t.k += n as i32,
          _ => return None,
        }
        continue;
      }
      let base = to_f64(&b).filter(|v| *v > 0.0 && is_constant_wrt(&b, var))?;
      // The exponent has to be affine in `var`.
      let slope = eval(call("D", vec![e.clone(), id_expr(var)]))?;
      let slope = to_f64(&slope)?;
      let offset = to_f64(&eval(substitute_expr(
        &e,
        &id_expr(var),
        &Expr::Integer(0),
      ))?)?;
      let rebuilt = eval(minus2(
        e.clone(),
        plus2(times2(Expr::Real(slope), id_expr(var)), Expr::Real(offset)),
      ))?;
      if !to_f64(&rebuilt).is_some_and(|v| v.abs() < 1e-12) {
        return None;
      }
      t.c *= base.powf(offset);
      t.a += slope * base.ln();
    }
    terms.push(t);
  }
  terms.iter().any(|t| t.a != 0.0).then_some(terms)
}

fn exponential_polynomial_value(terms: &[ExpTerm], x: f64) -> f64 {
  terms
    .iter()
    .map(|t| t.c * x.powi(t.k) * (t.a * x).exp())
    .sum()
}

/// The real roots of an exponential polynomial: past a bound `L` one term
/// dominates the rest on each side, so every root lies in [-L, L], where a
/// fine sampling finds the sign changes and bisection the roots.
fn exponential_polynomial_real_roots(terms: &[ExpTerm]) -> Option<Vec<f64>> {
  // The term that dominates as x -> +Infinity (largest rate, then largest
  // power) or -Infinity (smallest rate, then largest power).
  let dominant = |right: bool| {
    terms
      .iter()
      .enumerate()
      .filter(|(_, t)| t.c != 0.0)
      .max_by(|(_, p), (_, q)| {
        let (pa, qa) = if right { (p.a, q.a) } else { (-p.a, -q.a) };
        pa.total_cmp(&qa).then(p.k.cmp(&q.k))
      })
      .map(|(i, _)| i)
  };
  let dominates = |i: usize, x: f64| {
    let term = |t: &ExpTerm| (t.c * x.powi(t.k) * (t.a * x).exp()).abs();
    let others: f64 = terms
      .iter()
      .enumerate()
      .filter(|(j, _)| *j != i)
      .map(|(_, t)| term(t))
      .sum();
    term(&terms[i]) > 2.0 * others
  };
  let (right, left) = (dominant(true)?, dominant(false)?);
  let mut bound = 1.0f64;
  while !(dominates(right, bound) && dominates(left, -bound)) {
    bound *= 2.0;
    if bound > 4096.0 {
      return None;
    }
  }
  let f = |x: f64| exponential_polynomial_value(terms, x);
  let n = 20_000;
  let step = 2.0 * bound / n as f64;
  let mut roots: Vec<f64> = Vec::new();
  let mut prev = (-bound, f(-bound));
  for i in 1..=n {
    let x = -bound + step * i as f64;
    let y = f(x);
    if y == 0.0 {
      roots.push(x);
    } else if prev.1 != 0.0 && (prev.1 < 0.0) != (y < 0.0) {
      let (mut lo, mut hi) = (prev.0, x);
      let lo_negative = prev.1 < 0.0;
      for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        if mid == lo || mid == hi {
          break;
        }
        if (f(mid) < 0.0) == lo_negative {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      roots.push(f64::midpoint(lo, hi));
    }
    prev = (x, y);
  }
  Some(roots)
}

/// `Solve[lhs == rhs, x, Reals]` for an exponential polynomial: its real
/// roots, exact where a small rational is one, otherwise as
/// `Root[{f, x0}]` with `f` the pure function `lhs - rhs` — as wolframscript
/// gives `Solve[2^x + 3^x == 10, x, Reals]`.
fn solve_exponential_polynomial_reals(args: &[Expr]) -> Option<Expr> {
  let [equation, variable, domain] = args else {
    return None;
  };
  if !matches!(domain, Expr::Identifier(d) if d == "Reals") {
    return None;
  }
  let var = match variable {
    Expr::Identifier(v) => v.clone(),
    Expr::List(vs) => match vs.as_slice() {
      [Expr::Identifier(v)] => v.clone(),
      _ => return None,
    },
    _ => return None,
  };
  let Expr::Comparison {
    operands,
    operators,
  } = equation
  else {
    return None;
  };
  if operators.as_slice() != [ComparisonOp::Equal] || operands.len() != 2 {
    return None;
  }
  let f = crate::evaluator::evaluate_expr_to_expr(&minus2(
    operands[0].clone(),
    operands[1].clone(),
  ))
  .ok()?;
  let terms = exponential_polynomial_terms(&f, &var)?;
  let roots = exponential_polynomial_real_roots(&terms)?;
  let x = id_expr(&var);
  let body = substitute_expr(&f, &x, &Expr::Slot(1));
  let function = Expr::Function {
    body: Box::new(body),
  };
  let mut solutions: Vec<Expr> = Vec::with_capacity(roots.len());
  for r in roots {
    // A small rational that is a root exactly is given as itself.
    let exact = (1..=6).find_map(|q: i128| {
      let p = (r * q as f64).round();
      if (p / q as f64 - r).abs() > 1e-9 * r.abs().max(1.0) {
        return None;
      }
      let candidate = crate::evaluator::evaluate_expr_to_expr(&call(
        "Rational",
        vec![Expr::Integer(p as i128), Expr::Integer(q)],
      ))
      .ok()?;
      let value = crate::evaluator::evaluate_expr_to_expr(&substitute_expr(
        &f, &x, &candidate,
      ))
      .ok()?;
      matches!(value, Expr::Integer(0)).then_some(candidate)
    });
    let value = exact.unwrap_or_else(|| {
      // The bisection's root, polished at a higher working precision.
      let x0 = transcendental_root_value(&function, &Expr::Real(r))
        .unwrap_or(Expr::Real(r));
      call("Root", vec![Expr::List(vec![function.clone(), x0].into())])
    });
    solutions.push(Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(x.clone()),
        replacement: Box::new(value),
      }]
      .into(),
    ));
  }
  crate::evaluator::evaluate_function_call_ast(
    "Sort",
    &[Expr::List(solutions.into())],
  )
  .ok()
}

/// A variable tied, through the equations it shares with others, to an
/// equation holding an inexact number gets an inexact value, as in
/// wolframscript: `Solve[{0.5 x + y == 1., x == 0}, {x, y}]` gives
/// `x -> 0.`, while `z == 0` on its own keeps `z -> 0`. The variables and
/// equations form a graph, and every exact number a solution assigns to a
/// variable in a component with an inexact equation is numericized.
fn numericize_inexact_components(args: &[Expr], solved: Expr) -> Expr {
  use crate::functions::math_ast::numeric_utils::contains_inexact_real;
  if args.len() < 2 || !contains_inexact_real(&args[0]) {
    return solved;
  }
  let Expr::List(solutions) = &solved else {
    return solved;
  };
  let vars: Vec<String> = match &args[1] {
    Expr::Identifier(v) => vec![v.clone()],
    Expr::List(items) => items
      .iter()
      .filter_map(|v| match v {
        Expr::Identifier(n) => Some(n.clone()),
        _ => None,
      })
      .collect(),
    _ => return solved,
  };
  // The individual equations of a list or conjunction of them.
  fn split_equations(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::List(items) => {
        for i in items {
          split_equations(i, out);
        }
      }
      other => {
        let mut parts = Vec::new();
        collect_and_constraints(other, &mut parts);
        if parts.len() == 1 {
          out.extend(parts);
        } else {
          for p in &parts {
            split_equations(p, out);
          }
        }
      }
    }
  }
  let mut equations = Vec::new();
  split_equations(&args[0], &mut equations);
  // Union-find over the variables.
  let mut parent: Vec<usize> = (0..vars.len()).collect();
  fn root(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
      r = parent[r];
    }
    parent[i] = r;
    r
  }
  let mut inexact_vars = vec![false; vars.len()];
  for eq in &equations {
    let present: Vec<usize> = (0..vars.len())
      .filter(|&i| contains_var(eq, &vars[i]))
      .collect();
    for w in present.windows(2) {
      let (a, b) = (root(&mut parent, w[0]), root(&mut parent, w[1]));
      parent[a] = b;
    }
    if contains_inexact_real(eq) {
      for &i in &present {
        inexact_vars[i] = true;
      }
    }
  }
  let mut inexact_root = vec![false; vars.len()];
  for i in 0..vars.len() {
    if inexact_vars[i] {
      let r = root(&mut parent, i);
      inexact_root[r] = true;
    }
  }
  let numericize = |name: &str, value: &Expr| -> Option<Expr> {
    let i = vars.iter().position(|v| v == name)?;
    let r = root(&mut parent.clone(), i);
    let exact_number = matches!(value, Expr::Integer(_))
      || matches!(value, Expr::FunctionCall { name, .. } if name == "Rational");
    (inexact_root[r] && exact_number)
      .then(|| {
        crate::evaluator::evaluate_expr_to_expr(&call1("N", value.clone())).ok()
      })
      .flatten()
  };
  Expr::List(
    solutions
      .iter()
      .map(|sol| match sol {
        Expr::List(rules) => Expr::List(
          rules
            .iter()
            .map(|rule| match rule {
              Expr::Rule {
                pattern,
                replacement,
              } => match pattern.as_ref() {
                Expr::Identifier(name) => match numericize(name, replacement) {
                  Some(v) => rule_expr(pattern.as_ref().clone(), v),
                  None => rule.clone(),
                },
                _ => rule.clone(),
              },
              other => other.clone(),
            })
            .collect(),
        ),
        other => other.clone(),
      })
      .collect(),
  )
}

fn solve_ast_exact_parts(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // A part of an undefined list can be a variable (`s[[1]]` in a
  // spreadsheet-style system): each is solved for as a placeholder symbol
  // and put back into the solutions.
  if let Some(vars) = args.get(1) {
    let parts: Vec<&Expr> = match vars {
      Expr::List(items) => items
        .iter()
        .filter(|v| matches!(v, Expr::Part { .. }))
        .collect(),
      v @ Expr::Part { .. } => vec![v],
      _ => Vec::new(),
    };
    if !parts.is_empty() {
      let names: Vec<String> = (1..=parts.len())
        .map(|k| format!("Woxi`SolvePartVariable{k}"))
        .collect();
      let renamed: Vec<Expr> = args
        .iter()
        .map(|a| {
          parts.iter().zip(&names).fold(a.clone(), |e, (part, name)| {
            substitute_expr(&e, part, &Expr::Identifier(name.clone()))
          })
        })
        .collect();
      let solved = solve_ast(&renamed)?;
      if matches!(&solved, Expr::FunctionCall { name, .. } if name == "Solve") {
        return Ok(unevaluated("Solve", args));
      }
      return Ok(parts.iter().zip(&names).fold(solved, |e, (part, name)| {
        crate::syntax::substitute_variable(&e, name, part)
      }));
    }
  }
  // A disjunction (also one inside a conjunction) is solved branch by
  // branch: `x == 1 || x == 2`, `(x == -1 || x == 2) && x + 1 != 0`.
  if args.len() == 2
    && contains_or(&args[0])
    && let Ok(expanded) = crate::evaluator::evaluate_expr_to_expr(&call1(
      "LogicalExpand",
      args[0].clone(),
    ))
    && let Some(branches) = or_operands(&expanded)
  {
    let mut solutions: Vec<Expr> = Vec::new();
    // The branches report no svars of their own; the disjunction does, once.
    let depth = SOLVE_DEPTH.with(|d| {
      let v = d.get();
      d.set(v + 1);
      v
    });
    let solved: Result<Vec<Expr>, InterpreterError> = branches
      .into_iter()
      .map(|branch| {
        let mut branch_args = args.to_vec();
        branch_args[0] = branch;
        solve_ast(&branch_args)
      })
      .collect();
    SOLVE_DEPTH.with(|d| d.set(depth));
    for result in solved? {
      match &result {
        Expr::List(found) => {
          for sol in found {
            if !solutions
              .iter()
              .any(|s| crate::evaluator::pattern_matching::expr_equal(s, sol))
            {
              solutions.push(sol.clone());
            }
          }
        }
        _ => return Ok(unevaluated("Solve", args)),
      }
    }
    sort_solutions(&mut solutions);
    if depth == 0
      && !in_nsolve()
      && let Expr::List(requested) = &args[1]
      && solutions.iter().any(|sol| {
        let Expr::List(rules) = sol else { return false };
        requested.iter().any(|v| {
          !rules.iter().any(|r| {
            matches!(r, Expr::Rule { pattern, .. }
              if crate::evaluator::pattern_matching::expr_equal(pattern, v))
          })
        })
      })
    {
      crate::emit_message(
        "Solve::svars: Equations may not give solutions for all \"solve\" variables.",
      );
    }
    return Ok(Expr::List(solutions.into()));
  }
  // `a b^t == c d^t`: the exponentials of one product combine into a single
  // power, `(b/d)^t`, which the exponential solver inverts.
  if let [eq, Expr::Identifier(var)] = args
    && let Some(combined) = combine_exponential_factors(eq, var)
  {
    return solve_ast(&[combined, args[1].clone()]);
  }
  let result = drop_failing_system_solutions(args, solve_ast_entry(args)?);
  // A system with machine coefficients whose inexact elimination lost
  // solutions is re-solved exactly and numericized, with Solve::ratnz, as
  // wolframscript does (PhasePlaneAnalysisOfABiochemicalReactor).
  // A system written as a conjunction (`a == b && c == d`) is re-solved
  // only when the inexact route found nothing at all.
  let conjunction;
  let system: Option<&[Expr]> = match args.first() {
    Some(Expr::List(eqs)) if eqs.len() >= 2 => Some(eqs.as_slice()),
    Some(Expr::FunctionCall { name, args: conj })
      if name == "And" && matches!(&result, Expr::List(r) if r.is_empty()) =>
    {
      Some(conj.as_slice())
    }
    Some(
      e @ Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::And,
        ..
      },
    ) if matches!(&result, Expr::List(r) if r.is_empty()) => {
      conjunction = vec![e.clone()];
      Some(conjunction.as_slice())
    }
    _ => None,
  };
  if args.len() == 2
    && let Some(eqs) = system
    && crate::functions::math_ast::contains_inexact_real(&args[0])
    && !has_radical(&args[0])
    && let Ok(exact) = crate::evaluator::evaluate_function_call_ast(
      "Rationalize",
      &[args[0].clone(), Expr::Integer(0)],
    )
    && !crate::functions::math_ast::contains_inexact_real(&exact)
  {
    let found = |r: &Expr| match r {
      Expr::List(sols) => sols.len(),
      _ => 0,
    };
    // wolframscript takes the exact route for every nonlinear system
    // (some second derivative of an equation is nonzero).
    let vars: Vec<Expr> = match &args[1] {
      Expr::List(vs) => vs.to_vec(),
      v => vec![v.clone()],
    };
    // `Or[eq1, eq2]` and `And[…]` are read through to their equations: a
    // logical head is not itself nonlinear.
    fn equations(e: &Expr, out: &mut Vec<Expr>) {
      match e {
        Expr::FunctionCall { name, args } if name == "Or" || name == "And" => {
          args.iter().for_each(|a| equations(a, out));
        }
        Expr::BinaryOp {
          op:
            crate::syntax::BinaryOperator::And | crate::syntax::BinaryOperator::Or,
          left,
          right,
        } => {
          equations(left, out);
          equations(right, out);
        }
        other => out.push(other.clone()),
      }
    }
    let mut parts = Vec::new();
    for eq in eqs {
      equations(eq, &mut parts);
    }
    let nonlinear = parts.iter().any(|eq| {
      let g = match eq {
        Expr::Comparison { operands, .. } if operands.len() == 2 => {
          plus(vec![operands[0].clone(), neg(operands[1].clone())])
        }
        other => other.clone(),
      };
      // A vector equation differentiates to a list of zeros.
      fn is_zero(e: &Expr) -> bool {
        match e {
          Expr::Integer(0) => true,
          Expr::Real(r) => *r == 0.0,
          Expr::List(items) => items.iter().all(is_zero),
          _ => false,
        }
      }
      // The first derivatives settle a linear system with one `D` per
      // variable: when none of them mentions a variable any more, every
      // second derivative vanishes.
      vars.iter().any(|a| {
        let Ok(first) = crate::evaluator::evaluate_function_call_ast(
          "D",
          &[g.clone(), a.clone()],
        ) else {
          return true;
        };
        let free_of_vars = vars.iter().all(|b| match b {
          Expr::Identifier(name) => {
            crate::functions::calculus_ast::is_constant_wrt(&first, name)
          }
          _ => crate::evaluator::evaluate_function_call_ast(
            "FreeQ",
            &[first.clone(), b.clone()],
          )
          .is_ok_and(|r| matches!(&r, Expr::Identifier(t) if t == "True")),
        });
        !free_of_vars
          && vars.iter().any(|b| {
            !crate::evaluator::evaluate_function_call_ast(
              "D",
              &[first.clone(), b.clone()],
            )
            .is_ok_and(|d| is_zero(&d))
          })
      })
    });
    // A linear system is re-solved only when the inexact elimination found
    // nothing; the exact route is costly for large systems.
    if !nonlinear && found(&result) > 0 {
      return Ok(result);
    }
    let exact_result = {
      let _numeric = NumericRootsScope::new();
      solve_ast_entry(&[exact, args[1].clone()])?
    };
    if found(&exact_result) > found(&result)
      || (nonlinear && found(&exact_result) > 0)
    {
      // NSolve answers numerically by design and says nothing about it;
      // neither does Solve for equations with radicals of the unknowns
      // (`Norm[{x, y} - c] == r`), which Wolfram solves another way.
      let polynomial = parts.iter().all(|eq| {
        let difference = match eq {
          Expr::Comparison { operands, .. } if operands.len() == 2 => {
            plus(vec![operands[0].clone(), neg(operands[1].clone())])
          }
          other => other.clone(),
        };
        matches!(
          crate::evaluator::evaluate_function_call_ast(
            "PolynomialQ",
            &[difference, Expr::List(vars.clone().into())],
          ),
          Ok(Expr::Identifier(ref t)) if t == "True"
        )
      });
      if !in_nsolve() && polynomial {
        crate::emit_message(
          "Solve::ratnz: Solve was unable to solve the system with inexact coefficients. The answer was obtained by solving a corresponding exact system and numericizing the result.",
        );
      }
      // In the exact solution's order, as wolframscript numericizes it.
      return crate::evaluator::evaluate_function_call_ast(
        "N",
        &[exact_result],
      );
    }
  }
  if matches!(&result, Expr::FunctionCall { name, .. } if name == "Solve")
    && let [eq, Expr::Identifier(var)] = args
    && let Some(isolated) = solve_by_isolating_power(eq, var)
  {
    return Ok(isolated);
  }
  // A polynomial equation with inexact coefficients that stayed unsolved
  // (a cubic in a parameter) is solved exactly and numericized, as
  // wolframscript does.
  if matches!(&result, Expr::FunctionCall { name, .. } if name == "Solve")
    && let [
      eq @ Expr::Comparison { operands, .. },
      var @ Expr::Identifier(_),
    ] = args
    && operands.len() == 2
    && !in_nsolve()
    && crate::functions::math_ast::contains_inexact_real(eq)
    && matches!(
      crate::evaluator::evaluate_function_call_ast(
        "PolynomialQ",
        &[plus(vec![operands[0].clone(), neg(operands[1].clone())]), var.clone()],
      ),
      Ok(Expr::Identifier(ref t)) if t == "True"
    )
    && let Ok(exact) = crate::evaluator::evaluate_function_call_ast(
      "Rationalize",
      &[eq.clone(), Expr::Integer(0)],
    )
    && !crate::functions::math_ast::contains_inexact_real(&exact)
    && let Ok(exact_result @ Expr::List(_)) =
      solve_ast_entry(&[exact, var.clone()])
  {
    crate::emit_message(
      "Solve::ratnz: Solve was unable to solve the system with inexact coefficients. The answer was obtained by solving a corresponding exact system and numericizing the result.",
    );
    return crate::evaluator::evaluate_function_call_ast("N", &[exact_result]);
  }
  Ok(result)
}

/// Whether a logical combination holds an `Or` (at the top or under `And`).
fn contains_or(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args } if name == "Or" => args.len() >= 2,
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      ..
    } => true,
    Expr::FunctionCall { name, args } if name == "And" => {
      args.iter().any(contains_or)
    }
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => contains_or(left) || contains_or(right),
    _ => false,
  }
}

/// The disjuncts of an `Or`.
fn or_operands(e: &Expr) -> Option<Vec<Expr>> {
  match e {
    Expr::FunctionCall { name, args } if name == "Or" => Some(args.to_vec()),
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      left,
      right,
    } => {
      let mut out = or_operands(left).unwrap_or_else(|| vec![(**left).clone()]);
      out.extend(or_operands(right).unwrap_or_else(|| vec![(**right).clone()]));
      Some(out)
    }
    _ => None,
  }
}

/// Rewrites every product holding two or more powers `b^(c var)` (bases
/// free of `var`, exponents proportional to it) into one power
/// `(b1^c1 b2^c2 …)^var`, evaluated. `None` when no product has two.
fn combine_exponential_factors(expr: &Expr, var: &str) -> Option<Expr> {
  // `exponent = c var + d` with `c`, `d` free of `var`: `(c, d)`.
  fn affine(exponent: &Expr, var: &str) -> Option<(Expr, Expr)> {
    if is_constant_wrt(exponent, var) {
      return None;
    }
    let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
    let c = eval(call("D", vec![exponent.clone(), id_expr(var)]))?;
    if !is_constant_wrt(&c, var) {
      return None;
    }
    let d = eval(call(
      "ReplaceAll",
      vec![exponent.clone(), rule_expr(id_expr(var), Expr::Integer(0))],
    ))?;
    Some((c, d))
  }
  // `(b, exponent)` of a power, `(b^e)^k` read as `b^(e k)`.
  fn power_parts(e: &Expr) -> Option<(Expr, Expr)> {
    let (base, exponent) = match e {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref(), right.as_ref()),
      _ => return None,
    };
    match power_parts(base) {
      Some((inner_base, inner_exp)) => {
        Some((inner_base, times2(inner_exp, exponent.clone())))
      }
      None => Some((base.clone(), exponent.clone())),
    }
  }
  fn walk(e: &Expr, var: &str, changed: &mut bool) -> Expr {
    let rebuilt = crate::syntax::map_children(e, &|c| {
      let mut inner = false;
      let r = walk(c, var, &mut inner);
      if inner {
        CHANGED.with(|f| f.set(true));
      }
      r
    });
    if let Expr::FunctionCall { name, args } = &rebuilt
      && name == "Times"
    {
      let mut combined_base: Vec<Expr> = Vec::new();
      let mut rest: Vec<Expr> = Vec::new();
      for f in args {
        match power_parts(f) {
          Some((base, exponent))
            if is_constant_wrt(&base, var)
              && let Some((c, d)) = affine(&exponent, var) =>
          {
            combined_base.push(pow(base.clone(), c));
            if !matches!(d, Expr::Integer(0)) {
              rest.push(pow(base, d));
            }
          }
          _ => rest.push(f.clone()),
        }
      }
      if combined_base.len() >= 2 {
        *changed = true;
        let base =
          crate::evaluator::evaluate_expr_to_expr(&times(combined_base))
            .ok()
            .unwrap_or(Expr::Integer(1));
        rest.push(pow(base, id_expr(var)));
        return times(rest);
      }
    }
    rebuilt
  }
  thread_local! {
    static CHANGED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
  }
  // `p == q` (or `p + q == 0`) with both sides single products whose `var`
  // sits only in such exponents is `p/q == 1` (`-p/q == 1`), one product.
  let exponential_term = |t: &Expr| -> bool {
    if is_constant_wrt(t, var) {
      return false;
    }
    extract_times_factors(t).iter().all(|f| {
      is_constant_wrt(f, var)
        || power_parts(f).is_some_and(|(b, e)| {
          is_constant_wrt(&b, var) && affine(&e, var).is_some()
        })
    })
  };
  let as_ratio = |p: &Expr, q: &Expr| -> Option<Expr> {
    if !(exponential_term(p) && exponential_term(q)) {
      return None;
    }
    let ratio = crate::evaluator::evaluate_expr_to_expr(&times2(
      p.clone(),
      pow(q.clone(), Expr::Integer(-1)),
    ))
    .ok()?;
    Some(Expr::Comparison {
      operands: vec![ratio, Expr::Integer(1)],
      operators: vec![ComparisonOp::Equal],
    })
  };
  let ratio_form = match expr {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.as_slice() == [ComparisonOp::Equal] =>
    {
      let difference = crate::evaluator::evaluate_expr_to_expr(&minus2(
        operands[0].clone(),
        operands[1].clone(),
      ))
      .ok();
      match &difference {
        Some(Expr::FunctionCall { name, args })
          if name == "Plus" && args.len() == 2 =>
        {
          let negated =
            crate::evaluator::evaluate_expr_to_expr(&negate_expr(&args[1]))
              .ok()?;
          as_ratio(&args[0], &negated)
        }
        _ => as_ratio(&operands[0], &operands[1]),
      }
    }
    _ => None,
  };
  // A single product `K b1^(c1 var + d1) … == 1` becomes `B^var == 1/K`
  // with `B = b1^c1 …`, the shape the exponential solver inverts.
  if let Some(Expr::Comparison { operands, .. }) = &ratio_form {
    let mut bases = Vec::new();
    let mut constant = Vec::new();
    for f in extract_times_factors(&operands[0]) {
      match power_parts(&f) {
        Some((base, exponent))
          if is_constant_wrt(&base, var)
            && let Some((c, d)) = affine(&exponent, var) =>
        {
          bases.push(pow(base.clone(), c));
          constant.push(pow(base, d));
        }
        _ if is_constant_wrt(&f, var) => constant.push(f),
        _ => return None,
      }
    }
    let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
    let base = eval(times(bases))?;
    let rhs = eval(pow(times(constant), Expr::Integer(-1)))?;
    return Some(Expr::Comparison {
      operands: vec![pow(base, id_expr(var)), rhs],
      operators: vec![ComparisonOp::Equal],
    });
  }
  CHANGED.with(|f| f.set(false));
  let mut top = false;
  let out = walk(expr, var, &mut top);
  let any = top || CHANGED.with(std::cell::Cell::get);
  any.then_some(out)
}

/// `g(var) == 0` where `var` occurs only inside one power `(p var + q)^n` of
/// a linear expression, which the rest of `g` multiplies and shifts:
/// `2 (y + 1/3)^4 + a == 0`. The equation is a pure power in `z = p var +
/// q`, solved as such, and each root gives `var = (z - q)/p` — the way
/// wolframscript inverts the single occurrence instead of expanding a
/// quartic with symbolic coefficients.
fn solve_by_isolating_power(eq: &Expr, var: &str) -> Option<Expr> {
  let Expr::Comparison {
    operands,
    operators,
  } = eq
  else {
    return None;
  };
  if operands.len() != 2
    || operators.as_slice() != [crate::syntax::ComparisonOp::Equal]
  {
    return None;
  }
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let g = eval(minus2(operands[0].clone(), operands[1].clone()))?;
  let mentions = |e: &Expr| !is_constant_wrt(e, var);
  // The powers of an expression in `var`.
  fn powers<'a>(
    e: &'a Expr,
    var: &str,
    out: &mut Vec<(&'a Expr, &'a Expr, i128)>,
  ) {
    let pow_parts = match e {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => Some((left.as_ref(), right.as_ref())),
      _ => None,
    };
    if let Some((base, Expr::Integer(n))) = pow_parts
      && *n >= 2
      && !is_constant_wrt(base, var)
    {
      out.push((e, base, *n));
      return;
    }
    for child in crate::syntax::expr_children(e) {
      powers(child, var, out);
    }
  }
  let mut found = Vec::new();
  powers(&g, var, &mut found);
  let (power, base, n) = found.first().copied()?;
  let u = id_expr("Woxi`IsolatedPower");
  let replaced = eval(call(
    "ReplaceAll",
    vec![g.clone(), rule_expr(power.clone(), u.clone())],
  ))?;
  if mentions(&replaced) {
    return None;
  }
  // `replaced` is A u + B with A, B free of var.
  let a = eval(call(
    "Coefficient",
    vec![replaced.clone(), u.clone(), Expr::Integer(1)],
  ))?;
  let b = eval(call(
    "ReplaceAll",
    vec![replaced.clone(), rule_expr(u.clone(), Expr::Integer(0))],
  ))?;
  let check = eval(call(
    "Expand",
    vec![minus2(
      replaced.clone(),
      plus2(times2(a.clone(), u.clone()), b.clone()),
    )],
  ))?;
  if !matches!(check, Expr::Integer(0)) || matches!(a, Expr::Integer(0)) {
    return None;
  }
  // The base is p var + q.
  let p = eval(call("D", vec![base.clone(), id_expr(var)]))?;
  if mentions(&p) || matches!(p, Expr::Integer(0)) {
    return None;
  }
  let q = eval(call(
    "ReplaceAll",
    vec![base.clone(), rule_expr(id_expr(var), Expr::Integer(0))],
  ))?;
  let z = id_expr("Woxi`IsolatedBase");
  let roots = eval(call(
    "Solve",
    vec![
      Expr::Comparison {
        operands: vec![
          plus2(times2(a, pow(z.clone(), Expr::Integer(n))), b),
          Expr::Integer(0),
        ],
        operators: vec![crate::syntax::ComparisonOp::Equal],
      },
      z.clone(),
    ],
  ))?;
  let Expr::List(solutions) = &roots else {
    return None;
  };
  let mut out = Vec::with_capacity(solutions.len());
  for sol in solutions {
    let Expr::List(rules) = sol else {
      return None;
    };
    let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
      return None;
    };
    let value = eval(solve_divide(
      &minus2(replacement.as_ref().clone(), q.clone()),
      &p,
    ))?;
    out.push(Expr::List(vec![rule_expr(id_expr(var), value)].into()));
  }
  Some(Expr::List(out.into()))
}

/// `Solve[eq, x]` for an equation linear in `x` whose coefficients hold a
/// power of a sum with a large integer exponent (`(12 + r)^360`, a 30-year
/// monthly loan in PayThePoints): `x == -(eq /. x -> 0)/D[eq, x]`, read off
/// without expanding anything. The general path expands that power, which
/// takes practically forever; wolframscript answers instantly. Ordinary
/// linear equations keep the general path and its answer forms.
fn solve_linear_with_huge_powers(args: &[Expr]) -> Option<Expr> {
  const HUGE: i128 = 40;
  let [eq, Expr::Identifier(var)] = args else {
    return None;
  };
  fn has_huge_sum_power(e: &Expr) -> bool {
    let huge = |base: &Expr, exp: &Expr| {
      matches!(exp, Expr::Integer(n) if n.abs() >= HUGE)
        && matches!(base, Expr::FunctionCall { name, .. } if name == "Plus")
          | matches!(
            base,
            Expr::BinaryOp {
              op: BinaryOperator::Plus,
              ..
            }
          )
          | matches!(
            base,
            Expr::BinaryOp {
              op: BinaryOperator::Minus,
              ..
            }
          )
    };
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if huge(left, right) => true,
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 && huge(&args[0], &args[1]) =>
      {
        true
      }
      _ => crate::syntax::expr_children(e)
        .into_iter()
        .any(has_huge_sum_power),
    }
  }
  let (lhs, rhs) = match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.as_slice() == [ComparisonOp::Equal] =>
    {
      (&operands[0], &operands[1])
    }
    _ => return None,
  };
  if !has_huge_sum_power(eq) {
    return None;
  }
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let difference = minus2(lhs.clone(), rhs.clone());
  let slope = eval(call("D", vec![difference.clone(), id_expr(var)]))?;
  if contains_var(&slope, var) || matches!(slope, Expr::Integer(0)) {
    return None;
  }
  let intercept = eval(call(
    "ReplaceAll",
    vec![difference, rule_expr(id_expr(var), Expr::Integer(0))],
  ))?;
  let value = eval(div2(neg(intercept), slope))?;
  Some(Expr::List(
    vec![Expr::List(vec![rule_expr(id_expr(var), value)].into())].into(),
  ))
}

fn solve_ast_entry(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Some(result) = solve_piecewise_max_min(args) {
    return result;
  }
  if let Some(result) = solve_linear_with_huge_powers(args) {
    return Ok(result);
  }
  // Only a trailing rule can be an option — `Solve[eqns, vars, Reals]` passes a
  // domain in the same slot.
  let mut modulus: Option<i128> = None;
  let mut max_roots: Option<usize> = None;
  let mut positional: Vec<Expr> = Vec::with_capacity(args.len());
  for arg in args {
    let Expr::Rule {
      pattern,
      replacement,
    } = arg
    else {
      positional.push(arg.clone());
      continue;
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      positional.push(arg.clone());
      continue;
    };
    match name.as_str() {
      "Modulus" => match replacement.as_ref() {
        Expr::Integer(n) if *n > 1 => modulus = Some(*n),
        // Modulus -> 0 is the default: no modular arithmetic.
        Expr::Integer(0) => {}
        _ => positional.push(arg.clone()),
      },
      "MaxRoots" => match replacement.as_ref() {
        Expr::Integer(n) if *n >= 1 => max_roots = Some(*n as usize),
        Expr::Identifier(id) if id == "Infinity" || id == "Automatic" => {}
        other => {
          crate::emit_message(&format!(
            "Solve::maxrts: The value {} of the MaxRoots option is not a \
             positive integer, Infinity or Automatic.",
            crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
          ));
          return Ok(unevaluated("Solve", args));
        }
      },
      _ => positional.push(arg.clone()),
    }
  }

  // `Solve[eqns, vars, elims]` solves for `vars` with the `elims` eliminated.
  if positional.len() == 3
    && let (Expr::List(vars), Expr::List(elims)) =
      (&positional[1], &positional[2])
    && !elims.is_empty()
    && vars
      .iter()
      .chain(elims.iter())
      .all(|v| matches!(v, Expr::Identifier(_)))
    && let Some(solved) =
      solve_with_eliminated(&positional[0], vars.as_slice(), elims.as_slice())
  {
    return solved;
  }

  let mut squared_abs = false;
  // An unknown written as a curried call (`c[1][t]`, a method-of-lines
  // grid value) is solved for as a fresh symbol and put back afterwards.
  let curried: Vec<Expr> = match positional.get(1) {
    Some(Expr::List(vars)) => vars.to_vec(),
    Some(v) => vec![v.clone()],
    None => vec![],
  }
  .into_iter()
  .filter(|v| matches!(v, Expr::CurriedCall { .. }))
  .collect();
  if !curried.is_empty() {
    let renames: Vec<(Expr, String)> = curried
      .into_iter()
      .enumerate()
      .map(|(i, v)| (v, format!("Solve$var${i}")))
      .collect();
    let mut renamed: Vec<Expr> = args
      .iter()
      .map(|a| crate::functions::ode_ast::rename_compound_heads(a, &renames))
      .collect();
    // The renamed unknowns are plain symbols now; evaluate the equations
    // again so they simplify around them.
    renamed[0] = crate::evaluator::evaluate_expr_to_expr(&renamed[0])?;
    let solved = solve_ast(&renamed)?;
    return Ok(crate::functions::ode_ast::restore_compound_heads(
      &solved, &renames,
    ));
  }
  let several_unknowns =
    matches!(positional.get(1), Some(Expr::List(vars)) if vars.len() >= 2);
  if let Some(first) = positional.first_mut() {
    *first = expand_chained_equalities(first);
    let (rewritten, changed) = abs_squared_to_squares(first);
    *first = rewritten;
    squared_abs = changed;
    // A radical equation in several unknowns has no elimination path of its
    // own; squaring it leaves the equivalent polynomial system.
    if several_unknowns {
      *first = square_radical_equations(first);
    }
  }

  let depth = SOLVE_DEPTH.with(|d| {
    let v = d.get();
    d.set(v + 1);
    v
  });
  let solutions = solve_with_var_selection(&positional, modulus, args);
  SOLVE_DEPTH.with(|d| d.set(depth));
  let mut solutions = solutions?;
  if matches!(&solutions, Expr::FunctionCall { name, .. } if name == "Solve")
    && let Some(solved) = solve_by_linear_substitution(&positional)
  {
    solutions = solved;
  }
  if matches!(&solutions, Expr::FunctionCall { name, .. } if name == "Solve")
    && let Some(Expr::List(var_list)) = positional.get(1)
    && let Some(domain) = match positional.len() {
      2 => Some(false),
      3 if matches!(&positional[2], Expr::Identifier(d) | Expr::Constant(d) if d == "Reals") => {
        Some(true)
      }
      _ => None,
    }
    && let Some(solved) =
      solve_by_groebner_shape(&positional[0], var_list, domain)
  {
    solutions = solved;
  }
  // `Abs[u]^2` was solved as `u^2`, which also has the complex roots no
  // real `u` can reach.
  if squared_abs && let Expr::List(sols) = &solutions {
    solutions = Expr::List(
      sols
        .iter()
        .filter(|sol| !contains_complex(sol))
        .cloned()
        .collect(),
    );
  }
  // Solve gives the generic solutions: a branch that only holds for special
  // parameter values (`{…, a -> 0}` from a coefficient vanishing) is no
  // solution for the variables asked for, and a rule for anything but one
  // of them is no part of an answer.
  if let Some(spec) = positional.get(1)
    // `Solve[eqns, Reals]` names a domain, not the variables.
    && !matches!(spec, Expr::Identifier(d)
      if matches!(d.as_str(), "Reals" | "Integers" | "Complexes" | "Rationals"
        | "Booleans" | "Algebraics" | "Primes"))
    && let Expr::List(sols) = &solutions
  {
    let requested: Vec<Expr> = match spec {
      Expr::List(items) => items.to_vec(),
      single => vec![single.clone()],
    };
    let kept: Vec<Expr> = sols
      .iter()
      .filter(|sol| match sol {
        Expr::List(rules) => rules.iter().all(|r| match r {
          Expr::Rule { pattern, .. } => requested.iter().any(|v| {
            crate::evaluator::pattern_matching::expr_equal(pattern, v)
          }),
          _ => true,
        }),
        _ => true,
      })
      .cloned()
      .collect();
    if kept.len() != sols.len() {
      solutions = Expr::List(kept.into());
    }
  }
  // The outermost call reports when the equations left some of the
  // explicitly requested variables free.
  if depth == 0
    && !in_nsolve()
    && let Some(Expr::List(requested)) = positional.get(1)
    && let Expr::List(sols) = &solutions
    && let Some(Expr::List(first)) = sols.first()
    && requested.iter().any(|v| {
      !first.iter().any(|r| {
        matches!(r, Expr::Rule { pattern, .. }
          if crate::evaluator::pattern_matching::expr_equal(pattern, v))
      })
    })
  {
    let head = IFUN_HEAD.with(|h| *h.borrow());
    crate::emit_message(&format!(
      "{head}::svars: Equations may not give solutions for all \"solve\" variables."
    ));
  }
  // MaxRoots keeps the leading solutions of a solution list; anything else
  // (an unevaluated call, a conditional form) passes through untouched.
  match (max_roots, &solutions) {
    (Some(n), Expr::List(items)) if items.len() > n => {
      Ok(Expr::List(items.iter().take(n).cloned().collect()))
    }
    _ => Ok(solutions),
  }
}

/// Collect the sums that contain `var` and stand as a denominator: the base
/// of a negative integer power or the divisor of a quotient.
fn collect_denominator_sums(expr: &Expr, var: &str, out: &mut Vec<Expr>) {
  let is_sum = |e: &Expr| {
    matches!(
      e,
      Expr::BinaryOp {
        op: BinaryOperator::Plus | BinaryOperator::Minus,
        ..
      }
    ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Plus")
  };
  let note = |base: &Expr, out: &mut Vec<Expr>| {
    if is_sum(base) && super::eliminate::contains_var(base, var) {
      out.push(base.clone());
    }
  };
  match expr {
    Expr::BinaryOp { op, left, right } => {
      match op {
        BinaryOperator::Power if matches!(right.as_ref(), Expr::Integer(n) if *n < 0) =>
        {
          note(left, out);
        }
        BinaryOperator::Divide => note(right, out),
        _ => {}
      }
      collect_denominator_sums(left, var, out);
      collect_denominator_sums(right, var, out);
    }
    Expr::FunctionCall { name, args } => {
      if name == "Power"
        && args.len() == 2
        && matches!(&args[1], Expr::Integer(n) if *n < 0)
      {
        note(&args[0], out);
      }
      for a in args {
        collect_denominator_sums(a, var, out);
      }
    }
    Expr::UnaryOp { operand, .. } => {
      collect_denominator_sums(operand, var, out);
    }
    Expr::List(items) => {
      for i in items {
        collect_denominator_sums(i, var, out);
      }
    }
    Expr::Comparison { operands, .. } => {
      for o in operands {
        collect_denominator_sums(o, var, out);
      }
    }
    _ => {}
  }
}

/// Solve a zero-dimensional polynomial system with rational coefficients whose
/// lexicographic Groebner basis is in shape position: one polynomial `R` in
/// the first unknown alone, and for every other unknown a polynomial linear in
/// it (with a constant leading coefficient) and otherwise free of the rest.
/// The first unknown then runs over the roots of `R` as `Root` objects and the
/// others follow as polynomials in it. A fallback for systems in which no
/// single equation can be solved for one of the unknowns.
pub(crate) fn solve_by_groebner_shape(
  eqs: &Expr,
  var_list: &[Expr],
  reals_only: bool,
) -> Option<Expr> {
  if var_list.len() < 2
    || !var_list.iter().all(|v| matches!(v, Expr::Identifier(_)))
  {
    return None;
  }
  let mut conjuncts = Vec::new();
  crate::functions::polynomial_ast::reduce::collect_and_constraints(
    eqs,
    &mut conjuncts,
  );
  let eqs: Vec<Expr> = match eqs {
    Expr::List(items) => items.to_vec(),
    _ => conjuncts,
  };
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let eval_call = |name: &str, args: Vec<Expr>| {
    crate::evaluator::evaluate_function_call_ast(name, &args).ok()
  };
  let mut polys = Vec::new();
  for eq in &eqs {
    let (lhs, rhs, CompOp::Equal) =
      crate::functions::polynomial_ast::reduce::extract_comparison(eq)?
    else {
      return None;
    };
    let diff = eval(plus(vec![lhs, neg(rhs)]))?;
    let numer =
      eval_call("Numerator", vec![eval_call("Together", vec![diff])?])?;
    polys.push(eval_call("Expand", vec![numer])?);
  }
  let reversed: Vec<Expr> = var_list.iter().rev().cloned().collect();
  let basis = eval_call(
    "GroebnerBasis",
    vec![Expr::List(polys.into()), Expr::List(reversed.into())],
  )?;
  let Expr::List(basis) = &basis else {
    return None;
  };
  if basis.len() != var_list.len() {
    return None;
  }
  let is_rational = |c: &Expr| match c {
    Expr::Integer(_) => true,
    Expr::FunctionCall { name, args } if name == "Rational" => {
      args.len() == 2 && args.iter().all(|a| matches!(a, Expr::Integer(_)))
    }
    _ => false,
  };
  let x = &var_list[0];
  let Expr::Identifier(x_name) = x else {
    return None;
  };
  // R: the basis element free of every other unknown.
  let univariate = basis.iter().find(|b| {
    var_list[1..]
      .iter()
      .all(|v| matches!(&v, Expr::Identifier(n) if !contains_var(b, n)))
  })?;
  let coeff_list =
    eval_call("CoefficientList", vec![univariate.clone(), x.clone()])?;
  let Expr::List(coeffs) = &coeff_list else {
    return None;
  };
  let coeffs: Vec<Expr> = coeffs.to_vec();
  if coeffs.len() < 4 || !coeffs.iter().all(is_rational) {
    return None;
  }
  // Repeated roots would need multiplicities Solve reports differently.
  let gcd = eval_call(
    "PolynomialGCD",
    vec![
      univariate.clone(),
      eval_call("D", vec![univariate.clone(), x.clone()])?,
    ],
  )?;
  if !matches!(
    eval_call("Exponent", vec![gcd, x.clone()])?,
    Expr::Integer(0)
  ) {
    return None;
  }
  // For each other unknown: v = -c0(x)/c1.
  let mut others: Vec<(Expr, Expr)> = Vec::new();
  for v in &var_list[1..] {
    let Expr::Identifier(v_name) = v else {
      return None;
    };
    let element = basis.iter().find(|b| {
      contains_var(b, v_name)
        && var_list[1..].iter().all(|w| {
          matches!(w, Expr::Identifier(n) if n == v_name || !contains_var(b, n))
        })
    })?;
    if !matches!(
      eval_call("Exponent", vec![element.clone(), v.clone()])?,
      Expr::Integer(1)
    ) {
      return None;
    }
    let c1 = eval_call(
      "Coefficient",
      vec![element.clone(), v.clone(), Expr::Integer(1)],
    )?;
    let c0 = eval_call(
      "Coefficient",
      vec![element.clone(), v.clone(), Expr::Integer(0)],
    )?;
    if !is_rational(&c1) || matches!(c1, Expr::Integer(0)) {
      return None;
    }
    let value = eval(times(vec![
      Expr::Integer(-1),
      c0,
      pow(c1, Expr::Integer(-1)),
    ]))?;
    others.push((v.clone(), value));
  }
  let root_solutions = make_root_solutions(&coeffs, x_name)?;
  let Expr::List(root_sols) = &root_solutions else {
    return None;
  };
  let mut out = Vec::with_capacity(root_sols.len());
  for sol in root_sols {
    let Expr::List(rules) = sol else {
      return None;
    };
    // Over the reals only the roots of `R` that are real count, and each is
    // real exactly when its `Root` evaluates to a real number.
    if reals_only {
      let Expr::Rule { replacement, .. } = rules.first()? else {
        return None;
      };
      let im =
        eval_call("N", vec![eval_call("Im", vec![(**replacement).clone()])?])?;
      if !matches!(&im, Expr::Integer(0))
        && !matches!(expr_to_f64(&im), Ok(v) if v.abs() < 1e-9)
      {
        continue;
      }
    }
    let Expr::Rule { replacement, .. } = rules.first()? else {
      return None;
    };
    let mut row = vec![rule_expr(x.clone(), (**replacement).clone())];
    for (v, value) in &others {
      let at_root =
        crate::syntax::substitute_variable(value, x_name, replacement);
      row.push(rule_expr(v.clone(), eval(at_root)?));
    }
    out.push(Expr::List(row.into()));
  }
  Some(Expr::List(out.into()))
}

/// Solve a single equation whose only dependence on `var` runs through one
/// sum linear in `var` (`a + b q` standing as a denominator) by solving for
/// that sum as a fresh unknown and then solving the linear relation for `var`.
/// A fallback for equations the direct polynomial route cannot clear, e.g.
/// `1/(2 (a + b q)^2) - k/(a + b q) + c == 1/4`.
fn solve_by_linear_substitution(positional: &[Expr]) -> Option<Expr> {
  let [equation, Expr::Identifier(var)] = positional else {
    return None;
  };
  let (lhs, rhs, CompOp::Equal) =
    crate::functions::polynomial_ast::reduce::extract_comparison(equation)?
  else {
    return None;
  };
  let mut candidates = Vec::new();
  collect_denominator_sums(&lhs, var, &mut candidates);
  collect_denominator_sums(&rhs, var, &mut candidates);
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  for sum in candidates {
    // The sum has to be linear in `var`: a constant slope and offset.
    let slope =
      eval(call("D", vec![sum.clone(), Expr::Identifier(var.clone())]))?;
    if super::eliminate::contains_var(&slope, var) || is_zero_expr(&slope) {
      continue;
    }
    let offset = eval(crate::syntax::substitute_variable(
      &sum,
      var,
      &Expr::Integer(0),
    ))?;
    let name = "WoxiSolveSum";
    let fresh = id_expr(name);
    let new_lhs = substitute_expr(&lhs, &sum, &fresh);
    let new_rhs = substitute_expr(&rhs, &sum, &fresh);
    if super::eliminate::contains_var(&new_lhs, var)
      || super::eliminate::contains_var(&new_rhs, var)
    {
      continue;
    }
    let reduced = Expr::Comparison {
      operands: vec![new_lhs, new_rhs],
      operators: vec![ComparisonOp::Equal],
    };
    let Ok(Expr::List(ref sols)) = solve_ast(&[reduced, fresh]) else {
      continue;
    };
    let mut out = Vec::new();
    for sol in sols {
      let Expr::List(rules) = sol else {
        return None;
      };
      let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
        return None;
      };
      let value = eval(div2(
        minus2(replacement.as_ref().clone(), offset.clone()),
        slope.clone(),
      ))?;
      out.push(Expr::List(
        vec![rule_expr(Expr::Identifier(var.clone()), value)].into(),
      ));
    }
    return Some(Expr::List(out.into()));
  }
  None
}

/// Rewrites every `Abs[u]^2` of a symbolic `u` as `u^2` — the squared norm a
/// `Norm[{x - 1, y - 1}] == 2` equation reduces to — so the equation is an
/// ordinary polynomial one. The flag tells whether anything was rewritten.
fn abs_squared_to_squares(eqns: &Expr) -> (Expr, bool) {
  let changed = std::cell::Cell::new(false);
  let abs_argument = |e: &Expr| -> Option<Expr> {
    match e {
      Expr::FunctionCall { name, args } if name == "Abs" && args.len() == 1 => {
        Some(args[0].clone())
      }
      _ => None,
    }
  };
  let rewritten = crate::functions::string_ast::map_expr_tree(eqns, &|e| {
    let (base, exponent) = match e {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref(), right.as_ref()),
      _ => return None,
    };
    if !matches!(exponent, Expr::Integer(2)) {
      return None;
    }
    let inner = abs_argument(base)?;
    if try_eval_to_f64(&inner).is_some() {
      return None;
    }
    changed.set(true);
    Some(pow(inner, Expr::Integer(2)))
  });
  (rewritten, changed.get())
}

/// Squares the radical equations of a system: `Sqrt[p] == Sqrt[q]` becomes
/// `p == q`, and `Sqrt[p] == c` for a non-negative number `c` becomes
/// `p == c^2`. Both are equivalences, since the principal square root is
/// injective and non-negative. Looks through lists and `And`.
fn square_radical_equations(eqns: &Expr) -> Expr {
  use crate::functions::math_ast::is_sqrt;
  match eqns {
    Expr::List(items) => {
      Expr::List(items.iter().map(square_radical_equations).collect())
    }
    Expr::FunctionCall { name, args } if name == "And" => call(
      name.as_str(),
      args.iter().map(square_radical_equations).collect(),
    ),
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators[0] == ComparisonOp::Equal => {
      let equation = |lhs: Expr, rhs: Expr| Expr::Comparison {
        operands: vec![lhs, rhs],
        operators: vec![ComparisonOp::Equal],
      };
      let (lhs, rhs) = (&operands[0], &operands[1]);
      if let (Some(p), Some(q)) = (is_sqrt(lhs), is_sqrt(rhs)) {
        return equation(p.clone(), q.clone());
      }
      let (radical, other) = match (is_sqrt(lhs), is_sqrt(rhs)) {
        (Some(p), None) => (p, rhs),
        (None, Some(p)) => (p, lhs),
        _ => return eqns.clone(),
      };
      match try_eval_to_f64(other) {
        Some(c) if c >= 0.0 => {
          let squared = crate::evaluator::evaluate_expr_to_expr(&pow(
            other.clone(),
            Expr::Integer(2),
          ))
          .unwrap_or_else(|_| pow(other.clone(), Expr::Integer(2)));
          equation(radical.clone(), squared)
        }
        _ => eqns.clone(),
      }
    }
    _ => eqns.clone(),
  }
}

/// Rewrites a chained equality `a == b == c` as the conjunction
/// `a == b && b == c` it stands for, in the equation argument of `Solve`
/// and `Reduce` — at the top level, inside lists and inside `And`.
pub(crate) fn expand_chained_equalities(eqns: &Expr) -> Expr {
  match eqns {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() > 2
      && operators.iter().all(|op| *op == ComparisonOp::Equal) =>
    {
      call(
        "And",
        operands
          .windows(2)
          .map(|pair| Expr::Comparison {
            operands: pair.to_vec(),
            operators: vec![ComparisonOp::Equal],
          })
          .collect(),
      )
    }
    Expr::List(items) => {
      Expr::List(items.iter().map(expand_chained_equalities).collect())
    }
    Expr::FunctionCall { name, args } if name == "And" => call(
      name.as_str(),
      args.iter().map(expand_chained_equalities).collect(),
    ),
    _ => eqns.clone(),
  }
}

thread_local! {
  /// How deeply `solve_ast` is nested; only the outermost call reports
  /// `svars`.
  static SOLVE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
  /// How many `NSolve` calls are in progress. NSolve does not share Solve's
  /// handling of underdetermined systems: it solves a linear one for its
  /// *first* variables and reports no `svars`.
  static NSOLVE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Marks an `NSolve` in progress for as long as it is alive.
struct NSolveScope;

impl NSolveScope {
  fn new() -> Self {
    NSOLVE_DEPTH.with(|d| d.set(d.get() + 1));
    Self
  }
}

impl Drop for NSolveScope {
  fn drop(&mut self) {
    NSOLVE_DEPTH.with(|d| d.set(d.get() - 1));
  }
}

fn in_nsolve() -> bool {
  NSOLVE_DEPTH.with(std::cell::Cell::get) > 0
}

thread_local! {
  static NUMERIC_ROOTS_DEPTH: std::cell::Cell<usize> =
    const { std::cell::Cell::new(0) };
}

/// While alive, the polynomial system solver finds the roots of its
/// eliminants as numbers: the caller numericizes the solutions anyway (an
/// inexact system solved exactly), and radicals of rationalized machine
/// numbers are costly to build and simplify.
struct NumericRootsScope;

impl NumericRootsScope {
  fn new() -> Self {
    NUMERIC_ROOTS_DEPTH.with(|d| d.set(d.get() + 1));
    Self
  }
}

impl Drop for NumericRootsScope {
  fn drop(&mut self) {
    NUMERIC_ROOTS_DEPTH.with(|d| d.set(d.get() - 1));
  }
}

fn numeric_roots_wanted() -> bool {
  in_nsolve() || NUMERIC_ROOTS_DEPTH.with(std::cell::Cell::get) > 0
}

/// Solve with the explicit variable list narrowed first when the system is
/// underdetermined and nonlinear (see `select_underdetermined_vars`); a
/// linear one goes to the linear solver, which picks its own pivots.
fn solve_with_var_selection(
  positional: &[Expr],
  modulus: Option<i128>,
  original: &[Expr],
) -> Result<Expr, InterpreterError> {
  let over_integers = matches!(
    positional.get(2),
    Some(Expr::Identifier(d)) if d == "Integers"
  );
  // Over the integers, a system of linear equations has a lattice of
  // solutions, parametrized the way Reduce does it.
  if over_integers
    && modulus.is_none()
    && let Some(vars) = match &positional[1] {
      Expr::Identifier(name) => Some(vec![name.clone()]),
      Expr::List(items) => items
        .iter()
        .map(|item| match item {
          Expr::Identifier(name) => Some(name.clone()),
          _ => None,
        })
        .collect::<Option<Vec<_>>>(),
      _ => None,
    }
    && let Some(solutions) =
      super::reduce::solve_linear_diophantine_rules(&positional[0], &vars)?
  {
    return Ok(solutions);
  }
  // The real parametrization of an underdetermined system is no integer
  // solution.
  if modulus.is_none()
    && !over_integers
    && !in_nsolve()
    && positional.len() >= 2
    && let Expr::List(var_exprs) = &positional[1]
    && var_exprs.len() >= 2
    && let Some(vars) = var_exprs
      .iter()
      .map(|v| match v {
        Expr::Identifier(n) => Some(n.clone()),
        _ => None,
      })
      .collect::<Option<Vec<String>>>()
  {
    let eqs = solve_equalities(&positional[0]);
    let constraints =
      flatten_and_constraints(std::slice::from_ref(&positional[0]));
    // Only a system of plain equations: inequalities narrow the solution
    // set differently.
    if !eqs.is_empty()
      && eqs.len() < vars.len()
      && eqs.len() == constraints.len()
    {
      if let Some(result) = solve_linear_symbolic(&eqs, &vars) {
        return Ok(result);
      }
      let chosen = select_underdetermined_vars(&eqs, &vars);
      let mut narrowed = positional.to_vec();
      narrowed[1] =
        Expr::List(chosen.into_iter().map(Expr::Identifier).collect());
      return solve_core(&narrowed);
    }
  }
  match modulus {
    Some(n) => solve_modular(positional, n, original),
    None => solve_core(positional),
  }
}

/// Solve[eqns, vars, Modulus -> n] — delegate to the modular `Reduce` and turn
/// its conjunction/disjunction of equalities into Solve's list of rules.
fn solve_modular(
  positional: &[Expr],
  n: i128,
  original: &[Expr],
) -> Result<Expr, InterpreterError> {
  if positional.len() < 2 {
    return Ok(unevaluated("Solve", original));
  }
  // The variables, in the order they were given, so each solution lists its
  // rules in that order.
  let vars: Vec<String> = match &positional[1] {
    Expr::List(items) => items
      .iter()
      .filter_map(|v| match v {
        Expr::Identifier(name) => Some(name.clone()),
        _ => None,
      })
      .collect(),
    Expr::Identifier(name) => vec![name.clone()],
    _ => return Ok(unevaluated("Solve", original)),
  };
  if vars.is_empty() {
    return Ok(unevaluated("Solve", original));
  }
  let reduced = crate::evaluator::evaluate_expr_to_expr(&call(
    "Reduce",
    vec![
      positional[0].clone(),
      positional[1].clone(),
      rule_expr(id_expr("Modulus"), Expr::Integer(n)),
    ],
  ))?;
  // `False` means no solutions; `True` means every value works, which Reduce
  // does not report for a modular system, so anything else unrecognised leaves
  // the call alone rather than guessing.
  match &reduced {
    Expr::Identifier(s) if s == "False" => {
      return Ok(Expr::List(vec![].into()));
    }
    _ => {}
  }
  let Some(branches) = modular_solution_branches(&reduced, &vars) else {
    return Ok(unevaluated("Solve", original));
  };
  // Every value of a modular solution has to be a residue. `Reduce` still
  // ignores `Modulus` for a multivariate *nonlinear* system and answers over the
  // rationals, so refuse rather than dress that up as a modular solution.
  if branches.iter().any(|assignment| {
    assignment.iter().any(
      |(_, value)| !matches!(value, Expr::Integer(k) if (0..n).contains(k)),
    )
  }) {
    return Ok(unevaluated("Solve", original));
  }
  Ok(Expr::List(
    branches
      .into_iter()
      .map(|assignment| {
        Expr::List(
          vars
            .iter()
            .filter_map(|v| {
              assignment.iter().find(|(name, _)| name == v).map(
                |(name, value)| {
                  rule_expr(Expr::Identifier(name.clone()), value.clone())
                },
              )
            })
            .collect(),
        )
      })
      .collect(),
  ))
}

/// Split a modular `Reduce` result into one `(variable, value)` assignment list
/// per solution. `Or` separates solutions and `And` gathers the variables of
/// one; a bare `var == value` is a single one-variable solution. Returns `None`
/// for any other shape.
fn modular_solution_branches(
  reduced: &Expr,
  vars: &[String],
) -> Option<Vec<Vec<(String, Expr)>>> {
  // Flatten a nested Or/And of the given head into its leaves.
  fn parts(e: &Expr, head: &str) -> Vec<Expr> {
    match e {
      Expr::FunctionCall { name, args } if name == head => {
        args.iter().flat_map(|a| parts(a, head)).collect()
      }
      Expr::BinaryOp { op, left, right } if format!("{op:?}") == head => {
        let mut out = parts(left, head);
        out.extend(parts(right, head));
        out
      }
      other => vec![other.clone()],
    }
  }
  let equality = |e: &Expr| -> Option<(String, Expr)> {
    let (lhs, rhs) = match e {
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2
        && operators.len() == 1
        && operators[0] == ComparisonOp::Equal =>
      {
        (&operands[0], &operands[1])
      }
      Expr::FunctionCall { name, args }
        if name == "Equal" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    match lhs {
      Expr::Identifier(name) if vars.contains(name) => {
        Some((name.clone(), rhs.clone()))
      }
      _ => None,
    }
  };
  let mut branches = Vec::new();
  for branch in parts(reduced, "Or") {
    let mut assignment = Vec::new();
    for conjunct in parts(&branch, "And") {
      assignment.push(equality(&conjunct)?);
    }
    if assignment.is_empty() {
      return None;
    }
    branches.push(assignment);
  }
  Some(branches)
}

/// Thread an equality with at least one list operand into the element-wise
/// scalar equations (Wolfram's automatic listability of `Equal` inside
/// Solve): `{a, b} == {c, d}` → `[a == c, b == d]`. Scalar operands are
/// broadcast across the list, so `{a, b} == 0` → `[a == 0, b == 0]` — the
/// shape `Solve[N[Table[…] == 0, 10]]` produces. All list operands must
/// have the same length. Returns `None` when no operand is a list.
pub(super) fn thread_list_equation(eq: &Expr) -> Option<Vec<Expr>> {
  let operands: Vec<&Expr> = match eq {
    Expr::Comparison {
      operands,
      operators,
    } if !operands.is_empty()
      && operators.iter().all(|o| matches!(o, ComparisonOp::Equal)) =>
    {
      operands.iter().collect()
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() >= 2 => {
      args.iter().collect()
    }
    _ => return None,
  };
  // Length of the list operands; scalars broadcast to it.
  let mut len: Option<usize> = None;
  for op in &operands {
    if let Expr::List(items) = op {
      match len {
        None => len = Some(items.len()),
        Some(l) if l == items.len() => {}
        _ => return None,
      }
    }
  }
  let n = len?;
  if n == 0 {
    return None;
  }
  Some(
    (0..n)
      .map(|i| Expr::Comparison {
        operands: operands
          .iter()
          .map(|op| match op {
            Expr::List(items) => items[i].clone(),
            scalar => (*scalar).clone(),
          })
          .collect(),
        operators: vec![ComparisonOp::Equal; operands.len() - 1],
      })
      .collect(),
  )
}

/// A system whose equations have the unknowns in denominators, solved
/// through the numerators (see the call site). `None` when no equation has
/// such a denominator.
fn solve_rational_system(
  eqs: &[Expr],
  vars_exprs: &[Expr],
  var_names: &[String],
  rest_args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  use super::reduce::{CompOp, extract_comparison};
  let mut numerators = Vec::with_capacity(eqs.len());
  let mut denominators = Vec::new();
  for eq in eqs {
    let Some((lhs, rhs, CompOp::Equal)) = extract_comparison(eq) else {
      return Ok(None);
    };
    let diff = crate::evaluator::evaluate_expr_to_expr(&call(
      "Together",
      vec![call("Subtract", vec![lhs, rhs])],
    ))?;
    let (num, den) = super::together::extract_num_den(&diff);
    if var_names
      .iter()
      .any(|v| crate::functions::polynomial_ast::contains_var(&den, v))
    {
      denominators.push(den);
    }
    numerators.push(Expr::Comparison {
      operands: vec![num, Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    });
  }
  if denominators.is_empty() {
    return Ok(None);
  }
  let mut inner = vec![
    Expr::List(numerators.into()),
    Expr::List(vars_exprs.to_vec().into()),
  ];
  inner.extend(rest_args.iter().cloned());
  let solved = solve_ast(&inner)?;
  let Expr::List(solutions) = &solved else {
    return Ok(None);
  };
  let kept: Vec<Expr> = solutions
    .iter()
    .filter(|sol| {
      denominators.iter().all(|den| {
        let at = crate::evaluator::evaluate_expr_to_expr(&call(
          "ReplaceAll",
          vec![den.clone(), (*sol).clone()],
        ));
        // A vanishing denominator evaluates to an exact (or machine) zero;
        // a radical solution is not tested through machine numbers, whose
        // huge intermediate integers may not survive the conversion.
        !matches!(at, Ok(Expr::Integer(0)))
          && !matches!(at, Ok(Expr::Real(r)) if r == 0.0)
      })
    })
    .cloned()
    .collect();
  Ok(Some(Expr::List(kept.into())))
}

fn solve_core(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let _constrained = SuppressIfun::new(
    args.first().is_some_and(has_inequality)
      // The `Reals` domain is the third argument in the ordinary
      // `Solve[eqns, vars, Reals]` form, but the domain-only two-argument
      // form `Solve[eqns, Reals]` carries it as the last argument instead
      // — check `args.last()` so both are recognized alike.
      || (args.len() >= 2
        && matches!(args.last(), Some(Expr::Identifier(d)) if d == "Reals")),
  );

  // Pre-pass: thread equalities over equal-length lists, as Wolfram does
  // before solving. A vector equation like `r1 + t e1 == r2 + u e2` (both
  // sides 2-element lists) stands for the element-wise scalar equations, so
  // `Solve[{a1, b1} == {a2, b2}, {t, u}]` solves `a1 == a2 && b1 == b2`.
  // Runs before the one-argument form so `Solve[{x, y} == {1, 2}]` sees two
  // equations when auto-detecting its variables.
  let threaded_args_owned: Vec<Expr>;
  let args = {
    let threaded_first = match &args[0] {
      Expr::List(items) => {
        let mut changed = false;
        let mut out: Vec<Expr> = Vec::new();
        for e in items {
          match thread_list_equation(e) {
            Some(eqs) => {
              changed = true;
              out.extend(eqs);
            }
            None => out.push(e.clone()),
          }
        }
        changed.then(|| Expr::List(out.into()))
      }
      other => thread_list_equation(other).map(|eqs| Expr::List(eqs.into())),
    };
    match threaded_first {
      Some(first) => {
        let mut new_args = args.to_vec();
        new_args[0] = first;
        threaded_args_owned = new_args;
        threaded_args_owned.as_slice()
      }
      None => args,
    }
  };

  // A one-element equation list stands for the bare equation. Unwrapping it
  // keeps the bare form's root multiplicity: `Solve[{(x - 2)^2 == 0}, x]`
  // gives `{{x -> 2}, {x -> 2}}` just like `Solve[(x - 2)^2 == 0, x]`.
  let unwrapped_args_owned: Vec<Expr>;
  let args = match &args[0] {
    Expr::List(items)
      if items.len() == 1
        && matches!(
          &items[0],
          Expr::Comparison { .. } | Expr::FunctionCall { .. }
        )
        && !matches!(&items[0], Expr::FunctionCall { name, .. } if name == "List") =>
    {
      let mut new_args = args.to_vec();
      new_args[0] = items[0].clone();
      unwrapped_args_owned = new_args;
      unwrapped_args_owned.as_slice()
    }
    _ => args,
  };

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
    // In a list, a `True` constrains nothing and a `False` rules out
    // everything: `Solve[{True}]` is `{{}}`, `Solve[{x == 1, True}]` solves
    // `x == 1`.
    if let Expr::List(items) = &args[0] {
      let is = |e: &Expr, v: &str| matches!(e, Expr::Identifier(s) if s == v);
      if items.iter().any(|e| is(e, "False")) {
        return Ok(Expr::List(vec![].into()));
      }
      if items.iter().any(|e| is(e, "True")) {
        let kept: Vec<Expr> =
          items.iter().filter(|e| !is(e, "True")).cloned().collect();
        if kept.is_empty() {
          return Ok(Expr::List(vec![Expr::List(vec![].into())].into()));
        }
        return solve_ast(&[Expr::List(kept.into())]);
      }
    }
    return match auto_detect_solve_vars(&args[0]) {
      Some(va) => solve_ast(&[args[0].clone(), va]),
      None => Ok(unevaluated("Solve", args)),
    };
  }
  // Two-argument form Solve[eqns, dom] (a domain, not a variable list —
  // the Demonstrations idiom that restricts a system to real solutions
  // without spelling out which symbols are the unknowns, e.g.
  // `NSolve[{eqn1, eqn2, 0 < x < 1, 0 < y < 1}, Reals]`): auto-detect the
  // variables exactly as the one-argument form does, then delegate to the
  // three-argument form with the given domain. A domain name is a
  // protected symbol in Wolfram and can never be an ordinary equation
  // variable, so this never misfires against a genuine `Solve[eqns, x]`.
  if args.len() == 2
    && let Expr::Identifier(dom) = &args[1]
    && matches!(
      dom.as_str(),
      "Reals"
        | "Integers"
        | "Complexes"
        | "Rationals"
        | "Algebraics"
        | "Booleans"
    )
  {
    // Mirrors the one-argument form: an ambiguous (underdetermined) system
    // is left unevaluated rather than falling through into the generic
    // two-argument path below, which would otherwise treat the domain
    // name itself as the solve variable.
    return match auto_detect_solve_vars(&args[0]) {
      Some(va) => solve_ast(&[args[0].clone(), va, args[1].clone()]),
      None => Ok(unevaluated("Solve", args)),
    };
  }
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Solve expects 2 or 3 arguments".into(),
    ));
  }

  // Pre-pass: generalized variables (`Solve[eqns, f[x]]`).
  if let Some(result) = with_generalized_variables(args, solve_ast) {
    return result;
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
        for a in aargs {
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

  // Pre-pass: thread equations whose two sides are equal-length lists into
  // their component equations ({xx, yy} == {a, b} → xx == a, yy == b,
  // recursively), matching Wolfram. Vector equations like
  // `{xx, yy} == t*v + pos` (parametric-intersection systems) otherwise
  // carry no solvable content for the scalar equation paths below.
  fn thread_list_equations(eq: &Expr, out: &mut Vec<Expr>) -> bool {
    let sides: Option<(&Expr, &Expr)> = match eq {
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2
        && operators.len() == 1
        && operators[0] == ComparisonOp::Equal =>
      {
        Some((&operands[0], &operands[1]))
      }
      Expr::FunctionCall { name, args }
        if name == "Equal" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      _ => None,
    };
    if let Some((Expr::List(l), Expr::List(r))) = sides
      && l.len() == r.len()
    {
      for (li, ri) in l.iter().zip(r.iter()) {
        thread_list_equations(
          &Expr::Comparison {
            operands: vec![li.clone(), ri.clone()],
            operators: vec![ComparisonOp::Equal],
          },
          out,
        );
      }
      return true;
    }
    out.push(eq.clone());
    false
  }
  let threaded_args_owned: Vec<Expr>;
  let args = {
    let eqns: Vec<Expr> = match &args[0] {
      Expr::List(items) => items.to_vec(),
      other => vec![other.clone()],
    };
    let mut threaded = Vec::new();
    let mut any_threaded = false;
    for eq in &eqns {
      any_threaded |= thread_list_equations(eq, &mut threaded);
    }
    if any_threaded {
      let mut new_args = args.to_vec();
      new_args[0] = Expr::List(threaded.into());
      threaded_args_owned = new_args;
      threaded_args_owned.as_slice()
    } else {
      args
    }
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
  // equation, and continue with the remaining variables. Without this, an extra var like `y` in
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
  let single_eq;
  let eqs_view: Option<&[Expr]> = match &args[0] {
    Expr::List(eqs) => Some(eqs.as_slice()),
    eq @ Expr::Comparison { .. } => {
      single_eq = [eq.clone()];
      Some(&single_eq[..])
    }
    _ => None,
  };
  let args = if let (Some(eqs), Expr::List(vars)) = (eqs_view, &args[1]) {
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
      // The svars message is the outermost call's to report, once it knows
      // a solution leaves a requested variable free (an inconsistent
      // system has none to report).
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

  // A domain does not change a tautology or contradiction. Handle these
  // before domain-specific routing so Solve[True, x, Integers] agrees with the
  // already-supported two-argument form instead of being treated as an
  // unbounded affine search.
  if domain.is_some()
    && let Expr::Identifier(truth) = &args[0]
  {
    if truth == "True" {
      return Ok(Expr::List(vec![Expr::List(vec![].into())].into()));
    }
    if truth == "False" {
      return Ok(Expr::List(Vec::new().into()));
    }
  }

  // Integer affine formulas need a completeness-preserving route before the
  // ordinary algebraic solver. Filtering a partial/parametric algebraic answer
  // can turn an unsupported integer problem into a false empty list.
  if matches!(domain.as_deref(), Some("Integers")) {
    use super::reduce_backend::FiniteIntegerSolve;

    match super::reduce_backend::solve_finite_integer(&args[0], &args[1]) {
      FiniteIntegerSolve::Solved(solutions) => {
        let variables: Vec<String> = match &args[1] {
          Expr::Identifier(name) => vec![name.clone()],
          Expr::List(items) => items
            .iter()
            .filter_map(|item| match item {
              Expr::Identifier(name) => Some(name.clone()),
              _ => None,
            })
            .collect(),
          _ => Vec::new(),
        };
        return Ok(Expr::List(
          solutions
            .into_iter()
            .map(|values| {
              Expr::List(
                variables
                  .iter()
                  .zip(values)
                  .map(|(variable, value)| {
                    rule_expr(
                      Expr::Identifier(variable.clone()),
                      super::reduce_backend::integer_expr(value),
                    )
                  })
                  .collect(),
              )
            })
            .collect(),
        ));
      }
      FiniteIntegerSolve::Infeasible => {
        return Ok(Expr::List(Vec::new().into()));
      }
      FiniteIntegerSolve::Unsupported => {
        return Ok(unevaluated("Solve", args));
      }
      FiniteIntegerSolve::NotApplicable => {}
    }
  }

  // If another domain is specified, solve without domain first, then filter.
  // Non-affine integer equations also use this established algebraic path.
  if let Some(ref dom) = domain {
    // Over the reals the solutions found are all there are: wolframscript
    // reports no use of inverse functions.
    let base_solutions = {
      let _quiet = SuppressIfun::new(dom == "Reals");
      let outer = IFUN_SWALLOWED.with(std::cell::Cell::get);
      let solved = solve_ast(&args[..2]);
      IFUN_SWALLOWED.with(|w| w.set(outer));
      solved?
    };
    if matches!(&base_solutions, Expr::FunctionCall { name, .. } if name == "Solve")
    {
      return Ok(unevaluated("Solve", args));
    }
    if dom == "Reals" {
      // Filter out complex solutions
      if let Expr::List(solutions) = &base_solutions {
        let filtered: Vec<Expr> = solutions
          .iter()
          .map(real_member_of_imaginary_family)
          .filter(|sol| {
            !contains_complex(sol)
              && !has_nonreal_root_value(sol)
              && !has_numerically_nonreal_value(sol)
          })
          .collect();
        return Ok(Expr::List(filtered.into()));
      }
    }
    if dom == "Integers" {
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
    if dom == "Rationals" {
      // Keep only solutions whose values are all exact rationals; irrational
      // algebraic values like Sqrt[2] are excluded.
      if let Expr::List(solutions) = &base_solutions {
        let filtered: Vec<Expr> = solutions
          .iter()
          .filter(|sol| {
            if let Expr::List(rules) = sol {
              rules.iter().all(|rule| {
                if let Expr::Rule { replacement, .. } = rule {
                  is_rational_expr(replacement)
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
      // Equations with the unknowns in denominators: solve the numerators
      // of `lhs - rhs` brought together as a polynomial system, then drop
      // the solutions that make a denominator vanish. `-19/a^5 + 31/a^3 -
      // 8/(a^3 b^2) == 0` is the polynomial `-19 b^2 + 31 a^2 b^2 - 8 a^2`
      // over `a^5 b^2`, so a == 0 is no solution.
      if let Some(result) =
        solve_rational_system(eqs, vars_exprs, &var_names, &args[2..])?
      {
        return Ok(result);
      }
      // Try symbolic Gaussian elimination for linear systems (handles underdetermined case)
      if let Some(result) = solve_linear_symbolic(eqs, &var_names) {
        return Ok(result);
      }
      // A mostly linear system with a few nonlinear terms: unknowns that an
      // equation fixes linearly with a numeric coefficient are substituted
      // away first, leaving a small nonlinear core.
      if let Some(result) = solve_after_linear_elimination(eqs, &var_names) {
        return Ok(result);
      }
      // Nonlinear polynomial systems are eliminated with resultants, which
      // keeps every intermediate a polynomial.
      if let Some(result) = try_solve_polynomial_system(eqs, &var_names) {
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
        // An equation that just names an unknown's value (`u == x`, the
        // other side free of the unknowns) is substituted into the rest,
        // which is solved on its own.
        if let Some(result) = solve_with_defined_unknown(eqs, &var_names)? {
          return Ok(result);
        }
        return Ok(unevaluated("Solve", args));
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
      // The multi-variable elimination does not reach every constraint system
      // — a transcendental or `Abs` equation narrowed by an inequality comes
      // back as an unreduced `Reduce[...]`, which `ToRules` then leaves
      // wrapped. The conjunction spelling of the same system goes down the
      // single-expression path, which does handle those, so retry there
      // rather than hand back a `ToRules[Reduce[...]]` nobody asked for.
      //
      // But when every constraint is a plain equality, solve_ast's own
      // And-to-List pre-pass (above) immediately folds that conjunction
      // back into the identical `{eqs}, {vars}` pair this branch just
      // failed on, landing right back here with the same unreduced
      // `Reduce[...]` and recursing forever. Only retry when at least one
      // constraint isn't a plain equality, so the retry actually reaches a
      // different code path instead of looping back to this one.
      if matches!(&rules, Expr::FunctionCall { name, .. } if name == "ToRules")
        && !all_equalities(&constraints)
      {
        let conjunction = constraints
          .iter()
          .cloned()
          .reduce(|left, right| binop(BinaryOperator::And, left, right));
        if let Some(conjunction) = conjunction {
          let mut retry = args.to_vec();
          retry[0] = conjunction;
          return solve_ast(&retry);
        }
      }
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
        solutions.sort_by_key(|sol| i32::from(contains_complex(sol)));
      }
      return Ok(wrapped);
    }
  }

  // Handle a system of equations combined with inequality constraints when
  // solving for a list of variables: Solve[eq1 && eq2 && ... && ineq, {v1,
  // v2, ...}]. The pre-pass above only flattens an And into a List of
  // equations when every conjunct is an equality, so a mixed system with two
  // or more equalities plus an inequality is still a single And expression
  // here, and would otherwise be treated as one non-list "equation" by the
  // block below. Split it into its equalities and inequalities, solve the
  // equalities as a system, then discard any solution an inequality rules
  // out.
  if let Expr::List(var_items) = &args[1]
    && !var_items.is_empty()
  {
    let mut constraints = Vec::new();
    collect_and_constraints(&args[0], &mut constraints);
    let is_equality = |e: &Expr| -> bool {
      matches!(e, Expr::Comparison { operators, .. }
          if operators.len() == 1 && operators[0] == ComparisonOp::Equal)
        || matches!(e, Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2)
    };
    let (eqs, ineqs): (Vec<Expr>, Vec<Expr>) =
      constraints.into_iter().partition(|e| is_equality(e));
    if eqs.len() >= 2 && !ineqs.is_empty() {
      let eq_solutions = solve_ast(&[Expr::List(eqs.into()), args[1].clone()])?;
      return Ok(match &eq_solutions {
        Expr::List(solutions) => Expr::List(
          solutions
            .iter()
            .filter(|sol| {
              let Expr::List(rules) = sol else {
                return true;
              };
              !ineqs.iter().any(|ineq| {
                let substituted =
                  rules.iter().fold(ineq.clone(), |acc, rule| {
                    let Expr::Rule {
                      pattern,
                      replacement,
                    } = rule
                    else {
                      return acc;
                    };
                    let Expr::Identifier(name) = pattern.as_ref() else {
                      return acc;
                    };
                    crate::syntax::substitute_variable(&acc, name, replacement)
                  });
                matches!(
                  crate::evaluator::evaluate_expr_to_expr(&substituted),
                  Ok(Expr::Identifier(ref s)) if s == "False"
                )
              })
            })
            .cloned()
            .collect(),
        ),
        _ => eq_solutions,
      });
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
      let poly = minus2(lhs, rhs);
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
      return Ok(unevaluated("Solve", args));
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
        for sol in solutions {
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
              let key = expr_to_string(&c);
              if seen.insert(key) {
                out.push(c);
              }
            }
            continue;
          }
          // Otherwise keep the solution unless an inequality is definitely
          // violated.
          let ineq_false = |ineq: &Expr, replacement: &Expr| -> bool {
            let decide = |value: &Expr| -> Option<bool> {
              let subst =
                crate::syntax::substitute_variable(ineq, &var_name, value);
              match crate::evaluator::evaluate_expr_to_expr(&subst) {
                Ok(Expr::Identifier(ref s)) if s == "False" => Some(true),
                Ok(Expr::Identifier(ref s)) if s == "True" => Some(false),
                _ => None,
              }
            };
            if let Some(verdict) = decide(replacement) {
              return verdict;
            }
            // An exact root has no ordering the comparison operators can
            // settle symbolically (`0 <= Root[…] <= 1` stays as written), so
            // the bound is decided on the root's numeric value — which is
            // what wolframscript reports for a constrained polynomial.
            let numeric = crate::evaluator::evaluate_expr_to_expr(&call1(
              "N",
              replacement.clone(),
            ))
            .unwrap_or_else(|_| replacement.clone());
            if let Some(verdict) = decide(&numeric) {
              return verdict;
            }
            // Nothing orders a complex number, so an ordering bound rules
            // one out: `Solve[x^4 == 16 && x > 0, x]` keeps only the 2.
            let is_ordering = matches!(ineq, Expr::Comparison { operators, .. }
              if operators.iter().any(|o| matches!(o,
                ComparisonOp::Less
                  | ComparisonOp::LessEqual
                  | ComparisonOp::Greater
                  | ComparisonOp::GreaterEqual)));
            is_ordering
              && try_extract_complex_float(&numeric)
                .is_some_and(|(_re, im)| im.abs() > 1e-10)
          };
          let violated = matches!(sol, Expr::List(rules) if rules.iter().any(|rule| {
            matches!(rule, Expr::Rule { replacement, .. }
              if ineqs.iter().any(|ineq| ineq_false(ineq, replacement)))
          }));
          if !violated {
            let key = expr_to_string(sol);
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
              try_eval_to_f64(replacement).unwrap_or(f64::INFINITY)
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
        "Solve::ivar: {name} is not a valid variable."
      ));
      return Ok(unevaluated("Solve", args));
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
        let target_str = expr_to_string(target_expr);
        let lhs_str = expr_to_string(&lhs);
        let rhs_str = expr_to_string(&rhs);
        if lhs_str == target_str {
          return Ok(Expr::List(
            vec![Expr::List(vec![rule_expr(target_expr.clone(), rhs)].into())]
              .into(),
          ));
        }
        if rhs_str == target_str {
          return Ok(Expr::List(
            vec![Expr::List(vec![rule_expr(target_expr.clone(), lhs)].into())]
              .into(),
          ));
        }
      }
      return Ok(unevaluated("Solve", args));
    }
  };

  // Check if variable has Constant attribute (user-defined constants)
  use crate::evaluator::Attributes as A;
  let is_constant = crate::func_attrs_contains(var, A::Constant);
  if is_constant {
    crate::emit_message(&format!(
      "Solve::ivar: {var} is not a valid variable."
    ));
    return Ok(unevaluated("Solve", args));
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
      minus2(operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      minus2(fargs[0].clone(), fargs[1].clone())
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
      let expr_str = expr_to_string(&args[0]);
      crate::emit_message(&format!(
        "Solve::naqs: {expr_str} is not a quantified system of equations and inequalities."
      ));
      return Ok(unevaluated("Solve", args));
    }
  };

  // Absolute-value equations: Abs[f(x)] == c → f == c ∪ f == -c.
  if let Some(result) = try_solve_abs_eq(&args[0], var) {
    return result;
  }

  // `b^g(x) * h(x) == 0` with a positive constant base: the exponential never
  // vanishes, so only `h(x) == 0` contributes roots.
  if let Some(result) = try_solve_peel_exponential_factors(&args[0], var) {
    return result;
  }

  // `x^p E^(b x + c) == k` and `β E^(b x + c) + α x + d == 0` are solved
  // with ProductLog.
  if let Some(result) = try_solve_lambert(&poly, var)? {
    return Ok(result);
  }

  // Try to solve equations with invertible functions:
  // Log[expr] == a → expr == E^a, Sqrt[expr] == a → expr == a^2, etc.
  if let Some(result) = try_solve_inverse_function(&args[0], var) {
    return result;
  }
  if let Some(result) = try_solve_through_undefined_function(&args[0], var) {
    return result;
  }

  // Trigonometric equations of the form `Trig[var] == const` where the trig
  // head is Sin/Cos/Tan/Cot. Wolframscript prints the periodic solution set
  // as `{{var -> ConditionalExpression[base + 2*Pi*C[1], Element[C[1],
  // Integers]]}, …}` for Sin/Cos and `{var -> ConditionalExpression[base +
  // Pi*C[1], Element[C[1], Integers]]}` for Tan/Cot.
  if let Some(result) = try_solve_inexact_trig(&args[0], var) {
    return Ok(result);
  }
  if let Some(result) = try_solve_trig_eq(&args[0], var) {
    return Ok(result);
  }

  // Equations carrying a root of the unknown: raise them away and keep the
  // roots that survive the original equation.
  // Roots nested in roots are eliminated through norms; isolating one root
  // and squaring only shuffles them around.
  if has_nested_square_radical(&poly, var)
    && let Some(result) = try_solve_by_radical_norms(&poly, &args[0], var)
  {
    return result;
  }
  if let Some(result) = try_solve_radical_equation(&poly, &args[0], var) {
    return result;
  }
  if let Some(result) = try_solve_by_radical_norms(&poly, &args[0], var) {
    return result;
  }

  // Expand and collect polynomial coefficients
  // Clear denominators: f(x)/g(x) == 0 ↔ f(x) == 0
  let expanded_raw = expand_and_combine(&poly);
  let expanded = {
    // Cancel what `Together` leaves common to numerator and denominator (it
    // does not always reduce a multivariate fraction), so the numerator holds
    // no factor of the denominator and no pole is reported as a root.
    let together = together_expr(&expanded_raw);
    // A denominator free of the unknown never hides a pole: keep the plain
    // numerator.
    let plain_numerator = match &together {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: numerator,
        right: denominator,
      } if !super::eliminate::contains_var(denominator, var) => {
        Some(expand_and_combine(numerator))
      }
      _ => None,
    };
    if let Some(numerator) = plain_numerator {
      numerator
    } else {
      let together = super::cancel::cancel_expr_keep_quotient_sign(&together);
      let (numerator, denominator) =
        super::together::extract_num_den(&together);
      if super::eliminate::contains_var(&denominator, var) {
        expand_and_combine(&numerator)
      } else {
        expanded_raw
      }
    }
  };
  // Factor out constant factors (w.r.t. the solve variable) so the
  // quadratic formula sees integer leading coefficients when possible.
  // E.g. 2*a^2*k*q - 4*k*q*x^2  →  a^2 - 2*x^2
  let expanded = factor_out_constant_factors(&expanded, var);
  let terms = collect_additive_terms(&expanded);

  // Catch `atom == const` equations hiding inside a linear combination
  // (e.g. `b^f(var)/k == c` or `Log[var] + k == c`) before degree
  // extraction below mistakes them for polynomials — see
  // `try_solve_isolated_invertible_atom`.
  if let Some(result) = try_solve_isolated_invertible_atom(&expanded, var) {
    return result;
  }

  // Find maximum degree
  let Some(degree) = max_power_int(&expanded, var) else {
    // Non-polynomial: try factoring out common fractional-power
    // sub-expressions, then a polynomial in an exponential (`b^x` has no
    // degree in x).
    if let Some(result) = try_solve_factoring_powers(&expanded, var, args) {
      return result;
    }
    if let Some(result) = try_solve_exponential_polynomial(&expanded, var) {
      return result;
    }
    return Ok(unevaluated("Solve", args));
  };

  // A negative maximum power means a Laurent/rational expression (e.g. only
  // x^-2 terms). The coefficient extraction below builds an empty `0..=degree`
  // range, and the later `degree as usize` cast would index out of bounds, so
  // bail out with the unevaluated Solve.
  if degree < 0 {
    return Ok(unevaluated("Solve", args));
  }
  // Degree 0 with the variable still present (only inside exponents or
  // other functions) is no polynomial; reading it as the constant 0 claimed
  // every value a solution (`Solve[a b^t - c d^t == 0, t]` gave {{}}).
  if degree == 0 && !is_constant_wrt(&expanded, var) {
    if let Some(result) = try_solve_factoring_powers(&expanded, var, args) {
      return result;
    }
    if let Some(result) = try_solve_exponential_polynomial(&expanded, var) {
      return result;
    }
    return Ok(unevaluated("Solve", args));
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
        result = plus2(result, c);
      }
      coeffs.push(simplify(result));
    }
  }

  // A coefficient that still holds the variable (`x 2^x` read as degree 1
  // with coefficient 2^x) means this is no polynomial in it.
  if coeffs.iter().any(|c| !is_constant_wrt(c, var)) {
    if let Some(result) = try_solve_factoring_powers(&expanded, var, args) {
      return result;
    }
    return Ok(unevaluated("Solve", args));
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
    Expr::List(vec![rule_expr(id_expr(var), solution)].into())
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
        // `x = 0` is a root of multiplicity `zero_count`, and Solve reports
        // every root with its multiplicity: `x^3 - 4 x^2 == 0` has the
        // solutions {0, 0, 4}. An inexact coefficient anywhere makes the
        // whole solution set inexact, so the zero is `0.` there.
        let zero = if coeffs
          .iter()
          .any(crate::functions::predicate_ast::contains_real_literal)
        {
          Expr::Real(0.0)
        } else {
          Expr::Integer(0)
        };
        let mut all_solutions: Vec<Expr> =
          (0..zero_count).map(|_| make_rule(zero.clone())).collect();
        all_solutions.extend(reduced_sols.iter().cloned());
        sort_solutions(&mut all_solutions);
        return Ok(Expr::List(all_solutions.into()));
      }
    }
  }

  // A polynomial with symbolic coefficients that factors over the rationals
  // is solved factor by factor, each root as often as its factor occurs:
  // `(x - b)(x - a) == 0` gives `x -> a` and `x -> b`, not the quadratic
  // formula's radicals of a perfect square.
  // (A pure power a x^n + c keeps its own roots-of-unity form below.)
  let pure_power =
    (1..degree as usize).all(|i| matches!(&coeffs[i], Expr::Integer(0)));
  if degree >= 2
    && !pure_power
    && coeffs.iter().any(|c| {
      !crate::functions::predicate_ast::is_numeric_q(c)
        && !matches!(c, Expr::Integer(_))
    })
    && let Some(result) =
      solve_by_factor_list(&build_poly_from_coeffs(&coeffs, var), var, args)?
  {
    return Ok(result);
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
      // Put the root over one denominator, as wolframscript does: forms
      // like -((1 - z)/2) become (-1 + z)/2 and a + 1/b becomes
      // (1 + a b)/b.
      let solution = crate::evaluator::evaluate_function_call_ast(
        "Together",
        std::slice::from_ref(&solution),
      )
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
      let discriminant = simplify(minus2(b_sq, four_ac));

      let neg_b = negate_expr(b);
      let two_a = multiply_exprs(&Expr::Integer(2), a);

      // For integer coefficients, use exact arithmetic with simplified Sqrt
      // (Coefficients too large for i128 arithmetic — e.g. a rationalized
      // machine number — take the general path below.)
      let int_disc = match (a, b, c) {
        (Expr::Integer(ai), Expr::Integer(bi), Expr::Integer(ci)) => bi
          .checked_mul(*bi)
          .zip(ai.checked_mul(*ci).and_then(|p| p.checked_mul(4)))
          .and_then(|(bb, fac)| bb.checked_sub(fac))
          .filter(|_| ai.checked_mul(2).is_some()),
        _ => None,
      };
      if let (
        Expr::Integer(ai),
        Expr::Integer(bi),
        Expr::Integer(ci),
        Some(disc_int),
      ) = (a, b, c, int_disc)
      {
        let ai = *ai;
        let bi = *bi;
        let ci = *ci;

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
          }
          // Irrational roots: (-bi ± sqrt_out*Sqrt[sqrt_in]) / (2*ai)
          // Simplify by dividing common factors
          let g = gcd_i128(gcd_i128(-bi, sqrt_out), 2 * ai);
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
              let rational_arg = make_rational(sqrt_in, den * den);
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
            // A content the two terms share stays a factor of the sum:
            // `3 (-1 - Sqrt[2])`, `(3 (-45 - Sqrt[2585]))/16`.
            let content = if nb == 0 { 1 } else { gcd_i128(nb.abs(), so) };
            if content > 1 {
              let radical = if so == content {
                make_sqrt(Expr::Integer(sqrt_in))
              } else {
                multiply_exprs(
                  &Expr::Integer(so / content),
                  &make_sqrt(Expr::Integer(sqrt_in)),
                )
              };
              let join_op = if sign_minus {
                BinaryOperator::Minus
              } else {
                BinaryOperator::Plus
              };
              let sum = binop(join_op, Expr::Integer(nb / content), radical);
              let root = times2(make_rational(content, den), sum);
              return crate::evaluator::evaluate_expr_to_expr(&root)
                .unwrap_or(root);
            }
            let num = if nb == 0 {
              if sign_minus {
                negate_expr(&sqrt_part)
              } else {
                sqrt_part.clone()
              }
            } else {
              let nb_expr = Expr::Integer(nb);
              let join_op = if sign_minus {
                BinaryOperator::Minus
              } else {
                BinaryOperator::Plus
              };
              binop(join_op, nb_expr, sqrt_part.clone())
            };
            let root = if den == 1 {
              num
            } else {
              div2(num, Expr::Integer(den))
            };
            // Evaluated, so the root is canonical (2 Sqrt[5]/5 is 2/Sqrt[5]).
            crate::evaluator::evaluate_expr_to_expr(&root).unwrap_or(root)
          };
          let sol1 = make_sol(true);
          let sol2 = make_sol(false);
          return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)].into()));
        }
        // Check for cyclotomic polynomials before using quadratic formula
        // x^2 + x + 1 = 0 (Φ₃): roots are (-1)^(2/3) and -(-1)^(1/3)
        // x^2 - x + 1 = 0 (Φ₆): roots are (-1)^(1/3) and -(-1)^(2/3)
        // Multiplying the polynomial through by -1 doesn't change the
        // root set, so accept `(ai, ci) ∈ {(1, 1), (-1, -1)}` and pick
        // the cyclotomic branch by `Sign[bi*ai]` rather than `bi` alone.
        let cyclo_match = (ai == 1 && ci == 1) || (ai == -1 && ci == -1);
        if cyclo_match && bi.abs() == ai.abs() {
          let make_neg1_pow = |p: i128, q: i128| -> Expr {
            pow(Expr::Integer(-1), make_rational(p, q))
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
          }
          // Φ₆: x^2 - x + 1 → roots: (-1)^(1/3), -(-1)^(2/3)
          let sol1 = make_neg1_pow(1, 3);
          let sol2 = negate_expr(&make_neg1_pow(2, 3));
          return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)].into()));
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
            let i_part = multiply_exprs(&id_expr("I"), &imag_part);
            let join_op = if sign_minus {
              BinaryOperator::Minus
            } else {
              BinaryOperator::Plus
            };
            simplify(binop(join_op, real_part.clone(), i_part))
          };
          let sol1 = make_sol(true);
          let sol2 = make_sol(false);
          return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)].into()));
        }
        // Complex roots with irrational imaginary part
        let g = gcd_i128(gcd_i128(-bi, sqrt_out), 2 * ai);
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
          &id_expr("I"),
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
          // A shared content stays a factor of the sum, as for real roots.
          let content = if nb == 0 { 1 } else { gcd_i128(nb.abs(), so) };
          if content > 1 {
            let radical = multiply_exprs(
              &id_expr("I"),
              &if so == content {
                make_sqrt(Expr::Integer(sqrt_in))
              } else {
                multiply_exprs(
                  &Expr::Integer(so / content),
                  &make_sqrt(Expr::Integer(sqrt_in)),
                )
              },
            );
            let join_op = if sign_minus {
              BinaryOperator::Minus
            } else {
              BinaryOperator::Plus
            };
            let sum = binop(join_op, Expr::Integer(nb / content), radical);
            return times2(make_rational(content, den), sum);
          }
          let num = if nb == 0 {
            if sign_minus {
              negate_expr(&sqrt_part)
            } else {
              sqrt_part.clone()
            }
          } else {
            let join_op = if sign_minus {
              BinaryOperator::Minus
            } else {
              BinaryOperator::Plus
            };
            binop(join_op, Expr::Integer(nb), sqrt_part.clone())
          };
          if den == 1 {
            num
          } else {
            div2(num, Expr::Integer(den))
          }
        };
        // Re-evaluate so a raw negation collapses (e.g. -(I*Sqrt[2]) →
        // -I*Sqrt[2]), matching wolframscript's complex-root form.
        let finish =
          |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e);
        let sol1 = finish(make_sol(true));
        let sol2 = finish(make_sol(false));
        return Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)].into()));
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
        let monomial_root =
          crate::evaluator::evaluate_expr_to_expr(&numer_under_sqrt)
            .ok()
            .and_then(|m| split_monomial_sqrt(&m));
        let sqrt_numer = if let Some(root) = monomial_root {
          root
        } else {
          let raw =
            crate::functions::sqrt_ast(std::slice::from_ref(&numer_under_sqrt))
              .unwrap_or_else(|_| make_sqrt(numer_under_sqrt));
          let evaled =
            crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw);
          // In Solve context, Sqrt[expr^2] → expr because ± handles sign
          let evaled = strip_sqrt_square(evaled);
          simplify(evaled)
        };
        let sqrt_denom =
          crate::functions::sqrt_ast(std::slice::from_ref(&denom_under_sqrt))
            .unwrap_or_else(|_| make_sqrt(denom_under_sqrt));
        let sol_pos = if matches!(&sqrt_denom, Expr::Integer(1)) {
          sqrt_numer.clone()
        } else {
          div2(sqrt_numer.clone(), sqrt_denom)
        };
        let sol_neg = negate_expr(&sol_pos);
        return Ok(Expr::List(
          vec![make_rule(sol_neg), make_rule(sol_pos)].into(),
        ));
      }

      // b = 0 with a symbolic leading coefficient: x^2 == -c/a, whose root
      // splits like any monomial's (`b x^2 == a` gives ±Sqrt[a]/Sqrt[b]).
      if matches!(b, Expr::Integer(0))
        && let Ok(value) = crate::evaluator::evaluate_expr_to_expr(
          &solve_divide(&negate_expr(c), a),
        )
        && let Some(root) = split_monomial_sqrt(&value)
      {
        let negative =
          crate::evaluator::evaluate_expr_to_expr(&negate_expr(&root))
            .unwrap_or_else(|_| negate_expr(&root));
        return Ok(Expr::List(
          vec![make_rule(negative), make_rule(root)].into(),
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
      let num1 = eval_expr(minus2(neg_b.clone(), sqrt_disc.clone()));
      let sol1 = eval_expr(solve_divide(&num1, &two_a));
      let num2 = eval_expr(plus2(neg_b, sqrt_disc));
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
        // Machine-precision and rational coefficients flip the order just
        // the same; a symbolic one has no sign to go by.
        Expr::Real(r) => *r < 0.0,
        _ => try_eval_to_f64(a).is_some_and(|v| v < 0.0),
      };
      if leading_negative {
        Ok(Expr::List(vec![make_rule(sol2), make_rule(sol1)].into()))
      } else {
        Ok(Expr::List(vec![make_rule(sol1), make_rule(sol2)].into()))
      }
    }
    _ => {
      // Pure power equation: a*x^n + c = 0 (all middle coefficients zero)
      // Solve as x = (-c/a)^(1/n) * root_of_unity for each nth root of unity.
      // Only for exact coefficients: `x^3 == 8.` is answered numerically by
      // wolframscript, not with the radical `-2 (-1)^(1/3)` forms.
      let is_pure_power = (1..degree as usize)
        .all(|i| matches!(&coeffs[i], Expr::Integer(0)))
        && !coeffs
          .iter()
          .any(crate::functions::predicate_ast::contains_real_literal);

      if is_pure_power {
        let c_coeff = &coeffs[0];
        let a_coeff = &coeffs[degree as usize];
        let neg_c = negate_expr(c_coeff);
        let val = simplify(solve_divide(&neg_c, a_coeff));
        let val = crate::evaluator::evaluate_expr_to_expr(&val).unwrap_or(val);

        // x^n == 0 has the single root 0, reported with its multiplicity.
        if matches!(&val, Expr::Integer(0)) {
          let zeros = (0..degree)
            .map(|_| make_rule(Expr::Integer(0)))
            .collect::<Vec<_>>();
          return Ok(Expr::List(zeros.into()));
        }

        // The nth-root approach also carries exact numeric values, which is
        // what wolframscript reports: Solve[x^3 == 8, x] keeps the complex
        // roots as `-2 (-1)^(1/3)` and `2 (-1)^(2/3)` rather than expanding
        // them, and x^3 == 2 stays in radicals instead of falling back to
        // Root objects.
        {
          let n = degree;
          let mut roots = Vec::new();

          // Build val^(1/n). For an odd degree and a negative value the
          // generating root is the REAL one, -|val|^(1/n), which is the
          // basis wolframscript reports the remaining roots against:
          // Solve[x^3 == -2, x] gives -2^(1/3), not (-2)^(1/3), as its
          // second solution.
          let negative_val = matches!(
            try_eval_to_f64(&val),
            Some(v) if v < 0.0
          );
          let root_of = |value: &Expr| -> Expr {
            let raw = pow(
              value.clone(),
              call("Rational", vec![Expr::Integer(1), Expr::Integer(n)]),
            );
            crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw)
          };
          let monomial_root = (!negative_val)
            .then(|| split_monomial_root(&val, n))
            .flatten();
          let val_root = if let Some(r) = monomial_root {
            r
          } else if n % 2 == 1 && negative_val {
            let magnitude =
              crate::evaluator::evaluate_expr_to_expr(&negate_expr(&val))
                .unwrap_or_else(|_| negate_expr(&val));
            let positive_root = root_of(&magnitude);
            crate::evaluator::evaluate_expr_to_expr(&negate_expr(
              &positive_root,
            ))
            .unwrap_or_else(|_| negate_expr(&positive_root))
          } else {
            root_of(&val)
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
                let g = gcd_i128(j, n);
                let p = j / g;
                let q = n / g;
                let multiplier = pow(
                  Expr::Integer(-1),
                  call("Rational", vec![Expr::Integer(p), Expr::Integer(q)]),
                );
                let product = times2(multiplier, val_root.clone());
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
              let g = gcd_i128(frac_num, frac_den);

              if frac_num == 0 {
                // frac = 0: roots are -val^(1/n) and val^(1/n)
                roots.push(make_rule(negate_expr(&val_root)));
                roots.push(make_rule(val_root.clone()));
              } else {
                let p = frac_num / g;
                let q = frac_den / g;
                let multiplier = if p == 1 && q == 2 {
                  id_expr("I")
                } else {
                  pow(
                    Expr::Integer(-1),
                    call("Rational", vec![Expr::Integer(p), Expr::Integer(q)]),
                  )
                };
                let product = times2(multiplier, val_root.clone());
                roots.push(make_rule(negate_expr(&product)));
                roots.push(make_rule(product));
              }
            }
          }

          // The roots are built as raw products of a radical and a root of
          // unity; evaluating them folds `(-1)^(2/3) 2` into `2 (-1)^(2/3)`
          // and `(-1)^(2/5) (-1)^(1/5)` into `(-1)^(3/5)`. wolframscript
          // then reports them in plain canonical order.
          for root in &mut roots {
            if let Expr::List(rules) = root
              && rules.len() == 1
              && let Expr::Rule {
                pattern,
                replacement,
              } = &rules[0]
            {
              let evaluated =
                crate::evaluator::evaluate_expr_to_expr(replacement)
                  .unwrap_or_else(|_| (**replacement).clone());
              *root =
                Expr::List(vec![rule_expr(*pattern.clone(), evaluated)].into());
            }
          }
          let root_value = |e: &Expr| -> Expr {
            match e {
              Expr::List(rules) if rules.len() == 1 => match &rules[0] {
                Expr::Rule { replacement, .. } => (**replacement).clone(),
                other => other.clone(),
              },
              other => other.clone(),
            }
          };
          roots.sort_by(|a, b| {
            crate::functions::list_helpers_ast::canonical_cmp(
              &root_value(a),
              &root_value(b),
            )
          });
          return Ok(Expr::List(roots.into()));
        }
      }

      // Higher degree: try Factor-based solving. Arbitrary-precision
      // coefficients are left to the numeric solver below: factoring their
      // exact rational images (`N[BernoulliB[20, z], 10]`) runs a GCD on
      // enormous rationals for nothing.
      let has_bigfloat = coeffs.iter().any(contains_bigfloat);
      if !has_bigfloat
        && let Ok(factored) = crate::functions::polynomial_ast::factor_ast(
          std::slice::from_ref(&expanded),
        )
      {
        // Split repeated factors `base^n` into their base and multiplicity,
        // so a pure power like `(1 + x)^3` is solved through its base too.
        let factors: Vec<(Expr, usize)> = extract_times_factors(&factored)
          .into_iter()
          .map(|factor| match power_base_exponent(&factor) {
            Some((base, Expr::Integer(n))) if *n >= 2 && *n <= 64 => {
              (base.clone(), *n as usize)
            }
            _ => (factor, 1),
          })
          .collect();
        // Only a real factorization: one non-constant factor alongside
        // constants is the polynomial itself, rescaled.
        let varying = factors
          .iter()
          .filter(|(factor, _)| !is_constant_wrt(factor, var))
          .count();
        if varying > 1 || factors.iter().any(|(_, n)| *n > 1) {
          // Solve each factor separately, repeating each root by the
          // multiplicity of its factor
          let mut all_solutions: Vec<Expr> = Vec::new();
          for (factor, multiplicity) in &factors {
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
              for _ in 0..*multiplicity {
                all_solutions.extend(sols.iter().cloned());
              }
            }
          }
          if !all_solutions.is_empty() {
            sort_solutions(&mut all_solutions);
            return Ok(Expr::List(all_solutions.into()));
          }
        }
      }
      // A cubic with symbolic coefficients has Cardano's radicals, in the
      // order wolframscript lists them.
      if degree == 3
        && coeffs
          .iter()
          .any(|c| !crate::functions::predicate_ast::is_numeric_q(c))
        && !coeffs
          .iter()
          .any(crate::functions::predicate_ast::contains_real_literal)
        && let Some(roots) = symbolic_cubic_roots(
          // The coefficients as written, before denominators were cleared:
          // wolframscript's radicals are in terms of those.
          &match &args[0] {
            Expr::Comparison { operands, .. } if operands.len() == 2 => {
              match crate::evaluator::evaluate_function_call_ast(
                "CoefficientList",
                &[
                  plus(vec![operands[0].clone(), neg(operands[1].clone())]),
                  id_expr(var),
                ],
              ) {
                Ok(Expr::List(ref raw))
                  if raw.len() == 4
                    && raw.iter().all(|c| is_constant_wrt(c, var)) =>
                {
                  raw.to_vec()
                }
                _ => coeffs.clone(),
              }
            }
            _ => coeffs.clone(),
          },
        )
      {
        return Ok(Expr::List(roots.into_iter().map(make_rule).collect()));
      }
      // Machine-precision coefficients: wolframscript never answers those
      // with `Root[…]` objects, it solves them numerically. So a cubic like
      // `x^3 + 1.5 x^2 - 3.2 x + 4.7 == 0` comes back as three approximate
      // roots instead of staying unevaluated.
      if coeffs
        .iter()
        .any(crate::functions::predicate_ast::contains_real_literal)
        && let Some(numeric) = numeric_polynomial_solutions(&coeffs, var)
      {
        return Ok(numeric);
      }
      // An irreducible quartic in x^2 alone is a quadratic in u = x^2:
      // each of its roots gives the two square roots ±Sqrt[u].
      if degree == 4
        && matches!(&coeffs[1], Expr::Integer(0))
        && matches!(&coeffs[3], Expr::Integer(0))
      {
        let u = "Solve$u";
        let quadratic = Expr::Comparison {
          operands: vec![
            plus(vec![
              coeffs[0].clone(),
              times(vec![coeffs[2].clone(), id_expr(u)]),
              times(vec![coeffs[4].clone(), pow(id_expr(u), Expr::Integer(2))]),
            ]),
            Expr::Integer(0),
          ],
          operators: vec![ComparisonOp::Equal],
        };
        if let Ok(Expr::List(ref u_roots)) = solve_ast(&[quadratic, id_expr(u)])
        {
          let mut values: Vec<Expr> = u_roots
            .iter()
            .filter_map(|root| match root {
              Expr::List(rules) => match rules.first() {
                Some(Expr::Rule { replacement, .. }) => {
                  Some((**replacement).clone())
                }
                _ => None,
              },
              _ => None,
            })
            .collect();
          // The roots with real square roots (u >= 0) come first, each
          // group by the value of u.
          let keys: Vec<Option<(bool, f64, f64)>> = values
            .iter()
            .map(|v| {
              let (re, im) = try_extract_complex_f64(
                &crate::evaluator::evaluate_expr_to_expr(&call(
                  "N",
                  vec![v.clone()],
                ))
                .ok()?,
              )?;
              Some((!(im.abs() < 1e-12 && re >= 0.0), re, im))
            })
            .collect();
          if keys.iter().all(Option::is_some) {
            let mut order: Vec<usize> = (0..values.len()).collect();
            order.sort_by(|&a, &b| {
              let (ka, kb) = (keys[a].unwrap(), keys[b].unwrap());
              ka.0
                .cmp(&kb.0)
                .then(ka.1.total_cmp(&kb.1))
                .then(ka.2.total_cmp(&kb.2))
            });
            values = order.into_iter().map(|i| values[i].clone()).collect();
          }
          let mut solutions = Vec::with_capacity(4);
          for replacement in &values {
            let sqrt = crate::evaluator::evaluate_expr_to_expr(&call(
              "Sqrt",
              vec![replacement.clone()],
            ))?;
            let negated =
              crate::evaluator::evaluate_expr_to_expr(&times(vec![
                Expr::Integer(-1),
                sqrt.clone(),
              ]))?;
            solutions.push(make_rule(negated));
            solutions.push(make_rule(sqrt));
          }
          if solutions.len() == 4 {
            return Ok(Expr::List(solutions.into()));
          }
        }
      }
      // Last resort for irreducible polynomials of degree ≥ 3 with
      // integer/rational coefficients: emit the wolframscript-style
      // list of Root expressions (`Root[poly &, k, 0]` for k = 1..deg).
      if let Some(rs) = make_root_solutions(&coeffs, var) {
        return Ok(rs);
      }
      Ok(unevaluated("Solve", args))
    }
  }
}

/// Solve a system of polynomial equations by eliminating one variable at a
/// time.
///
/// Solving one equation for one variable and substituting — what the generic
/// `Reduce` path does — introduces a radical as soon as that equation is
/// quadratic in the variable, and the radical then has to be eliminated all
/// over again from the equations it was substituted into. Combining whole
/// equations instead keeps every intermediate a polynomial, so the system
/// loses a variable and stays a polynomial system.
///
/// Returns `None` for anything this cannot settle — a non-polynomial or
/// parameterised equation, a system that stays underdetermined, a polynomial
/// Woxi cannot solve in radicals — which leaves the caller on its previous
/// path.
/// True if `expr` carries a subexpression that is neither arithmetic
/// structure, a number (exact or a numeric constant like `Log[2]`), nor one
/// of the unknowns — an opaque symbolic coefficient such as `q[[1]]`,
/// `f[1]` or `Sin[a]`.
///
/// Such a coefficient is a free parameter exactly like a free symbol is,
/// which is what the caller uses it for. A subexpression that *does* mention
/// an unknown is left alone: it makes the equation non-polynomial in that
/// unknown, which the caller's own `is_polynomial` check already rejects.
fn has_opaque_parameter(expr: &Expr, vars: &[String]) -> bool {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(..)
    | Expr::Identifier(_)
    | Expr::Constant(_) => false,
    Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "Plus" | "Times" | "Power" | "Rational" | "Subtract" | "Divide"
      ) =>
    {
      args.iter().any(|a| has_opaque_parameter(a, vars))
    }
    Expr::BinaryOp {
      op:
        BinaryOperator::Plus
        | BinaryOperator::Minus
        | BinaryOperator::Times
        | BinaryOperator::Divide
        | BinaryOperator::Power,
      left,
      right,
    } => has_opaque_parameter(left, vars) || has_opaque_parameter(right, vars),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => has_opaque_parameter(operand, vars),
    other => {
      if crate::functions::math_ast::try_eval_to_f64(other).is_some() {
        return false;
      }
      let mut free = Vec::new();
      collect_solve_vars(other, &mut free);
      !free.iter().any(|f| vars.contains(f))
    }
  }
}

fn try_solve_polynomial_system(eqs: &[Expr], vars: &[String]) -> Option<Expr> {
  // One variable is the ordinary single-equation case, and a system with
  // fewer equations than variables is underdetermined.
  if vars.len() < 2 || eqs.len() < vars.len() || vars.len() > MAX_SYSTEM_VARS {
    return None;
  }
  let mut polys = Vec::with_capacity(eqs.len());
  for eq in eqs {
    let (lhs, rhs, op) =
      crate::functions::polynomial_ast::reduce::extract_comparison(eq)?;
    if !matches!(op, crate::functions::polynomial_ast::reduce::CompOp::Equal) {
      return None;
    }
    let poly = expand_and_combine(&minus2(lhs, rhs));
    // Every symbol has to be one of the unknowns: a free parameter leaves
    // the solutions symbolic, and a symbolic solution cannot be checked
    // against the equations it did not come from. An opaque coefficient
    // (`q[[1]]`, `f[1]`) is such a parameter too even though it is not a
    // symbol — without this, a system whose coefficients are all opaque
    // looked parameter-free and went down the resultant path, which
    // returned the uncancelled quotients the general elimination path
    // reports as the plain `{{x -> q[[1]], y -> q[[2]]}}`.
    let mut free = Vec::new();
    collect_solve_vars(&poly, &mut free);
    if free.iter().any(|f| !vars.contains(f))
      || has_opaque_parameter(&poly, vars)
    {
      return None;
    }
    if !vars
      .iter()
      .all(|v| crate::functions::polynomial_ast::is_polynomial(&poly, v))
    {
      return None;
    }
    polys.push(poly);
  }

  let coupled = coupled_variables(&polys, vars);
  let Some(solutions) = poly_system_solutions(&polys, vars, &coupled) else {
    // An inconsistent system — its Gröbner basis is {1} — has no solutions.
    let var_list =
      Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect());
    if let Ok(Expr::List(ref basis)) =
      crate::evaluator::evaluate_function_call_ast(
        "GroebnerBasis",
        &[Expr::List(polys.clone().into()), var_list],
      )
      && matches!(basis.as_slice(), [Expr::Integer(1)])
    {
      return Some(Expr::List(Vec::new().into()));
    }
    return solve_polynomial_system_by_factors(&polys, vars);
  };
  if solutions.is_empty() {
    return Some(Expr::List(Vec::new().into()));
  }
  let solutions = drop_spurious_multiplicities(&polys, vars, solutions);
  let mut wrapped: Vec<Expr> = solutions
    .into_iter()
    .map(|values| {
      Expr::List(
        vars
          .iter()
          .zip(values)
          .map(|(v, value)| rule_expr(Expr::Identifier(v.clone()), value))
          .collect(),
      )
    })
    .collect();
  // Real solutions come before complex ones, as everywhere else in Solve.
  wrapped.sort_by_key(|sol| i32::from(contains_complex(sol)));
  Some(Expr::List(wrapped.into()))
}

/// A system the eliminations cannot take whole is split on an equation
/// that factors: its solutions are those of the systems with each factor in
/// the equation's place, merged without repeats — `(x - y)(x + y) == 0`
/// is `x == y` or `x == -y`.
fn solve_polynomial_system_by_factors(
  polys: &[Expr],
  vars: &[String],
) -> Option<Expr> {
  let factors_of = |poly: &Expr| -> Option<Vec<Expr>> {
    let Ok(Expr::List(ref pairs)) =
      crate::evaluator::evaluate_function_call_ast(
        "FactorList",
        std::slice::from_ref(poly),
      )
    else {
      return None;
    };
    let factors: Vec<Expr> = pairs
      .iter()
      .filter_map(|p| match p {
        Expr::List(fm) if fm.len() == 2 => Some(fm[0].clone()),
        _ => None,
      })
      .filter(|f| vars.iter().any(|v| contains_var(f, v)))
      .collect();
    (factors.len() >= 2).then_some(factors)
  };
  let (index, factors) = polys
    .iter()
    .enumerate()
    .find_map(|(i, p)| factors_of(p).map(|f| (i, f)))?;
  let mut merged: Vec<Expr> = Vec::new();
  let mut seen = std::collections::HashSet::new();
  for factor in factors {
    let eqs: Vec<Expr> = polys
      .iter()
      .enumerate()
      .map(|(i, p)| Expr::Comparison {
        operands: vec![
          if i == index {
            factor.clone()
          } else {
            p.clone()
          },
          Expr::Integer(0),
        ],
        operators: vec![ComparisonOp::Equal],
      })
      .collect();
    let Expr::List(ref solutions) = try_solve_polynomial_system(&eqs, vars)?
    else {
      return None;
    };
    for sol in solutions {
      if seen.insert(expr_to_string(sol)) {
        merged.push(sol.clone());
      }
    }
  }
  // In the order of the factors, as wolframscript lists them.
  Some(Expr::List(merged.into()))
}

/// The unknowns that some equation ties to another unknown, directly or
/// through a chain of equations.
///
/// Whether a solution counts more than once depends on it: wolframscript
/// keeps a root's multiplicity only where the system falls apart into
/// separate one-variable problems — `Solve[{x^2 == 0, y == 1}, {x, y}]`
/// lists its solution twice — and reports the plain intersection points as
/// soon as a variable has to be eliminated, however tangential the meeting:
/// `Solve[{x^2 + y^2 == 1, (x - 2)^2 + y^2 == 1}, {x, y}]` is one point,
/// not two, and `Solve[{x^3 == 0, y == x}, {x, y}]` one, not three. Mixed
/// systems mix the two rules per group: `Solve[{x^2 == 0, y^2 == 0,
/// z == x}, {x, y, z}]` is `y`'s double root times the single `(x, z)`
/// point.
fn coupled_variables(
  polys: &[Expr],
  vars: &[String],
) -> std::collections::HashSet<String> {
  use std::collections::HashMap;
  // Union-find over the unknowns: every equation merges the groups of all
  // the unknowns it mentions.
  let mut group: Vec<usize> = (0..vars.len()).collect();
  fn find(group: &mut [usize], mut i: usize) -> usize {
    while group[i] != i {
      group[i] = group[group[i]];
      i = group[i];
    }
    i
  }
  for poly in polys {
    let present: Vec<usize> = (0..vars.len())
      .filter(|i| max_power_int(poly, &vars[*i]).unwrap_or(0) > 0)
      .collect();
    for pair in present.windows(2) {
      let (a, b) = (find(&mut group, pair[0]), find(&mut group, pair[1]));
      if a != b {
        group[a] = b;
      }
    }
  }
  let mut size: HashMap<usize, usize> = HashMap::new();
  for i in 0..vars.len() {
    let root = find(&mut group, i);
    *size.entry(root).or_insert(0) += 1;
  }
  (0..vars.len())
    .filter(|i| {
      let root = find(&mut group, *i);
      size.get(&root).copied().unwrap_or(1) > 1
    })
    .map(|i| vars[i].clone())
    .collect()
}

/// More variables than this and the eliminations multiply out of hand.
const MAX_SYSTEM_VARS: usize = 4;

/// The largest Sylvester matrix an elimination step is allowed to build.
/// Two cubics already produce a degree-9 polynomial in the variable that
/// survives, which is past what Solve reports in radicals anyway.
const MAX_SYLVESTER_SIZE: i128 = 8;

/// Values for `vars`, in order, at every common root of `polys`.
///
/// `var` (the last one) is eliminated first: the equations containing it are
/// divided into one another until only one still does, and what falls out
/// are equations in the remaining variables. Those are solved by recursion,
/// and each of their solutions is put back into the original equations,
/// which then only leave `var` to solve for.
fn poly_system_solutions(
  polys: &[Expr],
  vars: &[String],
  coupled: &std::collections::HashSet<String>,
) -> Option<Vec<Vec<Expr>>> {
  let (var, outer_vars) = vars.split_last()?;

  // Eliminate `var` down to a single equation. Every step replaces the
  // higher-degree of two equations with one of strictly lower degree in
  // `var`, so this terminates.
  let mut working: Vec<Expr> = polys.to_vec();
  loop {
    let mut in_var: Vec<usize> = (0..working.len())
      .filter(|i| max_power_int(&working[*i], var).unwrap_or(0) > 0)
      .collect();
    if in_var.len() < 2 {
      break;
    }
    in_var.sort_by_key(|i| max_power_int(&working[*i], var).unwrap_or(0));
    let divisor = in_var[0];
    let dividend = in_var[1];
    working[dividend] =
      eliminate_var(&working[dividend], &working[divisor], var)?;
  }

  let mut solved_for_var: Option<&Expr> = None;
  let mut reduced: Vec<Expr> = Vec::new();
  for poly in &working {
    if max_power_int(poly, var).unwrap_or(0) > 0 {
      solved_for_var = Some(poly);
    } else if !is_zero_expr(poly) {
      reduced.push(poly.clone());
    }
  }
  // Nothing left to pin `var` down: the system is underdetermined.
  let solved_for_var = solved_for_var?;

  if outer_vars.is_empty() {
    // A leftover equation in no variable at all is either the trivial
    // `0 == 0`, already dropped, or a contradiction.
    if reduced.iter().any(|p| !is_zero_expr(p)) {
      return Some(Vec::new());
    }
    let mut result: Vec<Vec<Expr>> = Vec::new();
    for value in roots_of_last_variable(polys, solved_for_var, var)? {
      for _ in 0..solution_multiplicity(polys, &value, var, coupled) {
        result.push(vec![value.clone()]);
      }
    }
    return Some(result);
  }

  if reduced.is_empty() {
    return None;
  }
  let outer_solutions = poly_system_solutions(&reduced, outer_vars, coupled)?;

  let mut result: Vec<Vec<Expr>> = Vec::new();
  let mut index = 0;
  while index < outer_solutions.len() {
    // Solutions repeat in the reduced system when several points of the
    // full system share those outer values, and also when the point they
    // share counts more than once.
    let outer = &outer_solutions[index];
    let mut outer_multiplicity = 0;
    while index < outer_solutions.len()
      && solution_values_equal(outer, &outer_solutions[index])
    {
      outer_multiplicity += 1;
      index += 1;
    }

    // With machine values substituted, a coefficient that vanishes at the
    // point comes out as rounding noise; it would split a multiple root
    // into nearby roots that fail the check against the system.
    let substituted: Vec<Expr> = polys
      .iter()
      .map(|p| drop_rounding_noise(&substitute_values(p, outer_vars, outer)))
      .collect();
    let pivot = substituted
      .iter()
      .filter(|p| max_power_int(p, var).unwrap_or(0) > 0)
      .min_by_key(|p| max_power_int(p, var).unwrap_or(i128::MAX))?;
    for value in roots_of_last_variable(&substituted, pivot, var)? {
      for _ in 0..outer_multiplicity
        * solution_multiplicity(&substituted, &value, var, coupled)
      {
        let mut full = outer.clone();
        full.push(value.clone());
        result.push(full);
      }
    }
  }
  Some(result)
}

/// Drops duplicate copies of a common root that eliminating a variable
/// spuriously multiplied up.
///
/// Elimination divides one equation's higher-degree-in-`var` terms down
/// using another's, and when that divisor's leading coefficient in `var`
/// happens to vanish at a candidate point, the division has effectively
/// multiplied through by (a factor that vanishes there), inflating the
/// point's reported multiplicity even though the two curves cross it
/// transversally. A transversal crossing is exactly one where the
/// system's Jacobian is nonsingular (the implicit function theorem), so
/// any extra copies of such a point are the artifact, not a real
/// multiplicity — where the Jacobian is singular, every copy is kept.
///
/// A second line of defence: a system whose variables have to be
/// eliminated against each other no longer reports a multiplicity at all
/// (see `coupled_variables`), so nothing should reach here with copies to
/// drop.
fn drop_spurious_multiplicities(
  polys: &[Expr],
  vars: &[String],
  solutions: Vec<Vec<Expr>>,
) -> Vec<Vec<Expr>> {
  if polys.len() != vars.len() || solutions.len() < 2 {
    return solutions;
  }
  let mut result: Vec<Vec<Expr>> = Vec::with_capacity(solutions.len());
  for point in solutions {
    let already_kept = result
      .iter()
      .any(|kept| solution_values_equal(kept, &point));
    if already_kept && jacobian_is_nonsingular(polys, vars, &point) {
      continue;
    }
    result.push(point);
  }
  result
}

/// Whether the Jacobian of `polys` with respect to `vars`, evaluated at
/// `point`, is nonsingular. Falls back to `false` (i.e. "treat as a real
/// multiplicity, don't drop it") whenever the entries cannot be pinned down
/// to real numbers, since that is the safe direction — it only ever costs
/// a duplicate that should have been dropped, never drops a genuine one.
fn jacobian_is_nonsingular(
  polys: &[Expr],
  vars: &[String],
  point: &[Expr],
) -> bool {
  let n = vars.len();
  let mut matrix = vec![vec![0.0_f64; n]; n];
  for (i, poly) in polys.iter().enumerate() {
    for (j, var) in vars.iter().enumerate() {
      let Ok(derivative) = differentiate_expr(poly, var) else {
        return false;
      };
      let value = substitute_values(&derivative, vars, point);
      let Some((re, im)) = try_extract_complex_f64(&value) else {
        return false;
      };
      if im.abs() > 1e-9 * (1.0 + re.abs()) {
        return false;
      }
      matrix[i][j] = re;
    }
  }
  // Compare the determinant against the Hadamard bound (the product of the
  // row norms) rather than an absolute epsilon, so the test scales with
  // the size of the entries instead of being fooled by them.
  let row_norms: Vec<f64> = matrix
    .iter()
    .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
    .collect();
  let hadamard_bound: f64 = row_norms.iter().product();
  if hadamard_bound <= 1e-12 {
    return false;
  }
  (determinant_f64(&matrix) / hadamard_bound).abs() > 1e-6
}

/// Determinant of a small square matrix via Laplace expansion (`vars` is
/// capped at `MAX_SYSTEM_VARS`, so this never sees more than a few rows).
fn determinant_f64(matrix: &[Vec<f64>]) -> f64 {
  match matrix.len() {
    0 => 1.0,
    1 => matrix[0][0],
    2 => matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
    n => (0..n)
      .map(|c| {
        let minor: Vec<Vec<f64>> = matrix[1..]
          .iter()
          .map(|row| {
            row
              .iter()
              .enumerate()
              .filter(|(cc, _)| *cc != c)
              .map(|(_, v)| *v)
              .collect()
          })
          .collect();
        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
        sign * matrix[0][c] * determinant_f64(&minor)
      })
      .sum(),
  }
}

/// How often the solution at `value` counts, given which unknowns the
/// system ties together. A variable that had to be eliminated against
/// another one contributes each of its values once — see
/// `coupled_variables` — and only a variable standing on its own carries
/// its root's multiplicity.
fn solution_multiplicity(
  polys: &[Expr],
  value: &Expr,
  var: &str,
  coupled: &std::collections::HashSet<String>,
) -> usize {
  if coupled.contains(var) {
    1
  } else {
    root_multiplicity(polys, value, var)
  }
}

/// How often the solution at `value` counts.
///
/// A curve can meet the line the other variables were fixed to twice over
/// while still crossing the other curve just once — the circle `x^2 + y^2
/// == 1` touches `x == 1` twice, but meets `(x - 1)^2 + (y - 1)^2 == 1`
/// transversally there. So the count is the smallest multiplicity `value`
/// has among the equations, not the one the equation solved for it happens
/// to report.
fn root_multiplicity(polys: &[Expr], value: &Expr, var: &str) -> usize {
  let mut multiplicity = usize::MAX;
  for poly in polys {
    if max_power_int(poly, var).unwrap_or(0) <= 0 {
      continue;
    }
    let Some(roots) = univariate_roots(poly, var) else {
      continue;
    };
    let count = roots.iter().filter(|r| values_equal(r, value)).count();
    if count > 0 {
      multiplicity = multiplicity.min(count);
    }
  }
  if multiplicity == usize::MAX {
    1
  } else {
    multiplicity
  }
}

/// The distinct roots of `pivot` in `var` that satisfy every polynomial in
/// `polys`. How often each counts is `root_multiplicity`'s business.
fn roots_of_last_variable(
  polys: &[Expr],
  pivot: &Expr,
  var: &str,
) -> Option<Vec<Expr>> {
  let vars = [var.to_string()];
  let mut roots: Vec<Expr> = Vec::new();
  for value in univariate_roots(pivot, var)? {
    if roots.iter().any(|kept| values_equal(kept, &value)) {
      continue;
    }
    if polys
      .iter()
      .all(|p| poly_vanishes_at(p, &vars, std::slice::from_ref(&value)))
    {
      roots.push(value);
    }
  }
  Some(roots)
}

/// Combine `dividend` and `divisor` into an equation of strictly lower
/// degree in `var` that every common root of the two still satisfies.
///
/// Polynomial division does it whenever the divisor's leading coefficient is
/// a number, and then the step is reversible — no root is gained or lost,
/// and the remainder of two circles is the line through their intersections,
/// which is far better conditioned than anything built from their squares.
/// A leading coefficient that itself depends on the other variables would
/// need dividing through by an expression that may vanish, so those go to
/// the resultant instead, which eliminates `var` outright.
fn eliminate_var(dividend: &Expr, divisor: &Expr, var: &str) -> Option<Expr> {
  let var_expr = id_expr(var);
  if leading_coefficient_is_numeric(divisor, var) {
    let remainder =
      crate::functions::polynomial_ast::polynomial_remainder_ast(&[
        dividend.clone(),
        divisor.clone(),
        var_expr,
      ])
      .ok()?;
    return Some(drop_rounding_noise(&expand_and_combine(&remainder)));
  }
  let dividend_degree = max_power_int(dividend, var)?;
  let divisor_degree = max_power_int(divisor, var)?;
  if dividend_degree + divisor_degree > MAX_SYLVESTER_SIZE {
    return None;
  }
  let eliminated = crate::functions::polynomial_ast::resultant_ast(&[
    dividend.clone(),
    divisor.clone(),
    var_expr,
  ])
  .ok()?;
  let eliminated = drop_rounding_noise(&expand_and_combine(&eliminated));
  // A resultant that vanishes identically means the two equations share a
  // whole curve, so the system is not zero-dimensional.
  if is_zero_expr(&eliminated) {
    return None;
  }
  Some(eliminated)
}

/// Drop the terms of an eliminated equation that are made of nothing but
/// rounding error.
///
/// Terms that cancel exactly in exact arithmetic only cancel to within a
/// rounding error in inexact arithmetic, and what is left behind claims the
/// eliminated variable is still there — with a coefficient small enough that
/// dividing by it swamps the equation in noise. A term more than twelve
/// orders of magnitude below the largest one in the same equation is that
/// leftover, not a coefficient anybody wrote. Exact equations are left
/// alone: nothing there is approximate.
fn drop_rounding_noise(poly: &Expr) -> Expr {
  if !crate::functions::math_ast::numeric_utils::expr_contains_real(poly) {
    return poly.clone();
  }
  let terms = collect_additive_terms(poly);
  let magnitudes: Vec<f64> = terms.iter().map(term_magnitude).collect();
  let largest = magnitudes.iter().copied().fold(0.0, f64::max);
  if largest == 0.0 {
    return poly.clone();
  }
  let kept: Vec<Expr> = terms
    .iter()
    .zip(&magnitudes)
    .filter(|(_, magnitude)| **magnitude > 1e-12 * largest)
    .map(|(term, _)| term.clone())
    .collect();
  if kept.len() == terms.len() {
    return poly.clone();
  }
  if kept.is_empty() {
    return Expr::Integer(0);
  }
  expand_and_combine(&build_sum(kept))
}

/// The size of a term's numeric part, taking anything symbolic as a factor
/// of one.
fn term_magnitude(term: &Expr) -> f64 {
  let mut magnitude = 1.0;
  for factor in collect_multiplicative_factors(term) {
    if let Some((re, im)) = try_extract_complex_f64(&factor) {
      magnitude *= re.hypot(im);
    }
  }
  magnitude
}

/// Whether any number in `expr` is a machine-precision one.
fn contains_bigfloat(expr: &Expr) -> bool {
  match expr {
    Expr::BigFloat(..) => true,
    Expr::List(items) => items.iter().any(contains_bigfloat),
    Expr::FunctionCall { args, .. } => args.iter().any(contains_bigfloat),
    Expr::BinaryOp { left, right, .. } => {
      contains_bigfloat(left) || contains_bigfloat(right)
    }
    Expr::UnaryOp { operand, .. } => contains_bigfloat(operand),
    _ => false,
  }
}

/// Whether the coefficient of the highest power of `var` in `poly` is a
/// number other than zero.
fn leading_coefficient_is_numeric(poly: &Expr, var: &str) -> bool {
  let Some(degree) = max_power_int(poly, var) else {
    return false;
  };
  let Ok(leading) = crate::functions::polynomial_ast::coefficient_ast(&[
    poly.clone(),
    id_expr(var),
    Expr::Integer(degree),
  ]) else {
    return false;
  };
  let Ok(leading) = crate::evaluator::evaluate_expr_to_expr(&leading) else {
    return false;
  };
  match try_extract_complex_f64(&leading) {
    Some((re, im)) => re != 0.0 || im != 0.0,
    None => false,
  }
}

/// The roots of `poly` (as `poly == 0`) in `var`, with multiplicity, or
/// `None` when Solve cannot report them explicitly.
/// The roots of `poly` in `var`, numericized from the closed form, when it is
/// at most quadratic with exact numeric coefficients.
fn numeric_quadratic_roots(poly: &Expr, var: &str) -> Option<Vec<Expr>> {
  let eval = |name: &str, args: Vec<Expr>| {
    crate::evaluator::evaluate_function_call_ast(name, &args).ok()
  };
  let list = eval("CoefficientList", vec![poly.clone(), id_expr(var)])?;
  let Expr::List(coeffs) = &list else {
    return None;
  };
  let exact = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::BigInteger(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  if !coeffs.iter().all(exact) {
    return None;
  }
  let n = |e: Expr| eval("N", vec![e]);
  match coeffs.as_slice() {
    [c, b] if !matches!(b, Expr::Integer(0)) => Some(vec![n(eval(
      "Times",
      vec![
        Expr::Integer(-1),
        c.clone(),
        eval("Power", vec![b.clone(), Expr::Integer(-1)])?,
      ],
    )?)?]),
    [c, b, a] => {
      let disc = eval(
        "Plus",
        vec![
          eval("Power", vec![b.clone(), Expr::Integer(2)])?,
          eval("Times", vec![Expr::Integer(-4), a.clone(), c.clone()])?,
        ],
      )?;
      let root = |sign: i128| -> Option<Expr> {
        let numerator = eval(
          "Plus",
          vec![
            eval("Times", vec![Expr::Integer(-1), b.clone()])?,
            eval(
              "Times",
              vec![Expr::Integer(sign), eval("Sqrt", vec![disc.clone()])?],
            )?,
          ],
        )?;
        n(eval(
          "Times",
          vec![
            numerator,
            eval(
              "Power",
              vec![
                eval("Times", vec![Expr::Integer(2), a.clone()])?,
                Expr::Integer(-1),
              ],
            )?,
          ],
        )?)
      };
      // In Solve's order: ascending, the conjugate with the negative
      // imaginary part first.
      let mut roots = vec![root(-1)?, root(1)?];
      let key = |e: &Expr| {
        try_eval_to_f64(e)
          .map(|re| (re, 0.0))
          .or_else(|| try_extract_complex_float(e))
      };
      if let (Some(a), Some(b)) = (key(&roots[0]), key(&roots[1]))
        && (if a.1 != 0.0 || b.1 != 0.0 {
          b.1 < a.1
        } else {
          b.0 < a.0
        })
      {
        roots.swap(0, 1);
      }
      Some(roots)
    }
    _ => None,
  }
}

fn univariate_roots(poly: &Expr, var: &str) -> Option<Vec<Expr>> {
  // Under NSolve the roots are wanted as numbers: solving an eliminant in
  // radicals (and simplifying them) is what made a rationalized geometry
  // system take seconds per intersection. A quadratic's closed form is
  // numericized directly, which rounds its roots correctly; higher degrees
  // go to the numeric root finder.
  if numeric_roots_wanted()
    && let Some(roots) = numeric_quadratic_roots(poly, var)
  {
    return Some(roots);
  }
  // Only a polynomial in `var` alone: one that still has another unknown
  // in its coefficients is no numeric root-finding problem.
  let numeric = numeric_roots_wanted()
    && !matches!(
      crate::evaluator::evaluate_function_call_ast(
        "Exponent",
        &[poly.clone(), id_expr(var)],
      ),
      Ok(Expr::Integer(d)) if d <= 2
    )
    && matches!(
      &crate::evaluator::evaluate_function_call_ast(
        "Variables",
        std::slice::from_ref(poly),
      ),
      Ok(Expr::List(vs)) if vs.len() == 1
    );
  let poly_exact = &poly.clone();
  let poly = if numeric {
    crate::evaluator::evaluate_function_call_ast(
      "N",
      std::slice::from_ref(poly),
    )
    .ok()?
  } else {
    poly.clone()
  };
  let equation = Expr::Comparison {
    operands: vec![poly, Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };
  let solved = if numeric {
    nsolve_ast(&[equation, id_expr(var)]).ok()?
  } else {
    solve_ast(&[equation, id_expr(var)]).ok()?
  };
  let Expr::List(ref solutions) = solved else {
    return None;
  };
  let mut roots = Vec::with_capacity(solutions.len());
  for solution in solutions {
    let Expr::List(rules) = solution else {
      return None;
    };
    let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
      return None;
    };
    // A root reported as an unsolved `Root[…]` object is no use for
    // substituting back into the remaining equations.
    if expr_to_string(replacement).contains("Root[") {
      return None;
    }
    roots.push(replacement.as_ref().clone());
  }
  if numeric {
    snap_to_square_free_roots(&mut roots, poly_exact, var);
  }
  Some(roots)
}

/// Repeated roots (a resultant is often a square) come out of a numeric root
/// finder good only to eps^(1/k) for a k-fold root, too rough to pass the
/// check against the system. Each root is replaced by the nearest root of
/// the exact square-free part `poly / gcd(poly, poly')`, where every root is
/// simple and found to full precision; the list keeps its length and order.
fn snap_to_square_free_roots(roots: &mut [Expr], poly: &Expr, var: &str) {
  if crate::functions::math_ast::contains_inexact_real(poly) {
    return;
  }
  let ev = |head: &str, a: Vec<Expr>| {
    crate::evaluator::evaluate_function_call_ast(head, &a)
  };
  let Ok(square_free) = ev("D", vec![poly.clone(), id_expr(var)])
    .and_then(|d| ev("PolynomialGCD", vec![poly.clone(), d]))
    .and_then(|g| {
      ev("PolynomialQuotient", vec![poly.clone(), g, id_expr(var)])
    })
  else {
    return;
  };
  let Ok(solved) = nsolve_ast(&[
    Expr::Comparison {
      operands: vec![square_free, Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    },
    id_expr(var),
  ]) else {
    return;
  };
  let Expr::List(solutions) = &solved else {
    return;
  };
  let simple: Vec<(Expr, (f64, f64))> = solutions
    .iter()
    .filter_map(|s| match s {
      Expr::List(rules) => match rules.as_slice() {
        [Expr::Rule { replacement, .. }] => {
          let v = replacement.as_ref().clone();
          let c = crate::functions::math_ast::try_extract_complex_float(&v)?;
          Some((v, c))
        }
        _ => None,
      },
      _ => None,
    })
    .collect();
  for root in roots.iter_mut() {
    let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(root)
    else {
      continue;
    };
    let nearest = simple.iter().min_by(|a, b| {
      let da = (a.1.0 - re).hypot(a.1.1 - im);
      let db = (b.1.0 - re).hypot(b.1.1 - im);
      da.total_cmp(&db)
    });
    if let Some((v, (sre, sim))) = nearest
      && (sre - re).hypot(sim - im) <= 1e-4 * re.hypot(im).max(1.0)
    {
      *root = v.clone();
    }
  }
}

/// Substitute `values` for `vars` in `expr` and evaluate.
fn substitute_values(expr: &Expr, vars: &[String], values: &[Expr]) -> Expr {
  let mut substituted = expr.clone();
  for (var, value) in vars.iter().zip(values) {
    substituted = crate::syntax::substitute_variable(&substituted, var, value);
  }
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&substituted)
    .unwrap_or(substituted);
  expand_and_combine(&evaluated)
}

/// Whether `poly` is zero once `values` are substituted for `vars`.
///
/// An exact substitution may leave a root such as `Sqrt[7]/2` in a form that
/// does not collapse to a literal zero, and an inexact one never lands on
/// zero exactly, so the residual is weighed against the size of the terms it
/// cancelled between.
fn poly_vanishes_at(poly: &Expr, vars: &[String], values: &[Expr]) -> bool {
  let substituted = substitute_values(poly, vars, values);
  if is_zero_expr(&substituted) {
    return true;
  }
  let Some((re, im)) = try_extract_complex_f64(&substituted) else {
    return false;
  };
  // The size of the terms that cancelled is the rounding error to expect, so
  // measure each term of the polynomial itself at the point: the evaluated
  // sum has already lost them (`1.4e15 - 1.4e15 + 1.` is just `1.`).
  let mut scale: f64 = 1.0;
  for term in collect_additive_terms(&expand_and_combine(poly)) {
    let at_point = substitute_values(&term, vars, values);
    if let Some((tre, tim)) = try_extract_complex_f64(&at_point) {
      scale += tre.hypot(tim);
    }
  }
  re.hypot(im) <= 1e-12 * scale
}

/// Whether `expr` is the literal zero left by a cancellation.
fn is_zero_expr(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(0)) || matches!(expr, Expr::Real(r) if *r == 0.0)
}

/// Whether two values name the same number — structurally, or numerically
/// for the inexact ones that never match structurally.
fn values_equal(a: &Expr, b: &Expr) -> bool {
  if expr_to_string(a) == expr_to_string(b) {
    return true;
  }
  let (Some((are, aim)), Some((bre, bim))) =
    (try_extract_complex_f64(a), try_extract_complex_f64(b))
  else {
    return false;
  };
  let scale = 1.0 + are.hypot(aim).max(bre.hypot(bim));
  (are - bre).hypot(aim - bim) <= 1e-9 * scale
}

/// Whether two solutions assign the same values to every variable.
fn solution_values_equal(a: &[Expr], b: &[Expr]) -> bool {
  a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y))
}

/// Highest power of `var` in `eq` (treated as `lhs - rhs`). Returns
/// `None` for non-polynomial equations or when `var` does not appear.
fn max_degree_of_var(eq: &Expr, var: &str) -> Option<i128> {
  let lhs_minus_rhs = match eq {
    Expr::Comparison { operands, .. } if operands.len() == 2 => {
      minus2(operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      minus2(args[0].clone(), args[1].clone())
    }
    other => other.clone(),
  };
  let expanded =
    crate::evaluator::evaluate_expr_to_expr(&lhs_minus_rhs).ok()?;
  crate::functions::polynomial_ast::max_power_int(&expanded, var)
}

/// Solve a polynomial whose coefficients are machine numbers by finding its
/// roots numerically, returning `{{var -> r1}, …}` ordered by ascending real
/// part and then ascending imaginary part — the order wolframscript reports.
/// Returns None when some coefficient is not a machine number or the leading
/// one vanishes.
fn numeric_polynomial_solutions(coeffs: &[Expr], var: &str) -> Option<Expr> {
  // Complex machine coefficients (`(0. - 4.4*^-16 I) z`, `0. + 0.*I`) are
  // solved by the complex-coefficient iteration.
  let mut complex = Vec::with_capacity(coeffs.len());
  let mut has_complex_coefficient = false;
  for c in coeffs {
    complex.push(if let Some(v) = try_eval_to_f64(c) {
      (v, 0.0)
    } else {
      has_complex_coefficient = true;
      try_extract_complex_float(c)?
    });
  }
  let degree = complex.len().checked_sub(1)?;
  if degree < 1 || complex[degree].0.hypot(complex[degree].1) < 1e-300 {
    return None;
  }
  let mut roots = if has_complex_coefficient {
    durand_kerner_complex_roots(&complex)
  } else {
    let numeric: Vec<f64> = complex.iter().map(|c| c.0).collect();
    durand_kerner_roots(&numeric)
  };
  roots.sort_by(|a, b| {
    a.0
      .partial_cmp(&b.0)
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
  });
  let solutions: Vec<Expr> = roots
    .into_iter()
    .map(|(re, im)| {
      let value = if im == 0.0 {
        Expr::Real(re)
      } else {
        crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
      };
      Expr::List(vec![rule_expr(id_expr(var), value)].into())
    })
    .collect();
  Some(Expr::List(solutions.into()))
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
      _ => Some(pow(slot.clone(), Expr::Integer(i as i128))),
    };
    let term = match (var_pow, c) {
      (None, c) => c.clone(),
      (Some(p), Expr::Integer(1)) => p,
      (Some(p), c) => times(vec![c.clone(), p]),
    };
    terms.push(term);
  }
  let body = match terms.len() {
    0 => return None,
    1 => terms.remove(0),
    _ => plus(terms),
  };
  let body = crate::evaluator::evaluate_expr_to_expr(&body).ok()?;
  let func = Expr::Function {
    body: Box::new(body),
  };
  let mut solutions = Vec::with_capacity(degree);
  for k in 1..=degree {
    let root = call(
      "Root",
      vec![func.clone(), Expr::Integer(k as i128), Expr::Integer(0)],
    );
    solutions.push(Expr::List(vec![rule_expr(id_expr(var), root)].into()));
  }
  Some(Expr::List(solutions.into()))
}

/// Sort a list of Solve solutions (each is `{var -> val}`) by root value.
/// Uses `solve_order` so complex roots interleave with reals by real
/// part, matching wolframscript's `Solve[x^5 == x, x]` output.
fn sort_solutions(solutions: &mut [Expr]) {
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

/// Whether a solution `{v -> value, …}` assigns a non-real number to some
/// unknown through a `Root` object, which hides its complex roots from the
/// syntactic `contains_complex` check.
fn has_nonreal_root_value(sol: &Expr) -> bool {
  let Expr::List(rules) = sol else {
    return false;
  };
  rules.iter().any(|rule| {
    let Expr::Rule { replacement, .. } = rule else {
      return false;
    };
    if !expr_to_string(replacement).contains("Root[") {
      return false;
    }
    let im = crate::evaluator::evaluate_function_call_ast(
      "Im",
      &[(**replacement).clone()],
    )
    .and_then(|v| crate::evaluator::evaluate_function_call_ast("N", &[v]));
    match im {
      Ok(v) => !matches!(expr_to_f64(&v), Ok(x) if x.abs() < 1e-9),
      Err(_) => false,
    }
  })
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
/// The `(base, exponent)` of a `Power` expression in either representation.
fn power_base_exponent(expr: &Expr) -> Option<(&Expr, &Expr)> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => Some((left, right)),
    _ => None,
  }
}

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
      (c, Some(v)) => times2(Expr::Integer(c), v),
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
    _ => times(coeff_factors),
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
    crate::evaluator::evaluate_expr_to_expr(&div2(val_side, coeff)).ok()?
  };

  // Solve `inner == value`, returning the outer list's solution entries.
  let solve_branch = |value: Expr| -> Option<Vec<Expr>> {
    let branch_eq = Expr::Comparison {
      operands: vec![inner.clone(), value],
      operators: vec![ComparisonOp::Equal],
    };
    let r = solve_ast(&[branch_eq, id_expr(var)]).ok()?;
    match r {
      Expr::List(ref items) => Some(items.to_vec()),
      _ => None,
    }
  };

  let mut solutions: Vec<Expr> = Vec::new();
  match try_eval_to_f64(&eff) {
    Some(v) if v < 0.0 => {} // no real solution → {}
    Some(0.0) => {
      solutions.extend(solve_branch(Expr::Integer(0))?);
    }
    _ => {
      // Inverting `Abs` splits the equation into two branches, and over the
      // complexes that throws solutions away — `Abs[x] == 2` also holds all
      // around the circle of radius 2, which only `Reduce` reports. Over the
      // reals nothing is lost, so the message is confined to the other
      // domains, exactly as wolframscript does it.
      report_inverse_function_use();
      // Positive numeric or symbolic value: both signs. The negative branch
      // is added first so the symbolic case keeps wolframscript's order
      // ({x -> -a} before {x -> a}); numeric cases are reordered by
      // sort_solutions anyway.
      let neg =
        crate::evaluator::evaluate_expr_to_expr(&neg1(eff.clone())).ok()?;
      solutions.extend(solve_branch(neg)?);
      solutions.extend(solve_branch(eff)?);
    }
  }
  sort_solutions(&mut solutions);
  Some(Ok(Expr::List(solutions.into())))
}

/// Drop a leading nonzero constant factor (one free of `var`) from a product,
/// returning the remaining factor. Anything that is not such a product comes
/// back unchanged.
fn strip_constant_factor(expr: &Expr, var: &str) -> Expr {
  let is_nonzero_const = |e: &Expr| -> bool {
    !contains_var(e, var) && try_eval_to_f64(e).is_some_and(|v| v != 0.0)
  };
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (**operand).clone(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if is_nonzero_const(left) => (**right).clone(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if is_nonzero_const(right) => (**left).clone(),
    Expr::FunctionCall { name, args }
      if name == "Times" && args.len() == 2 && is_nonzero_const(&args[0]) =>
    {
      args[1].clone()
    }
    other => other.clone(),
  }
}

/// Whether `expr` is `b^g` with a positive constant base `b` and an exponent
/// depending on `var` (`E^x`, `2^(1 - x/3)`): a factor that is never zero.
fn is_nonvanishing_exponential(expr: &Expr, var: &str) -> bool {
  let (base, exp) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return false,
  };
  !contains_var(base, var)
    && contains_var(exp, var)
    && try_eval_to_f64(base).is_some_and(|b| b > 0.0)
}

/// Collect (deduplicated by printed form) the nonvanishing exponentials
/// occurring anywhere inside `expr`.
fn collect_nonvanishing_exponentials(
  expr: &Expr,
  var: &str,
  out: &mut Vec<Expr>,
) {
  if is_nonvanishing_exponential(expr, var) {
    if !out
      .iter()
      .any(|e| expr_to_string(e) == expr_to_string(expr))
    {
      out.push(expr.clone());
    }
    return;
  }
  match expr {
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_nonvanishing_exponentials(a, var, out);
      }
    }
    Expr::List(items) => {
      for a in items {
        collect_nonvanishing_exponentials(a, var, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_nonvanishing_exponentials(left, var, out);
      collect_nonvanishing_exponentials(right, var, out);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_nonvanishing_exponentials(operand, var, out);
    }
    _ => {}
  }
}

/// For a sum whose terms all carry the same exponential (as `D` of
/// `x^2 E^(1 - x/3)` returns), divide that exponential out: the quotient has
/// the same roots. `None` unless some candidate divides out completely.
fn peel_common_exponential(lhs: &Expr, var: &str) -> Option<Expr> {
  let mut candidates = Vec::new();
  collect_nonvanishing_exponentials(lhs, var, &mut candidates);
  for c in candidates {
    let quotient = crate::evaluator::evaluate_expr_to_expr(&call1(
      "Expand",
      div2(lhs.clone(), c),
    ))
    .ok()?;
    let mut left = Vec::new();
    collect_nonvanishing_exponentials(&quotient, var, &mut left);
    if left.is_empty() {
      return Some(quotient);
    }
  }
  None
}

/// Solve `product == 0` by dropping the factors that can never vanish
/// (exponentials with a positive constant base). Returns `None` when the
/// product has no such factor.
fn try_solve_peel_exponential_factors(
  eq: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  let lhs = match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal
      && matches!(operands[1], Expr::Integer(0)) =>
    {
      &operands[0]
    }
    Expr::FunctionCall { name, args }
      if name == "Equal"
        && args.len() == 2
        && matches!(args[1], Expr::Integer(0)) =>
    {
      &args[0]
    }
    _ => return None,
  };
  let factors = extract_times_factors(lhs);
  if !factors.iter().any(|f| is_nonvanishing_exponential(f, var)) {
    return peel_common_exponential(lhs, var).map(|reduced| {
      let new_eq = Expr::Comparison {
        operands: vec![reduced, Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      };
      solve_ast(&[new_eq, id_expr(var)])
    });
  }
  let rest: Vec<Expr> = factors
    .into_iter()
    .filter(|f| !is_nonvanishing_exponential(f, var))
    .collect();
  if rest.is_empty() {
    return Some(Ok(Expr::List(vec![].into())));
  }
  let reduced = crate::evaluator::evaluate_expr_to_expr(&times(rest)).ok()?;
  let new_eq = Expr::Comparison {
    operands: vec![reduced, Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };
  Some(solve_ast(&[new_eq, id_expr(var)]))
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
  if let Some(result) = try_solve_trig_pair_eq(lhs, rhs, var) {
    return Some(result);
  }
  // A nonzero constant factor does not change the roots of `… == 0`, so peel
  // it off: `-Cos[x] == 0` and `3*Sin[x] == 0` solve like `Cos[x] == 0` and
  // `Sin[x] == 0`.
  let stripped;
  let lhs = if matches!(rhs, Expr::Integer(0)) {
    stripped = strip_constant_factor(lhs, var);
    &stripped
  } else {
    lhs
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
  // Inner argument has to be linear in the solve variable: `a*var + b`.
  // The solutions are found for the whole argument and then mapped back
  // through `var = (arg - b)/a`.
  let (lin_a, lin_b) = if matches!(trig_arg, Expr::Identifier(s) if s == var) {
    (Expr::Integer(1), Expr::Integer(0))
  } else {
    super::apart::extract_linear_coeffs(trig_arg, var)?
  };
  if contains_var(&lin_a, var) || contains_var(&lin_b, var) {
    return None;
  }
  // A negative `a` is turned positive the way wolframscript does it, through
  // the parity of the function: `Sin[1 - x] == 1/2` is solved as
  // `Sin[-1 + x] == -1/2`, giving `1 + (7*Pi)/6 + 2*Pi*C[1]`.
  let evaluated =
    |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e);
  let flipped_rhs;
  let (lin_a, lin_b, rhs) = if try_eval_to_f64(&lin_a).is_some_and(|v| v < 0.0)
  {
    let rhs = if trig_name == "Cos" {
      rhs
    } else {
      flipped_rhs = evaluated(neg1(rhs.clone()));
      &flipped_rhs
    };
    (evaluated(neg1(lin_a)), evaluated(neg1(lin_b)), rhs)
  } else {
    (lin_a, lin_b, rhs)
  };
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
    let _c = try_eval_to_f64(rhs)?;
  }

  let var_expr = id_expr(var);
  let pi = const_expr("Pi");
  let c1 = call1("C", Expr::Integer(1));
  let element_c1_integers =
    call("Element", vec![c1.clone(), id_expr("Integers")]);
  let two_pi_c1 = times2(Expr::Integer(2), times2(pi.clone(), c1.clone()));
  let pi_c1 = times2(pi.clone(), c1.clone());
  let neg_half_pi = times2(make_rational(-1, 2), pi.clone());
  let half_pi = div2(pi.clone(), Expr::Integer(2));
  // Helper: build `ConditionalExpression[expr, Element[C[1], Integers]]`.
  let cond = |body: Expr| {
    call(
      "ConditionalExpression",
      vec![body, element_c1_integers.clone()],
    )
  };
  let make_rule_list = |bodies: Vec<Expr>| -> Expr {
    Expr::List(
      bodies
        .into_iter()
        .map(|body| {
          Expr::List(vec![rule_expr(var_expr.clone(), cond(body))].into())
        })
        .collect(),
    )
  };

  // Build "base + 2*Pi*C[1]" / "base + Pi*C[1]" expressions.

  // Evaluate an expression (simplifies e.g. ArcSin[1/2] → Pi/6).
  let eval = |e: Expr| {
    crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or_else(|_| e.clone())
  };
  let inverse = |head: &str| eval(call1(head, rhs.clone()));
  let negate = |e: Expr| eval(neg1(e));

  let solutions: Vec<Expr> = match (trig_name, rhs_special) {
    ("Sin", Some(0)) => {
      vec![two_pi_c1.clone(), plus2(pi.clone(), two_pi_c1.clone())]
    }
    ("Sin", Some(1)) => vec![plus2(half_pi.clone(), two_pi_c1.clone())],
    ("Sin", Some(-1)) => vec![plus2(neg_half_pi.clone(), two_pi_c1.clone())],
    ("Cos", Some(0)) => vec![
      plus2(neg_half_pi.clone(), two_pi_c1.clone()),
      plus2(half_pi.clone(), two_pi_c1.clone()),
    ],
    ("Cos", Some(1)) => vec![two_pi_c1.clone()],
    ("Cos", Some(-1)) => {
      // Wolframscript returns the two-solution list `{x -> -Pi + 2*Pi*C[1],
      // x -> Pi + 2*Pi*C[1]}` even though they coincide modulo 2*Pi.
      let neg_pi = neg1(pi.clone());
      vec![
        plus2(neg_pi, two_pi_c1.clone()),
        plus2(pi.clone(), two_pi_c1.clone()),
      ]
    }
    ("Tan", Some(0)) => vec![pi_c1.clone()],
    ("Cot", Some(0)) => vec![plus2(half_pi.clone(), pi_c1.clone())],
    // General numeric rhs c: x = ArcSin[c] + 2πC, Pi - ArcSin[c] + 2πC for
    // Sin; -ArcCos[c] + 2πC, ArcCos[c] + 2πC for Cos; ArcTan[c] + πC for Tan.
    ("Sin", None) => {
      let a = inverse("ArcSin");
      let arcsin_sol = plus2(a.clone(), two_pi_c1.clone());
      let pi_minus_sol = plus2(
        eval(plus2(pi.clone(), negate(a.clone()))),
        two_pi_c1.clone(),
      );
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
        plus2(negate(a.clone()), two_pi_c1.clone()),
        plus2(a, two_pi_c1.clone()),
      ]
    }
    ("Tan", None) => vec![plus2(inverse("ArcTan"), pi_c1.clone())],
    _ => return None,
  };

  let solutions = if matches!(&lin_a, Expr::Integer(1))
    && matches!(&lin_b, Expr::Integer(0))
  {
    solutions
  } else {
    // `var = (family - b)/a`, as wolframscript leaves it — not expanded:
    // `(Pi/2 + 2*Pi*C[1])/2`.
    solutions
      .into_iter()
      .map(|body| eval(div2(minus2(body, lin_b.clone()), lin_a.clone())))
      .collect()
  };

  Some(make_rule_list(solutions))
}

/// An inexact equation that is a polynomial in `Sin[u]`, `Cos[u]` or `Tan[u]`
/// of one argument `u` linear in `var`. wolframscript inverts the functions
/// instead of returning the periodic family: it solves for the function
/// values and takes principal branches — `ArcSin[s]` when only `Sin` occurs,
/// `ArcTan[t]` when only `Tan` does, and `±ArcCos[c]` when only `Cos` does.
/// Equations mixing `Sin` and `Cos` are reduced to `Cos` alone through
/// `Sin[u]^2 == 1 - Cos[u]^2` (squaring away the odd `Sin` part), and only the
/// angles `ArcTan[c, ±Sqrt[1 - c^2]]` that satisfy the original equation are
/// kept.
fn try_solve_inexact_trig(eq: &Expr, var: &str) -> Option<Expr> {
  fn collect<'a>(e: &'a Expr, var: &str, out: &mut Vec<&'a Expr>) {
    match e {
      Expr::FunctionCall { name, args }
        if args.len() == 1
          && matches!(name.as_str(), "Sin" | "Cos" | "Tan")
          && contains_var(&args[0], var) =>
      {
        out.push(e);
      }
      Expr::FunctionCall { args, .. } => {
        args.iter().for_each(|a| collect(a, var, out));
      }
      Expr::List(items) => items.iter().for_each(|a| collect(a, var, out)),
      Expr::BinaryOp { left, right, .. } => {
        collect(left, var, out);
        collect(right, var, out);
      }
      Expr::UnaryOp { operand, .. } => collect(operand, var, out),
      _ => {}
    }
  }
  if !crate::functions::math_ast::contains_inexact_real(eq) {
    return None;
  }
  let (lhs, rhs, CompOp::Equal) =
    crate::functions::polynomial_ast::reduce::extract_comparison(eq)?
  else {
    return None;
  };
  let difference = minus2(lhs, rhs);
  let mut calls = Vec::new();
  collect(&difference, var, &mut calls);
  let Expr::FunctionCall { args, .. } = calls.first()? else {
    return None;
  };
  let argument = args[0].clone();
  let mut heads = std::collections::BTreeSet::new();
  let mut substituted = difference.clone();
  for trig_call in &calls {
    let Expr::FunctionCall { name, args } = trig_call else {
      return None;
    };
    if expr_to_string(&args[0]) != expr_to_string(&argument) {
      return None;
    }
    heads.insert(name.as_str());
    let symbol = id_expr(&format!("$SolveTrig{name}"));
    substituted = substitute_expr(&substituted, trig_call, &symbol);
  }
  if contains_var(&substituted, var) {
    return None;
  }
  let (lin_a, lin_b) = if matches!(&argument, Expr::Identifier(s) if s == var) {
    (Expr::Integer(1), Expr::Integer(0))
  } else {
    super::apart::extract_linear_coeffs(&argument, var)?
  };
  if contains_var(&lin_a, var) || contains_var(&lin_b, var) {
    return None;
  }
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let n =
    |e: Expr| crate::evaluator::evaluate_function_call_ast("N", &[e]).ok();
  let heads: Vec<&str> = heads.into_iter().collect();
  // For a mixed equation, the parts of its reduced form `even + odd*Sin[u]`.
  let mut sine_parts = None;
  let (unknown, polynomial) = match heads.as_slice() {
    [head] => (id_expr(&format!("$SolveTrig{head}")), substituted.clone()),
    ["Cos", "Sin"] => {
      let s = id_expr("$SolveTrigSin");
      let c = id_expr("$SolveTrigCos");
      let circle = eval(call(
        "Plus",
        vec![
          pow2(s.clone(), Expr::Integer(2)),
          pow2(c.clone(), Expr::Integer(2)),
          Expr::Integer(-1),
        ],
      ))?;
      let remainder = eval(call(
        "PolynomialRemainder",
        vec![substituted.clone(), circle, s.clone()],
      ))?;
      let coefficient = |k: i128| {
        eval(call(
          "Coefficient",
          vec![remainder.clone(), s.clone(), Expr::Integer(k)],
        ))
      };
      let (even, odd) = (coefficient(0)?, coefficient(1)?);
      sine_parts = Some((even.clone(), odd.clone()));
      let resolvent = if matches!(odd, Expr::Integer(0)) {
        even
      } else {
        call(
          "Subtract",
          vec![
            pow2(even, Expr::Integer(2)),
            times2(
              pow2(odd, Expr::Integer(2)),
              minus2(Expr::Integer(1), pow2(c.clone(), Expr::Integer(2))),
            ),
          ],
        )
      };
      (c, eval(call1("Expand", resolvent))?)
    }
    _ => return None,
  };
  let Expr::Identifier(unknown_name) = &unknown else {
    return None;
  };
  if !matches!(
    eval(call(
      "PolynomialQ",
      vec![polynomial.clone(), unknown.clone()]
    ))?,
    Expr::Identifier(ref t) if t == "True"
  ) {
    return None;
  }
  let roots = quietly(|| {
    solve_ast(&[
      Expr::Comparison {
        operands: vec![polynomial, Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      },
      unknown.clone(),
    ])
  })
  .ok()?;
  let Expr::List(ref roots) = roots else {
    return None;
  };
  let mut candidates = Vec::new();
  for root in roots {
    let Expr::List(rules) = root else {
      return None;
    };
    let [
      Expr::Rule {
        pattern,
        replacement,
      },
    ] = rules.as_slice()
    else {
      return None;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(p) if p == unknown_name) {
      return None;
    }
    let value = replacement.as_ref().clone();
    try_extract_complex_f64(&n(value.clone())?)?;
    // The angle of the point `(Cos[u], Sin[u])`; a missing `Cos[u]` takes
    // the principal (nonnegative) root, a missing `Sin[u]` both signs.
    let other = || {
      call1(
        "Sqrt",
        minus2(Expr::Integer(1), pow2(value.clone(), Expr::Integer(2))),
      )
    };
    match heads.as_slice() {
      ["Tan"] => candidates.push(call1("ArcTan", value)),
      ["Sin"] => candidates.push(call("ArcTan", vec![other(), value.clone()])),
      ["Cos"] => {
        let arc = call1("ArcCos", value.clone());
        candidates.push(neg1(arc.clone()));
        candidates.push(arc);
      }
      _ => {
        // Where the odd part does not vanish it pins `Sin[u]` down without
        // the cancellation `1 - c^2` suffers near `c = ±1`.
        let (even, odd) = sine_parts.as_ref()?;
        let at = |e: &Expr| n(substitute_expr(e, &unknown, &value));
        let odd_value = at(odd)?;
        let sine = match try_extract_complex_f64(&odd_value) {
          Some((re, im)) if re.hypot(im) > 1e-8 => {
            eval(neg1(div2(at(even)?, odd_value)))?
          }
          _ => other(),
        };
        candidates
          .push(call("ArcTan", vec![value.clone(), neg1(sine.clone())]));
        candidates.push(call("ArcTan", vec![value.clone(), sine]));
      }
    }
  }
  let mut solutions: Vec<Expr> = Vec::new();
  let mut seen: Vec<(f64, f64)> = Vec::new();
  for candidate in candidates {
    let value =
      n(eval(div2(minus2(candidate, lin_b.clone()), lin_a.clone()))?)?;
    let point = try_extract_complex_f64(&value)?;
    let solution =
      Expr::List(vec![rule_expr(id_expr(var), value.clone())].into());
    if numerically_violates(eq, &solution)
      || seen.iter().any(|p| {
        (p.0 - point.0).abs() <= 1e-12 * (1.0 + p.0.abs())
          && (p.1 - point.1).abs() <= 1e-12 * (1.0 + p.1.abs())
      })
    {
      continue;
    }
    seen.push(point);
    solutions.push(solution);
  }
  report_inverse_function_use();
  Some(sort_nsolve_solutions(Expr::List(solutions.into())))
}

/// `Sin/Cos[a1*var + b1] == Sin/Cos[a2*var + b2]` (both sides depend on `var`
/// through a linear argument). Writing `Cos[A]` as `Sin[Pi/2 + A]`, the
/// equation `Sin[A] == Sin[B]` holds iff `A == B + 2*Pi*k` or
/// `A == Pi - B + 2*Pi*k`; each branch is linear in `var`. A branch whose
/// `var` coefficient vanishes is either empty (skipped) or an identity, in
/// which case `None` leaves the equation to the generic paths.
fn try_solve_trig_pair_eq(lhs: &Expr, rhs: &Expr, var: &str) -> Option<Expr> {
  // Normalize one side to the linear coefficients `(a, b)` of an equivalent
  // `Sin[a*var + b]`.
  let side = |e: &Expr| -> Option<(Expr, Expr)> {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if args.len() != 1 || !matches!(name.as_str(), "Sin" | "Cos") {
      return None;
    }
    let (a, b) = if matches!(&args[0], Expr::Identifier(s) if s == var) {
      (Expr::Integer(1), Expr::Integer(0))
    } else {
      super::apart::extract_linear_coeffs(&args[0], var)?
    };
    if contains_var(&a, var) || contains_var(&b, var) {
      return None;
    }
    let b = if name == "Cos" {
      plus2(div2(const_expr("Pi"), Expr::Integer(2)), b)
    } else {
      b
    };
    Some((a, b))
  };
  let (a1, b1) = side(lhs)?;
  let (a2, b2) = side(rhs)?;
  let eval = |e: Expr| {
    crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or_else(|_| e.clone())
  };
  let pi = const_expr("Pi");
  let c1 = call1("C", Expr::Integer(1));
  let two_pi_c1 = times2(Expr::Integer(2), times2(pi.clone(), c1.clone()));
  // (var coefficient, constant) of each branch: `coef*var == const + 2*Pi*k`.
  let branches = [
    (
      eval(minus2(a1.clone(), a2.clone())),
      eval(minus2(b2.clone(), b1.clone())),
    ),
    (
      eval(plus2(a1.clone(), a2.clone())),
      eval(minus2(minus2(pi.clone(), b1), b2)),
    ),
  ];
  let mut bodies: Vec<Expr> = Vec::new();
  for (coef, constant) in branches {
    if matches!(&coef, Expr::Integer(0)) {
      // `0 == constant + 2*Pi*k` is solvable for an integer `k` iff
      // `constant/(2*Pi)` is an integer, which makes the equation an identity.
      let ratio = eval(div2(constant, times2(Expr::Integer(2), pi.clone())));
      let is_int = match &ratio {
        Expr::Integer(_) => true,
        other => {
          try_eval_to_f64(other).is_some_and(|v| (v - v.round()).abs() < 1e-12)
        }
      };
      if is_int {
        return None;
      }
      continue;
    }
    if contains_var(&coef, var) {
      return None;
    }
    // `k -> -k` keeps the family intact; use it to divide by a positive
    // coefficient so the parameter enters with a positive sign.
    let (constant, coef) = if try_eval_to_f64(&coef).is_some_and(|v| v < 0.0) {
      (eval(neg1(constant)), eval(neg1(coef)))
    } else {
      (constant, coef)
    };
    // `coef*var == constant + 2*Pi*k` with an integer coef m has m families
    // of period 2*Pi, which is how wolframscript writes them: each
    // representative (constant + 2*Pi*j)/m moved into (-Pi, Pi], plus
    // 2*Pi*C[1] (`Sin[x] == Cos[x]` is -3*Pi/4 and Pi/4).
    let mut new_bodies = Vec::new();
    let pi_multiple =
      matches!(eval(div2(constant.clone(), pi.clone())), Expr::Integer(_))
        || matches!(
          eval(div2(constant.clone(), pi.clone())),
          Expr::FunctionCall { ref name, .. } if name == "Rational"
        );
    match &coef {
      Expr::Integer(m) if *m >= 1 && *m <= 64 && pi_multiple => {
        for j in 0..*m {
          let rep = eval(div2(
            plus2(constant.clone(), times2(Expr::Integer(2 * j), pi.clone())),
            Expr::Integer(*m),
          ));
          let rep = match try_eval_to_f64(&rep) {
            Some(v) => {
              let turns = ((v - std::f64::consts::PI)
                / (2.0 * std::f64::consts::PI))
                .ceil();
              if turns == 0.0 {
                rep
              } else {
                eval(minus2(
                  rep,
                  times2(Expr::Integer(turns as i128 * 2), pi.clone()),
                ))
              }
            }
            None => rep,
          };
          new_bodies.push(eval(plus2(rep, two_pi_c1.clone())));
        }
      }
      _ => new_bodies.push(eval(call1(
        "Expand",
        div2(plus2(constant, two_pi_c1.clone()), coef),
      ))),
    }
    for body in new_bodies {
      if !bodies
        .iter()
        .any(|b| expr_to_string(b) == expr_to_string(&body))
      {
        bodies.push(body);
      }
    }
  }
  // In canonical order: the bare `2*Pi*C[1]` before every `c + 2*Pi*C[1]`,
  // and those by their constant, ascending.
  let constant_of = |body: &Expr| -> f64 {
    let at_zero = crate::evaluator::pattern_matching::apply_replace_all_ast(
      body,
      &rule_expr(c1.clone(), Expr::Integer(0)),
    )
    .ok()
    .and_then(|e| crate::evaluator::evaluate_expr_to_expr(&e).ok());
    at_zero
      .and_then(|e| try_eval_to_f64(&e))
      .unwrap_or(f64::INFINITY)
  };
  let mut keyed: Vec<(bool, f64, Expr)> = bodies
    .into_iter()
    .map(|b| {
      let c = constant_of(&b);
      (c != 0.0, c, b)
    })
    .collect();
  keyed.sort_by(|x, y| {
    (x.0, x.1)
      .partial_cmp(&(y.0, y.1))
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  let bodies: Vec<Expr> = keyed.into_iter().map(|(_, _, b)| b).collect();
  let element_c1_integers = call("Element", vec![c1, id_expr("Integers")]);
  let var_expr = id_expr(var);
  Some(Expr::List(
    bodies
      .into_iter()
      .map(|body| {
        Expr::List(
          vec![rule_expr(
            var_expr.clone(),
            call(
              "ConditionalExpression",
              vec![body, element_c1_integers.clone()],
            ),
          )]
          .into(),
        )
      })
      .collect(),
  ))
}

/// Shape `try_solve_inverse_function` knows how to invert once an
/// expression is isolated as `atom == value`: a bare function call or a
/// `Power`.
fn is_invertible_atom_shape(e: &Expr) -> bool {
  matches!(
    e,
    Expr::FunctionCall { .. }
      | Expr::CurriedCall { .. }
      | Expr::BinaryOp {
        op: BinaryOperator::Power,
        ..
      }
  )
}

/// Factor a single additive term into `(constant coefficient, invertible
/// atom)` when exactly one of its multiplicative factors depends on `var`
/// and that factor has an invertible-function shape (`Power[..]` or a
/// function call). Returns `None` when `var` appears in more than one
/// factor of the term or in a shape that isn't invertible — nothing to
/// isolate there.
fn factor_term_for_invertible_atom(
  term: &Expr,
  var: &str,
) -> Option<(Expr, Expr)> {
  let (sign, inner) = match term {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (Expr::Integer(-1), operand.as_ref().clone()),
    _ => (Expr::Integer(1), term.clone()),
  };
  let mut const_factors = vec![sign];
  let mut atom: Option<Expr> = None;
  for f in collect_multiplicative_factors(&inner) {
    if is_constant_wrt(&f, var) {
      const_factors.push(f);
    } else if atom.is_none() && is_invertible_atom_shape(&f) {
      atom = Some(f);
    } else {
      return None;
    }
  }
  let atom = atom?;
  let k = const_factors
    .into_iter()
    .reduce(|a, b| multiply_exprs(&a, &b))
    .unwrap_or(Expr::Integer(1));
  Some((k, atom))
}

/// `expanded == 0` where `expanded` sums exactly one invertible-function
/// atom (an exponential like `b^f(var)`, a `Log`, `Sqrt`, trig call, …)
/// against terms constant w.r.t. `var` looks like a degree-0 polynomial to
/// `max_power_int`: a constant base makes `base^f(var)` register as degree
/// 0, so the coefficient extraction below silently drops the atom term and
/// Solve reports no solutions instead of solving `atom == -c/k`. Detect
/// that shape directly and delegate to the same invertible-function/trig
/// solvers used for the bare `atom == const` case.
fn try_solve_isolated_invertible_atom(
  expanded: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  let terms = collect_additive_terms(expanded);
  let mut const_sum: Vec<Expr> = Vec::new();
  let mut atom_term: Option<(Expr, Expr)> = None;
  for term in &terms {
    if is_constant_wrt(term, var) {
      const_sum.push(term.clone());
      continue;
    }
    let (k, atom) = factor_term_for_invertible_atom(term, var)?;
    if atom_term.is_some() {
      // More than one term depends on `var` — outside this fallback's scope.
      return None;
    }
    atom_term = Some((k, atom));
  }
  let (k, atom) = atom_term?;
  let c = const_sum
    .into_iter()
    .reduce(plus2)
    .unwrap_or(Expr::Integer(0));
  // atom == -c/k
  let target =
    crate::evaluator::evaluate_expr_to_expr(&div2(neg1(c), k)).ok()?;
  let synthetic_eq = Expr::Comparison {
    operands: vec![atom, target],
    operators: vec![ComparisonOp::Equal],
  };
  if let Some(result) = try_solve_inverse_function(&synthetic_eq, var) {
    return Some(result);
  }
  if let Some(result) = try_solve_trig_eq(&synthetic_eq, var) {
    return Some(Ok(result));
  }
  None
}

/// Over the reals, a family `v + (i·p) C[k]` with a purely imaginary period
/// (the logarithms of `E^x == r`) has only its `C[k] = 0` member:
/// `Solve[E^(5 b) == 2, b, Reals]` is `{{b -> Log[2]/5}}`. A family with a
/// real period (`2 Pi C[1]` of a trigonometric equation) stays a family.
fn real_member_of_imaginary_family(solution: &Expr) -> Expr {
  let Expr::List(rules) = solution else {
    return solution.clone();
  };
  let member = |value: &Expr| -> Option<Expr> {
    let Expr::FunctionCall { name, args } = value else {
      return None;
    };
    if name != "ConditionalExpression" || args.len() != 2 {
      return None;
    }
    let Expr::FunctionCall {
      name: element,
      args: element_args,
    } = &args[1]
    else {
      return None;
    };
    if element != "Element" || element_args.len() != 2 {
      return None;
    }
    let parameter = &element_args[0];
    let at = |k: i128| {
      crate::evaluator::evaluate_expr_to_expr(&substitute_expr(
        &args[0],
        parameter,
        &Expr::Integer(k),
      ))
      .ok()
    };
    let (zero, one) = (at(0)?, at(1)?);
    let period = crate::evaluator::evaluate_function_call_ast(
      "N",
      &[minus2(one, zero.clone())],
    )
    .ok()?;
    let (re, im) = try_extract_complex_f64(&period)?;
    (re == 0.0 && im != 0.0).then_some(zero)
  };
  Expr::List(
    rules
      .iter()
      .map(|rule| match rule {
        Expr::Rule {
          pattern,
          replacement,
        } => match member(replacement) {
          Some(value) => rule_expr(pattern.as_ref().clone(), value),
          None => rule.clone(),
        },
        other => other.clone(),
      })
      .collect(),
  )
}

/// `n` as `r^m` with the smallest base `r`: `8` is `2^3`, `12` is `12^1`.
fn smallest_integer_root(n: i128) -> (Expr, i128) {
  for m in (2..=127).rev() {
    let r = (n as f64).powf(1.0 / m as f64).round() as i128;
    for candidate in [r - 1, r, r + 1] {
      if candidate >= 2
        && candidate
          .checked_pow(m as u32)
          .is_some_and(|power| power == n)
      {
        return (Expr::Integer(candidate), m);
      }
    }
  }
  (Expr::Integer(n), 1)
}

/// Solve `Σ c_i B^(k_i var + d_i) == 0` with a constant base `B` and
/// rational `k_i`: a polynomial in `t = B^(g var)`, `g` the gcd of the
/// `k_i`. Each nonzero root `r` of the polynomial gives the solutions of
/// `B^(g var) == r`:
///   Solve[2^b + 4^b == 6, b]
///     → 1 + (2 I Pi C[1])/Log[2], (2 I Pi C[1])/Log[2] + (I Pi + Log[3])/Log[2]
fn try_solve_exponential_polynomial(
  expanded: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let rational = |e: &Expr| -> Option<(i128, i128)> {
    match e {
      Expr::Integer(n) => Some((*n, 1)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) => Some((*p, *q)),
          _ => None,
        }
      }
      _ => None,
    }
  };
  let mut base: Option<Expr> = None;
  // (coefficient, exponent slope as p/q) per term.
  let mut terms: Vec<(Expr, (i128, i128))> = Vec::new();
  for term in collect_additive_terms(expanded) {
    let mut coefficient = Vec::new();
    let mut slope = (0i128, 1i128);
    let factors = match &term {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => std::iter::once(Expr::Integer(-1))
        .chain(extract_times_factors(operand))
        .collect(),
      _ => extract_times_factors(&term),
    };
    for factor in factors {
      if is_constant_wrt(&factor, var) {
        coefficient.push(factor);
        continue;
      }
      let (b, exponent) = match &factor {
        Expr::FunctionCall { name, args }
          if name == "Power" && args.len() == 2 =>
        {
          (args[0].clone(), args[1].clone())
        }
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => (left.as_ref().clone(), right.as_ref().clone()),
        _ => return None,
      };
      if !is_constant_wrt(&b, var) {
        return None;
      }
      // An integer base is written as a power of its smallest root, so
      // `2^b + 4^b` is a polynomial in `2^b`.
      let (root, multiplicity) = match &b {
        Expr::Integer(n) if *n >= 2 => smallest_integer_root(*n),
        _ => (b.clone(), 1),
      };
      match &base {
        Some(known)
          if !crate::evaluator::pattern_matching::expr_equal(known, &root) =>
        {
          return None;
        }
        Some(_) => {}
        None => base = Some(root.clone()),
      }
      let k = eval(call("D", vec![exponent.clone(), id_expr(var)]))?;
      let (p, q) = rational(&k)?;
      let p = p * multiplicity;
      let offset =
        eval(substitute_expr(&exponent, &id_expr(var), &Expr::Integer(0)))?;
      // The exponent has to be affine in `var`.
      let rebuilt = eval(minus2(
        exponent,
        plus2(times2(k.clone(), id_expr(var)), offset.clone()),
      ))?;
      if !matches!(rebuilt, Expr::Integer(0)) {
        return None;
      }
      if !matches!(offset, Expr::Integer(0)) {
        coefficient.push(pow2(b, offset));
      }
      slope = (slope.0 * q + p * slope.1, slope.1 * q);
    }
    terms.push((times(coefficient), slope));
  }
  let base = base?;
  // Common denominator, then the gcd of the numerators.
  let gcd = |mut a: i128, mut b: i128| {
    while b != 0 {
      (a, b) = (b, a % b);
    }
    a.abs()
  };
  let lcd = terms
    .iter()
    .fold(1i128, |l, (_, (_, q))| l / gcd(l, *q) * q.abs());
  let scaled: Vec<i128> =
    terms.iter().map(|(_, (p, q))| p * (lcd / q)).collect();
  let g = scaled.iter().fold(0i128, |acc, k| gcd(acc, *k));
  let distinct: std::collections::BTreeSet<i128> =
    scaled.iter().filter(|k| **k != 0).copied().collect();
  if g == 0 || distinct.len() < 2 {
    return None;
  }
  let powers: Vec<i128> = scaled.iter().map(|k| k / g).collect();
  let shift = powers.iter().copied().min().unwrap_or(0).min(0);
  let t = id_expr("Woxi`ExponentialBase");
  let polynomial = plus(
    terms
      .iter()
      .zip(&powers)
      .map(|((c, _), e)| {
        times2(c.clone(), pow2(t.clone(), Expr::Integer(e - shift)))
      })
      .collect(),
  );
  let polynomial = eval(polynomial)?;
  let roots = solve_ast(&[
    Expr::Comparison {
      operands: vec![polynomial, Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    },
    t.clone(),
  ]);
  let Ok(Expr::List(ref roots)) = roots else {
    return None;
  };
  let slope = eval(div2(Expr::Integer(g), Expr::Integer(lcd)))?;
  let power = pow2(base, times2(slope, id_expr(var)));
  let mut solutions: Vec<Expr> = Vec::new();
  // `None` gives up on the equation; `Some(Err(…))` is a failed sub-solve.
  let done = with_single_ifun(|| -> Option<Result<(), InterpreterError>> {
    for root in roots {
      let Expr::List(rules) = root else { return None };
      let [Expr::Rule { replacement, .. }] = &rules[..] else {
        return None;
      };
      if matches!(replacement.as_ref(), Expr::Integer(0)) {
        continue;
      }
      let solved = match solve_ast(&[
        Expr::Comparison {
          operands: vec![power.clone(), replacement.as_ref().clone()],
          operators: vec![ComparisonOp::Equal],
        },
        id_expr(var),
      ]) {
        Ok(found) => found,
        Err(e) => return Some(Err(e)),
      };
      let Expr::List(ref solved) = solved else {
        return None;
      };
      for solution in solved {
        if !solutions
          .iter()
          .any(|s| crate::evaluator::pattern_matching::expr_equal(s, solution))
        {
          solutions.push(solution.clone());
        }
      }
    }
    Some(Ok(()))
  });
  if let Err(e) = done? {
    return Some(Err(e));
  }
  // wolframscript lists the families in canonical order.
  Some(crate::evaluator::evaluate_function_call_ast(
    "Sort",
    &[Expr::List(solutions.into())],
  ))
}

/// Solve `g == 0` for `var` with the Lambert W function when `g` is
/// `a x^p E^(b x + c) + d` (p a positive integer) or
/// `β E^(b x + c) + α x + d`, the shapes wolframscript answers with
/// `ProductLog`. Every root gets the principal branch, and a real argument
/// in (-1/e, 0) the `-1` branch as well. Machine-real input gives numbers.
fn try_solve_lambert(
  g: &Expr,
  var: &str,
) -> Result<Option<Expr>, InterpreterError> {
  fn has_var_exponential(e: &Expr, var: &str) -> bool {
    let here = match e {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => !contains_var(left, var) && contains_var(right, var),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        !contains_var(&args[0], var) && contains_var(&args[1], var)
      }
      Expr::FunctionCall { name, args } if name == "Exp" && args.len() == 1 => {
        contains_var(&args[0], var)
      }
      _ => false,
    };
    here
      || crate::syntax::expr_children(e)
        .into_iter()
        .any(|c| has_var_exponential(c, var))
  }
  if !has_var_exponential(g, var) {
    return Ok(None);
  }
  let ev = crate::evaluator::evaluate_expr_to_expr;
  let x = id_expr(var);
  let free = |e: &Expr| !contains_var(e, var);
  let expanded = ev(&call("Expand", vec![g.clone()]))?;
  let terms: Vec<Expr> = match ev(&call(
    "Apply",
    vec![id_expr("List"), expanded.clone()],
  ))? {
    Expr::List(ref items)
      if matches!(&expanded, Expr::FunctionCall { name, .. } if name == "Plus")
        || matches!(
          &expanded,
          Expr::BinaryOp {
            op: BinaryOperator::Plus | BinaryOperator::Minus,
            ..
          }
        ) =>
    {
      items.to_vec()
    }
    _ => vec![expanded.clone()],
  };
  let (var_terms, const_terms): (Vec<Expr>, Vec<Expr>) =
    terms.into_iter().partition(|t| !free(t));
  let d = ev(&plus(const_terms))?;

  // `t` as `a x^p E^(b x + c)`: (a, p, b, c).
  let split_exp_term =
    |t: &Expr| -> Result<Option<(Expr, i128, Expr, Expr)>, InterpreterError> {
      let factors: Vec<Expr> =
        match ev(&call("Apply", vec![id_expr("List"), t.clone()]))? {
          Expr::List(ref items)
            if matches!(t, Expr::FunctionCall { name, .. } if name == "Times")
              || matches!(
                t,
                Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  ..
                }
              ) =>
          {
            items.to_vec()
          }
          _ => vec![t.clone()],
        };
      let mut exponent: Option<Expr> = None;
      let mut rest: Vec<Expr> = Vec::new();
      for f in factors {
        // `b^e` is `E^(e Log[b])`; the exponent is that of base E.
        let in_base_e = |base: &Expr, exponent: &Expr| -> Option<Expr> {
          if contains_var(base, var) || !contains_var(exponent, var) {
            return None;
          }
          if matches!(base, Expr::Constant(c) | Expr::Identifier(c) if c == "E")
          {
            Some(exponent.clone())
          } else {
            Some(times2(exponent.clone(), call1("Log", base.clone())))
          }
        };
        let e = match &f {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => in_base_e(left, right),
          Expr::FunctionCall { name, args }
            if name == "Power" && args.len() == 2 =>
          {
            in_base_e(&args[0], &args[1])
          }
          _ => None,
        };
        match e {
          Some(e) if exponent.is_none() => exponent = Some(e),
          Some(_) => return Ok(None),
          None => rest.push(f),
        }
      }
      let Some(e) = exponent else { return Ok(None) };
      let b = ev(&call(
        "Coefficient",
        vec![e.clone(), x.clone(), Expr::Integer(1)],
      ))?;
      let c = ev(&call(
        "Expand",
        vec![plus(vec![
          e.clone(),
          times(vec![Expr::Integer(-1), b.clone(), x.clone()]),
        ])],
      ))?;
      if !free(&b) || !free(&c) || matches!(b, Expr::Integer(0)) {
        return Ok(None);
      }
      let rest = ev(&times(rest))?;
      let p = match ev(&call("Exponent", vec![rest.clone(), x.clone()]))? {
        Expr::Integer(p) if p >= 0 => p,
        _ => return Ok(None),
      };
      let a = ev(&times(vec![rest, pow(x.clone(), Expr::Integer(-p))]))?;
      if !free(&a) {
        return Ok(None);
      }
      Ok(Some((a, p, b, c)))
    };

  // The roots `(scale, z, shift)`: solutions `scale W(z) + shift`.
  let mut roots: Vec<(Expr, Expr, Expr)> = Vec::new();
  match var_terms.as_slice() {
    // a x^p E^(b x + c) = -d  ⇒  x E^(b x / p) = r, r^p = -d / (a E^c)
    //   ⇒  x = (p/b) W((b/p) r)
    [t] => {
      let Some((a, p, b, c)) = split_exp_term(t)? else {
        return Ok(None);
      };
      if p == 0 {
        return Ok(None);
      }
      let k = ev(&times(vec![
        Expr::Integer(-1),
        d.clone(),
        pow(a, Expr::Integer(-1)),
        pow(const_expr("E"), neg(c)),
      ]))?;
      let rs: Vec<Expr> = if p == 1 {
        vec![k]
      } else {
        let y = id_expr("Solve$lambertRoot");
        match ev(&call(
          "Solve",
          vec![
            Expr::Comparison {
              operands: vec![pow(y.clone(), Expr::Integer(p)), k],
              operators: vec![ComparisonOp::Equal],
            },
            y.clone(),
          ],
        ))? {
          Expr::List(ref sols) => sols
            .iter()
            .filter_map(|s| match s {
              Expr::List(rules) => rules.first().and_then(|r| match r {
                Expr::Rule { replacement, .. } => {
                  Some(replacement.as_ref().clone())
                }
                _ => None,
              }),
              _ => None,
            })
            .collect(),
          _ => return Ok(None),
        }
      };
      let pb = ev(&times(vec![
        Expr::Integer(p),
        pow(b.clone(), Expr::Integer(-1)),
      ]))?;
      for r in rs {
        let z = ev(&times(vec![
          b.clone(),
          pow(Expr::Integer(p), Expr::Integer(-1)),
          r,
        ]))?;
        roots.push((pb.clone(), z, Expr::Integer(0)));
      }
    }
    // β E^(b x + c) + α x + d = 0. With y = x + d/α:
    //   y E^(-b y) = -(β/α) E^(c - b d/α) = k  ⇒  y = -(1/b) W(-b k)
    [t1, t2] => {
      let (exp_term, lin_term) =
        match (split_exp_term(t1)?, split_exp_term(t2)?) {
          (Some(e), None) => (e, t2),
          (None, Some(e)) => (e, t1),
          _ => return Ok(None),
        };
      let (beta, p, b, c) = exp_term;
      if p != 0 {
        return Ok(None);
      }
      let alpha = ev(&call(
        "Coefficient",
        vec![lin_term.clone(), x.clone(), Expr::Integer(1)],
      ))?;
      let check = ev(&call(
        "Expand",
        vec![plus(vec![
          lin_term.clone(),
          times(vec![Expr::Integer(-1), alpha.clone(), x.clone()]),
        ])],
      ))?;
      if !free(&alpha)
        || matches!(alpha, Expr::Integer(0))
        || !matches!(check, Expr::Integer(0))
      {
        return Ok(None);
      }
      let shift = ev(&times(vec![
        Expr::Integer(-1),
        d.clone(),
        pow(alpha.clone(), Expr::Integer(-1)),
      ]))?;
      let k = ev(&times(vec![
        Expr::Integer(-1),
        beta,
        pow(alpha, Expr::Integer(-1)),
        call(
          "Power",
          vec![
            const_expr("E"),
            plus(vec![c, times(vec![b.clone(), shift.clone()])]),
          ],
        ),
      ]))?;
      let z = ev(&times(vec![Expr::Integer(-1), b.clone(), k]))?;
      let scale =
        ev(&times(vec![Expr::Integer(-1), pow(b, Expr::Integer(-1))]))?;
      roots.push((scale, z, shift));
    }
    _ => return Ok(None),
  }

  let inexact =
    crate::functions::math_ast::numeric_utils::contains_inexact_real(g);
  let inv_e = -1.0 / std::f64::consts::E;
  let mut solutions: Vec<Expr> = Vec::new();
  for (scale, z, shift) in roots {
    let mut branches = vec![call("ProductLog", vec![z.clone()])];
    if let Some(zv) = try_eval_to_f64(&z)
      && zv > inv_e
      && zv < 0.0
    {
      branches.push(call("ProductLog", vec![Expr::Integer(-1), z.clone()]));
    }
    for w in branches {
      let mut sol =
        ev(&plus(vec![times(vec![scale.clone(), w]), shift.clone()]))?;
      if !matches!(shift, Expr::Integer(0)) {
        sol = ev(&call("Together", vec![sol]))?;
      }
      if inexact {
        sol = ev(&call("N", vec![sol]))?;
      }
      solutions.push(Expr::List(vec![rule_expr(x.clone(), sol)].into()));
    }
  }
  report_inverse_function_use();
  Ok(Some(Expr::List(solutions.into())))
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

  // Try both orientations: f[expr] == val and val == f[expr]
  let (func_call, val) = if is_invertible_atom_shape(&lhs)
    && is_constant_wrt(&rhs, var)
    && !is_constant_wrt(&lhs, var)
  {
    (&lhs, &rhs)
  } else if is_invertible_atom_shape(&rhs)
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
      // Skip if exponent is an integer — the polynomial solver handles
      // those (a negative one after clearing the denominator) and gives all
      // roots, not just the principal one: `1/x^2 == 4` is `x == ±1/2`.
      if matches!(&exp, Expr::Integer(n) if *n != 0) {
        return None;
      }
      // base^exp == val where exp is constant (non-integer), base contains var
      // → base == val^(1/exp)
      let inverse_exp = div2(Expr::Integer(1), exp.clone());
      let inverse_rhs = pow2(val.clone(), inverse_exp);
      let simplified_rhs =
        crate::evaluator::evaluate_expr_to_expr(&inverse_rhs).ok()?;
      let new_eq = Expr::Comparison {
        operands: vec![base, simplified_rhs],
        operators: vec![ComparisonOp::Equal],
      };
      let solved = solve_ast(&[new_eq, id_expr(var)]);
      // An even root takes only its principal value, so undoing it can
      // produce an answer the equation itself does not have.
      return Some(match rational_parts(&exp) {
        Some((_, denominator)) if denominator % 2 == 0 => {
          keep_solutions_satisfying(solved, eq, var)
        }
        _ => solved,
      });
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
      // With inexact numbers wolframscript inverts the function instead:
      // only the principal value, reported with `ifun`
      // (`Solve[E^x == 2., x]` → `{{x -> 0.693…}}`).
      let inexact =
        crate::functions::math_ast::numeric_utils::contains_inexact_real(eq);
      let base_gets_period = matches!(&base, Expr::Constant(n) if n == "E")
        || base_is_real_gt_one(&base);
      if base_gets_period && !inexact {
        // Log[base] (evaluates to 1 for E).
        let log_base =
          crate::evaluator::evaluate_expr_to_expr(&call1("Log", base.clone()))
            .ok()?;
        // Principal part: Log[val] / Log[base].
        let principal = crate::evaluator::evaluate_expr_to_expr(&div2(
          call1("Log", val.clone()),
          log_base.clone(),
        ))
        .ok()?;
        // Periodic part: (2*Pi*I*C[1]) / Log[base].
        let c1 = call1("C", Expr::Integer(1));
        let periodic = crate::evaluator::evaluate_expr_to_expr(&div2(
          times(vec![
            Expr::Integer(2),
            id_expr("I"),
            id_expr("Pi"),
            c1.clone(),
          ]),
          log_base,
        ))
        .ok()?;
        // A zero principal (val == 1) drops out entirely, matching
        // `Solve[E^x == 1, x] -> 2*I*Pi*C[1]`.
        let principal_is_zero = matches!(&principal, Expr::Integer(0))
          || matches!(&principal, Expr::Real(r) if *r == 0.0);
        let condition = call("Element", vec![c1, id_expr("Integers")]);
        let conditional = |value: Expr| {
          call("ConditionalExpression", vec![value, condition.clone()])
        };
        let family = |periodic: Expr| {
          if principal_is_zero {
            periodic
          } else {
            let sum = plus(vec![periodic, principal.clone()]);
            crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum)
          }
        };
        if matches!(&exp, Expr::Identifier(n) if n == var) {
          return Some(Ok(Expr::List(
            vec![Expr::List(
              vec![rule_expr(id_expr(var), conditional(family(periodic)))]
                .into(),
            )]
            .into(),
          )));
        }
        // An exponent linear in the variable equals that whole family:
        // solve `exp == t` for a placeholder `t` and put the family in its
        // place, so the family stays a sum of its own:
        //   Solve[2^(x - 2) == 15, x]
        //     → 2 + (2*I*Pi*C[1])/Log[2] + Log[15]/Log[2]
        // `C[1]` ranges over all integers, so a negative slope takes the
        // family with `-C[1]`, keeping the periodic term positive:
        //   Solve[3^(1 - x) == 5, x]
        //     → 1 + (2*I*Pi*C[1])/Log[3] - Log[5]/Log[3]
        let slope = crate::evaluator::evaluate_expr_to_expr(&call(
          "D",
          vec![exp.clone(), id_expr(var)],
        ))
        .ok()?;
        if is_constant_wrt(&slope, var) && !matches!(slope, Expr::Integer(0)) {
          let placeholder = id_expr("$SolvePeriodicFamily");
          let new_eq = Expr::Comparison {
            operands: vec![exp.clone(), placeholder.clone()],
            operators: vec![ComparisonOp::Equal],
          };
          let solved = solve_ast(&[new_eq, id_expr(var)]).ok()?;
          if let Expr::List(solutions) = &solved
            && let [solution] = &solutions[..]
            && let Expr::List(rules) = solution
            && let [Expr::Rule { replacement, .. }] = &rules[..]
          {
            let periodic =
              if crate::functions::math_ast::complex::negate_if_negative(&slope)
                .is_some()
              {
                crate::evaluator::evaluate_expr_to_expr(&neg(periodic)).ok()?
              } else {
                periodic
              };
            let value = crate::evaluator::evaluate_expr_to_expr(
              &substitute_expr(replacement, &placeholder, &family(periodic)),
            )
            .ok()?;
            return Some(Ok(Expr::List(
              vec![Expr::List(
                vec![rule_expr(id_expr(var), conditional(value))].into(),
              )]
              .into(),
            )));
          }
        }
      }
      // base^exp == val where base is constant, exp contains var
      // → exp == Log[val] / Log[base]. A machine value goes through the
      // two-argument `Log`, which rounds the way wolframscript's does
      // (`Solve[2^(x - 2) == 15., x]` → `5.906890595608519`).
      let inverse_rhs = if inexact {
        call("Log", vec![base, val.clone()])
      } else {
        div2(call1("Log", val.clone()), call1("Log", base))
      };
      let simplified_rhs =
        crate::evaluator::evaluate_expr_to_expr(&inverse_rhs).ok()?;
      let new_eq = Expr::Comparison {
        operands: vec![exp, simplified_rhs],
        operators: vec![ComparisonOp::Equal],
      };
      let solved = solve_ast(&[new_eq, id_expr(var)]);
      if inexact && base_gets_period && solved.is_ok() {
        report_inverse_function_use();
      }
      return Some(solved);
    }
  }

  if let Some((inner, inverse_value)) =
    undefined_function_inverse(func_call, val, var)
  {
    let solved = solve_ast(&[
      Expr::Comparison {
        operands: vec![inner, inverse_value],
        operators: vec![ComparisonOp::Equal],
      },
      id_expr(var),
    ]);
    if solved.is_ok() {
      report_inverse_function_use();
    }
    return Some(solved);
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
        pow2(const_expr("E"), val.clone())
      }
      "Sqrt" => {
        // Sqrt[inner] == val → inner == val^2
        pow2(val.clone(), Expr::Integer(2))
      }
      "Exp" => {
        // Exp[inner] == val → inner == Log[val]
        call1("Log", val.clone())
      }
      "ArcSin" => {
        // ArcSin[inner] == val → inner == Sin[val]
        call1("Sin", val.clone())
      }
      "ArcCos" => {
        // ArcCos[inner] == val → inner == Cos[val]
        call1("Cos", val.clone())
      }
      "ArcTan" => {
        // ArcTan[inner] == val → inner == Tan[val]
        call1("Tan", val.clone())
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
        let val_deg = times2(val.clone(), const_expr("Degree"));
        call1(inverse_name, val_deg)
      }
      "Log10" => {
        // Log10[inner] == val → inner == 10^val
        pow2(Expr::Integer(10), val.clone())
      }
      "Log2" => {
        // Log2[inner] == val → inner == 2^val
        pow2(Expr::Integer(2), val.clone())
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
    let solved = solve_ast(&[new_eq, id_expr(var)]);
    // `Sqrt` is never negative, so squaring the equation away can produce
    // an answer the equation itself does not have.
    Some(if name == "Sqrt" {
      keep_solutions_satisfying(solved, eq, var)
    } else {
      solved
    })
  } else {
    None
  }
}

/// An equation that involves `var` only through one call of an undefined
/// function, polynomially (`f[x]^2 == 4`): solved for the call's values
/// first, each of which is then inverted as in `undefined_function_inverse`.
fn try_solve_through_undefined_function(
  eq: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  fn atoms<'a>(e: &'a Expr, var: &str, out: &mut Vec<&'a Expr>) {
    if is_constant_wrt(e, var) {
      return;
    }
    match e {
      Expr::FunctionCall { name, .. }
        if !crate::evaluator::functions::is_builtin_symbol(name) =>
      {
        out.push(e);
      }
      Expr::CurriedCall { .. } => out.push(e),
      Expr::FunctionCall { args, .. } => {
        args.iter().for_each(|a| atoms(a, var, out));
      }
      Expr::List(items) => items.iter().for_each(|a| atoms(a, var, out)),
      Expr::BinaryOp { left, right, .. } => {
        atoms(left, var, out);
        atoms(right, var, out);
      }
      Expr::UnaryOp { operand, .. } => atoms(operand, var, out),
      Expr::Comparison { operands, .. } => {
        for a in operands {
          atoms(a, var, out);
        }
      }
      _ => {}
    }
  }
  let mut found = Vec::new();
  atoms(eq, var, &mut found);
  let atom = (*found.first()?).clone();
  if found
    .iter()
    .any(|a| expr_to_string(a) != expr_to_string(&atom))
  {
    return None;
  }
  let placeholder = id_expr("$SolveUndefinedCall");
  let substituted = substitute_expr(eq, &atom, &placeholder);
  if contains_var(&substituted, var) {
    return None;
  }
  let (lhs, rhs, CompOp::Equal) =
    crate::functions::polynomial_ast::reduce::extract_comparison(&substituted)?
  else {
    return None;
  };
  let is_polynomial = crate::evaluator::evaluate_expr_to_expr(&call(
    "PolynomialQ",
    vec![minus2(lhs, rhs), placeholder.clone()],
  ))
  .ok()?;
  if !matches!(is_polynomial, Expr::Identifier(ref t) if t == "True") {
    return None;
  }
  let values =
    quietly(|| solve_ast(&[substituted, placeholder.clone()])).ok()?;
  let Expr::List(ref values) = values else {
    return None;
  };
  let mut solutions = Vec::new();
  for value in values {
    let Expr::List(rules) = value else {
      return None;
    };
    let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
      return None;
    };
    let (inner, inverse_value) =
      undefined_function_inverse(&atom, replacement, var)?;
    let solved = quietly(|| {
      solve_ast(&[
        Expr::Comparison {
          operands: vec![inner, inverse_value],
          operators: vec![ComparisonOp::Equal],
        },
        id_expr(var),
      ])
    })
    .ok()?;
    let Expr::List(ref found) = solved else {
      return None;
    };
    solutions.extend(found.iter().cloned());
  }
  report_inverse_function_use();
  Some(Ok(Expr::List(solutions.into())))
}

/// `f[a1, …, ak, …, an] == val` for a function `f` with no definition (a
/// symbol that is not built in, or a compound head like `F[4]` built on one),
/// where only the argument `ak` involves `var`: that argument equals
/// `InverseFunction[f, k, n][a1, …, val, …, an]`, `val` taking its place.
fn undefined_function_inverse(
  func_call: &Expr,
  val: &Expr,
  var: &str,
) -> Option<(Expr, Expr)> {
  fn undefined_head(head: &Expr, var: &str) -> bool {
    match head {
      Expr::Identifier(name) => {
        !crate::evaluator::functions::is_builtin_symbol(name)
      }
      Expr::FunctionCall { name, args } => {
        !crate::evaluator::functions::is_builtin_symbol(name)
          && args.iter().all(|a| is_constant_wrt(a, var))
      }
      Expr::CurriedCall { func, args } => {
        undefined_head(func, var)
          && args.iter().all(|a| is_constant_wrt(a, var))
      }
      _ => false,
    }
  }
  let (head, args) = match func_call {
    Expr::FunctionCall { name, args } => (id_expr(name), args.as_slice()),
    Expr::CurriedCall { func, args } => {
      (func.as_ref().clone(), args.as_slice())
    }
    _ => return None,
  };
  if !undefined_head(&head, var) {
    return None;
  }
  let mut dependent = args
    .iter()
    .enumerate()
    .filter(|(_, a)| !is_constant_wrt(a, var));
  let (k, inner) = dependent.next()?;
  if dependent.next().is_some() {
    return None;
  }
  let mut inverse_args = args.to_vec();
  inverse_args[k] = val.clone();
  let inverse = call(
    "InverseFunction",
    vec![
      head,
      Expr::Integer(k as i128 + 1),
      Expr::Integer(args.len() as i128),
    ],
  );
  Some((
    inner.clone(),
    Expr::CurriedCall {
      func: Box::new(inverse),
      args: inverse_args,
    },
  ))
}

/// Drop the solutions that do not satisfy `equation`. A solution that
/// cannot be checked is kept: it would have to be dropped on a suspicion.
fn keep_solutions_satisfying(
  solved: Result<Expr, InterpreterError>,
  equation: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let solved = solved?;
  let Expr::List(ref solutions) = solved else {
    return Ok(solved);
  };
  let mut kept: Vec<Expr> = Vec::new();
  let mut undecided = false;
  for solution in solutions {
    let value = match solution {
      Expr::List(rules) => match rules.as_slice() {
        [Expr::Rule { replacement, .. }] => Some(replacement.as_ref()),
        _ => None,
      },
      _ => None,
    };
    if let Some(value) = value {
      match equation_holds_at(equation, var, value) {
        Some(true) => {}
        Some(false) => continue,
        // With other symbols in the equation the check cannot decide; it
        // is decided at sample values of them instead. A candidate that
        // holds for only some (`y -> x^2` of `Sqrt[y] == x`) is kept with
        // the warning wolframscript gives for it.
        None => match holds_generically(equation, var, value) {
          Some(GenericValidity::Fails) => continue,
          Some(GenericValidity::Always) => {}
          _ => undecided = true,
        },
      }
    }
    if !kept
      .iter()
      .any(|k| crate::evaluator::pattern_matching::expr_equal(k, solution))
    {
      kept.push(solution.clone());
    }
  }
  if undecided {
    report_nongen();
  }
  Ok(Expr::List(kept.into()))
}

/// Solve an equation that takes a root of the unknown.
///
/// One radical term is put on its own and both sides are raised to that
/// root's index, which clears it; what is left is solved as usual and each
/// answer is put back into the equation it came from. That last step is not
/// a formality — raising to a power throws away which branch of the root was
/// meant, so `Sqrt[2 x + 3] == x` picks up an `x == -1` that solves the
/// squared equation and not the original one. Anything that cannot be
/// checked that way is left to the caller rather than reported on trust.
fn try_solve_radical_equation(
  poly: &Expr,
  equation: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  if has_inequality(equation) {
    return None;
  }
  let (_, _, CompOp::Equal) =
    crate::functions::polynomial_ast::reduce::extract_comparison(equation)?
  else {
    return None;
  };
  let terms = collect_additive_terms(&expand_and_combine(poly));
  let radical_terms = terms
    .iter()
    .filter(|term| radical_index(term, var).is_some())
    .count();
  if radical_terms == 0 || terms.len() < 2 {
    return None;
  }
  let isolated = terms
    .iter()
    .position(|term| radical_index(term, var).is_some())?;
  let index = radical_index(&terms[isolated], var)?;
  if index > MAX_RADICAL_INDEX {
    return None;
  }

  let rest = build_sum(
    terms
      .iter()
      .enumerate()
      .filter(|(position, _)| *position != isolated)
      .map(|(_, term)| negate_expr(term))
      .collect(),
  );
  let raise = |base: &Expr| {
    crate::evaluator::evaluate_expr_to_expr(&pow2(
      base.clone(),
      Expr::Integer(index),
    ))
    .ok()
  };
  let cleared =
    expand_and_combine(&minus2(raise(&terms[isolated])?, raise(&rest)?));
  // Raising has to leave fewer roots behind than it found, or the equation
  // would come straight back here and never settle.
  let left_over = collect_additive_terms(&cleared)
    .iter()
    .filter(|term| radical_index(term, var).is_some())
    .count();
  if left_over >= radical_terms {
    return None;
  }

  let cleared_equation = Expr::Comparison {
    operands: vec![cleared, Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };
  let solved = solve_ast(&[cleared_equation, id_expr(var)]).ok()?;
  let Expr::List(ref candidates) = solved else {
    return None;
  };
  if !candidates.iter().all(|c| {
    matches!(c, Expr::List(rules) if matches!(rules.as_slice(), [Expr::Rule { .. }]))
  }) {
    return None;
  }
  Some(keep_solutions_satisfying(Ok(solved), equation, var))
}

/// `p^(n/2)` with odd `n`, in any of its spellings, as `(p, n)`.
fn as_half_power(expr: &Expr) -> Option<(&Expr, i128)> {
  let half = |e: &Expr| match e {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(2)) if n % 2 != 0 => Some(*n),
        _ => None,
      }
    }
    _ => None,
  };
  match expr {
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      Some((&args[0], 1))
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      half(&args[1]).map(|n| (&args[0], n))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => half(right).map(|n| (left.as_ref(), n)),
    _ => None,
  }
}

/// A square root of an expression in `var` inside `expr` — `Sqrt[p]` or
/// `p^(n/2)` with odd `n` — as its radicand `p`; one that no other root's
/// radicand contains, so that eliminating it leaves the others intact.
fn find_square_radical(expr: &Expr, var: &str) -> Option<Expr> {
  fn collect(e: &Expr, var: &str, out: &mut Vec<Expr>) {
    if let Some((p, _)) = as_half_power(e)
      && !is_constant_wrt(p, var)
    {
      out.push(p.clone());
    }
    for child in crate::syntax::expr_children(e) {
      collect(child, var, out);
    }
  }
  let mut radicands = Vec::new();
  collect(expr, var, &mut radicands);
  let inside = |inner: &Expr, outer: &Expr| -> bool {
    fn occurs(e: &Expr, target: &Expr) -> bool {
      as_half_power(e).is_some_and(|(p, _)| {
        crate::evaluator::pattern_matching::expr_equal(p, target)
      }) || crate::syntax::expr_children(e)
        .into_iter()
        .any(|c| occurs(c, target))
    }
    occurs(outer, inner)
  };
  radicands
    .iter()
    .find(|p| !radicands.iter().any(|q| inside(p, q)))
    .cloned()
}

/// Warn that a solution's validity depends on the parameters — from the
/// Solve the user called only, not from the equations a system's
/// elimination solves along the way, and not under NSolve.
fn report_nongen() {
  if SOLVE_DEPTH.with(std::cell::Cell::get) <= 1 && !in_nsolve() {
    crate::emit_message(
      "Solve::nongen: There may be values of the parameters for which some \
       or all solutions are not valid.",
    );
  }
}

/// Whether `expr` takes a square root of something that itself holds a
/// square root of `var`.
fn has_nested_square_radical(expr: &Expr, var: &str) -> bool {
  if let Some((r, _)) = as_half_power(expr)
    && find_square_radical(r, var).is_some()
  {
    return true;
  }
  crate::syntax::expr_children(expr)
    .into_iter()
    .any(|child| has_nested_square_radical(child, var))
}

/// Solve an equation in square roots of `var` that isolating one root and
/// squaring does not clear (roots nested in roots, several in one term):
/// each root `s = Sqrt[p]` is eliminated in turn by writing the equation as
/// `A + B s == 0` (with `s^2 = p`) and passing to its norm `A^2 - p B^2 ==
/// 0`, which every solution of the original also solves. The resulting
/// polynomial is solved and its roots checked against the original
/// equation, numerically or, with parameters, at sample points.
fn try_solve_by_radical_norms(
  poly: &Expr,
  equation: &Expr,
  var: &str,
) -> Option<Result<Expr, InterpreterError>> {
  use crate::evaluator::evaluate_expr_to_expr as eval;
  find_square_radical(poly, var)?;
  if has_inequality(equation) {
    return None;
  }
  let numerator = |e: Expr| -> Option<Expr> {
    eval(&call("Numerator", vec![call("Together", vec![e])])).ok()
  };
  let mut f = numerator(poly.clone())?;
  let s = Expr::Identifier("Solve$radical".to_string());
  for _ in 0..8 {
    let Some(p) = find_square_radical(&f, var) else {
      break;
    };
    let same = |e: &Expr| crate::evaluator::pattern_matching::expr_equal(e, &p);
    let replaced =
      crate::functions::string_ast::map_expr_tree(&f, &|node: &Expr| {
        match as_half_power(node) {
          Some((base, n)) if same(base) => {
            Some(pow(s.clone(), Expr::Integer(n)))
          }
          _ => None,
        }
      });
    let replaced = numerator(replaced)?;
    let coefficients =
      match &eval(&call("CoefficientList", vec![replaced, s.clone()])).ok()? {
        Expr::List(items) => items.to_vec(),
        _ => return None,
      };
    if coefficients
      .iter()
      .any(|c| expr_to_string(c).contains("Solve$radical"))
    {
      return None;
    }
    let (mut even, mut odd) = (Vec::new(), Vec::new());
    for (k, c) in coefficients.into_iter().enumerate() {
      let power = pow(p.clone(), Expr::Integer((k / 2) as i128));
      if k % 2 == 0 {
        even.push(times2(c, power));
      } else {
        odd.push(times2(c, power));
      }
    }
    let a = plus(even);
    let b = plus(odd);
    // Squaring multiplies the terms out; with several radicals of other
    // unknowns still in the coefficients (a Fermat-point objective in three
    // variables) the norm grows past anything Solve could use, and building
    // it took gigabytes. Give up instead.
    let size = |e: &Expr| super::simplify::leaf_count(e) as u64;
    if size(&a).pow(2) + size(&p) * size(&b).pow(2) > 400_000 {
      return None;
    }
    let norm = minus2(
      pow(a, Expr::Integer(2)),
      times2(p.clone(), pow(b, Expr::Integer(2))),
    );
    f = numerator(eval(&call("Expand", vec![norm])).ok()?)?;
  }
  if find_square_radical(&f, var).is_some() || is_constant_wrt(&f, var) {
    return None;
  }
  let reduced = Expr::Comparison {
    operands: vec![f, Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };
  let solved = quietly(|| solve_ast(&[reduced, id_expr(var)])).ok()?;
  let Expr::List(ref candidates) = solved else {
    return None;
  };
  if !candidates.iter().all(|c| {
    matches!(c, Expr::List(rules) if matches!(rules.as_slice(), [Expr::Rule { .. }]))
  }) {
    return None;
  }
  Some(keep_solutions_satisfying(Ok(solved), equation, var))
}

/// Numeric last resort for `NSolve[eqn, var]` when neither the quadratic /
/// pure-power shortcuts above nor `solve_ast`'s symbolic techniques —
/// including `try_solve_radical_equation`'s isolate-one-term-and-raise
/// elimination — find a closed form.
///
/// This covers equations that mix a sub-expression's positive and negative
/// fractional powers, e.g. a Cardano cube-root formula added to its own
/// reciprocal (`A^(1/3) + k/A^(1/3)`): isolating either term and cubing
/// reintroduces a *different* fractional power of the same base instead of
/// eliminating it, so the elimination loop gives up. Such an equation is
/// not unsolvable, just not solvable by that one technique — it is an
/// ordinary equation to root-find numerically, the way real Mathematica's
/// own numerical solver does internally. Woxi does not have that solver, so
/// this runs damped Newton iteration (central-difference derivative, since
/// the search variable need not be a plain identifier — `C[1]`, as produced
/// by `Solve`'s own generated constants, is common) from a spread of seed
/// points and keeps whichever seeds converge, deduplicated.
///
/// Only applies to `NSolve[eqn, var]` with a single equation and a single
/// (non-list) variable; systems and multi-variable solves are left alone.
fn try_nsolve_numeric_search(args: &[Expr]) -> Option<Expr> {
  if args.len() != 2 {
    return None;
  }
  let (lhs, rhs, op) =
    crate::functions::polynomial_ast::reduce::extract_comparison(&args[0])?;
  if op != crate::functions::polynomial_ast::reduce::CompOp::Equal {
    return None;
  }
  let var = &args[1];
  if matches!(var, Expr::List(_)) {
    return None;
  }
  let residual = minus2(lhs, rhs);
  if !expr_contains_expr(&residual, var) {
    return None;
  }

  let eval_at = |x: f64| -> Option<f64> {
    let substituted = substitute_expr(&residual, var, &Expr::Real(x));
    let evaled =
      quietly(|| crate::evaluator::evaluate_expr_to_expr(&substituted)).ok()?;
    try_eval_to_f64(&evaled).filter(|v| v.is_finite())
  };

  const SEEDS: &[f64] = &[
    0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 50.0,
    -50.0, 100.0, -100.0,
  ];
  const MAX_ITER: usize = 100;
  const MAX_BACKTRACKS: usize = 40;
  const TOL: f64 = 1e-13;
  const STEP_H: f64 = 1e-6;

  let mut roots: Vec<f64> = Vec::new();
  for &seed in SEEDS {
    let Some(mut fx) = eval_at(seed) else {
      continue;
    };
    let mut x = seed;
    let mut converged = fx.abs() < TOL;
    for _ in 0..MAX_ITER {
      if converged {
        break;
      }
      let (Some(f_plus), Some(f_minus)) =
        (eval_at(x + STEP_H), eval_at(x - STEP_H))
      else {
        break;
      };
      let deriv = (f_plus - f_minus) / (2.0 * STEP_H);
      if !deriv.is_finite() || deriv.abs() < 1e-14 {
        break;
      }
      let step = fx / deriv;
      let mut shrink = 1.0;
      let mut x_next = x - step;
      let mut f_next = eval_at(x_next);
      let mut tries = 0;
      while tries < MAX_BACKTRACKS && f_next.is_none_or(|v| v.abs() >= fx.abs())
      {
        shrink *= 0.5;
        x_next = x - step * shrink;
        f_next = eval_at(x_next);
        tries += 1;
      }
      let Some(fnext) = f_next else {
        break;
      };
      x = x_next;
      fx = fnext;
      if fx.abs() < TOL {
        converged = true;
      }
    }
    if converged && x.is_finite() {
      let is_duplicate = roots
        .iter()
        .any(|existing: &f64| (existing - x).abs() < 1e-6 * (1.0 + x.abs()));
      if !is_duplicate {
        roots.push(x);
      }
    }
  }
  if roots.is_empty() {
    return None;
  }
  roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
  Some(Expr::List(
    roots
      .into_iter()
      .map(|r| Expr::List(vec![rule_expr(var.clone(), Expr::Real(r))].into()))
      .collect(),
  ))
}

/// `NSolve[{eqn, bounds…}, x]`: one equation in one variable whose
/// inequalities confine `x` to a finite interval (`0 <= x < Pi`). The
/// residual is sampled across the interval and every sign change is
/// bisected to full precision, which finds the roots of transcendental
/// equations that no symbolic technique reaches; roots where the
/// inequalities fail are dropped.
fn try_nsolve_bracketed(args: &[Expr]) -> Option<Expr> {
  let Expr::List(items) = args.first()? else {
    return None;
  };
  let var = match args.get(1)? {
    Expr::List(vs) if vs.len() == 1 => vs[0].clone(),
    v @ Expr::Identifier(_) => v.clone(),
    _ => return None,
  };
  let Expr::Identifier(var_name) = &var else {
    return None;
  };
  let mut equation = None;
  let mut constraints = Vec::new();
  for item in items {
    match crate::functions::polynomial_ast::reduce::extract_comparison(item) {
      Some((
        lhs,
        rhs,
        crate::functions::polynomial_ast::reduce::CompOp::Equal,
      )) => {
        if equation.replace(minus2(lhs, rhs)).is_some() {
          return None;
        }
      }
      _ => constraints.push(item.clone()),
    }
  }
  let residual = equation?;
  if constraints.is_empty() {
    return None;
  }
  let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
  for c in &constraints {
    let (l, h) = variable_bounds(c, var_name)?;
    lo = lo.max(l);
    hi = hi.min(h);
  }
  if !(lo.is_finite() && hi.is_finite() && lo < hi) {
    return None;
  }
  let eval_at = |x: f64| -> Option<f64> {
    let substituted = substitute_expr(&residual, &var, &Expr::Real(x));
    let evaled =
      quietly(|| crate::evaluator::evaluate_expr_to_expr(&substituted)).ok()?;
    try_eval_to_f64(&evaled).filter(|v| v.is_finite())
  };
  let admissible = |x: f64| {
    constraints.iter().all(|c| {
      let substituted = substitute_expr(c, &var, &Expr::Real(x));
      matches!(
        quietly(|| crate::evaluator::evaluate_expr_to_expr(&substituted)),
        Ok(Expr::Identifier(ref t)) if t == "True"
      )
    })
  };
  const SAMPLES: usize = 2000;
  let points: Vec<(f64, Option<f64>)> = (0..=SAMPLES)
    .map(|i| {
      let x = lo + (hi - lo) * i as f64 / SAMPLES as f64;
      (x, eval_at(x))
    })
    .collect();
  let mut roots: Vec<f64> = Vec::new();
  for pair in points.windows(2) {
    let ((a, Some(fa)), (b, Some(fb))) = (pair[0], pair[1]) else {
      continue;
    };
    let root = if fa == 0.0 {
      a
    } else if fa.signum() == fb.signum() || fb == 0.0 {
      continue;
    } else {
      let (mut a, mut b, mut fa) = (a, b, fa);
      loop {
        let m = f64::midpoint(a, b);
        if m <= a || m >= b {
          break if eval_at(a)?.abs() <= eval_at(b)?.abs() {
            a
          } else {
            b
          };
        }
        let fm = eval_at(m)?;
        if fm == 0.0 {
          break m;
        }
        if fm.signum() == fa.signum() {
          a = m;
          fa = fm;
        } else {
          b = m;
        }
      }
    };
    // A sign change across a pole is not a root.
    let scale = fa.abs().max(1.0);
    if eval_at(root)?.abs() <= 1e-8 * scale && admissible(root) {
      roots.push(root);
    }
  }
  if let Some((x, Some(f))) = points.last()
    && *f == 0.0
    && admissible(*x)
  {
    roots.push(*x);
  }
  if roots.is_empty() {
    return None;
  }
  Some(Expr::List(
    roots
      .into_iter()
      .map(|r| Expr::List(vec![rule_expr(var.clone(), Expr::Real(r))].into()))
      .collect(),
  ))
}

/// The interval a constraint such as `0 <= x < 3`, `x > 1` or
/// `Inequality[0, LessEqual, x, Less, 3]` confines `var` to, open or closed
/// alike. `None` for a constraint of another shape.
fn variable_bounds(constraint: &Expr, var: &str) -> Option<(f64, f64)> {
  use crate::syntax::ComparisonOp as Op;
  let (operands, ops): (Vec<Expr>, Vec<Op>) = match constraint {
    Expr::Comparison {
      operands,
      operators,
    } => (operands.clone(), operators.clone()),
    Expr::FunctionCall { name, args }
      if name == "Inequality" && args.len() % 2 == 1 =>
    {
      let mut ops = Vec::new();
      for op in args.iter().skip(1).step_by(2) {
        ops.push(match op {
          Expr::Identifier(n) => match n.as_str() {
            "Less" => Op::Less,
            "LessEqual" => Op::LessEqual,
            "Greater" => Op::Greater,
            "GreaterEqual" => Op::GreaterEqual,
            _ => return None,
          },
          _ => return None,
        });
      }
      (args.iter().step_by(2).cloned().collect(), ops)
    }
    _ => return None,
  };
  let is_var = |e: &Expr| matches!(e, Expr::Identifier(n) if n == var);
  let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
  for (i, op) in ops.iter().enumerate() {
    let (left, right) = (&operands[i], &operands[i + 1]);
    let (bound, var_on_right) = if is_var(right) {
      (try_eval_to_f64(left)?, true)
    } else if is_var(left) {
      (try_eval_to_f64(right)?, false)
    } else {
      return None;
    };
    let below = matches!(op, Op::Less | Op::LessEqual);
    let above = matches!(op, Op::Greater | Op::GreaterEqual);
    if !(below || above) {
      return None;
    }
    // `bound < x` and `x > bound` give a lower bound.
    if below == var_on_right {
      lo = lo.max(bound);
    } else {
      hi = hi.min(bound);
    }
  }
  Some((lo, hi))
}

/// Whether a solution `{v -> value, …}` has an infinite or indeterminate
/// value.
fn has_non_finite_value(solution: &Expr) -> bool {
  fn bad(e: &Expr) -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s)
      if matches!(s.as_str(), "ComplexInfinity" | "Indeterminate" | "Infinity"))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedInfinity")
      || crate::syntax::expr_children(e).into_iter().any(bad)
  }
  bad(solution)
}

/// Numeric last resort for `NSolve[{eqns}, {vars}]` with as many equations
/// as unknowns that no elimination solves: damped Newton iteration with a
/// central-difference Jacobian from a grid of real starting points, keeping
/// the distinct points where every equation holds.
fn try_nsolve_numeric_system(args: &[Expr]) -> Option<Expr> {
  let Expr::List(eqs) = args.first()? else {
    return None;
  };
  let Expr::List(vars) = args.get(1)? else {
    return None;
  };
  let k = vars.len();
  if !(2..=8).contains(&k) || eqs.len() != k {
    return None;
  }
  let residuals: Vec<Expr> = eqs
    .iter()
    .map(|eq| {
      let (lhs, rhs, op) =
        crate::functions::polynomial_ast::reduce::extract_comparison(eq)?;
      (op == crate::functions::polynomial_ast::reduce::CompOp::Equal)
        .then(|| minus2(lhs, rhs))
    })
    .collect::<Option<_>>()?;
  // Residuals over plain symbols compile to machine code; the rest are
  // evaluated symbolically at each point.
  let var_names: Option<Vec<String>> = vars
    .iter()
    .map(|v| match v {
      Expr::Identifier(n) => Some(n.clone()),
      _ => None,
    })
    .collect();
  let compiled: Vec<Option<crate::functions::ode_ast::CompiledReal>> =
    residuals
      .iter()
      .map(|r| {
        var_names.as_ref().and_then(|names| {
          crate::functions::ode_ast::CompiledReal::new(r, names)
        })
      })
      .collect();
  let eval_at = |point: &[f64]| -> Option<Vec<f64>> {
    residuals
      .iter()
      .zip(&compiled)
      .map(|(r, c)| {
        if let Some(c) = c {
          let v = c.eval(point);
          if v.is_finite() {
            return Some(v);
          }
        }
        let mut e = r.clone();
        for (v, x) in vars.iter().zip(point) {
          e = substitute_expr(&e, v, &Expr::Real(*x));
        }
        let value =
          quietly(|| crate::evaluator::evaluate_expr_to_expr(&e)).ok()?;
        try_eval_to_f64(&value).filter(|v| v.is_finite())
      })
      .collect()
  };
  let norm = |f: &[f64]| f.iter().map(|v| v * v).sum::<f64>().sqrt();
  const COARSE: [f64; 7] = [-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0];
  // Two unknowns afford a finer grid, which reaches roots whose basin of
  // attraction is small (the off-axis equilibria of a three-body problem).
  const FINE: [f64; 13] = [
    -10.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0, 0.25,
  ];
  let starts_grid: &[f64] = if k == 2 { &FINE } else { &COARSE };
  // Up to three unknowns every combination of the start values is tried;
  // beyond that the grid is too large, so the diagonal starts and a fixed
  // pseudo-random scatter in [-3, 3]^k are used instead.
  let starts: Vec<Vec<f64>> = if k <= 3 {
    (0..starts_grid.len().pow(k as u32))
      .map(|mut n| {
        (0..k)
          .map(|_| {
            let v = starts_grid[n % starts_grid.len()];
            n /= starts_grid.len();
            v + 0.1234
          })
          .collect()
      })
      .collect()
  } else {
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
      seed ^= seed << 13;
      seed ^= seed >> 7;
      seed ^= seed << 17;
      (seed >> 11) as f64 / (1u64 << 53) as f64 * 6.0 - 3.0
    };
    starts_grid
      .iter()
      .map(|&c| vec![c + 0.1234; k])
      .chain((0..40).map(|_| (0..k).map(|_| next()).collect()))
      .collect()
  };
  let mut found: Vec<Vec<f64>> = Vec::new();
  for start in starts {
    let mut x = start;
    if let Some(mut fx) = eval_at(&x) {
      for _ in 0..60 {
        let size = norm(&fx);
        if size < 1e-12 {
          break;
        }
        // Jacobian by central differences.
        let mut jac = vec![vec![0.0; k]; k];
        let mut ok = true;
        for j in 0..k {
          let h = 1e-6 * x[j].abs().max(1.0);
          let mut up = x.clone();
          up[j] += h;
          let mut down = x.clone();
          down[j] -= h;
          let (Some(fu), Some(fd)) = (eval_at(&up), eval_at(&down)) else {
            ok = false;
            break;
          };
          for i in 0..k {
            jac[i][j] = (fu[i] - fd[i]) / (2.0 * h);
          }
        }
        if !ok {
          break;
        }
        let Some(step) = solve_mass_like(&jac, &fx) else {
          break;
        };
        let mut t = 1.0;
        let mut next = None;
        for _ in 0..30 {
          let trial: Vec<f64> =
            x.iter().zip(&step).map(|(a, d)| a - t * d).collect();
          if let Some(ft) = eval_at(&trial)
            && norm(&ft) < size
          {
            next = Some((trial, ft));
            break;
          }
          t /= 2.0;
        }
        let Some((nx, nf)) = next else { break };
        x = nx;
        fx = nf;
      }
      let scale = 1.0 + x.iter().map(|v| v.abs()).fold(0.0, f64::max);
      if norm(&fx) < 1e-9 * scale
        && !found
          .iter()
          .any(|y| y.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-6 * scale))
      {
        found.push(x);
      }
    }
  }
  if found.is_empty() {
    return None;
  }
  Some(Expr::List(
    found
      .into_iter()
      .map(|point| {
        Expr::List(
          vars
            .iter()
            .zip(point)
            .map(|(v, x)| rule_expr(v.clone(), Expr::Real(x)))
            .collect(),
        )
      })
      .collect(),
  ))
}

/// Solve the small dense system `a·d = b` by Gaussian elimination.
fn solve_mass_like(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
  let n = b.len();
  let mut m: Vec<Vec<f64>> = a.to_vec();
  let mut r = b.to_vec();
  for col in 0..n {
    let piv =
      (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
    if m[piv][col].abs() < 1e-300 {
      return None;
    }
    m.swap(col, piv);
    r.swap(col, piv);
    for row in col + 1..n {
      let f = m[row][col] / m[col][col];
      for c in col..n {
        m[row][c] -= f * m[col][c];
      }
      r[row] -= f * r[col];
    }
  }
  let mut d = vec![0.0; n];
  for row in (0..n).rev() {
    let mut acc = r[row];
    for c in row + 1..n {
      acc -= m[row][c] * d[c];
    }
    d[row] = acc / m[row][row];
  }
  Some(d)
}

/// True when `target` occurs (structurally, via string comparison like
/// `substitute_expr`) anywhere inside `expr`.
pub(crate) fn expr_contains_expr(expr: &Expr, target: &Expr) -> bool {
  if expr_to_string(expr) == expr_to_string(target) {
    return true;
  }
  match expr {
    Expr::List(items) => items.iter().any(|e| expr_contains_expr(e, target)),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|e| expr_contains_expr(e, target))
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_expr(left, target) || expr_contains_expr(right, target)
    }
    Expr::UnaryOp { operand, .. } => expr_contains_expr(operand, target),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|e| expr_contains_expr(e, target))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      expr_contains_expr(pattern, target)
        || expr_contains_expr(replacement, target)
    }
    _ => false,
  }
}

/// Roots beyond this index are left alone: raising to them makes a
/// polynomial no solver reports in radicals anyway.
const MAX_RADICAL_INDEX: i128 = 4;

/// The index of the root `term` takes of the unknown — 2 for a square root,
/// 3 for a cube root — or `None` for a term that takes none. A term under
/// several roots at once reports the index that clears all of them.
fn radical_index(term: &Expr, var: &str) -> Option<i128> {
  let root_here = |expr: &Expr| -> Option<i128> {
    if let Some(inner) = is_sqrt(expr) {
      return (!is_constant_wrt(inner, var)).then_some(2);
    }
    let (base, exponent) = extract_base_and_exp(expr);
    if is_constant_wrt(&base, var) {
      return None;
    }
    match rational_parts(&exponent) {
      Some((_, denominator)) if denominator > 1 => Some(denominator),
      _ => None,
    }
  };
  fn walk(
    expr: &Expr,
    root_here: &dyn Fn(&Expr) -> Option<i128>,
    index: &mut Option<i128>,
  ) {
    if let Some(root) = root_here(expr) {
      *index = Some(match *index {
        None => root,
        Some(current) => lcm_i128(current, root),
      });
    }
    match expr {
      Expr::BinaryOp { left, right, .. } => {
        walk(left, root_here, index);
        walk(right, root_here, index);
      }
      Expr::UnaryOp { operand, .. } => walk(operand, root_here, index),
      Expr::FunctionCall { args, .. } => {
        for arg in args {
          walk(arg, root_here, index);
        }
      }
      _ => {}
    }
  }
  let mut index = None;
  walk(term, &root_here, &mut index);
  index
}

/// An exponent written as a ratio of two integers, however it is spelled.
fn rational_parts(exponent: &Expr) -> Option<(i128, i128)> {
  match exponent {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(numerator), Expr::Integer(denominator)) => {
          Some((*numerator, *denominator))
        }
        _ => None,
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => match (left.as_ref(), right.as_ref()) {
      (Expr::Integer(numerator), Expr::Integer(denominator)) => {
        Some((*numerator, *denominator))
      }
      _ => None,
    },
    _ => None,
  }
}

/// Whether `equation` holds with `value` put in for `var`, or `None` when
/// that cannot be decided.
/// Whether a candidate solves an equation with other symbols in it, judged
/// at sample points where each of those symbols takes a number (positive,
/// negative, larger, complex): valid at all of them, at some (valid for
/// some parameter values, as `x == -Sqrt[4 - y^2]` for x <= 0), or at none.
/// `None` when no sample point decides.
fn holds_generically(
  equation: &Expr,
  var: &str,
  value: &Expr,
) -> Option<GenericValidity> {
  fn collect(e: &Expr, out: &mut std::collections::BTreeSet<String>) {
    if let Expr::Identifier(name) = e {
      out.insert(name.clone());
    }
    for child in crate::syntax::expr_children(e) {
      collect(child, out);
    }
  }
  let mut found = std::collections::BTreeSet::new();
  collect(equation, &mut found);
  collect(value, &mut found);
  let symbols: Vec<String> = found
    .into_iter()
    .filter(|s| s != var && !crate::evaluator::functions::is_builtin_symbol(s))
    .collect();
  if symbols.is_empty() {
    return None;
  }
  let at = |point: &dyn Fn(usize) -> Expr| -> Option<bool> {
    let mut equation = equation.clone();
    let mut value = value.clone();
    for (k, symbol) in symbols.iter().enumerate() {
      equation =
        crate::syntax::substitute_variable(&equation, symbol, &point(k));
      value = crate::syntax::substitute_variable(&value, symbol, &point(k));
    }
    equation_holds_at(&equation, var, &value)
  };
  let shift = |k: usize| 0.3127 * k as f64;
  let points: [&dyn Fn(usize) -> Expr; 4] = [
    &|k| Expr::Real(0.7349 + shift(k)),
    &|k| Expr::Real(-1.3127 - shift(k)),
    &|k| Expr::Real(7.3149 + shift(k)),
    &|k| {
      crate::functions::math_ast::build_complex_float_expr_keep_real(
        0.5213 + shift(k),
        0.8137,
      )
    },
  ];
  let verdicts: Vec<Option<bool>> = points.iter().map(|p| at(*p)).collect();
  if verdicts.iter().all(Option::is_none) {
    return None;
  }
  Some(if verdicts.iter().all(|v| *v == Some(true)) {
    GenericValidity::Always
  } else if verdicts.contains(&Some(true)) {
    GenericValidity::Partly
  } else {
    GenericValidity::Fails
  })
}

/// How a candidate with parameters fares across sample parameter values.
#[derive(PartialEq)]
enum GenericValidity {
  Always,
  Partly,
  Fails,
}

/// Drop the solutions of an equation system that cannot be right: one with
/// an infinite or indeterminate value, or one that turns an equation into
/// `False` once its exact values are put in (an elimination can leave a
/// stray candidate such as `p -> 0` of `p == t^(-7/2)`). Anything the exact
/// substitution cannot decide is kept.
fn drop_failing_system_solutions(args: &[Expr], result: Expr) -> Expr {
  let (Some(Expr::List(eqs)), Expr::List(solutions)) = (args.first(), &result)
  else {
    return result;
  };
  // Modular and domain-restricted systems are decided by their own rules.
  if eqs.len() < 2 || args.len() != 2 {
    return result;
  }
  let kept: Vec<Expr> = solutions
    .iter()
    .filter(|solution| {
      if has_non_finite_value(solution) {
        return false;
      }
      let Expr::List(rules) = solution else {
        return true;
      };
      let mut point = Vec::new();
      for rule in rules {
        let Expr::Rule {
          pattern,
          replacement,
        } = rule
        else {
          return true;
        };
        let Expr::Identifier(var) = pattern.as_ref() else {
          return true;
        };
        // A `Root[…]` value only compares numerically, which is not
        // reliable enough to reject a solution on.
        if crate::functions::math_ast::contains_inexact_real(replacement)
          || !is_exact_number(replacement)
          || contains_head(replacement, "Root")
        {
          return true;
        }
        point.push((var.clone(), replacement.as_ref().clone()));
      }
      !eqs.iter().any(|eq| {
        let substituted = point.iter().fold(eq.clone(), |e, (var, value)| {
          crate::syntax::substitute_variable(&e, var, value)
        });
        matches!(
          quietly(|| crate::evaluator::evaluate_expr_to_expr(&substituted)),
          Ok(Expr::Identifier(ref f)) if f == "False"
        )
      })
    })
    .cloned()
    .collect();
  Expr::List(kept.into())
}

/// Whether `e` contains a call with head `head`.
fn contains_head(e: &Expr, head: &str) -> bool {
  matches!(e, Expr::FunctionCall { name, .. } if name == head)
    || crate::syntax::expr_children(e)
      .into_iter()
      .any(|c| contains_head(c, head))
}

/// An exact number without symbols: integers, rationals and their
/// arithmetic with radicals (`405`, `1/2`, `Sqrt[2]`).
fn is_exact_number(e: &Expr) -> bool {
  matches!(
    crate::evaluator::evaluate_function_call_ast(
      "NumericQ",
      std::slice::from_ref(e)
    ),
    Ok(Expr::Identifier(ref t)) if t == "True"
  )
}

fn equation_holds_at(equation: &Expr, var: &str, value: &Expr) -> Option<bool> {
  let (lhs, rhs, CompOp::Equal) =
    crate::functions::polynomial_ast::reduce::extract_comparison(equation)?
  else {
    return None;
  };
  let at_value = |side: &Expr| {
    let substituted = crate::syntax::substitute_variable(side, var, value);
    quietly(|| crate::evaluator::evaluate_expr_to_expr(&substituted)).ok()
  };
  let left = at_value(&lhs)?;
  let right = at_value(&rhs)?;
  if expr_to_string(&left) == expr_to_string(&right) {
    return Some(true);
  }
  // An exact root only cancels to a literal zero once it is worked out, and
  // an inexact one never quite does, so the two sides are compared as
  // numbers against the size of what they are made of.
  let (left_re, left_im) = try_extract_complex_f64(&left)?;
  let (right_re, right_im) = try_extract_complex_f64(&right)?;
  let scale = 1.0 + left_re.hypot(left_im).max(right_re.hypot(right_im));
  Some((left_re - right_re).hypot(left_im - right_im) <= 1e-9 * scale)
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
  if all_power_factors.iter().any(std::vec::Vec::is_empty) {
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
              let (new_num, new_den) = rat_reduce(new_num, new_den);

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
                  call(
                    "Rational",
                    vec![Expr::Integer(new_num), Expr::Integer(new_den)],
                  )
                };
                remaining_factors[idx] = pow2(base_expr, new_exp);
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
    (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => make_rational(*n, *d),
    // Non-integer denominator (a rational such as -1/2, or a symbolic
    // expression): evaluate the quotient so it is fully simplified
    // (e.g. -x / (-1/2) -> 2*x) rather than left as a nested fraction.
    _ => {
      let div = div2(num.clone(), den.clone());
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
      times2(c.clone(), id_expr(var))
    } else {
      times2(c.clone(), pow2(id_expr(var), Expr::Integer(i as i128)))
    };
    terms.push(term);
  }
  let poly_expr = if terms.is_empty() {
    Expr::Integer(0)
  } else {
    let mut result = terms.remove(0);
    for t in terms {
      result = plus2(result, t);
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
    Expr::Slot(n) if *n == k => id_expr(name),
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| rs_subst_slot(e, k, name)).collect())
    }
    Expr::FunctionCall { name: fname, args } => call(
      fname.as_str(),
      args.iter().map(|e| rs_subst_slot(e, k, name)).collect(),
    ),
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      rs_subst_slot(left, k, name),
      rs_subst_slot(right, k, name),
    ),
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
  let unevaluated = || Ok(unevaluated("RootSum", args));
  if args.len() != 2 {
    return unevaluated();
  }
  let var = "__rootsum_x__";
  let var_sym = id_expr(var);

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

/// The polynomial of a `Root` the way wolframscript keeps it: integer
/// coefficients without a common factor and a positive leading one
/// (`Root[10 + 99 #1 - 4 #1^7 &, 1]` is `Root[-10 - 99 #1 + 4 #1^7 &, 1]`).
/// `None` for coefficients that are not all rational.
fn primitive_root_polynomial(poly: &Expr, var: &str) -> Option<Expr> {
  let coefficients = crate::evaluator::evaluate_function_call_ast(
    "CoefficientList",
    &[poly.clone(), id_expr(var)],
  )
  .ok()?;
  let Expr::List(ref coefficients) = coefficients else {
    return None;
  };
  let mut parts: Vec<(num_bigint::BigInt, num_bigint::BigInt)> = Vec::new();
  for c in coefficients {
    parts.push(match c {
      Expr::Integer(n) => ((*n).into(), 1.into()),
      Expr::BigInteger(n) => (n.clone(), 1.into()),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        let big = |e: &Expr| match e {
          Expr::Integer(n) => Some(num_bigint::BigInt::from(*n)),
          Expr::BigInteger(n) => Some(n.clone()),
          _ => None,
        };
        (big(&args[0])?, big(&args[1])?)
      }
      _ => return None,
    });
  }
  use crate::functions::math_ast::gcd_bigint;
  use num_traits::{Signed, Zero};
  let lcm = parts.iter().fold(num_bigint::BigInt::from(1), |l, (_, d)| {
    &l / gcd_bigint(&l, d) * d.abs()
  });
  let numerators: Vec<num_bigint::BigInt> =
    parts.iter().map(|(n, d)| n * (&lcm / d)).collect();
  let gcd = numerators
    .iter()
    .fold(num_bigint::BigInt::zero(), |g, n| gcd_bigint(&g, n).abs());
  let leading = numerators.iter().rev().find(|n| !n.is_zero())?;
  if gcd.is_zero() {
    return None;
  }
  let gcd = if leading.is_negative() { -gcd } else { gcd };
  let terms: Vec<Expr> = numerators
    .iter()
    .enumerate()
    .filter(|(_, n)| !n.is_zero())
    .map(|(i, n)| {
      times2(
        crate::functions::math_ast::bigint_to_expr(n / &gcd),
        pow2(id_expr(var), Expr::Integer(i as i128)),
      )
    })
    .collect();
  Some(plus(terms))
}

/// The root `Root[{f, x0}]` stands for: `f`'s zero found by `FindRoot` from
/// `x0`, as a machine number, or None when it cannot be found.
pub fn transcendental_root_value(f: &Expr, x0: &Expr) -> Option<Expr> {
  let t = Expr::Identifier("Internal`RootVariable".to_string());
  let applied = crate::evaluator::apply_function_to_arg(f, &t).ok()?;
  let found = crate::evaluator::evaluate_expr_to_expr(&call(
    "Quiet",
    vec![call(
      "FindRoot",
      vec![
        call("Equal", vec![applied.clone(), Expr::Integer(0)]),
        Expr::List(vec![t.clone(), call("N", vec![x0.clone()])].into()),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("WorkingPrecision".into())),
          replacement: Box::new(Expr::Integer(30)),
        },
      ],
    )],
  ))
  .ok()?;
  let Expr::List(rules) = &found else {
    return None;
  };
  let Some(Expr::Rule { replacement, .. }) = rules.first() else {
    return None;
  };
  crate::functions::math_ast::try_eval_to_f64(replacement)?;
  // A few Newton steps at 40 digits make the machine root correctly
  // rounded.
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let derivative = eval(call("D", vec![applied.clone(), t.clone()]))?;
  let step = call(
    "Plus",
    vec![
      t.clone(),
      call(
        "Times",
        vec![
          Expr::Integer(-1),
          applied.clone(),
          call("Power", vec![derivative, Expr::Integer(-1)]),
        ],
      ),
    ],
  );
  let mut x = eval(call(
    "SetPrecision",
    vec![replacement.as_ref().clone(), Expr::Integer(40)],
  ))?;
  for _ in 0..4 {
    x = eval(call(
      "N",
      vec![substitute_expr(&step, &t, &x), Expr::Integer(40)],
    ))?;
  }
  crate::functions::math_ast::try_eval_to_f64(&x).map(Expr::Real)
}

/// `Root[{f, x0}]` — the root of the transcendental function `f` that the
/// approximation `x0` identifies; it stays as written, and `N` gives the
/// root. An `x0` too far from a root (the refined root moves by more than
/// 10^-6 relative) is reported (`::invrt`), anything but such a pair is
/// not a root specification (`::trsp`), and a pattern (`Root[_]`) is left
/// alone.
fn transcendental_root_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  let held = || Ok(call("Root", vec![arg.clone()]));
  let Expr::List(spec) = arg else {
    if !crate::evaluator::pattern_matching::contains_pattern(arg) {
      crate::emit_message(&format!(
        "Root::trsp: Nonpolynomial root specification {} is not a list with \
         two elements.",
        crate::syntax::format_expr(arg, crate::syntax::ExprForm::Output)
      ));
    }
    return held();
  };
  let [f, x0] = spec.as_slice() else {
    return held();
  };
  let Some(guess) = crate::functions::math_ast::try_eval_to_f64(x0) else {
    return held();
  };
  let close = transcendental_root_value(f, x0)
    .and_then(|r| crate::functions::math_ast::try_eval_to_f64(&r))
    .is_some_and(|r| (r - guess).abs() <= 1e-6 * r.abs().max(1.0));
  if !close {
    let shown = crate::evaluator::evaluate_expr_to_expr(&call(
      "ToString",
      vec![call("SetPrecision", vec![x0.clone(), Expr::Integer(16)])],
    ))?;
    let Expr::String(ref shown) = shown else {
      return held();
    };
    crate::emit_message(&format!(
      "Root::invrt: {shown} is not equal to a root of {}.",
      crate::syntax::format_expr(f, crate::syntax::ExprForm::Output)
    ));
  }
  held()
}

pub fn root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let [arg] = args {
    return transcendental_root_ast(arg);
  }
  // A `Root[…]` is re-evaluated whenever an expression holding it is, and
  // canonicalizing it factors its polynomial; the outcome only depends on
  // the arguments. A call that reported a message is not remembered, so
  // that it reports it again.
  thread_local! {
    static ROOT_FORMS: std::cell::RefCell<
      std::collections::HashMap<String, Expr>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
  }
  let key = args
    .iter()
    .map(expr_to_string)
    .collect::<Vec<_>>()
    .join("\u{1}");
  if let Some(hit) = ROOT_FORMS.with(|c| c.borrow().get(&key).cloned()) {
    return Ok(hit);
  }
  let side_effects = crate::epoch::side_effect_count();
  let result = root_ast_uncached(args)?;
  if crate::epoch::side_effect_count() == side_effects {
    ROOT_FORMS.with(|c| {
      let mut c = c.borrow_mut();
      if c.len() >= 4096 {
        c.clear();
      }
      c.insert(key, result.clone());
    });
  }
  Ok(result)
}

fn root_ast_uncached(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Ok(unevaluated("Root", args));
  }

  // Root[f, k, 0] is the exact form (same as Root[f, k]). Root[f, k, 1]
  // requests fast numerical evaluation — we leave it symbolic for now,
  // matching wolframscript's behaviour when it cannot simplify to a
  // closed form.
  if args.len() == 3 && !matches!(&args[2], Expr::Integer(0 | 1)) {
    return Ok(unevaluated("Root", args));
  }

  let k = match &args[1] {
    Expr::Integer(n) => *n,
    _ => {
      return Ok(unevaluated("Root", args));
    }
  };

  if k < 1 {
    return Err(InterpreterError::EvaluationError(
      "Root: index k must be a positive integer".into(),
    ));
  }

  // A polynomial expression in a single variable (rather than a pure function)
  // is normalized to the pure-function form by replacing that variable with
  // Slot[1], then re-dispatched. Matches wolframscript, e.g.
  //   Root[x^2 - 2, 2] = Sqrt[2],  Root[x^3 - 2, 1] = Root[-2 + #1^3 &, 1, 0].
  // (Constants like Pi/E are Expr::Constant, so they are not mistaken for the
  // variable; zero or more than one variable stays unevaluated.)
  if !matches!(&args[0], Expr::Function { .. }) {
    let mut vars = std::collections::HashSet::new();
    super::simplify::collect_variables(&args[0], &mut vars);
    if vars.len() == 1 {
      let var = vars.into_iter().next().unwrap();
      let body =
        crate::syntax::substitute_variable(&args[0], &var, &Expr::Slot(1));
      let mut new_args = vec![Expr::Function {
        body: Box::new(body),
      }];
      new_args.extend_from_slice(&args[1..]);
      return root_ast(&new_args);
    }
    return Ok(unevaluated("Root", args));
  }

  // Extract pure function body
  let Expr::Function { body } = &args[0] else {
    return Ok(unevaluated("Root", args));
  };

  // Substitute Slot(1) with a temporary variable
  let var_name = "\u{2620}root\u{2620}"; // unique internal variable
  let poly = crate::syntax::substitute_slots(body, &[id_expr(var_name)]);

  // wolframscript reports an explicit value only when the polynomial factors
  // over the rationals into pieces of degree at most 2, and then in the
  // quadratic-formula form: Root[x^3 - 8, 2] is -1 - I Sqrt[3], while the
  // irreducible Root[x^3 - 2, 1] keeps the canonical Root form even though
  // Solve writes that root as 2^(1/3).
  let expanded_poly = expand_and_combine(&poly);
  // An index past the degree names no root.
  if let Some(degree) = max_power_int(&expanded_poly, var_name)
    && degree >= 1
    && k > degree
  {
    return Ok(root_degree_message(args));
  }
  let factor_degree = |factor: &Expr| -> Option<i128> {
    let base = match factor {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        ..
      } => left.as_ref(),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        &args[0]
      }
      other => other,
    };
    max_power_int(base, var_name)
  };
  let mut factored_roots: Option<Vec<Expr>> = None;
  if matches!(max_power_int(&expanded_poly, var_name), Some(d) if d >= 3) {
    let factors = crate::functions::polynomial_ast::factor_ast(
      std::slice::from_ref(&expanded_poly),
    )
    .map(|factored| extract_times_factors(&factored))
    .unwrap_or_default();
    let irreducible = factors.is_empty()
      || factors
        .iter()
        .any(|f| matches!(factor_degree(f), Some(d) if d >= 3));
    if !irreducible {
      // Each factor is solved on its own so the quadratic ones report the
      // quadratic-formula form rather than the binomial radicals Solve
      // would give for the product.
      let mut collected: Vec<Expr> = Vec::new();
      for factor in &factors {
        if factor_degree(factor).is_none_or(|d| d < 1) {
          continue;
        }
        // A repeated factor `base^m` contributes the roots of `base`, each
        // m times; solving the power itself would come back here as a Root
        // of degree 3 or more (`Root[(#1 - 1)^3 &, 1]` recursed forever).
        let (base, multiplicity) = match factor {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => match right.as_ref() {
            Expr::Integer(m) if *m >= 1 => (left.as_ref().clone(), *m as usize),
            _ => (factor.clone(), 1),
          },
          Expr::FunctionCall { name, args }
            if name == "Power" && args.len() == 2 =>
          {
            match &args[1] {
              Expr::Integer(m) if *m >= 1 => (args[0].clone(), *m as usize),
              _ => (factor.clone(), 1),
            }
          }
          _ => (factor.clone(), 1),
        };
        let factor_eq = Expr::Comparison {
          operands: vec![base, Expr::Integer(0)],
          operators: vec![ComparisonOp::Equal],
        };
        let solved = solve_ast(&[factor_eq, id_expr(var_name)])?;
        if let Expr::List(outer) = &solved {
          for item in outer {
            if let Expr::List(inner) = item {
              for rule in inner {
                if let Expr::Rule { replacement, .. } = rule {
                  for _ in 0..multiplicity {
                    collected.push((**replacement).clone());
                  }
                }
              }
            }
          }
        }
      }
      if !collected.is_empty() {
        factored_roots = Some(collected);
      }
    }
    if irreducible {
      let canonical_body = match &args[0] {
        Expr::Function { body } => Expr::Function {
          body: Box::new(crate::evaluator::evaluate_expr_to_expr(
            &primitive_root_polynomial(&expanded_poly, var_name).map_or_else(
              || body.as_ref().clone(),
              |p| {
                crate::syntax::substitute_variable(&p, var_name, &Expr::Slot(1))
              },
            ),
          )?),
        },
        _ => args[0].clone(),
      };
      return Ok(call(
        "Root",
        vec![canonical_body, args[1].clone(), Expr::Integer(0)],
      ));
    }
  }

  // Solve the polynomial equation poly == 0
  let eq = Expr::Comparison {
    operands: vec![poly, Expr::Integer(0)],
    operators: vec![ComparisonOp::Equal],
  };

  let solutions = match &factored_roots {
    Some(_) => Expr::List(Vec::new().into()),
    None => solve_ast(&[eq, id_expr(var_name)])?,
  };

  // Extract root values from {{var -> val1}, {var -> val2}, ...}
  let mut roots: Vec<Expr> = factored_roots.unwrap_or_default();
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
    return Ok(call(
      "Root",
      vec![canonical_body, args[1].clone(), Expr::Integer(0)],
    ));
  }

  // Sort roots: real roots first (ascending), then complex roots
  roots.sort_by(root_order);

  let idx = (k as usize) - 1;
  if idx >= roots.len() {
    return Ok(root_degree_message(args));
  }

  // Simplify the result
  crate::evaluator::evaluate_expr_to_expr(&roots[idx])
}

/// `Root[f, k]` with `k` beyond the degree of `f`: `Root::deg`, and the call
/// stays unevaluated.
fn root_degree_message(args: &[Expr]) -> Expr {
  let body = match &args[0] {
    Expr::Function { body } => body.as_ref().clone(),
    other => other.clone(),
  };
  let body = crate::evaluator::evaluate_expr_to_expr(&body).unwrap_or(body);
  crate::emit_message(&crate::syntax::message_with_expr_2d(
    "Root::deg: ",
    &body,
    &format!(
      " has fewer than {} root(s) as a polynomial in #1.",
      expr_to_string(&args[1])
    ),
  ));
  unevaluated("Root", args)
}

/// Order roots the way Wolfram's `Root` does: real roots first, sorted
/// ascending, then complex roots sorted by (real, imag).
pub fn root_order(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  use crate::functions::list_helpers_ast::expr_to_complex_parts;
  // Radical roots such as (-1)^(1/3) carry no syntactic real and imaginary
  // part, so fall back to their numeric value rather than leaving the pair
  // unordered.
  let parts = |e: &Expr| -> Option<(f64, f64)> {
    expr_to_complex_parts(e).or_else(|| {
      let numeric = crate::evaluator::evaluate_function_call_ast(
        "N",
        std::slice::from_ref(e),
      )
      .ok()?;
      expr_to_complex_parts(&numeric)
    })
  };
  let pa = parts(a);
  let pb = parts(b);

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

/// wolframscript's `MaxIterations` default for `FindRoot`.
const FIND_ROOT_DEFAULT_MAX_ITERATIONS: usize = 100;

/// How many times one Newton step may be halved before it is accepted anyway.
const FIND_ROOT_MAX_BACKTRACKS: usize = 60;

/// Parse the `MaxIterations` option out of FindRoot's trailing option
/// arguments, shared by the single-variable and multivariate forms so the
/// validation and message stay in one place. `Ok(None)` means the option was
/// absent or `Automatic` (caller should use its own default); an invalid
/// value emits the `ioppfa` message and returns `Err(())`, on which the
/// caller must return the whole call unevaluated, matching wolframscript.
fn find_root_parse_max_iterations(opts: &[Expr]) -> Result<Option<usize>, ()> {
  for opt in opts {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      continue;
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      continue;
    };
    if name != "MaxIterations" {
      continue;
    }
    return match replacement.as_ref() {
      Expr::Integer(n) if *n >= 1 => Ok(Some(*n as usize)),
      Expr::Identifier(id) if id == "Automatic" => Ok(None),
      Expr::Identifier(id) if id == "Infinity" => Ok(Some(usize::MAX)),
      other => {
        crate::emit_message(&format!(
          "FindRoot::ioppfa: The value of the option MaxIterations -> {} \
           should be a positive integer, Infinity or Automatic.",
          crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
        ));
        Err(())
      }
    };
  }
  Ok(None)
}

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

  // FindRoot holds its arguments, so a spec built programmatically and
  // passed by name — `initguess = Flatten[Table[{u[i], 0.5}, ...]];
  // FindRoot[sys, initguess]`, the idiom collocation methods use to name
  // one unknown per grid point — reaches here as a bare identifier rather
  // than a literal list. Resolve it once to reveal that list structure. A
  // `{var, x0}` written directly at the call site is left untouched (not
  // re-evaluated) so a variable that happens to carry an unrelated global
  // value isn't substituted into the "variable" slot.
  let arg1_owned;
  let arg1: &Expr = if matches!(&args[1], Expr::List(_)) {
    &args[1]
  } else {
    arg1_owned = crate::evaluator::evaluate_expr_to_expr(&args[1])
      .unwrap_or_else(|_| args[1].clone());
    &arg1_owned
  };

  // `FindRoot[f, {x0}]`: a root of the function `f` itself, from the
  // starting value x0, reported as `{x}` (no variable to name).
  if let Expr::List(start) = arg1
    && let [x0] = start.as_slice()
    && crate::evaluator::evaluate_expr_to_expr(x0)
      .is_ok_and(|v| crate::functions::predicate_ast::is_numeric_q(&v))
    && !matches!(&args[0], Expr::Comparison { .. } | Expr::List(_))
  {
    let var = Expr::Identifier("FindRoot$x".to_string());
    let mut inner = vec![
      Expr::CurriedCall {
        func: Box::new(args[0].clone()),
        args: vec![var.clone()],
      },
      Expr::List(vec![var.clone(), x0.clone()].into()),
    ];
    inner.extend(args[2..].iter().cloned());
    let result = find_root_ast(&inner)?;
    return Ok(match &result {
      Expr::List(rules) => Expr::List(
        rules
          .iter()
          .map(|r| match r {
            Expr::Rule { replacement, .. } => replacement.as_ref().clone(),
            other => other.clone(),
          })
          .collect(),
      ),
      _ => result,
    });
  }

  // A starting value that is not a number leaves the call unevaluated with
  // FindRoot::srect, as in wolframscript — never a hard error.
  {
    let specs: Vec<&Expr> = match arg1 {
      Expr::List(items) if items.iter().all(|s| matches!(s, Expr::List(_))) => {
        items.iter().collect()
      }
      _ => args[1..]
        .iter()
        .take_while(|s| matches!(s, Expr::List(_)))
        .collect(),
    };
    for spec in specs {
      let Expr::List(items) = spec else { continue };
      if !(2..=4).contains(&items.len()) || !is_findroot_var_expr(&items[0]) {
        continue;
      }
      for value in items.iter().skip(1) {
        let numeric = crate::evaluator::evaluate_expr_to_expr(&call(
          "AllTrue",
          vec![
            call1("Flatten", Expr::List(vec![value.clone()].into())),
            id_expr("NumericQ"),
          ],
        ))
        .is_ok_and(|q| matches!(q, Expr::Identifier(ref t) if t == "True"));
        if !numeric {
          let shown = crate::evaluator::evaluate_expr_to_expr(value)
            .unwrap_or_else(|_| value.clone());
          let spec_shown = crate::evaluator::evaluate_expr_to_expr(spec)
            .unwrap_or_else(|_| spec.clone());
          crate::emit_message(&format!(
            "FindRoot::srect: Value {} in search specification {} is not a number or array of numbers.",
            expr_to_string(&shown),
            expr_to_string(&spec_shown)
          ));
          return Ok(unevaluated("FindRoot", args));
        }
      }
    }
  }

  // Multivariate form: FindRoot[{eqns}, {{x, x0}, {y, y0}, ...}] — every
  // variable spec is itself a {var, start} list, optionally {var, x0, x1}
  // for a per-variable secant hint. Solved with multidimensional Newton
  // iteration.
  if let Expr::List(specs) = arg1
    && !specs.is_empty()
    && specs.iter().all(|s| {
      matches!(s, Expr::List(p)
      if (2..=4).contains(&p.len()) && is_findroot_var_expr(&p[0]))
    })
  {
    let max_iter = match find_root_parse_max_iterations(&args[2..]) {
      Ok(v) => v.unwrap_or(FIND_ROOT_DEFAULT_MAX_ITERATIONS),
      Err(()) => return Ok(unevaluated("FindRoot", args)),
    };
    return find_root_multivariate(
      &args[0],
      specs,
      max_iter,
      find_option(&args[2..], "StepMonitor"),
      args,
    );
  }

  // Multivariate form: FindRoot[{eqns}, {x, x0}, {y, y0}, ...] — the
  // documented form (used throughout Wolfram's own examples) where each
  // variable spec is its own trailing argument instead of being wrapped in
  // one outer list. Collect every leading trailing argument that looks like
  // a {var, start} or {var, x0, x1} pair; two or more means this is the
  // multivariate form rather than the single-variable {var, x0} / {var, x0,
  // x1} form handled below.
  if args.len() >= 3 {
    let candidate_specs: Vec<Expr> = args[1..]
      .iter()
      .take_while(|s| {
        matches!(s, Expr::List(p)
        if (2..=4).contains(&p.len()) && is_findroot_var_expr(&p[0]))
      })
      .cloned()
      .collect();
    if candidate_specs.len() >= 2 {
      let opts = &args[1 + candidate_specs.len()..];
      let max_iter = match find_root_parse_max_iterations(opts) {
        Ok(v) => v.unwrap_or(FIND_ROOT_DEFAULT_MAX_ITERATIONS),
        Err(()) => return Ok(unevaluated("FindRoot", args)),
      };
      return find_root_multivariate(
        &args[0],
        &candidate_specs,
        max_iter,
        find_option(opts, "StepMonitor"),
        args,
      );
    }
  }

  // `FindRoot[{eqn}, {x, x0}]`: a one-equation system is that equation.
  if let Expr::List(eqs) = &args[0]
    && eqs.len() == 1
  {
    let mut single = args.to_vec();
    single[0] = eqs[0].clone();
    return find_root_ast(&single);
  }

  // Parse the options we honour: Method -> "Secant" picks the secant iteration,
  // and MaxIterations caps the number of steps. The rest are accepted and
  // ignored (see the note on AccuracyGoal/PrecisionGoal below).
  let mut use_secant = false;
  let max_iter: usize = match find_root_parse_max_iterations(&args[2..]) {
    Ok(v) => v.unwrap_or(FIND_ROOT_DEFAULT_MAX_ITERATIONS),
    Err(()) => return Ok(unevaluated("FindRoot", args)),
  };
  for opt in &args[2..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      continue;
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      continue;
    };
    if name == "Method"
      && let Expr::String(m) = replacement.as_ref()
      && m == "Secant"
    {
      use_secant = true;
    }
  }

  let step_monitor = find_option(&args[2..], "StepMonitor");

  // Parse second argument: {var, x0} or {var, x0, x1}
  // First peek at the variable name and start point; if the start point
  // evaluates to a complex number we route to a complex Newton iteration
  // before the real-only path below.
  let (var_name, x_start_expr) = match arg1 {
    // An indexed variable (`{T[1], 350}`) goes through the multivariate
    // solver, which renames such variables to plain symbols.
    Expr::List(items)
      if (2..=4).contains(&items.len())
        && !matches!(&items[0], Expr::Identifier(_))
        && is_findroot_var_expr(&items[0]) =>
    {
      return find_root_multivariate(
        &args[0],
        std::slice::from_ref(arg1),
        max_iter,
        step_monitor,
        args,
      );
    }
    Expr::List(items) if (2..=5).contains(&items.len()) => {
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
    let func = build_find_root_func(&args[0], &[&var_name]);
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
  let mut bounds: Option<(f64, f64)> = None;
  let (var, x0, x1_opt) = match arg1 {
    Expr::List(items) if items.len() == 4 => {
      // {var, x0, xmin, xmax}: Newton search confined to [xmin, xmax].
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "FindRoot: variable must be a symbol".into(),
          ));
        }
      };
      let x0 = find_root_eval_number(&items[1])?;
      let lo = find_root_eval_number(&items[2])?;
      let hi = find_root_eval_number(&items[3])?;
      bounds = Some((lo.min(hi), lo.max(hi)));
      (var_name, x0, None)
    }
    Expr::List(items) if items.len() == 5 => {
      // {var, x0, x1, xmin, xmax}: secant search confined to [xmin, xmax].
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
      let lo = find_root_eval_number(&items[3])?;
      let hi = find_root_eval_number(&items[4])?;
      let (lo, hi) = (lo.min(hi), lo.max(hi));
      // A start outside the search region is reported, not clamped.
      let outside = [x0, x1].into_iter().find(|v| *v < lo || *v > hi);
      if let Some(v) = outside {
        let real = |x: f64| crate::syntax::expr_to_string(&Expr::Real(x));
        crate::emit_message(&format!(
          "FindRoot::streg: The starting point {{{}}} is not in the search region {{{{{}}}, {{{}}}}}.",
          real(v),
          real(lo),
          real(hi)
        ));
        return Ok(unevaluated("FindRoot", args));
      }
      bounds = Some((lo, hi));
      (var_name, x0, Some(x1))
    }
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
  let x0 = match bounds {
    Some((lo, hi)) => x0.clamp(lo, hi),
    None => x0,
  };
  // Use secant method if x1 is provided or Method -> "Secant"
  let use_secant = use_secant || x1_opt.is_some();

  // Extract the function to find root of: expr or lhs - rhs for equations
  let func = build_find_root_func(&args[0], &[&var]);

  // Secant method when requested
  if use_secant {
    let max_iter = 100;
    let tol = 1e-15;
    let mut x_prev = x0;
    let mut x_curr = x1_opt.unwrap_or(x0 + 0.1);
    // A residual that is complex at a starting point (a logarithm of a
    // negative number) sends the whole iteration into the complex plane, as
    // in wolframscript, whose root then carries a vanishing imaginary part.
    let (Ok(f0), Ok(f1)) = (
      find_root_eval_at(&func, &var, x_prev),
      find_root_eval_at(&func, &var, x_curr),
    ) else {
      return complex_secant(&func, &var, x_prev, x_curr, step_monitor);
    };
    let mut f_prev = f0;
    let mut f_curr = f1;

    for _ in 0..max_iter {
      if f_curr.abs() < tol {
        break;
      }
      let denom = f_curr - f_prev;
      if denom.abs() < 1e-30 {
        break;
      }
      let step = f_curr * (x_curr - x_prev) / denom;
      // A step can overshoot out of the function's real domain — e.g. a
      // fractional power's base going negative — where evaluating the
      // residual there yields a complex number rather than a real one.
      // Halve the step toward the current point until it lands somewhere
      // evaluable, the same backtracking the damped-Newton path above
      // uses, instead of failing the whole solve over one bad iterate.
      let mut shrink = 1.0;
      let mut x_next = x_curr - step;
      let mut f_next = find_root_eval_at(&func, &var, x_next)
        .ok()
        .filter(|v| v.is_finite());
      for _ in 0..FIND_ROOT_MAX_BACKTRACKS {
        if f_next.is_some() {
          break;
        }
        shrink *= 0.5;
        x_next = x_curr - step * shrink;
        f_next = find_root_eval_at(&func, &var, x_next)
          .ok()
          .filter(|v| v.is_finite());
      }
      let Some(f_next_val) = f_next else {
        return Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression numerically".into(),
        ));
      };
      x_prev = x_curr;
      f_prev = f_curr;
      x_curr = x_next;
      f_curr = f_next_val;
      fire_step_monitor(step_monitor, std::slice::from_ref(&var), &[x_curr]);
    }

    let result_val = Expr::Real(x_curr);
    return Ok(Expr::List(
      vec![rule_expr(Expr::Identifier(var), result_val)].into(),
    ));
  }

  // Try symbolic derivative; fall back to numerical if unavailable
  let deriv_expr =
    crate::functions::calculus_ast::differentiate_expr(&func, &var)
      .ok()
      .filter(|d| !contains_unevaluated_d(d));

  // Newton's method, damped: a step that makes |f| worse is halved until it
  // does not, which is what lets a badly scaled function like `Exp[x] - 1000`
  // walk back from the huge first step plain Newton takes. Stops when |f| is
  // negligible or the iterate stops moving; running out of iterations first
  // reports FindRoot::cvmit and hands back the point reached, as
  // wolframscript does.
  let tol = 1e-15;
  let mut x = x0;
  let mut converged = false;
  let mut settled: Vec<(usize, f64)> = Vec::new();

  for k in 0..max_iter {
    let fx = find_root_eval_at(&func, &var, x)?;
    if fx.abs() < tol {
      converged = true;
      break;
    }
    // Compute derivative: symbolic if available, else numerical. A symbolic
    // derivative that does not reduce to a number *at this iterate* is no
    // better than having none — differentiating a non-smooth function leaves
    // heads like `Derivative[1, 0][Max][…]` standing, and a `Piecewise`
    // branch outside its condition gives `Indeterminate`. Wolfram falls back
    // to a difference quotient there rather than giving up on the solve, so
    // an unusable symbolic derivative drops through to the numerical branch.
    let symbolic_fpx = deriv_expr
      .as_ref()
      .and_then(|d| quietly(|| find_root_eval_at(d, &var, x)).ok())
      .filter(|v| v.is_finite());
    let fpx = if let Some(fpx) = symbolic_fpx {
      fpx
    } else {
      // 4th-order central difference for high-precision derivative
      let h = x.abs().max(1.0) * 1e-4;
      let fp1 = find_root_eval_at(&func, &var, x + h)?;
      let fm1 = find_root_eval_at(&func, &var, x - h)?;
      let fp2 = find_root_eval_at(&func, &var, x + 2.0 * h)?;
      let fm2 = find_root_eval_at(&func, &var, x - 2.0 * h)?;
      (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
    };
    let fpx = if fpx.abs() < 1e-30 {
      // Derivative too small — fall back to a finite-difference slope.
      let h = 1e-8;
      let fx_plus = find_root_eval_at(&func, &var, x + h)?;
      (fx_plus - fx) / h
    } else {
      fpx
    };
    if fpx.abs() < 1e-30 {
      // Nothing to divide by: wolframscript reports the singular Jacobian and
      // hands back the point it stalled at rather than failing outright.
      crate::emit_message(&format!(
        "FindRoot::jsing: Encountered a singular Jacobian at the point \
         {{{}}} = {{{}}}. Try perturbing the initial point(s).",
        var,
        crate::syntax::format_expr(
          &Expr::Real(x),
          crate::syntax::ExprForm::Output
        )
      ));
      converged = true;
      break;
    }
    let step = fx / fpx;
    // Backtrack while the step overshoots into a worse residual.
    let mut trial = x - step;
    let mut shrink = 1.0;
    for _ in 0..FIND_ROOT_MAX_BACKTRACKS {
      let ft = find_root_eval_at(&func, &var, trial)?;
      if ft.is_finite() && ft.abs() <= fx.abs() {
        break;
      }
      shrink *= 0.5;
      trial = x - step * shrink;
    }
    if let Some((lo, hi)) = bounds {
      trial = trial.clamp(lo, hi);
    }
    // A step this small means the iteration has settled, even for a function
    // whose residual cannot reach `tol` because of its scale. Note it and keep
    // going: the last iterations only jitter the final digit, and stopping here
    // instead would move the answer off the one wolframscript reports.
    if (trial - x).abs() <= f64::EPSILON * x.abs().max(1.0) {
      converged = true;
    }
    // Once settled the iteration only jitters the last digits, and each
    // iterate is a pure function of the previous one, so the sequence is
    // eventually periodic. Detect the period and jump to the iterate the
    // remaining iterations would end on instead of evaluating them all.
    if converged && step_monitor.is_none() {
      settled.push((k, x));
      if let Some(m) = settled
        .iter()
        .position(|(_, s)| s.to_bits() == trial.to_bits())
      {
        let start = settled[m].0;
        let period = k + 1 - start;
        x = settled[m + (max_iter - start) % period].1;
        break;
      }
    }
    x = trial;
    fire_step_monitor(step_monitor, std::slice::from_ref(&var), &[x]);
  }
  if !converged {
    crate::emit_message(&format!(
      "FindRoot::cvmit: Failed to converge to the requested accuracy or \
       precision within {max_iter} iterations."
    ));
  }

  // Format the result
  // Clean up -0.0
  let result_val = if x == 0.0 {
    Expr::Real(0.0)
  } else {
    Expr::Real(x)
  };

  // Re-evaluate the LHS so a held variable name with an OwnValue (e.g.
  // `x = "I am the result!"; FindRoot[…, {x, 1}]`) gets surfaced as that
  // bound value in the returned rule, matching wolframscript's
  // `{I am the result! -> 1.149…}` behaviour.
  let lhs_ident = Expr::Identifier(var);
  let lhs =
    crate::evaluator::evaluate_expr_to_expr(&lhs_ident).unwrap_or(lhs_ident);
  Ok(Expr::List(vec![rule_expr(lhs, result_val)].into()))
}

/// Convert FindRoot's first argument into the function whose root we
/// seek. For equations `lhs == rhs` this is `lhs - rhs`; otherwise the
/// expression is used directly. `vars` are every search variable the
/// caller is solving for — see `evaluate_with_vars_localized`.
///
/// FindRoot is `HoldAll` (so the iteration variable isn't looked up as an
/// OwnValue before the search starts), which means this raw `lhs - rhs`
/// still contains any held computation the equation wrote — most commonly
/// `D[f[x], x]`, an unevaluated derivative. Substituting a numeric trial
/// value straight into that raw form would land the number inside `D[…]`'s
/// variable slot (`D[f[3.8], 3.8]`) and fail with `D::ivar`. Evaluating once
/// here — with the search variable(s) still free symbols — resolves `D`,
/// `Integrate`, `Sum`, etc. into a plain closed-form expression first, the
/// same way `find_root_multivariate` already does per equation; only that
/// closed form is substituted into afterwards. An expression that fails to
/// evaluate (e.g. it stays opaque) is used as written, matching the
/// multivariate path's fallback.
fn build_find_root_func(arg: &Expr, vars: &[&str]) -> Expr {
  // FindRoot holds its arguments, so an equation passed by name
  // (`eq = f[x] == 0; FindRoot[eq, {x, x0}]`, typically a Module local)
  // arrives as a bare symbol. Resolve it first to reveal the `lhs == rhs`
  // structure the match below converts to `lhs - rhs`.
  let resolved;
  let arg = if matches!(arg, Expr::Identifier(_)) {
    resolved = evaluate_with_vars_localized(arg, vars);
    &resolved
  } else {
    arg
  };
  let raw = match arg {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      minus2(operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Equal" && fargs.len() == 2 =>
    {
      minus2(fargs[0].clone(), fargs[1].clone())
    }
    // An equation named by a symbol (`e1 = T + y == 5; FindRoot[{e1, ...}`)
    // only reveals itself as an `Equal` once evaluated; resolve it first,
    // then rewrite it like a literal equation.
    Expr::Identifier(_) => {
      let resolved = evaluate_with_vars_localized(arg, vars);
      return match &resolved {
        Expr::Identifier(_) => resolved,
        _ => build_find_root_func(&resolved, vars),
      };
    }
    // A call to a user-defined function that returns an equation
    // (`eq[i_] := y[i] == k[i] x[i]; FindRoot[{eq[1], ...}, ...]`) only
    // reveals the `Equal` once evaluated; resolve it, then rewrite it like a
    // literal equation.
    Expr::FunctionCall { .. } => {
      let resolved = evaluate_with_vars_localized(arg, vars);
      return match &resolved {
        Expr::Comparison { operators, .. }
          if operators.len() == 1 && operators[0] == ComparisonOp::Equal =>
        {
          build_find_root_func(&resolved, vars)
        }
        Expr::FunctionCall { name, .. } if name == "Equal" => {
          build_find_root_func(&resolved, vars)
        }
        _ => resolved,
      };
    }
    // Any other held form (`eq /. rules`, `If[...]`, `Which[...]`, ...) may
    // only reveal an `Equal` once evaluated; resolve it, then rewrite that
    // equation like a literal one.
    other => {
      let resolved = evaluate_with_vars_localized(other, vars);
      return match &resolved {
        Expr::Comparison { operators, .. }
          if operators.len() == 1 && operators[0] == ComparisonOp::Equal =>
        {
          build_find_root_func(&resolved, vars)
        }
        Expr::FunctionCall { name, .. } if name == "Equal" => {
          build_find_root_func(&resolved, vars)
        }
        _ => resolved,
      };
    }
  };
  evaluate_with_vars_localized(&raw, vars)
}

/// Evaluate `expr` with each of `vars`' global bindings temporarily
/// removed, restoring them all afterward regardless of the outcome —
/// `Block[vars, expr]`'s scoping, applied to a single speculative
/// evaluation rather than a full dynamic-scope body.
///
/// `build_find_root_func` needs exactly this: FindRoot's search variable(s)
/// must be free while the equation's held computation resolves (so
/// `D[f[x], x]` becomes a closed form in `x`), but must not pick up
/// whatever unrelated global value `x` happens to carry — `x = "prior
/// result"; FindRoot[f[x] == 0, {x, x0}]` searches fresh, the same as
/// Wolfram's `HoldAll`, rather than evaluating `f["prior result"]` and
/// reporting `FindRoot::nlnum`.
fn evaluate_with_vars_localized(expr: &Expr, vars: &[&str]) -> Expr {
  let saved: Vec<(&str, Option<StoredValue>)> = vars
    .iter()
    .map(|v| (*v, ENV.with(|e| e.borrow_mut().remove(*v))))
    .collect();
  let result = crate::evaluator::evaluate_expr_to_expr(expr)
    .unwrap_or_else(|_| expr.clone());
  for (v, prev) in saved {
    ENV.with(|e| {
      let mut env = e.borrow_mut();
      match prev {
        Some(val) => {
          env.insert(v.to_string(), val);
        }
        None => {
          env.remove(v);
        }
      }
    });
  }
  result
}

/// Try to evaluate `expr` to a complex `(re, im)` pair using f64
/// arithmetic. Returns None when the expression isn't fully numeric.
fn try_extract_complex_f64(expr: &Expr) -> Option<(f64, f64)> {
  let n_result = n_ast(std::slice::from_ref(expr)).ok()?;
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
      for a in args {
        let (r, i) = expr_to_complex_f64(a)?;
        re += r;
        im += i;
      }
      Some((re, im))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut re = 1.0;
      let mut im = 0.0;
      for a in args {
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
    call("Complex", vec![Expr::Real(re), Expr::Real(im)])
  };
  let substituted = crate::syntax::substitute_variable(expr, var, &value);
  let evaled = crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
  // First try the natural form; if not yet a Complex/Real, push through
  // N[] which collapses things like Sin[Complex[…]] into a numeric form.
  if let Some(c) = expr_to_complex_f64(&evaled) {
    return Some(c);
  }
  let n_result = n_ast(&[evaled]).ok()?;
  expr_to_complex_f64(&n_result)
}

/// Newton's method on the complex plane. Mirrors the real-only path
/// but works in (re, im) pairs throughout. Uses a numerical derivative
/// when the symbolic derivative isn't available.
/// The real-valued search failed because an iterate left the function's real
/// domain (`x Hypergeometric2F1[1/2, 1/2, 2, x]` past x = 1): Wolfram's
/// Newton iteration simply continues in the complex plane. Retries a
/// single-variable `FindRoot[f, {x, x0}]` that way when the function is a
/// number at the (real) start; None when that does not apply.
pub fn find_root_complex_retry(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::List(spec) = args.get(1)? else {
    return None;
  };
  let [Expr::Identifier(var), start] = spec.as_slice() else {
    return None;
  };
  if matches!(&args[0], Expr::List(eqs) if eqs.len() != 1) {
    return None;
  }
  let (re0, im0) = try_extract_complex_f64(start)?;
  let func = build_find_root_func(&args[0], &[var]);
  find_root_eval_complex_at(&func, var, re0, im0)?;
  let deriv = crate::functions::calculus_ast::differentiate_expr(&func, var)
    .ok()
    .map(simplify);
  Some(find_root_complex_newton(
    &func,
    deriv.as_ref(),
    var,
    re0,
    im0,
  ))
}

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
  let mut iterations = 0;
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
    let Some((fr, fi)) = find_root_eval_complex_at(func, var, re, im) else {
      return Err(InterpreterError::EvaluationError(
        "FindRoot: cannot evaluate expression at complex point".into(),
      ));
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
      let Some((fr_p, fi_p)) = find_root_eval_complex_at(func, var, re + h, im)
      else {
        return Err(InterpreterError::EvaluationError(
          "FindRoot: cannot evaluate expression for derivative".into(),
        ));
      };
      ((fr_p - fr) / h, (fi_p - fi) / h)
    };
    let Some((sr, si)) = cdiv(fr, fi, dr, di) else {
      return Err(InterpreterError::EvaluationError(
        "FindRoot: derivative is zero, cannot converge".into(),
      ));
    };
    // A step below the last bit of the iterate means it has settled even
    // though the residual's rounding noise stays above `tol`.
    if cabs(sr, si) <= f64::EPSILON * cabs(re, im).max(1.0) {
      re -= sr;
      im -= si;
      break;
    }
    // Damped like the real iteration: a step that makes |f| worse (or
    // leaves the function's domain) is halved until it does not.
    let current = cabs(fr, fi);
    let mut lambda = 1.0;
    let mut accepted = false;
    for _ in 0..40 {
      let (nr, ni) = (re - lambda * sr, im - lambda * si);
      if let Some((gr, gi)) = find_root_eval_complex_at(func, var, nr, ni)
        && cabs(gr, gi) <= current
      {
        re = nr;
        im = ni;
        accepted = true;
        break;
      }
      lambda *= 0.5;
    }
    if !accepted {
      re -= sr;
      im -= si;
    }
    iterations += 1;
  }
  if iterations == max_iter
    && find_root_eval_complex_at(func, var, re, im)
      .is_none_or(|(fr, fi)| cabs(fr, fi) >= tol)
  {
    crate::emit_message(&format!(
      "FindRoot::cvmit: Failed to converge to the requested accuracy or precision within {max_iter} iterations."
    ));
  }
  // Build the complex result. Drop the imaginary part if it collapsed
  // to zero so the rule reads like a real solution. Otherwise build
  // `re + im*I` (or `re - |im|*I`) so it formats like wolframscript.
  let value = if im.abs() < 1e-14 {
    Expr::Real(re)
  } else {
    let im_term = times2(Expr::Real(im.abs()), id_expr("I"));
    let combined = if im >= 0.0 {
      plus2(Expr::Real(re), im_term)
    } else {
      minus2(Expr::Real(re), im_term)
    };
    crate::evaluator::evaluate_expr_to_expr(&combined).unwrap_or(combined)
  };
  let lhs_ident = id_expr(var);
  let lhs =
    crate::evaluator::evaluate_expr_to_expr(&lhs_ident).unwrap_or(lhs_ident);
  Ok(Expr::List(vec![rule_expr(lhs, value)].into()))
}

/// A FindRoot residual that evaluates to a one-element list (e.g. the
/// derivative of a replacement by a list of solution rules) stands for the
/// residual itself.
fn unwrap_singleton_list(e: Expr) -> Expr {
  // Every level: a residual such as `{{x^2}} == 4` is one number.
  match &e {
    Expr::List(items) if items.len() == 1 => {
      unwrap_singleton_list(items[0].clone())
    }
    _ => e,
  }
}

/// Evaluate an expression numerically at a specific value of var.
/// The secant iteration for `func` in `var` in complex arithmetic, from the
/// real starting points `a` and `b`. The root keeps its imaginary part
/// unless that is exactly zero.
fn complex_secant(
  func: &Expr,
  var: &str,
  a: f64,
  b: f64,
  step_monitor: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  type C = (f64, f64);
  let eval = |x: C| -> Option<C> {
    let point = if x.1 == 0.0 {
      Expr::Real(x.0)
    } else {
      crate::evaluator::evaluate_expr_to_expr(&call(
        "Complex",
        vec![Expr::Real(x.0), Expr::Real(x.1)],
      ))
      .ok()?
    };
    let value = crate::evaluator::evaluate_expr_to_expr(&call(
      "N",
      vec![crate::syntax::substitute_variable(func, var, &point)],
    ))
    .ok()?;
    crate::functions::math_ast::numeric_utils::try_extract_complex_f64(
      &unwrap_singleton_list(value),
    )
    .filter(|(r, i)| r.is_finite() && i.is_finite())
  };
  let sub = |p: C, q: C| (p.0 - q.0, p.1 - q.1);
  let mul = |p: C, q: C| (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0);
  let div = |p: C, q: C| {
    let d = q.0 * q.0 + q.1 * q.1;
    ((p.0 * q.0 + p.1 * q.1) / d, (p.1 * q.0 - p.0 * q.1) / d)
  };
  let abs = |p: C| p.0.hypot(p.1);
  let (mut x_prev, mut x_curr) = ((a, 0.0), (b, 0.0));
  let (Some(mut f_prev), Some(mut f_curr)) = (eval(x_prev), eval(x_curr))
  else {
    crate::emit_message(
      "FindRoot::nlnum: The function value is not a number at the starting point.",
    );
    return Ok(unevaluated("FindRoot", &[]));
  };
  for _ in 0..100 {
    if abs(f_curr) < 1e-15 {
      break;
    }
    let denom = sub(f_curr, f_prev);
    if abs(denom) < 1e-30 {
      break;
    }
    // Damped: a step is halved until the residual does not grow (an
    // undamped complex secant wanders off from a start like this one).
    let step = div(mul(f_curr, sub(x_curr, x_prev)), denom);
    let mut scale = 1.0;
    let mut next = None;
    for _ in 0..FIND_ROOT_MAX_BACKTRACKS {
      let x_try = sub(x_curr, (step.0 * scale, step.1 * scale));
      if let Some(f_try) = eval(x_try) {
        next = Some((x_try, f_try));
        if abs(f_try) < abs(f_curr) {
          break;
        }
      }
      scale *= 0.5;
    }
    let Some((x_next, f_next)) = next else { break };
    (x_prev, f_prev) = (x_curr, f_curr);
    (x_curr, f_curr) = (x_next, f_next);
    fire_step_monitor(step_monitor, &[var.to_string()], &[x_curr.0]);
  }
  let value = if x_curr.1 == 0.0 {
    Expr::Real(x_curr.0)
  } else {
    crate::evaluator::evaluate_expr_to_expr(&call(
      "Complex",
      vec![Expr::Real(x_curr.0), Expr::Real(x_curr.1)],
    ))?
  };
  Ok(Expr::List(
    vec![rule_expr(Expr::Identifier(var.to_string()), value)].into(),
  ))
}

fn find_root_eval_at(
  expr: &Expr,
  var: &str,
  x: f64,
) -> Result<f64, InterpreterError> {
  let substituted =
    crate::syntax::substitute_variable(expr, var, &Expr::Real(x));
  let evaled = unwrap_singleton_list(crate::evaluator::evaluate_expr_to_expr(
    &substituted,
  )?);
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
      let n_result = n_ast(&[evaled])?;
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

/// Run `probe` with every message it emits discarded.
///
/// FindRoot's symbolic derivative is speculative: when it does not reduce to a
/// number the iteration falls back to a difference quotient, so whatever the
/// attempt complained about on the way (`D::ivar` from differentiating a
/// user function at an already-substituted point, a division by zero in a
/// branch that is never used, …) is internal bookkeeping. wolframscript
/// reports none of it, so neither does Woxi.
fn quietly<T>(probe: impl FnOnce() -> T) -> T {
  let snapshot = crate::snapshot_warnings();
  crate::push_quiet();
  let result = probe();
  crate::pop_quiet();
  crate::restore_warnings(snapshot);
  result
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
          let n_expr = call1("N", evaled.clone());
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

/// True when `e` is a valid FindRoot search variable: a plain symbol, an
/// indexed variable such as `u[1]` or `a[2, 3]` — a symbol applied to
/// literal numeric or string indices — or a curried chain of those such as
/// `T[0][t]`. Collocation methods for boundary-value problems name one
/// unknown per grid point this way (`FindRoot[{eqns}, {{u[1], 0.5}, {u[2],
/// 0.5}, ...}]`), and time-dependent families use the curried form
/// (`FindRoot[eqns, {T[0][t], 354.3}, {T[1][t], 357.7}, ...]`); real
/// Wolfram accepts all of these.
fn is_findroot_var_expr(e: &Expr) -> bool {
  match e {
    Expr::Identifier(_) => true,
    Expr::FunctionCall { .. } | Expr::CurriedCall { .. } => {
      flatten_findroot_var(e).is_some()
    }
    _ => false,
  }
}

/// A hashable key for a FindRoot indexed-variable's literal or symbolic
/// arguments (`u[1]` → `[Int(1)]`, `a[2, "x"]` → `[Int(2), Str("x")]`,
/// `T[0][t]` → `[Int(0), Sym("t")]`).
#[derive(PartialEq, Eq, Hash, Clone)]
enum FindRootIndexKey {
  Int(i128),
  RealBits(u64),
  Str(String),
  Sym(String),
}

fn find_root_index_key(e: &Expr) -> Option<FindRootIndexKey> {
  match e {
    Expr::Integer(n) => Some(FindRootIndexKey::Int(*n)),
    Expr::Real(r) => Some(FindRootIndexKey::RealBits(r.to_bits())),
    Expr::String(s) => Some(FindRootIndexKey::Str(s.clone())),
    Expr::Identifier(s) => Some(FindRootIndexKey::Sym(s.clone())),
    _ => None,
  }
}

/// Flatten an indexed FindRoot variable — a plain `u[1]` or a curried chain
/// of those like `T[0][t]` — into its base name and the sequence of index
/// keys across every level, one inner `Vec` per application level so that
/// a single multi-arg call (`T[0, 1]`) and a curried chain of single-arg
/// calls (`T[0][1]`) — structurally distinct expressions in Wolfram — key
/// differently instead of colliding on the same flattened index sequence.
/// `None` when any level isn't a recognized indexed-variable shape (e.g. an
/// empty-arg call, or a level whose argument isn't a literal or symbol).
fn flatten_findroot_var(
  e: &Expr,
) -> Option<(&str, Vec<Vec<FindRootIndexKey>>)> {
  match e {
    Expr::FunctionCall { name, args } => {
      if args.is_empty() {
        return None;
      }
      let keys = args
        .iter()
        .map(find_root_index_key)
        .collect::<Option<Vec<_>>>()?;
      Some((name.as_str(), vec![keys]))
    }
    Expr::CurriedCall { func, args } => {
      if args.is_empty() {
        return None;
      }
      let (name, mut levels) = flatten_findroot_var(func)?;
      let keys = args
        .iter()
        .map(find_root_index_key)
        .collect::<Option<Vec<_>>>()?;
      levels.push(keys);
      Some((name, levels))
    }
    _ => None,
  }
}

/// Replace every var in `vars` (matched by exact structural equality —
/// including indexed variables like `u[1]`) with the corresponding
/// expression in `replacements`, in a single pass over `expr`.
///
/// `find_root_multivariate` uses this once, up front, to rename every
/// search variable — plain or indexed — to a fresh plain symbol, so the
/// rest of the solve (symbolic differentiation, localization, per-iteration
/// substitution) can reuse the existing name-based machinery unchanged
/// instead of paying a structural-match cost at every Newton step. Doing
/// the rename with a per-node hashmap lookup, in one pass, also keeps this
/// one-off step itself linear in the equations' size rather than linear in
/// the number of variables times their size.
///
/// Covers the constructs a FindRoot equation can contain: arithmetic,
/// comparisons and their containers. Any other construct is left as
/// written — the caller detects an unrenamed variable when the result
/// fails to reduce to a number, rather than this silently skipping it.
fn find_root_rename_vars(
  expr: &Expr,
  vars: &[Expr],
  replacements: &[Expr],
) -> Expr {
  let mut id_map: std::collections::HashMap<&str, &Expr> =
    std::collections::HashMap::with_capacity(vars.len());
  let mut idx_map: std::collections::HashMap<
    (&str, Vec<Vec<FindRootIndexKey>>),
    &Expr,
  > = std::collections::HashMap::new();
  for (v, r) in vars.iter().zip(replacements) {
    match v {
      Expr::Identifier(name) => {
        id_map.insert(name.as_str(), r);
      }
      Expr::FunctionCall { .. } | Expr::CurriedCall { .. } => {
        if let Some((name, keys)) = flatten_findroot_var(v) {
          idx_map.insert((name, keys), r);
        }
      }
      _ => {}
    }
  }
  find_root_rename_walk(expr, &id_map, &idx_map)
}

fn find_root_rename_walk(
  expr: &Expr,
  id_map: &std::collections::HashMap<&str, &Expr>,
  idx_map: &std::collections::HashMap<
    (&str, Vec<Vec<FindRootIndexKey>>),
    &Expr,
  >,
) -> Expr {
  match expr {
    Expr::Identifier(name) => match id_map.get(name.as_str()) {
      Some(r) => (*r).clone(),
      None => expr.clone(),
    },
    Expr::FunctionCall { name, args } => {
      if let Some((flat_name, keys)) = flatten_findroot_var(expr)
        && let Some(r) = idx_map.get(&(flat_name, keys))
      {
        return (*r).clone();
      }
      call(
        name.as_str(),
        args
          .iter()
          .map(|a| find_root_rename_walk(a, id_map, idx_map))
          .collect(),
      )
    }
    Expr::CurriedCall { func, args } => {
      if let Some((flat_name, keys)) = flatten_findroot_var(expr)
        && let Some(r) = idx_map.get(&(flat_name, keys))
      {
        return (*r).clone();
      }
      Expr::CurriedCall {
        func: Box::new(find_root_rename_walk(func, id_map, idx_map)),
        args: args
          .iter()
          .map(|a| find_root_rename_walk(a, id_map, idx_map))
          .collect(),
      }
    }
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| find_root_rename_walk(a, id_map, idx_map))
        .collect(),
    ),
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      find_root_rename_walk(left, id_map, idx_map),
      find_root_rename_walk(right, id_map, idx_map),
    ),
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(find_root_rename_walk(operand, id_map, idx_map)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|a| find_root_rename_walk(a, id_map, idx_map))
        .collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => Expr::CompoundExpr(
      exprs
        .iter()
        .map(|a| find_root_rename_walk(a, id_map, idx_map))
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => rule_expr(
      find_root_rename_walk(pattern, id_map, idx_map),
      find_root_rename_walk(replacement, id_map, idx_map),
    ),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => rule_expr(
      find_root_rename_walk(pattern, id_map, idx_map),
      find_root_rename_walk(replacement, id_map, idx_map),
    ),
    other => other.clone(),
  }
}

/// Evaluate `expr` to an f64 with every variable in `vars` bound to the
/// corresponding value in `vals`, in a single substitution pass.
///
/// Builds a fresh `vars`-length binding table on every call, so it is only
/// cheap when called a handful of times per point. The Newton loop below
/// evaluates a whole equation *and* Jacobian row at the same point `x` —
/// call [`find_root_eval_multivar_at`] there instead, sharing one binding
/// table across all of those calls (see its doc comment).
fn find_root_eval_multivar(
  expr: &Expr,
  vars: &[String],
  vals: &[f64],
) -> Result<f64, InterpreterError> {
  let reals: Vec<Expr> = vals.iter().map(|&x| Expr::Real(x)).collect();
  let bindings: Vec<(&str, &Expr)> =
    vars.iter().map(String::as_str).zip(reals.iter()).collect();
  find_root_eval_multivar_at(expr, &bindings)
}

/// Same as [`find_root_eval_multivar`], but takes an already-built binding
/// table instead of rebuilding one from `vars`/`vals`.
///
/// A Newton iteration over `n` variables evaluates `n` equations plus an
/// `n`×`n` Jacobian at the *same* point — up to `n^2 + n` calls that all
/// need the identical `var -> value` table. Rebuilding that O(n) table
/// inside every one of those calls (as [`find_root_eval_multivar`] does)
/// turns one Newton iteration into O(n^3) work regardless of how sparse
/// the equations are, since the rebuild cost is paid even when a given
/// equation or Jacobian entry mentions only a few variables. Building the
/// table once per iteration and sharing it here keeps that cost O(n) per
/// iteration instead.
fn find_root_eval_multivar_at(
  expr: &Expr,
  bindings: &[(&str, &Expr)],
) -> Result<f64, InterpreterError> {
  let e = crate::syntax::substitute_variables(expr, bindings);
  let evaled =
    unwrap_singleton_list(crate::evaluator::evaluate_expr_to_expr(&e)?);
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
      let n_result = n_ast(&[evaled])?;
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
  max_iter: usize,
  step_monitor: Option<&Expr>,
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Variables and starting points. A spec's variable is either a plain
  // symbol or an indexed variable like `u[1]` — see `is_findroot_var_expr`.
  // A three-element spec `{var, x0, x1}` carries a second point used as a
  // per-variable finite-difference step hint for the numerical Jacobian,
  // mirroring the secant hint the single-variable form uses.
  let mut raw_vars: Vec<Expr> = Vec::new();
  let mut x: Vec<f64> = Vec::new();
  let mut step_hint: Vec<Option<f64>> = Vec::new();
  let mut bounds: Vec<Option<(f64, f64)>> = Vec::new();
  for spec in specs {
    if let Expr::List(p) = spec {
      if !is_findroot_var_expr(&p[0]) {
        return Err(InterpreterError::EvaluationError(
          "FindRoot: variable must be a symbol".into(),
        ));
      }
      raw_vars.push(p[0].clone());
      let x0 = find_root_eval_number(&p[1])?;
      x.push(x0);
      step_hint.push(if p.len() == 3 {
        let x1 = find_root_eval_number(&p[2])?;
        let h = (x1 - x0).abs();
        if h > 1e-14 { Some(h) } else { None }
      } else {
        None
      });
      // `{var, x0, xmin, xmax}` keeps the search within [xmin, xmax].
      bounds.push(if p.len() == 4 {
        let lo = find_root_eval_number(&p[2])?;
        let hi = find_root_eval_number(&p[3])?;
        Some((lo.min(hi), lo.max(hi)))
      } else {
        None
      });
    }
  }
  let n = raw_vars.len();
  let clamp_to_bounds = |v: Vec<f64>| -> Vec<f64> {
    v.into_iter()
      .zip(&bounds)
      .map(|(xj, b)| match b {
        Some((lo, hi)) => xj.clamp(*lo, *hi),
        None => xj,
      })
      .collect()
  };
  let mut x = clamp_to_bounds(x);

  // `StepMonitor :> expr` fires after every accepted step with the plain
  // search variables bound to the new point. An indexed variable like
  // `u[1]` has no symbol to bind, so it is left out.
  let monitor_names: Vec<(usize, String)> = raw_vars
    .iter()
    .enumerate()
    .filter_map(|(i, v)| match v {
      Expr::Identifier(name) => Some((i, name.clone())),
      _ => None,
    })
    .collect();
  let fire_monitor = |x: &[f64]| {
    if step_monitor.is_some() {
      let names: Vec<String> =
        monitor_names.iter().map(|(_, n)| n.clone()).collect();
      let vals: Vec<f64> = monitor_names.iter().map(|(i, _)| x[*i]).collect();
      fire_step_monitor(step_monitor, &names, &vals);
    }
  };

  // Only plain-symbol variables can carry a stale global binding under a
  // literal name — an indexed variable like `u[1]` has no such binding to
  // remove.
  let raw_var_refs: Vec<&str> = raw_vars
    .iter()
    .filter_map(|v| match v {
      Expr::Identifier(name) => Some(name.as_str()),
      _ => None,
    })
    .collect();
  // The equations are often built and named separately (`sys =
  // Join[{bc1, ...}, Table[Eq[...], ...]] // Flatten; FindRoot[sys, ...]`)
  // rather than written literally, so `eqns_arg` may reach here as a bare
  // identifier rather than a literal list. Resolve it (search variables
  // localized, see `evaluate_with_vars_localized`) to reveal the list of
  // equations, so a variable that happens to carry a stale global value
  // elsewhere doesn't leak into this evaluation. FindRoot holds its
  // arguments, so this is the first chance to do so.
  let eqns_owned;
  let eqns_arg: &Expr = if matches!(eqns_arg, Expr::List(_)) {
    eqns_arg
  } else {
    eqns_owned = evaluate_with_vars_localized(eqns_arg, &raw_var_refs);
    &eqns_owned
  };

  // Each equation is built (`Equal` rewritten to a `lhs - rhs` difference,
  // then fully evaluated with the plain-symbol search variables' global
  // bindings held free — the indexed ones were never bound under a literal
  // name to begin with, see above) *before* renaming, not after. A
  // stage-by-stage model typically writes its equations in terms of
  // helper functions (`H[i]`, `h[i]`, ...) rather than the search
  // variables directly — a literal `T[5]` only appears once such a
  // function's Module-scoped `SetDelayed` body is expanded during
  // evaluation, not in the pre-evaluation source `eqns_arg` a syntactic
  // rename pass walks. Renaming first therefore missed every occurrence
  // reached only through such a call: the built equation still carried an
  // unrenamed, unbound `T[5]`, which could not be substituted with the
  // search point's numeric value and made the residual fail to evaluate
  // to a number at all (`FindRoot::nlnum`) instead of merely being
  // differentiated incorrectly. Evaluating first and renaming the
  // resulting closed form instead catches every occurrence, wherever it
  // came from.
  let raw_eq_list: Vec<Expr> = match eqns_arg {
    Expr::List(es) => es.to_vec(),
    other => vec![other.clone()],
  };
  // An equation between lists (`{a, b} == {c, d}`) is one equation per
  // component, as Equal threads over lists.
  let mut evaluated_eqns: Vec<Expr> = raw_eq_list
    .iter()
    .map(|e| build_find_root_func(e, &raw_var_refs))
    .flat_map(|f| match &f {
      Expr::List(items) => items.to_vec(),
      _ => vec![f],
    })
    .collect();
  // A function that only evaluates at numbers (`eqns[x_?NumericQ, …] :=
  // {…}`) stays one unevaluated call here; its value at the starting point
  // shows how many equations it stands for, and each is a component of it.
  if let [black_box] = evaluated_eqns.as_slice()
    && n > 1
  {
    let at_start = raw_vars
      .iter()
      .zip(&x)
      .fold(black_box.clone(), |e, (v, x0)| {
        substitute_expr(&e, v, &Expr::Real(*x0))
      });
    if let Ok(Expr::List(ref values)) =
      quietly(|| crate::evaluator::evaluate_expr_to_expr(&at_start))
      && values.len() == n
    {
      let black_box = black_box.clone();
      evaluated_eqns = (1..=n)
        .map(|i| {
          call("Part", vec![black_box.clone(), Expr::Integer(i as i128)])
        })
        .collect();
    }
  }

  // Every search variable — plain or indexed — is then renamed to a fresh
  // plain symbol. Collocation methods can name hundreds of `u[i]`
  // variables for a dense system; treating them as plain symbols from here
  // on lets the rest of the solve reuse the existing name-based
  // substitution (`substitute_variables`, one pass over the equation) and
  // symbolic differentiation (`differentiate_expr`) instead of a much
  // slower structural-match or finite-difference fallback per indexed
  // variable. The rename is reversed only in the final `var -> value`
  // rules.
  let vars: Vec<String> = (0..n).map(|i| format!("$FindRootVar{i}$")).collect();
  let syn_idents: Vec<Expr> =
    vars.iter().map(|s| Expr::Identifier(s.clone())).collect();
  let eqns: Vec<Expr> = evaluated_eqns
    .iter()
    .map(|e| find_root_rename_vars(e, &raw_vars, &syn_idents))
    .collect();
  if eqns.len() != n {
    crate::emit_message(&format!(
      "FindRoot::nveq: The number of equations does not match the number of variables in {}.",
      expr_to_string(&unevaluated("FindRoot", call_args))
    ));
    return Ok(unevaluated("FindRoot", call_args));
  }

  // Jacobian J[i][j] = d f_i / d x_j (symbolic where possible — every
  // variable is now a plain symbol, so this applies uniformly to indexed
  // variables too). An entry that cannot be differentiated symbolically
  // (e.g. the equation still contains an opaque function call) is left
  // `None` and approximated by a central finite difference at each
  // iteration point.
  //
  // A large system built from per-index equations (a stage-by-stage
  // process model, a PDE discretization, ...) has a Jacobian that is
  // mostly zero: equation `i` typically only involves a handful of the
  // `n` search variables. Actually differentiating (and then simplifying)
  // every one of the n*n pairs costs time proportional to the equation's
  // full expression size regardless of whether `v` appears in it, so for
  // hundreds of variables that quadratic blowup dominates the runtime.
  // `is_constant_wrt` proves the zero case with a plain structural
  // presence check — no expression construction or simplification — so
  // checking it first turns the dense n*n differentiate+simplify pass
  // into one that costs full differentiation only for the (sparse) pairs
  // that actually need it.
  //
  // An equation that is itself huge (e.g. an equation-of-state model whose
  // closed form expands to thousands of nodes) is the opposite case: its
  // symbolic derivative is larger still and `simplify` on it runs for
  // minutes, far longer than the finite-difference/Broyden iteration that
  // the `None` fallback gives. Such equations are not differentiated
  // symbolically at all.
  const MAX_SYMBOLIC_JACOBIAN_LEAVES: usize = 2000;
  // Equations of plain arithmetic are compiled once: their values and
  // exact gradients (forward-mode differentiation over the compiled tree)
  // then cost a pass over each equation, where the symbolic Jacobian of a
  // large stage-by-stage model took most of a minute to differentiate and
  // simplify, and every entry was re-evaluated as an expression.
  // Only for larger systems: a small one keeps the expression evaluation,
  // whose last digits agree with Wolfram's.
  let compiled: Option<Vec<NumNode>> = if n >= 8 {
    eqns
      .iter()
      .map(|f| compile_numeric(f, &vars).filter(|node| !node.has_fallback()))
      .collect()
  } else {
    None
  };
  let mut jac: Vec<Vec<Option<Expr>>> = Vec::with_capacity(n);
  for f in &eqns {
    if compiled.is_some() {
      // Not used: the compiled gradient stands for every entry.
      jac.push(vec![Some(Expr::Integer(0)); n]);
      continue;
    }
    let mut row = Vec::with_capacity(n);
    let too_large =
      super::simplify::leaf_count(f) > MAX_SYMBOLIC_JACOBIAN_LEAVES;
    for v in &vars {
      let d = if crate::functions::calculus_ast::is_constant_wrt(f, v) {
        Some(Expr::Integer(0))
      } else if too_large {
        None
      } else {
        crate::functions::calculus_ast::differentiate_expr(f, v)
          .map(simplify)
          .ok()
          .filter(|d| !contains_unevaluated_d(d))
      };
      row.push(d);
    }
    jac.push(row);
  }

  // Every entry that has no symbolic derivative needs two extra function
  // evaluations (central difference) per iteration to refill it — for an
  // opaque function like one wrapping `NDSolve` that cost dominates and a
  // plain Newton iteration recomputing the whole Jacobian every step turns
  // into hundreds of expensive re-solves. Broyden's quasi-Newton method
  // amortizes that: the Jacobian is built by finite difference just once,
  // then cheaply rank-1 updated from the step already taken, so each
  // further iteration costs exactly the `n` evaluations needed for the
  // residual itself.
  let needs_finite_diff = jac
    .iter()
    .any(|row| row.iter().any(std::option::Option::is_none));
  let tol = 1e-13;

  // The point with the smallest residual seen so far, and that residual.
  // A large, ill-conditioned system (e.g. a PDE collocation matrix, whose
  // condition number grows with the grid size) can have an achievable
  // residual floor well above `tol` — f64 rounding in the linear solve
  // limits it, not the loop's remaining iterations. Once the residual
  // stops improving from one iteration to the next, every further
  // iteration is spending equation/Jacobian evaluations (each O(n)-plus)
  // on numerical noise, not progress: burning through the rest of
  // `max_iter` this way costs seconds to minutes on a large system
  // without moving the answer. Tracking the best iterate and stopping the
  // moment progress stalls returns that already-best point immediately
  // instead, while leaving genuinely (even slowly) converging cases to
  // keep iterating exactly as before. Both branches below update this.
  let mut best_x = x.clone();
  // Set by both branches below before their first read.
  let mut best_resid;
  // The two-point spec's step hint only seeds the very first Jacobian
  // estimate — reusing it as a fixed-width central difference for the rest
  // of the solve would keep probing the same offset from whatever point the
  // iteration has moved to (potentially outside the domain the caller
  // implied was valid) and would stay just as coarse once the iterate has
  // moved far from `x0`, where the adaptive default is a better estimate.
  let no_hint: Vec<Option<f64>> = vec![None; n];

  let eval_jacobian_at = |x: &[f64],
                          bindings: &[(&str, &Expr)],
                          hint: &[Option<f64>]|
   -> Result<Vec<Vec<f64>>, InterpreterError> {
    let mut jm = vec![vec![0.0; n]; n];
    if let Some(nodes) = &compiled {
      for (i, node) in nodes.iter().enumerate() {
        let (_, grad) = node.eval_dual(x);
        for (j, d) in grad {
          jm[i][j] = d;
        }
      }
      if jm.iter().flatten().all(|v| v.is_finite()) {
        return Ok(jm);
      }
      jm = vec![vec![0.0; n]; n];
    }
    for (i, row) in jac.iter().enumerate() {
      for (j, dij) in row.iter().enumerate() {
        let entry = match dij {
          Some(d) => quietly(|| find_root_eval_multivar_at(d, bindings))
            .ok()
            .filter(|v| v.is_finite()),
          None => None,
        };
        jm[i][j] = if let Some(v) = entry {
          v
        } else {
          let h = hint[j].unwrap_or(1e-7 * x[j].abs().max(1.0));
          let mut xp = x.to_vec();
          xp[j] += h;
          let mut xm = x.to_vec();
          xm[j] -= h;
          let fp = find_root_eval_multivar(&eqns[i], &vars, &xp)?;
          let fm = find_root_eval_multivar(&eqns[i], &vars, &xm)?;
          (fp - fm) / (2.0 * h)
        };
      }
    }
    Ok(jm)
  };

  if needs_finite_diff {
    // Broyden's method: one finite-difference Jacobian up front, then a
    // rank-1 update from each accepted step. A step is only accepted if it
    // improves the residual (backtracking, as the single-variable secant
    // path above does); a rank-1 update built from a rejected long step
    // would corrupt the Jacobian estimate rather than merely take a slow
    // step, so a run of failed backtracks instead re-seeds the Jacobian
    // with a fresh finite difference at the current point.
    let residual_norm =
      |fv: &[f64]| fv.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let eval_all = |x: &[f64]| -> Result<Vec<f64>, InterpreterError> {
      let mut fv = vec![0.0; n];
      for (i, f) in eqns.iter().enumerate() {
        fv[i] = find_root_eval_multivar(f, &vars, x)?;
      }
      Ok(fv)
    };
    let mut fv = eval_all(&x)?;
    let reals: Vec<Expr> = x.iter().map(|&v| Expr::Real(v)).collect();
    let bindings: Vec<(&str, &Expr)> =
      vars.iter().map(String::as_str).zip(reals.iter()).collect();
    let mut jm = eval_jacobian_at(&x, &bindings, &step_hint)?;
    best_resid = residual_norm(&fv);
    best_x.clone_from(&x);
    let mut converged = best_resid < tol;
    let mut iter_count = 0;
    while !converged && iter_count < max_iter {
      iter_count += 1;
      let mut progressed = false;
      for attempt in 0..2 {
        let neg_f: Vec<f64> = fv.iter().map(|v| -v).collect();
        let Some(delta) = find_root_solve_linear(jm.clone(), neg_f) else {
          if attempt == 0 {
            let reals: Vec<Expr> = x.iter().map(|&v| Expr::Real(v)).collect();
            let bindings: Vec<(&str, &Expr)> =
              vars.iter().map(String::as_str).zip(reals.iter()).collect();
            jm = eval_jacobian_at(&x, &bindings, &no_hint)?;
            continue;
          }
          break;
        };
        let base_norm = residual_norm(&fv);
        let mut shrink = 1.0;
        let mut accepted = None;
        for _ in 0..FIND_ROOT_MAX_BACKTRACKS {
          let x_trial: Vec<f64> = x
            .iter()
            .zip(&delta)
            .map(|(&xj, &dj)| xj + dj * shrink)
            .collect();
          let x_trial = clamp_to_bounds(x_trial);
          // An opaque residual (e.g. one wrapping NDSolve) can fail outright
          // at a wild trial point rather than merely return a non-finite
          // value — treat that the same as a rejected trial (shrink and
          // retry) instead of aborting the whole FindRoot call over one bad
          // backtrack guess.
          let fv_trial =
            eval_all(&x_trial).unwrap_or_else(|_| vec![f64::NAN; n]);
          if fv_trial.iter().all(|v| v.is_finite())
            && residual_norm(&fv_trial) < base_norm
          {
            accepted = Some((x_trial, fv_trial, shrink));
            break;
          }
          shrink *= 0.5;
        }
        let Some((x_new, fv_new, shrink)) = accepted else {
          // No backtrack improved on this Jacobian estimate — it has drifted
          // too far from the true derivative. Re-seed it once and retry the
          // step from scratch before giving up on this outer iteration.
          if attempt == 0 {
            let reals: Vec<Expr> = x.iter().map(|&v| Expr::Real(v)).collect();
            let bindings: Vec<(&str, &Expr)> =
              vars.iter().map(String::as_str).zip(reals.iter()).collect();
            jm = eval_jacobian_at(&x, &bindings, &no_hint)?;
            continue;
          }
          break;
        };
        let step: Vec<f64> = delta.iter().map(|d| d * shrink).collect();
        // Rank-1 update: J += (Δf - JΔx) ΔxᵀΔx⁻¹ / (Δx · Δx).
        let step_sq: f64 = step.iter().map(|d| d * d).sum();
        if step_sq > 1e-300 {
          for i in 0..n {
            let j_dx: f64 = (0..n).map(|k| jm[i][k] * step[k]).sum();
            let coeff = (fv_new[i] - fv[i] - j_dx) / step_sq;
            for (k, &sk) in step.iter().enumerate() {
              jm[i][k] += coeff * sk;
            }
          }
        }
        x = x_new;
        fv = fv_new;
        fire_monitor(&x);
        progressed = true;
        let resid = residual_norm(&fv);
        if resid < best_resid {
          best_resid = resid;
          best_x.clone_from(&x);
        }
        if resid < tol {
          converged = true;
        }
        break;
      }
      if !progressed {
        break;
      }
      if converged {
        break;
      }
    }
    // Only report exhausting the iteration budget when that is actually
    // what happened — a stall break (`!progressed`, backtracking found no
    // improving step even after re-seeding the Jacobian) settles on the
    // best point found so far exactly like the plain-Newton branch below
    // does silently, not a failure to converge within `max_iter` steps.
    if !converged && iter_count >= max_iter {
      crate::emit_message(&format!(
        "FindRoot::cvmit: Failed to converge to the requested accuracy or \
         precision within {max_iter} iterations."
      ));
    }
  } else {
    // Fully symbolic Jacobian: recomputing it every iteration costs no
    // extra function evaluations, so plain damped-free Newton is simplest.
    // The residual is re-checked and `best_x` updated immediately after
    // every step (not just at the top of the next iteration) so a small
    // `MaxIterations` still reflects the step just taken — otherwise the
    // very last step of a run that stops because it hit `max_iter` would be
    // computed and then silently discarded, exactly like the Broyden branch
    // above already avoids by updating `best_x` right after each step.
    let eval_residual =
      |x: &[f64]| -> Result<(Vec<f64>, f64), InterpreterError> {
        if let Some(nodes) = &compiled {
          let fv: Vec<f64> = nodes.iter().map(|node| node.eval(x)).collect();
          if fv.iter().all(|v| v.is_finite()) {
            let resid = fv.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            return Ok((fv, resid));
          }
        }
        let reals: Vec<Expr> = x.iter().map(|&v| Expr::Real(v)).collect();
        let bindings: Vec<(&str, &Expr)> =
          vars.iter().map(String::as_str).zip(reals.iter()).collect();
        let mut fv = vec![0.0; n];
        for (i, f) in eqns.iter().enumerate() {
          fv[i] = find_root_eval_multivar_at(f, &bindings)?;
        }
        let resid = fv.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        Ok((fv, resid))
      };
    let (mut fv, mut resid) = eval_residual(&x)?;
    best_resid = resid;
    best_x.clone_from(&x);
    // Newton steps are damped by backtracking on the Euclidean norm of the
    // residual vector (the merit function). Demanding the max-norm drop on
    // every full step wrongly stalls systems where one component's residual
    // is traded for another's (e.g. `{x*z == 2, z == 3}` from `{1, 1}` lands
    // on `{0, 3}` first), so the iteration never left its starting point.
    let merit = |fv: &[f64]| fv.iter().map(|v| v * v).sum::<f64>();
    for _ in 0..max_iter {
      if resid < tol {
        break;
      }
      let reals: Vec<Expr> = x.iter().map(|&v| Expr::Real(v)).collect();
      let bindings: Vec<(&str, &Expr)> =
        vars.iter().map(String::as_str).zip(reals.iter()).collect();
      let jm = eval_jacobian_at(&x, &bindings, &no_hint)?;
      let neg_f: Vec<f64> = fv.iter().map(|v| -v).collect();
      let Some(delta) = find_root_solve_linear(jm, neg_f) else {
        break;
      };
      let base_merit = merit(&fv);
      let mut shrink = 1.0;
      let mut accepted = None;
      for _ in 0..FIND_ROOT_MAX_BACKTRACKS {
        let x_trial: Vec<f64> = x
          .iter()
          .zip(&delta)
          .map(|(&xj, &dj)| xj + dj * shrink)
          .collect();
        let x_trial = clamp_to_bounds(x_trial);
        if let Ok((fv_trial, resid_trial)) = eval_residual(&x_trial)
          && fv_trial.iter().all(|v| v.is_finite())
          && merit(&fv_trial) < base_merit
        {
          accepted = Some((x_trial, fv_trial, resid_trial, shrink));
          break;
        }
        shrink *= 0.5;
      }
      let Some((x_new, fv_new, resid_new, shrink)) = accepted else {
        break;
      };
      let max_d = delta.iter().fold(0.0f64, |a, &d| a.max((d * shrink).abs()));
      x = x_new;
      fv = fv_new;
      resid = resid_new;
      fire_monitor(&x);
      if resid < best_resid {
        best_resid = resid;
        best_x.clone_from(&x);
      }
      if max_d < tol {
        break;
      }
    }
  }
  x = best_x;
  let rules: Vec<Expr> = raw_vars
    .iter()
    .zip(&x)
    .map(|(v, &xv)| rule_expr(v.clone(), Expr::Real(xv)))
    .collect();
  Ok(Expr::List(rules.into()))
}

/// Check if an expression contains an unevaluated D[...] or Dt[...] call.
fn contains_unevaluated_d(expr: &Expr) -> bool {
  match expr {
    // The derivative of an interpolating function evaluates at any point.
    Expr::CurriedCall { func, args }
      if matches!(func.as_ref(), Expr::CurriedCall { func: d, args: f }
        if matches!(d.as_ref(), Expr::FunctionCall { name, .. } if name == "Derivative")
          && matches!(f.as_slice(), [Expr::FunctionCall { name, .. }] if name == "InterpolatingFunction")) =>
    {
      args.iter().any(contains_unevaluated_d)
    }
    // "D"/"Dt" catches an unevaluated D[...] left as-is; "Derivative" catches
    // the `Derivative[n1, n2, ...][f][args]` form a partial derivative of a
    // function with no symbolic derivative rule (e.g. CDF of a distribution
    // Woxi only evaluates numerically) is left in.
    Expr::FunctionCall { name, .. }
      if name == "D" || name == "Dt" || name == "Derivative" =>
    {
      true
    }
    Expr::FunctionCall { args, .. } => args.iter().any(contains_unevaluated_d),
    Expr::CurriedCall { func, args } => {
      contains_unevaluated_d(func) || args.iter().any(contains_unevaluated_d)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_unevaluated_d(left) || contains_unevaluated_d(right)
    }
    Expr::UnaryOp { operand, .. } => contains_unevaluated_d(operand),
    // A malformed derivative can surface wrapped in a List (e.g. the
    // mis-threaded per-component result of differentiating a function whose
    // argument is itself a list), so recurse into list elements too.
    Expr::List(items) => items.iter().any(contains_unevaluated_d),
    _ => false,
  }
}

// ─── Minimize / Maximize ─────────────────────────────────────────────

/// Minimize[f, x] or Minimize[f, {x, y, ...}] — find the global minimum.
/// Minimize[{f, cons1, cons2, ...}, vars] — constrained minimization.
/// Returns {min_val, {x -> x_min, ...}} with exact results when possible.
///
/// Maximize[f, vars] is the dual (negates objective and result).
/// True when the optimizer of a `{value, {var -> a, …}}` result satisfies every
/// constraint. The real optimum of a strictly bounded problem sits *on* the
/// excluded boundary — `Maximize[{x, x < 3}, x]` reports `{3, {x -> 3}}`, which
/// wolframscript warns about — and such a point is not a solution over the
/// integers, so it must not be handed on as one.
fn optimizer_is_feasible(result: &Expr, constraints: &[Expr]) -> bool {
  let Expr::List(pair) = result else {
    return false;
  };
  let Some(Expr::List(rules)) = pair.get(1) else {
    return false;
  };
  let point: Vec<(String, Expr)> = rules
    .iter()
    .filter_map(|rule| match rule {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => match pattern.as_ref() {
        Expr::Identifier(var) => {
          Some((var.clone(), replacement.as_ref().clone()))
        }
        _ => None,
      },
      _ => None,
    })
    .collect();
  if point.len() != rules.len() {
    return false;
  }
  point_satisfies(&point, constraints)
}

/// True when every constraint evaluates to `True` at the `var -> value` point.
fn point_satisfies(point: &[(String, Expr)], constraints: &[Expr]) -> bool {
  constraints.iter().all(|constraint| {
    let mut substituted = constraint.clone();
    for (var, value) in point {
      substituted =
        crate::syntax::substitute_variable(&substituted, var, value);
    }
    matches!(
      crate::evaluator::evaluate_expr_to_expr(&substituted),
      Ok(Expr::Identifier(ref s)) if s == "True"
    )
  })
}

/// The real relaxation of an `Integers`-domain problem often lands just outside
/// the feasible integer set: `Maximize[{x, x < 3}, x, Integers]` relaxes to
/// `x -> 3`, which the strict `<` excludes, while wolframscript answers
/// `{2, {x -> 2}}`. Search the integer points within one step of the relaxation
/// optimum and keep the best feasible one.
///
/// The window is deliberately tiny (one integer either side of the relaxation
/// value, at most four variables), so this can only ever refine an answer the
/// relaxation already located — it never goes hunting for an optimum somewhere
/// else, and returns `None` when nothing nearby is feasible.
fn snap_relaxation_to_integers(
  relaxation: &Expr,
  objective: &Expr,
  constraints: &[Expr],
  maximize: bool,
) -> Option<Expr> {
  let Expr::List(pair) = relaxation else {
    return None;
  };
  if pair.len() != 2 {
    return None;
  }
  let Expr::List(rules) = &pair[1] else {
    return None;
  };
  if rules.is_empty() || rules.len() > 4 {
    return None;
  }

  // Each optimizer must be a finite real number to bracket it with integers.
  let mut ranges: Vec<(String, Vec<i64>)> = Vec::with_capacity(rules.len());
  for rule in rules {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = rule
    else {
      return None;
    };
    let Expr::Identifier(var) = pattern.as_ref() else {
      return None;
    };
    let value = crate::functions::math_ast::try_eval_to_f64(replacement)?;
    if !value.is_finite() || value.abs() > 1e12 {
      return None;
    }
    let lo = value.floor() as i64 - 1;
    let hi = value.ceil() as i64 + 1;
    ranges.push((var.clone(), (lo..=hi).collect()));
  }

  // Cartesian product of the per-variable windows (at most 4^4 points).
  let mut points: Vec<Vec<(String, Expr)>> = vec![Vec::new()];
  for (var, candidates) in &ranges {
    points = points
      .iter()
      .flat_map(|prefix| {
        candidates.iter().map(move |c| {
          let mut next = prefix.clone();
          next.push((var.clone(), Expr::Integer(*c as i128)));
          next
        })
      })
      .collect();
  }

  let mut best: Option<(f64, Expr, Vec<(String, Expr)>)> = None;
  for point in points {
    if !point_satisfies(&point, constraints) {
      continue;
    }
    let mut value = objective.clone();
    for (var, v) in &point {
      value = crate::syntax::substitute_variable(&value, var, v);
    }
    let Ok(value) = crate::evaluator::evaluate_expr_to_expr(&value) else {
      continue;
    };
    let Some(numeric) = crate::functions::math_ast::try_eval_to_f64(&value)
    else {
      continue;
    };
    let better = match &best {
      None => true,
      Some((current, _, _)) => {
        if maximize {
          numeric > *current
        } else {
          numeric < *current
        }
      }
    };
    if better {
      best = Some((numeric, value, point));
    }
  }

  let (_, value, point) = best?;
  Some(Expr::List(
    vec![
      value,
      Expr::List(
        point
          .into_iter()
          .map(|(var, v)| rule_expr(Expr::Identifier(var), v))
          .collect(),
      ),
    ]
    .into(),
  ))
}

/// True when every optimizer in a `{value, {var -> a, …}}` result is an integer,
/// or the extremum is infinite. Together with feasibility this makes the result
/// optimal over the integers as well: a *feasible* real optimum that happens to
/// be integral is feasible there and nothing integral can beat it, and an
/// unbounded real problem is unbounded over the integers too.
fn optimizer_is_integral(result: &Expr) -> bool {
  let Expr::List(pair) = result else {
    return false;
  };
  if pair.len() != 2 {
    return false;
  }
  let infinite = |e: &Expr| {
    matches!(e, Expr::Identifier(s) if s == "Infinity")
      || crate::functions::math_ast::is_neg_infinity(e)
  };
  if infinite(&pair[0]) {
    return true;
  }
  let Expr::List(rules) = &pair[1] else {
    return false;
  };
  rules.iter().all(|rule| {
    let value = match rule {
      Expr::Rule { replacement, .. }
      | Expr::RuleDelayed { replacement, .. } => replacement.as_ref(),
      _ => return false,
    };
    matches!(value, Expr::Integer(_) | Expr::BigInteger(_)) || infinite(value)
  })
}

pub fn minimize_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize { "Maximize" } else { "Minimize" };
  // The critical points come from Solve, whose inverted functions are its
  // own business: wolframscript's Maximize never reports Solve::ifun.
  let _no_ifun = SuppressIfun::new(true);
  // An `Integers` domain becomes `Element[var, Integers]` constraints below,
  // which the solver only handles for a bounded problem. When that comes back
  // with nothing, solve over the reals instead and keep the answer only if every
  // optimizer is already an integer — a fractional one says nothing about the
  // integer optimum, so the call is left alone rather than rounded and guessed.
  if args.len() == 3
    && matches!(&args[2], Expr::Identifier(d) if d == "Integers")
  {
    // Both attempts below are probes, and a probe's complaint only belongs in
    // the output when its answer is the one handed back — "the maximum is not
    // attained" describes a relaxation Woxi may well discard. So each runs
    // quietly and its messages are replayed only on acceptance.
    let probe =
      |probe_args: &[Expr]| -> Result<(Expr, Vec<String>), InterpreterError> {
        let before = crate::snapshot_warnings();
        let seen = before.messages().len();
        crate::push_quiet();
        let result = minimize_ast_inner(probe_args, maximize);
        // Read before leaving the quiet block, which drops them.
        let raised = crate::snapshot_warnings().messages()[seen..].to_vec();
        crate::pop_quiet();
        crate::restore_warnings(before);
        result.map(|value| (value, raised))
      };
    let replay = |messages: Vec<String>| {
      for message in messages {
        crate::emit_message(&message);
      }
    };

    let (constrained, messages) = probe(args)?;
    if matches!(&constrained, Expr::List(pair) if pair.len() == 2) {
      replay(messages);
      return Ok(constrained);
    }
    let (over_reals, messages) = probe(&args[..2])?;
    let (objective, constraints) = minimize_parse_objective(&args[0]);
    if optimizer_is_integral(&over_reals)
      && optimizer_is_feasible(&over_reals, &constraints)
    {
      replay(messages);
      return Ok(over_reals);
    }
    if let Some(snapped) = snap_relaxation_to_integers(
      &over_reals,
      &objective,
      &constraints,
      maximize,
    ) {
      return Ok(snapped);
    }
    replay(messages);
    return Ok(unevaluated(func_name, args));
  }
  minimize_ast_inner(args, maximize)
}

fn minimize_ast_inner(
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
    return Ok(unevaluated(func_name, args));
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
  // The variables are real, so an even power of the absolute value of a
  // real expression is that power of the expression itself
  // (EuclideanDistance gives Sqrt[Abs[x - 1]^2 + …]).
  let raw_objective = crate::evaluator::evaluate_function_call_ast(
    "ReplaceAll",
    &[
      raw_objective.clone(),
      crate::syntax::string_to_expr(
        "Abs[u$_]^(k$_?EvenQ) /; FreeQ[u$, Complex] :> u$^k$",
      )?,
    ],
  )
  .unwrap_or(raw_objective);
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
        constraints.push(call(
          "Element",
          vec![Expr::Identifier(var.clone()), id_expr("Integers")],
        ));
      }
    }
  }

  if vars.len() == 1
    && let Some(result) = optimize_parametric_quadratic(
      &objective,
      &constraints,
      &vars[0],
      maximize,
    )?
  {
    return Ok(result);
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

/// The optimum of a quadratic in `var` whose coefficients hold parameters:
/// `C + B var + A var^2` has its extremum `Together[C - B^2/(4 A)]` at
/// `var -> Together[-B/(2 A)]` when `A` has the bounding sign (negative for
/// a maximum). Without constraints that sign must be numeric; parameter
/// constraints (free of `var`) must imply it, and the answer then holds
/// under them: `{Piecewise[{{max, cond}}, -Infinity], {var ->
/// Piecewise[{{x*, cond}}, Indeterminate]}}`. `None` for anything else.
fn optimize_parametric_quadratic(
  objective: &Expr,
  constraints: &[Expr],
  var: &str,
  maximize: bool,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::polynomial_ast::eliminate::contains_var;
  if !contains_var(objective, var)
    || constraints.iter().any(|c| contains_var(c, var))
  {
    return Ok(None);
  }
  let x = Expr::Identifier(var.to_string());
  let eval = crate::evaluator::evaluate_expr_to_expr;
  // Only an objective with parameters: a purely numeric one is the
  // ordinary solver's.
  let others = eval(&call(
    "DeleteCases",
    vec![call("Variables", vec![objective.clone()]), x.clone()],
  ))?;
  if matches!(&others, Expr::List(v) if v.is_empty()) {
    return Ok(None);
  }
  if !matches!(
    eval(&call("PolynomialQ", vec![objective.clone(), x.clone()]))?,
    Expr::Identifier(ref t) if t == "True"
  ) {
    return Ok(None);
  }
  let coefficient_list =
    eval(&call("CoefficientList", vec![objective.clone(), x.clone()]))?;
  let Expr::List(coeffs) = &coefficient_list else {
    return Ok(None);
  };
  if coeffs.len() != 3 {
    return Ok(None);
  }
  let (c, b, a) = (&coeffs[0], &coeffs[1], &coeffs[2]);
  let bounding_sign = call(
    if maximize { "Less" } else { "Greater" },
    vec![a.clone(), Expr::Integer(0)],
  );
  let condition = match constraints {
    [] => None,
    [one] => Some(one.clone()),
    _ => Some(call("And", constraints.to_vec())),
  };
  let sign_holds = match &condition {
    None => eval(&bounding_sign)?,
    Some(cond) => eval(&call("Simplify", vec![bounding_sign, cond.clone()]))?,
  };
  if !matches!(sign_holds, Expr::Identifier(ref t) if t == "True") {
    return Ok(None);
  }
  let int = Expr::Integer;
  let pow = |e: &Expr, k: i128| call("Power", vec![e.clone(), int(k)]);
  // C - B^2/(4 A) and -B/(2 A).
  let value = eval(&call(
    "Together",
    vec![call(
      "Plus",
      vec![
        c.clone(),
        call(
          "Times",
          vec![
            int(-1),
            pow(b, 2),
            pow(&call("Times", vec![int(4), a.clone()]), -1),
          ],
        ),
      ],
    )],
  ))?;
  let optimizer = eval(&call(
    "Together",
    vec![call(
      "Times",
      vec![
        int(-1),
        b.clone(),
        pow(&call("Times", vec![int(2), a.clone()]), -1),
      ],
    )],
  ))?;
  let (value, optimizer) = match condition {
    None => (value, optimizer),
    Some(cond) => {
      let unbounded = if maximize {
        eval(&crate::syntax::string_to_expr("-Infinity")?)?
      } else {
        id_expr("Infinity")
      };
      let piecewise = |e: Expr, otherwise: Expr| {
        call(
          "Piecewise",
          vec![
            Expr::List(vec![Expr::List(vec![e, cond.clone()].into())].into()),
            otherwise,
          ],
        )
      };
      (
        piecewise(value, unbounded),
        piecewise(optimizer, id_expr("Indeterminate")),
      )
    }
  };
  Ok(Some(Expr::List(
    vec![
      value,
      Expr::List(
        vec![Expr::Rule {
          pattern: Box::new(x),
          replacement: Box::new(optimizer),
        }]
        .into(),
      ),
    ]
    .into(),
  )))
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
  if let Expr::List(items) = expr {
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
  } else {
    let (n, e) = parse_one(expr, func_name)?;
    Ok((vec![n], vec![e]))
  }
}

/// Recursively replace every occurrence of `from` with `to` in `expr`.
pub(crate) fn substitute_expr(expr: &Expr, from: &Expr, to: &Expr) -> Expr {
  // Only a node shaped like `from` can equal it; spelling every node out as
  // a string made the walk quadratic for large systems.
  fn name_of(e: &Expr) -> Option<&str> {
    match e {
      Expr::Identifier(n) | Expr::Constant(n) => Some(n.as_str()),
      _ => None,
    }
  }
  let may_match = match (expr, from) {
    (
      Expr::FunctionCall { name: a, args: x },
      Expr::FunctionCall { name: b, args: y },
    ) => a == b && x.len() == y.len(),
    // A symbol only equals a symbol of that name.
    (_, Expr::Identifier(_) | Expr::Constant(_)) => {
      if name_of(expr) == name_of(from) {
        return to.clone();
      }
      false
    }
    (Expr::List(_), _) | (_, Expr::List(_)) => {
      matches!((expr, from), (Expr::List(_), Expr::List(_)))
    }
    _ => true,
  };
  if may_match && expr_to_string(expr) == expr_to_string(from) {
    return to.clone();
  }
  match expr {
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| substitute_expr(e, from, to)).collect())
    }
    Expr::FunctionCall { name, args } => call(
      name.as_str(),
      args.iter().map(|e| substitute_expr(e, from, to)).collect(),
    ),
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      substitute_expr(left, from, to),
      substitute_expr(right, from, to),
    ),
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
    } => rule_expr(
      substitute_expr(pattern, from, to),
      substitute_expr(replacement, from, to),
    ),
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(substitute_expr(func, from, to)),
      args: args.iter().map(|e| substitute_expr(e, from, to)).collect(),
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
        call("Rational", vec![Expr::Integer(rn), Expr::Integer(rd)])
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
      if let Ok(n_result) = n_ast(std::slice::from_ref(expr)) {
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
          call("Rational", vec![Expr::Integer(num), Expr::Integer(den)])
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
              call("Rational", vec![Expr::Integer(n1), Expr::Integer(d1)])
            });
            if n1 != n2 || d1 != d2 {
              roots.push(if d2 == 1 {
                Expr::Integer(n2)
              } else {
                call("Rational", vec![Expr::Integer(n2), Expr::Integer(d2)])
              });
            }
          } else {
            // Irrational roots: (-b ± sqrt_out * √sqrt_in) / (2a)
            let g = gcd_i128(gcd_i128(-b, sqrt_out), 2 * a);
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
                let rational_arg = make_rational(sqrt_in, den * den);
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
                let join_op = if sign_minus {
                  BinaryOperator::Minus
                } else {
                  BinaryOperator::Plus
                };
                binop(join_op, nb_expr, sqrt_part.clone())
              };
              let root = if den == 1 {
                num
              } else {
                div2(num, Expr::Integer(den))
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
                call("Rational", vec![Expr::Integer(rn), Expr::Integer(rd)])
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
  rat_reduce(n, d)
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
      if let Some(val) = crate::evaluator::evaluate_expr_to_expr(&substituted)
        .ok()
        .and_then(|e| minimize_try_f64(&e))
      {
        vals.push(val);
      } else {
        vals.clear();
        break;
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

/// Build the unbounded result for minimize/maximize (no optimum exists):
/// `{-Infinity, {x -> ±Infinity}}`, or `{Infinity, …}` when maximizing.
/// `var_toward_positive` says which end of the axis the objective runs off
/// at, which is independent of the direction of the optimization:
/// `Maximize[-x, x]` is unbounded as `x -> -Infinity`.
fn minimize_neg_infinity_result(
  vars: &[String],
  maximize: bool,
  var_toward_positive: bool,
) -> Expr {
  let neg_infinity = || neg1(id_expr("Infinity"));
  let inf_val = if maximize {
    id_expr("Infinity")
  } else {
    neg_infinity()
  };
  let x_val = if var_toward_positive {
    id_expr("Infinity")
  } else {
    neg_infinity()
  };
  let rules: Vec<Expr> = vars
    .iter()
    .map(|v| rule_expr(Expr::Identifier(v.clone()), x_val.clone()))
    .collect();
  Expr::List(vec![inf_val, Expr::List(rules.into())].into())
}

/// The end of the axis at which the (already sign-normalized, i.e. always
/// minimized) objective `f` runs off to -Infinity: `true` for
/// `var -> +Infinity`. Probed numerically at large magnitudes; an objective
/// that cannot be evaluated there falls back to `var -> -Infinity`.
fn minimize_unbounded_direction(f: &Expr, var: &str) -> bool {
  let probe = |x: f64| -> Option<f64> {
    let substituted =
      crate::syntax::substitute_variable(f, var, &Expr::Real(x));
    crate::evaluator::evaluate_expr_to_expr(&substituted)
      .ok()
      .and_then(|e| minimize_try_f64(&e))
  };
  match (probe(1e6), probe(-1e6)) {
    (Some(at_pos), Some(at_neg)) => at_pos < at_neg,
    _ => false,
  }
}

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
      for a in args {
        collect_abs_args(a, var, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_abs_args(left, var, out);
      collect_abs_args(right, var, out);
    }
    Expr::UnaryOp { operand, .. } => collect_abs_args(operand, var, out),
    Expr::List(items) => {
      for it in items {
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
    let solved = solve_ast(&[eq, id_expr(var)]);
    if let Ok(Expr::List(sol_sets)) = &solved {
      for sol_set in sol_sets {
        if let Expr::List(rules) = sol_set {
          for rule in rules {
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
      "{head}::natt: The {kind} is not attained at any point satisfying the given constraints."
    ));
    return Ok(minimize_neg_infinity_result(
      &[var.to_string()],
      maximize,
      minimize_unbounded_direction(&f_inner, var),
    ));
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
    return Ok(call(func_name, vec![f.clone(), id_expr(var)]));
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

  let (Some(min_val), Some(min_x)) = (best_exact, best_x) else {
    return Ok(call(func_name, vec![f.clone(), id_expr(var)]));
  };

  // For maximize, negate the value back
  let result_val = if maximize {
    simplify(negate_expr(&min_val))
  } else {
    min_val
  };

  let rule = rule_expr(id_expr(var), min_x);
  Ok(Expr::List(
    vec![result_val, Expr::List(vec![rule].into())].into(),
  ))
}

/// Find critical points of f' = 0 in one variable.
/// Periodic equations come back from `Solve` as a family
/// `ConditionalExpression[Pi/2 + 2*Pi*C[1], Element[C[1], Integers]]`, which
/// carries no usable numeric value. Instantiate the integer parameter over a
/// small window around zero so the concrete critical points (`Pi/2`,
/// `Pi/2 + 2*Pi`, …) become candidates; the caller's feasibility filter drops
/// the ones outside the constraint region.
fn minimize_instantiate_periodic_roots(roots: &[Expr]) -> Vec<Expr> {
  const WINDOW: i128 = 3;
  let mut out = Vec::new();
  for root in roots {
    let Expr::FunctionCall { name, args } = root else {
      out.push(root.clone());
      continue;
    };
    if name != "ConditionalExpression" || args.len() != 2 {
      out.push(root.clone());
      continue;
    }
    // The condition names the parameter: Element[C[k], Integers].
    let Expr::FunctionCall {
      name: cond,
      args: cond_args,
    } = &args[1]
    else {
      out.push(root.clone());
      continue;
    };
    if cond != "Element"
      || cond_args.len() != 2
      || !matches!(&cond_args[1], Expr::Identifier(d) if d == "Integers")
    {
      out.push(root.clone());
      continue;
    }
    // Walk outwards from k = 0 so the member nearest the origin is tried
    // first: on a periodic objective every member is equally optimal, and
    // wolframscript reports the one near the origin.
    let order = std::iter::once(0).chain((1..=WINDOW).flat_map(|k| [-k, k]));
    for k in order {
      let instantiated =
        substitute_expr(&args[0], &cond_args[0], &Expr::Integer(k));
      // `simplify` alone only folds a zero factor out of a two-argument
      // `BinaryOp` product, not an n-ary `Times[...]` call — so a
      // once-imaginary member like `2*I*Pi*C[1]` at `C[1] -> 0` stayed
      // `Times[2, I, Pi, 0]` instead of collapsing to the real `0`, and the
      // literal `I` still in the tree made `minimize_cp_is_complex` reject
      // it as complex even though the whole product is zero. Run it through
      // the real evaluator first so a genuinely-zero coefficient collapses
      // the product before that complexity check sees it.
      let evaluated = crate::evaluator::evaluate_expr_to_expr(&instantiated)
        .unwrap_or(instantiated);
      out.push(simplify(evaluated));
    }
  }
  out
}

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
  match solve_ast(&[df_eq, id_expr(var)]) {
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
        let roots = minimize_instantiate_periodic_roots(&roots);
        return Ok(roots);
      }
      // Unevaluated Solve result - try numerical
      minimize_find_critical_points_numerical(f, var)
    }
    Err(_) => minimize_find_critical_points_numerical(f, var),
  }
}

/// Numerically find critical points of f using Newton's method with multiple starts.
/// Newton's method from a few starts on `f'`. Its probes may land where the
/// objective is singular; that is no message of the optimization's.
fn minimize_find_critical_points_numerical(
  f: &Expr,
  var: &str,
) -> Result<Vec<Expr>, InterpreterError> {
  quietly(|| minimize_find_critical_points_newton(f, var))
}

fn minimize_find_critical_points_newton(
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
/// The minimum of a quadratic polynomial `f` in `vars` whose Hessian is
/// positive definite: its single critical point, from the (linear) gradient
/// system. `{value, {var -> point, …}}`, or None for anything else.
fn convex_quadratic_minimum(
  f: &Expr,
  vars: &[String],
) -> Result<Option<Expr>, InterpreterError> {
  let var_list =
    Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect());
  if !matches!(
    crate::evaluator::evaluate_function_call_ast(
      "PolynomialQ",
      &[f.clone(), var_list.clone()]
    )?,
    Expr::Identifier(ref t) if t == "True"
  ) {
    return Ok(None);
  }
  let mut gradient = Vec::with_capacity(vars.len());
  let mut hessian = Vec::with_capacity(vars.len());
  for v in vars {
    let d = crate::functions::calculus_ast::differentiate_expr(f, v)?;
    let mut row = Vec::with_capacity(vars.len());
    for w in vars {
      let dd = crate::evaluator::evaluate_expr_to_expr(
        &crate::functions::calculus_ast::differentiate_expr(&d, w)?,
      )?;
      // Second derivatives must be numbers: degree two at most.
      if crate::functions::math_ast::try_eval_to_f64(&dd).is_none() {
        return Ok(None);
      }
      row.push(dd);
    }
    hessian.push(Expr::List(row.into()));
    gradient.push(Expr::Comparison {
      operands: vec![d, Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    });
  }
  let hessian = Expr::List(hessian.into());
  if !matches!(
    crate::evaluator::evaluate_function_call_ast(
      "PositiveDefiniteMatrixQ",
      &[hessian]
    )?,
    Expr::Identifier(ref t) if t == "True"
  ) {
    return Ok(None);
  }
  let solved = solve_ast(&[Expr::List(gradient.into()), var_list])?;
  let Expr::List(sets) = &solved else {
    return Ok(None);
  };
  let [Expr::List(rules)] = sets.as_slice() else {
    return Ok(None);
  };
  // The rules in the variables' order.
  let mut ordered = Vec::with_capacity(vars.len());
  for v in vars {
    let Some(rule) = rules.iter().find(|r| {
      matches!(r, Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == v))
    }) else {
      return Ok(None);
    };
    ordered.push(rule.clone());
  }
  if ordered.iter().any(|r| {
    matches!(r, Expr::Rule { replacement, .. }
      if vars.iter().any(|v| contains_var(replacement, v)))
  }) {
    return Ok(None);
  }
  let value = crate::evaluator::evaluate_function_call_ast(
    "Together",
    &[substitute_rules(f, &ordered)?],
  )?;
  Ok(Some(Expr::List(
    vec![value, Expr::List(ordered.into())].into(),
  )))
}

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

  // Sqrt[q] is smallest where q is: minimize the radicand, then take the
  // root of its minimum.
  let radicand = match f {
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      Some(&args[0])
    }
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && expr_to_string(&args[1]) == "1/2" =>
    {
      Some(&args[0])
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } if expr_to_string(right) == "1/2" => Some(left.as_ref()),
    _ => None,
  };
  if !maximize
    && let Some(radicand) = radicand
    && let Some(inner) = convex_quadratic_minimum(radicand, vars)?
    && let Expr::List(pair) = &inner
    && let [value, rules] = pair.as_slice()
  {
    let root = crate::evaluator::evaluate_function_call_ast(
      "Sqrt",
      std::slice::from_ref(value),
    )?;
    return Ok(Expr::List(vec![root, rules.clone()].into()));
  }
  if let Some(result) = convex_quadratic_minimum(&f_inner, vars)? {
    let Expr::List(pair) = &result else {
      unreachable!()
    };
    let value = if maximize {
      simplify(negate_expr(&pair[0]))
    } else {
      pair[0].clone()
    };
    return Ok(Expr::List(vec![value, pair[1].clone()].into()));
  }

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
    if let Ok(sol) = solve_ast(&[grad_eq, Expr::Identifier(var.clone())]) {
      if let Expr::List(sol_sets) = &sol
        && sol_sets.len() == 1
        && let Some(Expr::List(rules)) = sol_sets.first()
        && rules.len() == 1
        && let Some(Expr::Rule { replacement, .. }) = rules.first()
        // A coupled gradient (x in terms of y) is no critical point.
        && !vars.iter().any(|v| contains_var(replacement, v))
      {
        solutions[i] = Some(*replacement.clone());
        continue;
      }
      all_solved = false;
      break;
    }
    all_solved = false;
    break;
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
        .map(|(v, val)| rule_expr(Expr::Identifier(v.clone()), val.clone()))
        .collect();
      return Ok(Expr::List(
        vec![result_val, Expr::List(rules.into())].into(),
      ));
    }
  }

  // The gradient solved jointly: the least value over its exact critical
  // points. A critical set left in terms of a free variable (`y -> 3/x`
  // where f is constant along it) is represented by the free variable at 1.
  {
    let equations: Vec<Expr> = grad
      .iter()
      .map(|g| Expr::Comparison {
        operands: vec![g.clone(), Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      })
      .collect();
    let var_list =
      Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect());
    crate::push_quiet();
    let solved = solve_ast(&[Expr::List(equations.into()), var_list]);
    crate::pop_quiet();
    let mut best: Option<(f64, Expr, Vec<Expr>)> = None;
    if let Ok(Expr::List(sets)) = &solved {
      for set in sets {
        let Expr::List(rules) = set else { continue };
        let mut point = Vec::with_capacity(n);
        let mut ok = true;
        for v in vars {
          let found = rules.iter().find_map(|r| match r {
            Expr::Rule {
              pattern,
              replacement,
            } if matches!(pattern.as_ref(), Expr::Identifier(p) if p == v) => {
              Some((**replacement).clone())
            }
            _ => None,
          });
          point.push(found.unwrap_or(Expr::Integer(1)));
        }
        // Free variables at 1, then the solved ones in terms of them.
        let free_at_one: Vec<Expr> = vars
          .iter()
          .zip(point.iter())
          .filter(|(v, p)| matches!(p, Expr::Integer(1)) && !rules.iter().any(|r| matches!(r, Expr::Rule { pattern, .. } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == *v))))
          .map(|(v, _)| rule_expr(Expr::Identifier(v.clone()), Expr::Integer(1)))
          .collect();
        let mut values = Vec::with_capacity(n);
        for p in &point {
          let at = substitute_rules(p, &free_at_one)?;
          if vars.iter().any(|v| contains_var(&at, v))
            || crate::functions::math_ast::try_eval_to_f64(&at).is_none()
          {
            ok = false;
            break;
          }
          values.push(at);
        }
        if !ok {
          continue;
        }
        let Ok(value) = minimize_eval_exact_multi(&f_inner, vars, &values)
        else {
          continue;
        };
        let Some(numeric) = crate::functions::math_ast::try_eval_to_f64(&value)
        else {
          continue;
        };
        if best.as_ref().is_none_or(|(b, _, _)| numeric < *b) {
          best = Some((numeric, value, values));
        }
      }
    }
    if let Some((_, value, values)) = best {
      let value = if maximize {
        simplify(negate_expr(&value))
      } else {
        value
      };
      let rules: Vec<Expr> = vars
        .iter()
        .zip(values)
        .map(|(v, val)| rule_expr(Expr::Identifier(v.clone()), val))
        .collect();
      return Ok(Expr::List(vec![value, Expr::List(rules.into())].into()));
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
    return Ok(call(
      func_name,
      vec![
        f.clone(),
        Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
      ],
    ));
  }

  let result_val = if maximize {
    Expr::Real(-fval)
  } else {
    Expr::Real(fval)
  };
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(v, &val)| rule_expr(Expr::Identifier(v.clone()), Expr::Real(val)))
    .collect();
  Ok(Expr::List(
    vec![result_val, Expr::List(rules.into())].into(),
  ))
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

  // A linear objective under linear constraints is a linear program, solved
  // exactly when its coefficients are integers.
  if vars.len() > 1
    && let Some(result) =
      minimize_linear_program(f, &constraints, vars, maximize, false)
  {
    return Ok(result);
  }

  // Try ILP if any Element[x, Integers] constraint is present
  if constraints
    .iter()
    .any(|c| matches!(c, Expr::FunctionCall { name, .. } if name == "Element"))
    && let Some(result) =
      minimize_try_ilp(&f_inner, &constraints, vars, maximize, func_name)
  {
    return Ok(restore(result));
  }

  // Equations alone with one parametrized solution (a linear system) are
  // eliminated: the objective is optimized over the variables they leave
  // free, and the solved ones follow from those.
  if constraints.iter().all(|c| {
    matches!(c, Expr::Comparison { operators, .. }
      if operators.iter().all(|o| *o == ComparisonOp::Equal))
  }) && let Some(result) =
    minimize_eliminating_equations(f, &constraints, vars, maximize, func_name)?
  {
    return Ok(result);
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
  )))
}

/// `expr` with each `var -> value` rule substituted syntactically and the
/// result evaluated. (`ReplaceAll` would read the solver's internal
/// `__ilp_i` variable names as patterns.)
fn substitute_rules(
  expr: &Expr,
  rules: &[Expr],
) -> Result<Expr, InterpreterError> {
  let mut out = expr.clone();
  for r in rules {
    if let Expr::Rule {
      pattern,
      replacement,
    } = r
      && let Expr::Identifier(v) = pattern.as_ref()
    {
      out = crate::syntax::substitute_variable(&out, v, replacement);
    }
  }
  crate::evaluator::evaluate_expr_to_expr(&out)
}

/// `Minimize[{f, eqns}, vars]` with equations `Solve` parametrizes by a
/// single solution: optimize `f` with the solved variables substituted over
/// the remaining ones, then read the solved ones off the optimum. None when
/// the equations have no such solution or the reduced problem is not solved.
fn minimize_eliminating_equations(
  f: &Expr,
  equations: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let var_list =
    Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect());
  // Leaving variables free is the point here, not worth a Solve::svars.
  crate::push_quiet();
  let solved = solve_ast(&[Expr::List(equations.to_vec().into()), var_list]);
  crate::pop_quiet();
  let solved = solved?;
  let Expr::List(sets) = &solved else {
    return Ok(None);
  };
  let [Expr::List(rules)] = sets.as_slice() else {
    return Ok(None);
  };
  let solved_name = |r: &Expr| match r {
    Expr::Rule { pattern, .. } => match pattern.as_ref() {
      Expr::Identifier(n) => Some(n.clone()),
      _ => None,
    },
    _ => None,
  };
  if rules.is_empty() || rules.iter().any(|r| solved_name(r).is_none()) {
    return Ok(None);
  }
  let free: Vec<String> = vars
    .iter()
    .filter(|v| !rules.iter().any(|r| solved_name(r).as_ref() == Some(v)))
    .cloned()
    .collect();
  let reduced = substitute_rules(f, rules)?;
  let (value, free_rules) = if free.is_empty() {
    (reduced, Vec::new())
  } else {
    let optimum = if free.len() == 1 {
      minimize_single_var(&reduced, &free[0], maximize, func_name)?
    } else {
      minimize_multi_var(&reduced, &free, maximize, func_name)?
    };
    let Expr::List(pair) = &optimum else {
      return Ok(None);
    };
    let [value, Expr::List(free_rules)] = pair.as_slice() else {
      return Ok(None);
    };
    if free_rules.iter().any(|r| {
      matches!(r, Expr::Rule { replacement, .. }
        if free.iter().any(|v| contains_var(replacement, v)))
    }) {
      return Ok(None);
    }
    (value.clone(), free_rules.to_vec())
  };
  let mut out = Vec::with_capacity(vars.len());
  for v in vars {
    if let Some(r) = free_rules
      .iter()
      .find(|r| solved_name(r).as_ref() == Some(v))
    {
      out.push(r.clone());
      continue;
    }
    let Some(Expr::Rule { replacement, .. }) =
      rules.iter().find(|r| solved_name(r).as_ref() == Some(v))
    else {
      return Ok(None);
    };
    let at = substitute_rules(replacement, &free_rules)?;
    out.push(rule_expr(Expr::Identifier(v.clone()), at));
  }
  Ok(Some(Expr::List(vec![value, Expr::List(out.into())].into())))
}

/// Collect the symbols an `Element[…, dom]` subject names: a single
/// identifier, a list, `Alternatives[…]` or the `x | y` chain.
fn collect_element_symbols(
  e: &Expr,
  out: &mut std::collections::HashSet<String>,
) {
  match e {
    Expr::Identifier(var) => {
      out.insert(var.clone());
    }
    Expr::List(items) => {
      for a in items {
        collect_element_symbols(a, out);
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Alternatives" || name == "List" =>
    {
      for a in args {
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

/// `expr` as `Σ cᵢ varᵢ + k` in one pass over its expanded terms, sparse in
/// the variables; `None` when some term is not a numeric multiple of a single
/// variable (or a number).
fn sparse_linear_form(
  expr: &Expr,
  index: &std::collections::HashMap<&str, usize>,
) -> Option<(Vec<(usize, f64)>, f64)> {
  // Evaluated so every term is in canonical `Times[c, v]` form.
  let expanded =
    crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(expr)).ok()?;
  let mut coeffs: Vec<(usize, f64)> = Vec::new();
  let mut constant = 0.0;
  for term in collect_additive_terms(&expanded) {
    if let Some(v) = try_eval_to_f64(&term) {
      constant += v;
      continue;
    }
    let factors: Vec<Expr> = match &term {
      Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => vec![(**left).clone(), (**right).clone()],
      other => vec![other.clone()],
    };
    let mut var = None;
    let mut c = 1.0;
    for f in &factors {
      if let Expr::Identifier(name) = f
        && let Some(&i) = index.get(name.as_str())
      {
        if var.is_some() {
          return None;
        }
        var = Some(i);
      } else {
        c *= try_eval_to_f64(f)?;
      }
    }
    coeffs.push((var?, c));
  }
  Some((coeffs, constant))
}

/// Minimize/NMinimize of a linear objective under linear constraints (some
/// variables possibly restricted by `Element[…, Integers]`), solved exactly by
/// `LinearProgramming`'s simplex / branch and bound. Returns
/// `{value, {var -> val, …}}`, or `None` when the problem is not linear, is
/// infeasible or unbounded, or (with `numeric` false) has a non-integral
/// coefficient whose machine value would not be exact.
fn minimize_linear_program(
  objective: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  numeric: bool,
) -> Option<Expr> {
  use std::collections::{HashMap, HashSet};
  let index: HashMap<&str, usize> = vars
    .iter()
    .enumerate()
    .map(|(i, v)| (v.as_str(), i))
    .collect();
  let n = vars.len();
  let mut flat = Vec::new();
  for c in constraints {
    flatten_and_expr(c, &mut flat);
  }
  let mut integer: HashSet<String> = HashSet::new();
  let mut lo = vec![f64::NEG_INFINITY; n];
  let mut hi = vec![f64::INFINITY; n];
  let mut rows: Vec<(Vec<(usize, f64)>, f64, i32)> = Vec::new();
  let mut linear = Vec::new();
  for c in &flat {
    match c {
      Expr::FunctionCall { name, args }
        if name == "Element" && args.len() == 2 =>
      {
        match &args[1] {
          Expr::Identifier(d) if d == "Integers" => {
            collect_element_symbols(&args[0], &mut integer);
          }
          Expr::Identifier(d) if d == "Reals" => {}
          _ => return None,
        }
      }
      Expr::Identifier(t) if t == "True" => {}
      other => linear.push(other),
    }
  }
  let all_integer = |terms: &[(usize, f64)]| {
    terms
      .iter()
      .all(|&(i, c)| integer.contains(&vars[i]) && c.fract() == 0.0)
  };
  for c in linear {
    let Expr::Comparison {
      operands,
      operators,
    } = c
    else {
      return None;
    };
    let ([l, r], [op]) = (operands.as_slice(), operators.as_slice()) else {
      return None;
    };
    let (terms, k) = sparse_linear_form(&minus2(l.clone(), r.clone()), &index)?;
    // terms + k op 0  ⇒  terms op -k
    let mut rhs = -k;
    let sense = match op {
      ComparisonOp::Equal => 0,
      ComparisonOp::GreaterEqual => 1,
      ComparisonOp::LessEqual => -1,
      // A strict inequality over integer terms is the next integer bound.
      ComparisonOp::Greater if all_integer(&terms) => {
        rhs = rhs.floor() + 1.0;
        1
      }
      ComparisonOp::Less if all_integer(&terms) => {
        rhs = rhs.ceil() - 1.0;
        -1
      }
      _ => return None,
    };
    let nonzero: Vec<(usize, f64)> =
      terms.iter().copied().filter(|(_, c)| *c != 0.0).collect();
    match (nonzero.as_slice(), sense) {
      ([], _) => {
        let ok = match sense {
          0 => rhs == 0.0,
          1 => 0.0 >= rhs,
          _ => 0.0 <= rhs,
        };
        if !ok {
          return None;
        }
      }
      (&[(i, a)], s) if s != 0 => {
        let bound = rhs / a;
        if (s == 1) == (a > 0.0) {
          lo[i] = lo[i].max(bound);
        } else {
          hi[i] = hi[i].min(bound);
        }
      }
      _ => rows.push((terms, rhs, sense)),
    }
  }
  let (obj_terms, obj_const) = sparse_linear_form(objective, &index)?;
  if !numeric {
    let exact = obj_terms.iter().all(|(_, c)| c.fract() == 0.0)
      && obj_const.fract() == 0.0
      && rows.iter().all(|(t, r, _)| {
        r.fract() == 0.0 && t.iter().all(|(_, c)| c.fract() == 0.0)
      })
      && lo
        .iter()
        .chain(&hi)
        .all(|b| !b.is_finite() || b.fract() == 0.0);
    if !exact {
      return None;
    }
  }
  let num = |x: f64| {
    if x.fract() == 0.0 && x.abs() < 1e15 {
      Expr::Integer(x as i128)
    } else {
      Expr::Real(x)
    }
  };
  let bound = |x: f64| {
    if x == f64::INFINITY {
      id_expr("Infinity")
    } else if x == f64::NEG_INFINITY {
      negate_expr(&id_expr("Infinity"))
    } else {
      num(x)
    }
  };
  let sign = if maximize { -1.0 } else { 1.0 };
  let mut cvec = vec![0.0; n];
  for &(i, c) in &obj_terms {
    cvec[i] += sign * c;
  }
  if rows.is_empty() {
    rows.push((Vec::new(), 0.0, 1));
  }
  let m: Vec<Expr> = rows
    .iter()
    .map(|(t, _, _)| {
      let mut row = vec![0.0; n];
      for &(i, c) in t {
        row[i] += c;
      }
      Expr::List(row.into_iter().map(num).collect())
    })
    .collect();
  let b: Vec<Expr> = rows
    .iter()
    .map(|(_, r, s)| {
      Expr::List(vec![num(*r), Expr::Integer(*s as i128)].into())
    })
    .collect();
  let lu: Vec<Expr> = (0..n)
    .map(|i| Expr::List(vec![bound(lo[i]), bound(hi[i])].into()))
    .collect();
  let dom: Vec<Expr> = vars
    .iter()
    .map(|v| {
      Expr::Identifier(
        if integer.contains(v) {
          "Integers"
        } else {
          "Reals"
        }
        .into(),
      )
    })
    .collect();
  let solution = {
    let _quiet = QuietGuard::new();
    crate::functions::polynomial_ast::linear_programming_ast(&[
      Expr::List(cvec.into_iter().map(num).collect()),
      Expr::List(m.into()),
      Expr::List(b.into()),
      Expr::List(lu.into()),
      Expr::List(dom.into()),
    ])
    .ok()?
  };
  let Expr::List(values) = &solution else {
    return None;
  };
  if values.len() != n
    || values
      .iter()
      .any(|v| matches!(v, Expr::Identifier(s) if s == "Indeterminate"))
  {
    return None;
  }
  let values: Vec<Expr> = values
    .iter()
    .zip(vars)
    .map(|(v, name)| {
      if numeric && !integer.contains(name) {
        Expr::Real(try_eval_to_f64(v).unwrap_or(f64::NAN))
      } else {
        v.clone()
      }
    })
    .collect();
  let mut value = num(obj_const);
  for &(i, c) in &obj_terms {
    value = crate::evaluator::evaluate_expr_to_expr(&plus(vec![
      value,
      times(vec![num(c), values[i].clone()]),
    ]))
    .ok()?;
  }
  if numeric {
    value = Expr::Real(try_eval_to_f64(&value)?);
  }
  let rules: Vec<Expr> = vars
    .iter()
    .zip(values)
    .map(|(v, x)| rule_expr(Expr::Identifier(v.clone()), x))
    .collect();
  Some(Expr::List(vec![value, Expr::List(rules.into())].into()))
}

/// Try Integer Linear Programming. Returns Some(result) if ILP was solved, None if unsupported.
fn minimize_try_ilp(
  f: &Expr,
  constraints: &[Expr],
  vars: &[String],
  maximize: bool,
  func_name: &str,
) -> std::option::Option<Expr> {
  use std::collections::HashSet;

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
    return None;
  }

  // Extract linear objective coefficients
  let (obj_coeffs, _) = minimize_extract_linear_expr(f, vars)?;

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
      return None; // non-linear constraint
    }
  }

  // Only support single equality constraint for DP
  if equalities.len() != 1 {
    return None;
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
    return None;
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
  let mut all_values: Vec<f64> = eq_coeffs.clone();
  all_values.push(*eq_rhs);
  let scale = find_integer_scale(&all_values);
  if scale == 0 {
    return None;
  }

  let mut weights: Vec<i64> = Vec::with_capacity(vars.len());
  for &c in eq_coeffs {
    let scaled = c * scale as f64;
    let ci = scaled.round() as i64;
    if (scaled - ci as f64).abs() > 1e-4 || ci <= 0 {
      return None;
    }
    weights.push(ci);
  }
  let target_f = *eq_rhs * scale as f64;
  let target_i = target_f.round() as i64;
  if (target_f - target_i as f64).abs() > 1e-4 || target_i < 0 {
    return None;
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
    // Infeasible integer program: there is no divergence direction to
    // report, so keep the -Infinity end for the variable values.
    return Some(minimize_neg_infinity_result(vars, maximize, false));
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
      return None;
    }
    obj_int.push(ci);
  }

  // Guard: if the target is too large for DP (> 10M), bail out
  if shifted_target > 10_000_000 {
    return None;
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
            // `clone_from` would need `&dp_assign[t - wi]` while `dp_assign[t]`
            // is already mutably borrowed, so the clone stays explicit.
            #[allow(clippy::assigning_clones)]
            {
              dp_assign[t] = dp_assign[t - wi].clone();
            }
            dp_assign[t][i] += 1;
          }
        }
      }
    }
  }

  if dp[shifted_target] == INF {
    // Infeasible
    return Some(call(
      func_name,
      vec![
        f.clone(),
        Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
      ],
    ));
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
    .map(|(v, &val)| {
      rule_expr(Expr::Identifier(v.clone()), Expr::Integer(val as i128))
    })
    .collect();
  Some(Expr::List(
    vec![result_val, Expr::List(rules.into())].into(),
  ))
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
        coeffs[i] += minimize_try_f64(&c)?;
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
/// `Infinity` or `-Infinity` as an expression.
fn signed_infinity(positive: bool) -> Expr {
  let infinity = id_expr("Infinity");
  if positive {
    infinity
  } else {
    negate_expr(&infinity)
  }
}

/// Does `f` tend to -Infinity as `var` runs off to +/-Infinity? Used to
/// detect an unbounded 1-D minimization; anything the limit machinery cannot
/// decide counts as "no", so the caller falls back to its finite-candidate
/// search.
fn minimize_diverges_to_neg_infinity(
  f: &Expr,
  var: &str,
  toward_positive: bool,
) -> Result<bool, InterpreterError> {
  let limit = crate::functions::calculus_ast::limit_ast(&[
    f.clone(),
    rule_expr(id_expr(var), signed_infinity(toward_positive)),
  ])?;
  // `-Infinity` reaches here in several shapes (Times[-1, Infinity],
  // DirectedInfinity[-1], a negated identifier); compare the rendered form.
  Ok(expr_to_string(&limit) == "-Infinity")
}

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

  // A chained comparison (`1 < x < 3`) carries several relations in one node;
  // split it into the pairwise relations the bound scan below understands.
  let mut pairwise: Vec<Expr> = Vec::new();
  for con in constraints {
    match con {
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() > 2 && operators.len() == operands.len() - 1 => {
        for (i, op) in operators.iter().enumerate() {
          pairwise.push(Expr::Comparison {
            operands: vec![operands[i].clone(), operands[i + 1].clone()],
            operators: vec![*op],
          });
        }
      }
      other => pairwise.push(other.clone()),
    }
  }

  for con in &pairwise {
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
          // Strict inequalities bound the same closure as their non-strict
          // counterparts, so the boundary point is kept either way. Both
          // orientations occur: `x > 1` and the `1 < x` a chained
          // `1 < x < 3` splits into.
          ComparisonOp::Greater => {
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
          ComparisonOp::Less => {
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
    return Ok(call(func_name, vec![obj_with_cons, id_expr(var)]));
  }

  // An unbounded feasible direction can carry the objective off to infinity,
  // in which case there is no finite extremum. `f` here is the internally
  // minimized objective (already negated for Maximize), so a limit of
  // -Infinity toward a missing bound means the problem is unbounded.
  for (bound, toward_positive) in [(lb, false), (ub, true)] {
    if bound.is_some() {
      continue;
    }
    if !minimize_diverges_to_neg_infinity(f, var, toward_positive)? {
      continue;
    }
    crate::emit_message(&format!(
      "{}::natt: The {} is not attained at any point satisfying the given constraints.",
      func_name,
      if maximize { "maximum" } else { "minimum" }
    ));
    let rule = rule_expr(id_expr(var), signed_infinity(toward_positive));
    return Ok(Expr::List(
      vec![signed_infinity(maximize), Expr::List(vec![rule].into())].into(),
    ));
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
    return Ok(call(func_name, vec![obj_with_cons, id_expr(var)]));
  }

  // Find the minimum among candidates
  let mut best_f = f64::INFINITY;
  let mut best_x_f64 = candidates[0];
  for &cx in &candidates {
    // A bound can be a singular point of the objective; its value there
    // just loses the comparison.
    let fx = quietly(|| find_root_eval_at(f, var, cx)).unwrap_or(f64::INFINITY);
    if fx < best_f {
      best_f = fx;
      best_x_f64 = cx;
    }
  }

  // An unbounded end with a finite limit is an infimum the interior can only
  // approach, so it competes with the finite candidates (`1/x` on `x > 1`
  // bottoms out at 0, not at any point of the region).
  for (bound, toward_positive) in [(lb, false), (ub, true)] {
    if bound.is_some() {
      continue;
    }
    let limit = crate::functions::calculus_ast::limit_ast(&[
      f.clone(),
      rule_expr(id_expr(var), signed_infinity(toward_positive)),
    ])?;
    let Some(lv) = minimize_try_f64(&limit) else {
      continue;
    };
    if lv >= best_f {
      continue;
    }
    let rule = rule_expr(id_expr(var), signed_infinity(toward_positive));
    let value = if maximize {
      simplify(negate_expr(&limit))
    } else {
      limit
    };
    return Ok(Expr::List(
      vec![value, Expr::List(vec![rule].into())].into(),
    ));
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

  let rule = rule_expr(id_expr(var), result_x);
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
) -> Expr {
  // First try pure LP (linear objective + linear constraints)
  if vars.len() >= 2
    && let Some(result) = minimize_lp_2d(f, constraints, vars, maximize)
  {
    return result;
  }

  // For any dimension, try boundary reduction for linear constraints
  if let Some(result) =
    minimize_constrained_boundary(f, constraints, vars, maximize)
  {
    return result;
  }

  // Return unevaluated
  let obj_with_cons = Expr::List(
    std::iter::once(f.clone())
      .chain(constraints.iter().cloned())
      .collect(),
  );
  call(
    func_name,
    vec![
      obj_with_cons,
      Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
    ],
  )
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
  let diff = minus2(operands[0].clone(), operands[1].clone());
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
        // A non-constant coefficient means this is not a linear system.
        coeffs[i] += minimize_try_f64(&c)?;
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
) -> std::option::Option<Expr> {
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
        let term =
          times2(Expr::Real(coeffs[j]), Expr::Identifier(var_j.clone()));
        elim_expr = minus2(elim_expr, term);
      }
    }
    if (elim_coeff - 1.0).abs() > 1e-12 {
      elim_expr = div2(elim_expr, Expr::Real(elim_coeff));
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
    return None;
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
      rule_expr(Expr::Identifier(v.clone()), exact_val)
    })
    .collect();

  Some(Expr::List(
    vec![result_fval, Expr::List(rules.into())].into(),
  ))
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
    let diff = minus2(lhs.clone(), rhs.clone());
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
            call("Rational", vec![Expr::Integer(rn), Expr::Integer(rd)])
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
            call("Rational", vec![Expr::Integer(rn), Expr::Integer(rd)])
          };
          return Some(Expr::List(
            vec![
              e,
              Expr::List(
                vec![
                  rule_expr(
                    Expr::Identifier(x_name.clone()),
                    make_exact(best_vertex.0),
                  ),
                  rule_expr(
                    Expr::Identifier(y_name.clone()),
                    make_exact(best_vertex.1),
                  ),
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
          rule_expr(
            Expr::Identifier(x_name.clone()),
            make_exact(best_vertex.0),
          ),
          rule_expr(
            Expr::Identifier(y_name.clone()),
            make_exact(best_vertex.1),
          ),
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
    _ => Ok(call(
      if maximize {
        "FindMaxValue"
      } else {
        "FindMinValue"
      },
      args.to_vec(),
    )),
  }
}

/// FindMinimum[f, {x, x0}] — find a local minimum of f starting at x0
/// FindMinimum[f, {{x, x0}, {y, y0}}] — multivariable
/// Returns {min_value, {x -> x_min, ...}}
///
/// FindMaximum is implemented by negating f and negating the result.
/// The variable names of an optimization specification: `x`, `{x, x0}`,
/// `{x, y}` or `{{x, x0}, {y, y0}}`. `None` when the shape is not one of
/// those.
pub fn optimization_variable_names(spec: &Expr) -> Option<Vec<String>> {
  match spec {
    Expr::Identifier(name) => Some(vec![name.clone()]),
    Expr::List(items) if items.is_empty() => Some(Vec::new()),
    Expr::List(items) if items.iter().all(|i| matches!(i, Expr::List(_))) => {
      // {{x, x0}, {y, y0}, …}
      items
        .iter()
        .map(|item| match item {
          Expr::List(pair) if !pair.is_empty() => match &pair[0] {
            Expr::Identifier(name) => Some(name.clone()),
            _ => None,
          },
          _ => None,
        })
        .collect()
    }
    Expr::List(items)
      if items.iter().all(|i| matches!(i, Expr::Identifier(_))) =>
    {
      // {x, y, …}
      items
        .iter()
        .map(|i| match i {
          Expr::Identifier(name) => Some(name.clone()),
          _ => None,
        })
        .collect()
    }
    // {x, x0} — one variable with a starting value.
    Expr::List(items) if items.len() == 2 => match &items[0] {
      Expr::Identifier(name) => Some(vec![name.clone()]),
      _ => None,
    },
    _ => None,
  }
}

/// Quasi-Newton (BFGS) minimization of `phi` from `z0` with gradient
/// `grad` and a backtracking (Armijo) line search; `phi` is infinite
/// wherever a point is not allowed, which the line search then avoids.
/// `on_accept` sees every accepted point. The first step moves no
/// coordinate by more than a tenth of its own size (or 0.1 near zero): a
/// raw gradient step can be arbitrarily long and land on a far plateau the
/// search never climbs back from. A full step accepted outright is
/// extended while it keeps improving, and the first update takes the
/// Shanno–Phua scale `s.y / y.y`.
fn bfgs_minimize(
  phi: &dyn Fn(&[f64]) -> f64,
  grad: &dyn Fn(&[f64]) -> Vec<f64>,
  mut z: Vec<f64>,
  max_iter: usize,
  on_accept: &mut dyn FnMut(&[f64]),
) -> Vec<f64> {
  let m = z.len();
  let mut fz = phi(&z);
  let mut g = grad(&z);
  let first_scale = z
    .iter()
    .zip(&g)
    .filter(|(_, gi)| gi.abs() > 0.0)
    .map(|(zi, gi)| 0.1 * zi.abs().max(1.0) / gi.abs())
    .fold(1.0f64, f64::min);
  let mut hinv: Vec<Vec<f64>> = (0..m)
    .map(|i| {
      (0..m)
        .map(|j| if i == j { first_scale } else { 0.0 })
        .collect()
    })
    .collect();
  let mut updated = false;
  for _ in 0..max_iter {
    let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    if gnorm < 1e-10 * fz.abs().max(1.0) {
      break;
    }
    let dir: Vec<f64> = (0..m)
      .map(|i| -(0..m).map(|j| hinv[i][j] * g[j]).sum::<f64>())
      .collect();
    let slope: f64 = dir.iter().zip(&g).map(|(d, gi)| d * gi).sum();
    let dir = if slope < 0.0 {
      dir
    } else {
      g.iter().map(|v| -v).collect()
    };
    let slope: f64 = dir.iter().zip(&g).map(|(d, gi)| d * gi).sum();
    let mut t = 1.0;
    let mut accepted = None;
    for _ in 0..60 {
      let trial: Vec<f64> =
        z.iter().zip(&dir).map(|(zi, di)| zi + t * di).collect();
      let ft = phi(&trial);
      if ft.is_finite() && ft <= fz + 1e-4 * t * slope {
        accepted = Some((trial, ft));
        break;
      }
      t *= 0.5;
    }
    let Some((mut znew, mut fnew)) = accepted else {
      break;
    };
    if t == 1.0 {
      for _ in 0..20 {
        t *= 2.0;
        let trial: Vec<f64> =
          z.iter().zip(&dir).map(|(zi, di)| zi + t * di).collect();
        let ft = phi(&trial);
        if ft.is_finite() && ft < fnew && ft <= fz + 1e-4 * t * slope {
          znew = trial;
          fnew = ft;
        } else {
          break;
        }
      }
    }
    on_accept(&znew);
    let gnew = grad(&znew);
    let s_vec: Vec<f64> = znew.iter().zip(&z).map(|(a, b)| a - b).collect();
    let y_vec: Vec<f64> = gnew.iter().zip(&g).map(|(a, b)| a - b).collect();
    let sy: f64 = s_vec.iter().zip(&y_vec).map(|(a, b)| a * b).sum();
    if sy > 1e-14 {
      if !updated {
        let yy: f64 = y_vec.iter().map(|v| v * v).sum();
        if yy > 0.0 {
          for (i, row) in hinv.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
              *v = if i == j { sy / yy } else { 0.0 };
            }
          }
        }
        updated = true;
      }
      // H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T
      let rho = 1.0 / sy;
      let hy: Vec<f64> = (0..m)
        .map(|i| (0..m).map(|j| hinv[i][j] * y_vec[j]).sum())
        .collect();
      let yhy: f64 = y_vec.iter().zip(&hy).map(|(a, b)| a * b).sum();
      for i in 0..m {
        for j in 0..m {
          hinv[i][j] += (1.0 + rho * yhy) * rho * s_vec[i] * s_vec[j]
            - rho * (hy[i] * s_vec[j] + s_vec[i] * hy[j]);
        }
      }
    }
    let converged = (fz - fnew).abs() <= 1e-15 * fz.abs().max(1.0);
    z = znew;
    fz = fnew;
    g = gnew;
    if converged {
      break;
    }
  }
  z
}

/// Whether `e` is a polynomial in `vars` of total degree at most `max`.
fn polynomial_degree_at_most(e: &Expr, vars: &[String], max: i64) -> bool {
  let var_list =
    Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect());
  let is_poly = crate::evaluator::evaluate_expr_to_expr(&call(
    "PolynomialQ",
    vec![e.clone(), var_list],
  ));
  if !matches!(is_poly, Ok(Expr::Identifier(ref t)) if t == "True") {
    return false;
  }
  // The total degree is the degree in `t` once every variable is scaled
  // by `t`.
  let t = Expr::Identifier("FindMinimum$degree".to_string());
  let scaled: Vec<(String, Expr)> = vars
    .iter()
    .map(|v| {
      (
        v.clone(),
        call("Times", vec![t.clone(), Expr::Identifier(v.clone())]),
      )
    })
    .collect();
  let pairs: Vec<(&str, &Expr)> =
    scaled.iter().map(|(v, e)| (v.as_str(), e)).collect();
  let scaled_e = crate::syntax::substitute_variables(e, &pairs);
  matches!(
    crate::evaluator::evaluate_expr_to_expr(&call(
      "Exponent",
      vec![call("Expand", vec![scaled_e]), t],
    )),
    Ok(Expr::Integer(d)) if d <= i128::from(max)
  )
}

/// A local `FindMinimum` of `f` from the given starting values under
/// equalities and inequalities. The linear equalities (the mass balances of
/// a chemical-equilibrium Gibbs minimization) are solved for some of the
/// variables and each nonlinear one (an equation of state) defines one more
/// implicitly, so the search runs over the rest; the inequalities are kept
/// strictly, every trial point outside them being rejected — a logarithm of
/// a mole number must never see a negative argument. `monitor` is the
/// caller's name and `StepMonitor`. `None` when the problem has another
/// shape (or no feasible point is found), for the general solver.
fn find_minimum_local_constrained(
  f: &Expr,
  constraints: &[Expr],
  var_specs: &[Expr],
  maximize: bool,
  monitor: Option<(&str, &Expr)>,
) -> Option<Expr> {
  use super::reduce::{CompOp, extract_comparison};
  // Variables and starting values, `{x, x0}` each (possibly in one list).
  let mut vars: Vec<String> = Vec::new();
  let mut start: Vec<f64> = Vec::new();
  // `{x, x0, xmin, xmax}` also confines x to [xmin, xmax].
  let mut spec_bounds: Vec<Expr> = Vec::new();
  let mut take_spec = |spec: &Expr| -> Option<()> {
    let Expr::List(items) = spec else { return None };
    let (v, x0) = match items.as_slice() {
      [Expr::Identifier(v), x0] => (v, x0),
      [Expr::Identifier(v), x0, lo, hi] => {
        spec_bounds.push(Expr::Comparison {
          operands: vec![lo.clone(), Expr::Identifier(v.clone()), hi.clone()],
          operators: vec![ComparisonOp::LessEqual, ComparisonOp::LessEqual],
        });
        (v, x0)
      }
      _ => return None,
    };
    vars.push(v.clone());
    start.push(find_root_eval_number(x0).ok()?);
    Some(())
  };
  for spec in var_specs {
    match spec {
      Expr::List(items)
        if !items.is_empty()
          && items.iter().all(|i| matches!(i, Expr::List(_))) =>
      {
        for i in items {
          take_spec(i)?;
        }
      }
      other => take_spec(other)?,
    }
  }
  let n = vars.len();
  if n == 0 {
    return None;
  }
  // Flatten the constraints into single comparisons.
  fn flatten(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::List(items) => items.iter().for_each(|i| flatten(i, out)),
      Expr::FunctionCall { name, args } if name == "And" => {
        args.iter().for_each(|i| flatten(i, out));
      }
      Expr::Comparison {
        operands,
        operators,
      } if operators.len() > 1 => {
        for (k, op) in operators.iter().enumerate() {
          out.push(Expr::Comparison {
            operands: vec![operands[k].clone(), operands[k + 1].clone()],
            operators: vec![*op],
          });
        }
      }
      other => out.push(other.clone()),
    }
  }
  let mut flat = Vec::new();
  for c in constraints.iter().chain(&spec_bounds) {
    flatten(c, &mut flat);
  }
  let minus = |a: Expr, b: Expr| {
    crate::evaluator::evaluate_expr_to_expr(&call(
      "Plus",
      vec![a, call("Times", vec![Expr::Integer(-1), b])],
    ))
    .ok()
  };
  // Linear equalities as rows of A x = b; inequalities as g >= 0 / g > 0;
  // any other equality as g == 0.
  let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
  let mut nonlinear: Vec<Expr> = Vec::new();
  let mut ineqs: Vec<(Expr, Option<Expr>)> = Vec::new();
  // The position in `flat` each inequality came from.
  let mut ineq_source: Vec<usize> = Vec::new();
  for (ci, c) in flat.iter().enumerate() {
    if !matches!(extract_comparison(c), Some((_, _, CompOp::Equal)) | None) {
      ineq_source.push(ci);
    }
    let (lhs, rhs, op) = extract_comparison(c)?;
    match op {
      CompOp::Equal => {
        let g = minus(lhs, rhs)?;
        let mut coeffs = Vec::with_capacity(n);
        for v in &vars {
          let d =
            crate::functions::calculus_ast::differentiate_expr(&g, v).ok()?;
          let d = crate::evaluator::evaluate_expr_to_expr(&d).ok()?;
          match crate::functions::math_ast::try_eval_to_f64(&d) {
            Some(c) => coeffs.push(c),
            None => break,
          }
        }
        if coeffs.len() < n {
          nonlinear.push(g);
          continue;
        }
        let zero = Expr::Integer(0);
        let zero_point: Vec<(&str, &Expr)> =
          vars.iter().map(|v| (v.as_str(), &zero)).collect();
        let at_zero = crate::syntax::substitute_variables(&g, &zero_point);
        let c0 = crate::functions::math_ast::try_eval_to_f64(
          &crate::evaluator::evaluate_expr_to_expr(&at_zero).ok()?,
        )?;
        rows.push((coeffs, -c0));
      }
      CompOp::Greater => ineqs.push((minus(lhs, rhs)?, None)),
      CompOp::Less => ineqs.push((minus(rhs, lhs)?, None)),
      // A non-strict one may end up active: it can then be imposed as the
      // equality `lhs == rhs`.
      CompOp::GreaterEqual => {
        ineqs.push((
          minus(lhs.clone(), rhs.clone())?,
          Some(call("Equal", vec![lhs, rhs])),
        ));
      }
      CompOp::LessEqual => {
        ineqs.push((
          minus(rhs.clone(), lhs.clone())?,
          Some(call("Equal", vec![lhs, rhs])),
        ));
      }
      CompOp::NotEqual => return None,
    }
  }
  let has_equalities = !rows.is_empty() || !nonlinear.is_empty();
  if !has_equalities && ineqs.is_empty() {
    return None;
  }
  // wolframscript solves a linear or quadratic objective under linear
  // constraints (and any problem with equalities) by a method that does
  // not report its steps: a `StepMonitor` then gets `noopmon` instead.
  let monitorable = monitor.is_some()
    && !has_equalities
    && !(polynomial_degree_at_most(f, &vars, 2)
      && ineqs
        .iter()
        .all(|(g, _)| polynomial_degree_at_most(g, &vars, 1)));
  let report_unmonitored = || {
    if let Some((func_name, _)) = monitor {
      crate::emit_message(&format!(
        "{func_name}::noopmon: The optimization was solved by an algorithm that does not provide monitoring information. Choose a specific iterative method if this information is necessary."
      ));
    }
  };
  // Reduced row echelon form: pivot variables in terms of the free ones.
  let mut a: Vec<Vec<f64>> = rows
    .iter()
    .map(|(c, b)| {
      let mut r = c.clone();
      r.push(*b);
      r
    })
    .collect();
  let mut pivots: Vec<usize> = Vec::new();
  let mut row = 0;
  for col in 0..n {
    if row == a.len() {
      break;
    }
    let (best, val) = (row..a.len())
      .map(|r| (r, a[r][col].abs()))
      .max_by(|x, y| x.1.total_cmp(&y.1))?;
    if val < 1e-12 {
      continue;
    }
    a.swap(row, best);
    let p = a[row][col];
    for v in &mut a[row] {
      *v /= p;
    }
    for r in 0..a.len() {
      if r != row && a[r][col] != 0.0 {
        let factor = a[r][col];
        let pivot_row = a[row].clone();
        for (x, y) in a[r].iter_mut().zip(pivot_row) {
          *x -= factor * y;
        }
      }
    }
    pivots.push(col);
    row += 1;
  }
  // An inconsistent system has no feasible point at all.
  if a[row..].iter().any(|r| r[n].abs() > 1e-9) {
    return None;
  }
  let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
  // The point from the values of the free variables, the pivots following.
  let linear_point = |w: &[f64]| -> Vec<f64> {
    let mut x = vec![0.0; n];
    for (k, &c) in free.iter().enumerate() {
      x[c] = w[k];
    }
    for (r, &pc) in pivots.iter().enumerate() {
      let mut v = a[r][n];
      for &c in &free {
        v -= a[r][c] * x[c];
      }
      x[pc] = v;
    }
    x
  };
  // The derivative along free variable `c` of a function with gradient
  // `gx`, the pivots moving with it.
  let reduced = |gx: &[f64], c: usize| -> f64 {
    let mut v = gx[c];
    for (r, &pc) in pivots.iter().enumerate() {
      v -= a[r][c] * gx[pc];
    }
    v
  };
  let dense_gradient = |node: &NumNode, x: &[f64]| -> (f64, Vec<f64>) {
    let (v, sparse) = node.eval_dual(x);
    let mut g = vec![0.0; n];
    for (k, d) in sparse {
      g[k] = d;
    }
    (v, g)
  };
  // Each nonlinear equality determines one free variable implicitly (the
  // one it depends on most strongly at the start, by column pivoting); the
  // search runs over the remaining free variables, the dependent ones
  // re-solved by Newton's method at every point.
  let nl_nodes: Vec<NumNode> = nonlinear
    .iter()
    .map(|g| compile_numeric(g, &vars))
    .collect::<Option<_>>()?;
  let k_nl = nl_nodes.len();
  let w0: Vec<f64> = free.iter().map(|&c| start[c]).collect();
  let mut dependent: Vec<usize> = Vec::new();
  if k_nl > 0 {
    let x0 = linear_point(&w0);
    let mut jac: Vec<Vec<f64>> = nl_nodes
      .iter()
      .map(|h| {
        let (_, g) = dense_gradient(h, &x0);
        free.iter().map(|&c| reduced(&g, c)).collect()
      })
      .collect();
    for j in 0..k_nl {
      let (col, val) = (0..free.len())
        .filter(|c| !dependent.contains(c))
        .map(|c| (c, jac[j][c].abs()))
        .max_by(|x, y| x.1.total_cmp(&y.1))?;
      if val.is_nan() || val <= 1e-12 {
        return None;
      }
      for r in j + 1..k_nl {
        let factor = jac[r][col] / jac[j][col];
        let pivot_row = jac[j].clone();
        for (x, y) in jac[r].iter_mut().zip(pivot_row) {
          *x -= factor * y;
        }
      }
      dependent.push(col);
    }
  }
  let search: Vec<usize> =
    (0..free.len()).filter(|c| !dependent.contains(c)).collect();
  // The dependent values Newton's method starts from: those of the last
  // accepted point, so the solution stays on one branch.
  let anchor = std::cell::RefCell::new(
    dependent.iter().map(|&d| w0[d]).collect::<Vec<f64>>(),
  );
  let full_point_from =
    |z: &[f64], y0: &[f64]| -> Option<(Vec<f64>, Vec<f64>)> {
      let mut w = vec![0.0; free.len()];
      for (k, &c) in search.iter().enumerate() {
        w[c] = z[k];
      }
      let mut y = y0.to_vec();
      if k_nl == 0 {
        return Some((linear_point(&w), y));
      }
      for _ in 0..100 {
        for (k, &c) in dependent.iter().enumerate() {
          w[c] = y[k];
        }
        let x = linear_point(&w);
        let mut jac = Vec::with_capacity(k_nl);
        let mut rhs = Vec::with_capacity(k_nl);
        for h in &nl_nodes {
          let (v, g) = dense_gradient(h, &x);
          jac.push(
            dependent
              .iter()
              .map(|&c| reduced(&g, free[c]))
              .collect::<Vec<_>>(),
          );
          rhs.push(-v);
        }
        let step = solve_dense_linear_system(&jac, &rhs)?;
        if !step.iter().all(|d| d.is_finite()) {
          return None;
        }
        let mut small = true;
        for (yk, d) in y.iter_mut().zip(&step) {
          small &= d.abs() <= 1e-14 * yk.abs().max(1.0);
          *yk += d;
        }
        if small {
          for (k, &c) in dependent.iter().enumerate() {
            w[c] = y[k];
          }
          return Some((linear_point(&w), y));
        }
      }
      None
    };
  let full_point =
    |z: &[f64]| full_point_from(z, &anchor.borrow()).map(|(x, _)| x);
  let objective = compile_numeric(f, &vars)?;
  let ineq_nodes: Vec<NumNode> = ineqs
    .iter()
    .map(|(g, _)| compile_numeric(g, &vars))
    .collect::<Option<_>>()?;
  let sign = if maximize { -1.0 } else { 1.0 };
  // Strictly inside every inequality (a boundary point is no more use to a
  // logarithm than an outside one).
  let feasible = |x: &[f64]| ineq_nodes.iter().all(|g| g.eval(x) > 0.0);
  let phi = |z: &[f64]| -> f64 {
    let Some(x) = full_point(z) else {
      return f64::INFINITY;
    };
    if !feasible(&x) {
      return f64::INFINITY;
    }
    let v = sign * objective.eval(&x);
    if v.is_finite() { v } else { f64::INFINITY }
  };
  let mut z: Vec<f64> = search.iter().map(|&c| w0[c]).collect();
  // Phase one: walk into the feasible region when the start lies outside.
  if !phi(&z).is_finite() {
    let violation = |zz: &[f64]| -> f64 {
      let Some(x) = full_point(zz) else {
        return 1e6 * ineq_nodes.len().max(1) as f64;
      };
      ineq_nodes
        .iter()
        .map(|g| {
          let v = g.eval(&x);
          if v.is_finite() {
            (1e-6 - v).max(0.0).powi(2)
          } else {
            1e6
          }
        })
        .sum()
    };
    for _ in 0..20 {
      z = nelder_mead_minimize(&violation, &|_: &mut [f64]| {}, &z, 2000);
      if phi(&z).is_finite() {
        break;
      }
    }
    if !phi(&z).is_finite() {
      return None;
    }
  }
  // The points the search visits, for a `StepMonitor`.
  let steps: std::cell::RefCell<Vec<Vec<f64>>> =
    std::cell::RefCell::new(full_point(&z).into_iter().collect());
  // Phase two: quasi-Newton (BFGS) with central-difference gradients and a
  // backtracking line search that stays inside the feasible region.
  let m = z.len();
  // The gradient in the free variables from the symbolic partials of f:
  // dx_free/dz is the identity and dx_pivot/dz the negated echelon entries.
  let partials: Option<Vec<NumNode>> = vars
    .iter()
    .map(|v| {
      let d = crate::functions::calculus_ast::differentiate_expr(f, v).ok()?;
      compile_numeric(&d, &vars).filter(|node| !node.has_fallback())
    })
    .collect();
  // With dependent variables, the implicit function theorem: the
  // multipliers solve J_y^T lambda = grad_y f and the gradient along a
  // search variable is its own partial less lambda . J_z.
  let exact_grad = |zz: &[f64]| -> Option<Vec<f64>> {
    let x = full_point(zz)?;
    let gx: Vec<f64> = match partials.as_ref() {
      Some(parts) => parts.iter().map(|p| sign * p.eval(&x)).collect(),
      None if k_nl > 0 => dense_gradient(&objective, &x)
        .1
        .iter()
        .map(|d| sign * d)
        .collect(),
      None => return None,
    };
    let mut g: Vec<f64> =
      search.iter().map(|&c| reduced(&gx, free[c])).collect();
    if k_nl > 0 {
      let hx: Vec<Vec<f64>> =
        nl_nodes.iter().map(|h| dense_gradient(h, &x).1).collect();
      let jac_t: Vec<Vec<f64>> = dependent
        .iter()
        .map(|&c| hx.iter().map(|gh| reduced(gh, free[c])).collect())
        .collect();
      let gy: Vec<f64> =
        dependent.iter().map(|&c| reduced(&gx, free[c])).collect();
      let lambda = solve_dense_linear_system(&jac_t, &gy)?;
      for (gk, &c) in g.iter_mut().zip(&search) {
        for (l, gh) in lambda.iter().zip(&hx) {
          *gk -= l * reduced(gh, free[c]);
        }
      }
    }
    g.iter().all(|v| v.is_finite()).then_some(g)
  };
  let grad = |zz: &[f64]| -> Vec<f64> {
    if let Some(g) = exact_grad(zz) {
      return g;
    }
    (0..m)
      .map(|k| {
        let h = 1e-6 * zz[k].abs().max(1.0);
        let mut p = zz.to_vec();
        p[k] += h;
        let fp = phi(&p);
        p[k] -= 2.0 * h;
        let fm = phi(&p);
        if fp.is_finite() && fm.is_finite() {
          (fp - fm) / (2.0 * h)
        } else if fp.is_finite() {
          (fp - phi(zz)) / h
        } else {
          (phi(zz) - fm) / h
        }
      })
      .collect()
  };
  if m > 0 {
    z = bfgs_minimize(&phi, &grad, z, 500, &mut |znew: &[f64]| {
      let solved = full_point_from(znew, &anchor.borrow());
      if let Some((x, y)) = solved {
        *anchor.borrow_mut() = y;
        if monitorable {
          steps.borrow_mut().push(x);
        }
      }
    });
  }
  let x = full_point(&z)?;
  let value = objective.eval(&x);
  if !value.is_finite() {
    return None;
  }
  // A non-strict inequality the search ended up pressed against is active:
  // impose it as an equality and solve again from here, which lands on the
  // boundary exactly rather than as close as the line search got.
  let scale = x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
  if let Some(k) = ineqs.iter().enumerate().position(|(k, (_, eq))| {
    eq.is_some() && ineq_nodes[k].eval(&x).abs() < 1e-6 * scale
  }) {
    let mut tightened: Vec<Expr> = flat
      .iter()
      .enumerate()
      .filter(|(i, _)| *i != ineq_source[k])
      .map(|(_, c)| c.clone())
      .collect();
    tightened.push(ineqs[k].1.clone()?);
    let specs: Vec<Expr> = vars
      .iter()
      .zip(&x)
      .map(|(v, xv)| {
        Expr::List(vec![Expr::Identifier(v.clone()), Expr::Real(*xv)].into())
      })
      .collect();
    if let Some(r) = find_minimum_local_constrained(
      f,
      &tightened,
      &[Expr::List(specs.into())],
      maximize,
      None,
    ) {
      report_unmonitored();
      return Some(r);
    }
  }
  // The search's steps are the caller's to monitor when it ran unhindered
  // to a stationary point (wolframscript's interior method); otherwise
  // the problem was solved another way that reports nothing.
  let stationary = {
    let g = grad(&z);
    let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    gnorm.is_finite() && gnorm <= 1e-6 * value.abs().max(1.0)
  };
  if let Some((_, m)) = monitor
    && monitorable
    && stationary
  {
    for point in steps.borrow().iter() {
      fire_step_monitor(Some(m), &vars, point);
    }
  } else {
    report_unmonitored();
  }
  let rules: Vec<Expr> = vars
    .iter()
    .zip(&x)
    .map(|(v, xv)| Expr::Rule {
      pattern: Box::new(Expr::Identifier(v.clone())),
      replacement: Box::new(Expr::Real(*xv)),
    })
    .collect();
  Some(Expr::List(
    vec![Expr::Real(value), Expr::List(rules.into())].into(),
  ))
}

/// Derivative-free Nelder–Mead simplex search for a local minimum of
/// `eval` (already signed for maximize, and never returning an error —
/// callers score a failed trial point as `f64::INFINITY`). Used by
/// [`find_minimum_ast`] whenever `f` isn't symbolically differentiable.
pub(crate) fn nelder_mead_minimize(
  eval: &impl Fn(&[f64]) -> f64,
  clamp: &impl Fn(&mut [f64]),
  x0: &[f64],
  max_iter: usize,
) -> Vec<f64> {
  let n = x0.len();
  if n == 0 {
    return x0.to_vec();
  }
  // Initial simplex: x0 plus one vertex per axis, offset by 5% of that
  // coordinate's own scale (or a fixed 0.05 at/near the origin).
  let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
  for i in 0..n {
    let mut p = x0.to_vec();
    p[i] += if p[i].abs() > 1e-8 {
      p[i].abs() * 0.05
    } else {
      0.05
    };
    clamp(&mut p);
    simplex.push(p);
  }
  let mut fvals: Vec<f64> = simplex.iter().map(|p| eval(p)).collect();

  for _ in 0..max_iter {
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
      fvals[a]
        .partial_cmp(&fvals[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    simplex = order.iter().map(|&i| simplex[i].clone()).collect();
    fvals = order.iter().map(|&i| fvals[i]).collect();

    // Equal values alone are not convergence: a simplex straddling the
    // minimum symmetrically (3.95 and 4.05 around 4) has them too.
    let spread = simplex[1..]
      .iter()
      .flat_map(|p| p.iter().zip(simplex[0].iter()).map(|(a, b)| (a - b).abs()))
      .fold(0.0_f64, f64::max);
    let scale = simplex[0].iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    if (fvals[n] - fvals[0]).abs() < 1e-14 * (1.0 + fvals[0].abs())
      && spread < 1e-8 * scale
    {
      break;
    }

    // Centroid of every vertex but the worst.
    let mut centroid = vec![0.0; n];
    for p in simplex.iter().take(n) {
      for (c, pj) in centroid.iter_mut().zip(p.iter()) {
        *c += pj;
      }
    }
    for c in &mut centroid {
      *c /= n as f64;
    }

    let worst = simplex[n].clone();
    let reflect = |scale: f64| -> Vec<f64> {
      centroid
        .iter()
        .zip(worst.iter())
        .map(|(c, w)| c + scale * (c - w))
        .collect()
    };

    let mut xr = reflect(1.0);
    clamp(&mut xr);
    let fr = eval(&xr);
    if fr < fvals[0] {
      let mut xe = reflect(2.0);
      clamp(&mut xe);
      let fe = eval(&xe);
      if fe < fr {
        simplex[n] = xe;
        fvals[n] = fe;
      } else {
        simplex[n] = xr;
        fvals[n] = fr;
      }
    } else if fr < fvals[n - 1] {
      simplex[n] = xr;
      fvals[n] = fr;
    } else {
      let mut xc = reflect(-0.5);
      clamp(&mut xc);
      let fc = eval(&xc);
      if fc < fvals[n] {
        simplex[n] = xc;
        fvals[n] = fc;
      } else {
        // Shrink every vertex but the best toward the best.
        let best = simplex[0].clone();
        for k in 1..=n {
          for (sk, bk) in simplex[k].iter_mut().zip(best.iter()) {
            *sk = bk + 0.5 * (*sk - bk);
          }
          clamp(&mut simplex[k]);
          fvals[k] = eval(&simplex[k]);
        }
      }
    }
  }

  let best_i = (0..=n)
    .min_by(|&a, &b| {
      fvals[a]
        .partial_cmp(&fvals[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    })
    .unwrap_or(0);
  let mut best = simplex[best_i].clone();
  let mut best_f = fvals[best_i];
  // The simplex only brackets the minimum to its own tolerance; a few
  // parabolic steps per coordinate (the vertex of the parabola through
  // `x - h`, `x`, `x + h`) finish a smooth minimum to machine precision.
  for _ in 0..3 {
    for i in 0..n {
      let h = 1e-4 * best[i].abs().max(1.0);
      let probe = |delta: f64| {
        let mut p = best.clone();
        p[i] += delta;
        clamp(&mut p);
        (eval(&p), p)
      };
      let (f_minus, _) = probe(-h);
      let (f_plus, _) = probe(h);
      let curvature = f_plus - 2.0 * best_f + f_minus;
      if !(curvature > 0.0 && curvature.is_finite()) {
        continue;
      }
      let (f_new, p_new) = probe(-h * (f_plus - f_minus) / (2.0 * curvature));
      if f_new <= best_f {
        best = p_new;
        best_f = f_new;
      }
    }
  }
  best
}

/// `FindMinimum`/`FindMaximum` over indexed variables (`{n[j], 0.15}`):
/// the objective (and constraints) are evaluated with the variables free,
/// every variable is renamed to a fresh symbol the way `FindRoot` does, and
/// the result's rules are mapped back. None when every variable is already
/// a symbol.
fn find_minimum_indexed_variables(
  args: &[Expr],
  maximize: bool,
) -> Option<Result<Expr, InterpreterError>> {
  let spec_arg_in = &args[1];
  let is_var = |e: &Expr| {
    matches!(
      e,
      Expr::Identifier(_)
        | Expr::FunctionCall { .. }
        | Expr::CurriedCall { .. }
    )
  };
  // `{c[1], c[2]}`: bare variables, no starting values.
  let bare = matches!(spec_arg_in, Expr::List(items)
    if items.len() > 1 && items.iter().all(is_var));
  let specs: Vec<Expr> = match spec_arg_in {
    Expr::List(items) if bare => items.to_vec(),
    Expr::List(items) if items.iter().all(|i| matches!(i, Expr::List(_))) => {
      items.to_vec()
    }
    single @ Expr::List(_) => vec![single.clone()],
    _ => return None,
  };
  let vars: Vec<Expr> = specs
    .iter()
    .map(|spec| match spec {
      Expr::List(parts) if !bare => parts.first().cloned(),
      v if bare => Some(v.clone()),
      _ => None,
    })
    .collect::<Option<_>>()?;
  if !vars
    .iter()
    .any(|v| matches!(v, Expr::FunctionCall { .. } | Expr::CurriedCall { .. }))
  {
    return None;
  }
  let fresh: Vec<Expr> = (0..vars.len())
    .map(|i| Expr::Identifier(format!("$FindMinimumVar{i}$")))
    .collect();
  let objective = crate::evaluator::evaluate_expr_to_expr(&args[0])
    .unwrap_or_else(|_| args[0].clone());
  let renamed = find_root_rename_vars(&objective, &vars, &fresh);
  let new_specs: Vec<Expr> = specs
    .iter()
    .zip(&fresh)
    .map(|(spec, name)| match spec {
      Expr::List(parts) if !bare => {
        let mut parts = parts.to_vec();
        parts[0] = name.clone();
        Expr::List(parts.into())
      }
      _ if bare => name.clone(),
      other => other.clone(),
    })
    .collect();
  let spec_arg = if bare
    || matches!(spec_arg_in, Expr::List(items) if items.iter().all(|i| matches!(i, Expr::List(_))))
  {
    Expr::List(new_specs.into())
  } else {
    new_specs.into_iter().next()?
  };
  let mut new_args = vec![renamed, spec_arg];
  new_args.extend_from_slice(&args[2..]);
  Some(
    find_minimum_ast(&new_args, maximize)
      .map(|result| find_root_rename_vars(&result, &fresh, &vars)),
  )
}

/// Largest variable count for which `FindMinimum` builds the Hessian
/// symbolically; beyond it the Hessian is differenced from the gradient.
const SYMBOLIC_HESSIAN_MAX_VARS: usize = 6;

pub fn find_minimum_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  // A search that ran into a non-number reports it as Indeterminate, never
  // as a machine NaN (which no comparison can order).
  fn nan_free(e: &Expr) -> Expr {
    match e {
      Expr::Real(f) if f.is_nan() => Expr::Identifier("Indeterminate".into()),
      Expr::List(items) => Expr::List(items.iter().map(nan_free).collect()),
      Expr::Rule {
        pattern,
        replacement,
      } => Expr::Rule {
        pattern: pattern.clone(),
        replacement: Box::new(nan_free(replacement)),
      },
      other => other.clone(),
    }
  }
  find_minimum_inner(args, maximize).map(|r| nan_free(&r))
}

/// A local minimum of `f` near the starting values `a` and `b`: the
/// golden-section point between them brackets one when it lies below both,
/// otherwise the bracket is grown downhill from the lower start; Brent's
/// parabolic/golden-section search then narrows it. None when no finite
/// bracket turns up.
fn two_point_brent_minimum(
  f: impl Fn(f64) -> f64,
  a: f64,
  b: f64,
) -> Option<f64> {
  const GOLD: f64 = std::f64::consts::GOLDEN_RATIO;
  const CGOLD: f64 = 0.381_966_011_250_105;
  let (fa, fb) = (f(a), f(b));
  let m = a + CGOLD * (b - a);
  let fm = f(m);
  let (mut ax, bx, mut cx);
  if fm < fa && fm < fb {
    (ax, bx, cx) = (a, m, b);
  } else {
    // Downhill from the higher start towards (and past) the lower one.
    let (mut p, mut fp, mut q, mut fq) = if fb <= fa {
      (a, fa, b, fb)
    } else {
      (b, fb, a, fa)
    };
    let mut r = q + GOLD * (q - p);
    let mut fr = f(r);
    let mut steps = 0;
    while fr < fq {
      (p, fp, q, fq) = (q, fq, r, fr);
      r = q + GOLD * (q - p);
      fr = f(r);
      steps += 1;
      if steps > 100 || !r.is_finite() {
        return None;
      }
    }
    let _ = fp;
    (ax, bx, cx) = (p, q, r);
  }
  if !f(bx).is_finite() {
    return None;
  }
  // Brent's method on the bracket (ax, bx, cx).
  if ax > cx {
    std::mem::swap(&mut ax, &mut cx);
  }
  let (mut lo, mut hi) = (ax, cx);
  let (mut x, mut w, mut v) = (bx, bx, bx);
  let (mut fx, mut fw, mut fv) = (f(bx), f(bx), f(bx));
  let (mut d, mut e) = (0.0f64, 0.0f64);
  for _ in 0..200 {
    let xm = f64::midpoint(lo, hi);
    let tol1 = 1.0e-10 * x.abs() + 1.0e-20;
    let tol2 = 2.0 * tol1;
    if (x - xm).abs() <= tol2 - 0.5 * (hi - lo) {
      break;
    }
    let mut golden = true;
    if e.abs() > tol1 {
      let r = (x - w) * (fx - fv);
      let mut q = (x - v) * (fx - fw);
      let mut p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if q > 0.0 {
        p = -p;
      }
      q = q.abs();
      let etemp = e;
      e = d;
      if p.abs() < (0.5 * q * etemp).abs()
        && p > q * (lo - x)
        && p < q * (hi - x)
      {
        d = p / q;
        let u = x + d;
        if u - lo < tol2 || hi - u < tol2 {
          d = tol1.copysign(xm - x);
        }
        golden = false;
      }
    }
    if golden {
      e = if x >= xm { lo - x } else { hi - x };
      d = CGOLD * e;
    }
    let u = if d.abs() >= tol1 {
      x + d
    } else {
      x + tol1.copysign(d)
    };
    let fu = f(u);
    if fu <= fx {
      if u >= x {
        lo = x;
      } else {
        hi = x;
      }
      (v, fv, w, fw, x, fx) = (w, fw, x, fx, u, fu);
    } else {
      if u < x {
        lo = u;
      } else {
        hi = u;
      }
      if fu <= fw || w == x {
        (v, fv, w, fw) = (w, fw, u, fu);
      } else if fu <= fv || v == x || v == w {
        (v, fv) = (u, fu);
      }
    }
  }
  fx.is_finite().then_some(x)
}

fn find_minimum_inner(
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
  // The variables may be named by a symbol holding their list
  // (`vars = {x, y}; FindMinimum[f, vars]`).
  if let Expr::Identifier(_) = &args[1]
    && let Ok(value @ Expr::List(_)) =
      crate::evaluator::evaluate_expr_to_expr(&args[1])
  {
    let mut resolved = args.to_vec();
    resolved[1] = value;
    return find_minimum_ast(&resolved, maximize);
  }
  if let Some(result) = find_minimum_indexed_variables(args, maximize) {
    return result;
  }
  // A `{f, cons…}` first argument states a constrained problem, which the
  // constrained solver handles; the starting values are only a hint and it
  // takes the variables on their own. wolframscript reports the same
  // `{value, {var -> …}}` shape from either.
  // `{x, x0}, {y, y0}, …` as separate arguments name one variable each.
  let single_var_spec = |e: &Expr| {
    matches!(e, Expr::List(items)
      if (2..=4).contains(&items.len())
        && matches!(items[0], Expr::Identifier(_))
        && !items.iter().all(|i| matches!(i, Expr::Identifier(_))))
  };
  let mut var_args_end = 2;
  if single_var_spec(&args[1]) {
    while var_args_end < args.len() && single_var_spec(&args[var_args_end]) {
      var_args_end += 1;
    }
  }
  if let Expr::List(items) = &args[0]
    && items.len() >= 2
    && let Some(result) = find_minimum_local_constrained(
      &items[0],
      &items[1..],
      &args[1..var_args_end],
      maximize,
      find_option(&args[var_args_end..], "StepMonitor").map(|m| (func_name, m)),
    )
  {
    return Ok(result);
  }
  if let Expr::List(items) = &args[0]
    && items.len() >= 2
    && let Some(vars) = args[1..var_args_end]
      .iter()
      .map(optimization_variable_names)
      .collect::<Option<Vec<_>>>()
      .map(|names| names.concat())
    && !vars.is_empty()
  {
    let step_monitor = monitor_after_noopmon(func_name, &args[var_args_end..]);
    let _quiet = QuietGuard::new();
    return nminimize_ast_impl(
      &[
        args[0].clone(),
        Expr::List(vars.into_iter().map(Expr::Identifier).collect()),
      ],
      maximize,
      step_monitor,
    );
  }

  // `StepMonitor :> …` fires once per accepted iteration below.
  let step_monitor = find_option(&args[2..], "StepMonitor");
  let evaluation_monitor = find_option(&args[2..], "EvaluationMonitor");

  // Trailing arguments are options (e.g. MaxIterations -> 2,
  // Method -> "Newton"). They aren't honoured yet, but we accept them
  // silently rather than aborting so call shapes match Wolfram.
  // Only the first two positional arguments drive the optimisation.
  //
  // `{f}` (a one-element list) is `{f, constraints...}` with zero
  // constraints — plain unconstrained `f`, same as writing it bare. Unwrap
  // it up front so the rest of this function (symbolic differentiation in
  // particular) never has to differentiate a `List`-headed expression: a
  // second differentiation of an already-`List`-wrapped derivative hits a
  // `differentiate_expr` case the ordinary `D[…]` entry point (which
  // threads over a `List` argument itself before ever calling
  // `differentiate_expr` on a bare, non-`List` factor) does not exercise.
  let unwrapped_f;
  let f = match &args[0] {
    Expr::List(items) if items.len() == 1 => {
      unwrapped_f = items[0].clone();
      &unwrapped_f
    }
    other => other,
  };

  // A single variable spec: {x, x0} (unconstrained) or {x, x0, xmin, xmax}
  // (`x` constrained to stay within [xmin, xmax] throughout the search).
  // Returns `None` when `pair` doesn't have that shape.
  let parse_bounded_var_spec = |pair: &[Expr]| -> Option<
    Result<(String, f64, Option<(f64, f64)>), InterpreterError>,
  > {
    let Expr::Identifier(name) = &pair[0] else {
      return None;
    };
    match pair.len() {
      // `{x, x0, x1}` names two starting values; the search starts from
      // the first (Wolfram's derivative-free method uses both).
      2 | 3 => {
        Some(find_root_eval_number(&pair[1]).map(|x0| (name.clone(), x0, None)))
      }
      4 => Some((|| {
        let x0 = find_root_eval_number(&pair[1])?;
        let xmin = find_root_eval_number(&pair[2])?;
        let xmax = find_root_eval_number(&pair[3])?;
        Ok((name.clone(), x0, Some((xmin.min(xmax), xmin.max(xmax)))))
      })()),
      _ => None,
    }
  };

  // Wolfram also accepts each variable spec as its own trailing positional
  // argument: `FindMinimum[f, {x, x0}, {y, y0}]` means the same as
  // `FindMinimum[f, {{x, x0}, {y, y0}}]`. Detect that shape — args[1] itself
  // a bare `{var, start}`/`{var, start, min, max}` pair — and fold any
  // further same-shaped trailing arguments into one combined multivariable
  // spec before the parser below runs, so both forms share one code path.
  // Stops at the first argument that isn't shaped like a variable spec
  // (the start of options such as `MaxIterations -> n`).
  let is_var_spec_shape = |e: &Expr| -> bool {
    matches!(e, Expr::List(items) if (items.len() == 2 || items.len() == 4)
      && matches!(items.first(), Some(Expr::Identifier(_))))
  };
  let mut var_spec_end = 2;
  if is_var_spec_shape(&args[1]) {
    while var_spec_end < args.len() && is_var_spec_shape(&args[var_spec_end]) {
      var_spec_end += 1;
    }
  }
  let combined_spec: Expr = if var_spec_end > 2 {
    let mut items = vec![args[1].clone()];
    items.extend(args[2..var_spec_end].iter().cloned());
    Expr::List(items.into())
  } else {
    args[1].clone()
  };

  // `{x, x0, x1}` names two starting values for a derivative-free search:
  // bracket a minimum from them and narrow it with Brent's method, then let
  // the search below polish it from there. Started from x0 alone, a point
  // where the objective is flat or not differentiable (`Abs[BesselJ[1, x]]`
  // at 0) would stay put.
  if let Expr::List(items) = &combined_spec
    && let [Expr::Identifier(var), x0, x1] = items.as_slice()
    && let (Ok(x0v), Ok(x1v)) =
      (find_root_eval_number(x0), find_root_eval_number(x1))
    && x0v != x1v
  {
    let sign = if maximize { -1.0 } else { 1.0 };
    let objective = |x: f64| {
      quietly(|| find_root_eval_at(f, var, x))
        .ok()
        .filter(|v| v.is_finite())
        .map_or(f64::INFINITY, |v| sign * v)
    };
    if let Some(best) = two_point_brent_minimum(objective, x0v, x1v) {
      let mut polished = args.to_vec();
      polished[1] = Expr::List(
        vec![Expr::Identifier(var.clone()), Expr::Real(best)].into(),
      );
      return find_minimum_inner(&polished, maximize);
    }
  }

  // Parse variables and starting points: x, {x, y}, {x, x0}, {x, x0, xmin,
  // xmax} or {{x, x0}, {y, y0}} (each pair optionally carrying its own
  // xmin/xmax bounds too). Bare symbols get Wolfram's automatic starting
  // point of 1 (FindMinimum[f, x] == FindMinimum[f, {x, 1}]).
  let var_specs = match &combined_spec {
    Expr::Identifier(name) => vec![(name.clone(), 1.0, None)],
    Expr::List(items)
      if !items.is_empty() && matches!(&items[0], Expr::List(_)) =>
    {
      // Multivariable: {{x, x0}, {y, y0}, ...}
      let mut specs = Vec::new();
      for item in items {
        if let Expr::List(pair) = item
          && let Some(spec) = parse_bounded_var_spec(pair)
        {
          specs.push(spec?);
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
          Expr::Identifier(name) => (name.clone(), 1.0, None),
          _ => unreachable!(),
        })
        .collect()
    }
    // Single variable: {x, x0}, {x, x0, x1} or {x, x0, xmin, xmax}
    Expr::List(items) if (2..=4).contains(&items.len()) => {
      match parse_bounded_var_spec(items) {
        Some(spec) => vec![spec?],
        None => {
          return Err(InterpreterError::EvaluationError(format!(
            "{func_name}: variable spec must be {{var, start}}"
          )));
        }
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "{func_name}: second argument must be {{var, start}} or {{{{x, x0}}, {{y, y0}}}}"
      )));
    }
  };

  let vars: Vec<String> = var_specs.iter().map(|(v, ..)| v.clone()).collect();
  let mut x: Vec<f64> = var_specs.iter().map(|(_, x0, _)| *x0).collect();
  let bounds: Vec<Option<(f64, f64)>> =
    var_specs.iter().map(|(_, _, b)| *b).collect();
  let clamp = |point: &mut [f64]| {
    for (xi, b) in point.iter_mut().zip(bounds.iter()) {
      if let Some((lo, hi)) = b {
        *xi = xi.clamp(*lo, *hi);
      }
    }
  };
  clamp(&mut x);
  let n = vars.len();

  // Compute the symbolic gradient and Hessian (for Newton's method). Either
  // step can fail outright — not just fail to reduce to a real number,
  // which `symbolic_works` below already screens for, but return a hard
  // `Err` — when `f` calls an opaque (unrecognized) function with a large
  // List-valued argument that isn't itself a differentiation variable
  // (e.g. a fixed dataset passed alongside the parameters being fit):
  // `differentiate_expr` represents "no dependence here" for such an
  // argument with a same-shape list of zero orders, matching how Wolfram's
  // own `Derivative` generalizes over vector arguments, but building the
  // *second* derivative from an already-List-shaped first one this way is
  // not yet implemented for every case. Since this is exactly the same
  // "not symbolically differentiable" situation the `_?NumericQ`-guarded
  // case below already falls back from, a failure here falls back the same
  // way — the derivative-free branch never touches `grad_exprs`/
  // `hess_exprs`, so leaving them empty is safe.
  // With many variables the symbolic Hessian (n^2 derivatives of what is
  // often a large sum) costs far more than the search itself: an objective
  // that compiles is minimized by BFGS — wolframscript's default
  // quasi-Newton method — on forward-mode gradients instead.
  let compiled_objective = if n >= 8 && bounds.iter().all(Option::is_none) {
    compile_numeric(f, &vars).filter(|node| !node.has_fallback())
  } else {
    None
  };
  let mut grad_exprs: Vec<Expr> = Vec::with_capacity(n);
  let mut hess_exprs: Vec<Vec<Expr>> = Vec::new();
  let symbolic_derivatives: Option<()> = (|| {
    if compiled_objective.is_some() {
      return None;
    }
    for var in &vars {
      let deriv =
        crate::functions::calculus_ast::differentiate_expr(f, var).ok()?;
      grad_exprs.push(simplify(deriv));
    }
    // The symbolic Hessian has n² entries, each differentiating an already
    // large gradient; past a handful of variables that cost explodes (a
    // 28-parameter regression objective never finishes), so the Newton step
    // below differences the symbolic gradient numerically instead.
    if n > SYMBOLIC_HESSIAN_MAX_VARS {
      return Some(());
    }
    for i in 0..n {
      let mut row = Vec::new();
      for j in 0..n {
        let h = crate::functions::calculus_ast::differentiate_expr(
          &grad_exprs[i],
          &vars[j],
        )
        .ok()?;
        row.push(simplify(h));
      }
      hess_exprs.push(row);
    }
    Some(())
  })();

  // Evaluate expression at point
  // The objective, gradient and Hessian are evaluated at many points, so
  // each is compiled once into a plain float tree (keyed by address; they
  // are not mutated from here on). Anything the compiler hands to the AST
  // evaluator, and any non-finite result, takes the general path below so
  // messages and exotic values behave as before.
  let compiled: std::cell::RefCell<
    std::collections::HashMap<*const Expr, Option<NumNode>>,
  > = std::cell::RefCell::new(std::collections::HashMap::new());
  let eval_at = |expr: &Expr, point: &[f64]| -> Result<f64, InterpreterError> {
    let fast = {
      let mut cache = compiled.borrow_mut();
      cache
        .entry(std::ptr::from_ref::<Expr>(expr))
        .or_insert_with(|| {
          compile_numeric(expr, &vars).filter(|node| !node.has_fallback())
        })
        .as_ref()
        .map(|node| node.eval(point))
        .filter(|v| v.is_finite())
    };
    if let Some(v) = fast {
      return Ok(v);
    }
    let mut e = expr.clone();
    for (i, var) in vars.iter().enumerate() {
      e = crate::syntax::substitute_variable(&e, var, &Expr::Real(point[i]));
    }
    let evaled = crate::evaluator::evaluate_expr_to_expr(&e)?;
    expr_to_f64(&evaled)
  };

  // Every evaluation of the objective itself (not its derivatives) fires
  // `EvaluationMonitor :> …` with the variables bound to the trial point.
  let eval_obj =
    |expr: &Expr, point: &[f64]| -> Result<f64, InterpreterError> {
      let value = eval_at(expr, point)?;
      fire_step_monitor(evaluation_monitor, &vars, point);
      Ok(value)
    };

  // `f` may be a call to a plain (non-symbolic) function — most commonly
  // one guarded by `_?NumericQ` patterns, the standard idiom for a
  // least-squares objective built from `NDSolve`/`ReplaceAll` that
  // intentionally blocks symbolic differentiation. `differentiate_expr`
  // then falls back to an unevaluated `Derivative[…][…]` form, which
  // doesn't reduce to a real number at any point. Detect that once at the
  // starting point, and fall back to a derivative-free search below
  // instead of failing outright — matching Wolfram, which switches
  // methods automatically whenever the objective isn't symbolically
  // differentiable.
  let symbolic_works = symbolic_derivatives.is_some()
    && (0..n).all(|i| eval_at(&grad_exprs[i], &x).is_ok())
    && hess_exprs
      .iter()
      .all(|row| row.iter().all(|h| eval_at(h, &x).is_ok()));

  // Pre-flight: if f doesn't reduce to a real number at the starting
  // point (e.g. contains an unbound symbol like phi[x]), emit
  // {func_name}::nrnum and return the call unevaluated. Matches
  // wolframscript's behaviour and lets script chains keep flowing
  // instead of aborting with a hard "Cannot evaluate numerically".
  if eval_obj(f, &x).is_err() {
    let mut substituted = f.clone();
    for (i, var) in vars.iter().enumerate() {
      substituted = crate::syntax::substitute_variable(
        &substituted,
        var,
        &Expr::Real(x[i]),
      );
    }
    let value_str = expr_to_output(&substituted);
    let var_str: String = if vars.len() == 1 {
      format!("{{{}}} = {{{}}}", vars[0], x[0])
    } else {
      let names = vars.join(", ");
      let vals = x
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
      format!("{{{names}}} = {{{vals}}}")
    };
    crate::emit_message(&format!(
      "{func_name}::nrnum: The function value {value_str} is not a real number at {var_str}.",
    ));
    return Ok(unevaluated(func_name, args));
  }

  let sign = if maximize { -1.0 } else { 1.0 };
  // `MaxIterations -> n` caps the iterations (default 200).
  let max_iter = match find_option(&args[2..], "MaxIterations") {
    Some(Expr::Integer(n)) if *n > 0 => *n as usize,
    _ => 200,
  };
  let tol = 1e-15;

  // Like wolframscript, a trial point where the objective is not a real
  // number ends the search with FindMinimum::nrnum / FindMaximum::nrnum
  // and the last point reached.
  let head = if maximize {
    "FindMaximum"
  } else {
    "FindMinimum"
  };
  let report_nonreal = |point: &[f64]| {
    let mut e = f.clone();
    for (i, var) in vars.iter().enumerate() {
      e = crate::syntax::substitute_variable(&e, var, &Expr::Real(point[i]));
    }
    let value = crate::evaluator::evaluate_expr_to_expr(&call1("N", e))
      .unwrap_or(Expr::Identifier("Indeterminate".into()));
    // Message arguments read as ToString renders them (6 digits).
    let fmt = |e: &Expr| match crate::evaluator::evaluate_expr_to_expr(&call1(
      "ToString",
      e.clone(),
    )) {
      Ok(Expr::String(ref text)) => text.clone(),
      _ => crate::syntax::format_expr(e, crate::syntax::ExprForm::Output),
    };
    crate::emit_message(&format!(
      "{head}::nrnum: The function value {} is not a real number at {} = {}.",
      fmt(&value),
      fmt(&Expr::List(
        vars.iter().map(|v| Expr::Identifier(v.clone())).collect()
      )),
      fmt(&Expr::List(point.iter().map(|v| Expr::Real(*v)).collect()))
    ));
  };

  if let Some(node) = &compiled_objective {
    let phi = |z: &[f64]| -> f64 {
      fire_step_monitor(evaluation_monitor, &vars, z);
      let v = sign * node.eval(z);
      if v.is_finite() { v } else { f64::INFINITY }
    };
    let grad = |z: &[f64]| -> Vec<f64> {
      let mut g = vec![0.0; n];
      for (k, d) in node.eval_dual(z).1 {
        g[k] = sign * d;
      }
      g
    };
    x = bfgs_minimize(&phi, &grad, x, max_iter, &mut |z: &[f64]| {
      fire_step_monitor(step_monitor, &vars, z);
    });
  } else if !symbolic_works {
    // `f` isn't symbolically differentiable — most commonly a call to a
    // `_?NumericQ`-guarded user function, the standard idiom for a
    // least-squares objective built from `NDSolve`/`ReplaceAll` that
    // intentionally blocks symbolic differentiation (`differentiate_expr`
    // then falls back to an unevaluated `Derivative[…][…]` form, which
    // never reduces to a real number). A finite-difference gradient would
    // amplify whatever numerical noise a black-box objective like that
    // carries (e.g. `NDSolve`'s own fixed-step accuracy) into a wild
    // Newton step; Nelder–Mead's derivative-free simplex search is the
    // same fallback Wolfram itself uses once symbolic differentiation
    // isn't available, and is far more robust to that noise. A failed
    // trial evaluation (`f` erroring at some wild point) is scored as
    // +infinity rather than aborting the search.
    let signed_eval_clamped = |point: &[f64]| -> f64 {
      let mut p = point.to_vec();
      clamp(&mut p);
      eval_obj(f, &p).map_or(f64::INFINITY, |v| v * sign)
    };
    x = nelder_mead_minimize(&signed_eval_clamped, &clamp, &x, max_iter);
  } else if n == 1 {
    // Single variable: damped Newton's method on the derivative
    // Uses line search to ensure we actually decrease/increase the function
    'newton: for _ in 0..max_iter {
      let (Ok(gval), Ok(hval)) =
        (eval_at(&grad_exprs[0], &x), eval_at(&hess_exprs[0][0], &x))
      else {
        report_nonreal(&x);
        break 'newton;
      };

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
      // The threshold scales with the iterate: near a large `x` the step
      // is stuck at rounding noise (a few ulps) and would otherwise never
      // drop below the absolute tolerance, burning every iteration.
      if step.abs() < tol * (1.0 + x[0].abs()) {
        break;
      }

      // Line search along Newton direction to ensure improvement.
      // Use `<=`: if the function is flat to machine precision (e.g. a
      // quadratic scaled by 10^-30 added to 2., which evaluates to 2.0
      // identically in f64), we still want to take the Newton step rather
      // than backtracking to a near-zero alpha and freezing in place.
      let current_f = eval_obj(f, &x)? * sign;
      let mut alpha = 1.0;
      let mut best_x = x[0] - step;
      clamp(std::slice::from_mut(&mut best_x));
      let Ok(mut best_f) = eval_obj(f, &[best_x]).map(|v| v * sign) else {
        report_nonreal(&[best_x]);
        break 'newton;
      };

      // Backtracking: reduce step only if it strictly worsens the value.
      for _ in 0..30 {
        if best_f <= current_f {
          break;
        }
        alpha *= 0.5;
        best_x = x[0] - alpha * step;
        clamp(std::slice::from_mut(&mut best_x));
        let Ok(v) = eval_obj(f, &[best_x]) else {
          report_nonreal(&[best_x]);
          break 'newton;
        };
        best_f = v * sign;
      }
      // The line search could not move off the current point, so every
      // further iteration would repeat this one exactly.
      if best_x == x[0] {
        break;
      }
      x[0] = best_x;
      fire_step_monitor(step_monitor, &vars, &x);
    }
  } else {
    // Multivariable: damped Newton on sign*f using the symbolic Hessian.
    // Falls back to steepest descent when the Newton system is singular
    // or its direction is not a descent direction (e.g. near a saddle).
    // Plain gradient descent only converges linearly and used to stall
    // short of the optimum within the iteration budget.
    'newton: for _ in 0..max_iter {
      // Signed gradient (of sign*f, so "descent" always means improvement)
      let mut grad = vec![0.0; n];
      for i in 0..n {
        let Ok(g) = eval_at(&grad_exprs[i], &x) else {
          report_nonreal(&x);
          break 'newton;
        };
        grad[i] = g * sign;
      }

      let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
      if grad_norm < tol {
        break;
      }

      // Signed Hessian
      let mut hess = vec![vec![0.0; n]; n];
      if hess_exprs.is_empty() {
        // Forward differences of the symbolic gradient, symmetrised.
        for j in 0..n {
          let step = 1e-6 * (1.0 + x[j].abs());
          let mut shifted = x.clone();
          shifted[j] += step;
          for (i, row) in hess.iter_mut().enumerate() {
            let g = eval_at(&grad_exprs[i], &shifted)? * sign;
            row[j] = (g - grad[i]) / step;
          }
        }
        for i in 0..n {
          for j in 0..i {
            let avg = f64::midpoint(hess[i][j], hess[j][i]);
            hess[i][j] = avg;
            hess[j][i] = avg;
          }
        }
      } else {
        for (i, row) in hess.iter_mut().enumerate() {
          for (j, h) in row.iter_mut().enumerate() {
            *h = eval_at(&hess_exprs[i][j], &x)? * sign;
          }
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
      let current_f = eval_obj(f, &x)? * sign;
      let decrease: f64 =
        grad.iter().zip(dir.iter()).map(|(g, d)| g * d).sum::<f64>();
      let mut alpha = 1.0;

      for _ in 0..50 {
        let mut x_new: Vec<f64> = x
          .iter()
          .zip(dir.iter())
          .map(|(xi, di)| xi + alpha * di)
          .collect();
        clamp(&mut x_new);
        let Ok(new_f) = eval_obj(f, &x_new).map(|v| v * sign) else {
          report_nonreal(&x_new);
          break 'newton;
        };
        if new_f <= current_f + c * alpha * decrease || alpha < 1e-15 {
          x = x_new;
          fire_step_monitor(step_monitor, &vars, &x);
          break;
        }
        alpha *= 0.5;
      }
    }
  }

  // Compute final function value
  let min_val = eval_obj(f, &x)?;
  let min_val_expr = Expr::Real(min_val);

  // Build result: {min_val, {x -> x_min, y -> y_min, ...}}
  let rules: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(var, val)| {
      rule_expr(Expr::Identifier(var.clone()), Expr::Real(*val))
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
    eval(call(
      "ReplaceAll",
      vec![body.clone(), rule_expr(param.clone(), value)],
    ))
  };
  // Linear coefficients: a = body | C=0, b = Coefficient[body, C, 1].
  let a = try_eval_to_f64(&subst_param(Expr::Integer(0))?)?;
  let b_expr = eval(call(
    "Coefficient",
    vec![body.clone(), param.clone(), Expr::Integer(1)],
  ))?;
  let b = try_eval_to_f64(&b_expr)?;
  if b.abs() < 1e-12 {
    return None;
  }

  // Finite numeric bounds on `var` taken from the inequality operands.
  let mut x_bounds: Vec<f64> = Vec::new();
  for ineq in ineqs {
    if let Expr::Comparison { operands, .. } = ineq {
      for op in operands {
        if expr_to_string(op) == var_name {
          continue;
        }
        if let Some(v) = try_eval_to_f64(op) {
          x_bounds.push(v);
        }
      }
    }
  }
  if x_bounds.len() < 2 {
    return None; // not bounded on both sides
  }
  let x_lo = x_bounds.iter().copied().fold(f64::INFINITY, f64::min);
  let x_hi = x_bounds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
      result.push(Expr::List(vec![rule_expr(id_expr(var_name), value)].into()));
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

/// Gauss-Jordan elimination of an augmented matrix whose entries are all
/// machine numbers (at least one inexact), with partial pivoting.
///
/// The symbolic elimination takes the first non-zero entry as pivot and tests
/// for zero exactly, so rounding noise in a larger inexact system leaves
/// phantom pivots and loses the solution. Entries below a relative tolerance
/// count as zero here. Returns `None` for any other matrix.
fn eliminate_machine_numbers(
  matrix: &[Vec<Expr>],
  col_order: &mut dyn Iterator<Item = usize>,
) -> Option<(Vec<Vec<Expr>>, Vec<(usize, usize)>)> {
  let mut inexact = false;
  let mut m: Vec<Vec<f64>> = Vec::with_capacity(matrix.len());
  for row in matrix {
    let mut r = Vec::with_capacity(row.len());
    for e in row {
      match e {
        Expr::Real(x) => {
          inexact = true;
          r.push(*x);
        }
        Expr::Integer(i) => r.push(*i as f64),
        // Exact rationals beside the machine numbers (`-1/4`) are numbers
        // too.
        Expr::BigInteger(_) => r.push(try_eval_to_f64(e)?),
        Expr::FunctionCall { name, .. } if name == "Rational" => {
          r.push(try_eval_to_f64(e)?);
        }
        _ => return None,
      }
    }
    m.push(r);
  }
  if !inexact {
    return None;
  }
  let nrows = m.len();
  let scale = m
    .iter()
    .flatten()
    .fold(0.0f64, |acc, v| acc.max(v.abs()))
    .max(f64::MIN_POSITIVE);
  let tol = scale * 1e-12;

  let mut pivot_row = 0;
  let mut pivot_cols: Vec<(usize, usize)> = Vec::new();
  for col in col_order {
    if pivot_row >= nrows {
      break;
    }
    let (best, best_abs) = (pivot_row..nrows)
      .map(|r| (r, m[r][col].abs()))
      .fold((pivot_row, 0.0), |acc, c| if c.1 > acc.1 { c } else { acc });
    if best_abs <= tol {
      continue;
    }
    m.swap(pivot_row, best);
    let pivot = m[pivot_row][col];
    for v in &mut m[pivot_row] {
      *v /= pivot;
    }
    m[pivot_row][col] = 1.0;
    let pivot_values = m[pivot_row].clone();
    for (row, values) in m.iter_mut().enumerate() {
      if row == pivot_row {
        continue;
      }
      let factor = values[col];
      if factor != 0.0 {
        for (v, p) in values.iter_mut().zip(&pivot_values) {
          *v -= factor * p;
        }
      }
      values[col] = 0.0;
    }
    pivot_cols.push((pivot_row, col));
    pivot_row += 1;
  }
  let reduced = m
    .into_iter()
    .map(|row| {
      row
        .into_iter()
        .map(|v| {
          if v.abs() <= tol {
            Expr::Integer(0)
          } else {
            Expr::Real(v)
          }
        })
        .collect()
    })
    .collect();
  Some((reduced, pivot_cols))
}

/// A system in which one equation is `x == value` (either way round) with
/// `value` free of every unknown: the value is substituted into the other
/// equations, they are solved for the remaining unknowns, and `x -> value`
/// joins each of their solutions in the order the unknowns were given.
/// `Ok(None)` when no equation is such a definition or the rest stays
/// unsolved.
fn solve_with_defined_unknown(
  eqs: &[Expr],
  var_names: &[String],
) -> Result<Option<Expr>, InterpreterError> {
  if eqs.len() < 2 || var_names.len() < 2 {
    return Ok(None);
  }
  let free_of_unknowns = |e: &Expr| {
    var_names
      .iter()
      .all(|v| crate::functions::calculus_ast::is_constant_wrt(e, v))
  };
  let definition = eqs.iter().enumerate().find_map(|(i, eq)| {
    let Expr::Comparison {
      operands,
      operators,
    } = eq
    else {
      return None;
    };
    if operands.len() != 2 || operators[0] != ComparisonOp::Equal {
      return None;
    }
    [(&operands[0], &operands[1]), (&operands[1], &operands[0])]
      .into_iter()
      .find_map(|(side, value)| match side {
        Expr::Identifier(v)
          if var_names.contains(v) && free_of_unknowns(value) =>
        {
          Some((i, v.clone(), value.clone()))
        }
        _ => None,
      })
  });
  let Some((index, var, value)) = definition else {
    return Ok(None);
  };
  let rule = rule_expr(id_expr(&var), value.clone());
  let rest: Vec<Expr> = eqs
    .iter()
    .enumerate()
    .filter(|(i, _)| *i != index)
    .map(|(_, eq)| {
      crate::evaluator::evaluate_function_call_ast(
        "ReplaceAll",
        &[eq.clone(), rule.clone()],
      )
    })
    .collect::<Result<_, _>>()?;
  let others: Vec<String> =
    var_names.iter().filter(|v| **v != var).cloned().collect();
  let (eq_arg, var_arg) = if rest.len() == 1 && others.len() == 1 {
    (rest[0].clone(), id_expr(&others[0]))
  } else {
    (
      Expr::List(rest.into()),
      Expr::List(others.iter().map(|v| id_expr(v)).collect()),
    )
  };
  let Expr::List(ref solutions) = solve_ast(&[eq_arg, var_arg])? else {
    return Ok(None);
  };
  let position = |e: &Expr| match e {
    Expr::Rule { pattern, .. } => match pattern.as_ref() {
      Expr::Identifier(n) => var_names.iter().position(|v| v == n),
      _ => None,
    },
    _ => None,
  };
  let mut combined = Vec::with_capacity(solutions.len());
  for solution in solutions {
    let Expr::List(rules) = solution else {
      return Ok(None);
    };
    let mut all: Vec<Expr> = rules.to_vec();
    all.push(rule.clone());
    all.sort_by_key(|r| position(r).unwrap_or(usize::MAX));
    combined.push(Expr::List(all.into()));
  }
  Ok(Some(Expr::List(combined.into())))
}

/// Solve a system of linear equations using symbolic Gaussian elimination.
///
/// Returns `Some(Expr::List([Expr::List([rules...])]))` if the system is linear and consistent,
/// `Some(Expr::List([]))` if inconsistent, or `None` if the system is not linear.
/// Solve a polynomial system by first eliminating every unknown some
/// equation determines linearly with a numeric coefficient
/// (`m1 == m2 + m3`), substituting it into the rest, then solving what is
/// left (the nonlinear core, `60 + 33/100 mr == m1 xk1`) and substituting
/// back. Only used for five or more unknowns and when the elimination
/// leaves a square or overdetermined core, so no choice of free variables
/// is involved. `None` when nothing
/// can be eliminated this way or the system is not polynomial.
fn solve_after_linear_elimination(
  eqs: &[Expr],
  var_names: &[String],
) -> Option<Expr> {
  let eval = |name: &str, args: Vec<Expr>| {
    crate::evaluator::evaluate_function_call_ast(name, &args).ok()
  };
  let var_list = Expr::List(
    var_names
      .iter()
      .map(|v| id_expr(v))
      .collect::<Vec<_>>()
      .into(),
  );
  // Each equation as an expanded polynomial `lhs - rhs`.
  let mut polys: Vec<Expr> = Vec::with_capacity(eqs.len());
  for eq in eqs {
    let Expr::Comparison {
      operands,
      operators,
    } = eq
    else {
      return None;
    };
    if operands.len() != 2 || operators[0] != ComparisonOp::Equal {
      return None;
    }
    let poly = eval(
      "Expand",
      vec![minus2(operands[0].clone(), operands[1].clone())],
    )?;
    let polynomial = eval("PolynomialQ", vec![poly.clone(), var_list.clone()])?;
    if !matches!(polynomial, Expr::Identifier(ref t) if t == "True") {
      return None;
    }
    polys.push(poly);
  }
  // Only for larger systems: the general elimination below is what decides
  // the order and multiplicity of a small system's solutions the way
  // Wolfram lists them, and only grows too slow (tens of seconds for seven
  // unknowns with one bilinear term) once there are many unknowns.
  if var_names.len() < 5 {
    return None;
  }
  let mut remaining: Vec<String> = var_names.to_vec();
  let mut eliminated: Vec<(String, Expr)> = Vec::new();
  'outer: loop {
    for (pi, poly) in polys.iter().enumerate() {
      for v in &remaining {
        let degree = eval("Exponent", vec![poly.clone(), id_expr(v)])?;
        if !matches!(degree, Expr::Integer(1)) {
          continue;
        }
        let coeff = eval(
          "Coefficient",
          vec![poly.clone(), id_expr(v), Expr::Integer(1)],
        )?;
        let numeric = matches!(
          coeff,
          Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_)
        ) || matches!(&coeff, Expr::FunctionCall { name, .. } if name == "Rational");
        if !numeric || matches!(coeff, Expr::Integer(0)) {
          continue;
        }
        // v == v - poly/c
        let value = eval(
          "Expand",
          vec![minus2(id_expr(v), div2(poly.clone(), coeff))],
        )?;
        let v = v.clone();
        let rule = rule_expr(id_expr(&v), value.clone());
        let mut next: Vec<Expr> = Vec::with_capacity(polys.len() - 1);
        for (pj, other) in polys.iter().enumerate() {
          if pj != pi {
            next.push(eval(
              "Expand",
              vec![eval("ReplaceAll", vec![other.clone(), rule.clone()])?],
            )?);
          }
        }
        // Earlier eliminations are written in terms of `v` too.
        for (_, earlier) in &mut eliminated {
          *earlier = eval(
            "Expand",
            vec![eval("ReplaceAll", vec![earlier.clone(), rule.clone()])?],
          )?;
        }
        polys = next;
        remaining.retain(|r| r != &v);
        eliminated.push((v, value));
        continue 'outer;
      }
    }
    break;
  }
  if eliminated.is_empty() || polys.len() < remaining.len() {
    return None;
  }
  // What the eliminated equations leave over: identities, contradictions,
  // or the nonlinear core.
  polys.retain(|p| !matches!(p, Expr::Integer(0)));
  let core_solutions: Vec<Vec<Expr>> = if remaining.is_empty() {
    if polys.iter().any(|p| {
      matches!(p, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
        || matches!(p, Expr::FunctionCall { name, .. } if name == "Rational")
    }) {
      return Some(Expr::List(vec![].into()));
    }
    if !polys.is_empty() {
      return None;
    }
    vec![Vec::new()]
  } else {
    // A linear-only remainder is not this function's case.
    if polys.iter().all(|p| {
      remaining.iter().all(|v| {
        matches!(
          eval("Exponent", vec![p.clone(), id_expr(v)]),
          Some(Expr::Integer(0 | 1))
        )
      })
    }) && polys.len() == eqs.len()
    {
      return None;
    }
    let core_eqs: Vec<Expr> = polys
      .iter()
      .map(|p| Expr::Comparison {
        operands: vec![p.clone(), Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      })
      .collect();
    let solved = solve_ast(&[
      Expr::List(core_eqs.into()),
      Expr::List(
        remaining
          .iter()
          .map(|v| id_expr(v))
          .collect::<Vec<_>>()
          .into(),
      ),
    ])
    .ok()?;
    let Expr::List(ref solutions) = solved else {
      return None;
    };
    let mut out = Vec::with_capacity(solutions.len());
    for sol in solutions {
      let Expr::List(rules) = sol else {
        return None;
      };
      out.push(rules.iter().cloned().collect());
    }
    out
  };
  let mut results: Vec<Expr> = Vec::with_capacity(core_solutions.len());
  for core in core_solutions {
    let core_rules = Expr::List(core.clone().into());
    let mut values: Vec<(String, Expr)> = Vec::new();
    for rule in &core {
      if let Expr::Rule {
        pattern,
        replacement,
      } = rule
        && let Expr::Identifier(name) = pattern.as_ref()
      {
        values.push((name.clone(), replacement.as_ref().clone()));
      }
    }
    for (v, value) in &eliminated {
      let substituted =
        eval("ReplaceAll", vec![value.clone(), core_rules.clone()])?;
      let simplified = eval("Together", vec![substituted])?;
      values.push((v.clone(), simplified));
    }
    let rules: Vec<Expr> = var_names
      .iter()
      .filter_map(|v| {
        values
          .iter()
          .find(|(n, _)| n == v)
          .map(|(_, value)| rule_expr(id_expr(v), value.clone()))
      })
      .collect();
    if rules.len() != var_names.len() {
      return None;
    }
    results.push(Expr::List(rules.into()));
  }
  Some(Expr::List(results.into()))
}

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
    let poly_raw = minus2(lhs.clone(), rhs.clone());
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
        // A factor this function cannot decompose (e.g. the `1/(1+z)` inside
        // `z/(1+z)`) reports itself back as a "coefficient" that still
        // contains `var` — its (-1)-power sentinel, meant to signal
        // "unrecognised structure", is indistinguishable from a genuine
        // negative power once it has combined multiplicatively with a real
        // `var^1` factor elsewhere in the same term (1 + -1 = 0, disguising a
        // rational term as a constant one). Requiring the reported
        // coefficient to actually be free of `var` catches that collision
        // for both the constant (power 0) and linear (power 1) cases.
        if (power == 0 || power == 1) && !is_constant_wrt(&coeff, var) {
          valid = false;
          break;
        }
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

  // Gauss-Jordan elimination, trying the columns in `col_order` for pivots.
  // Each pivot row is normalized so its pivot is 1; returns the reduced
  // matrix and its `(row, col)` pivots.
  let eliminate = |mut matrix: Vec<Vec<Expr>>,
                   col_order: &mut dyn Iterator<Item = usize>|
   -> (Vec<Vec<Expr>>, Vec<(usize, usize)>) {
    if let Some(result) = eliminate_machine_numbers(&matrix, col_order) {
      return result;
    }
    let mut pivot_row = 0;
    let mut pivot_cols: Vec<(usize, usize)> = Vec::new();
    for col in col_order {
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
            matrix[row][j] = eval_entry(minus2(t1, t2));
          }
        }
      }
      pivot_row += 1;
    }
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
    (matrix, pivot_cols)
  };

  let (mut reduced, mut pivot_cols) = eliminate(matrix.clone(), &mut (0..n));
  let underdetermined = pivot_cols.len() < n;
  if underdetermined && !in_nsolve() {
    // Wolfram solves an underdetermined system for its *last* variables,
    // keeping the earlier ones as the free parameters
    // (`Solve[x + y == 2, {x, y}]` gives `y -> 2 - x`), so the pivots are
    // searched for from the last column backwards.
    (reduced, pivot_cols) = eliminate(matrix, &mut (0..n).rev());
  }
  let matrix = reduced;

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

  // vars[pivot_col] = rhs - sum_fc(coeff_fc * vars[fc]) where fc are the
  // free columns. Rules are sorted by variable index to match Wolfram's
  // output order.
  let mut sorted_pivots = pivot_cols.clone();
  sorted_pivots.sort_by_key(|&(_, c)| c);
  let mut rules = Vec::new();
  for &(row, col) in &sorted_pivots {
    let mut rhs_expr = matrix[row][n].clone();
    for &fc in &free_var_cols {
      let coeff = matrix[row][fc].clone();
      if !is_expr_zero(&coeff) {
        let term =
          multiply_exprs(&coeff, &Expr::Identifier(var_names[fc].clone()));
        let neg_term = negate_expr(&eval_entry(term));
        rhs_expr = eval_entry(add_exprs(&rhs_expr, &neg_term));
      }
    }
    let intermediate = eval_entry(rhs_expr);
    // A parametrized solution keeps the elimination's term-by-term form
    // (`3/2 - x/2`, `c/b - (a*x)/b`), as Wolfram's does. A unique one runs
    // through the user-level Simplify so the RHS collapses forms like
    // -1*(1 - E^3)/2 into (-1 + E^3)/2 (matching wolframscript).
    let rhs = if underdetermined
      || matches!(intermediate, Expr::Integer(_) | Expr::Real(_))
    {
      intermediate
    } else {
      crate::functions::polynomial_ast::simplify_ast(std::slice::from_ref(
        &intermediate,
      ))
      .unwrap_or(intermediate)
    };
    rules.push(rule_expr(Expr::Identifier(var_names[col].clone()), rhs));
  }

  Some(Expr::List(vec![Expr::List(rules.into())].into()))
}

/// The first `n` primes, the bases of a Halton sequence.
fn first_primes(n: usize) -> Vec<usize> {
  let mut primes = Vec::with_capacity(n);
  let mut candidate = 2;
  while primes.len() < n {
    if primes.iter().all(|p| candidate % p != 0) {
      primes.push(candidate);
    }
    candidate += 1;
  }
  primes
}

/// The van der Corput radical inverse of `k` in base `b`, in [0, 1).
fn radical_inverse(mut k: usize, b: usize) -> f64 {
  let (mut value, mut scale) = (0.0, 1.0 / b as f64);
  while k > 0 {
    value += (k % b) as f64 * scale;
    k /= b;
    scale /= b as f64;
  }
  value
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
      let n_result = n_ast(std::slice::from_ref(expr))?;
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
/// The value of option `key` among a call's trailing `opt -> value` /
/// `opt :> value` arguments.
fn find_option<'a>(opts: &'a [Expr], key: &str) -> Option<&'a Expr> {
  opts.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(k) if k == key) => {
      Some(replacement.as_ref())
    }
    _ => None,
  })
}

/// The `StepMonitor` a constrained optimization should actually fire.
///
/// The constrained solver is a global sampler with a local refinement, not
/// an iterative method whose steps mean anything to the caller, so Wolfram
/// reports `noopmon` and fires no monitor at all — `Reap` comes back empty.
/// Only an explicitly requested iterative `Method` monitors its steps.
fn monitor_after_noopmon<'a>(
  func_name: &str,
  opts: &'a [Expr],
) -> Option<&'a Expr> {
  let monitor = find_option(opts, "StepMonitor");
  if (monitor.is_some() || find_option(opts, "EvaluationMonitor").is_some())
    && find_option(opts, "Method").is_none()
  {
    crate::emit_message(&format!(
      "{func_name}::noopmon: The optimization was solved by an algorithm that does not provide monitoring information. Choose a specific iterative method if this information is necessary."
    ));
    return None;
  }
  monitor
}

/// Evaluate a `StepMonitor :> expr` option at one solver step: substitute
/// each variable with its current numeric value and evaluate purely for
/// side effect (e.g. `Sow[{x, y}]`), discarding the result.
fn fire_step_monitor(monitor: Option<&Expr>, vars: &[String], x: &[f64]) {
  let Some(monitor) = monitor else {
    return;
  };
  // `Block[{x = …, y = …}, monitor]` binds the variables dynamically, so a
  // monitor that reaches them indirectly (`Sow[vars]` with `vars = {x, y}`)
  // sees the current point just like one that names them directly.
  let bindings: Vec<Expr> = vars
    .iter()
    .zip(x.iter())
    .map(|(var, val)| {
      call("Set", vec![Expr::Identifier(var.clone()), Expr::Real(*val)])
    })
    .collect();
  let e = call("Block", vec![Expr::List(bindings.into()), monitor.clone()]);
  let _ = crate::evaluator::evaluate_expr_to_expr(&e);
}

/// Silences messages for as long as it lives. The numeric optimizer probes
/// the objective at points where it may complain (edges of the search
/// region, speculative symbolic gradients) and wolframscript reports none
/// of that.
struct QuietGuard(crate::WarningSnapshot);

impl QuietGuard {
  fn new() -> Self {
    let saved = crate::snapshot_warnings();
    crate::push_quiet();
    Self(saved)
  }
}

impl Drop for QuietGuard {
  fn drop(&mut self) {
    crate::pop_quiet();
    crate::restore_warnings(std::mem::take(&mut self.0));
  }
}

pub fn nminimize_ast(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize { "NMaximize" } else { "NMinimize" };
  match nminimize_ast_checked(args, maximize) {
    // An objective that never evaluates to a number (an undefined `f[r]`)
    // is NMinimize::nnum and the call stays as it is, not a hard error.
    Err(InterpreterError::EvaluationError(msg))
      if msg.contains("evaluate expression numerically") =>
    {
      let vars = match args.get(1) {
        Some(Expr::List(v)) => v.to_vec(),
        Some(v) => vec![v.clone()],
        None => vec![],
      };
      let objective = match args.first() {
        Some(Expr::List(items)) if !items.is_empty() => items[0].clone(),
        Some(o) => o.clone(),
        None => Expr::Integer(0),
      };
      let probe: Vec<Expr> = vars.iter().map(|_| Expr::Real(0.5)).collect();
      let mut value = objective;
      for (v, p) in vars.iter().zip(&probe) {
        if let Expr::Identifier(name) = v {
          value = crate::syntax::substitute_variable(&value, name, p);
        }
      }
      if maximize {
        value = negate_expr(&value);
      }
      let value =
        crate::evaluator::evaluate_expr_to_expr(&value).unwrap_or(value);
      crate::emit_message(&format!(
        "{func_name}::nnum: The function value {} is not a number at {} = {}.",
        expr_to_string(&value),
        expr_to_string(&Expr::List(vars.clone().into())),
        expr_to_string(&Expr::List(probe.into()))
      ));
      Ok(unevaluated(func_name, args))
    }
    other => other,
  }
}

fn nminimize_ast_checked(
  args: &[Expr],
  maximize: bool,
) -> Result<Expr, InterpreterError> {
  let func_name = if maximize { "NMaximize" } else { "NMinimize" };
  // Trailing `opt -> value` arguments are accepted (and, apart from a
  // monitor's `noopmon` report, ignored) rather than rejected on arity.
  let (positional, opts) = args.split_at(args.len().min(2));
  let step_monitor = monitor_after_noopmon(func_name, opts);

  // The optimizer probes the objective at the edges of the search region
  // (e.g. the excluded endpoint of `x < 1`), which would print Power::infy
  // and similar messages; wolframscript's NMinimize stays silent there.
  let _quiet = QuietGuard::new();

  // An indexed variable like `n[1]` (a FunctionCall, not a plain Identifier)
  // is a valid variable for Minimize/Maximize's symbolic solver; NMinimize
  // should accept the same shape. Rename each such variable to a fresh
  // plain identifier throughout the objective/constraints so the rest of
  // the (identifier-only) numeric pipeline below can stay unchanged, then
  // substitute the original expressions back into the result's rules.
  if positional.len() == 2
    && let Ok((_, var_exprs)) =
      minimize_parse_vars_full(&positional[1], func_name)
    && var_exprs
      .iter()
      .any(|e| matches!(e, Expr::FunctionCall { .. }))
  {
    let fresh_names: Vec<String> =
      (0..var_exprs.len()).map(|i| format!("__ilp_{i}")).collect();
    let mut renamed_objective_arg = positional[0].clone();
    for (orig, fresh) in var_exprs.iter().zip(fresh_names.iter()) {
      if matches!(orig, Expr::FunctionCall { .. }) {
        renamed_objective_arg = substitute_expr(
          &renamed_objective_arg,
          orig,
          &Expr::Identifier(fresh.clone()),
        );
      }
    }
    let renamed_vars_arg =
      Expr::List(fresh_names.iter().cloned().map(Expr::Identifier).collect());
    let renamed_args = [renamed_objective_arg, renamed_vars_arg];

    let result = nminimize_ast_impl(&renamed_args, maximize, step_monitor)?;
    let result = fresh_names.iter().zip(var_exprs.iter()).fold(
      result,
      |acc, (fresh, orig)| {
        substitute_expr(&acc, &Expr::Identifier(fresh.clone()), orig)
      },
    );
    return Ok(result);
  }

  nminimize_ast_impl(positional, maximize, step_monitor)
}

/// Rewrite a linear, single-free-variable inequality/equality like
/// `1 - n3 >= 0` into the canonical `var op value` shape that
/// `extract_bound_from_comparison` understands (it only recognizes a bare
/// variable on one side and a literal on the other). This matters after
/// eliminating equality constraints below: substituting `n1 -> 1 - n3` into
/// `n1 >= 0` produces exactly this un-normalized shape. Constraints that
/// touch more than one of `free_vars`, or aren't linear, are returned
/// unchanged (the fallback that was already in place).
fn normalize_bound_constraint(c: &Expr, free_vars: &[String]) -> Expr {
  let Expr::Comparison {
    operands,
    operators,
  } = c
  else {
    return c.clone();
  };
  if operators.len() != 1 || operands.len() != 2 {
    return c.clone();
  }
  let touched: Vec<String> = free_vars
    .iter()
    .filter(|v| crate::functions::polynomial_ast::contains_var(c, v))
    .cloned()
    .collect();
  let [var] = touched.as_slice() else {
    return c.clone();
  };
  if matches!(&operands[0], Expr::Identifier(n) if n == var)
    || matches!(&operands[1], Expr::Identifier(n) if n == var)
  {
    // Already canonical.
    return c.clone();
  }
  let diff = minus2(operands[0].clone(), operands[1].clone());
  let Some((coeffs, constant)) =
    minimize_extract_linear_expr(&diff, std::slice::from_ref(var))
  else {
    return c.clone();
  };
  let coeff = coeffs[0];
  if coeff == 0.0 {
    return c.clone();
  }
  // `coeff * var + constant op 0` ⇒ `var op' (-constant / coeff)`, flipping
  // the comparator when dividing by a negative coefficient.
  let bound = -constant / coeff;
  let op = if coeff < 0.0 {
    match operators[0] {
      ComparisonOp::Less => ComparisonOp::Greater,
      ComparisonOp::LessEqual => ComparisonOp::GreaterEqual,
      ComparisonOp::Greater => ComparisonOp::Less,
      ComparisonOp::GreaterEqual => ComparisonOp::LessEqual,
      other => other,
    }
  } else {
    operators[0]
  };
  Expr::Comparison {
    operands: vec![Expr::Identifier(var.clone()), Expr::Real(bound)],
    operators: vec![op],
  }
}

fn nminimize_ast_impl(
  args: &[Expr],
  maximize: bool,
  step_monitor: Option<&Expr>,
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

  // Linear objective and constraints: solve the linear (integer) program
  // directly instead of searching numerically.
  if !constraints.is_empty()
    && let Some(result) =
      minimize_linear_program(&objective, &constraints, &vars, maximize, true)
  {
    return Ok(result);
  }

  // Equality constraints that are linear in the optimization variables
  // (e.g. `n[1] + n[2] == 1`) define an equality manifold that the sampler
  // and penalty search below struggle to land on exactly, especially once
  // the true optimum sits at a vertex of the feasible region (as any linear
  // objective's does). Eliminate them symbolically first — solving for as
  // many variables as possible in terms of the rest — so the search below
  // only ever has to explore the remaining free variables (typically with
  // plain box bounds once the equalities are gone), regardless of how
  // nonlinear the objective itself is.
  let equalities: Vec<Expr> = flat_constraints
    .iter()
    .filter(|c| {
      matches!(c, Expr::Comparison { operators, .. } if operators.len() == 1 && operators[0] == ComparisonOp::Equal)
    })
    .map(|c| (**c).clone())
    .collect();
  if !equalities.is_empty()
    && let Some(Expr::List(ref solutions)) =
      solve_linear_symbolic(&equalities, &vars)
  {
    if solutions.is_empty() {
      // The equalities are mutually inconsistent: no point can satisfy them.
      return Ok(nminimize_infeasible_result(&constraints, &vars, maximize));
    }
    if let Some(Expr::List(rules)) = solutions.first() {
      let eliminated: Vec<&String> = rules
        .iter()
        .filter_map(|r| match r {
          Expr::Rule { pattern, .. } => match pattern.as_ref() {
            Expr::Identifier(n) => Some(n),
            _ => None,
          },
          _ => None,
        })
        .collect();
      if !eliminated.is_empty() {
        let substitute_rules = |e: &Expr| -> Expr {
          let mut e = e.clone();
          for r in rules {
            if let Expr::Rule {
              pattern,
              replacement,
            } = r
              && let Expr::Identifier(name) = pattern.as_ref()
            {
              e = crate::syntax::substitute_variable(&e, name, replacement);
            }
          }
          e
        };
        let free_vars: Vec<String> = vars
          .iter()
          .filter(|v| !eliminated.contains(v))
          .cloned()
          .collect();
        let reduced_objective = substitute_rules(&objective);
        let non_equality_constraints: Vec<Expr> = flat_constraints
          .iter()
          .filter(|c| {
            !matches!(c, Expr::Comparison { operators, .. } if operators.len() == 1 && operators[0] == ComparisonOp::Equal)
          })
          .map(|c| substitute_rules(c))
          .map(|c| normalize_bound_constraint(&c, &free_vars))
          .collect();

        // Fully determined (no free variables left): evaluate the unique
        // point directly instead of recursing with an empty variable list.
        if free_vars.is_empty() {
          let feasible = non_equality_constraints.iter().all(|c| {
            matches!(
              crate::evaluator::evaluate_expr_to_expr(c),
              Ok(Expr::Identifier(ref s)) if s == "True"
            )
          });
          if !feasible {
            return Ok(nminimize_infeasible_result(
              &constraints,
              &vars,
              maximize,
            ));
          }
          let val = crate::evaluator::evaluate_expr_to_expr(&call1(
            "N",
            reduced_objective,
          ))?;
          let full_rules: Vec<Expr> = rules
            .iter()
            .filter_map(|r| match r {
              Expr::Rule {
                pattern,
                replacement,
              } => {
                let v = crate::evaluator::evaluate_expr_to_expr(&call1(
                  "N",
                  (**replacement).clone(),
                ))
                .unwrap_or_else(|_| (**replacement).clone());
                Some(Expr::Rule {
                  pattern: pattern.clone(),
                  replacement: Box::new(v),
                })
              }
              _ => None,
            })
            .collect();
          return Ok(Expr::List(
            vec![val, Expr::List(full_rules.into())].into(),
          ));
        }

        let reduced_args_obj = if non_equality_constraints.is_empty() {
          reduced_objective
        } else {
          let mut items = vec![reduced_objective];
          items.extend(non_equality_constraints);
          Expr::List(items.into())
        };
        let reduced_vars_arg =
          Expr::List(free_vars.iter().cloned().map(Expr::Identifier).collect());
        let reduced_args = [reduced_args_obj, reduced_vars_arg];
        let reduced_result =
          nminimize_ast_impl(&reduced_args, maximize, step_monitor)?;

        // Reconstruct the full rule list: the free variables come straight
        // from the reduced result; the eliminated ones are recovered by
        // evaluating their elimination rule at that point.
        if let Expr::List(items) = &reduced_result
          && items.len() == 2
          && let Expr::List(free_rules) = &items[1]
        {
          let mut full_rules = free_rules.clone();
          for r in rules {
            if let Expr::Rule {
              pattern,
              replacement,
            } = r
            {
              let mut val = (**replacement).clone();
              for fr in free_rules {
                if let Expr::Rule {
                  pattern: fp,
                  replacement: fv,
                } = fr
                  && let Expr::Identifier(fname) = fp.as_ref()
                {
                  val = crate::syntax::substitute_variable(&val, fname, fv);
                }
              }
              let val =
                crate::evaluator::evaluate_expr_to_expr(&val).unwrap_or(val);
              full_rules.push(Expr::Rule {
                pattern: pattern.clone(),
                replacement: Box::new(val),
              });
            }
          }
          full_rules.sort_by_key(|r| match r {
            Expr::Rule { pattern, .. } => match pattern.as_ref() {
              Expr::Identifier(n) => {
                vars.iter().position(|v| v == n).unwrap_or(usize::MAX)
              }
              _ => usize::MAX,
            },
            _ => usize::MAX,
          });
          return Ok(Expr::List(
            vec![items[0].clone(), Expr::List(full_rules)].into(),
          ));
        }
        return Ok(reduced_result);
      }
    }
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
      crate::evaluator::evaluate_expr_to_expr(&unevaluated(sym_name, args))
        .ok()
        .filter(|sym| matches!(sym, Expr::List(items) if items.len() == 2))
        .and_then(|sym| {
          crate::evaluator::evaluate_expr_to_expr(&call1("N", sym)).ok()
        });

    // Symbolic first: when it produces an exact, optimal answer we want to
    // keep it rather than overwrite it with float noise from the numeric
    // optimizer (which is only adopted when meaningfully better).
    let candidates: Vec<Expr> =
      [symbolic, numeric].into_iter().flatten().collect();
    return Ok(pick_best_optimum(
      candidates,
      &objective,
      &constraints,
      &vars,
      maximize,
    ));
  }

  // Extract bounds from constraints (e.g. 0 < x < Pi/2)
  let bounds = extract_bounds(&constraints, &vars);

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
    let key = std::ptr::from_ref::<Expr>(expr);
    if !compiled_cache.borrow().contains_key(&key) {
      let compiled = compile_numeric(expr, &vars);
      compiled_cache.borrow_mut().insert(key, compiled);
    }
    if let Some(node) = compiled_cache.borrow().get(&key).unwrap() {
      let v = node.eval(point);
      if v.is_finite() {
        return Ok(v);
      }
      // Pure arithmetic that is non-finite here (log of a negative, a
      // fractional power of a negative base, a division by zero) is not a
      // real number for the AST evaluator either, so skip its much slower
      // second opinion — a wide unbounded search would otherwise spend
      // nearly all its time in it.
      if !node.has_fallback() {
        return Err(InterpreterError::EvaluationError(
          "objective is not a finite real number".into(),
        ));
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
  // Best sample restricted to the tightest (nearest-origin) scale range.
  // wolframscript's NMinimize starts its search in a small default region
  // around the origin, so when a distant sample refines to the same optimum
  // as a near-origin one (e.g. the periodic Sin[x]), it reports the
  // near-origin optimum. Track this candidate separately so we can prefer
  // it on ties after refinement.
  let mut tight_x = vec![0.0; n];
  let mut tight_f = if maximize {
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
          let mid = f64::midpoint(lo, hi);
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

  // Index of the tightest scale (smallest total range).
  let range_sum =
    |sb: &[(f64, f64)]| sb.iter().map(|&(lo, hi)| hi - lo).sum::<f64>();
  let tight_idx = scale_bounds
    .iter()
    .enumerate()
    .min_by(|a, b| range_sum(a.1).total_cmp(&range_sum(b.1)))
    .map_or(0, |(i, _)| i);

  // A full grid has (samples + 1)^n points: it stays at 51 per dimension
  // up to three variables, thins out above that to keep about
  // MAX_GRID_POINTS, and gives way to a Halton sequence once even a
  // two-point grid would be larger (a support-vector QP in 40 unknowns
  // tried to build 51^40 points).
  const MAX_GRID_POINTS: f64 = 200_000.0;
  const HALTON_POINTS: usize = 4000;
  let per_dim =
    if ((samples_per_dim + 1) as f64).powi(n as i32) <= MAX_GRID_POINTS {
      Some(samples_per_dim)
    } else {
      let k = MAX_GRID_POINTS.powf(1.0 / n as f64).floor() as usize;
      (k >= 2).then(|| k - 1)
    };
  for (si, sb) in scale_bounds.iter().enumerate() {
    let mut visit = |pt: &[f64]| {
      update_best(pt, &mut best_x, &mut best_f, &eval_at);
      if si == tight_idx {
        update_best(pt, &mut tight_x, &mut tight_f, &eval_at);
      }
    };
    if let Some(per_dim) = per_dim {
      // Odometer over the grid indices: no point list is materialized.
      let mut index = vec![0usize; n];
      let mut pt = vec![0.0; n];
      'grid: loop {
        for i in 0..n {
          let (lo, hi) = sb[i];
          pt[i] = lo + index[i] as f64 / per_dim as f64 * (hi - lo);
        }
        visit(&pt);
        for i in (0..n).rev() {
          if index[i] < per_dim {
            index[i] += 1;
            continue 'grid;
          }
          index[i] = 0;
        }
        break;
      }
    } else {
      // The centre, then a Halton sequence over the box.
      let centre: Vec<f64> =
        sb.iter().map(|&(lo, hi)| f64::midpoint(lo, hi)).collect();
      visit(&centre);
      let primes = first_primes(n);
      for k in 1..=HALTON_POINTS {
        let pt: Vec<f64> = sb
          .iter()
          .zip(&primes)
          .map(|(&(lo, hi), &p)| lo + radical_inverse(k, p) * (hi - lo))
          .collect();
        visit(&pt);
      }
    }
  }

  // Phase 2: Local refinement using golden section / gradient-free search
  // For each variable, refine using Brent-like narrowing.
  // The descent contracts roughly geometrically (step ≈ 0.1·gradient), so a
  // generous iteration cap is needed to converge optima like Sin[x] at -Pi/2
  // to machine precision (wolframscript prints the objective there as an
  // exact -1.); a stalled line search exits early, keeping this cheap.
  let sign = if maximize { -1.0 } else { 1.0 };
  let max_iter = 1000;
  let tol = 1e-12;

  // Try to compute symbolic gradients for gradient-based refinement
  let grad_exprs: Option<Vec<Expr>> = {
    let mut grads = Vec::new();
    let mut ok = true;
    for var in &vars {
      if let Ok(d) =
        crate::functions::calculus_ast::differentiate_expr(&objective, var)
      {
        let d = simplify(d);
        // Check for unevaluated D
        if contains_unevaluated_d(&d) {
          ok = false;
          break;
        }
        grads.push(d);
      } else {
        ok = false;
        break;
      }
    }
    if ok { Some(grads) } else { None }
  };

  let mut x = best_x.clone();

  // The objective to minimize (negated for NMaximize). A point where it is
  // not a finite number is never an improvement, whichever direction the
  // search goes — the sign is applied before that fallback, not after it.
  let signed_objective = |p: &[f64]| match eval_at(&objective, p) {
    Ok(v) if v.is_finite() => v * sign,
    _ => f64::INFINITY,
  };

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
        let current_f = signed_objective(&x);

        let mut moved = false;
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
            moved = true;
            break;
          }
          alpha *= 0.5;
          if alpha < 1e-20 {
            break;
          }
        }
        if !moved {
          // Line search stalled: no step improves, so further iterations
          // would recompute the same failure.
          break;
        }
        fire_step_monitor(step_monitor, &vars, &x);
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
    let mut prev_f = eval_at(&objective, &x).ok();
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

          let fc = signed_objective(&xc);
          let fd = signed_objective(&xd);

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

        let new_val = f64::midpoint(a, b);
        let mut moved = x.clone();
        moved[i] = new_val;
        if (new_val - x[i]).abs() > tol
          && signed_objective(&moved) <= signed_objective(&x)
        {
          x = moved;
          improved = true;
        }
      }
      if !improved {
        break;
      }
      fire_step_monitor(step_monitor, &vars, &x);
      // A coordinate can keep drifting by noise-level amounts (e.g. from an
      // objective built on an adaptively-integrated CDF) long after the
      // objective value itself has stopped meaningfully improving. Stop the
      // sweep once a full pass over all coordinates no longer moves the
      // objective by more than a relative 1e-10, rather than waiting out
      // `max_iter` sweeps of pure jitter.
      if let (Some(prev), Ok(cur)) = (prev_f, eval_at(&objective, &x)) {
        if (cur - prev).abs() < 1e-10 * (1.0 + prev.abs()) {
          break;
        }
        prev_f = Some(cur);
      }
    }
  }

  // Also refine the best near-origin sample and prefer it when it reaches
  // the same optimum. wolframscript's NMinimize starts from a small default
  // region around the origin, so for objectives with many equally good
  // optima (e.g. Sin[x] over the default ±10^6 box) it reports the one near
  // the origin, not a distant twin that happened to sample marginally better.
  if let Some(ref grads) = grad_exprs
    && tight_f.is_finite()
    && tight_x != best_x
  {
    let (x_near, f_near) = run_gradient_descent(tight_x, grads);
    if let Ok(f_cur) = eval_at(&objective, &x)
      && f_near.is_finite()
    {
      let tie_tol = 1e-8 * (1.0 + f_cur.abs());
      let improves = if maximize {
        f_near > f_cur + tie_tol
      } else {
        f_near < f_cur - tie_tol
      };
      let ties = (f_near - f_cur).abs() <= tie_tol;
      let dist2 = |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>();
      if improves || (ties && dist2(&x_near) < dist2(&x)) {
        x = x_near;
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
    let cur = eval_at(&objective, &x).map_or(f64::INFINITY, |v| sign_p * v);
    let polished = nelder_mead_min(&polish_obj, &x, 0.05, n);
    if polish_obj(&polished) < cur {
      x = polished;
    }
  }

  // Newton polish on the gradient. Value-comparison descent stalls once the
  // objective hits its float plateau (|x - x*| ≈ 1e-8 leaves e.g. Sin at
  // -0.9999999999999997 instead of wolframscript's -1.), but the gradient is
  // still well-resolved there, so a few damped Newton steps per coordinate
  // (finite-difference second derivative) close the remaining gap to machine
  // precision. Guarded: a step is only kept when it shrinks the gradient
  // component and does not worsen the objective.
  if let Some(ref grads) = grad_exprs {
    for _ in 0..5 {
      let mut moved = false;
      for i in 0..n {
        let Ok(gi) = eval_at(&grads[i], &x) else {
          continue;
        };
        if !gi.is_finite() || gi == 0.0 {
          continue;
        }
        let h = 1e-6 * (1.0 + x[i].abs());
        let mut xp = x.clone();
        xp[i] += h;
        let mut xm = x.clone();
        xm[i] -= h;
        let (Ok(gp), Ok(gm)) =
          (eval_at(&grads[i], &xp), eval_at(&grads[i], &xm))
        else {
          continue;
        };
        let d2 = (gp - gm) / (2.0 * h);
        if !d2.is_finite() || d2.abs() < 1e-12 {
          continue;
        }
        let step = gi / d2;
        if !step.is_finite() || step.abs() > 1.0 {
          continue;
        }
        let mut cand = x.clone();
        cand[i] = (cand[i] - step).clamp(bounds[i].0, bounds[i].1);
        let (Ok(g_new), Ok(f_new), Ok(f_cur)) = (
          eval_at(&grads[i], &cand),
          eval_at(&objective, &cand),
          eval_at(&objective, &x),
        ) else {
          continue;
        };
        if g_new.is_finite()
          && f_new.is_finite()
          && g_new.abs() < gi.abs()
          && f_new * sign <= f_cur * sign
        {
          x = cand;
          moved = true;
        }
      }
      if !moved {
        break;
      }
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
    .map(|(var, val)| {
      rule_expr(Expr::Identifier(var.clone()), Expr::Real(*val))
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
  //
  // That cleanup only matters for small, closed-form objectives. On a large
  // one (e.g. an option-pricing formula with nested `Erfc` terms) the exact
  // solver's factoring and cancelling never finishes, so the numeric optimum
  // stands on its own.
  let sym_name = if maximize { "Maximize" } else { "Minimize" };
  // A radical of the unknowns in several variables (a sum of distances,
  // the Fermat point) sends the exact solver through squaring
  // eliminations that run for minutes; the numeric optimum stands.
  fn radical_of(e: &Expr, vars: &[String]) -> bool {
    let fractional = |exp: &Expr| {
      matches!(exp, Expr::FunctionCall { name, .. } if name == "Rational")
        || matches!(exp, Expr::Real(_))
    };
    let depends = |b: &Expr| {
      vars
        .iter()
        .any(|v| crate::functions::polynomial_ast::contains_var(b, v))
    };
    match e {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (fractional(&args[1]) && depends(&args[0]))
          || args.iter().any(|a| radical_of(a, vars))
      }
      Expr::FunctionCall { name, args }
        if name == "Sqrt" && args.len() == 1 =>
      {
        depends(&args[0]) || radical_of(&args[0], vars)
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (fractional(right) && depends(left)) || radical_of(left, vars),
      _ => crate::syntax::expr_children(e)
        .into_iter()
        .any(|c| radical_of(c, vars)),
    }
  }
  let symbolic = if super::simplify::leaf_count(&objective)
    <= SYMBOLIC_REFINEMENT_MAX_LEAVES
    && !(vars.len() >= 2 && radical_of(&objective, &vars))
  {
    crate::evaluator::evaluate_expr_to_expr(&unevaluated(sym_name, args))
      .ok()
      .filter(|sym| matches!(sym, Expr::List(items) if items.len() == 2))
      .and_then(|sym| {
        crate::evaluator::evaluate_expr_to_expr(&call1("N", sym)).ok()
      })
  } else {
    None
  };

  let candidates: Vec<Expr> =
    [symbolic, Some(numeric)].into_iter().flatten().collect();
  Ok(pick_best_optimum(
    candidates,
    &objective,
    &constraints,
    &vars,
    maximize,
  ))
}

/// Largest objective (in leaves) for which `NMinimize`-style searches still
/// consult the exact `Minimize`/`Maximize` solver to clean up float noise.
const SYMBOLIC_REFINEMENT_MAX_LEAVES: usize = 150;

/// A compiled numeric expression tree over the optimization variables.
///
/// Optimizers evaluate the objective and constraints thousands of times per
/// run. Re-cloning and re-evaluating the full `Expr` AST at every point is
/// orders of magnitude too slow to afford a thorough search. Compiling the
/// arithmetic once into this closed enum makes each evaluation a few hundred
/// floating-point ops, so many restarts / iterations stay cheap. Anything the
/// compiler doesn't recognise yields `None`, and the caller falls back to the
/// slow AST path.
pub(crate) enum NumNode {
  Const(f64),
  Var(usize),
  Neg(Box<Self>),
  Add(Vec<Self>),
  Mul(Vec<Self>),
  Sub(Box<Self>, Box<Self>),
  Div(Box<Self>, Box<Self>),
  Pow(Box<Self>, Box<Self>),
  Unary(fn(f64) -> f64, Box<Self>),
  // Exp, Sqrt and Log get their own nodes so the wide re-evaluation can
  // keep them exact outside machine range.
  Exp(Box<Self>),
  Sqrt(Box<Self>),
  Ln(Box<Self>),
  Binary(fn(f64, f64) -> f64, Box<Self>, Box<Self>),
  // `Re`, `Im`, `Abs`, … of a subexpression: while the argument stays
  // real, the function applied to its value; once it leaves the reals
  // (machine arithmetic gives NaN, e.g. `Log` of a negative number), the
  // whole call through the `Fallback` node, in complex arithmetic.
  RealOr(fn(f64) -> f64, Box<Self>, Box<Self>),
  // An `InterpolatingFunction[…]` (its arguments, in Woxi's own layout)
  // applied to compiled arguments: evaluated directly, without copying its
  // data into an expression for every point.
  // A grid function is also kept in machine numbers.
  Interpolation(
    std::rc::Rc<[Expr]>,
    Option<std::rc::Rc<crate::functions::ode_ast::NumericGrid>>,
    Vec<Self>,
  ),
  // A subexpression compile_numeric doesn't otherwise recognise (e.g. a
  // distribution's CDF/PDF), evaluated via the full AST evaluator with the
  // current point substituted in. Isolating just this piece — rather than
  // failing the whole compile and re-cloning/substituting/evaluating the
  // entire objective through the slow path on every call — keeps the rest
  // of a large expression on the fast numeric path.
  Fallback(std::rc::Rc<Expr>, std::rc::Rc<[String]>),
}

impl NumNode {
  /// Whether any subexpression is evaluated through the AST evaluator
  /// rather than compiled arithmetic.
  pub(crate) fn has_fallback(&self) -> bool {
    match self {
      Self::Const(_) | Self::Var(_) => false,
      Self::Neg(a)
      | Self::Unary(_, a)
      | Self::Exp(a)
      | Self::Sqrt(a)
      | Self::Ln(a) => a.has_fallback(),
      Self::Add(xs) | Self::Mul(xs) => xs.iter().any(Self::has_fallback),
      Self::Sub(a, b)
      | Self::Div(a, b)
      | Self::Pow(a, b)
      | Self::Binary(_, a, b) => a.has_fallback() || b.has_fallback(),
      // The fallback only stands in for complex intermediate values.
      Self::RealOr(_, a, _) => a.has_fallback(),
      Self::Interpolation(_, _, xs) => xs.iter().any(Self::has_fallback),
      Self::Fallback(..) => true,
    }
  }

  /// Value at `point`. When plain machine arithmetic overflows or
  /// underflows on the way (e.g. `E^878 * (E^-880 - …)`, which is `inf * 0`),
  /// the tree is re-evaluated with an unbounded exponent, the way Wolfram
  /// leaves machine range instead of producing `Indeterminate`.
  pub(crate) fn eval(&self, point: &[f64]) -> f64 {
    let v = self.eval_plain(point);
    if v.is_finite() {
      v
    } else {
      self.eval_wide(point).to_f64()
    }
  }

  fn eval_plain(&self, point: &[f64]) -> f64 {
    match self {
      Self::Const(c) => *c,
      Self::Var(i) => point[*i],
      Self::Neg(a) => -a.eval_plain(point),
      Self::Add(xs) => xs.iter().map(|x| x.eval_plain(point)).sum(),
      Self::Mul(xs) => xs.iter().map(|x| x.eval_plain(point)).product(),
      Self::Sub(a, b) => a.eval_plain(point) - b.eval_plain(point),
      Self::Div(a, b) => a.eval_plain(point) / b.eval_plain(point),
      Self::Pow(a, b) => {
        let base = a.eval_plain(point);
        let exp = b.eval_plain(point);
        // Integer exponents via powi keep `(-x)^2` real instead of NaN.
        if exp.fract() == 0.0 && exp.abs() < 1e9 {
          base.powi(exp as i32)
        } else {
          base.powf(exp)
        }
      }
      Self::Unary(f, a) => f(a.eval_plain(point)),
      Self::Exp(a) => a.eval_plain(point).exp(),
      Self::Sqrt(a) => a.eval_plain(point).sqrt(),
      Self::Ln(a) => a.eval_plain(point).ln(),
      Self::Binary(f, a, b) => f(a.eval_plain(point), b.eval_plain(point)),
      Self::Interpolation(func, grid, xs) => {
        let values: Vec<f64> = xs.iter().map(|x| x.eval_plain(point)).collect();
        Self::interpolation_value(func, grid.as_deref(), &values)
      }
      Self::RealOr(f, a, whole) => {
        let v = a.eval_plain(point);
        if v.is_nan() {
          whole.eval_fallback(point)
        } else {
          f(v)
        }
      }
      Self::Fallback(..) => self.eval_fallback(point),
    }
  }

  /// The value at `point` with its gradient, as `(variable, derivative)`
  /// pairs sorted by variable (forward-mode differentiation; only the
  /// variables the node mentions appear). A function known only as a
  /// pointer (`Unary`, `Binary`) is differentiated by central differences
  /// of that one function; a `Fallback` node by central differences of the
  /// whole subexpression.
  pub(crate) fn eval_dual(&self, point: &[f64]) -> (f64, Vec<(usize, f64)>) {
    fn axpy(
      a: f64,
      x: &[(usize, f64)],
      b: f64,
      y: &[(usize, f64)],
    ) -> Vec<(usize, f64)> {
      let mut out = Vec::with_capacity(x.len() + y.len());
      let (mut i, mut j) = (0, 0);
      while i < x.len() || j < y.len() {
        if j >= y.len() || (i < x.len() && x[i].0 < y[j].0) {
          out.push((x[i].0, a * x[i].1));
          i += 1;
        } else if i >= x.len() || y[j].0 < x[i].0 {
          out.push((y[j].0, b * y[j].1));
          j += 1;
        } else {
          out.push((x[i].0, a * x[i].1 + b * y[j].1));
          i += 1;
          j += 1;
        }
      }
      out
    }
    let scale = |a: f64, x: Vec<(usize, f64)>| -> Vec<(usize, f64)> {
      x.into_iter().map(|(k, d)| (k, a * d)).collect()
    };
    let numeric_derivative = |f: &dyn Fn(f64) -> f64, a: f64| {
      let h = 1e-6 * a.abs().max(1.0);
      (f(a + h) - f(a - h)) / (2.0 * h)
    };
    match self {
      Self::Const(c) => (*c, Vec::new()),
      Self::Var(i) => (point[*i], vec![(*i, 1.0)]),
      Self::Neg(a) => {
        let (v, g) = a.eval_dual(point);
        (-v, scale(-1.0, g))
      }
      Self::Add(xs) => xs.iter().fold((0.0, Vec::new()), |(v, g), x| {
        let (xv, xg) = x.eval_dual(point);
        (v + xv, axpy(1.0, &g, 1.0, &xg))
      }),
      Self::Mul(xs) => xs.iter().fold((1.0, Vec::new()), |(v, g), x| {
        let (xv, xg) = x.eval_dual(point);
        (v * xv, axpy(xv, &g, v, &xg))
      }),
      Self::Sub(a, b) => {
        let (av, ag) = a.eval_dual(point);
        let (bv, bg) = b.eval_dual(point);
        (av - bv, axpy(1.0, &ag, -1.0, &bg))
      }
      Self::Div(a, b) => {
        let (av, ag) = a.eval_dual(point);
        let (bv, bg) = b.eval_dual(point);
        (av / bv, axpy(1.0 / bv, &ag, -av / (bv * bv), &bg))
      }
      Self::Pow(a, b) => {
        let (av, ag) = a.eval_dual(point);
        let (bv, bg) = b.eval_dual(point);
        let v = Self::Pow(Box::new(Self::Const(av)), Box::new(Self::Const(bv)))
          .eval_plain(point);
        // d(a^b) = b a^(b-1) da + a^b ln(a) db
        let da = if bv == 0.0 {
          0.0
        } else {
          bv * Self::Pow(
            Box::new(Self::Const(av)),
            Box::new(Self::Const(bv - 1.0)),
          )
          .eval_plain(point)
        };
        let db = if bg.is_empty() { 0.0 } else { v * av.ln() };
        (v, axpy(da, &ag, db, &bg))
      }
      Self::Exp(a) => {
        let (av, ag) = a.eval_dual(point);
        let v = av.exp();
        (v, scale(v, ag))
      }
      Self::Sqrt(a) => {
        let (av, ag) = a.eval_dual(point);
        let v = av.sqrt();
        (v, scale(0.5 / v, ag))
      }
      Self::Ln(a) => {
        let (av, ag) = a.eval_dual(point);
        (av.ln(), scale(1.0 / av, ag))
      }
      Self::Unary(f, a) => {
        let (av, ag) = a.eval_dual(point);
        (f(av), scale(numeric_derivative(f, av), ag))
      }
      Self::Binary(f, a, b) => {
        let (av, ag) = a.eval_dual(point);
        let (bv, bg) = b.eval_dual(point);
        let da = numeric_derivative(&|t| f(t, bv), av);
        let db = numeric_derivative(&|t| f(av, t), bv);
        (f(av, bv), axpy(da, &ag, db, &bg))
      }
      Self::RealOr(f, a, whole) => {
        let (av, ag) = a.eval_dual(point);
        if av.is_nan() {
          whole.eval_dual(point)
        } else {
          (f(av), scale(numeric_derivative(f, av), ag))
        }
      }
      // Each argument's partial derivative by central differences of the
      // interpolation, chained with that argument's gradient.
      Self::Interpolation(func, grid, xs) => {
        let grid = grid.as_deref();
        let duals: Vec<(f64, Vec<(usize, f64)>)> =
          xs.iter().map(|x| x.eval_dual(point)).collect();
        let values: Vec<f64> = duals.iter().map(|(v, _)| *v).collect();
        let v = Self::interpolation_value(func, grid, &values);
        let mut grad: Vec<(usize, f64)> = Vec::new();
        for (k, (xk, gk)) in duals.iter().enumerate() {
          if gk.is_empty() {
            continue;
          }
          let h = 1e-6 * xk.abs().max(1.0);
          let mut shifted = values.clone();
          shifted[k] = xk + h;
          let up = Self::interpolation_value(func, grid, &shifted);
          shifted[k] = xk - h;
          let down = Self::interpolation_value(func, grid, &shifted);
          grad = axpy(1.0, &grad, (up - down) / (2.0 * h), gk);
        }
        (v, grad)
      }
      Self::Fallback(..) => {
        let v = self.eval_fallback(point);
        let grad = (0..point.len())
          .filter_map(|k| {
            let h = 1e-7 * point[k].abs().max(1.0);
            let mut p = point.to_vec();
            p[k] += h;
            let fp = self.eval_fallback(&p);
            p[k] -= 2.0 * h;
            let fm = self.eval_fallback(&p);
            let d = (fp - fm) / (2.0 * h);
            (d != 0.0).then_some((k, d))
          })
          .collect();
        (v, grad)
      }
    }
  }

  /// `eval_plain` with every intermediate kept as a `WideFloat`.
  fn eval_wide(&self, point: &[f64]) -> WideFloat {
    match self {
      Self::Const(c) => WideFloat::new(*c),
      Self::Var(i) => WideFloat::new(point[*i]),
      Self::Neg(a) => a.eval_wide(point).neg(),
      Self::Add(xs) => xs
        .iter()
        .fold(WideFloat::new(0.0), |acc, x| acc.add(x.eval_wide(point))),
      Self::Mul(xs) => xs
        .iter()
        .fold(WideFloat::new(1.0), |acc, x| acc.mul(x.eval_wide(point))),
      Self::Sub(a, b) => a.eval_wide(point).add(b.eval_wide(point).neg()),
      Self::Div(a, b) => a.eval_wide(point).div(b.eval_wide(point)),
      Self::Pow(a, b) => a.eval_wide(point).pow(b.eval_wide(point).to_f64()),
      // Any other function sees its argument rounded into machine range.
      Self::Unary(f, a) => WideFloat::new(f(a.eval_wide(point).to_f64())),
      Self::Exp(a) => WideFloat::exp(a.eval_wide(point).to_f64()),
      Self::Sqrt(a) => a.eval_wide(point).sqrt(),
      Self::Ln(a) => WideFloat::new(a.eval_wide(point).ln()),
      Self::Binary(f, a, b) => WideFloat::new(f(
        a.eval_wide(point).to_f64(),
        b.eval_wide(point).to_f64(),
      )),
      Self::Interpolation(..) => WideFloat::new(self.eval_plain(point)),
      Self::RealOr(f, a, whole) => {
        let v = a.eval_wide(point).to_f64();
        WideFloat::new(if v.is_nan() {
          whole.eval_fallback(point)
        } else {
          f(v)
        })
      }
      Self::Fallback(..) => WideFloat::new(self.eval_fallback(point)),
    }
  }

  /// The value of `InterpolatingFunction[func…][values…]`.
  fn interpolation_value(
    func: &[Expr],
    grid: Option<&crate::functions::ode_ast::NumericGrid>,
    values: &[f64],
  ) -> f64 {
    let args = || values.iter().map(|v| Expr::Real(*v)).collect::<Vec<_>>();
    if let Some(grid) = grid {
      if grid.outside(values) {
        crate::functions::ode_ast::emit_dmval(&args());
      }
      return grid.eval_real(values);
    }
    let args = args();
    crate::functions::ode_ast::evaluate_interpolating_function(func, &args)
      .ok()
      .and_then(|v| expr_to_f64(&v).ok())
      .unwrap_or(f64::NAN)
  }

  fn eval_fallback(&self, point: &[f64]) -> f64 {
    match self {
      Self::Fallback(expr, vars) => {
        let mut e = (**expr).clone();
        for (i, var) in vars.iter().enumerate() {
          e =
            crate::syntax::substitute_variable(&e, var, &Expr::Real(point[i]));
        }
        crate::evaluator::evaluate_expr_to_expr(&e)
          .ok()
          .and_then(|v| expr_to_f64(&v).ok())
          .unwrap_or(f64::NAN)
      }
      _ => f64::NAN,
    }
  }
}

/// A float with an unbounded binary exponent: `mant * 2^exp`, `mant` in
/// `[0.5, 1)` in magnitude (or zero / non-finite, with `exp == 0`).
///
/// Like Wolfram's machine arithmetic, only overflow leaves machine range:
/// a value below the smallest normal machine number flushes to zero
/// (`Exp[800.] (Exp[-800.] - Exp[-801.])` is `0.`).
#[derive(Clone, Copy, Debug)]
struct WideFloat {
  mant: f64,
  exp: i64,
}

impl WideFloat {
  fn new(x: f64) -> Self {
    Self::normalized(x, 0)
  }

  fn normalized(mant: f64, exp: i64) -> Self {
    if mant == 0.0 || !mant.is_finite() {
      return Self { mant, exp: 0 };
    }
    let (m, e) = libm::frexp(mant);
    let exp = exp.saturating_add(i64::from(e));
    if exp < f64::MIN_EXP as i64 {
      return Self {
        mant: 0.0_f64.copysign(m),
        exp: 0,
      };
    }
    Self { mant: m, exp }
  }

  fn to_f64(self) -> f64 {
    if self.exp > 2000 {
      self.mant * f64::INFINITY
    } else if self.exp < -2000 {
      self.mant * 0.0
    } else {
      libm::ldexp(self.mant, self.exp as i32)
    }
  }

  fn neg(self) -> Self {
    Self {
      mant: -self.mant,
      exp: self.exp,
    }
  }

  fn mul(self, other: Self) -> Self {
    Self::normalized(self.mant * other.mant, self.exp.saturating_add(other.exp))
  }

  fn div(self, other: Self) -> Self {
    Self::normalized(self.mant / other.mant, self.exp.saturating_sub(other.exp))
  }

  fn add(self, other: Self) -> Self {
    if self.mant == 0.0 || !other.mant.is_finite() {
      return other;
    }
    if other.mant == 0.0 || !self.mant.is_finite() {
      return self;
    }
    let (big, small) = if self.exp >= other.exp {
      (self, other)
    } else {
      (other, self)
    };
    let shift = small.exp - big.exp;
    if shift < -1100 {
      return big;
    }
    Self::normalized(big.mant + libm::ldexp(small.mant, shift as i32), big.exp)
  }

  /// `e^x` for a machine `x`, whatever the size of the result.
  fn exp(x: f64) -> Self {
    if !x.is_finite() || x.abs() > 1e15 {
      return Self::new(x.exp());
    }
    let k = (x / std::f64::consts::LN_2).floor();
    let r = x - k * std::f64::consts::LN_2;
    Self::normalized(r.exp(), k as i64)
  }

  /// Natural log, which always fits a machine real.
  fn ln(self) -> f64 {
    if self.mant <= 0.0 || !self.mant.is_finite() {
      return self.to_f64().ln();
    }
    self.mant.ln() + self.exp as f64 * std::f64::consts::LN_2
  }

  fn sqrt(self) -> Self {
    if self.mant < 0.0 || !self.mant.is_finite() {
      return Self::new(self.to_f64().sqrt());
    }
    // Make the exponent even so it halves exactly.
    let (m, e) = if self.exp % 2 == 0 {
      (self.mant, self.exp)
    } else {
      (self.mant * 2.0, self.exp - 1)
    };
    Self::normalized(m.sqrt(), e / 2)
  }

  fn pow(self, p: f64) -> Self {
    if p.fract() == 0.0 && p.abs() <= 64.0 {
      // Exact repeated squaring keeps full precision and the sign.
      let mut result = Self::new(1.0);
      let mut base = self;
      let mut n = p.abs() as u32;
      while n > 0 {
        if n & 1 == 1 {
          result = result.mul(base);
        }
        base = base.mul(base);
        n >>= 1;
      }
      return if p < 0.0 {
        Self::new(1.0).div(result)
      } else {
        result
      };
    }
    if self.mant <= 0.0 || !self.mant.is_finite() || !p.is_finite() {
      return Self::new(self.to_f64().powf(p));
    }
    Self::exp(p * self.ln())
  }
}

/// Compile an `Expr` over `vars` into a fast numeric closure tree, or `None`
/// if it contains anything not handled here (forcing the slow AST fallback).
pub(crate) fn compile_numeric(expr: &Expr, vars: &[String]) -> Option<NumNode> {
  let c = |e: &Expr| compile_numeric(e, vars);
  match expr {
    Expr::Integer(n) => Some(NumNode::Const(*n as f64)),
    Expr::BigInteger(n) => Some(NumNode::Const(n.to_string().parse().ok()?)),
    Expr::Real(r) => Some(NumNode::Const(*r)),
    // An arbitrary-precision number (`N[x, $MachinePrecision]` makes them)
    // enters the machine arithmetic rounded.
    Expr::BigFloat(..) => Some(NumNode::Const(expr_to_f64(expr).ok()?)),
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
      use BinaryOperator as B;
      let l = Box::new(c(left)?);
      let r = Box::new(c(right)?);
      match op {
        B::Plus => Some(NumNode::Add(vec![*l, *r])),
        B::Minus => Some(NumNode::Sub(l, r)),
        B::Times => Some(NumNode::Mul(vec![*l, *r])),
        B::Divide => Some(NumNode::Div(l, r)),
        B::Power => Some(NumNode::Pow(l, r)),
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
      let real_or =
        |f: fn(f64) -> f64, args: &crate::ExprList| -> Option<NumNode> {
          Some(NumNode::RealOr(
            f,
            Box::new(compile_numeric(&args[0], vars)?),
            Box::new(NumNode::Fallback(
              std::rc::Rc::new(expr.clone()),
              std::rc::Rc::from(vars.to_vec()),
            )),
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
        "Sqrt" if args.len() == 1 => {
          Some(NumNode::Sqrt(Box::new(c(&args[0])?)))
        }
        "Exp" if args.len() == 1 => Some(NumNode::Exp(Box::new(c(&args[0])?))),
        "Sin" => unary(f64::sin, args),
        "Cos" => unary(f64::cos, args),
        "Tan" => unary(f64::tan, args),
        "Cot" => unary(|x| 1.0 / x.tan(), args),
        "Sec" => unary(|x| 1.0 / x.cos(), args),
        "Csc" => unary(|x| 1.0 / x.sin(), args),
        "ArcSin" => unary(f64::asin, args),
        "ArcCos" => unary(f64::acos, args),
        "ArcTan" if args.len() == 2 => Some(NumNode::Binary(
          f64::atan2,
          Box::new(c(&args[0])?),
          Box::new(c(&args[1])?),
        )),
        "ArcTan" => unary(f64::atan, args),
        "Sinh" => unary(f64::sinh, args),
        "Cosh" => unary(f64::cosh, args),
        "Tanh" => unary(f64::tanh, args),
        "Re" | "Conjugate" if args.len() == 1 => real_or(|x| x, args),
        "Im" if args.len() == 1 => real_or(|_| 0.0, args),
        "Abs" if args.len() == 1 => real_or(f64::abs, args),
        "Sign" => unary(f64::signum, args),
        "Erf" => {
          unary(crate::functions::math_ast::numeric_utils::erf_f64, args)
        }
        "Erfc" => unary(
          |x| {
            if x >= 0.0 {
              crate::functions::math_ast::numeric_utils::erfc_f64(x)
            } else {
              2.0 - crate::functions::math_ast::numeric_utils::erfc_f64(-x)
            }
          },
          args,
        ),
        "Log" if args.len() == 2 => Some(NumNode::Binary(
          |b, x| x.ln() / b.ln(),
          Box::new(c(&args[0])?),
          Box::new(c(&args[1])?),
        )),
        "Log" if args.len() == 1 => Some(NumNode::Ln(Box::new(c(&args[0])?))),
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
        // An unrecognised function (e.g. CDF of a distribution) still
        // compiles: it's isolated in a `Fallback` node rather than failing
        // the whole expression's compilation.
        _ => Some(NumNode::Fallback(
          std::rc::Rc::new(expr.clone()),
          std::rc::Rc::from(vars.to_vec()),
        )),
      }
    }
    Expr::CurriedCall { func, args }
      if let Expr::FunctionCall {
        name,
        args: func_args,
      } = func.as_ref()
        && name == "InterpolatingFunction" =>
    {
      // A kernel-layout function is read into Woxi's own layout once here
      // rather than at every evaluation.
      let native =
        crate::functions::ode_ast::kernel_grid_interpolating_function(
          func_args,
        )
        .or_else(|| {
          crate::functions::ode_ast::kernel_interpolating_function_to_native(
            func_args,
          )
        })
        .unwrap_or_else(|| func_args.to_vec());
      let grid = crate::functions::ode_ast::NumericGrid::from_function(&native)
        .map(std::rc::Rc::new);
      Some(NumNode::Interpolation(
        std::rc::Rc::from(native),
        grid,
        args.iter().map(&c).collect::<Option<_>>()?,
      ))
    }
    // Anything else unrecognised (comparisons, curried calls, …) also falls
    // back in isolation rather than failing the whole compile.
    _ => Some(NumNode::Fallback(
      std::rc::Rc::new(expr.clone()),
      std::rc::Rc::from(vars.to_vec()),
    )),
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
    "GoldenRatio" => f64::midpoint(1.0, 5.0_f64.sqrt()),
    "EulerGamma" => std::f64::consts::EULER_GAMMA,
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
        constraint_strs.push(expr_to_output(&Expr::Comparison {
          operands: vec![operands[i].clone(), operands[i + 1].clone()],
          operators: vec![operators[i]],
        }));
      }
    } else {
      constraint_strs.push(expr_to_output(term));
    }
  }
  crate::emit_message(&format!(
    "{func_name}::nsol: There are no points that satisfy the constraints {{{}}}.",
    constraint_strs.join(", ")
  ));

  let inf = if maximize {
    neg1(id_expr("Infinity"))
  } else {
    id_expr("Infinity")
  };
  let rules: Vec<Expr> = vars
    .iter()
    .map(|var| {
      rule_expr(Expr::Identifier(var.clone()), id_expr("Indeterminate"))
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
    id_expr("Infinity")
  } else {
    neg1(id_expr("Infinity"))
  };
  let rules: Vec<Expr> = vars
    .iter()
    .map(|var| {
      rule_expr(Expr::Identifier(var.clone()), id_expr("Indeterminate"))
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
  use ComparisonOp as C;
  let mut total = 0.0;
  for c in comparisons {
    let l = eval_expr_at_point(&c.left, vars, point);
    let r = eval_expr_at_point(&c.right, vars, point);
    if !l.is_finite() || !r.is_finite() {
      return f64::INFINITY;
    }
    total += match c.op {
      C::Less | C::LessEqual => (l - r).max(0.0),
      C::Greater | C::GreaterEqual => (r - l).max(0.0),
      C::Equal => (l - r).abs(),
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
) -> Expr {
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
      for r in rules {
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
      if let Some(v) = found {
        point[vi] = v;
      } else {
        ok = false;
        break;
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
    Some((c, _, true)) => c,
    _ => nminimize_infeasible_result(constraints, vars, maximize),
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
  let bounds = extract_bounds(constraints, vars);

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
    use ComparisonOp as C;
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
        C::Less | C::LessEqual => (l - r).max(0.0),
        C::Greater | C::GreaterEqual => (r - l).max(0.0),
        C::Equal => (l - r).abs(),
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
  // The full grid has per_dim^n points; past 50 000 (more than seven
  // variables) the starts are a Halton sequence over the region instead.
  const MAX_STARTS: f64 = 50_000.0;
  const HALTON_STARTS: usize = 2000;
  let mut starts: Vec<Vec<f64>> = vec![vec![]];
  if (per_dim as f64).powi(n as i32) <= MAX_STARTS {
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
  } else {
    let primes = first_primes(n);
    starts = std::iter::once(
      start_region
        .iter()
        .map(|&(lo, hi)| f64::midpoint(lo, hi))
        .collect(),
    )
    .chain((1..=HALTON_STARTS).map(|k| {
      start_region
        .iter()
        .zip(&primes)
        .map(|(&(lo, hi), &p)| lo + radical_inverse(k, p) * (hi - lo))
        .collect()
    }))
    .collect();
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
    .map(|(var, val)| {
      rule_expr(Expr::Identifier(var.clone()), Expr::Real(*val))
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
) -> std::vec::Vec<(f64, f64)> {
  let mut bounds: Vec<(f64, f64)> = vars.iter().map(|_| (-1e6, 1e6)).collect();

  for constraint in constraints {
    // Flatten And expressions
    let mut flat = Vec::new();
    flatten_and_constraints_ref(constraint, &mut flat);
    for c in flat {
      extract_bound_from_comparison(c, vars, &mut bounds);
    }
  }

  bounds
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
      for a in args {
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
) {
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
}

/// Try to evaluate an expression to f64.
fn eval_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(r) => Ok(*r),
    _ => {
      // Try evaluating via N[]
      let n_expr = call1("N", expr.clone());
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

  // A Boolean system — over `Booleans`, or a formula of plain connectives
  // in every variable — has its instances read off its truth table. (A
  // variable the formula does not mention is complex then, not Boolean.)
  let var_exprs: Vec<Expr> = match vars {
    Expr::List(items) => items.to_vec(),
    single => vec![single.clone()],
  };
  // Equations between truth values only count over `Booleans`.
  fn has_comparison(e: &Expr) -> bool {
    matches!(e, Expr::Comparison { .. })
      || crate::syntax::expr_children(e)
        .into_iter()
        .any(has_comparison)
  }
  if domain.as_deref() == Some("Booleans")
    || (domain.is_none()
      && !var_exprs.is_empty()
      && var_names.len() == var_exprs.len()
      && var_names.iter().all(|v| !is_constant_wrt(cond, v))
      && !has_comparison(cond)
      && crate::functions::boolean_ast::is_connective_formula(cond, &var_exprs))
  {
    // A list of conditions must hold together.
    let conjunction;
    let cond = match cond {
      Expr::List(items) => {
        conjunction = call("And", items.to_vec());
        &conjunction
      }
      other => other,
    };
    let rules = |assignment: Vec<bool>| {
      Expr::List(
        var_exprs
          .iter()
          .zip(assignment)
          .map(|(v, b)| rule_expr(v.clone(), bool_expr(b)))
          .collect(),
      )
    };
    match crate::functions::boolean_ast::satisfying_assignments(
      cond, &var_exprs, n,
    )? {
      Some(Ok(instances)) => {
        return Ok(Expr::List(instances.into_iter().map(rules).collect()));
      }
      Some(Err(assignment)) => {
        crate::emit_message(&format!(
          "FindInstance::boolv: The system is not Boolean valued at {}.",
          crate::syntax::format_expr(
            &rules(assignment),
            crate::syntax::ExprForm::Output
          )
        ));
        return Ok(call("FindInstance", args.to_vec()));
      }
      None => {}
    }
  }

  // Linear inequalities (and equations) over the reals.
  if matches!(domain.as_deref(), None | Some("Reals"))
    && var_names.len() == var_exprs.len()
    && let Some(instances) = find_instance_linear(cond, &var_names, n)?
  {
    return Ok(Expr::List(instances.into()));
  }

  // Whether the solution set below is provably complete: Solve returned the
  // full (finite) solution set and the concreteness filter dropped nothing.
  // Only then may an empty result become `{}` ("no instance exists");
  // otherwise an exhausted search returns the call unevaluated, matching
  // wolframscript, which never claims emptiness it cannot prove.
  let mut complete = false;

  // Try Solve first (suppress warnings from Solve)
  let mut solutions: Vec<Expr> = {
    // Messages Solve raises about a condition it cannot handle (an
    // inequality, say) belong to its internal attempt, not to FindInstance.
    let saved = crate::snapshot_warnings();
    crate::push_quiet();
    let solve_result = solve_ast(&[cond.clone(), vars.clone()]);
    crate::pop_quiet();
    crate::restore_warnings(saved);
    match &solve_result {
      Ok(Expr::List(sols)) if sols.is_empty() => {
        // Solve proved the solution set empty.
        complete = true;
        Vec::new()
      }
      Ok(Expr::List(sols)) => {
        // Filter out parametric solutions (solutions with free variables)
        // FindInstance needs concrete values, not expressions in terms of
        // other variables.
        let concrete: Vec<Expr> = sols
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
          .collect();
        complete = concrete.len() == sols.len();
        concrete
      }
      _ => Vec::new(),
    }
  };

  // If Solve failed or returned no solutions, try numerical search
  if solutions.is_empty() && !var_names.is_empty() && !complete {
    solutions = find_instance_numerical(cond, &var_names, n);
  }

  if solutions.is_empty() {
    if complete {
      return Ok(Expr::List(vec![].into()));
    }
    // Search exhausted without proving emptiness: leave unevaluated.
    return Ok(call("FindInstance", args.to_vec()));
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
    if complete {
      return Ok(Expr::List(vec![].into()));
    }
    // A partial (e.g. numerically found) solution set that the domain
    // filter emptied proves nothing: leave unevaluated.
    return Ok(call("FindInstance", args.to_vec()));
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

/// The atoms of a conjunction of conditions: the members of lists and
/// `And`s, with a comparison chain `a < x <= b` split into its links.
fn condition_atoms(
  cond: &Expr,
  out: &mut Vec<(Expr, ComparisonOp, Expr)>,
) -> bool {
  match cond {
    Expr::List(items) => items.iter().all(|c| condition_atoms(c, out)),
    Expr::FunctionCall { name, args } if name == "And" => {
      args.iter().all(|c| condition_atoms(c, out))
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::And,
      left,
      right,
    } => condition_atoms(left, out) && condition_atoms(right, out),
    Expr::Identifier(s) if s == "True" => true,
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() + 1 == operands.len() => {
      for (i, op) in operators.iter().enumerate() {
        out.push((operands[i].clone(), *op, operands[i + 1].clone()));
      }
      true
    }
    _ => false,
  }
}

/// `FindInstance` for a conjunction of linear equations and inequalities
/// over the reals: an exact linear program finds a point, maximizing the
/// slack `t` of the strict inequalities (`a.x + c > 0` becomes
/// `a.x + c >= t`; capped at 1 where it is unbounded) so that a positive
/// optimum proves them satisfiable and a zero optimum, like an infeasible
/// program, proves there is no instance. The largest slack centres a
/// bounded interval: `3 < x < 10` gives `x -> 13/2`. Further instances are
/// the extreme points of the same system with the slack fixed at half its
/// optimum, which keeps them strictly inside, and then midpoints of those.
/// `None` when the condition is not of that shape.
fn find_instance_linear(
  cond: &Expr,
  var_names: &[String],
  n: usize,
) -> Result<Option<Vec<Expr>>, InterpreterError> {
  use crate::evaluator::evaluate_expr_to_expr as eval;
  let mut atoms = Vec::new();
  if var_names.is_empty() || !condition_atoms(cond, &mut atoms) {
    return Ok(None);
  }
  let vars: Vec<Expr> = var_names
    .iter()
    .map(|v| Expr::Identifier(v.clone()))
    .collect();
  let zero_rules = Expr::List(
    vars
      .iter()
      .map(|v| rule_expr(v.clone(), Expr::Integer(0)))
      .collect(),
  );
  let is_real_number = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };

  // Each atom as `a.x + c  rel  0` with rel `>`, `>=` or `==`.
  let mut rows: Vec<(Vec<Expr>, Expr, ComparisonOp)> = Vec::new();
  let mut inexact = false;
  let mut has_inequality = false;
  for (lhs, op, rhs) in &atoms {
    let (difference, op) = match op {
      ComparisonOp::Equal => (minus2(lhs.clone(), rhs.clone()), *op),
      ComparisonOp::Greater | ComparisonOp::GreaterEqual => {
        (minus2(lhs.clone(), rhs.clone()), *op)
      }
      ComparisonOp::Less => {
        (minus2(rhs.clone(), lhs.clone()), ComparisonOp::Greater)
      }
      ComparisonOp::LessEqual => {
        (minus2(rhs.clone(), lhs.clone()), ComparisonOp::GreaterEqual)
      }
      _ => return Ok(None),
    };
    has_inequality |= op != ComparisonOp::Equal;
    let difference = eval(&call("Expand", vec![difference]))?;
    let mut coefficients = Vec::with_capacity(vars.len());
    for v in &vars {
      let a = eval(&call("Coefficient", vec![difference.clone(), v.clone()]))?;
      if !is_real_number(&a) {
        return Ok(None);
      }
      coefficients.push(a);
    }
    let constant = eval(&call(
      "ReplaceAll",
      vec![difference.clone(), zero_rules.clone()],
    ))?;
    if !is_real_number(&constant) {
      return Ok(None);
    }
    // Linear: nothing is left once the linear part is taken off.
    let mut linear = constant.clone();
    for (a, v) in coefficients.iter().zip(&vars) {
      linear = plus2(linear, times2(a.clone(), v.clone()));
    }
    let rest = eval(&call("Expand", vec![minus2(difference.clone(), linear)]))?;
    if !matches!(rest, Expr::Integer(0))
      && !matches!(rest, Expr::Real(r) if r == 0.0)
    {
      return Ok(None);
    }
    inexact |= coefficients
      .iter()
      .chain(std::iter::once(&constant))
      .any(|e| matches!(e, Expr::Real(_)));
    rows.push((coefficients, constant, op));
  }
  if !has_inequality {
    return Ok(None);
  }
  let strict = rows.iter().any(|(_, _, op)| *op == ComparisonOp::Greater);

  // `LinearProgramming[c, m, b, bounds]` over the variables and the slack
  // `t` every strict inequality must clear (`a.x + c >= t`); the solution
  // values, or None when infeasible or unbounded. A non-strict inequality
  // may hold with equality, and wolframscript, too, likes to pick such a
  // boundary point: `x >= 3` gives `x -> 3`.
  let k = vars.len();
  let solve = |objective: &[Expr],
               slack: &Expr|
   -> Result<Option<Vec<Expr>>, InterpreterError> {
    let mut m = Vec::with_capacity(rows.len());
    let mut b = Vec::with_capacity(rows.len());
    for (coefficients, constant, op) in &rows {
      let mut row = coefficients.clone();
      row.push(Expr::Integer(if *op == ComparisonOp::Greater {
        -1
      } else {
        0
      }));
      m.push(Expr::List(row.into()));
      b.push(Expr::List(
        vec![
          eval(&negate_expr(constant))?,
          Expr::Integer(i128::from(*op != ComparisonOp::Equal)),
        ]
        .into(),
      ));
    }
    let infinity = Expr::Identifier("Infinity".to_string());
    let mut bounds: Vec<Expr> = (0..k)
      .map(|_| {
        Expr::List(vec![negate_expr(&infinity), infinity.clone()].into())
      })
      .collect();
    bounds.push(slack.clone());
    let program = call(
      "LinearProgramming",
      vec![
        Expr::List(objective.to_vec().into()),
        Expr::List(m.into()),
        Expr::List(b.into()),
        Expr::List(bounds.into()),
      ],
    );
    let saved = crate::snapshot_warnings();
    crate::push_quiet();
    let result = eval(&program);
    crate::pop_quiet();
    crate::restore_warnings(saved);
    Ok(match &result? {
      Expr::List(values)
        if values.len() == k + 1 && values.iter().all(is_real_number) =>
      {
        Some(values.to_vec())
      }
      _ => None,
    })
  };
  let slack_between = |lo: Expr, hi: Expr| Expr::List(vec![lo, hi].into());

  // The largest slack, which centres a bounded interval; where it can grow
  // without bound it is capped at 1.
  let mut objective = vec![Expr::Integer(0); k + 1];
  objective[k] = Expr::Integer(-1);
  let first = match solve(
    &objective,
    &slack_between(Expr::Integer(0), Expr::Identifier("Infinity".to_string())),
  )? {
    Some(p) => Some(p),
    None => solve(
      &objective,
      &slack_between(Expr::Integer(0), Expr::Integer(1)),
    )?,
  };
  let Some(first) = first else {
    // An exact system without a solution has no instance.
    return Ok(if inexact { None } else { Some(Vec::new()) });
  };
  let t = eval(&first[k])?;
  // No room at all: the strict inequalities cannot hold.
  if strict && matches!(t, Expr::Integer(0)) {
    return Ok(if inexact { None } else { Some(Vec::new()) });
  }
  let half = eval(&call(
    "Times",
    vec![
      t,
      call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
    ],
  ))?;
  let slack = slack_between(half.clone(), half);
  let same = |p: &[Expr], q: &[Expr]| {
    p.iter()
      .zip(q)
      .all(|(a, b)| crate::evaluator::pattern_matching::expr_equal(a, b))
  };
  let mut points: Vec<Vec<Expr>> = vec![first[..k].to_vec()];
  'more: for j in 0..k {
    for sign in [1, -1] {
      if points.len() >= n {
        break 'more;
      }
      let mut objective = vec![Expr::Integer(0); k + 1];
      objective[j] = Expr::Integer(sign);
      if let Some(p) = solve(&objective, &slack)?
        && !points.iter().any(|q| same(q, &p[..k]))
      {
        points.push(p[..k].to_vec());
      }
    }
  }
  // Still short (a ray has a single extreme point): midpoints between
  // found points, which the convexity of the system keeps inside it.
  let mut i = 0;
  while points.len() < n && points.len() >= 2 && i + 1 < points.len() {
    let mid: Vec<Expr> = points[i]
      .iter()
      .zip(&points[i + 1])
      .map(|(a, b)| {
        eval(&call(
          "Times",
          vec![
            plus2(a.clone(), b.clone()),
            call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
          ],
        ))
      })
      .collect::<Result<_, _>>()?;
    if !points.iter().any(|q| same(q, &mid)) {
      points.insert(i + 1, mid);
    }
    i += 2;
    if i + 1 >= points.len() {
      i = 0;
    }
  }
  points.truncate(n);

  let mut instances = Vec::with_capacity(points.len());
  for point in points {
    let mut rules = Vec::with_capacity(k);
    for (v, value) in vars.iter().zip(point) {
      let value = if inexact {
        eval(&call("N", vec![value]))?
      } else {
        value
      };
      rules.push(rule_expr(v.clone(), value));
    }
    instances.push(Expr::List(rules.into()));
  }
  Ok(Some(instances))
}

/// Try to find instances numerically by evaluating the condition at sample points.
fn find_instance_numerical(
  cond: &Expr,
  var_names: &[String],
  n: usize,
) -> Vec<Expr> {
  use crate::evaluator::evaluate_expr_to_expr;
  use crate::functions::plot::substitute_var;

  // Sample range and step. Real domains are scanned on the same integer
  // grid — only instances at integer points are found either way.
  let (range_lo, range_hi, step) = (-100i64, 100i64, 1i64);

  let mut results: Vec<Expr> = Vec::new();

  // For single variable, simple scan
  if var_names.len() == 1 {
    let var = &var_names[0];
    let mut val = range_lo;
    while val <= range_hi && results.len() < n {
      let test_val = Expr::Integer(val as i128);
      let subst = substitute_var(cond, var, &test_val);
      if let Ok(evaled) = evaluate_expr_to_expr(&subst)
        && matches!(evaled, Expr::Identifier(ref s) if s == "True")
      {
        results.push(Expr::List(
          vec![rule_expr(Expr::Identifier(var.clone()), test_val)].into(),
        ));
      }
      val += step;
    }
    // Nothing inside the scan window: the solution set may be a ray or an
    // interval far from the origin (e.g. `x > 1000`). Probe growing
    // magnitudes, then bisect back towards the origin to land on the
    // integer closest to it.
    if results.is_empty() {
      let holds = |v: i128| {
        let subst = substitute_var(cond, var, &Expr::Integer(v));
        matches!(
          evaluate_expr_to_expr(&subst),
          Ok(Expr::Identifier(ref s)) if s == "True"
        )
      };
      let mut found: Option<(i128, i128)> = None;
      let mut near: i128 = range_hi as i128;
      for _ in 0..30 {
        let far = near * 10;
        if let Some(&hi) = [-far, far].iter().find(|&&v| holds(v)) {
          found = Some((hi, if hi < 0 { -near } else { near }));
          break;
        }
        near = far;
      }
      if let Some((mut hi, mut lo)) = found {
        while (hi - lo).abs() > 1 {
          let mid = lo + (hi - lo) / 2;
          if holds(mid) {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        let dir = if hi < 0 { -1 } else { 1 };
        let mut v = hi;
        while results.len() < n && v.abs() <= hi.abs() + n as i128 * 4 {
          if holds(v) {
            results.push(Expr::List(
              vec![rule_expr(Expr::Identifier(var.clone()), Expr::Integer(v))]
                .into(),
            ));
          }
          v += dir;
        }
      }
    }
  } else if var_names.len() == 2 {
    // For two variables, scan a grid
    let var1 = &var_names[0];
    let var2 = &var_names[1];
    let step2 = step;
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
              rule_expr(Expr::Identifier(var1.clone()), test1.clone()),
              rule_expr(Expr::Identifier(var2.clone()), test2),
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
  } else {
    // Three or more variables: a full grid is far too big, but the
    // instances wolframscript reports are small, so walk each variable
    // outwards from zero — 0, 1, -1, 2, -2, … — and stop at whatever radius
    // keeps the whole sweep inside the evaluation budget. Scanning in that
    // order is also what makes the first hit the one wolframscript names:
    // `x^5 + y^5 + z^5 == w^5 && x > 0` yields `{1, 0, 0, 1}`.
    const BUDGET: usize = 200_000;
    let k = var_names.len();
    let mut radius = 0i64;
    while radius < 100 {
      let side = (2 * (radius + 1) + 1) as f64;
      if side.powi(k as i32) > BUDGET as f64 {
        break;
      }
      radius += 1;
    }
    let values: Vec<i64> = std::iter::once(0)
      .chain((1..=radius).flat_map(|v| [v, -v]))
      .collect();

    let mut idx = vec![0usize; k];
    loop {
      let mut subst = cond.clone();
      for (var, &i) in var_names.iter().zip(&idx) {
        subst = substitute_var(&subst, var, &Expr::Integer(values[i] as i128));
      }
      if let Ok(evaled) = evaluate_expr_to_expr(&subst)
        && matches!(evaled, Expr::Identifier(ref s) if s == "True")
      {
        results.push(Expr::List(
          var_names
            .iter()
            .zip(&idx)
            .map(|(var, &i)| {
              rule_expr(
                Expr::Identifier(var.clone()),
                Expr::Integer(values[i] as i128),
              )
            })
            .collect::<Vec<_>>()
            .into(),
        ));
        if results.len() >= n {
          break;
        }
      }
      // Odometer over the value indices, last variable fastest.
      let mut pos = k;
      loop {
        if pos == 0 {
          break;
        }
        pos -= 1;
        idx[pos] += 1;
        if idx[pos] < values.len() {
          break;
        }
        idx[pos] = 0;
        if pos == 0 {
          pos = usize::MAX;
          break;
        }
      }
      if pos == usize::MAX {
        break;
      }
    }
  }

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

/// Check whether a solution value is an exact rational (an integer or a
/// ratio of integers). Irrational algebraic values such as `Sqrt[2]` are
/// not rational, so `Solve[x^2 == 2, x, Rationals]` filters them out.
fn is_rational_expr(expr: &Expr) -> bool {
  fn is_rational_atom(e: &Expr) -> bool {
    match e {
      Expr::Integer(_) | Expr::BigInteger(_) => true,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        matches!(&args[0], Expr::Integer(_) | Expr::BigInteger(_))
          && matches!(&args[1], Expr::Integer(_) | Expr::BigInteger(_))
      }
      _ => false,
    }
  }
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_rational_expr(operand),
    e if is_rational_atom(e) => true,
    _ => crate::evaluator::evaluate_expr_to_expr(expr)
      .is_ok_and(|evaled| is_rational_atom(&evaled)),
  }
}

/// Check if all values in a solution are real (not complex)
/// Whether a solution assigns a numeric value that is not real, such as
/// `2 ProductLog[-1/2]` (the principal Lambert W below -1/E is complex).
fn has_numerically_nonreal_value(sol: &Expr) -> bool {
  let Expr::List(rules) = sol else {
    return false;
  };
  rules.iter().any(|rule| {
    let Expr::Rule { replacement, .. } = rule else {
      return false;
    };
    let numeric = matches!(
      crate::evaluator::evaluate_function_call_ast(
        "NumericQ",
        std::slice::from_ref(replacement.as_ref()),
      ),
      Ok(Expr::Identifier(ref t)) if t == "True"
    );
    if !numeric {
      return false;
    }
    let im = crate::evaluator::evaluate_expr_to_expr(&call(
      "Im",
      vec![call("N", vec![replacement.as_ref().clone()])],
    ));
    matches!(im, Ok(ref v) if crate::functions::math_ast::try_eval_to_f64(v)
      .is_some_and(|i| i.abs() > 1e-10))
  })
}

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

/// `Solve[eqns, vars, elims]`: the unknowns are taken one at a time, the
/// eliminated ones first, each from an equation linear in it, and substituted
/// into the remaining equations; whatever is left is handed to `Solve`. The
/// rules for `vars` are then resolved back through the eliminated unknowns.
/// `None` when the equations are not plain equalities.
fn solve_with_eliminated(
  eqns: &Expr,
  vars: &[Expr],
  elims: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let same = crate::evaluator::pattern_matching::expr_equal;
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let numerator_of = |e: Expr| {
    let together = eval(call("Together", vec![e]))?;
    eval(call("Numerator", vec![together]))
  };
  let eq_list = solve_equalities(eqns);
  if eq_list.is_empty()
    || eq_list.len()
      != flatten_and_constraints(std::slice::from_ref(eqns)).len()
  {
    return None;
  }
  let mut polys: Vec<Expr> = Vec::new();
  for eq in &eq_list {
    let Expr::Comparison { operands, .. } = eq else {
      return None;
    };
    polys.push(numerator_of(minus2(
      operands[0].clone(),
      operands[1].clone(),
    ))?);
  }
  let order: Vec<Expr> = elims
    .iter()
    .chain(vars.iter().filter(|v| !elims.iter().any(|e| same(e, v))))
    .cloned()
    .collect();
  let mut steps: Vec<(Expr, Expr)> = Vec::new();
  for var in &order {
    let Expr::Identifier(name) = var else {
      continue;
    };
    let mut chosen = None;
    for (i, g) in polys.iter().enumerate() {
      let Some(slope) = eval(call("D", vec![g.clone(), var.clone()])) else {
        continue;
      };
      if is_zero_expr(&slope) || super::eliminate::contains_var(&slope, name) {
        continue;
      }
      let offset = eval(crate::syntax::substitute_variable(
        g,
        name,
        &Expr::Integer(0),
      ))?;
      let size = match eval(call("LeafCount", vec![g.clone()])) {
        Some(Expr::Integer(n)) => n,
        _ => i128::MAX,
      };
      if chosen.as_ref().is_none_or(|(_, _, _, best)| size < *best) {
        chosen = Some((i, slope, offset, size));
      }
    }
    let Some((i, slope, offset, _)) = chosen else {
      continue;
    };
    let value = eval(call(
      "Together",
      vec![div2(minus2(Expr::Integer(0), offset), slope)],
    ))?;
    polys.remove(i);
    for g in &mut polys {
      if super::eliminate::contains_var(g, name) {
        *g = numerator_of(crate::syntax::substitute_variable(g, name, &value))?;
      }
    }
    steps.push((var.clone(), value));
  }
  // What the substitution could not settle.
  polys.retain(|g| !is_zero_expr(g));
  if polys
    .iter()
    .any(|g| matches!(g, Expr::Integer(_) | Expr::Real(_)))
  {
    return Some(Ok(Expr::List(vec![].into())));
  }
  let unsolved: Vec<Expr> = order
    .iter()
    .filter(|v| !steps.iter().any(|(s, _)| same(s, v)))
    .cloned()
    .collect();
  let tails: Vec<Vec<(Expr, Expr)>> = if polys.is_empty() {
    vec![vec![]]
  } else {
    let rest = Expr::List(
      polys
        .iter()
        .map(|g| Expr::Comparison {
          operands: vec![g.clone(), Expr::Integer(0)],
          operators: vec![ComparisonOp::Equal],
        })
        .collect::<Vec<_>>()
        .into(),
    );
    let solved = match solve_ast(&[rest, Expr::List(unsolved.into())]) {
      Ok(s) => s,
      Err(e) => return Some(Err(e)),
    };
    let Expr::List(ref sols) = solved else {
      return None;
    };
    let mut tails = Vec::new();
    for sol in sols {
      let Expr::List(rules) = sol else {
        return None;
      };
      let mut tail = Vec::new();
      for r in rules {
        let Expr::Rule {
          pattern,
          replacement,
        } = r
        else {
          return None;
        };
        tail.push((pattern.as_ref().clone(), replacement.as_ref().clone()));
      }
      tails.push(tail);
    }
    tails
  };
  let mut out = Vec::new();
  for tail in tails {
    let mut known: Vec<(Expr, Expr)> = tail;
    for (var, value) in steps.iter().rev() {
      let mut v = value.clone();
      for (k, kv) in &known {
        if let Expr::Identifier(kn) = k {
          v = crate::syntax::substitute_variable(&v, kn, kv);
        }
      }
      let v = eval(call("Together", vec![v]))?;
      known.push((
        var.clone(),
        eval(call("Cancel", vec![v.clone()])).unwrap_or(v),
      ));
    }
    let mut rules = Vec::new();
    for var in vars {
      if let Some((_, val)) = known.iter().find(|(k, _)| same(k, var)) {
        rules.push(rule_expr(var.clone(), val.clone()));
      }
    }
    out.push(Expr::List(rules.into()));
  }
  Some(Ok(Expr::List(out.into())))
}

/// The roots of the cubic `a x^3 + b x^2 + c x + d` with symbolic
/// coefficients, in wolframscript's Cardano form: with
/// `Δ0 = 3 a c - b^2`, `Δ1 = -2 b^3 + 9 a b c - 27 a^2 d` and
/// `C = (Δ1 + Sqrt[4 Δ0^3 + Δ1^2])^(1/3)`,
///
/// ```text
/// x1 = -b/(3 a) - 2^(1/3) Δ0/(3 a C) + C/(3 2^(1/3) a)
/// x2 = -b/(3 a) + (1 + I Sqrt[3]) Δ0/(3 2^(2/3) a C) - (1 - I Sqrt[3]) C/(6 2^(1/3) a)
/// ```
///
/// and `x3` the conjugate of `x2`'s units. A discriminant whose expansion
/// has an integer content `k > 1` is written `Sqrt[k] Sqrt[rest]`
/// (`Sqrt[3] Sqrt[4 p^3 + 27 q^2]` for `x^3 + p x + q`), and an integer
/// factor `g` common to `Δ1` and that radical comes out of the cube root
/// as `g^(1/3)`.
fn symbolic_cubic_roots(coeffs: &[Expr]) -> Option<Vec<Expr>> {
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let parse = |s: &str| crate::syntax::string_to_expr(s).ok();
  let fill = |template: &str, values: &[(&str, &Expr)]| -> Option<Expr> {
    let mut e = parse(template)?;
    for (name, value) in values {
      e = crate::syntax::substitute_variable(&e, name, value);
    }
    eval(e)
  };
  // A negative leading number is negated away (`v == (1 + y)^3` is
  // solved as `(1 + y)^3 - v`).
  let negated: Vec<Expr>;
  let coeffs = if let Some(lead) = coeffs.last()
    && let Some(v) = crate::functions::math_ast::try_eval_to_f64(&match lead {
      Expr::FunctionCall { name, args }
        if name == "Times" && !args.is_empty() =>
      {
        args[0].clone()
      }
      other => other.clone(),
    })
    && v < 0.0
  {
    negated = coeffs
      .iter()
      .map(|c| eval(call("Times", vec![Expr::Integer(-1), c.clone()])))
      .collect::<Option<_>>()?;
    negated.as_slice()
  } else {
    coeffs
  };
  let [d, c, b, a] = coeffs else {
    return None;
  };
  let abcd = [
    ("Solve$a", a),
    ("Solve$b", b),
    ("Solve$c", c),
    ("Solve$d", d),
  ];
  let d0 = fill("Expand[3 Solve$a Solve$c - Solve$b^2]", &abcd)?;
  let d1 = fill(
    "Expand[-2 Solve$b^3 + 9 Solve$a Solve$b Solve$c - 27 Solve$a^2 Solve$d]",
    &abcd,
  )?;
  // The integer content of an expanded polynomial and what is left.
  let integer_content = |e: &Expr| -> Option<(i128, Expr)> {
    let expanded = eval(call("Expand", vec![e.clone()]))?;
    let list = crate::functions::polynomial_ast::factor::factor_terms_list_ast(
      std::slice::from_ref(&expanded),
    )
    .ok()?;
    let Expr::List(ref parts) = list else {
      return None;
    };
    match parts.as_slice() {
      [Expr::Integer(k), rest @ ..] if !rest.is_empty() => {
        let rest = if rest.len() == 1 {
          rest[0].clone()
        } else {
          eval(call("Times", rest.to_vec()))?
        };
        Some((*k, rest))
      }
      _ => None,
    }
  };
  // A shifted perfect cube plus a constant (Δ0 == 0): the radical is Δ1
  // itself and the Δ0 terms vanish, x = -b/(3 a) + ω^k (2 Δ1)^(1/3)/(3 a).
  if matches!(d0, Expr::Integer(0)) {
    // The integer content leaves the cube root: (54 (1 - c))^(1/3) is
    // 3 2^(1/3) (1 - c)^(1/3).
    let twice = fill("2 Solve$q", &[("Solve$q", &d1)])?;
    let cube = match integer_content(&twice) {
      Some((k, rest)) if k.abs() > 1 => fill(
        "Solve$k^(1/3) Solve$r^(1/3)",
        &[
          ("Solve$k", &Expr::Integer(k.abs())),
          (
            "Solve$r",
            &eval(call("Times", vec![Expr::Integer(k.signum()), rest]))?,
          ),
        ],
      )?,
      _ => fill("Solve$x^(1/3)", &[("Solve$x", &twice)])?,
    };
    let values = [("Solve$a", a), ("Solve$b", b), ("Solve$C", &cube)];
    return [
      "Expand[-Solve$b/(3 Solve$a)] + Solve$C/(3 2^(1/3) Solve$a)",
      "Expand[-Solve$b/(3 Solve$a)] - (1 - I Sqrt[3]) Solve$C/(6 2^(1/3) Solve$a)",
      "Expand[-Solve$b/(3 Solve$a)] - (1 + I Sqrt[3]) Solve$C/(6 2^(1/3) Solve$a)",
    ]
    .iter()
    .map(|t| fill(t, &values))
    .collect();
  }
  let disc = fill(
    "4 Solve$p^3 + Solve$q^2",
    &[("Solve$p", &d0), ("Solve$q", &d1)],
  )?;

  // wolframscript expands the discriminant unless that makes it clearly
  // bigger (more than two leaves: `4 (-b^2 + 3 a c)^3 + (…)^2` stays as
  // it is, `32 + (7 - 27 c)^2` becomes `27 (3 - 14 c + 27 c^2)`).
  let leaves = |e: &Expr| -> Option<i128> {
    match eval(call("LeafCount", vec![e.clone()]))? {
      Expr::Integer(n) => Some(n),
      _ => None,
    }
  };
  let expanded_disc = eval(call("Expand", vec![disc.clone()]))?;
  let expand = leaves(&expanded_disc)? - leaves(&disc)? <= 2;
  let content = if expand { integer_content(&disc) } else { None };
  let (radical, radical_scale) = match content {
    Some((k, rest)) if k.abs() > 1 => {
      // Sqrt[k] = m Sqrt[s] with s squarefree.
      let k_abs = k.unsigned_abs();
      let mut m: u128 = 1;
      let mut s = k_abs;
      let mut f: u128 = 2;
      while f * f <= s {
        while s % (f * f) == 0 {
          s /= f * f;
          m *= f;
        }
        f += 1;
      }
      let rest = if k < 0 {
        eval(call("Times", vec![Expr::Integer(-1), rest]))?
      } else {
        rest
      };
      (
        eval(call(
          "Times",
          vec![
            call("Sqrt", vec![Expr::Integer(s as i128)]),
            call("Sqrt", vec![rest]),
          ],
        ))?,
        m as i128,
      )
    }
    _ if expand => (call("Sqrt", vec![expanded_disc]), 1),
    _ => (call("Sqrt", vec![disc.clone()]), 1),
  };
  // A factor common to Δ1 and the radical's integer scale.
  let g = match integer_content(&d1) {
    Some((k, _)) => gcd_i128(k.abs(), radical_scale),
    None => match &d1 {
      Expr::Integer(n) => gcd_i128(n.abs(), radical_scale),
      _ => 1,
    },
  };
  let inner = fill(
    "Expand[Solve$q / Solve$g] + (Solve$m / Solve$g) Solve$r",
    &[
      ("Solve$q", &d1),
      ("Solve$g", &Expr::Integer(g)),
      ("Solve$m", &Expr::Integer(radical_scale)),
      ("Solve$r", &radical),
    ],
  )?;
  let cube = fill(
    "Solve$g^(1/3) Solve$x^(1/3)",
    &[("Solve$g", &Expr::Integer(g)), ("Solve$x", &inner)],
  )?;
  let values = [
    ("Solve$a", a),
    ("Solve$b", b),
    ("Solve$p", &d0),
    ("Solve$C", &cube),
  ];
  [
    "Expand[-Solve$b/(3 Solve$a)] - 2^(1/3) Solve$p/(3 Solve$a Solve$C) + Solve$C/(3 2^(1/3) Solve$a)",
    "Expand[-Solve$b/(3 Solve$a)] + (1 + I Sqrt[3]) Solve$p/(3 2^(2/3) Solve$a Solve$C) - (1 - I Sqrt[3]) Solve$C/(6 2^(1/3) Solve$a)",
    "Expand[-Solve$b/(3 Solve$a)] + (1 - I Sqrt[3]) Solve$p/(3 2^(2/3) Solve$a Solve$C) - (1 + I Sqrt[3]) Solve$C/(6 2^(1/3) Solve$a)",
  ]
  .iter()
  .map(|t| fill(t, &values))
  .collect()
}

#[cfg(test)]
mod tests {
  use super::NumNode;

  fn exp_of_var() -> NumNode {
    NumNode::Exp(Box::new(NumNode::Var(0)))
  }

  #[test]
  fn compiled_eval_survives_intermediate_overflow() {
    // E^800 * 1e-300 is finite although E^800 is not a machine number.
    let node = NumNode::Mul(vec![exp_of_var(), NumNode::Const(1e-300)]);
    let v = node.eval(&[800.0]);
    let expected = 300.0f64.mul_add(-std::f64::consts::LN_10, 800.0).exp();
    assert!((v / expected - 1.0).abs() < 1e-12, "{v} vs {expected}");
    // Sums, quotients and integer powers stay in the wide range too.
    let node = NumNode::Div(
      Box::new(NumNode::Pow(
        Box::new(NumNode::Add(vec![exp_of_var(), exp_of_var()])),
        Box::new(NumNode::Const(2.0)),
      )),
      Box::new(NumNode::Exp(Box::new(NumNode::Const(1599.0)))),
    );
    let v = node.eval(&[800.0]);
    assert!((v / (4.0 * std::f64::consts::E) - 1.0).abs() < 1e-12, "{v}");
  }

  #[test]
  fn compiled_eval_flushes_underflow_like_machine_arithmetic() {
    // Wolfram: Exp[800.] (Exp[-800.] - Exp[-801.]) is 0., not 0.63.
    let shifted = |c: f64| {
      NumNode::Exp(Box::new(NumNode::Add(vec![
        NumNode::Neg(Box::new(NumNode::Var(0))),
        NumNode::Const(c),
      ])))
    };
    let node = NumNode::Mul(vec![
      exp_of_var(),
      NumNode::Sub(Box::new(shifted(0.0)), Box::new(shifted(-1.0))),
    ]);
    assert_eq!(node.eval(&[800.0]), 0.0);
    // A plain machine result is untouched.
    assert_eq!(node.eval(&[1.0]), 1.0 - (-1.0f64).exp());
  }

  #[test]
  fn compiled_real_part_compiles_and_handles_complex_values() {
    // Re[Log[x]] + Abs[Sqrt[x]]: real arithmetic while x > 0; for x < 0
    // the call is evaluated in complex arithmetic (Log[2] and Sqrt[2]).
    let expr =
      crate::syntax::string_to_expr("Re[Log[x]] + Abs[Sqrt[x]] + Im[x]")
        .unwrap();
    let node = super::compile_numeric(&expr, &["x".to_string()]).unwrap();
    assert!(!node.has_fallback());
    let at_e = node.eval(&[std::f64::consts::E]);
    assert!((at_e - (1.0 + std::f64::consts::E.sqrt())).abs() < 1e-15);
    let at_minus_two = node.eval(&[-2.0]);
    let expected = 2.0f64.ln() + 2.0f64.sqrt();
    assert!(
      (at_minus_two - expected).abs() < 1e-14,
      "{at_minus_two} vs {expected}"
    );
  }
}
