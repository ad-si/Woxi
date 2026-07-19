use crate::InterpreterError;
use crate::functions::math_ast::{gcd as gcd_i128, lcm_i128};
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, unevaluated,
};

/// RSolve[{recurrence, initial_conditions...}, a, n]
/// RSolve[recurrence, a[n], n] — single equation, return rule for `a[n]`
/// Solve constant-coefficient linear recurrence relations.
pub fn rsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("RSolve", args);

  if args.len() != 3 {
    return Ok(unevaluated());
  }

  // Extract the variable name (e.g. "n")
  let var_name = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated()),
  };

  // Second arg is either `a` (Identifier — return rule for the function)
  // or `a[var]` (FunctionCall — return rule for `a[var]` directly).
  let (func_name, return_as_func_call) = match &args[1] {
    Expr::Identifier(name) => (name.clone(), false),
    Expr::FunctionCall { name, args: fargs }
      if fargs.len() == 1
        && matches!(&fargs[0], Expr::Identifier(v) if *v == var_name) =>
    {
      (name.clone(), true)
    }
    _ => return Ok(unevaluated()),
  };

  // Extract equations from the first argument. Accept a list, a single
  // equation (Comparison or Equal[...]), or a conjunction joined with `&&`
  // (e.g. `a[n] == a[n-1] + 1 && a[1] == 1`) by flattening it into a list.
  let equations = match &args[0] {
    Expr::List(eqs) => eqs.clone(),
    Expr::Comparison { .. } => vec![args[0].clone()].into(),
    Expr::FunctionCall { name, .. } if name == "Equal" => {
      vec![args[0].clone()].into()
    }
    Expr::BinaryOp {
      op: BinaryOperator::And,
      ..
    } => flatten_and(&args[0]).into(),
    Expr::FunctionCall { name, .. } if name == "And" => {
      flatten_and(&args[0]).into()
    }
    _ => return Ok(unevaluated()),
  };

  // Separate the recurrence relation from initial conditions
  let mut recurrence = None;
  let mut initial_conditions: Vec<(i128, Expr)> = Vec::new(); // (index, value)

  for eq in &equations {
    // Each equation should be lhs == rhs
    let (lhs, rhs) = match extract_equation(eq) {
      Some(pair) => pair,
      None => return Ok(unevaluated()),
    };

    // Check if this is an initial condition: a[integer] == value
    if let Some((name, idx)) = extract_func_at_integer(&lhs)
      && name == func_name
    {
      initial_conditions.push((idx, rhs));
      continue;
    }
    if let Some((name, idx)) = extract_func_at_integer(&rhs)
      && name == func_name
    {
      initial_conditions.push((idx, lhs));
      continue;
    }

    // Otherwise this is the recurrence relation
    if recurrence.is_some() {
      return Ok(unevaluated()); // Multiple recurrences not supported
    }
    recurrence = Some((lhs, rhs));
  }

  let (rec_lhs, rec_rhs) = match recurrence {
    Some(r) => r,
    None => return Ok(unevaluated()),
  };

  // Try to solve as constant-coefficient linear recurrence
  // Extract: a[n+k] = c_{k-1}*a[n+k-1] + ... + c_0*a[n]
  if let Some(solution) = solve_const_coeff_linear(
    &rec_lhs,
    &rec_rhs,
    &func_name,
    &var_name,
    &initial_conditions,
  ) {
    return Ok(wrap_rsolve_result(
      solution,
      &func_name,
      &var_name,
      return_as_func_call,
    ));
  }

  // First-order arithmetic progression a[n] == a[n-1] + d (d free of n).
  if let Some(solution) = solve_first_order_arithmetic(
    &rec_lhs,
    &rec_rhs,
    &func_name,
    &var_name,
    &initial_conditions,
  ) {
    return Ok(wrap_rsolve_result(
      solution,
      &func_name,
      &var_name,
      return_as_func_call,
    ));
  }

  Ok(unevaluated())
}

/// True if `expr` references `func_name[...]` anywhere.
fn contains_func(expr: &Expr, func_name: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == func_name || args.iter().any(|a| contains_func(a, func_name))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_func(left, func_name) || contains_func(right, func_name)
    }
    Expr::UnaryOp { operand, .. } => contains_func(operand, func_name),
    Expr::List(items) => items.iter().any(|a| contains_func(a, func_name)),
    _ => false,
  }
}

/// Like `collect_recurrence_terms`, but instead of bailing on a non-`func`
/// term it collects it (with the running sign) into `forcing`. Returns false
/// only for structures it cannot decompose (e.g. a non-linear `func` argument
/// or a symbolic-coefficient `func` term).
fn collect_terms_with_forcing(
  expr: &Expr,
  func_name: &str,
  var_name: &str,
  sign: i128,
  terms: &mut Vec<(i128, i128)>,
  forcing: &mut Vec<Expr>,
) -> bool {
  let push_forcing = |e: &Expr, sign: i128, forcing: &mut Vec<Expr>| -> bool {
    if contains_func(e, func_name) {
      return false; // non-linear dependence on func — unsupported
    }
    forcing.push(if sign >= 0 {
      e.clone()
    } else {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(e.clone()),
      }
    });
    true
  };
  match expr {
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      match extract_var_offset(&args[0], var_name) {
        Some(offset) => {
          terms.push((offset, sign));
          true
        }
        None => false,
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if collect_coeff_times_term(left, right, func_name, var_name, sign, terms)
      {
        return true;
      }
      push_forcing(expr, sign, forcing)
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if collect_coeff_times_term(
        &args[0], &args[1], func_name, var_name, sign, terms,
      ) {
        return true;
      }
      push_forcing(expr, sign, forcing)
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().all(|a| {
        collect_terms_with_forcing(a, func_name, var_name, sign, terms, forcing)
      })
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_terms_with_forcing(
        left, func_name, var_name, sign, terms, forcing,
      ) && collect_terms_with_forcing(
        right, func_name, var_name, sign, terms, forcing,
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      collect_terms_with_forcing(
        left, func_name, var_name, sign, terms, forcing,
      ) && collect_terms_with_forcing(
        right, func_name, var_name, -sign, terms, forcing,
      )
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => collect_terms_with_forcing(
      operand, func_name, var_name, -sign, terms, forcing,
    ),
    Expr::Integer(0) => true,
    _ => push_forcing(expr, sign, forcing),
  }
}

/// First-order recurrence a[n] == a[n-1] + d with a constant increment `d`
/// (free of the index n). The closed form is the arithmetic progression
///   no initial condition:  d*n + C[1]
///   a[k0] == v:            v + d*(n - k0)   (simplified)
/// Returns None for higher order, a leading coefficient other than ±1, a
/// non-unit step ratio, an index-dependent forcing term (e.g. a[n-1] + n), or
/// more than one initial condition.
fn solve_first_order_arithmetic(
  lhs: &Expr,
  rhs: &Expr,
  func_name: &str,
  var_name: &str,
  ics: &[(i128, Expr)],
) -> Option<Expr> {
  let mut terms: Vec<(i128, i128)> = Vec::new();
  let mut forcing: Vec<Expr> = Vec::new();
  if !collect_terms_with_forcing(
    lhs,
    func_name,
    var_name,
    1,
    &mut terms,
    &mut forcing,
  ) || !collect_terms_with_forcing(
    rhs,
    func_name,
    var_name,
    -1,
    &mut terms,
    &mut forcing,
  ) {
    return None;
  }

  // Combine func terms by offset.
  let mut combined: std::collections::HashMap<i128, i128> =
    std::collections::HashMap::new();
  for (offset, coeff) in &terms {
    *combined.entry(*offset).or_insert(0) += coeff;
  }
  combined.retain(|_, c| *c != 0);
  if combined.len() != 2 {
    return None;
  }
  let lo = *combined.keys().min().unwrap();
  let hi = *combined.keys().max().unwrap();
  if hi - lo != 1 {
    return None; // not first order
  }
  let c_hi = combined[&hi];
  let c_lo = combined[&lo];
  // Arithmetic progression requires the step ratio -c_lo/c_hi to be 1.
  if c_hi == 0 || c_lo != -c_hi {
    return None;
  }

  // increment d = -forcing_total / c_hi, where the equation is
  // c_hi*a[n+hi] + c_lo*a[n+lo] + forcing_total == 0.
  let forcing_total = match forcing.len() {
    0 => Expr::Integer(0),
    1 => forcing.remove(0),
    _ => crate::functions::math_ast::plus_ast(&forcing).ok()?,
  };
  let d = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(forcing_total),
    }),
    right: Box::new(Expr::Integer(c_hi)),
  })
  .ok()?;
  if crate::functions::polynomial_ast::contains_var(&d, var_name) {
    return None; // index-dependent forcing — not an arithmetic progression
  }

  let n = Expr::Identifier(var_name.to_string());
  // d * n
  let dn = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(d.clone()),
    right: Box::new(n.clone()),
  })
  .ok()?;

  // Build the Plus terms in wolframscript's display order with a raw BinaryOp
  // (the canonical sorter would reorder, e.g. `C[1] + n` instead of `n + C[1]`).
  let raw_plus = |a: Expr, b: Expr| Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(a),
    right: Box::new(b),
  };
  match ics.len() {
    0 => {
      // d*n + C[1]
      let c1 = Expr::FunctionCall {
        name: "C".to_string(),
        args: vec![Expr::Integer(1)].into(),
      };
      if matches!(&dn, Expr::Integer(0)) {
        return Some(c1);
      }
      Some(raw_plus(dn, c1))
    }
    1 => {
      // a[n] = v + d*(n - k0) = (v - d*k0) + d*n, with the constant first.
      let (k0, v) = &ics[0];
      let constant = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(v.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(d),
          right: Box::new(Expr::Integer(*k0)),
        }),
      })
      .ok()?;
      if matches!(&dn, Expr::Integer(0)) {
        return Some(constant);
      }
      if matches!(&constant, Expr::Integer(0)) {
        return Some(dn);
      }
      Some(raw_plus(constant, dn))
    }
    _ => None, // over-determined: leave to the general solver / unevaluated
  }
}

/// Wrap a closed-form `solution` (in `var_name`) as Wolfram's RSolve output.
/// When `return_as_func_call` is true, returns `{{a[n] -> solution}}`;
/// otherwise returns `{{a -> Function[{n}, solution]}}`.
fn wrap_rsolve_result(
  solution: Expr,
  func_name: &str,
  var_name: &str,
  return_as_func_call: bool,
) -> Expr {
  let pattern = if return_as_func_call {
    Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![Expr::Identifier(var_name.to_string())].into(),
    }
  } else {
    Expr::Identifier(func_name.to_string())
  };
  let replacement = if return_as_func_call {
    solution
  } else {
    Expr::FunctionCall {
      name: "Function".to_string(),
      args: vec![
        Expr::List(vec![Expr::Identifier(var_name.to_string())].into()),
        solution,
      ]
      .into(),
    }
  };
  let rule = Expr::Rule {
    pattern: Box::new(pattern),
    replacement: Box::new(replacement),
  };
  Expr::List(vec![Expr::List(vec![rule].into())].into())
}

/// Flatten a conjunction (`a && b && c`, whether represented as nested
/// `BinaryOp::And` or a flattened `And[...]` FunctionCall) into the list of
/// its conjuncts. Non-And expressions yield a single-element list.
fn flatten_and(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => {
      let mut out = flatten_and(left);
      out.extend(flatten_and(right));
      out
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      args.iter().flat_map(flatten_and).collect()
    }
    _ => vec![expr.clone()],
  }
}

/// RSolveValue[eqns, expr, n] — like RSolve, but returns the value of `expr`
/// under the solution instead of a list of replacement rules.
pub fn rsolve_value_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("RSolveValue", args);

  if args.len() != 3 {
    return Ok(unevaluated());
  }

  // Determine the unknown function from the second argument: either the
  // bare symbol `a` or any call `a[...]` (e.g. `a[n]`, `a[3]`).
  let func_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    Expr::FunctionCall { name, .. } => name.clone(),
    _ => return Ok(unevaluated()),
  };

  // Solve for the function itself: {{a -> Function[{n}, body]}}
  let rsolve_args = [
    args[0].clone(),
    Expr::Identifier(func_name),
    args[2].clone(),
  ];
  let solved = rsolve_ast(&rsolve_args)?;

  let rule = match &solved {
    Expr::List(outer) if outer.len() == 1 => match &outer[0] {
      Expr::List(inner) if inner.len() == 1 => match &inner[0] {
        rule @ Expr::Rule { .. } => rule.clone(),
        _ => return Ok(unevaluated()),
      },
      _ => return Ok(unevaluated()),
    },
    _ => return Ok(unevaluated()),
  };

  // Bare symbol requested: return the Function[...] itself.
  if let Expr::Rule { replacement, .. } = &rule
    && matches!(&args[1], Expr::Identifier(_))
  {
    return Ok((**replacement).clone());
  }

  // Otherwise substitute the solution into the requested expression
  // (e.g. a[n] or a[3]) and evaluate.
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "ReplaceAll".to_string(),
    args: vec![args[1].clone(), rule].into(),
  })
}

/// Extract lhs and rhs from an equation (Comparison or FunctionCall Equal)
fn extract_equation(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      Some((operands[0].clone(), operands[1].clone()))
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// Check if expr is f[integer] and return (f_name, integer_value)
fn extract_func_at_integer(expr: &Expr) -> Option<(String, i128)> {
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        Some((name.clone(), *n))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Try to solve a constant-coefficient linear recurrence.
/// Handles: a[n+k] == c_{k-1}*a[n+k-1] + ... + c_0*a[n] + constant
fn solve_const_coeff_linear(
  lhs: &Expr,
  rhs: &Expr,
  func_name: &str,
  var_name: &str,
  initial_conditions: &[(i128, Expr)],
) -> Option<Expr> {
  // Collect all terms involving func_name[var_name + offset]
  // Move everything to lhs - rhs = 0
  let mut terms: Vec<(i128, i128)> = Vec::new(); // (offset, coefficient)

  // Extract terms from lhs (positive) and rhs (negative when moved).
  // Bail if any term is not a recognized homogeneous piece (e.g. a constant
  // or a free `n` term) — silently dropping it would solve a different
  // (homogeneous) recurrence and return a wrong answer.
  if !collect_recurrence_terms(lhs, func_name, var_name, 1, &mut terms)
    || !collect_recurrence_terms(rhs, func_name, var_name, -1, &mut terms)
  {
    return None;
  }

  if terms.is_empty() {
    return None;
  }

  // Combine terms with same offset
  let mut combined: std::collections::HashMap<i128, i128> =
    std::collections::HashMap::new();
  for (offset, coeff) in &terms {
    *combined.entry(*offset).or_insert(0) += coeff;
  }

  // Find the order (max offset - min offset)
  let min_offset = *combined.keys().min()?;
  let max_offset = *combined.keys().max()?;
  let order = (max_offset - min_offset) as usize;

  if order == 0 || order > 10 {
    return None;
  }

  // Build characteristic polynomial coefficients
  // If recurrence is c_k*a[n+k] + c_{k-1}*a[n+k-1] + ... + c_0*a[n] = 0
  // Characteristic equation: c_k*r^k + c_{k-1}*r^{k-1} + ... + c_0 = 0
  let mut char_coeffs = vec![0i128; order + 1];
  for (&offset, &coeff) in &combined {
    let idx = (offset - min_offset) as usize;
    char_coeffs[idx] = coeff;
  }

  // Golden-ratio recurrence r^2 == r + 1 (e.g. Fibonacci): the characteristic
  // roots (1 ± Sqrt[5])/2 are irrational, so the rational-root solver below
  // cannot handle it. wolframscript expresses these solutions in the
  // Fibonacci/LucasL basis instead of surds.
  if order == 2
    && char_coeffs[2] != 0
    && char_coeffs[1] == -char_coeffs[2]
    && char_coeffs[0] == -char_coeffs[2]
  {
    return solve_fibonacci_lucas(initial_conditions, var_name);
  }

  // Find roots of the characteristic polynomial
  let roots = find_characteristic_roots(&char_coeffs)?;

  if roots.len() != order {
    return None; // Need all roots for general solution
  }

  // No initial conditions: emit the general solution
  // a[n] = C[1]*r1^n + C[2]*r2^n + ...
  if initial_conditions.is_empty() {
    return build_general_solution(&roots, var_name);
  }

  // Under-determined: fewer ICs than order. Keep the first
  // (order - ics.len()) constants symbolic (as `C[1]`, `C[2]`, …) and use
  // the ICs to eliminate the remaining ones, matching wolframscript's
  // convention of preserving low-indexed C[k]s as the free parameters.
  if initial_conditions.len() < order {
    return build_partial_solution(&roots, initial_conditions, var_name);
  }

  // Check initial conditions match order
  if initial_conditions.len() != order {
    return None;
  }

  // First-order homogeneous with one initial condition: wolframscript writes
  // the closed form as `v * r^(n - k)` and folds the coefficient's factors of
  // r into the exponent (e.g. a[1]==6, r=2 → 3*2^n, not 6*2^(n-1)). Woxi's core
  // Times/Power renderer doesn't fold powers of the base out of an integer
  // coefficient, so build the folded form directly here.
  if order == 1
    && let Some(sol) =
      build_first_order_with_ic(&roots[0], &initial_conditions[0], var_name)
  {
    return Some(sol);
  }

  // Build and solve the system for constants
  // General solution: a[n] = c1*r1^n + c2*r2^n + ...
  // Apply initial conditions to find c1, c2, ...
  let constants = solve_initial_conditions(&roots, initial_conditions)?;

  // Build the solution expression: c1*r1^n + c2*r2^n + ...
  build_solution(&constants, &roots, var_name)
}

/// `(Fibonacci[k], LucasL[k])` for any integer index `k`, extended to
/// negative indices via `F[-m] = (-1)^(m+1) F[m]` and `L[-m] = (-1)^m L[m]`.
fn fib_lucas_at(k: i128) -> Option<(i128, i128)> {
  let steps = k.checked_abs()?;
  let (mut f0, mut f1) = (0i128, 1i128);
  let (mut l0, mut l1) = (2i128, 1i128);
  for _ in 0..steps {
    let f2 = f0.checked_add(f1)?;
    f0 = f1;
    f1 = f2;
    let l2 = l0.checked_add(l1)?;
    l0 = l1;
    l1 = l2;
  }
  let (mut f, mut l) = (f0, l0);
  if k < 0 {
    if steps % 2 == 0 {
      f = -f;
    } else {
      l = -l;
    }
  }
  Some((f, l))
}

/// Reduce a rational to lowest terms with a positive denominator.
fn rat_reduce(n: i128, d: i128) -> Option<(i128, i128)> {
  if d == 0 {
    return None;
  }
  let g = gcd_i128(n, d);
  let (mut n, mut d) = if g == 0 { (0, 1) } else { (n / g, d / g) };
  if d < 0 {
    n = -n;
    d = -d;
  }
  Some((n, d))
}

/// Solve the golden-ratio recurrence (characteristic polynomial r^2 - r - 1,
/// e.g. the Fibonacci recurrence) in the basis wolframscript uses:
/// `a[n] = alpha*Fibonacci[n] + beta*LucasL[n]`.
///
/// With no initial conditions both constants stay free
/// (`C[1]*Fibonacci[n] + C[2]*LucasL[n]`). One IC keeps `alpha = C[1]` free
/// and eliminates `beta`; two ICs at distinct indices determine both
/// constants exactly (rational values allowed).
fn solve_fibonacci_lucas(
  ics: &[(i128, Expr)],
  var_name: &str,
) -> Option<Expr> {
  let ic_value = |e: &Expr| -> Option<(i128, i128)> {
    let evaled =
      crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
    crate::functions::math_ast::expr_to_rational(&evaled)
      .and_then(|(n, d)| rat_reduce(n, d))
  };

  match ics {
    [] => {
      // C[1]*Fibonacci[n] + C[2]*LucasL[n]
      build_fib_lucas_combination((0, 1), (1, 1), (0, 1), (1, 1), 2, var_name)
    }
    [(k, v_expr)] => {
      // alpha = C[1] stays free; beta = (v - C[1]*F[k]) / L[k]. (L[k] is
      // never 0 for integer k, so the elimination is always valid.)
      let (vn, vd) = ic_value(v_expr)?;
      let (fk, lk) = fib_lucas_at(*k)?;
      let beta_const = rat_reduce(vn, vd.checked_mul(lk)?)?;
      let beta_c1 = rat_reduce(-fk, lk)?;
      build_fib_lucas_combination(
        (0, 1),
        (1, 1),
        beta_const,
        beta_c1,
        1,
        var_name,
      )
    }
    [(k1, v1_expr), (k2, v2_expr)] => {
      if k1 == k2 {
        return None;
      }
      let (v1n, v1d) = ic_value(v1_expr)?;
      let (v2n, v2d) = ic_value(v2_expr)?;
      let (f1, l1) = fib_lucas_at(*k1)?;
      let (f2, l2) = fib_lucas_at(*k2)?;
      // Solve alpha*F[k] + beta*L[k] == v for both indices (Cramer's rule).
      let det = f1.checked_mul(l2)?.checked_sub(f2.checked_mul(l1)?)?;
      if det == 0 {
        return None;
      }
      // alpha = (v1*L[k2] - v2*L[k1]) / det, over common denominator v1d*v2d.
      let alpha_num = v1n
        .checked_mul(v2d)?
        .checked_mul(l2)?
        .checked_sub(v2n.checked_mul(v1d)?.checked_mul(l1)?)?;
      let beta_num = v2n
        .checked_mul(v1d)?
        .checked_mul(f1)?
        .checked_sub(v1n.checked_mul(v2d)?.checked_mul(f2)?)?;
      let denom = v1d.checked_mul(v2d)?.checked_mul(det)?;
      let alpha = rat_reduce(alpha_num, denom)?;
      let beta = rat_reduce(beta_num, denom)?;
      build_fib_lucas_combination(alpha, (0, 1), beta, (0, 1), 1, var_name)
    }
    _ => None, // over-determined — leave unevaluated
  }
}

/// Render `(ac + a_c1*C[1])*Fibonacci[n] + (bc + b_c1*C[c_idx_b])*LucasL[n]`
/// the way wolframscript displays it: the common denominator of all four
/// rational coefficients is pulled out as a trailing `/d`, and the numerator
/// terms appear as `m*C[k]*Fibonacci[n]`-style products in the fixed order
/// Fibonacci-constant, Fibonacci-C, LucasL-constant, LucasL-C.
/// `c_idx_b` is the index of the arbitrary constant attached to the LucasL
/// term (2 for the fully general solution, 1 when eliminating against ICs).
fn build_fib_lucas_combination(
  a_const: (i128, i128),
  a_c1: (i128, i128),
  b_const: (i128, i128),
  b_c1: (i128, i128),
  c_idx_b: i128,
  var_name: &str,
) -> Option<Expr> {
  let coeffs = [a_const, a_c1, b_const, b_c1];
  let mut denom = 1i128;
  for &(_, d) in &coeffs {
    denom = lcm_i128(denom, d);
  }
  // Integer numerators over the common denominator, reduced by their
  // collective gcd (so e.g. 2*C[1]*Fibonacci[n]/2 collapses to
  // C[1]*Fibonacci[n]).
  let mut nums = [0i128; 4];
  for (i, &(n, d)) in coeffs.iter().enumerate() {
    nums[i] = n.checked_mul(denom / d)?;
  }
  let mut g = denom;
  for &n in &nums {
    g = gcd_i128(g, n);
  }
  if g > 1 {
    for n in &mut nums {
      *n /= g;
    }
    denom /= g;
  }
  if nums.iter().all(|&n| n == 0) {
    return Some(Expr::Integer(0));
  }

  let n_var = Expr::Identifier(var_name.to_string());
  let base = |name: &str| Expr::FunctionCall {
    name: name.to_string(),
    args: vec![n_var.clone()].into(),
  };
  let c = |i: i128| Expr::FunctionCall {
    name: "C".to_string(),
    args: vec![Expr::Integer(i)].into(),
  };

  // (coefficient, optional C[k] factor, basis function)
  let specs = [
    (nums[0], None, "Fibonacci"),
    (nums[1], Some(c(1)), "Fibonacci"),
    (nums[2], None, "LucasL"),
    (nums[3], Some(c(c_idx_b)), "LucasL"),
  ];
  let mut terms = Vec::new();
  for (m, c_factor, basis) in specs {
    if m == 0 {
      continue;
    }
    // A negative coefficient rides on the integer factor (`-5*Fibonacci[n]`,
    // not `-(5*Fibonacci[n])`); only a bare -1 becomes an outer negation.
    let mut factors: Vec<Expr> = Vec::new();
    if m.abs() != 1 {
      factors.push(Expr::Integer(m));
    }
    if let Some(cf) = c_factor {
      factors.push(cf);
    }
    factors.push(base(basis));
    let mut iter = factors.into_iter();
    let mut term = iter.next().unwrap();
    for f in iter {
      term = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(term),
        right: Box::new(f),
      };
    }
    if m == -1 {
      term = Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(term),
      };
    }
    terms.push(term);
  }

  let numerator = if terms.len() == 1 {
    terms.remove(0)
  } else {
    crate::functions::math_ast::plus_ast(&terms).ok()?
  };
  if denom == 1 {
    Some(numerator)
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(Expr::Integer(denom)),
    })
  }
}

/// First-order homogeneous recurrence with a single initial condition.
/// The closed form is `v * r^(n - k)` where `(k, v)` is the IC. wolframscript
/// folds the coefficient's factors of `r` into the exponent, e.g.
/// `a[1]==6, r=2` → `3*2^n` (since `6*2^(n-1) = 3*2^n`) and `a[2]==5, r=2` →
/// `5*2^(-2 + n)`. Restricted to integer roots `r >= 2`; other roots
/// (`1`, `-1`, negatives, rationals) fall through to the generic solver.
fn build_first_order_with_ic(
  root: &(i128, i128),
  ic: &(i128, Expr),
  var_name: &str,
) -> Option<Expr> {
  let (rn, rd) = *root;
  if rd != 1 || rn < 2 {
    return None;
  }
  let (k, v_expr) = ic;
  let mut v = match v_expr {
    Expr::Integer(v) => *v,
    _ => return None,
  };
  if v == 0 {
    return Some(Expr::Integer(0));
  }
  // Factor powers of r out of v, tracking the exponent shift m so that
  // v_original * r^(n - k) == v_reduced * r^(n - k + m).
  let mut m: i128 = 0;
  while v % rn == 0 {
    v /= rn;
    m += 1;
  }
  let off = m - *k;
  let n_var = Expr::Identifier(var_name.to_string());
  // Exponent `n` (off == 0) or `off + n` (constant first, matching display).
  let exponent = if off == 0 {
    n_var
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(off)),
      right: Box::new(n_var),
    }
  };
  let power = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Integer(rn)),
    right: Box::new(exponent),
  };
  let result = if v == 1 {
    power
  } else if v == -1 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(power),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(v)),
      right: Box::new(power),
    }
  };
  Some(result)
}

/// Build the solution when there are fewer initial conditions than
/// recurrence order. The first `free_count = order - ics.len()` constants
/// stay symbolic (as `C[1], C[2], …`) and the remaining constants are
/// expressed as linear combinations of the free constants and the IC
/// values, matching wolframscript's parameterization.
///
/// Currently handles `free_count == 1`, which covers the common
/// 2nd-order-with-one-IC case (e.g. `RSolve[{a[n+2] == a[n], a[0] == 1},
/// a, n]`).
fn build_partial_solution(
  roots: &[(i128, i128)],
  ics: &[(i128, Expr)],
  var_name: &str,
) -> Option<Expr> {
  let order = roots.len();
  let free_count = order - ics.len();
  // Generalising beyond a single free parameter requires solving a
  // multivariate symbolic system; keep the implementation focused on the
  // shapes wolframscript actually surfaces in tests for now.
  if free_count != 1 || order != 2 || ics.len() != 1 {
    return None;
  }

  let (k, v_expr) = &ics[0];
  let v = match v_expr {
    Expr::Integer(n) => *n,
    _ => return None,
  };

  // Only support integer roots for now (rational roots add denominator
  // bookkeeping that the test cases don't exercise).
  let (r1, r1d) = roots[0];
  let (r2, r2d) = roots[1];
  if r1d != 1 || r2d != 1 {
    return None;
  }

  // Equation: c_1 * r_1^k + c_2 * r_2^k = v   (k is the IC index)
  // With c_1 = C[1] free, eliminate c_2:
  //   c_2 = (v - C[1] * r_1^k) / r_2^k
  // General solution: a(n) = c_1*r_1^n + c_2*r_2^n
  //   = C[1] * r_1^n + v * r_2^(n-k) - C[1] * r_1^k * r_2^(n-k)
  //
  // For the common k = 0 case this collapses to:
  //   a(n) = C[1] * r_1^n + v * r_2^n - C[1] * r_2^n
  //
  // Non-zero k would need a rational scaling by r_2^(-k). Skip for now
  // since the under-determined cases the tests cover all use a[0].
  if *k != 0 {
    return None;
  }

  let n_var = Expr::Identifier(var_name.to_string());
  let c1 = Expr::FunctionCall {
    name: "C".to_string(),
    args: vec![Expr::Integer(1)].into(),
  };

  fn root_pow(r: i128, n_var: &Expr) -> Expr {
    if r == 1 {
      Expr::Integer(1)
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Integer(r)),
        right: Box::new(n_var.clone()),
      }
    }
  }

  let r1_pow_n = root_pow(r1, &n_var);
  let r2_pow_n = root_pow(r2, &n_var);

  // Term 1: C[1] * r_1^n   (collapses to just C[1] when r_1 == 1)
  let term1 = if matches!(&r1_pow_n, Expr::Integer(1)) {
    c1.clone()
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(r1_pow_n.clone()),
      right: Box::new(c1.clone()),
    }
  };

  // Term 2: v * r_2^n  (collapsing on v == 1 / -1 / r_2 == 1)
  let term2 = if matches!(&r2_pow_n, Expr::Integer(1)) {
    Expr::Integer(v)
  } else if v == 1 {
    r2_pow_n.clone()
  } else if v == -1 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(r2_pow_n.clone()),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(v)),
      right: Box::new(r2_pow_n.clone()),
    }
  };

  // Term 3: -C[1] * r_2^n
  let neg_c1_r2n = Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand: Box::new(if matches!(&r2_pow_n, Expr::Integer(1)) {
      c1.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(r2_pow_n),
        right: Box::new(c1.clone()),
      }
    }),
  };

  crate::functions::math_ast::plus_ast(&[term1, term2, neg_c1_r2n]).ok()
}

/// Build the general solution `C[1]*r1^n + C[2]*r2^n + ...`. Roots equal to
/// 1 contribute a bare `C[k]` (since `1^n = 1`); negative-1 roots produce
/// `(-1)^n*C[k]`. A root of multiplicity m contributes m terms whose j-th
/// occurrence (j = 0, 1, …) carries the extra factor `n^j`, e.g. a double root
/// 2 gives `2^n*C[1] + 2^n*n*C[2]`.
fn build_general_solution(
  roots: &[(i128, i128)],
  var_name: &str,
) -> Option<Expr> {
  // A first-order recurrence anchors its free constant at n = 1, so its single
  // root carries the exponent n-1 (matching wolframscript, e.g.
  // `a[n] -> 2^(-1 + n)*C[1]`). Higher-order solutions use the exponent n.
  let single_root = roots.len() == 1;
  let mut terms = Vec::new();
  for (i, &(rn, rd)) in roots.iter().enumerate() {
    let const_expr = Expr::FunctionCall {
      name: "C".to_string(),
      args: vec![Expr::Integer((i + 1) as i128)].into(),
    };
    // Occurrence index of this root among the earlier roots — its power of `n`.
    let mult_index = roots[..i].iter().filter(|&&r| r == (rn, rd)).count();

    // Factors in wolframscript's display order: r^n (omitted when r == 1),
    // then n^mult_index (omitted when 0), then the arbitrary constant.
    let mut factors: Vec<Expr> = Vec::new();
    if !(rn == 1 && rd == 1) {
      let root_expr = if rd == 1 {
        Expr::Integer(rn)
      } else {
        crate::functions::math_ast::make_rational(rn, rd)
      };
      let exponent = if single_root {
        // `-1 + n` (constant first) to match wolframscript's display order.
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(Expr::Identifier(var_name.to_string())),
        }
      } else {
        Expr::Identifier(var_name.to_string())
      };
      factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(root_expr),
        right: Box::new(exponent),
      });
    }
    if mult_index >= 1 {
      factors.push(if mult_index == 1 {
        Expr::Identifier(var_name.to_string())
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Identifier(var_name.to_string())),
          right: Box::new(Expr::Integer(mult_index as i128)),
        }
      });
    }
    factors.push(const_expr);

    // Combine left-associatively so the rendered factor order is preserved.
    let mut iter = factors.into_iter();
    let mut term = iter.next().unwrap();
    for f in iter {
      term = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(term),
        right: Box::new(f),
      };
    }
    terms.push(term);
  }
  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return Some(terms.remove(0));
  }
  crate::functions::math_ast::plus_ast(&terms).ok()
}

/// Collect terms of the form coeff * func[var + offset] from an expression.
/// Returns false if the expression contains a term that is not a recognized
/// homogeneous piece (a constant, a free variable term, a non-integer
/// coefficient, …) so the caller can bail instead of solving the wrong
/// recurrence.
fn collect_recurrence_terms(
  expr: &Expr,
  func_name: &str,
  var_name: &str,
  sign: i128,
  terms: &mut Vec<(i128, i128)>,
) -> bool {
  match expr {
    // Direct: a[n + k] or a[n]
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      if let Some(offset) = extract_var_offset(&args[0], var_name) {
        terms.push((offset, sign));
        true
      } else {
        false
      }
    }
    // c * a[n + k] (parsed form)
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_coeff_times_term(left, right, func_name, var_name, sign, terms)
    }
    // c * a[n + k] (evaluated form: Times[c, a[n + k]])
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      collect_coeff_times_term(
        &args[0], &args[1], func_name, var_name, sign, terms,
      )
    }
    // Plus: recurse into sub-terms
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().all(|arg| {
        collect_recurrence_terms(arg, func_name, var_name, sign, terms)
      })
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_recurrence_terms(left, func_name, var_name, sign, terms)
        && collect_recurrence_terms(right, func_name, var_name, sign, terms)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      collect_recurrence_terms(left, func_name, var_name, sign, terms)
        && collect_recurrence_terms(right, func_name, var_name, -sign, terms)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => collect_recurrence_terms(operand, func_name, var_name, -sign, terms),
    // A zero constant contributes nothing (e.g. `a[n+2] - a[n] == 0`)
    Expr::Integer(0) => true,
    _ => false,
  }
}

/// Recognize `c * a[n + k]` with an integer coefficient on either side.
fn collect_coeff_times_term(
  left: &Expr,
  right: &Expr,
  func_name: &str,
  var_name: &str,
  sign: i128,
  terms: &mut Vec<(i128, i128)>,
) -> bool {
  for (coeff_side, call_side) in [(left, right), (right, left)] {
    if let Expr::Integer(c) = coeff_side
      && let Expr::FunctionCall { name, args } = call_side
      && name == func_name
      && args.len() == 1
      && let Some(offset) = extract_var_offset(&args[0], var_name)
    {
      terms.push((offset, sign * c));
      return true;
    }
  }
  false
}

/// Extract offset from expressions like `n`, `n + 2`, `2 + n`
fn extract_var_offset(expr: &Expr, var_name: &str) -> Option<i128> {
  match expr {
    Expr::Identifier(name) if name == var_name => Some(0),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      // n + k or k + n
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Expr::Integer(k) = right.as_ref()
      {
        return Some(*k);
      }
      if let Expr::Identifier(name) = right.as_ref()
        && name == var_name
        && let Expr::Integer(k) = left.as_ref()
      {
        return Some(*k);
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      // Plus[n, k] or Plus[k, n]
      for i in 0..2 {
        if let Expr::Identifier(name) = &args[i]
          && name == var_name
          && let Expr::Integer(k) = &args[1 - i]
        {
          return Some(*k);
        }
      }
      None
    }
    _ => None,
  }
}

/// Find integer/rational roots of the characteristic polynomial
fn find_characteristic_roots(coeffs: &[i128]) -> Option<Vec<(i128, i128)>> {
  // Returns roots as (numerator, denominator) pairs
  let mut remaining = coeffs.to_vec();
  let mut roots = Vec::new();

  while remaining.len() > 1 {
    // Try integer roots using rational root theorem
    let last_nonzero = remaining.iter().rposition(|&c| c != 0)?;
    let leading = remaining[last_nonzero];
    let constant = remaining[0];

    if constant == 0 {
      // x = 0 is a root
      roots.push((0i128, 1i128));
      // Divide by x (shift coefficients)
      remaining = remaining[1..].to_vec();
      continue;
    }

    let lead_divs = divisors(leading.abs());
    let const_divs = divisors(constant.abs());

    let mut found = false;
    for &p in &const_divs {
      for &q in &lead_divs {
        for &sign in &[1i128, -1] {
          let num = sign * p;
          let den = q;
          // Evaluate polynomial at num/den: multiply through by den^n
          let mut val = 0i128;
          let _den_power = 1i128;
          for (i, &c) in remaining.iter().enumerate() {
            let num_power = num.checked_pow(i as u32)?;
            // We need: c * num^i * den^(degree-i)
            let degree = remaining.len() - 1;
            let den_p = den.checked_pow((degree - i) as u32)?;
            val =
              val.checked_add(c.checked_mul(num_power)?.checked_mul(den_p)?)?;
          }
          if val == 0 {
            roots.push((num, den));
            // Synthetic division by (den*x - num)
            remaining = synthetic_divide(&remaining, num, den)?;
            found = true;
            break;
          }
        }
        if found {
          break;
        }
      }
      if found {
        break;
      }
    }

    if !found {
      return None; // Can't find all roots
    }
  }

  Some(roots)
}

/// Synthetic division of polynomial by (den*x - num)
fn synthetic_divide(
  coeffs: &[i128],
  num: i128,
  den: i128,
) -> Option<Vec<i128>> {
  if coeffs.len() <= 1 {
    return Some(vec![]);
  }
  let n = coeffs.len() - 1;
  // Division of sum(c_i * x^i) by (den*x - num)
  // Equivalently, divide by (x - num/den) then divide all coefficients by den
  let mut result = vec![0i128; n];
  result[n - 1] = coeffs[n];
  for i in (0..n - 1).rev() {
    // result[i] = coeffs[i+1] + result[i+1] * num / den
    let prod = result[i + 1].checked_mul(num)?;
    if prod % den != 0 {
      return None; // Not exact division
    }
    result[i] = coeffs[i + 1] + prod / den;
  }
  // Verify: coeffs[0] + result[0] * num / den should be 0
  let check = result[0].checked_mul(num)?;
  if coeffs[0] * den + check != 0 {
    return None;
  }
  Some(result)
}

fn divisors(n: i128) -> Vec<i128> {
  let mut result = Vec::new();
  let mut i = 1i128;
  while i * i <= n {
    if n % i == 0 {
      result.push(i);
      if i != n / i {
        result.push(n / i);
      }
    }
    i += 1;
  }
  result
}

/// Solve for constants in the general solution given initial conditions.
/// General solution: a[n] = c1*r1^n + c2*r2^n + ...
/// Returns coefficients as (numerator, denominator) pairs.
fn solve_initial_conditions(
  roots: &[(i128, i128)],
  ics: &[(i128, Expr)],
) -> Option<Vec<(i128, i128)>> {
  let n = roots.len();
  if ics.len() != n {
    return None;
  }

  // Build matrix: A[i][j] = (roots[j].num / roots[j].den)^ics[i].index
  // Working with rationals: (num/den)^k = num^k / den^k
  // We need to solve A * c = b where b[i] = ics[i].value

  // For simplicity, work with integer matrix by multiplying through by lcm of denominators
  // First, build the rational matrix
  let mut matrix: Vec<Vec<(i128, i128)>> = Vec::new(); // (num, den)
  let mut rhs: Vec<(i128, i128)> = Vec::new();

  for (idx, val) in ics {
    let mut row = Vec::new();
    for (j, &(rnum, rden)) in roots.iter().enumerate() {
      // Occurrence index of this root among the earlier roots — the power of
      // `n` in its basis function `n^mult * r^idx`. A repeated root therefore
      // contributes the independent solutions r^n, n r^n, n^2 r^n, …; without
      // this the matrix has duplicate columns and the system is unsolvable.
      let mult_index =
        roots[..j].iter().filter(|&&r| r == (rnum, rden)).count() as u32;
      let k = *idx as u32;
      let num_k = rnum.checked_pow(k)?;
      let den_k = rden.checked_pow(k)?;
      let idx_pow = (*idx).checked_pow(mult_index)?;
      row.push((num_k.checked_mul(idx_pow)?, den_k));
    }
    matrix.push(row);
    // RHS value must be an integer
    match val {
      Expr::Integer(v) => rhs.push((*v, 1)),
      _ => return None,
    }
  }

  // Gaussian elimination with rational arithmetic
  // Augmented matrix: [A | b]
  let mut aug: Vec<Vec<(i128, i128)>> = matrix
    .iter()
    .zip(rhs.iter())
    .map(|(row, b)| {
      let mut r = row.clone();
      r.push(*b);
      r
    })
    .collect();

  for col in 0..n {
    // Find pivot
    let pivot_row = (col..n).find(|&r| aug[r][col].0 != 0)?;
    aug.swap(col, pivot_row);

    let pivot = aug[col][col];
    for row in (col + 1)..n {
      let factor = aug[row][col];
      if factor.0 == 0 {
        continue;
      }
      for j in col..=n {
        // aug[row][j] -= factor/pivot * aug[col][j]
        // = (aug[row][j] * pivot - factor * aug[col][j]) / pivot
        let (an, ad) = aug[row][j];
        let (bn, bd) = aug[col][j];
        let (fn_, fd) = factor;
        let (pn, pd) = pivot;
        // new = an/ad - (fn/fd)/(pn/pd) * bn/bd
        //     = an/ad - (fn*pd)/(fd*pn) * bn/bd
        //     = an/ad - (fn*pd*bn)/(fd*pn*bd)
        let lhs_n = an * fd * pn * bd;
        let lhs_d = ad * fd * pn * bd;
        let rhs_n = fn_ * pd * bn * ad;
        let new_n = lhs_n - rhs_n;
        let new_d = lhs_d;
        let g = gcd_i128(new_n, new_d);
        aug[row][j] = if g == 0 {
          (0, 1)
        } else {
          let mut nn = new_n / g;
          let mut nd = new_d / g;
          if nd < 0 {
            nn = -nn;
            nd = -nd;
          }
          (nn, nd)
        };
      }
    }
  }

  // Back substitution
  let mut solution = vec![(0i128, 1i128); n];
  for i in (0..n).rev() {
    let (mut sn, mut sd) = aug[i][n]; // RHS
    for j in (i + 1)..n {
      // sn/sd -= aug[i][j] * solution[j]
      let (an, ad) = aug[i][j];
      let (cn, cd) = solution[j];
      let sub_n = an * cn;
      let sub_d = ad * cd;
      // sn/sd - sub_n/sub_d
      sn = sn * sub_d - sub_n * sd;
      sd *= sub_d;
      let g = gcd_i128(sn, sd);
      if g != 0 {
        sn /= g;
        sd /= g;
      }
      if sd < 0 {
        sn = -sn;
        sd = -sd;
      }
    }
    // solution[i] = (sn/sd) / aug[i][i]
    let (pn, pd) = aug[i][i];
    sn *= pd;
    sd *= pn;
    let g = gcd_i128(sn, sd);
    if g != 0 {
      sn /= g;
      sd /= g;
    }
    if sd < 0 {
      sn = -sn;
      sd = -sd;
    }
    solution[i] = (sn, sd);
  }

  Some(solution)
}

/// Build the solution expression: (c1_num*r1^n + c2_num*r2^n + ...) / common_denom
fn build_solution(
  constants: &[(i128, i128)],
  roots: &[(i128, i128)],
  var_name: &str,
) -> Option<Expr> {
  // Find common denominator for all constants
  let mut common_denom = 1i128;
  for &(_, cd) in constants {
    common_denom = lcm_i128(common_denom, cd);
  }

  // Build terms with integer numerators: (cn * common_denom/cd) * r^n
  let mut terms = Vec::new();

  for (i, &(cn, cd)) in constants.iter().enumerate() {
    let num = cn * (common_denom / cd);
    if num == 0 {
      continue;
    }
    let (rn, rd) = roots[i];
    // Occurrence index among earlier equal roots — the power of `n` carried by
    // a repeated root's basis term (e.g. a double root 2 gives 2^n and 2^n n).
    let mult_index = roots[..i].iter().filter(|&&r| r == (rn, rd)).count();

    // Display order matching wolframscript: numeric coefficient, then r^n,
    // then n^mult_index.
    let abs_num = num.abs();
    let mut factors: Vec<Expr> = Vec::new();
    if abs_num != 1 {
      factors.push(Expr::Integer(abs_num));
    }
    if !(rn == 1 && rd == 1) {
      let root_expr = if rd == 1 {
        Expr::Integer(rn)
      } else {
        crate::functions::math_ast::make_rational(rn, rd)
      };
      factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(root_expr),
        right: Box::new(Expr::Identifier(var_name.to_string())),
      });
    }
    if mult_index >= 1 {
      factors.push(if mult_index == 1 {
        Expr::Identifier(var_name.to_string())
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Identifier(var_name.to_string())),
          right: Box::new(Expr::Integer(mult_index as i128)),
        }
      });
    }

    // Combine left-associatively; an empty factor list means the whole term is
    // the bare coefficient (root 1, multiplicity 0).
    let mut term = if factors.is_empty() {
      Expr::Integer(abs_num)
    } else {
      let mut iter = factors.into_iter();
      let mut t = iter.next().unwrap();
      for f in iter {
        t = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(t),
          right: Box::new(f),
        };
      }
      t
    };
    if num < 0 {
      term = Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(term),
      };
    }

    terms.push(term);
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }

  // Build the numerator sum
  let numerator = if terms.len() == 1 {
    terms.remove(0)
  } else {
    crate::functions::math_ast::plus_ast(&terms).ok()?
  };

  // Divide by common denominator if > 1
  if common_denom == 1 {
    Some(numerator)
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(Expr::Integer(common_denom)),
    })
  }
}

/// RecurrenceTable[{recurrence, initial_conditions...}, a, {n, nmin, nmax}]
/// Iteratively evaluate a recurrence relation.
pub fn recurrence_table_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("RecurrenceTable", args);
  if args.len() != 3 {
    return Ok(unevaluated());
  }

  // Extract function name (e.g. "a")
  let func_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated()),
  };

  // Extract the range spec: {n, nmax} (nmin defaults to 1) or {n, nmin, nmax}.
  let (var_name, nmin, nmax) = match &args[2] {
    Expr::List(items) if items.len() == 2 || items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(s) => s.clone(),
        _ => return Ok(unevaluated()),
      };
      let ints: Vec<i128> = items[1..]
        .iter()
        .filter_map(|e| match e {
          Expr::Integer(n) => Some(*n),
          _ => None,
        })
        .collect();
      if ints.len() != items.len() - 1 {
        return Ok(unevaluated());
      }
      if ints.len() == 1 {
        (var, 1, ints[0])
      } else {
        (var, ints[0], ints[1])
      }
    }
    _ => return Ok(unevaluated()),
  };

  // Extract equations from first arg
  let equations = match &args[0] {
    Expr::List(eqs) => eqs.clone(),
    _ => return Ok(unevaluated()),
  };

  // Separate initial conditions from the recurrence
  let mut recurrence_eq: Option<(Expr, Expr)> = None;
  let mut initial_conditions: std::collections::HashMap<i128, Expr> =
    std::collections::HashMap::new();

  for eq in &equations {
    let (lhs, rhs) = match extract_equation(eq) {
      Some(pair) => pair,
      None => return Ok(unevaluated()),
    };

    // Check if this is an initial condition: a[integer] == value
    if let Some((name, idx)) = extract_func_at_integer(&lhs)
      && name == func_name
    {
      let val = crate::evaluator::evaluate_expr_to_expr(&rhs)?;
      initial_conditions.insert(idx, val);
      continue;
    }
    if let Some((name, idx)) = extract_func_at_integer(&rhs)
      && name == func_name
    {
      let val = crate::evaluator::evaluate_expr_to_expr(&lhs)?;
      initial_conditions.insert(idx, val);
      continue;
    }

    // This is the recurrence relation
    if recurrence_eq.is_some() {
      return Ok(unevaluated());
    }
    recurrence_eq = Some((lhs, rhs));
  }

  let (rec_lhs, rec_rhs) = match recurrence_eq {
    Some(r) => r,
    None => return Ok(unevaluated()),
  };

  // Determine which side has the "highest" a[n+k] to solve for
  // Normalize: solve for a[n+k] on the LHS, expression on the RHS
  // Find the highest-offset term on LHS that is just a[n+offset]
  let (target_offset, solve_expr) = if let Some(offset) =
    extract_single_func_offset(&rec_lhs, &func_name, &var_name)
  {
    // LHS is a[n+offset], RHS is the expression
    (offset, rec_rhs)
  } else if let Some(offset) =
    extract_single_func_offset(&rec_rhs, &func_name, &var_name)
  {
    // RHS is a[n+offset], LHS is the expression
    (offset, rec_lhs)
  } else {
    return Ok(unevaluated());
  };

  // Now iterate: for each n from nmin to nmax, compute a[n]
  let mut results = Vec::new();
  let mut values = initial_conditions;

  for n in nmin..=nmax {
    if values.contains_key(&n) {
      results.push(values[&n].clone());
      continue;
    }

    // We need to compute a[n]. The recurrence says:
    // a[var + target_offset] = solve_expr
    // So when var = n - target_offset, we get a[n] = solve_expr(var = n - target_offset)
    let var_val = n - target_offset;

    // Substitute var_name = var_val in solve_expr
    let substituted = crate::syntax::substitute_variable(
      &solve_expr,
      &var_name,
      &Expr::Integer(var_val),
    );

    // Now substitute all a[k] references with known values
    let resolved = substitute_func_values(&substituted, &func_name, &values);
    let val = crate::evaluator::evaluate_expr_to_expr(&resolved)?;
    values.insert(n, val.clone());
    results.push(val);
  }

  Ok(Expr::List(results.into()))
}

/// Extract the offset from an expression like a[n+k] or a[n] or a[n-1]
/// Returns Some(k) if the expression is func_name[var_name + k]
fn extract_single_func_offset(
  expr: &Expr,
  func_name: &str,
  var_name: &str,
) -> Option<i128> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      extract_var_offset(&args[0], var_name)
    }
    _ => None,
  }
}

/// Substitute all occurrences of func_name[integer] with known values
fn substitute_func_values(
  expr: &Expr,
  func_name: &str,
  values: &std::collections::HashMap<i128, Expr>,
) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      // Try to evaluate the argument to an integer
      let evaled_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])
        .unwrap_or(args[0].clone());
      if let Expr::Integer(n) = &evaled_arg
        && let Some(val) = values.get(n)
      {
        return val.clone();
      }
      // Recurse into args
      Expr::FunctionCall {
        name: name.clone(),
        args: args
          .iter()
          .map(|a| substitute_func_values(a, func_name, values))
          .collect(),
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_func_values(a, func_name, values))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| substitute_func_values(a, func_name, values))
        .collect(),
    ),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_func_values(left, func_name, values)),
      right: Box::new(substitute_func_values(right, func_name, values)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_func_values(operand, func_name, values)),
    },
    _ => expr.clone(),
  }
}
