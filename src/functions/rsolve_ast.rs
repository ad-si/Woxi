use super::*;
use crate::functions::math_ast::{gcd_i128, lcm_i128};

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
  // Conditions at any index free of the variable (`f[n] == 1` with a
  // symbolic `n`), for the solvers that take boundary values.
  let mut conditions: Vec<(Expr, Expr)> = Vec::new();
  let symbolic_point = |e: &Expr| -> Option<Expr> {
    match e {
      Expr::FunctionCall { name, args }
        if name == &func_name
          && args.len() == 1
          && !crate::functions::polynomial_ast::contains_var(
            &args[0], &var_name,
          ) =>
      {
        Some(args[0].clone())
      }
      _ => None,
    }
  };

  for eq in &equations {
    // Each equation should be lhs == rhs
    let Some((lhs, rhs)) = extract_equation(eq) else {
      return Ok(unevaluated());
    };
    if let Some(at) = symbolic_point(&lhs) {
      conditions.push((at, rhs.clone()));
    } else if let Some(at) = symbolic_point(&rhs) {
      conditions.push((at, lhs.clone()));
    }
    if extract_func_at_integer(&lhs).is_none()
      && extract_func_at_integer(&rhs).is_none()
      && (symbolic_point(&lhs).is_some() || symbolic_point(&rhs).is_some())
    {
      continue;
    }

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

  let Some((rec_lhs, rec_rhs)) = recurrence else {
    return Ok(unevaluated());
  };
  // A condition at a symbolic index only the symbolic solver can use.
  let symbolic_conditions = conditions.len() > initial_conditions.len();

  if symbolic_conditions {
    return Ok(
      match solve_symbolic_const_coeff_linear(
        &rec_lhs,
        &rec_rhs,
        &func_name,
        &var_name,
        &conditions,
      ) {
        Some(solution) => wrap_rsolve_result(
          solution,
          &func_name,
          &var_name,
          return_as_func_call,
        ),
        None => unevaluated(),
      },
    );
  }

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

  // First-order linear a[n+1] == c a[n] + d with any constant c and d,
  // symbolic ones included.
  if let Some(solution) = solve_first_order_linear(
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

  // Riccati (Möbius) recurrence k0 + k1 a[n] + k2 a[n+1] + k3 a[n] a[n+1]
  // == 0.
  if let Some(solution) = solve_first_order_mobius(
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

  // Homogeneous linear recurrence with symbolic constant coefficients.
  if let Some(solution) = solve_symbolic_const_coeff_linear(
    &rec_lhs,
    &rec_rhs,
    &func_name,
    &var_name,
    &conditions,
  ) {
    return Ok(wrap_rsolve_result(
      solution,
      &func_name,
      &var_name,
      return_as_func_call,
    ));
  }

  // Nonlinear logistic map a[n+1] == 4 a[n] (1 - a[n]).
  if let Some(solution) = solve_logistic_map(
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

/// A homogeneous linear recurrence `Σ c_k a[n + k] == 0` whose coefficients
/// are free of `n` (symbols included). Its characteristic polynomial is
/// factored, the roots taken in `Roots`' order (the reverse of `FactorList`'s
/// factors), and each root ρ of multiplicity m contributes
/// `C[i] ρ^n, C[i+1] n ρ^n, …`. Conditions `a[k] == v` (`k` free of `n`, one
/// per order) fix the constants.
fn solve_symbolic_const_coeff_linear(
  lhs: &Expr,
  rhs: &Expr,
  func_name: &str,
  var_name: &str,
  conditions: &[(Expr, Expr)],
) -> Option<Expr> {
  use crate::evaluator::{evaluate_expr_to_expr, evaluate_function_call_ast};
  let difference = minus2(lhs.clone(), rhs.clone());
  let mut offsets = Vec::new();
  if !all_func_offsets(&difference, func_name, var_name, &mut offsets) {
    return None;
  }
  offsets.sort_unstable();
  offsets.dedup();
  let (lo, hi) = (*offsets.first()?, *offsets.last()?);
  if hi - lo < 1 || hi - lo > 6 {
    return None;
  }
  let at = |k: i128| {
    let index = plus2(id_expr(var_name), Expr::Integer(k));
    call1(func_name, evaluate_expr_to_expr(&index).unwrap_or(index))
  };
  let placeholder = |k: i128| id_expr(&format!("Woxi`RSolveTerm{}", k - lo));
  let rules: Vec<Expr> = offsets
    .iter()
    .map(|&k| rule_expr(at(k), placeholder(k)))
    .collect();
  let replaced = evaluate_expr_to_expr(&call(
    "ReplaceAll",
    vec![difference, Expr::List(rules.into())],
  ))
  .ok()?;
  let expanded = evaluate_function_call_ast("Expand", &[replaced]).ok()?;
  let free = |e: &Expr| {
    !crate::functions::polynomial_ast::contains_var(e, var_name)
      && !contains_func(e, func_name)
  };
  let mut coefficients = Vec::new();
  let mut rest = expanded.clone();
  for k in lo..=hi {
    let c = evaluate_function_call_ast(
      "Coefficient",
      &[expanded.clone(), placeholder(k), Expr::Integer(1)],
    )
    .ok()?;
    if !free(&c) {
      return None;
    }
    rest = evaluate_expr_to_expr(&call(
      "ReplaceAll",
      vec![rest, rule_expr(placeholder(k), Expr::Integer(0))],
    ))
    .ok()?;
    coefficients.push(c);
  }
  // Homogeneous only, and linear in the terms.
  let check = evaluate_function_call_ast(
    "Expand",
    &[minus2(
      expanded,
      plus(
        std::iter::once(rest.clone())
          .chain(
            (lo..=hi)
              .zip(&coefficients)
              .map(|(k, c)| times2(c.clone(), placeholder(k))),
          )
          .collect(),
      ),
    )],
  )
  .ok()?;
  // The equation must be linear in the terms; a constant inhomogeneity
  // (free of n) is allowed.
  if !matches!(check, Expr::Integer(0)) || !free(&rest) {
    return None;
  }
  let r = id_expr("r$RSolve");
  let characteristic = plus(
    coefficients
      .iter()
      .enumerate()
      .map(|(i, c)| times2(c.clone(), pow(r.clone(), Expr::Integer(i as i128))))
      .collect(),
  );
  let factors = evaluate_function_call_ast(
    "FactorList",
    &[evaluate_expr_to_expr(&characteristic).ok()?],
  )
  .ok()?;
  let Expr::List(factors) = &factors else {
    return None;
  };
  let mut roots: Vec<(Expr, i128)> = Vec::new();
  for f in factors.iter().rev() {
    let Expr::List(pair) = f else { return None };
    let [factor, Expr::Integer(m)] = pair.as_slice() else {
      return None;
    };
    if !crate::functions::polynomial_ast::contains_var(factor, "r$RSolve") {
      continue;
    }
    let solved = crate::functions::polynomial_ast::solve_ast(&[
      Expr::Comparison {
        operands: vec![factor.clone(), Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      },
      r.clone(),
    ])
    .ok()?;
    let Expr::List(sols) = &solved else {
      return None;
    };
    for sol in sols {
      let Expr::List(rule) = sol else { return None };
      let [Expr::Rule { replacement, .. }] = rule.as_slice() else {
        return None;
      };
      roots.push(((**replacement).clone(), *m));
    }
  }
  if roots.iter().map(|(_, m)| m).sum::<i128>() != hi - lo {
    return None;
  }
  let n = id_expr(var_name);
  let mut terms = Vec::new();
  // A constant inhomogeneity `rest` has the particular solution A n^m,
  // m the multiplicity of the characteristic root 1.
  if !matches!(rest, Expr::Integer(0)) {
    let m: i128 = roots
      .iter()
      .filter(|(root, _)| matches!(root, Expr::Integer(1)))
      .map(|(_, m)| *m)
      .sum();
    // P(E) n^m = P^(m)(1) for P the characteristic polynomial with the
    // root 1 of multiplicity m, so A = -rest/P(1) for m = 0 and
    // rest/(-P^(m)(1)) otherwise: the arrangement wolframscript's own
    // answers come in (c/(1 - q - s), -(k/(-1 + 2 p))).
    let falling =
      |k: i128, m: i128| -> i128 { (0..m).map(|j| k - j).product() };
    let derivative_at_one = evaluate_function_call_ast(
      "Expand",
      &[plus(
        coefficients
          .iter()
          .enumerate()
          .map(|(i, c)| times2(Expr::Integer(falling(i as i128, m)), c.clone()))
          .collect(),
      )],
    )
    .ok()?;
    if matches!(derivative_at_one, Expr::Integer(0)) {
      return None;
    }
    let amplitude = if m == 0 {
      evaluate_expr_to_expr(&div2(
        negate_expr(rest.clone()),
        derivative_at_one.clone(),
      ))
      .ok()?
    } else {
      let negated = evaluate_function_call_ast(
        "Expand",
        &[negate_expr(derivative_at_one.clone())],
      )
      .ok()?;
      evaluate_expr_to_expr(&div2(rest.clone(), negated)).ok()?
    };
    if !free(&amplitude) {
      return None;
    }
    terms.push(times2(amplitude, pow(n.clone(), Expr::Integer(m))));
  }
  let mut constant = 1;
  for (root, m) in &roots {
    for j in 0..*m {
      let mut factors = vec![
        pow(root.clone(), n.clone()),
        call1("C", Expr::Integer(constant)),
      ];
      if j > 0 {
        factors.push(pow(n.clone(), Expr::Integer(j)));
      }
      terms.push(times(factors));
      constant += 1;
    }
  }
  let general = evaluate_expr_to_expr(&plus(terms)).ok()?;
  if conditions.is_empty() {
    return Some(general);
  }
  if conditions.len() as i128 != hi - lo {
    return None;
  }
  let equations: Vec<Expr> = conditions
    .iter()
    .map(|(k, v)| Expr::Comparison {
      operands: vec![
        crate::syntax::substitute_variable(&general, var_name, k),
        v.clone(),
      ],
      operators: vec![ComparisonOp::Equal],
    })
    .collect();
  let unknowns: Vec<Expr> = (1..constant)
    .map(|i| call1("C", Expr::Integer(i)))
    .collect();
  let solved = crate::functions::polynomial_ast::solve_ast(&[
    Expr::List(equations.into()),
    Expr::List(unknowns.into()),
  ])
  .ok()?;
  let Expr::List(sols) = &solved else {
    return None;
  };
  let first = sols.first()?;
  let particular =
    evaluate_expr_to_expr(&call("ReplaceAll", vec![general, first.clone()]))
      .ok()?;
  // wolframscript factors the common denominator of the combined solution:
  // -(num/((-1 + q^n)*(-1 + 2 p))), not num/(-1 + 2 p + q^n - 2 p q^n).
  let combined = evaluate_function_call_ast("Together", &[particular]).ok()?;
  let numerator =
    evaluate_function_call_ast("Numerator", std::slice::from_ref(&combined))
      .ok()?;
  let denominator =
    evaluate_function_call_ast("Denominator", std::slice::from_ref(&combined))
      .ok()?;
  if matches!(denominator, Expr::Integer(_)) {
    return Some(combined);
  }
  let factored = evaluate_function_call_ast("Factor", &[denominator]).ok()?;
  evaluate_expr_to_expr(&div2(numerator, factored)).ok()
}

/// Collect every offset `k` at which `func[var + k]` occurs in `expr`,
/// pushing into `out`. Returns false if `func` appears with a non-affine
/// argument or wrong arity, so the caller can bail instead of mis-classifying.
fn all_func_offsets(
  expr: &Expr,
  func_name: &str,
  var_name: &str,
  out: &mut Vec<i128>,
) -> bool {
  match expr {
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      match extract_var_offset(&args[0], var_name) {
        Some(o) => {
          out.push(o);
          true
        }
        None => false,
      }
    }
    Expr::FunctionCall { name, .. } if name == func_name => false,
    Expr::FunctionCall { args, .. } => args
      .iter()
      .all(|a| all_func_offsets(a, func_name, var_name, out)),
    Expr::BinaryOp { left, right, .. } => {
      all_func_offsets(left, func_name, var_name, out)
        && all_func_offsets(right, func_name, var_name, out)
    }
    Expr::UnaryOp { operand, .. } => {
      all_func_offsets(operand, func_name, var_name, out)
    }
    Expr::List(items) => items
      .iter()
      .all(|a| all_func_offsets(a, func_name, var_name, out)),
    _ => true,
  }
}

/// True if every occurrence of `func` in `expr` is at index offset 0 (and at
/// least one occurrence exists) — i.e. the expression is `f(a[n])`.
fn side_only_offset_zero(expr: &Expr, func_name: &str, var_name: &str) -> bool {
  let mut offs = Vec::new();
  all_func_offsets(expr, func_name, var_name, &mut offs)
    && !offs.is_empty()
    && offs.iter().all(|&o| o == 0)
}

/// Replace every `func[var]` (offset 0) in `expr` with the placeholder symbol
/// `sym`, leaving the rest of the tree intact.
fn replace_func0_with_symbol(
  expr: &Expr,
  func_name: &str,
  var_name: &str,
  sym: &str,
) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == func_name
        && args.len() == 1
        && extract_var_offset(&args[0], var_name) == Some(0) =>
    {
      id_expr(sym)
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| replace_func0_with_symbol(a, func_name, var_name, sym))
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      replace_func0_with_symbol(left, func_name, var_name, sym),
      replace_func0_with_symbol(right, func_name, var_name, sym),
    ),
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_func0_with_symbol(
        operand, func_name, var_name, sym,
      )),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_func0_with_symbol(a, func_name, var_name, sym))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// If `poly` is `r*a[n]*(1 - a[n])` (in any algebraically equivalent form),
/// return the integer parameter `r`. Works by substituting `a[n] -> y` and
/// checking the polynomial in `y` is exactly `r*y - r*y^2`.
fn logistic_parameter(
  poly: &Expr,
  func_name: &str,
  var_name: &str,
) -> Option<i128> {
  // A placeholder guaranteed distinct from the index and function names.
  let sym = format!("{func_name}$Logistic${var_name}");
  let poly_y = replace_func0_with_symbol(poly, func_name, var_name, &sym);

  let clist = crate::evaluator::evaluate_expr_to_expr(&call(
    "CoefficientList",
    vec![poly_y, Expr::Identifier(sym)],
  ))
  .ok()?;

  let Expr::List(items) = &clist else {
    return None;
  };
  // Need exactly {0, r, -r}: constant 0, linear r, quadratic -r, no higher.
  if items.len() != 3 || !matches!(items[0], Expr::Integer(0)) {
    return None;
  }
  let Expr::Integer(r) = items[1] else {
    return None;
  };
  match items[2] {
    Expr::Integer(c2) if c2 == -r => Some(r),
    _ => None,
  }
}

/// Nonlinear logistic map `a[n+1] == 4 a[n] (1 - a[n])`. Only the fully chaotic
/// case r == 4 has the closed form this solver returns:
///   IC `a[k0] == c`:  a[n] = (1 - Cos[2^(n-k0)*ArcCos[1 - 2 c]]) / 2
///   no IC:            a[n] = 1/2 - Cos[2^n*C[1]] / 2
/// (r == 2 is also solvable in closed form, but wolframscript displays it in a
/// rewritten rational form this solver doesn't reproduce, so it's left
/// unevaluated.) Returns None for any other shape.
fn solve_logistic_map(
  lhs: &Expr,
  rhs: &Expr,
  func_name: &str,
  var_name: &str,
  ics: &[(i128, Expr)],
) -> Option<Expr> {
  // One side must be the bare advanced term a[n + d] (d >= 1); the other must
  // depend on `func` only through offset 0.
  let d_lhs = extract_single_func_offset(lhs, func_name, var_name);
  let d_rhs = extract_single_func_offset(rhs, func_name, var_name);
  let (adv_offset, poly_side) = match (d_lhs, d_rhs) {
    (Some(d), _)
      if d >= 1 && side_only_offset_zero(rhs, func_name, var_name) =>
    {
      (d, rhs)
    }
    (_, Some(d))
      if d >= 1 && side_only_offset_zero(lhs, func_name, var_name) =>
    {
      (d, lhs)
    }
    _ => return None,
  };
  if adv_offset != 1 {
    return None; // only first-order maps a[n+1] = f(a[n])
  }

  if logistic_parameter(poly_side, func_name, var_name)? != 4 {
    return None;
  }

  let n_var = id_expr(var_name);
  let cos = |arg: Expr| call1("Cos", arg);

  match ics.len() {
    0 => {
      // 1/2 - Cos[2^n*C[1]]/2
      let c1 = call1("C", Expr::Integer(1));
      let exp2 = pow2(Expr::Integer(2), n_var);
      let body = times2(
        crate::functions::math_ast::make_rational(-1, 2),
        cos(times2(exp2, c1)),
      );
      Some(plus2(crate::functions::math_ast::make_rational(1, 2), body))
    }
    1 => {
      let (k0, c) = &ics[0];
      // theta = ArcCos[1 - 2 c], evaluated so e.g. 1 - 2*(1/10) -> 4/5.
      let inner = crate::evaluator::evaluate_expr_to_expr(&plus2(
        Expr::Integer(1),
        times2(Expr::Integer(-2), c.clone()),
      ))
      .ok()?;
      let theta =
        crate::evaluator::evaluate_expr_to_expr(&call1("ArcCos", inner))
          .ok()?;
      // Exponent n (k0 == 0) or `-k0 + n` (constant first, matching display).
      let exponent = if *k0 == 0 {
        n_var.clone()
      } else {
        plus2(Expr::Integer(-k0), n_var.clone())
      };
      let exp2 = pow2(Expr::Integer(2), exponent);
      // (1 - Cos[2^(n-k0)*theta]) / 2
      let numerator = plus2(Expr::Integer(1), neg1(cos(times2(exp2, theta))));
      Some(div2(numerator, Expr::Integer(2)))
    }
    _ => None, // over-determined
  }
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
      neg1(e.clone())
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
  let d = crate::evaluator::evaluate_expr_to_expr(&div2(
    neg1(forcing_total),
    Expr::Integer(c_hi),
  ))
  .ok()?;
  if crate::functions::polynomial_ast::contains_var(&d, var_name) {
    return None; // index-dependent forcing — not an arithmetic progression
  }

  let n = id_expr(var_name);
  // d * n
  let dn =
    crate::evaluator::evaluate_expr_to_expr(&times2(d.clone(), n.clone()))
      .ok()?;

  // Build the Plus terms in wolframscript's display order with a raw BinaryOp
  // (the canonical sorter would reorder, e.g. `C[1] + n` instead of `n + C[1]`).
  let raw_plus = |a: Expr, b: Expr| plus2(a, b);
  match ics.len() {
    0 => {
      // d*n + C[1]
      let c1 = call1("C", Expr::Integer(1));
      if matches!(&dn, Expr::Integer(0)) {
        return Some(c1);
      }
      Some(raw_plus(dn, c1))
    }
    1 => {
      // a[n] = v + d*(n - k0) = (v - d*k0) + d*n, with the constant first.
      let (k0, v) = &ics[0];
      let constant = crate::evaluator::evaluate_expr_to_expr(&minus2(
        v.clone(),
        times2(d, Expr::Integer(*k0)),
      ))
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

/// First-order linear recurrence `a[m + 1] == c a[m] + d` whose `c` and `d`
/// are free of the index (they may be symbols: `p[t + 1] == (1 + r) p[t] -
/// k`), in whatever form it is written (`a[n] == c a[n - 1] + d`, `2 a[n + 1]
/// == 6 a[n] + 4`). The closed forms are wolframscript's:
///   no initial condition:  (c^n (1 - c^-n) d)/(c - 1) + c^(n - 1) C[1]
///   a[k0] == v:            Together[c^(n - k0) v + d (c^(n - k0) - 1)/(c - 1)]
/// `None` for anything else, including `c == 1` (an arithmetic progression,
/// handled by [`solve_first_order_arithmetic`]).
/// The Riccati recurrence `k0 + k1 a[n] + k2 a[n+1] + k3 a[n] a[n+1] == 0`
/// (`k3 != 0`), that is the Möbius map `a[n+1] = (p a[n] + q)/(r a[n] + s)`
/// with `p = -k1, q = -k0, r = k3, s = k2`. With `a[n] = y[n+1]/(r y[n]) -
/// s/r` it becomes the linear `y[n+2] - (p + s) y[n+1] + (p s - q r) y[n]
/// == 0`, whose characteristic roots `λ1 != λ2` give
///   a[n] = (λ1^(n+1) (λ1 - s) + C[1] λ2^(n+1) (λ2 - s)) /
///          (r (λ1^(n+1) + C[1] λ2^(n+1))),
/// with `λ1` the root wolframscript's `C[1]` leaves out (see below), fixed
/// by an initial value when one is given.
fn solve_first_order_mobius(
  lhs: &Expr,
  rhs: &Expr,
  func_name: &str,
  var_name: &str,
  ics: &[(i128, Expr)],
) -> Option<Expr> {
  use crate::evaluator::{evaluate_expr_to_expr, evaluate_function_call_ast};
  let eval = |e: Expr| evaluate_expr_to_expr(&e).ok();
  let difference = minus2(lhs.clone(), rhs.clone());
  let mut offsets = Vec::new();
  if !all_func_offsets(&difference, func_name, var_name, &mut offsets) {
    return None;
  }
  offsets.sort_unstable();
  offsets.dedup();
  let [lo, hi] = offsets.as_slice() else {
    return None;
  };
  if hi - lo != 1 {
    return None;
  }
  let at = |k: i128| {
    let index = plus2(id_expr(var_name), Expr::Integer(k));
    call1(func_name, eval(index.clone()).unwrap_or(index))
  };
  let u_hi = id_expr("Woxi`RSolveNext");
  let u_lo = id_expr("Woxi`RSolveThis");
  let replaced = eval(call(
    "ReplaceAll",
    vec![
      difference,
      Expr::List(
        vec![
          rule_expr(at(*hi), u_hi.clone()),
          rule_expr(at(*lo), u_lo.clone()),
        ]
        .into(),
      ),
    ],
  ))?;
  let expanded = evaluate_function_call_ast("Expand", &[replaced]).ok()?;
  let coefficient = |e: &Expr, u: &Expr| {
    evaluate_function_call_ast(
      "Coefficient",
      &[e.clone(), u.clone(), Expr::Integer(1)],
    )
    .ok()
  };
  let at_zero = |e: &Expr, u: &Expr| {
    eval(call(
      "ReplaceAll",
      vec![e.clone(), rule_expr(u.clone(), Expr::Integer(0))],
    ))
  };
  let with_hi = coefficient(&expanded, &u_hi)?;
  let k3 = coefficient(&with_hi, &u_lo)?;
  let k2 = at_zero(&with_hi, &u_lo)?;
  let without_hi = at_zero(&expanded, &u_hi)?;
  let k1 = coefficient(&without_hi, &u_lo)?;
  let k0 = at_zero(&without_hi, &u_lo)?;
  let check = evaluate_function_call_ast(
    "Expand",
    &[minus2(
      expanded.clone(),
      plus(vec![
        k0.clone(),
        times2(k1.clone(), u_lo.clone()),
        times2(k2.clone(), u_hi.clone()),
        times(vec![k3.clone(), u_lo.clone(), u_hi.clone()]),
      ]),
    )],
  )
  .ok()?;
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0))
      || crate::functions::math_ast::try_eval_to_f64(e) == Some(0.0)
  };
  let free = |e: &Expr| {
    !crate::functions::polynomial_ast::contains_var(e, var_name)
      && !contains_func(e, func_name)
      && !crate::functions::polynomial_ast::contains_var(e, "Woxi`RSolveNext")
      && !crate::functions::polynomial_ast::contains_var(e, "Woxi`RSolveThis")
  };
  if !is_zero(&check)
    || is_zero(&k3)
    || ![&k0, &k1, &k2, &k3].iter().all(|k| free(k))
  {
    return None;
  }
  let numeric = [&k0, &k1, &k2, &k3]
    .iter()
    .all(|k| crate::functions::predicate_ast::is_numeric_q(k));
  let (p, q, r, s) = (neg1(k1), neg1(k0), k3, k2);
  // λ² - (p + s) λ + (p s - q r) == 0.
  let trace = plus2(p.clone(), s.clone());
  let det = minus2(times2(p, s.clone()), times2(q, r.clone()));
  let discriminant = eval(minus2(
    pow(trace.clone(), Expr::Integer(2)),
    times2(Expr::Integer(4), det),
  ))?;
  if is_zero(&discriminant) {
    return None;
  }
  let root = call1("Sqrt", discriminant);
  let half = |e: Expr| div2(e, Expr::Integer(2));
  let plus_root = eval(half(plus2(trace.clone(), root.clone())))?;
  let minus_root = eval(half(minus2(trace, root)))?;
  // wolframscript's C[1] goes with the -Sqrt root when the coefficients are
  // numbers and with the +Sqrt root when they are symbols: keep both
  // meanings, so that a C[1] solved for later (once numbers are put in for
  // the symbols) is the same number in both.
  let (l1, l2) = if numeric {
    (plus_root, minus_root)
  } else {
    (minus_root, plus_root)
  };
  let n = id_expr(var_name);
  // The term of each root runs with the base -1/(other root): equal to the
  // root's own power at every integer n (the factor (-λ1 λ2)^(n+1) common to
  // all terms cancels), and wolframscript's continuation to real n — real
  // where the roots are negative, which a power of the roots is not.
  let b1 = eval(div2(Expr::Integer(-1), l2.clone()))?;
  let b2 = eval(div2(Expr::Integer(-1), l1.clone()))?;
  let constant = match ics {
    [] => call1("C", Expr::Integer(1)),
    [(k, v)] => {
      // a[k] = v fixes C[1].
      let kk = Expr::Integer(*k + 1);
      let p1 = pow(b1.clone(), kk.clone());
      let p2 = pow(b2.clone(), kk);
      eval(div2(
        minus2(
          times2(p1.clone(), minus2(l1.clone(), s.clone())),
          times(vec![r.clone(), v.clone(), p1]),
        ),
        minus2(
          times(vec![r.clone(), v.clone(), p2.clone()]),
          times2(p2, minus2(l2.clone(), s.clone())),
        ),
      ))?
    }
    _ => return None,
  };
  // wolframscript's C[1] multiplies the second term at n + 1: the same
  // family, written so that a C[1] found for its solution means the same
  // here.
  let n1 = plus2(n, Expr::Integer(1));
  let p1 = pow(b1, n1.clone());
  let p2 = pow(b2, n1);
  eval(div2(
    plus2(
      times2(p1.clone(), minus2(l1, s.clone())),
      times(vec![constant.clone(), p2.clone(), minus2(l2, s)]),
    ),
    times2(r, plus2(p1, times2(constant, p2))),
  ))
}

fn solve_first_order_linear(
  lhs: &Expr,
  rhs: &Expr,
  func_name: &str,
  var_name: &str,
  ics: &[(i128, Expr)],
) -> Option<Expr> {
  use crate::evaluator::{evaluate_expr_to_expr, evaluate_function_call_ast};
  let eval = |e: Expr| evaluate_expr_to_expr(&e).ok();
  let difference = minus2(lhs.clone(), rhs.clone());
  let mut offsets = Vec::new();
  if !all_func_offsets(&difference, func_name, var_name, &mut offsets) {
    return None;
  }
  offsets.sort_unstable();
  offsets.dedup();
  let [lo, hi] = offsets.as_slice() else {
    return None;
  };
  if hi - lo != 1 {
    return None;
  }
  // The two occurrences as placeholder symbols
  let at = |k: i128| {
    let index = plus2(id_expr(var_name), Expr::Integer(k));
    call1(func_name, eval(index.clone()).unwrap_or(index))
  };
  let u_hi = id_expr("Woxi`RSolveNext");
  let u_lo = id_expr("Woxi`RSolveThis");
  let replaced = eval(call(
    "ReplaceAll",
    vec![
      difference,
      Expr::List(
        vec![
          rule_expr(at(*hi), u_hi.clone()),
          rule_expr(at(*lo), u_lo.clone()),
        ]
        .into(),
      ),
    ],
  ))?;
  let coefficient = |u: &Expr| {
    evaluate_function_call_ast(
      "Coefficient",
      &[replaced.clone(), u.clone(), Expr::Integer(1)],
    )
    .ok()
  };
  let a_hi = coefficient(&u_hi)?;
  let a_lo = coefficient(&u_lo)?;
  let rest = eval(call(
    "ReplaceAll",
    vec![
      replaced.clone(),
      Expr::List(
        vec![
          rule_expr(u_hi.clone(), Expr::Integer(0)),
          rule_expr(u_lo.clone(), Expr::Integer(0)),
        ]
        .into(),
      ),
    ],
  ))?;
  // The equation must be exactly a_hi u_hi + a_lo u_lo + rest
  let check = evaluate_function_call_ast(
    "Expand",
    &[minus2(
      replaced.clone(),
      plus(vec![
        times2(a_hi.clone(), u_hi.clone()),
        times2(a_lo.clone(), u_lo.clone()),
        rest.clone(),
      ]),
    )],
  )
  .ok()?;
  let free = |e: &Expr| {
    !crate::functions::polynomial_ast::contains_var(e, var_name)
      && !contains_func(e, func_name)
      && !crate::functions::polynomial_ast::contains_var(e, "Woxi`RSolveNext")
      && !crate::functions::polynomial_ast::contains_var(e, "Woxi`RSolveThis")
  };
  // With machine coefficients the remainder is a machine zero
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0))
      || crate::functions::math_ast::try_eval_to_f64(e) == Some(0.0)
  };
  if !is_zero(&check)
    || is_zero(&a_hi)
    || !free(&a_hi)
    || !free(&a_lo)
    || !free(&rest)
  {
    return None;
  }
  let c = eval(div2(neg1(a_lo), a_hi.clone()))?;
  let d = eval(div2(neg1(rest), a_hi))?;
  if matches!(c, Expr::Integer(1 | 0)) {
    return None;
  }
  let n = id_expr(var_name);
  let c_minus_1 = plus2(c.clone(), Expr::Integer(-1));
  match ics {
    [] => {
      let homogeneous = times2(
        pow(c.clone(), plus2(Expr::Integer(-1), n.clone())),
        call1("C", Expr::Integer(1)),
      );
      if matches!(d, Expr::Integer(0)) {
        return eval(homogeneous);
      }
      let particular = div2(
        times(vec![
          pow(c.clone(), n.clone()),
          minus2(Expr::Integer(1), pow(c.clone(), neg1(n.clone()))),
          d,
        ]),
        c_minus_1,
      );
      eval(plus2(particular, homogeneous))
    }
    [(k0, v)] => {
      let steps = plus2(n.clone(), Expr::Integer(-k0));
      let power = pow(c.clone(), steps);
      let solution = plus2(
        times2(power.clone(), v.clone()),
        div2(times2(d, plus2(power, Expr::Integer(-1))), c_minus_1),
      );
      evaluate_function_call_ast("Together", &[eval(solution)?]).ok()
    }
    _ => None,
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
    call1(func_name, id_expr(var_name))
  } else {
    id_expr(func_name)
  };
  let replacement = if return_as_func_call {
    solution
  } else {
    call(
      "Function",
      vec![Expr::List(vec![id_expr(var_name)].into()), solution],
    )
  };
  let rule = rule_expr(pattern, replacement);
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
  crate::evaluator::evaluate_expr_to_expr(&call(
    "ReplaceAll",
    vec![args[1].clone(), rule],
  ))
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
/// Reject +/-Infinity.
fn rat_reduce_inf(n: i128, d: i128) -> Option<(i128, i128)> {
  if d == 0 {
    return None;
  }
  Some(rat_reduce(n, d))
}

/// Solve the golden-ratio recurrence (characteristic polynomial r^2 - r - 1,
/// e.g. the Fibonacci recurrence) in the basis wolframscript uses:
/// `a[n] = alpha*Fibonacci[n] + beta*LucasL[n]`.
///
/// With no initial conditions both constants stay free
/// (`C[1]*Fibonacci[n] + C[2]*LucasL[n]`). One IC keeps `alpha = C[1]` free
/// and eliminates `beta`; two ICs at distinct indices determine both
/// constants exactly (rational values allowed).
fn solve_fibonacci_lucas(ics: &[(i128, Expr)], var_name: &str) -> Option<Expr> {
  let ic_value = |e: &Expr| -> Option<(i128, i128)> {
    let evaled =
      crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
    crate::functions::math_ast::expr_to_rational(&evaled)
      .and_then(|(n, d)| rat_reduce_inf(n, d))
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
      let beta_const = rat_reduce_inf(vn, vd.checked_mul(lk)?)?;
      let beta_c1 = rat_reduce_inf(-fk, lk)?;
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
      let alpha = rat_reduce_inf(alpha_num, denom)?;
      let beta = rat_reduce_inf(beta_num, denom)?;
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

  let n_var = id_expr(var_name);
  let base = |name: &str| call1(name, n_var.clone());
  let c = |i: i128| call1("C", Expr::Integer(i));

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
      term = times2(term, f);
    }
    if m == -1 {
      term = neg1(term);
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
    Some(div2(numerator, Expr::Integer(denom)))
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
  let n_var = id_expr(var_name);
  // Exponent `n` (off == 0) or `off + n` (constant first, matching display).
  let exponent = if off == 0 {
    n_var
  } else {
    plus2(Expr::Integer(off), n_var)
  };
  let power = pow2(Expr::Integer(rn), exponent);
  let result = if v == 1 {
    power
  } else if v == -1 {
    neg1(power)
  } else {
    times2(Expr::Integer(v), power)
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

  let n_var = id_expr(var_name);
  let c1 = call1("C", Expr::Integer(1));

  fn root_pow(r: i128, n_var: &Expr) -> Expr {
    if r == 1 {
      Expr::Integer(1)
    } else {
      pow2(Expr::Integer(r), n_var.clone())
    }
  }

  let r1_pow_n = root_pow(r1, &n_var);
  let r2_pow_n = root_pow(r2, &n_var);

  // Term 1: C[1] * r_1^n   (collapses to just C[1] when r_1 == 1)
  let term1 = if matches!(&r1_pow_n, Expr::Integer(1)) {
    c1.clone()
  } else {
    times2(r1_pow_n.clone(), c1.clone())
  };

  // Term 2: v * r_2^n  (collapsing on v == 1 / -1 / r_2 == 1)
  let term2 = if matches!(&r2_pow_n, Expr::Integer(1)) {
    Expr::Integer(v)
  } else if v == 1 {
    r2_pow_n.clone()
  } else if v == -1 {
    neg1(r2_pow_n.clone())
  } else {
    times2(Expr::Integer(v), r2_pow_n.clone())
  };

  // Term 3: -C[1] * r_2^n
  let neg_c1_r2n = neg1(if matches!(&r2_pow_n, Expr::Integer(1)) {
    c1.clone()
  } else {
    times2(r2_pow_n, c1.clone())
  });

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
    let const_expr = call1("C", Expr::Integer((i + 1) as i128));
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
        plus2(Expr::Integer(-1), id_expr(var_name))
      } else {
        id_expr(var_name)
      };
      factors.push(pow2(root_expr, exponent));
    }
    if mult_index >= 1 {
      factors.push(if mult_index == 1 {
        id_expr(var_name)
      } else {
        pow2(id_expr(var_name), Expr::Integer(mult_index as i128))
      });
    }
    factors.push(const_expr);

    // Combine left-associatively so the rendered factor order is preserved.
    let mut iter = factors.into_iter();
    let mut term = iter.next().unwrap();
    for f in iter {
      term = times2(term, f);
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
        let (lhs_n, lhs_d) = (an * fd * pn * bd, ad * fd * pn * bd);
        let rhs_n = fn_ * pd * bn * ad;
        let (new_n, new_d) = (lhs_n - rhs_n, lhs_d);
        let (nn, nd) = rat_reduce(new_n, new_d);
        aug[row][j] = (nn, nd);
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
      let (sub_n, sub_d) = (an * cn, ad * cd);
      // sn/sd - sub_n/sub_d
      (sn, sd) = (sn * sub_d - sub_n * sd, sd * sub_d);
      (sn, sd) = rat_reduce(sn, sd);
    }
    // solution[i] = (sn/sd) / aug[i][i]
    let (pn, pd) = aug[i][i];
    (sn, sd) = rat_reduce(sn * pd, sd * pn);
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
      factors.push(pow2(root_expr, id_expr(var_name)));
    }
    if mult_index >= 1 {
      factors.push(if mult_index == 1 {
        id_expr(var_name)
      } else {
        pow2(id_expr(var_name), Expr::Integer(mult_index as i128))
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
        t = times2(t, f);
      }
      t
    };
    if num < 0 {
      term = neg1(term);
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
    Some(div2(numerator, Expr::Integer(common_denom)))
  }
}

/// `RecurrenceTable[{recurrence, initial conditions…}, a, {n, nmin, nmax}]`,
/// and the coupled form `RecurrenceTable[{…}, {x, y, …}, {n, …}]` where each
/// function has its own recurrence and the table lists one tuple per step.
/// Iteratively evaluates the relations.
pub fn recurrence_table_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("RecurrenceTable", args);
  if args.len() != 3 {
    return Ok(unevaluated());
  }

  // The functions being tabulated: one symbol, or a list of them for a
  // system whose equations refer to each other.
  let func_names: Vec<String> = match &args[1] {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::List(items) if !items.is_empty() => {
      let mut names = Vec::with_capacity(items.len());
      for item in items {
        match item {
          Expr::Identifier(name) => names.push(name.clone()),
          _ => return Ok(unevaluated()),
        }
      }
      names
    }
    _ => return Ok(unevaluated()),
  };
  let is_system = matches!(&args[1], Expr::List(_));

  // Extract the range spec: {n, nmax} (nmin defaults to 1) or {n, nmin, nmax}.
  let (var_name, nmin, nmax) = match &args[2] {
    Expr::List(items) if items.len() == 2 || items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(s) => s.clone(),
        _ => return Ok(unevaluated()),
      };
      let ints: Vec<i128> = items[1..]
        .iter()
        .filter_map(|e| match crate::evaluator::evaluate_expr_to_expr(e) {
          Ok(Expr::Integer(n)) => Some(n),
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

  // Extract equations from first arg: a list of them, or a conjunction.
  let equations = match &args[0] {
    Expr::List(eqs) => eqs.clone(),
    Expr::FunctionCall { name, args: eqs } if name == "And" => eqs.clone(),
    _ => return Ok(unevaluated()),
  };

  // Sort the equations into one recurrence per function and the initial
  // values they start from, both keyed by function name.
  let mut recurrences: std::collections::HashMap<String, (i128, Expr)> =
    std::collections::HashMap::new();
  let mut values: std::collections::HashMap<(String, i128), Expr> =
    std::collections::HashMap::new();

  // A machine real anywhere in the equations makes the whole table machine
  // precision: wolframscript numericizes the initial values too, so
  // `{a[n+1] == 0.5 a[n], a[1] == 1}` starts from `1.`.
  let numericize = equations
    .iter()
    .any(crate::functions::math_ast::numeric_utils::contains_inexact_real);

  for eq in &equations {
    let Some((lhs, rhs)) = extract_equation(eq) else {
      return Ok(unevaluated());
    };

    // An initial condition: f[integer] == value, either way round.
    let initial = extract_func_at_integer(&lhs)
      .filter(|(name, _)| func_names.contains(name))
      .map(|(name, idx)| (name, idx, rhs.clone()))
      .or_else(|| {
        extract_func_at_integer(&rhs)
          .filter(|(name, _)| func_names.contains(name))
          .map(|(name, idx)| (name, idx, lhs.clone()))
      });
    if let Some((name, idx, val)) = initial {
      values.insert((name, idx), val);
      continue;
    }

    // Otherwise a recurrence: the side that is a bare f[var + k] names the
    // function it defines, and the other side is the expression for it.
    // Both sides can look like one — `b[i + 1] == a[i]` — so the equation
    // defines whichever side steps furthest forward, the left one on a tie.
    let solved = func_names
      .iter()
      .flat_map(|name| {
        [
          extract_single_func_offset(&lhs, name, &var_name)
            .map(|off| (name.clone(), off, rhs.clone(), 0u8)),
          extract_single_func_offset(&rhs, name, &var_name)
            .map(|off| (name.clone(), off, lhs.clone(), 1u8)),
        ]
      })
      .flatten()
      .max_by_key(|(_, off, _, side)| (*off, std::cmp::Reverse(*side)))
      .or_else(|| {
        // Neither side is the bare term (`2 y[n + 1] == y[n - 1] + y[n]`):
        // solve the equation for its furthest-forward term, which it has to
        // be linear in.
        func_names.iter().find_map(|name| {
          solve_for_leading_term(&lhs, &rhs, name, &var_name)
            .map(|(off, expr)| (name.clone(), off, expr, 0u8))
        })
      });
    let Some((name, offset, expr, _)) = solved else {
      return recurrence_table_by_solving(args, &func_names, &var_name, nmin, nmax);
    };
    if recurrences.insert(name, (offset, expr)).is_some() {
      return recurrence_table_by_solving(args, &func_names, &var_name, nmin, nmax);
    }
  }

  if recurrences.len() != func_names.len() {
    return recurrence_table_by_solving(args, &func_names, &var_name, nmin, nmax);
  }

  // An initial value may be given by another one (`r[1] == r[0]`): resolve
  // those references before evaluating, and numericizing, the values.
  let keys: Vec<(String, i128)> = values.keys().cloned().collect();
  for _ in 0..keys.len() {
    let mut changed = false;
    for key in &keys {
      let resolved = substitute_func_values(&values[key], &func_names, &values);
      if !crate::evaluator::pattern_matching::expr_equal(
        &resolved,
        &values[key],
      ) {
        values.insert(key.clone(), resolved);
        changed = true;
      }
    }
    if !changed {
      break;
    }
  }
  for key in &keys {
    let mut val = crate::evaluator::evaluate_expr_to_expr(&values[key])?;
    if numericize {
      val = crate::evaluator::evaluate_expr_to_expr(&call("N", vec![val]))?;
    }
    values.insert(key.clone(), val);
  }

  // Step through the range, computing every function at each index. A
  // system's equations refer to the previous step, so one pass in the
  // declared order resolves them. The walk starts at the earliest initial
  // condition even when the table asked for a later window, since the
  // steps in between are what carries the recurrence forward.
  let start = values
    .keys()
    .map(|(_, idx)| *idx)
    .min()
    .map_or(nmin, |first| first.min(nmin));
  let mut results = Vec::new();
  for n in start..=nmax {
    let mut row = Vec::with_capacity(func_names.len());
    for name in &func_names {
      if let Some(known) = values.get(&(name.clone(), n)) {
        row.push(known.clone());
        continue;
      }
      // The recurrence gives f[var + offset]; at var = n - offset it gives
      // f[n].
      let (offset, solve_expr) = &recurrences[name];
      let substituted = crate::syntax::substitute_variable(
        solve_expr,
        &var_name,
        &Expr::Integer(n - offset),
      );
      let resolved = substitute_func_values(&substituted, &func_names, &values);
      let val = crate::evaluator::evaluate_expr_to_expr(&resolved)?;
      values.insert((name.clone(), n), val.clone());
      row.push(val);
    }
    if n < nmin {
      continue;
    }
    results.push(if is_system {
      Expr::List(row.into())
    } else {
      row.into_iter().next().unwrap_or(Expr::Integer(0))
    });
  }

  Ok(Expr::List(results.into()))
}

/// A system `RecurrenceTable` whose equations do not each define one
/// function — a coupled step `y[n + 1] + z[n + 1] == …` alongside an
/// algebraic constraint `y[n] + 2 z[n] == 1`. Every equation is placed so
/// that its furthest-forward term lands on the step being computed, and the
/// resulting equations are solved together for the functions' values there.
fn recurrence_table_by_solving(
  args: &[Expr],
  func_names: &[String],
  var_name: &str,
  nmin: i128,
  nmax: i128,
) -> Result<Expr, InterpreterError> {
  use std::collections::HashMap;
  let unevaluated = || Ok(unevaluated("RecurrenceTable", args));
  if !matches!(&args[1], Expr::List(_)) {
    return unevaluated();
  }
  let equations = match &args[0] {
    Expr::List(eqs) => eqs.clone(),
    Expr::FunctionCall { name, args: eqs } if name == "And" => eqs.clone(),
    _ => return unevaluated(),
  };
  let numericize = equations
    .iter()
    .any(crate::functions::math_ast::numeric_utils::contains_inexact_real);
  fn offsets(e: &Expr, names: &[String], var: &str, out: &mut Vec<i128>) {
    for name in names {
      if let Some(k) = extract_single_func_offset(e, name, var) {
        out.push(k);
      }
    }
    for child in crate::syntax::expr_children(e) {
      offsets(child, names, var, out);
    }
  }
  let mut values: HashMap<(String, i128), Expr> = HashMap::new();
  // (difference lhs - rhs, its furthest-forward offset)
  let mut steps: Vec<(Expr, i128)> = Vec::new();
  for eq in &equations {
    let Some((lhs, rhs)) = extract_equation(eq) else {
      return unevaluated();
    };
    let initial = extract_func_at_integer(&lhs)
      .filter(|(name, _)| func_names.contains(name))
      .map(|(name, idx)| (name, idx, rhs.clone()))
      .or_else(|| {
        extract_func_at_integer(&rhs)
          .filter(|(name, _)| func_names.contains(name))
          .map(|(name, idx)| (name, idx, lhs.clone()))
      });
    if let Some((name, idx, val)) = initial {
      let mut val = crate::evaluator::evaluate_expr_to_expr(&val)?;
      if numericize {
        val = crate::evaluator::evaluate_expr_to_expr(&call("N", vec![val]))?;
      }
      values.insert((name, idx), val);
      continue;
    }
    let mut found = Vec::new();
    offsets(&lhs, func_names, var_name, &mut found);
    offsets(&rhs, func_names, var_name, &mut found);
    let Some(k) = found.iter().max().copied() else {
      return unevaluated();
    };
    steps.push((call("Subtract", vec![lhs, rhs]), k));
  }
  if steps.is_empty() || values.is_empty() {
    return unevaluated();
  }
  let start = values.keys().map(|(_, idx)| *idx).min().unwrap().min(nmin);
  let mut results = Vec::new();
  for n in start..=nmax {
    let missing: Vec<&String> = func_names
      .iter()
      .filter(|name| !values.contains_key(&((*name).clone(), n)))
      .collect();
    if !missing.is_empty() {
      // Each equation instantiated with its leading term at step n.
      let mut eqs = Vec::new();
      for (diff, k) in &steps {
        let at = crate::syntax::substitute_variable(
          diff,
          var_name,
          &Expr::Integer(n - k),
        );
        let resolved = substitute_func_values(&at, func_names, &values);
        let resolved = crate::evaluator::evaluate_expr_to_expr(&resolved)?;
        if matches!(resolved, Expr::Integer(0)) {
          continue;
        }
        eqs.push(call("Equal", vec![resolved, Expr::Integer(0)]));
      }
      let unknowns: Vec<Expr> = missing
        .iter()
        .map(|name| Expr::FunctionCall {
          name: (*name).clone(),
          args: vec![Expr::Integer(n)].into(),
        })
        .collect();
      let solved = crate::functions::solve_ast(&[
        Expr::List(eqs.into()),
        Expr::List(unknowns.clone().into()),
      ])?;
      let Expr::List(solutions) = &solved else {
        return unevaluated();
      };
      let [Expr::List(rules)] = solutions.as_slice() else {
        return unevaluated();
      };
      for (name, unknown) in missing.iter().zip(&unknowns) {
        let Some(value) = rules.iter().find_map(|r| match r {
          Expr::Rule {
            pattern,
            replacement,
          } if crate::evaluator::pattern_matching::expr_equal(
            pattern, unknown,
          ) =>
          {
            Some(replacement.as_ref().clone())
          }
          _ => None,
        }) else {
          return unevaluated();
        };
        values.insert(((*name).clone(), n), value);
      }
    }
    if n < nmin {
      continue;
    }
    let row: Vec<Expr> = func_names
      .iter()
      .map(|name| values[&(name.clone(), n)].clone())
      .collect();
    results.push(Expr::List(row.into()));
  }
  Ok(Expr::List(results.into()))
}

/// Extract the offset from an expression like a[n+k] or a[n] or a[n-1]
/// Returns Some(k) if the expression is func_name[var_name + k]
/// `lhs == rhs` solved for `name[var + k]` with the largest `k` that
/// occurs in it, as `(k, solution)`, when the equation is linear in that
/// term.
fn solve_for_leading_term(
  lhs: &Expr,
  rhs: &Expr,
  name: &str,
  var: &str,
) -> Option<(i128, Expr)> {
  fn offsets(e: &Expr, name: &str, var: &str, out: &mut Vec<i128>) {
    if let Some(k) = extract_single_func_offset(e, name, var) {
      out.push(k);
    }
    for child in crate::syntax::expr_children(e) {
      offsets(child, name, var, out);
    }
  }
  let mut found = Vec::new();
  offsets(lhs, name, var, &mut found);
  offsets(rhs, name, var, &mut found);
  let k = *found.iter().max()?;
  let shift = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![Expr::Identifier(var.to_string()), Expr::Integer(k)].into(),
  };
  let term = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: name.to_string(),
    args: vec![shift].into(),
  })
  .ok()?;
  let marker = Expr::Identifier("RecurrenceTable$lead".to_string());
  let ev = |src: Expr| crate::evaluator::evaluate_expr_to_expr(&src).ok();
  let call = |head: &str, args: Vec<Expr>| Expr::FunctionCall {
    name: head.to_string(),
    args: args.into(),
  };
  let difference = ev(call("Subtract", vec![lhs.clone(), rhs.clone()]))?;
  let poly = ev(call(
    "ReplaceAll",
    vec![difference, call("Rule", vec![term, marker.clone()])],
  ))?;
  let poly = ev(call("Expand", vec![poly]))?;
  if !matches!(
    ev(call(
      "Coefficient",
      vec![poly.clone(), marker.clone(), Expr::Integer(2)]
    ))?,
    Expr::Integer(0)
  ) {
    return None;
  }
  let a = ev(call(
    "Coefficient",
    vec![poly.clone(), marker.clone(), Expr::Integer(1)],
  ))?;
  if matches!(a, Expr::Integer(0)) {
    return None;
  }
  let b = ev(call(
    "ReplaceAll",
    vec![poly, call("Rule", vec![marker, Expr::Integer(0)])],
  ))?;
  let solution = ev(call(
    "Together",
    vec![call(
      "Times",
      vec![
        Expr::Integer(-1),
        b,
        call("Power", vec![a, Expr::Integer(-1)]),
      ],
    )],
  ))?;
  Some((k, solution))
}

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

/// Substitute every `f[integer]` of the tabulated functions with the value
/// already computed for it.
fn substitute_func_values(
  expr: &Expr,
  func_names: &[String],
  values: &std::collections::HashMap<(String, i128), Expr>,
) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if func_names.contains(name) && args.len() == 1 =>
    {
      // Try to evaluate the argument to an integer
      let evaled_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])
        .unwrap_or(args[0].clone());
      if let Expr::Integer(n) = &evaled_arg
        && let Some(val) = values.get(&(name.clone(), *n))
      {
        return val.clone();
      }
      // Recurse into args
      Expr::FunctionCall {
        name: name.clone(),
        args: args
          .iter()
          .map(|a| substitute_func_values(a, func_names, values))
          .collect(),
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_func_values(a, func_names, values))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| substitute_func_values(a, func_names, values))
        .collect(),
    ),
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      substitute_func_values(left, func_names, values),
      substitute_func_values(right, func_names, values),
    ),
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_func_values(operand, func_names, values)),
    },
    _ => expr.clone(),
  }
}

/// `RecurrenceTable[eqns, a, {n, nmin, nmax}, {k, kmin, kmax}, …]`: a
/// recurrence in several indices. The equations are one recurrence
/// `a[n + p, k + q, …] == rhs` and boundary values such as
/// `a[0, k] == f[k]`, whose slots are integers or bare index variables.
/// Each entry is computed recursively from the boundary values (memoized),
/// so values outside the tabulated range are reached through the
/// recurrence as they are needed.
pub fn recurrence_table_multi_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use std::collections::HashMap;
  let unevaluated = || Ok(unevaluated("RecurrenceTable", args));
  let Expr::Identifier(func) = &args[1] else {
    return unevaluated();
  };
  // Iterators {v, max} (from 1) or {v, min, max}.
  let mut iters: Vec<(String, i128, i128)> = Vec::new();
  for spec in &args[2..] {
    let Expr::List(items) = spec else {
      return unevaluated();
    };
    let Some(Expr::Identifier(v)) = items.first() else {
      return unevaluated();
    };
    let mut ints = Vec::new();
    for e in &items[1..] {
      match crate::evaluator::evaluate_expr_to_expr(e)? {
        Expr::Integer(i) => ints.push(i),
        _ => return unevaluated(),
      }
    }
    match ints.as_slice() {
      [max] => iters.push((v.clone(), 1, *max)),
      [min, max] => iters.push((v.clone(), *min, *max)),
      _ => return unevaluated(),
    }
  }
  let dims = iters.len();
  let vars: Vec<&str> = iters.iter().map(|(v, _, _)| v.as_str()).collect();
  let Expr::List(eqs) = &args[0] else {
    return unevaluated();
  };

  // The integer offset of `slot` from the index variable `v`, when the slot
  // is `v + offset`.
  let offset_of =
    |slot: &Expr, v: &str| -> Result<Option<i128>, InterpreterError> {
      let at = |x: i128| {
        crate::evaluator::evaluate_expr_to_expr(
          &crate::syntax::substitute_variable(slot, v, &Expr::Integer(x)),
        )
      };
      Ok(match (at(0)?, at(1)?) {
        (Expr::Integer(a), Expr::Integer(b)) if b - a == 1 => Some(a),
        _ => None,
      })
    };
  enum Slot {
    Fixed(i128),
    Free,
  }
  // (slots, value expression in the index variables)
  let mut boundaries: Vec<(Vec<Slot>, Expr)> = Vec::new();
  // (offsets, rhs)
  let mut recurrence: Option<(Vec<i128>, Expr)> = None;
  let call_slots = |e: &Expr| -> Option<Vec<Expr>> {
    match e {
      Expr::FunctionCall { name, args }
        if name == func && args.len() == dims =>
      {
        Some(args.to_vec())
      }
      _ => None,
    }
  };
  for eq in eqs {
    let Some((lhs, rhs)) = extract_equation(eq) else {
      return unevaluated();
    };
    let mut handled = false;
    for (side, other) in [(&lhs, &rhs), (&rhs, &lhs)] {
      let Some(slots) = call_slots(side) else {
        continue;
      };
      // A boundary: every slot an integer or its own bare index variable.
      let boundary: Option<Vec<Slot>> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| match s {
          Expr::Integer(c) => Some(Slot::Fixed(*c)),
          Expr::Identifier(v) if v == vars[i] => Some(Slot::Free),
          _ => None,
        })
        .collect();
      if let Some(b) = boundary
        && b.iter().any(|s| matches!(s, Slot::Fixed(_)))
      {
        boundaries.push((b, other.clone()));
        handled = true;
        break;
      }
      let mut offsets = Vec::with_capacity(dims);
      for (i, s) in slots.iter().enumerate() {
        match offset_of(s, vars[i])? {
          Some(o) => offsets.push(o),
          None => break,
        }
      }
      if offsets.len() == dims && recurrence.is_none() {
        recurrence = Some((offsets, other.clone()));
        handled = true;
        break;
      }
    }
    if !handled {
      return unevaluated();
    }
  }
  let Some((offsets, rhs)) = recurrence else {
    return unevaluated();
  };

  struct Ctx<'a> {
    func: &'a str,
    vars: &'a [&'a str],
    boundaries: &'a [(Vec<Slot>, Expr)],
    offsets: &'a [i128],
    rhs: &'a Expr,
    memo: HashMap<Vec<i128>, Expr>,
    depth: usize,
  }
  fn value(
    ctx: &mut Ctx,
    idx: &[i128],
  ) -> Result<Option<Expr>, InterpreterError> {
    if let Some(v) = ctx.memo.get(idx) {
      return Ok(Some(v.clone()));
    }
    if ctx.depth > 2000 {
      return Ok(None);
    }
    let bindings: Vec<(String, Expr)>;
    let template: Expr;
    if let Some((_, val)) = ctx.boundaries.iter().find(|(slots, _)| {
      slots
        .iter()
        .zip(idx)
        .all(|(s, i)| !matches!(s, Slot::Fixed(c) if c != i))
    }) {
      bindings = ctx
        .vars
        .iter()
        .zip(idx)
        .map(|(v, i)| (v.to_string(), Expr::Integer(*i)))
        .collect();
      template = val.clone();
    } else {
      bindings = ctx
        .vars
        .iter()
        .zip(idx.iter().zip(ctx.offsets))
        .map(|(v, (i, o))| (v.to_string(), Expr::Integer(i - o)))
        .collect();
      template = ctx.rhs.clone();
    }
    let refs: Vec<(&str, &Expr)> =
      bindings.iter().map(|(n, e)| (n.as_str(), e)).collect();
    let instantiated = crate::syntax::substitute_variables(&template, &refs);
    ctx.depth += 1;
    let replaced = replace_calls(ctx, &instantiated);
    ctx.depth -= 1;
    let Some(replaced) = replaced? else {
      return Ok(None);
    };
    let v = crate::evaluator::evaluate_expr_to_expr(&replaced)?;
    ctx.memo.insert(idx.to_vec(), v.clone());
    Ok(Some(v))
  }
  // Replace every call of the function by its computed value.
  fn replace_calls(
    ctx: &mut Ctx,
    e: &Expr,
  ) -> Result<Option<Expr>, InterpreterError> {
    match e {
      Expr::FunctionCall { name, args }
        if name == ctx.func && args.len() == ctx.vars.len() =>
      {
        let mut idx = Vec::with_capacity(args.len());
        for a in args {
          match crate::evaluator::evaluate_expr_to_expr(a)? {
            Expr::Integer(i) => idx.push(i),
            _ => return Ok(None),
          }
        }
        value(ctx, &idx)
      }
      Expr::FunctionCall { name, args } => {
        let mut out = Vec::with_capacity(args.len());
        for a in args {
          match replace_calls(ctx, a)? {
            Some(x) => out.push(x),
            None => return Ok(None),
          }
        }
        Ok(Some(Expr::FunctionCall {
          name: name.clone(),
          args: out.into(),
        }))
      }
      Expr::List(items) => {
        let mut out = Vec::with_capacity(items.len());
        for a in items {
          match replace_calls(ctx, a)? {
            Some(x) => out.push(x),
            None => return Ok(None),
          }
        }
        Ok(Some(Expr::List(out.into())))
      }
      other => {
        // Operators kept in their own node types are evaluated whole.
        let ev = crate::evaluator::evaluate_expr_to_expr(other)?;
        if matches!(&ev, Expr::FunctionCall { .. } | Expr::List(_)) {
          return replace_calls(ctx, &ev);
        }
        Ok(Some(ev))
      }
    }
  }
  let mut ctx = Ctx {
    func,
    vars: &vars,
    boundaries: &boundaries,
    offsets: &offsets,
    rhs: &rhs,
    memo: HashMap::new(),
    depth: 0,
  };
  fn table(
    ctx: &mut Ctx,
    iters: &[(String, i128, i128)],
    idx: &mut Vec<i128>,
  ) -> Result<Option<Expr>, InterpreterError> {
    let level = idx.len();
    if level == iters.len() {
      return value(ctx, idx);
    }
    let (_, lo, hi) = iters[level];
    let mut row = Vec::new();
    for i in lo..=hi {
      idx.push(i);
      let v = table(ctx, iters, idx)?;
      idx.pop();
      match v {
        Some(v) => row.push(v),
        None => return Ok(None),
      }
    }
    Ok(Some(Expr::List(row.into())))
  }
  match table(&mut ctx, &iters, &mut Vec::new())? {
    Some(t) => Ok(t),
    None => unevaluated(),
  }
}
