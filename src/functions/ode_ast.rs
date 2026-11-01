//! AST-native ODE solving functions (DSolve, NDSolve).
//!
//! DSolve solves ordinary differential equations symbolically.
//! NDSolve solves initial-value problems numerically using RK4.

use super::*;
use crate::functions::calculus_ast::differentiate_expr;
use crate::functions::math_ast::{make_sqrt, rat_reduce};
use crate::syntax::{expr_children, map_children};

mod complex_ode;
mod complex_pde;
mod curve;
mod dde;
mod stiff;

// ─── DSolve ────────────────────────────────────────────────────────────

/// DSolve[eqn, y[x], x] or DSolve[{eqn, ic1, ...}, y[x], x]
/// Also DSolve[eqn, y, x] (returns Function form)
pub fn dsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  dsolve_ast_with_head(args, "DSolve")
}

/// Same as [`dsolve_ast`], but an unevaluated result is returned under `head`.
/// `DSolveValue` delegates here, and an equation neither can solve must come
/// back as `DSolveValue[…]`, not as the `DSolve[…]` it delegated to.
pub fn dsolve_ast_with_head(
  args: &[Expr],
  head: &str,
) -> Result<Expr, InterpreterError> {
  // An ODE Woxi can't classify/solve should stay unevaluated (like
  // wolframscript for genuinely unsolvable equations) rather than leaking an
  // internal "DSolve: …" error to the user.
  // Nested lists of equations (`{{eqns…}, {conditions…}}`) are one system.
  let flat_args;
  let solved_args = match args.first() {
    Some(Expr::List(items))
      if items.iter().any(|i| matches!(i, Expr::List(_))) =>
    {
      fn flatten(e: &Expr, out: &mut Vec<Expr>) {
        match e {
          Expr::List(items) => items.iter().for_each(|i| flatten(i, out)),
          other => out.push(other.clone()),
        }
      }
      let mut eqns = Vec::new();
      flatten(&args[0], &mut eqns);
      flat_args = std::iter::once(Expr::List(eqns.into()))
        .chain(args[1..].iter().cloned())
        .collect::<Vec<_>>();
      &flat_args[..]
    }
    _ => args,
  };
  if let Some(result) = dsolve_single_variable_pde(solved_args)? {
    return Ok(result);
  }
  let result = match dsolve_ast_inner(solved_args) {
    Err(InterpreterError::EvaluationError(msg))
      if msg.starts_with("DSolve:") =>
    {
      Ok(unevaluated("DSolve", args))
    }
    // Left unevaluated, the call keeps the equations as written.
    Ok(Expr::FunctionCall { ref name, .. }) if name == "DSolve" => {
      Ok(unevaluated("DSolve", args))
    }
    other => other,
  };
  result.map(|expr| retag_unevaluated(expr, "DSolve", head))
}

/// A PDE in which the unknown is differentiated with respect to one variable
/// only is an ODE in that variable with the others as parameters:
/// `D[u[x, y], x] == x y` gives `u -> x^2 y/2 + C[1][y]`, every constant of
/// integration becoming a function of the other variables. `None` when the
/// equations are not of that shape (or carry conditions).
fn dsolve_single_variable_pde(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let [eqns, target, Expr::List(vars)] = args else {
    return Ok(None);
  };
  if vars.len() < 2 || !vars.iter().all(|v| matches!(v, Expr::Identifier(_))) {
    return Ok(None);
  }
  let (fname, as_function) = match target {
    Expr::Identifier(f) => (f.clone(), true),
    Expr::FunctionCall { name, args: targs }
      if targs.len() == vars.len()
        && targs.iter().zip(vars.iter()).all(|(a, v)| {
          matches!((a, v), (Expr::Identifier(a), Expr::Identifier(v)) if a == v)
        }) =>
    {
      (name.clone(), false)
    }
    _ => return Ok(None),
  };
  let same_args = |a: &[Expr]| {
    a.len() == vars.len()
      && a.iter().zip(vars.iter()).all(|(x, v)| {
        matches!((x, v), (Expr::Identifier(x), Expr::Identifier(v)) if x == v)
      })
  };
  // Every occurrence is `u[vars…]` or `Derivative[…][u][vars…]` with one
  // common differentiated position.
  fn scan(
    e: &Expr,
    fname: &str,
    same_args: &dyn Fn(&[Expr]) -> bool,
    axis: &mut Option<usize>,
  ) -> bool {
    match e {
      Expr::FunctionCall { name, args } if name == fname => same_args(args),
      Expr::CurriedCall { func, args } => {
        if let Expr::CurriedCall {
          func: d,
          args: inner,
        } = func.as_ref()
          && matches!(inner.as_slice(), [Expr::Identifier(n)] if n == fname)
          && let Expr::FunctionCall { name, args: orders } = d.as_ref()
          && name == "Derivative"
        {
          if !same_args(args) {
            return false;
          }
          let nonzero: Vec<usize> = orders
            .iter()
            .enumerate()
            .filter(|(_, o)| !matches!(o, Expr::Integer(0)))
            .map(|(i, _)| i)
            .collect();
          if nonzero.len() != 1
            || !matches!(orders[nonzero[0]], Expr::Integer(_))
          {
            return false;
          }
          match axis {
            Some(a) if *a != nonzero[0] => return false,
            _ => *axis = Some(nonzero[0]),
          }
          return true;
        }
        scan(func, fname, same_args, axis)
          && args.iter().all(|a| scan(a, fname, same_args, axis))
      }
      Expr::Identifier(n) if n == fname => false,
      other => expr_children(other)
        .into_iter()
        .all(|c| scan(c, fname, same_args, axis)),
    }
  }
  let mut axis = None;
  if !scan(eqns, &fname, &same_args, &mut axis) {
    return Ok(None);
  }
  let Some(axis) = axis else {
    return Ok(None);
  };
  let x = vars[axis].clone();
  let g = format!("{fname}$pde");
  // u[vars…] → g[x] and Derivative[…, n, …][u][vars…] → Derivative[n][g][x].
  fn reduce(e: &Expr, fname: &str, g: &str, axis: usize, x: &Expr) -> Expr {
    match e {
      Expr::FunctionCall { name, .. } if name == fname => {
        call(g, vec![x.clone()])
      }
      Expr::CurriedCall { func, .. }
        if matches!(func.as_ref(), Expr::CurriedCall { args: inner, .. }
          if matches!(inner.as_slice(), [Expr::Identifier(n)] if n == fname)) =>
      {
        let Expr::CurriedCall { func: d, .. } = func.as_ref() else {
          unreachable!()
        };
        let Expr::FunctionCall { args: orders, .. } = d.as_ref() else {
          unreachable!()
        };
        Expr::CurriedCall {
          func: Box::new(Expr::CurriedCall {
            func: Box::new(call("Derivative", vec![orders[axis].clone()])),
            args: vec![Expr::Identifier(g.to_string())],
          }),
          args: vec![x.clone()],
        }
      }
      other => map_children(other, &|c| reduce(c, fname, g, axis, x)),
    }
  }
  let reduced = reduce(eqns, &fname, &g, axis, &x);
  let solved = dsolve_ast(&[reduced, call(&g, vec![x.clone()]), x.clone()])?;
  let Expr::List(solutions) = &solved else {
    return Ok(None);
  };
  let others: Vec<Expr> = vars
    .iter()
    .enumerate()
    .filter(|(i, _)| *i != axis)
    .map(|(_, v)| v.clone())
    .collect();
  // C[k] → C[k][other variables].
  fn lift_constants(e: &Expr, others: &[Expr]) -> Expr {
    match e {
      Expr::FunctionCall { name, args }
        if name == "C"
          && args.len() == 1
          && matches!(args[0], Expr::Integer(_)) =>
      {
        Expr::CurriedCall {
          func: Box::new(e.clone()),
          args: others.to_vec(),
        }
      }
      other => map_children(other, &|c| lift_constants(c, others)),
    }
  }
  let mut out = Vec::with_capacity(solutions.len());
  for sol in solutions {
    let Expr::List(rules) = sol else {
      return Ok(None);
    };
    let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
      return Ok(None);
    };
    let value = crate::evaluator::evaluate_expr_to_expr(&lift_constants(
      replacement,
      &others,
    ))?;
    let rule = if as_function {
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(fname.clone())),
        replacement: Box::new(call(
          "Function",
          vec![Expr::List(vars.clone()), value],
        )),
      }
    } else {
      Expr::Rule {
        pattern: Box::new(target.clone()),
        replacement: Box::new(value),
      }
    };
    out.push(Expr::List(vec![rule].into()));
  }
  Ok(Some(Expr::List(out.into())))
}

/// Rewrite the head of an unevaluated `solver[args…]` result, leaving genuine
/// solutions untouched.
fn retag_unevaluated(expr: Expr, from: &str, to: &str) -> Expr {
  if from != to
    && let Expr::FunctionCall { name, args } = &expr
    && name == from
  {
    return Expr::FunctionCall {
      name: to.to_string(),
      args: args.clone(),
    };
  }
  expr
}

/// The name of the (first) unknown function in DSolve's second argument:
/// `y`, `y[x]`, or the first of a list of them.
fn dsolve_function_name(dep: &Expr) -> Option<String> {
  match dep {
    Expr::Identifier(name) => Some(name.clone()),
    Expr::FunctionCall { name, .. } => Some(name.clone()),
    Expr::List(items) => items.first().and_then(dsolve_function_name),
    _ => None,
  }
}

/// Rewrite each equation `lhs == rhs` whose common denominator involves the
/// unknown function `fname` in explicit form, solved for its highest
/// derivative: `u'[x]/u[x] == x` becomes `u'[x] == x u[x]`. The classifiers
/// read explicit and polynomial forms; substituting into the quotient gave
/// 0/0. Equations already explicit in that derivative are left alone.
fn clear_unknown_denominators(
  eqns: &Expr,
  fname: &str,
) -> Result<Expr, InterpreterError> {
  if let Expr::List(items) = eqns {
    Ok(Expr::List(
      items
        .iter()
        .map(|e| clear_unknown_denominators(e, fname))
        .collect::<Result<Vec<_>, _>>()?
        .into(),
    ))
  } else {
    let Some((lhs, rhs)) = as_equal_pair(eqns) else {
      return Ok(eqns.clone());
    };
    let order = max_derivative_order(eqns, fname);
    if order == 0 {
      return Ok(eqns.clone());
    }
    let ev = crate::evaluator::evaluate_expr_to_expr;
    let together = ev(&call(
      "Together",
      vec![call(
        "Plus",
        vec![
          lhs.clone(),
          call("Times", vec![Expr::Integer(-1), rhs.clone()]),
        ],
      )],
    ))?;
    let denominator = ev(&call("Denominator", vec![together]))?;
    if !mentions_symbol(&denominator, fname) {
      return Ok(eqns.clone());
    }
    // The highest derivative `Derivative[order][fname][x]` as it appears.
    fn find_derivative(e: &Expr, fname: &str, order: usize) -> Option<Expr> {
      if let Some((k, _)) = extract_derivative_order_and_point(e, fname)
        && k == order
      {
        return Some(e.clone());
      }
      expr_children(e)
        .into_iter()
        .find_map(|c| find_derivative(c, fname, order))
    }
    let Some(top) = find_derivative(eqns, fname, order) else {
      return Ok(eqns.clone());
    };
    let solved = crate::functions::solve_ast(&[eqns.clone(), top.clone()])?;
    if let Expr::List(sols) = &solved
      && let [Expr::List(rules)] = sols.as_slice()
      && let [Expr::Rule { replacement, .. }] = rules.as_slice()
    {
      return Ok(Expr::Comparison {
        operands: vec![top, replacement.as_ref().clone()],
        operators: vec![ComparisonOp::Equal],
      });
    }
    Ok(eqns.clone())
  }
}

fn dsolve_ast_inner(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("DSolve", args));
  }

  let dep_arg = &args[1];
  let indep_var = &args[2];

  // An ODE with the unknown in a denominator (`u'[x]/u[x] == x`,
  // `-k u'[x]/u''[x] == x`) is solved as its numerator over a common
  // denominator, `u'[x] - x u[x] == 0`: the classifiers below read
  // polynomial forms, and substituting into the quotient gave 0/0.
  let cleared;
  let eqns_arg = if matches!(indep_var, Expr::Identifier(_))
    && let Some(fname) = dsolve_function_name(dep_arg)
  {
    cleared = clear_unknown_denominators(&args[0], &fname)?;
    &cleared
  } else {
    &args[0]
  };

  // Purely algebraic case: when the equation(s) contain no derivatives of the
  // dependent function, DSolve reduces to Solve for the dependent function(s),
  // e.g. DSolve[y[x] + 2 == 5, y[x], x] -> {{y[x] -> 3}}. Delegate to Solve,
  // which treats the applications `y[x]` as generalized unknowns.
  if !contains_derivative(eqns_arg) {
    let solved =
      crate::functions::solve_ast(&[eqns_arg.clone(), dep_arg.clone()])?;
    // Only accept a genuine solution list; otherwise fall through so the ODE
    // machinery (or the unevaluated fallback) handles it.
    if matches!(&solved, Expr::List(_)) {
      return Ok(solved);
    }
  }

  // PDE branch: `DSolve[eqn, f, {x, y}]` (or `DSolve[eqn, f[x, y], {x, y}]`)
  // for a first-order linear PDE in two variables. Three recognised
  // shapes:
  //   * Constant coefficients with f-divided derivatives:
  //       a*D[f,x]/f + b*D[f,y]/f == c
  //     Solution: f -> Function[{x, y}, E^((c/a)*x) * C[1][y - (b/a)*x]]
  //   * Constant coefficients on bare derivatives, RHS constant:
  //       a*D[f,x] + b*D[f,y] == c
  //     Solution: f[x, y] -> (c/a)*x + C[1][y - (b/a)*x]
  //   * Euler-type with the variable as coefficient:
  //       x*D[f,x] + y*D[f,y] == c
  //     Solution: f[x, y] -> c*Log[x] + C[1][y/x]
  // The output rule's LHS depends on whether `dep_arg` is `f` (Function
  // form) or `f[x, y]` (rule on the call form).
  if let Expr::List(vars) = indep_var
    && vars.len() == 2
    && let (Expr::Identifier(xn), Expr::Identifier(yn)) = (&vars[0], &vars[1])
  {
    let (fname_opt, return_call_form) = match dep_arg {
      Expr::Identifier(name) => (Some(name.clone()), false),
      Expr::FunctionCall { name, args: fargs }
        if fargs.len() == 2
          && matches!(&fargs[0], Expr::Identifier(s) if s == xn)
          && matches!(&fargs[1], Expr::Identifier(s) if s == yn) =>
      {
        (Some(name.clone()), true)
      }
      _ => (None, false),
    };
    if let Some(fname) = fname_opt {
      if let Some(body) =
        try_linear_first_order_pde_body(eqns_arg, &fname, xn, yn)
      {
        return Ok(wrap_pde_solution(body, &fname, xn, yn, return_call_form));
      }
      if let Some(body) = try_direct_linear_pde_body(eqns_arg, &fname, xn, yn) {
        return Ok(wrap_pde_solution(body, &fname, xn, yn, return_call_form));
      }
      if let Some(body) = try_euler_pde_body(eqns_arg, &fname, xn, yn) {
        return Ok(wrap_pde_solution(body, &fname, xn, yn, return_call_form));
      }
      if let Some(body) =
        try_second_order_constant_pde_body(eqns_arg, &fname, xn, yn)
      {
        return Ok(wrap_pde_solution(body, &fname, xn, yn, return_call_form));
      }
    }
  }

  // Extract independent variable name
  let x_name = match indep_var {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("DSolve", args));
    }
  };

  // A list naming one function is that function: the single-equation
  // solver below covers far more than the linear-system one.
  if let Expr::List(dep_items) = dep_arg
    && let [single] = dep_items.as_slice()
  {
    let mut unwrapped = args.to_vec();
    unwrapped[1] = single.clone();
    let solved = dsolve_ast_inner(&unwrapped)?;
    if !matches!(&solved, Expr::FunctionCall { name, .. } if name == "DSolve") {
      return Ok(solved);
    }
  }

  // `DSolve[{…}, {y1[t], y2[t], …}, t]` — a coupled system of dependent
  // functions, handled by the operator-elimination solver above instead of
  // the single-variable machinery below.
  // A bare function name (`{x, y}`) asks for `x -> Function[{t}, …]`.
  if let Expr::List(dep_items) = dep_arg {
    let function_names: Vec<String> = dep_items
      .iter()
      .filter_map(|item| match item {
        Expr::Identifier(name) => Some(name.clone()),
        _ => None,
      })
      .collect();
    let items: Vec<Expr> = dep_items
      .iter()
      .map(|item| match item {
        Expr::Identifier(name) => Expr::FunctionCall {
          name: name.clone(),
          args: vec![Expr::Identifier(x_name.clone())].into(),
        },
        other => other.clone(),
      })
      .collect();
    // A system whose equations each involve a single unknown splits into
    // independent ODEs; otherwise the linear-system solver takes it. A
    // system that solver cannot handle (variable or symbolic coefficients,
    // non-constant forcing) stays unevaluated.
    let solved = if let Some(solved) =
      dsolve_eliminating_algebraic(eqns_arg, &items, &x_name)?
    {
      solved
    } else if let Some(solved) =
      dsolve_decoupled_system(eqns_arg, &items, &x_name)?
    {
      solved
    } else {
      match dsolve_linear_system(eqns_arg, &items, &x_name) {
        Ok(solved) => solved,
        Err(_) => return Ok(unevaluated("DSolve", args)),
      }
    };
    if function_names.is_empty() {
      return Ok(solved);
    }
    let to_function_form = |rule: &Expr| -> Expr {
      if let Expr::Rule {
        pattern,
        replacement,
      } = rule
        && let Expr::FunctionCall { name, .. } = pattern.as_ref()
        && function_names.contains(name)
      {
        return function_form_rule(
          name.clone(),
          x_name.clone(),
          replacement.as_ref().clone(),
        );
      }
      rule.clone()
    };
    return Ok(match &solved {
      Expr::List(solutions) => Expr::List(
        solutions
          .iter()
          .map(|solution| match solution {
            Expr::List(rules) => {
              Expr::List(rules.iter().map(to_function_form).collect())
            }
            other => other.clone(),
          })
          .collect(),
      ),
      other => other.clone(),
    });
  }

  // Determine dependent function name and whether Function form is requested
  let (y_name, function_form) = match dep_arg {
    // y[x] form → return y[x] -> expr
    Expr::FunctionCall { name, args: fargs } if fargs.len() == 1 => {
      if let Expr::Identifier(xn) = &fargs[0] {
        if xn == &x_name {
          (name.clone(), false)
        } else {
          return Ok(unevaluated("DSolve", args));
        }
      } else {
        return Ok(unevaluated("DSolve", args));
      }
    }
    // y form → return y -> Function[{x}, expr]
    Expr::Identifier(name) => (name.clone(), true),
    _ => {
      return Ok(unevaluated("DSolve", args));
    }
  };

  // Separate equations and initial conditions
  let (ode_expr, initial_conditions) = match eqns_arg {
    Expr::List(items) => {
      // First item should be the ODE, rest are initial conditions
      if items.is_empty() {
        return Ok(unevaluated("DSolve", args));
      }
      let mut ics = Vec::new();
      let mut ode = None;
      for item in items {
        if is_initial_condition(item, &y_name, &x_name) {
          ics.push(item.clone());
        } else {
          if ode.is_some() {
            // Multiple ODEs not supported
            return Ok(unevaluated("DSolve", args));
          }
          ode = Some(item.clone());
        }
      }
      match ode {
        Some(o) => (o, ics),
        None => {
          return Ok(unevaluated("DSolve", args));
        }
      }
    }
    // Single equation, no ICs
    _ => (eqns_arg.clone(), Vec::new()),
  };

  // Parse the ODE: extract lhs == rhs, move everything to lhs - rhs = 0
  let ode_normalized = normalize_equation(&ode_expr)?;

  // `y'' + (α + β Cos[ω x + φ]) y == 0` is Mathieu's equation.
  if initial_conditions.is_empty()
    && let Some(solution) =
      dsolve_mathieu_equation(&ode_normalized, &y_name, &x_name)
  {
    return Ok(build_dsolve_result(solution, y_name, x_name, function_form));
  }

  // Collect terms: classify each additive term by derivative order
  // A product such as `k*(a - y[x] + f[x])` hides the linear terms inside a
  // parenthesised sum, so retry on the expanded form before giving up.
  let collected =
    collect_ode_terms(&ode_normalized, &y_name, &x_name).or_else(|err| {
      let expanded =
        crate::functions::polynomial_ast::expand_and_combine(&ode_normalized);
      collect_ode_terms(&expanded, &y_name, &x_name).map_err(|_| err)
    });
  let terms = match collected {
    Ok(terms) => terms,
    // The term classifier only understands equations that are linear in y.
    // An equation it rejects may still be separable (`y' == g(x) h(y)`), so
    // try quadrature before giving up.
    Err(err) => match solve_separable_first_order(
      &ode_expr,
      &y_name,
      &x_name,
      &initial_conditions,
    ) {
      Some(solution) => {
        return Ok(build_dsolve_result(
          solution,
          y_name,
          x_name,
          function_form,
        ));
      }
      None => return Err(err),
    },
  };

  // Determine max order
  let max_order = terms.iter().map(|t| t.order).max().unwrap_or(0);
  if max_order <= 0 {
    // Not actually an ODE
    return Ok(unevaluated("DSolve", args));
  }

  // Check if all coefficients are constant w.r.t. x
  let all_constant_coeffs = terms.iter().filter(|t| t.order >= 0).all(|t| {
    crate::functions::calculus_ast::is_constant_wrt(&t.coefficient, &x_name)
  });

  // Check if forcing term is also constant
  let forcing_is_constant = terms.iter().filter(|t| t.order == -1).all(|t| {
    crate::functions::calculus_ast::is_constant_wrt(&t.coefficient, &x_name)
  });

  // Try to solve based on ODE type
  // For first-order ODEs, always try the first-order solver (handles more cases)
  let general_solution = if max_order == 1 {
    solve_first_order_linear(&terms, &x_name)?
  } else if all_constant_coeffs && forcing_is_constant {
    solve_constant_coefficient_ode(&terms, max_order as usize, &x_name)?
  } else if all_constant_coeffs {
    // Constant-coefficient with non-constant forcing — solve homogeneous part
    // TODO: add variation of parameters for non-constant forcing
    solve_constant_coefficient_ode(&terms, max_order as usize, &x_name)?
  } else if max_order == 2 && !terms.iter().any(|t| t.order == 0) {
    // The unknown itself is absent (`x y'' + k y' == 0`): with v = y' the
    // equation is first order linear in v, and y is the integral of v.
    match solve_second_order_without_unknown(&terms, &x_name)? {
      Some(solution) => solution,
      None => return Ok(unevaluated("DSolve", args)),
    }
  } else {
    // Unsupported
    return Ok(unevaluated("DSolve", args));
  };

  // Apply initial conditions if any
  let solution = if initial_conditions.is_empty() {
    general_solution
  } else {
    apply_initial_conditions(
      &general_solution,
      &initial_conditions,
      &y_name,
      &x_name,
      max_order as usize,
    )?
  };

  // Simplify the solution
  let solution =
    crate::evaluator::evaluate_expr_to_expr(&solution).unwrap_or(solution);

  Ok(build_dsolve_result(solution, y_name, x_name, function_form))
}

/// `DSolve[{eq1, eq2, …}, {y1[x], y2[x], …}, x]` where every equation (and
/// condition) mentions exactly one of the unknowns: each unknown is solved on
/// its own, and the constants of the k-th solution are shifted past those of
/// the earlier ones, so `{y' == x^2 y, z' == 5 z}` gives
/// `{{y[x] -> E^(x^3/3) C[1], z[x] -> E^(5 x) C[2]}}`. `None` when the
/// system is coupled or a component cannot be solved.
fn dsolve_decoupled_system(
  eqns_arg: &Expr,
  dep_items: &[Expr],
  x_name: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let mut names = Vec::with_capacity(dep_items.len());
  for item in dep_items {
    match item {
      Expr::FunctionCall { name, args }
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(s) if s == x_name) =>
      {
        names.push(name.clone());
      }
      _ => return Ok(None),
    }
  }
  if names.len() < 2 {
    return Ok(None);
  }
  let items: Vec<Expr> = match eqns_arg {
    Expr::List(items) => items.iter().cloned().collect(),
    other => vec![other.clone()],
  };
  let mut groups: Vec<Vec<Expr>> = vec![Vec::new(); names.len()];
  for item in &items {
    let mentioned: Vec<usize> = names
      .iter()
      .enumerate()
      .filter(|(_, n)| expr_mentions_function(item, n))
      .map(|(i, _)| i)
      .collect();
    let [only] = mentioned.as_slice() else {
      return Ok(None);
    };
    groups[*only].push(item.clone());
  }
  // Each unknown needs exactly one differential equation of its own.
  for (group, name) in groups.iter().zip(&names) {
    let odes = group
      .iter()
      .filter(|e| !is_initial_condition(e, name, x_name))
      .count();
    if odes != 1 || !group.iter().any(contains_derivative) {
      return Ok(None);
    }
  }

  // The solution lists of each component, constants already shifted.
  let mut offset: i128 = 0;
  let mut components: Vec<Vec<Expr>> = Vec::new();
  for (group, item) in groups.into_iter().zip(dep_items) {
    let eqns = if group.len() == 1 {
      group[0].clone()
    } else {
      Expr::List(group.into())
    };
    let solved = match dsolve_ast_inner(&[
      eqns,
      item.clone(),
      Expr::Identifier(x_name.to_string()),
    ]) {
      Ok(Expr::List(ref solutions)) if !solutions.is_empty() => {
        solutions.to_vec()
      }
      Ok(_) | Err(InterpreterError::EvaluationError(_)) => return Ok(None),
      Err(e) => return Err(e),
    };
    let max_c = solved.iter().map(max_constant_index).max().unwrap_or(0);
    let mut rules = Vec::new();
    for solution in &solved {
      let Expr::List(rs) = solution else {
        return Ok(None);
      };
      let [rule] = rs.as_slice() else {
        return Ok(None);
      };
      rules.push(shift_constants(rule, offset));
    }
    offset += max_c;
    components.push(rules);
  }

  // Every combination of the components' solution branches.
  let mut combined: Vec<Vec<Expr>> = vec![Vec::new()];
  for rules in components {
    combined = combined
      .into_iter()
      .flat_map(|prefix| {
        rules.iter().map(move |rule| {
          let mut next = prefix.clone();
          next.push(rule.clone());
          next
        })
      })
      .collect();
  }
  let result = Expr::List(
    combined
      .into_iter()
      .map(|rules| Expr::List(rules.into()))
      .collect(),
  );
  Ok(Some(
    crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
  ))
}

/// A differential-algebraic system `DSolve[{y'[x] - 3 z[x] == Sin[x],
/// y[x] + z[x] == 1/5, …}, {y[x], z[x]}, x]`: an equation free of
/// derivatives that is linear in an unknown never differentiated anywhere
/// gives that unknown in terms of the others. It is substituted into the
/// remaining equations, which are solved for the remaining unknowns, and
/// the eliminated unknown is recovered from their solution. Each solution
/// is brought over a common denominator, as wolframscript writes it:
/// `{{y[x] -> (2 - Cos[x] + 3*Sin[x])/10, z[x] -> (Cos[x] - 3*Sin[x])/10}}`.
fn dsolve_eliminating_algebraic(
  eqns_arg: &Expr,
  dep_items: &[Expr],
  x_name: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let mut names = Vec::with_capacity(dep_items.len());
  for item in dep_items {
    match item {
      Expr::FunctionCall { name, args }
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(s) if s == x_name) =>
      {
        names.push(name.clone());
      }
      _ => return Ok(None),
    }
  }
  if names.len() < 2 {
    return Ok(None);
  }
  let items: Vec<Expr> = match eqns_arg {
    Expr::List(items) => items.iter().cloned().collect(),
    other => vec![other.clone()],
  };
  let is_condition =
    |e: &Expr| names.iter().any(|n| is_initial_condition(e, n, x_name));
  // An unknown that some equation differentiates cannot be eliminated.
  let differentiated = |name: &str| {
    items.iter().any(|e| {
      contains_derivative(e)
        && derivative_orders(e).get(name).is_some_and(|&k| k > 0)
    })
  };
  let ev = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
  for (eq_index, eq) in items.iter().enumerate() {
    if contains_derivative(eq) || is_condition(eq) {
      continue;
    }
    for (k, name) in names.iter().enumerate() {
      if !expr_mentions_function(eq, name) || differentiated(name) {
        continue;
      }
      // A condition on the eliminated unknown would need translating.
      if items.iter().any(|e| is_initial_condition(e, name, x_name)) {
        continue;
      }
      let solved =
        crate::functions::solve_ast(&[eq.clone(), dep_items[k].clone()])?;
      let Expr::List(sols) = &solved else { continue };
      let [Expr::List(rules)] = sols.as_slice() else {
        continue;
      };
      let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
        continue;
      };
      let value = replacement.as_ref().clone();
      if expr_mentions_function(&value, name) {
        continue;
      }
      let substitution = Expr::Rule {
        pattern: Box::new(dep_items[k].clone()),
        replacement: Box::new(value.clone()),
      };
      let rest: Vec<Expr> = items
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != eq_index)
        .map(|(_, e)| {
          ev(Expr::FunctionCall {
            name: "ReplaceAll".to_string(),
            args: vec![e.clone(), substitution.clone()].into(),
          })
        })
        .collect::<Result<_, _>>()?;
      let remaining: Vec<Expr> = dep_items
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != k)
        .map(|(_, d)| d.clone())
        .collect();
      let sub_dep = if remaining.len() == 1 {
        remaining[0].clone()
      } else {
        Expr::List(remaining.clone().into())
      };
      let sub_eqns = if rest.len() == 1 {
        rest[0].clone()
      } else {
        Expr::List(rest.into())
      };
      let sub = match dsolve_ast_inner(&[
        sub_eqns,
        sub_dep,
        Expr::Identifier(x_name.to_string()),
      ]) {
        Ok(Expr::List(ref sols)) if !sols.is_empty() => sols.to_vec(),
        Ok(_) | Err(InterpreterError::EvaluationError(_)) => return Ok(None),
        Err(e) => return Err(e),
      };
      let mut solutions = Vec::with_capacity(sub.len());
      for solution in &sub {
        let Expr::List(sub_rules) = solution else {
          return Ok(None);
        };
        let eliminated = ev(Expr::FunctionCall {
          name: "ReplaceAll".to_string(),
          args: vec![value.clone(), solution.clone()].into(),
        })?;
        let mut rules = Vec::with_capacity(dep_items.len());
        for (i, d) in dep_items.iter().enumerate() {
          let rhs = if i == k {
            eliminated.clone()
          } else {
            let Some(r) = sub_rules.iter().find_map(|r| match r {
              Expr::Rule {
                pattern,
                replacement,
              } if crate::evaluator::pattern_matching::expr_equal(
                pattern, d,
              ) =>
              {
                Some(replacement.as_ref().clone())
              }
              _ => None,
            }) else {
              return Ok(None);
            };
            r
          };
          let rhs = ev(Expr::FunctionCall {
            name: "Together".to_string(),
            args: vec![rhs].into(),
          })?;
          rules.push(Expr::Rule {
            pattern: Box::new(d.clone()),
            replacement: Box::new(rhs),
          });
        }
        solutions.push(Expr::List(rules.into()));
      }
      return Ok(Some(Expr::List(solutions.into())));
    }
  }
  Ok(None)
}

/// The largest `k` among the constants `C[k]` in `expr` (0 when none).
fn max_constant_index(expr: &Expr) -> i128 {
  let own = match expr {
    Expr::FunctionCall { name, args } if name == "C" && args.len() == 1 => {
      match &args[0] {
        Expr::Integer(k) => *k,
        _ => 0,
      }
    }
    _ => 0,
  };
  expr_children(expr)
    .into_iter()
    .map(max_constant_index)
    .fold(own, i128::max)
}

/// `expr` with every constant `C[k]` renamed to `C[k + offset]`.
fn shift_constants(expr: &Expr, offset: i128) -> Expr {
  if offset == 0 {
    return expr.clone();
  }
  match expr {
    Expr::FunctionCall { name, args }
      if name == "C"
        && args.len() == 1
        && matches!(&args[0], Expr::Integer(_)) =>
    {
      let Expr::Integer(k) = &args[0] else {
        unreachable!()
      };
      Expr::FunctionCall {
        name: "C".to_string(),
        args: vec![Expr::Integer(k + offset)].into(),
      }
    }
    _ => crate::syntax::map_children(expr, &|c| shift_constants(c, offset)),
  }
}

/// `y -> Function[{x}, sol]`.
fn function_form_rule(y_name: String, x_name: String, solution: Expr) -> Expr {
  rule_expr(
    Expr::Identifier(y_name),
    Expr::NamedFunction {
      params: vec![x_name],
      body: Box::new(solution),
      bracketed: true,
    },
  )
}

/// Wrap a solved right-hand side as `{{y[x] -> sol}}`, or
/// `{{y -> Function[{x}, sol]}}` when the caller asked for `y` rather than
/// `y[x]`.
/// The general solution of `c y'' + P y == 0` (`expr` its left side) when
/// `P/c = α + β Cos[ω x + φ]`: with `z = (ω x + φ)/2` it is Mathieu's
/// equation `y_zz + (a − 2 q Cos[2z]) y == 0`, `a = 4α/ω²`, `q = −2β/ω²`,
/// solved by `C[1] MathieuC[a, q, z] + C[2] MathieuS[a, q, z]`. Machine
/// numbers in the equation make the parameters machine numbers too.
fn dsolve_mathieu_equation(
  expr: &Expr,
  y_name: &str,
  x_name: &str,
) -> Option<Expr> {
  use crate::evaluator::evaluate_function_call_ast as call_ast;
  let x = Expr::Identifier(x_name.to_string());
  let y_of_x = Expr::FunctionCall {
    name: y_name.to_string(),
    args: vec![x.clone()].into(),
  };
  let deriv = |k: usize| {
    crate::evaluator::evaluate_expr_to_expr(
      &crate::syntax::string_to_expr(&format!(
        "Derivative[{k}][{y_name}][{x_name}]"
      ))
      .ok()?,
    )
    .ok()
  };
  let (y1, y2) = (deriv(1)?, deriv(2)?);
  let expanded = call_ast("Expand", std::slice::from_ref(expr)).ok()?;
  let c2 = call_ast("Coefficient", &[expanded.clone(), y2.clone()]).ok()?;
  let c1 = call_ast("Coefficient", &[expanded.clone(), y1.clone()]).ok()?;
  let c0 = call_ast("Coefficient", &[expanded.clone(), y_of_x.clone()]).ok()?;
  if matches!(c2, Expr::Integer(0)) || !matches!(c1, Expr::Integer(0)) {
    return None;
  }
  // Nothing else: the equation is homogeneous in y.
  let rest = call_ast(
    "Expand",
    &[call_ast(
      "Subtract",
      &[
        expanded,
        call_ast(
          "Plus",
          &[
            call_ast("Times", &[c2.clone(), y2]).ok()?,
            call_ast("Times", &[c0.clone(), y_of_x.clone()]).ok()?,
          ],
        )
        .ok()?,
      ],
    )
    .ok()?],
  )
  .ok()?;
  if !matches!(rest, Expr::Integer(0) | Expr::Real(_))
    || matches!(rest, Expr::Real(r) if r != 0.0)
  {
    return None;
  }
  let inexact = crate::functions::math_ast::contains_inexact_real(expr);
  let mut p =
    call_ast("Expand", &[call_ast("Divide", &[c0, c2]).ok()?]).ok()?;
  // Machine numbers in the equation: its constants are numericized first,
  // and the parameters follow from them (`a` comes out as `1./l^2`).
  if inexact {
    p = call_ast("Expand", &[call_ast("N", &[p]).ok()?]).ok()?;
  }
  if mentions_head(&p, y_name) {
    return None;
  }
  // The single Cos[θ] with θ = ω x + φ.
  fn cosines(e: &Expr, x: &str, out: &mut Vec<Expr>) {
    if let Expr::FunctionCall { name, args } = e
      && name == "Cos"
      && args.len() == 1
      && crate::functions::polynomial_ast::contains_var(&args[0], x)
    {
      if !out.iter().any(|o| exprs_match(o, &args[0])) {
        out.push(args[0].clone());
      }
      return;
    }
    for c in expr_children(e) {
      cosines(c, x, out);
    }
  }
  let mut found = Vec::new();
  cosines(&p, x_name, &mut found);
  let [theta] = found.as_slice() else {
    return None;
  };
  let cos_theta = Expr::FunctionCall {
    name: "Cos".to_string(),
    args: vec![theta.clone()].into(),
  };
  let omega = call_ast("D", &[theta.clone(), x.clone()]).ok()?;
  if crate::functions::polynomial_ast::contains_var(&omega, x_name)
    || matches!(omega, Expr::Integer(0))
  {
    return None;
  }
  let beta = call_ast("Coefficient", &[p.clone(), cos_theta.clone()]).ok()?;
  let alpha = call_ast(
    "Expand",
    &[call_ast(
      "Subtract",
      &[p, call_ast("Times", &[beta.clone(), cos_theta]).ok()?],
    )
    .ok()?],
  )
  .ok()?;
  if crate::functions::polynomial_ast::contains_var(&alpha, x_name)
    || crate::functions::polynomial_ast::contains_var(&beta, x_name)
  {
    return None;
  }
  let omega2 = call_ast("Power", &[omega, Expr::Integer(2)]).ok()?;
  let simplify = |e: Expr| {
    if inexact {
      Some(e)
    } else {
      call_ast("Simplify", &[e]).ok()
    }
  };
  let a = simplify(
    call_ast(
      "Divide",
      &[
        call_ast("Times", &[Expr::Integer(4), alpha]).ok()?,
        omega2.clone(),
      ],
    )
    .ok()?,
  )?;
  let q = simplify(
    call_ast(
      "Divide",
      &[call_ast("Times", &[Expr::Integer(-2), beta]).ok()?, omega2],
    )
    .ok()?,
  )?;
  let z = call_ast("Divide", &[theta.clone(), Expr::Integer(2)]).ok()?;
  let c = |i: i128| Expr::FunctionCall {
    name: "C".to_string(),
    args: vec![Expr::Integer(i)].into(),
  };
  let mathieu = |name: &str| Expr::FunctionCall {
    name: name.to_string(),
    args: vec![a.clone(), q.clone(), z.clone()].into(),
  };
  call_ast(
    "Plus",
    &[
      call_ast("Times", &[c(1), mathieu("MathieuC")]).ok()?,
      call_ast("Times", &[c(2), mathieu("MathieuS")]).ok()?,
    ],
  )
  .ok()
}

fn build_dsolve_result(
  solution: Expr,
  y_name: String,
  x_name: String,
  function_form: bool,
) -> Expr {
  let rule = if function_form {
    function_form_rule(y_name, x_name, solution)
  } else {
    rule_expr(
      Expr::FunctionCall {
        name: y_name,
        args: vec![Expr::Identifier(x_name)].into(),
      },
      solution,
    )
  };

  Expr::List(vec![Expr::List(vec![rule].into())].into())
}

// ─── NDSolve ───────────────────────────────────────────────────────────

/// NDSolve[{eqn, ic1, ...}, y[x], {x, xmin, xmax}]
/// Also NDSolve[{eqn, ic1, ...}, y, {x, xmin, xmax}]
pub fn ndsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  ndsolve_ast_with_head(args, "NDSolve")
}

/// Same as [`ndsolve_ast`], but an unevaluated result is returned under
/// `head`, so `NDSolveValue` keeps its own head instead of the `NDSolve` it
/// delegates to.
pub fn ndsolve_ast_with_head(
  args: &[Expr],
  head: &str,
) -> Result<Expr, InterpreterError> {
  // Options may come wrapped in a list (`NDSolve[eqns, u, {t, 0, 1},
  // {MaxSteps -> 100}]`, or a symbol holding one); splice such a list into
  // the trailing options so it is not taken for a fourth (PDE) domain.
  let is_rule =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let mut evaluated: Vec<Expr> = Vec::with_capacity(args.len());
  for (i, a) in evaluate_ndsolve_args(args)?.into_iter().enumerate() {
    match &a {
      Expr::List(items)
        if i >= 3 && !items.is_empty() && items.iter().all(is_rule) =>
      {
        evaluated.extend(items.iter().cloned());
      }
      _ => evaluated.push(a),
    }
  }
  // A domain listing more points (`{t, 0, 5, 1}`) spans the smallest to the
  // largest of them, as in wolframscript.
  let spanned = match evaluated.get(2) {
    Some(Expr::List(domain))
      if domain.len() > 3
        && matches!(domain.first(), Some(Expr::Identifier(_))) =>
    {
      domain[1..]
        .iter()
        .map(|p| nval_to_f64(p).map(|v| (v, p.clone())))
        .collect::<Option<Vec<_>>>()
        .map(|points| {
          let lo = points.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
          let hi = points.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
          Expr::List(vec![domain[0].clone(), lo.1.clone(), hi.1.clone()].into())
        })
    }
    _ => None,
  };
  if let Some(domain) = spanned {
    evaluated[2] = domain;
  }
  let evaluated: Vec<Expr> = evaluated
    .iter()
    .map(total_derivative_to_derivative)
    .map(|e| expand_defined_derivatives(&e))
    .map(|e| unwrap_singleton_equation_sides(&e))
    .collect();
  ndsolve_ast_inner(&evaluated)
    .map(|expr| retag_unevaluated(expr, "NDSolve", head))
}

/// Rewrite `lhs == {rhs}` (or `{lhs} == rhs`) — an initial value taken from
/// a solution list, as in `v[0] == (v /. sol)` — to `lhs == rhs`, so the
/// one-element list does not hide the condition from the solver.
fn unwrap_singleton_equation_sides(expr: &Expr) -> Expr {
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && let [ComparisonOp::Equal] = operators.as_slice()
    && let [a, b] = operands.as_slice()
  {
    let single = |e: &Expr| match e {
      Expr::List(items) if items.len() == 1 => Some(items[0].clone()),
      _ => None,
    };
    let is_list = |e: &Expr| matches!(e, Expr::List(_));
    let (a2, b2) = match (single(a), single(b)) {
      (Some(x), None) if !is_list(b) => (x, b.clone()),
      (None, Some(y)) if !is_list(a) => (a.clone(), y),
      _ => (a.clone(), b.clone()),
    };
    return Expr::Comparison {
      operands: vec![a2, b2],
      operators: operators.clone(),
    };
  }
  map_children(expr, &unwrap_singleton_equation_sides)
}

/// Rewrite `Derivative[n][g[i]][v]` — the derivative of an indexed function
/// whose definition (`g[i_][v_] := …`) gives `g[i][v]` a value — into the
/// derivative of that value, `D[g[i][v], {v, n}]`. Unknowns, whose
/// `f[i][v]` stays unevaluated, are left as they are.
fn expand_defined_derivatives(expr: &Expr) -> Expr {
  if let Expr::CurriedCall { func, args } = expr
    && let [var] = args.as_slice()
    && let Expr::CurriedCall {
      func: head,
      args: orders,
    } = func.as_ref()
    && let Expr::FunctionCall { name, args: dargs } = head.as_ref()
    && name == "Derivative"
    && let [order] = dargs.as_slice()
    && let [inner @ Expr::FunctionCall { .. }] = orders.as_slice()
  {
    let applied = Expr::CurriedCall {
      func: Box::new(inner.clone()),
      args: vec![var.clone()],
    };
    if let Ok(value) = crate::evaluator::evaluate_expr_to_expr(&applied)
      && expr_to_string(&value) != expr_to_string(&applied)
      && let Ok(d) = crate::evaluator::evaluate_function_call_ast(
        "D",
        &[value, Expr::List(vec![var.clone(), order.clone()].into())],
      )
    {
      return d;
    }
  }
  map_children(expr, &expand_defined_derivatives)
}

/// Rewrite `Dt[f[v], v]` / `Dt[f[v], {v, n}]` — a total derivative of a
/// function of its own differentiation variable, the way ODEs are often
/// written, and the shape `Dt[x[t], t] /. t -> 0` leaves behind in an
/// initial condition (`Dt[x[0], 0]`) — into `Derivative[n][f][v]`.
fn total_derivative_to_derivative(expr: &Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Dt"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: fname,
      args: fargs,
    } = &args[0]
    && let [arg] = fargs.as_slice()
  {
    let order = match &args[1] {
      Expr::List(items) => match items.as_slice() {
        [v, Expr::Integer(n)]
          if *n >= 1 && expr_to_string(v) == expr_to_string(arg) =>
        {
          Some(*n)
        }
        _ => None,
      },
      v if expr_to_string(v) == expr_to_string(arg) => Some(1),
      _ => None,
    };
    if let Some(n) = order {
      return Expr::CurriedCall {
        func: Box::new(Expr::CurriedCall {
          func: Box::new(Expr::FunctionCall {
            name: "Derivative".to_string(),
            args: vec![Expr::Integer(n)].into(),
          }),
          args: vec![Expr::Identifier(fname.clone())],
        }),
        args: vec![arg.clone()],
      };
    }
  }
  map_children(expr, &total_derivative_to_derivative)
}

/// NDSolve holds its arguments (matching Wolfram — see the `NDSolve`/
/// `NDSolveValue` entry in `core_eval.rs`'s held-function list) so that a
/// `Block`/Manipulate-bound variable sharing a name with the independent
/// variable — e.g. a Demonstration's own animation clock, when its ODE
/// spells the domain as `{t, 0, tMax}` — can't get substituted into the
/// iterator position before the solve ever sees `t` as symbolic. But
/// holding everything would just as wrongly leave ordinary named
/// equations unresolved (`eq1 = y'[t] == 1; NDSolve[{eq1, …}, …]`, a
/// common idiom, needs `eq1` looked up). So evaluate the whole argument
/// list normally except the domain spec's own iterator variable name(s)
/// (one per positional domain argument, for the PDE form's two): the
/// symbol is temporarily removed from scope — the same
/// take/restore-symbol-values pair `Block` itself uses, applied directly
/// rather than through a nested `Block[…]` call (which, evaluated through
/// the ordinary interpreter, doesn't shadow an *already*-`Block`-bound
/// outer value of the same name the way re-entering it directly does) —
/// so it evaluates to itself rather than the caller's current value,
/// exactly like `Table`/`Do`'s own iterator.
fn evaluate_ndsolve_args(args: &[Expr]) -> Result<Vec<Expr>, InterpreterError> {
  let n_pos = args
    .iter()
    .take_while(|a| !matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .count();
  let mut iter_names: Vec<String> = Vec::new();
  for &i in &[2usize, 3usize] {
    if i < n_pos
      && let Some(Expr::List(items)) = args.get(i)
      && let Some(Expr::Identifier(name)) = items.first()
    {
      iter_names.push(name.clone());
    }
  }
  if iter_names.is_empty() {
    // No recognizable `{var, …}` domain spec (malformed call) — evaluate
    // plainly so it still falls through to the ordinary "not a valid
    // NDSolve call" outcome below instead of silently skipping evaluation.
    return args
      .iter()
      .map(crate::evaluator::evaluate_expr_to_expr)
      .collect();
  }
  let saved: Vec<_> = iter_names
    .iter()
    .map(|n| crate::evaluator::symbol_values::take_symbol_values(n))
    .collect();
  let result: Result<Vec<Expr>, InterpreterError> = args
    .iter()
    .map(crate::evaluator::evaluate_expr_to_expr)
    .collect();
  for s in saved {
    crate::evaluator::symbol_values::restore_symbol_values(s);
  }
  result
}

thread_local! {
  /// The step size `StartingStepSize`/`MaxStepSize` asked for, while the
  /// NDSolve call that gave it runs.
  static STEP_HINT: std::cell::Cell<Option<f64>> =
    const { std::cell::Cell::new(None) };
}

struct StepHintScope(Option<f64>);

impl StepHintScope {
  fn new(hint: Option<f64>) -> Self {
    Self(STEP_HINT.with(|c| c.replace(hint)))
  }
}

impl Drop for StepHintScope {
  fn drop(&mut self) {
    STEP_HINT.with(|c| c.set(self.0));
  }
}

/// The unknowns of a PDE `NDSolve` written as applied functions of the two
/// domain variables (`u[x, t]`, or a list of them) as `(name, arguments)`.
/// `None` if any unknown is a bare symbol or applied to anything else.
fn pde_applied_unknowns(
  spec: &Expr,
  dom_a: &Expr,
  dom_b: &Expr,
) -> Option<Vec<(String, Vec<Expr>)>> {
  let var_of = |d: &Expr| match d {
    Expr::List(items) => items.first().map(expr_to_string),
    _ => None,
  };
  let (va, vb) = (var_of(dom_a)?, var_of(dom_b)?);
  let single = std::slice::from_ref(spec);
  let items: &[Expr] = match spec {
    Expr::List(items) => items,
    _ => single,
  };
  let mut out = Vec::with_capacity(items.len());
  for item in items {
    let Expr::FunctionCall { name, args } = item else {
      return None;
    };
    let vars: Vec<String> = args.iter().map(expr_to_string).collect();
    let is_pair = vars.len() == 2
      && ((vars[0] == va && vars[1] == vb) || (vars[0] == vb && vars[1] == va));
    if !is_pair {
      return None;
    }
    out.push((name.clone(), args.to_vec()));
  }
  (!out.is_empty()).then_some(out)
}

/// Rewrite `{{u -> f}}` into `{{u[x, t] -> f[x, t]}}` for the unknowns
/// asked for in applied form.
fn apply_pde_unknown_rules(
  solved: &Expr,
  applied: &[(String, Vec<Expr>)],
) -> Expr {
  let Expr::List(solutions) = solved else {
    return solved.clone();
  };
  let mut out = Vec::with_capacity(solutions.len());
  for sol in solutions {
    let Expr::List(rules) = sol else {
      return solved.clone();
    };
    let mut new_rules = Vec::with_capacity(rules.len());
    for rule in rules {
      let Expr::Rule {
        pattern,
        replacement,
      } = rule
      else {
        return solved.clone();
      };
      let Expr::Identifier(name) = pattern.as_ref() else {
        return solved.clone();
      };
      let Some((_, call_args)) = applied.iter().find(|(n, _)| n == name) else {
        return solved.clone();
      };
      new_rules.push(Expr::Rule {
        pattern: Box::new(Expr::FunctionCall {
          name: name.clone(),
          args: call_args.clone().into(),
        }),
        replacement: Box::new(Expr::CurriedCall {
          func: replacement.clone(),
          args: call_args.clone(),
        }),
      });
    }
    out.push(Expr::List(new_rules.into()));
  }
  Expr::List(out.into())
}

fn ndsolve_ast_inner(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Split trailing option rules (`Method -> …`, `MaxSteps -> …`, …) from
  // the positional arguments: three for an ODE/DAE system in one
  // independent variable, four for a scalar PDE
  // `NDSolve[eqns, u, {t, t0, t1}, {x, x0, x1}]` in two.
  let n_pos = args
    .iter()
    .take_while(|a| !matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .count();
  if n_pos == 4 {
    // The equations may nest (`{pdeAndIcs, {bcs}}`); the PDE solvers read a
    // flat list.
    let mut flat_eqs = Vec::new();
    flatten_eq_list(&args[0], &mut flat_eqs);
    let mut args = args.to_vec();
    args[0] = Expr::List(flat_eqs.into());
    // Unknowns written with their arguments (`u[x, t]`) are solved as the
    // bare functions, then the result rules are keyed by the same forms.
    if let Some(applied) = pde_applied_unknowns(&args[1], &args[2], &args[3]) {
      let mut bare = args.clone();
      bare[1] = match &args[1] {
        Expr::List(_) => Expr::List(
          applied
            .iter()
            .map(|(n, _)| Expr::Identifier(n.clone()))
            .collect(),
        ),
        _ => Expr::Identifier(applied[0].0.clone()),
      };
      let solved = ndsolve_ast_inner(&bare)?;
      return Ok(apply_pde_unknown_rules(&solved, &applied));
    }
    let args = &args[..];
    // Unknowns written as compound expressions (`Subscript[c, A]`) are keyed
    // by a fresh symbol while integrating, then restored in the result —
    // same treatment as the ODE branch below.
    let renames = compound_head_renamings(&args[1], &args[2]);
    let positional: Vec<Expr> = if renames.is_empty() {
      args[..4].to_vec()
    } else {
      args[..4]
        .iter()
        .map(|a| rename_compound_heads(a, &renames))
        .collect()
    };
    return match ndsolve_pde(&positional) {
      Ok(Some(result)) => Ok(restore_compound_heads(&result, &renames)),
      Ok(None) => match ndsolve_pde_hyperbolic(&positional) {
        Ok(Some(result)) => Ok(restore_compound_heads(&result, &renames)),
        Ok(None) => match complex_pde::ndsolve_pde_complex(&positional) {
          Ok(Some(result)) => Ok(restore_compound_heads(&result, &renames)),
          Ok(None) => Ok(unevaluated("NDSolve", args)),
          Err(e) => Err(e),
        },
        Err(e) => Err(e),
      },
      Err(e) => Err(e),
    };
  }
  if n_pos != 3 {
    return Ok(unevaluated("NDSolve", args));
  }
  if let Some(rewritten) = dirac_delta_events(args)? {
    return ndsolve_ast_inner(&rewritten);
  }
  if let Some(result) = ndsolve_vector_unknowns(args, n_pos)? {
    return Ok(result);
  }
  // Derivatives asked for as unknowns (`{x''[t], x'[t], x[t]}`): solve for
  // the functions themselves, then give a rule for each requested form in
  // order, its value read off the solution.
  if let Expr::List(unknowns) = &args[1]
    && unknowns
      .iter()
      .any(|u| derivative_unknown_base(u).is_some())
  {
    let mut bases: Vec<Expr> = Vec::new();
    for u in unknowns {
      let base = derivative_unknown_base(u).or_else(|| match u {
        Expr::Identifier(_) => Some(u.clone()),
        Expr::FunctionCall { name, .. } => Some(Expr::Identifier(name.clone())),
        _ => None,
      });
      let Some(base) = base else {
        return Ok(unevaluated("NDSolve", args));
      };
      if !bases
        .iter()
        .any(|b| expr_to_string(b) == expr_to_string(&base))
      {
        bases.push(base);
      }
    }
    let mut reduced = args.to_vec();
    reduced[1] = Expr::List(bases.into());
    let solved = ndsolve_ast_inner(&reduced)?;
    let Expr::List(solutions) = &solved else {
      return Ok(solved);
    };
    let mut out = Vec::new();
    for sol in solutions {
      let mut rules = Vec::new();
      for u in unknowns {
        let value = crate::evaluator::evaluate_function_call_ast(
          "ReplaceAll",
          &[u.clone(), sol.clone()],
        )?;
        rules.push(Expr::Rule {
          pattern: Box::new(u.clone()),
          replacement: Box::new(value),
        });
      }
      out.push(Expr::List(rules.into()));
    }
    return Ok(Expr::List(out.into()));
  }
  // An unknown the equations never mention is solved as itself (`w -> w`),
  // the others as usual.
  if let Expr::List(unknowns) = &args[1] {
    let mentions = |name: &str| mentions_symbol(&args[0], name);
    let (absent, present): (Vec<&Expr>, Vec<&Expr>) = unknowns
      .iter()
      .partition(|u| matches!(u, Expr::Identifier(name) if !mentions(name)));
    if !absent.is_empty() && !present.is_empty() {
      let mut reduced = args.to_vec();
      reduced[1] = Expr::List(present.into_iter().cloned().collect());
      let solved = ndsolve_ast_inner(&reduced)?;
      let Expr::List(solutions) = &solved else {
        return Ok(solved);
      };
      let complete = solutions
        .iter()
        .map(|sol| {
          let Expr::List(rules) = sol else {
            return sol.clone();
          };
          let rule_for = |u: &Expr| {
            rules
              .iter()
              .find(|r| matches!(r, Expr::Rule { pattern, .. } if crate::syntax::expr_to_string(pattern) == crate::syntax::expr_to_string(u)))
              .cloned()
              .unwrap_or_else(|| Expr::Rule {
                pattern: Box::new(u.clone()),
                replacement: Box::new(u.clone()),
              })
          };
          Expr::List(unknowns.iter().map(rule_for).collect())
        })
        .collect();
      return Ok(Expr::List(complete));
    }
  }
  // An independent variable written as a compound expression
  // (`{Subscript[A, m], 0, w}`) is keyed by a fresh symbol while
  // integrating, then restored in the result.
  if let Expr::List(domain) = &args[2]
    && matches!(
      domain.first(),
      Some(Expr::FunctionCall { .. } | Expr::CurriedCall { .. })
    )
  {
    let renames = vec![(domain[0].clone(), "NDSolve$var".to_string())];
    let renamed: Vec<Expr> = args
      .iter()
      .map(|a| rename_compound_heads(a, &renames))
      .collect();
    let result = ndsolve_ast_inner(&renamed)?;
    return Ok(restore_compound_heads(&result, &renames));
  }
  // A first-order system with a stated history (`f[t /; t <= t0] == …`)
  // and possibly delayed arguments `f[t - c]`.
  if let Some(result) = dde::ndsolve_dde(&args[..3])? {
    return Ok(result);
  }
  let opts = &args[n_pos..];
  let event = parse_event_locator_option(opts);

  // A stage-indexed family (`Table[x[i, t], {i, 1, n}]` as the dependent
  // variable spec — a discretized PDE or a staged process model) is keyed
  // by a fresh single-argument symbol per index, the same way a compound
  // head is below, so the rest of the solver only ever sees ordinary
  // `f[t]`-shaped unknowns.
  let idx_renames = indexed_family_renamings(&args[1], &args[2]);
  let indexed: Vec<Expr> = if idx_renames.is_empty() {
    args[..3].to_vec()
  } else {
    args[..3]
      .iter()
      .map(|a| rename_indexed_vars(a, &idx_renames))
      .collect()
  };

  // Unknowns written as compound expressions (`Subscript[c, 1]`) are keyed
  // by a fresh symbol while integrating, then restored in the result.
  let renames = compound_head_renamings(&indexed[1], &indexed[2]);
  let positional: Vec<Expr> = if renames.is_empty() {
    indexed.clone()
  } else {
    indexed
      .iter()
      .map(|a| rename_compound_heads(a, &renames))
      .collect()
  };

  // Every shape — one equation or a coupled system, linear or not — goes
  // through the same integrator. A single scalar equation used to take a
  // separate path built on DSolve's *linear* term classifier, which
  // refused anything nonlinear in the dependent variable (the pendulum's
  // `Sin[θ[t]]`) even though nothing about integrating it numerically
  // needs the equation to be linear.
  // `StartingStepSize -> h` (with or without a fixed-step method) or
  // `MaxStepSize -> h` asks for steps no longer than h: the nominal grid is
  // refined to it.
  let step_hint = opts.iter().find_map(|o| {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = o
    else {
      return None;
    };
    match pattern.as_ref() {
      Expr::Identifier(n) if n == "StartingStepSize" || n == "MaxStepSize" => {
        nval_to_f64(replacement).filter(|h| *h > 0.0 && h.is_finite())
      }
      _ => None,
    }
  });
  let _step_hint = StepHintScope::new(step_hint);
  // A system of algebraic equations plus unit-speed conditions on the
  // derivatives is an implicit curve rather than an initial value problem.
  let solved = match ndsolve_system(&positional, event.as_ref()) {
    Ok(None) if event.is_none() => {
      match curve::ndsolve_implicit_curve(&positional) {
        // A complex-valued system is solved as the real system of its parts.
        Ok(None) => complex_ode::ndsolve_complex_split(&positional),
        other => other,
      }
    }
    other => other,
  };
  match solved {
    Ok(Some(result)) => {
      let result = restore_compound_heads(&result, &renames);
      Ok(restore_indexed_vars(&result, &idx_renames))
    }
    Ok(None) => Ok(unevaluated("NDSolve", args)),
    Err(e) => Err(e),
  }
}

/// A vector-valued unknown of an `NDSolve` system and the scalar symbols
/// standing for its components while integrating.
struct VectorUnknown {
  name: String,
  components: Vec<String>,
}

/// Number of components of the vector-valued unknown `name`, read off an
/// initial condition `name[t0] == {…}` / `name'[t0] == {…}`.
fn vector_unknown_dimension(eqs: &[Expr], name: &str) -> Option<usize> {
  eqs.iter().find_map(|eq| {
    let (a, b) = as_equal_pair(eq)?;
    // `name[t0]` or `Derivative[k, name, t0]`.
    let at_point = |e: &Expr| match e {
      Expr::FunctionCall { name: n, args } if n == "Derivative" => {
        args.len() == 3
          && matches!(&args[1], Expr::Identifier(f) if f == name)
          && nval_to_f64(&args[2]).is_some()
      }
      Expr::FunctionCall { name: n, args } => {
        n == name && args.len() == 1 && nval_to_f64(&args[0]).is_some()
      }
      _ => false,
    };
    match (a, b) {
      (lhs, Expr::List(items)) if at_point(lhs) => Some(items.len()),
      (Expr::List(items), rhs) if at_point(rhs) => Some(items.len()),
      _ => None,
    }
  })
}

/// The `(expression, index)` of `expr[[index]]`, whether parsed as a `Part`
/// node or evaluated into `Part[expr, index]`.
fn part_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
  match expr {
    Expr::Part { expr, index } => Some((expr, index)),
    Expr::FunctionCall { name, args } if name == "Part" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

/// Replace every use of a vector-valued unknown by the list of its scalar
/// components (`X[t]` -> `{X$1[t], X$2[t]}`, `X''[t]` likewise), folding the
/// `[[i]]` picks that then select a component directly. With `pick` set only
/// that component is substituted, for a list element that earlier listable
/// arithmetic already threaded over the components.
fn expand_vector_unknowns(
  expr: &Expr,
  vecs: &[VectorUnknown],
  pick: Option<usize>,
) -> Expr {
  let components_of = |name: &str| vecs.iter().find(|v| v.name == name);
  let gather = |v: &VectorUnknown, make: &dyn Fn(&String) -> Expr| match pick {
    Some(i) => make(&v.components[i]),
    None => Expr::List(v.components.iter().map(make).collect()),
  };
  match expr {
    Expr::FunctionCall { name, args } if name == "Derivative" => {
      if let [k, Expr::Identifier(f), at] = args.as_slice()
        && let Some(v) = components_of(f)
      {
        let at = expand_vector_unknowns(at, vecs, pick);
        return gather(v, &|c| Expr::FunctionCall {
          name: "Derivative".to_string(),
          args: vec![k.clone(), Expr::Identifier(c.clone()), at.clone()].into(),
        });
      }
    }
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      if let Some(v) = components_of(name) {
        let at = expand_vector_unknowns(&args[0], vecs, pick);
        return gather(v, &|c| Expr::FunctionCall {
          name: c.clone(),
          args: vec![at.clone()].into(),
        });
      }
    }
    Expr::Part { .. } | Expr::FunctionCall { .. }
      if let Some((base, index)) = part_operands(expr) =>
    {
      let base = expand_vector_unknowns(base, vecs, pick);
      let index = expand_vector_unknowns(index, vecs, pick);
      if let (Expr::List(items), Expr::Integer(i)) = (&base, &index)
        && *i >= 1
        && let Some(item) = items.get(*i as usize - 1)
      {
        return item.clone();
      }
      return call("Part", vec![base, index]);
    }
    _ => {}
  }
  map_children(expr, &|c| expand_vector_unknowns(c, vecs, pick))
}

/// `NDSolve` for unknowns that are vectors (`X''[t] == f[X[t], X'[t]]` with
/// `X[0] == {x0, y0}`): each is split into scalar components, the vector
/// equations are threaded over them, and the component solutions are
/// recombined into one tensor-valued `InterpolatingFunction` per unknown.
fn ndsolve_vector_unknowns(
  args: &[Expr],
  n_pos: usize,
) -> Result<Option<Expr>, InterpreterError> {
  let unknowns: Vec<&Expr> = match &args[1] {
    Expr::List(items) => items.iter().collect(),
    single => vec![single],
  };
  let mut eqs = Vec::new();
  flatten_eq_list(&args[0], &mut eqs);

  // A vector unknown asked for applied (`swan[t]`) is solved as the bare
  // function, and its rule comes back applied: `swan[t] -> IF[…][t]`.
  if let Expr::List(domain) = &args[2]
    && let Some(Expr::Identifier(t)) = domain.first()
  {
    let applied: Vec<String> = unknowns
      .iter()
      .filter_map(|u| match u {
        Expr::FunctionCall { name, args: fargs }
          if matches!(fargs.as_slice(), [Expr::Identifier(a)] if a == t)
            && vector_unknown_dimension(&eqs, name).is_some() =>
        {
          Some(name.clone())
        }
        _ => None,
      })
      .collect();
    if !applied.is_empty() {
      let bare = |u: &Expr| match u {
        Expr::FunctionCall { name, .. } if applied.contains(name) => {
          Expr::Identifier(name.clone())
        }
        other => other.clone(),
      };
      let mut bare_args = args.to_vec();
      bare_args[1] = match &args[1] {
        Expr::List(items) => Expr::List(items.iter().map(bare).collect()),
        single => bare(single),
      };
      let Some(Expr::List(ref solutions)) =
        ndsolve_vector_unknowns(&bare_args, n_pos)?
      else {
        return Ok(None);
      };
      let var = Expr::Identifier(t.clone());
      let applied_rule = |r: &Expr| match r {
        Expr::Rule {
          pattern,
          replacement,
        } if matches!(pattern.as_ref(), Expr::Identifier(n) if applied.contains(n)) => {
          rule_expr(
            call(&expr_to_string(pattern), vec![var.clone()]),
            Expr::CurriedCall {
              func: replacement.clone(),
              args: vec![var.clone()],
            },
          )
        }
        other => other.clone(),
      };
      return Ok(Some(Expr::List(
        solutions
          .iter()
          .map(|sol| match sol {
            Expr::List(rules) => {
              Expr::List(rules.iter().map(applied_rule).collect())
            }
            other => other.clone(),
          })
          .collect(),
      )));
    }
  }
  let mut vecs: Vec<VectorUnknown> = Vec::new();
  for u in &unknowns {
    if let Expr::Identifier(name) = u
      && let Some(dim) = vector_unknown_dimension(&eqs, name)
    {
      vecs.push(VectorUnknown {
        name: name.clone(),
        components: (1..=dim).map(|i| format!("{name}$vec{i}")).collect(),
      });
    }
  }
  if vecs.is_empty() {
    return Ok(None);
  }

  let mut scalar_eqs: Vec<Expr> = Vec::new();
  for eq in &eqs {
    let Some((lhs, rhs)) = as_equal_pair(eq) else {
      scalar_eqs.push(expand_vector_unknowns(eq, &vecs, None));
      continue;
    };
    let new_lhs = expand_vector_unknowns(lhs, &vecs, None);
    let Expr::List(lhs_items) = &new_lhs else {
      scalar_eqs.push(Expr::Comparison {
        operands: vec![new_lhs, expand_vector_unknowns(rhs, &vecs, None)],
        operators: vec![ComparisonOp::Equal],
      });
      continue;
    };
    // A right-hand side already written (or threaded) as one list element
    // per component is read element-wise; anything else is a vector-valued
    // expression whose components are picked out of it.
    // Evaluated once: a product like `A . y[t]` would otherwise be
    // recomputed in full for every component picked out of it. A
    // user-defined function is left alone: it may only accept numeric
    // vectors, so it has to be evaluated per step.
    let whole = expand_vector_unknowns(rhs, &vecs, None);
    let whole = match crate::evaluator::evaluate_expr_to_expr(&whole) {
      Ok(evaluated)
        if !calls_user_function(rhs)
          && matches!(&evaluated, Expr::List(items) if items.len() == lhs_items.len()) =>
      {
        evaluated
      }
      _ => whole,
    };
    for (i, l) in lhs_items.iter().enumerate() {
      let r = match rhs {
        Expr::List(items) if items.len() == lhs_items.len() => {
          expand_vector_unknowns(&items[i], &vecs, Some(i))
        }
        _ => match &whole {
          Expr::List(items) if items.len() == lhs_items.len() => {
            items[i].clone()
          }
          _ => call("Part", vec![whole.clone(), Expr::Integer(i as i128 + 1)]),
        },
      };
      scalar_eqs.push(Expr::Comparison {
        operands: vec![l.clone(), r],
        operators: vec![ComparisonOp::Equal],
      });
    }
  }
  let mut scalar_unknowns: Vec<Expr> = Vec::new();
  for u in &unknowns {
    match u {
      Expr::Identifier(name) if vecs.iter().any(|v| v.name == *name) => {
        let v = vecs.iter().find(|v| v.name == *name).unwrap();
        scalar_unknowns
          .extend(v.components.iter().map(|c| Expr::Identifier(c.clone())));
      }
      other => scalar_unknowns.push((*other).clone()),
    }
  }
  let mut scalar_args: Vec<Expr> = vec![
    Expr::List(scalar_eqs.into()),
    Expr::List(scalar_unknowns.into()),
    args[2].clone(),
  ];
  scalar_args.extend(
    args[n_pos..]
      .iter()
      .map(|o| expand_vector_unknowns(o, &vecs, None)),
  );

  let solved = ndsolve_ast_inner(&scalar_args)?;
  let Expr::List(solutions) = &solved else {
    return Ok(Some(unevaluated("NDSolve", args)));
  };
  let mut merged_solutions = Vec::with_capacity(solutions.len());
  for sol in solutions {
    let Expr::List(rules) = sol else {
      return Ok(Some(unevaluated("NDSolve", args)));
    };
    let interp_of = |name: &str| {
      rules.iter().find_map(|r| match r {
        Expr::Rule {
          pattern,
          replacement,
        } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == name) => {
          match replacement.as_ref() {
            Expr::FunctionCall { name: h, args }
              if h == "InterpolatingFunction" =>
            {
              Some(args.clone())
            }
            _ => None,
          }
        }
        _ => None,
      })
    };
    let mut merged_rules = Vec::new();
    for u in &unknowns {
      let Expr::Identifier(name) = u else {
        continue;
      };
      let Some(v) = vecs.iter().find(|v| v.name == *name) else {
        // A scalar unknown keeps the rule the component solve produced.
        if let Some(rule) = rules.iter().find(|r| {
          matches!(r, Expr::Rule { pattern, .. } if expr_to_string(pattern) == *name)
        }) {
          merged_rules.push(rule.clone());
        }
        continue;
      };
      let parts: Option<Vec<_>> =
        v.components.iter().map(|c| interp_of(c)).collect();
      let Some(parts) = parts else {
        return Ok(Some(unevaluated("NDSolve", args)));
      };
      let columns: Vec<&[Expr]> = parts
        .iter()
        .map(|p| match p.get(1) {
          Some(Expr::List(rows)) => rows.as_slice(),
          _ => &[],
        })
        .collect();
      let n_rows = columns.first().map_or(0, |c| c.len());
      if n_rows == 0 || columns.iter().any(|c| c.len() != n_rows) {
        return Ok(Some(unevaluated("NDSolve", args)));
      }
      let mut rows = Vec::with_capacity(n_rows);
      for r in 0..n_rows {
        let mut t = None;
        let mut ys = Vec::with_capacity(columns.len());
        for col in &columns {
          let Expr::List(pair) = &col[r] else {
            return Ok(Some(unevaluated("NDSolve", args)));
          };
          let [x, y] = pair.as_slice() else {
            return Ok(Some(unevaluated("NDSolve", args)));
          };
          t.get_or_insert_with(|| x.clone());
          ys.push(y.clone());
        }
        rows.push(Expr::List(vec![t.unwrap(), Expr::List(ys.into())].into()));
      }
      let mut interp_args: Vec<Expr> = parts[0].to_vec();
      interp_args[1] = Expr::List(rows.into());
      merged_rules.push(rule_expr(
        Expr::Identifier(name.clone()),
        call("InterpolatingFunction", interp_args),
      ));
    }
    merged_solutions.push(Expr::List(merged_rules.into()));
  }
  Ok(Some(Expr::List(merged_solutions.into())))
}

// ─── PDE NDSolve (method of lines) ─────────────────────────────────────

/// A domain spec `{name, min, max}` for one PDE independent variable.
#[derive(Clone)]
struct PdeDomain {
  name: String,
  min: f64,
  max: f64,
}

fn parse_pde_domain(expr: &Expr) -> Option<PdeDomain> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.len() != 3 {
    return None;
  }
  let Expr::Identifier(name) = &items[0] else {
    return None;
  };
  let min = nval_to_f64(&items[1])?;
  let max = nval_to_f64(&items[2])?;
  // NaN bounds must bail out too, so compare via `partial_cmp` rather than
  // a negated float comparison.
  if max.partial_cmp(&min) != Some(std::cmp::Ordering::Greater) {
    return None;
  }
  Some(PdeDomain {
    name: name.clone(),
    min,
    max,
  })
}

/// Match `Derivative[dt, dx][u_name][t_arg, x_arg]` — the parsed form of a
/// mixed partial `D[u[t, x], {t, dt}, {x, dx}]` — or the bare
/// `u_name[t_arg, x_arg]` (order `(0, 0)`). Returns `(dt, dx, t_arg, x_arg)`.
fn match_pde_term(
  expr: &Expr,
  u_name: &str,
) -> Option<(usize, usize, Expr, Expr)> {
  if let Expr::FunctionCall { name, args } = expr
    && name == u_name
    && args.len() == 2
  {
    return Some((0, 0, args[0].clone(), args[1].clone()));
  }
  if let Expr::CurriedCall { func, args } = expr
    && args.len() == 2
    && let Expr::CurriedCall {
      func: deriv_head,
      args: fname_args,
    } = func.as_ref()
    && fname_args.len() == 1
    && matches!(&fname_args[0], Expr::Identifier(n) if n == u_name)
    && let Expr::FunctionCall {
      name: deriv_name,
      args: orders,
    } = deriv_head.as_ref()
    && deriv_name == "Derivative"
    && orders.len() == 2
    && let (Expr::Integer(dt), Expr::Integer(dx)) = (&orders[0], &orders[1])
  {
    return Some((
      *dt as usize,
      *dx as usize,
      args[0].clone(),
      args[1].clone(),
    ));
  }
  None
}

/// Like [`match_pde_term`], but normalizes the returned order/arguments so
/// the first element of each always corresponds to whichever call-argument
/// position plays the "t" (time-like) role for this match attempt and the
/// second to the "x" (space-like) role. `swap` is true when the system's
/// equations call the unknown as `u[x, t]` rather than the conventional
/// `u[t, x]` — which one it is isn't fixed by the `NDSolve` call's own
/// domain-argument order, so callers try both.
fn match_pde_term_roles(
  expr: &Expr,
  u_name: &str,
  swap: bool,
) -> Option<(usize, usize, Expr, Expr)> {
  let (dt, dx, arg0, arg1) = match_pde_term(expr, u_name)?;
  if swap {
    Some((dx, dt, arg1, arg0))
  } else {
    Some((dt, dx, arg0, arg1))
  }
}

fn as_equal_pair(eq: &Expr) -> Option<(&Expr, &Expr)> {
  let Expr::Comparison {
    operands,
    operators,
  } = eq
  else {
    return None;
  };
  if operands.len() == 2
    && operators.len() == 1
    && operators[0] == ComparisonOp::Equal
  {
    Some((&operands[0], &operands[1]))
  } else {
    None
  }
}

/// Flatten a product into its individual factors — a left-associative
/// chain of `Times` `BinaryOp`s (the parser's shape) or the evaluator's own
/// canonical n-ary `Times[...]` `FunctionCall` (what a coefficient survives
/// as once the derivative term next to it has been evaluated). A single
/// non-product expression is its own one-element flattening.
fn flatten_times(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut out = flatten_times(left);
      out.extend(flatten_times(right));
      out
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().flat_map(flatten_times).collect()
    }
    other => vec![other.clone()],
  }
}

/// Whether `expr` contains a call to any of `names` (as a bare function or
/// under a `Derivative[…]`) anywhere in its tree — used to reject a
/// coefficient or boundary value that unexpectedly depends on one of the
/// system's own unknowns, a shape the method-of-lines solver below doesn't
/// attempt to handle.
fn expr_references_any(expr: &Expr, names: &[String]) -> bool {
  for name in names {
    if match_pde_term(expr, name).is_some() {
      return true;
    }
  }
  expr_children(expr)
    .iter()
    .any(|c| expr_references_any(c, names))
}

/// Recognise `[coeff *] Derivative[1, 0][u][t, x] == rhs` (or the reverse,
/// or the coefficient-scaled term on either side) — the evolution equation
/// of a parabolic PDE, optionally scaled by a coefficient (which may depend
/// on `x` or `t`, but not on any unknown) multiplying the time-derivative
/// term, e.g. the local convection velocity in a falling-film transport
/// equation. Returns `(coeff, rhs)`; `coeff` is the literal `1` when the
/// term is bare.
fn try_pde_evolution_term(
  eq: &Expr,
  u_name: &str,
  t_name: &str,
  x_name: &str,
  swap: bool,
) -> Option<(Expr, Expr)> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let is_dudt = |e: &Expr| {
    matches!(
      match_pde_term_roles(e, u_name, swap),
      Some((1, 0, t_arg, x_arg))
        if matches!(&t_arg, Expr::Identifier(n) if n == t_name)
          && matches!(&x_arg, Expr::Identifier(n) if n == x_name)
    )
  };
  // `side` is tried as the (possibly coefficient-scaled) time-derivative
  // term; `other` becomes the raw RHS once `side`'s coefficient is
  // factored out.
  let try_side = |side: &Expr, other: &Expr| -> Option<(Expr, Expr)> {
    if is_dudt(side) {
      return Some((Expr::Integer(1), other.clone()));
    }
    let factors = flatten_times(side);
    if factors.len() < 2 {
      return None;
    }
    let deriv_at = factors.iter().position(is_dudt)?;
    let coeff = factors
      .into_iter()
      .enumerate()
      .filter(|(j, _)| *j != deriv_at)
      .map(|(_, f)| f)
      .reduce(times2)
      .unwrap_or(Expr::Integer(1));
    Some((coeff, other.clone()))
  };
  // `u_t - u_xx == 0`: the time-derivative term is one summand of a side
  // that carries other terms too; those move to the other side.
  let try_summand = |side: &Expr, other: &Expr| -> Option<(Expr, Expr)> {
    let terms = flatten_plus_terms(side);
    if terms.len() < 2 {
      return None;
    }
    let at = terms.iter().position(|t| try_side(t, other).is_some())?;
    let (coeff, _) = try_side(&terms[at], other)?;
    let moved = terms
      .iter()
      .enumerate()
      .filter(|(j, _)| *j != at)
      .map(|(_, t)| neg1(t.clone()))
      .fold(other.clone(), plus2);
    Some((coeff, moved))
  };
  try_side(lhs, rhs)
    .or_else(|| try_side(rhs, lhs))
    .or_else(|| try_summand(lhs, rhs))
    .or_else(|| try_summand(rhs, lhs))
}

/// Recognise `Derivative[2, 0][u][t, x] == rhs` (or the reverse) — the
/// evolution equation of a hyperbolic (second-order-in-time) PDE. Unlike
/// [`try_pde_evolution_term`], no coefficient scaling on the
/// time-derivative term is recognised.
fn try_pde_evolution_term_order2(
  eq: &Expr,
  u_name: &str,
  t_name: &str,
  x_name: &str,
  swap: bool,
) -> Option<Expr> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let is_d2udt2 = |e: &Expr| {
    matches!(
      match_pde_term_roles(e, u_name, swap),
      Some((2, 0, t_arg, x_arg))
        if matches!(&t_arg, Expr::Identifier(n) if n == t_name)
          && matches!(&x_arg, Expr::Identifier(n) if n == x_name)
    )
  };
  if is_d2udt2(lhs) {
    return Some(rhs.clone());
  }
  if is_d2udt2(rhs) {
    return Some(lhs.clone());
  }
  None
}

/// Recognise `u[t0, x] == rhs(x)` (or `u[x, t0]` when `swap` is set, or
/// either reversed) — the initial condition at `t == t0`. Returns the RHS.
fn try_pde_initial_condition(
  eq: &Expr,
  u_name: &str,
  x_name: &str,
  t0: f64,
  swap: bool,
) -> Option<Expr> {
  try_pde_initial_condition_ord(eq, u_name, x_name, t0, 0, swap)
}

/// Like [`try_pde_initial_condition`], but for the `time_order`-th time
/// derivative's initial value: `time_order == 1` recognises
/// `Derivative[1, 0][u][t0, x] == rhs(x)` — the initial *velocity*
/// condition a second-order-in-time (hyperbolic) PDE needs in addition to
/// the ordinary initial value.
fn try_pde_initial_condition_ord(
  eq: &Expr,
  u_name: &str,
  x_name: &str,
  t0: f64,
  time_order: usize,
  swap: bool,
) -> Option<Expr> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let is_ic = |e: &Expr| {
    let Some((dt, dx, t_arg, x_arg)) = match_pde_term_roles(e, u_name, swap)
    else {
      return false;
    };
    dt == time_order
      && dx == 0
      && matches!(&x_arg, Expr::Identifier(n) if n == x_name)
      && nval_to_f64(&t_arg)
        .is_some_and(|v| (v - t0).abs() <= 1e-9 * t0.abs().max(1.0))
  };
  if is_ic(lhs) {
    return Some(rhs.clone());
  }
  if is_ic(rhs) {
    return Some(lhs.clone());
  }
  None
}

/// A boundary condition at one end of the space domain: a Dirichlet value
/// `u(t, x0) = rhs(t)`, or a Neumann flux `∂u/∂x(t, x0) = rhs(t)` — the
/// parsed form of `D[u[t, x], x] /. x -> x0`.
enum PdeBc {
  Dirichlet(Expr),
  Neumann(Expr),
  /// Both ends of the space domain tied together (`u[t, xmin] == u[t,
  /// xmax]`), set at both `bc_lo` and `bc_hi` at once by
  /// `try_pde_periodic_boundary`. Only the hyperbolic solver
  /// (`compile_hyperbolic_boundary`) acts on this; the coupled-system
  /// solver treats it like any other unrecognised shape and bails out.
  Periodic,
}

/// Recognise `u[t, x0] == rhs(t)` (Dirichlet) or
/// `Derivative[0, 1][u][t, x0] == rhs(t)` (Neumann) — or either reversed —
/// a boundary condition at `x == x0`. Returns the condition kind and RHS.
fn try_pde_boundary_condition(
  eq: &Expr,
  u_name: &str,
  t_name: &str,
  x0: f64,
  swap: bool,
) -> Option<PdeBc> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let kind = |e: &Expr| -> Option<fn(Expr) -> PdeBc> {
    let (dt, dx, t_arg, x_arg) = match_pde_term_roles(e, u_name, swap)?;
    if !matches!(&t_arg, Expr::Identifier(n) if n == t_name) {
      return None;
    }
    if !nval_to_f64(&x_arg)
      .is_some_and(|v| (v - x0).abs() <= 1e-9 * x0.abs().max(1.0))
    {
      return None;
    }
    match (dt, dx) {
      (0, 0) => Some(PdeBc::Dirichlet),
      (0, 1) => Some(PdeBc::Neumann),
      _ => None,
    }
  };
  if let Some(ctor) = kind(lhs) {
    return Some(ctor(rhs.clone()));
  }
  if let Some(ctor) = kind(rhs) {
    return Some(ctor(lhs.clone()));
  }
  None
}

/// Recognise `u[t, x_min] == u[t, x_max]` (or reversed) — a periodic
/// boundary tying both ends of the space domain together, rather than one
/// end's value to a function of time. Unlike `try_pde_boundary_condition`,
/// a single matching equation supplies both `bc_lo` and `bc_hi` at once, so
/// this must be tried before that function: `u[t, x_max]` also happens to
/// satisfy `try_pde_boundary_condition`'s shape at `x_min` (matching `u[t,
/// x_min]`) with the *unevaluated* call `u[t, x_max]` as its "value", which
/// silently consumes the equation as a bogus Dirichlet condition and leaves
/// the other end unmatched.
fn try_pde_periodic_boundary(
  eq: &Expr,
  u_name: &str,
  t_name: &str,
  x_min: f64,
  x_max: f64,
  swap: bool,
) -> bool {
  let Some((lhs, rhs)) = as_equal_pair(eq) else {
    return false;
  };
  let at_end = |e: &Expr, x0: f64| -> bool {
    let Some((dt, dx, t_arg, x_arg)) = match_pde_term_roles(e, u_name, swap)
    else {
      return false;
    };
    dt == 0
      && dx == 0
      && matches!(&t_arg, Expr::Identifier(n) if n == t_name)
      && nval_to_f64(&x_arg)
        .is_some_and(|v| (v - x0).abs() <= 1e-9 * x0.abs().max(1.0))
  };
  (at_end(lhs, x_min) && at_end(rhs, x_max))
    || (at_end(lhs, x_max) && at_end(rhs, x_min))
}

/// Rewrite every unknown's occurrences in a PDE evolution right-hand side
/// into the placeholder identifiers `NumFn` compiles the finite-difference
/// stencil's numeric values into (see `ndsolve_pde`): `u_names[i][t, x]`
/// becomes `NDSolve$U{i}` for every `i` (bare value — how a reaction term
/// couples to another unknown), while `self_idx`'s own
/// `Derivative[0, 1][…][t, x]` and `Derivative[0, 2][…][t, x]` become
/// `NDSolve$UX`/`NDSolve$UXX`. Returns `None` when some other unknown
/// appears through a derivative (a coupling shape this solver doesn't
/// support) or any other construct this rewrite can't place.
fn rewrite_pde_rhs(
  expr: &Expr,
  u_names: &[String],
  self_idx: usize,
  t_name: &str,
  x_name: &str,
  swap: bool,
) -> Option<Expr> {
  for (j, name) in u_names.iter().enumerate() {
    let Some((dt, dx, t_arg, x_arg)) = match_pde_term_roles(expr, name, swap)
    else {
      continue;
    };
    if !matches!(&t_arg, Expr::Identifier(n) if n == t_name)
      || !matches!(&x_arg, Expr::Identifier(n) if n == x_name)
    {
      return None;
    }
    let placeholder = if j == self_idx {
      match (dt, dx) {
        (0, 0) => Some(format!("NDSolve$U{j}")),
        (0, 1) => Some("NDSolve$UX".to_string()),
        (0, 2) => Some("NDSolve$UXX".to_string()),
        _ => None,
      }
    } else {
      // Coupling to another unknown is only supported through its plain
      // value, not its derivatives.
      match (dt, dx) {
        (0, 0) => Some(format!("NDSolve$U{j}")),
        _ => None,
      }
    };
    return placeholder.map(Expr::Identifier);
  }
  let failed = std::cell::Cell::new(false);
  let mapped = map_children(expr, &|c| {
    if failed.get() {
      return c.clone();
    }
    if let Some(r) = rewrite_pde_rhs(c, u_names, self_idx, t_name, x_name, swap)
    {
      r
    } else {
      failed.set(true);
      c.clone()
    }
  });
  if failed.get() { None } else { Some(mapped) }
}

/// Rewrite every unknown's bare occurrence at the boundary point `x0`
/// (`u_names[j][t_name, x0]`, in whichever argument order `swap` selects)
/// into the placeholder identifier `NDSolve$U{j}` — how a Neumann boundary
/// condition's right-hand side couples to the unknowns' own current
/// boundary values, e.g. a Robin-type flux condition
/// `Λ Exp[-α (a[0, t] - b[0, t])]` that depends on both species'
/// concentrations at the electrode surface. The method-of-lines
/// integrator (`try_solve_pde_system`'s `derivative` closure) evaluates
/// the compiled result each step from the already-known grid values at
/// that boundary. Returns `None` when some unknown appears at a point
/// other than `x0`, or through a derivative — shapes this boundary solver
/// doesn't attempt.
fn rewrite_pde_bc_rhs(
  expr: &Expr,
  u_names: &[String],
  t_name: &str,
  x0: f64,
  swap: bool,
) -> Option<Expr> {
  for (j, name) in u_names.iter().enumerate() {
    if let Some((dt, dx, t_arg, x_arg)) = match_pde_term_roles(expr, name, swap)
    {
      if dt != 0
        || dx != 0
        || !matches!(&t_arg, Expr::Identifier(n) if n == t_name)
        || !nval_to_f64(&x_arg)
          .is_some_and(|v| (v - x0).abs() <= 1e-9 * x0.abs().max(1.0))
      {
        return None;
      }
      return Some(Expr::Identifier(format!("NDSolve$U{j}")));
    }
  }
  let failed = std::cell::Cell::new(false);
  let mapped = map_children(expr, &|c| {
    if failed.get() {
      return c.clone();
    }
    if let Some(r) = rewrite_pde_bc_rhs(c, u_names, t_name, x0, swap) {
      r
    } else {
      failed.set(true);
      c.clone()
    }
  });
  if failed.get() { None } else { Some(mapped) }
}

/// The additive terms of `expr` — a `Plus` call or the parser's `+`
/// `BinaryOp` chain — or `expr` itself as the single term.
fn flatten_plus_terms(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut out = flatten_plus_terms(left);
      out.extend(flatten_plus_terms(right));
      out
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().flat_map(flatten_plus_terms).collect()
    }
    other => vec![other.clone()],
  }
}

/// `Derivative[orders][u][args]` — the parsed form of a partial derivative.
fn pde_derivative_call(u_name: &str, orders: &[i64], args: Vec<Expr>) -> Expr {
  Expr::CurriedCall {
    func: Box::new(Expr::CurriedCall {
      func: Box::new(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: orders.iter().map(|o| Expr::Integer(*o as _)).collect(),
      }),
      args: vec![Expr::Identifier(u_name.to_string())],
    }),
    args,
  }
}

/// Replace every `Derivative[orders][u][…]` call in `expr` by `with`.
fn replace_pde_derivative(
  expr: &Expr,
  u_name: &str,
  orders: &[i64],
  with: &Expr,
) -> Expr {
  if let Expr::CurriedCall { func, args } = expr
    && let Expr::CurriedCall {
      func: head,
      args: fname,
    } = func.as_ref()
    && matches!(fname.as_slice(), [Expr::Identifier(n)] if n == u_name)
    && let Expr::FunctionCall { name, args: o } = head.as_ref()
    && name == "Derivative"
    && o.len() == orders.len()
    && o
      .iter()
      .zip(orders)
      .all(|(a, b)| matches!(a, Expr::Integer(i) if *i as i64 == *b))
    && !args.is_empty()
  {
    return with.clone();
  }
  map_children(expr, &|c| replace_pde_derivative(c, u_name, orders, with))
}

/// The call-argument position at which `u_name` is called with the plain
/// identifier `x_name` somewhere in `expr` (the space variable's slot).
fn pde_space_position(
  expr: &Expr,
  u_name: &str,
  x_name: &str,
) -> Option<usize> {
  if let Expr::FunctionCall { name, args } = expr
    && name == u_name
    && let Some(p) = args
      .iter()
      .position(|a| matches!(a, Expr::Identifier(n) if n == x_name))
  {
    return Some(p);
  }
  if let Expr::CurriedCall { func, args } = expr
    && let Expr::CurriedCall { args: fname, .. } = func.as_ref()
    && matches!(fname.as_slice(), [Expr::Identifier(n)] if n == u_name)
    && let Some(p) = args
      .iter()
      .position(|a| matches!(a, Expr::Identifier(n) if n == x_name))
  {
    return Some(p);
  }
  expr_children(expr)
    .iter()
    .find_map(|c| pde_space_position(c, u_name, x_name))
}

/// Replace every `If[cond, a, a]` whose two branches are identical by `a`.
fn collapse_identical_if_branches(expr: &Expr) -> Expr {
  let mapped = map_children(expr, &collapse_identical_if_branches);
  if let Expr::FunctionCall { name, args } = &mapped
    && name == "If"
    && args.len() == 3
    && crate::syntax::expr_to_string(&args[1])
      == crate::syntax::expr_to_string(&args[2])
  {
    return args[1].clone();
  }
  mapped
}

/// Lower `NeumannValue[g, x == x0]` terms in PDE equations to explicit
/// Neumann boundary equations. A term `NeumannValue[g, x == x0]` on the
/// right-hand side of `… - c u_xx … == rest + NeumannValue[…]` is the
/// natural boundary condition `n·(c ∇u) = g` with `n` the outward normal,
/// i.e. `c u_x = -g` at the lower end and `c u_x = g` at the upper end of
/// the space domain. The PDE keeps `rest`. An unknown that has no spatial
/// derivative at all (a coupled ODE evolving independently at every grid
/// point) gets a homogeneous Neumann condition at both ends, which the
/// method-of-lines grid needs but which does not influence it.
///
/// Returns `None` when there is no `NeumannValue` or it is not in the
/// supported shape.
fn lower_neumann_values(
  eqs: &[Expr],
  u_names: &[String],
  doms: &[&PdeDomain],
) -> Option<(Vec<Expr>, String)> {
  if !eqs.iter().any(|e| mentions_head(e, "NeumannValue")) {
    return None;
  }
  let mut out: Vec<Expr> = Vec::new();
  let mut x_info: Option<(String, f64, f64)> = None;
  let mut has_bc = vec![false; u_names.len()];
  for eq in eqs {
    if !mentions_head(eq, "NeumannValue") {
      out.push(eq.clone());
      continue;
    }
    let (lhs, rhs) = as_equal_pair(eq)?;
    let (lhs, rhs) = if flatten_plus_terms(rhs)
      .iter()
      .any(|t| matches!(t, Expr::FunctionCall { name, .. } if name == "NeumannValue"))
    {
      (lhs, rhs)
    } else {
      (rhs, lhs)
    };
    let (mut nvs, mut rest) = (Vec::new(), Vec::new());
    for t in flatten_plus_terms(rhs) {
      match &t {
        Expr::FunctionCall { name, args }
          if name == "NeumannValue" && args.len() == 2 =>
        {
          nvs.push((args[0].clone(), args[1].clone()));
        }
        _ => rest.push(t),
      }
    }
    if mentions_head(lhs, "NeumannValue") {
      return None;
    }
    let rest_expr = if rest.is_empty() {
      Expr::Integer(0)
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: rest.into(),
      }
    };
    let residual = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        lhs.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), rest_expr.clone()].into(),
        },
      ]
      .into(),
    };
    for (g, pred) in nvs {
      let (x_name, x0_expr) = match as_equal_pair(&pred)? {
        (Expr::Identifier(n), v) | (v, Expr::Identifier(n)) => {
          (n.clone(), v.clone())
        }
        _ => return None,
      };
      let dom = doms.iter().find(|d| d.name == x_name)?;
      let x0 = nval_to_f64(&x0_expr)?;
      let is_lo = (x0 - dom.min).abs() <= 1e-9 * dom.min.abs().max(1.0);
      let is_hi = (x0 - dom.max).abs() <= 1e-9 * dom.max.abs().max(1.0);
      if !is_lo && !is_hi {
        return None;
      }
      // The unknown whose equation this is, and the coefficient c of its
      // second space derivative (`-c u_xx` in the residual).
      let mut found = None;
      for (i, u) in u_names.iter().enumerate() {
        let Some(p) = pde_space_position(&residual, u, &x_name) else {
          continue;
        };
        let mut orders = vec![0i64; 2];
        orders[p] = 2;
        let sym = Expr::Identifier("NDSolve$Uxx".to_string());
        let replaced = replace_pde_derivative(&residual, u, &orders, &sym);
        let coef =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "D".to_string(),
            args: vec![replaced, sym.clone()].into(),
          })
          .ok()?;
        // `D` of a piecewise coefficient leaves `sym*If[cond, 0, 0]`
        // behind; collapse such branches so linearity can be recognised.
        let coef = crate::evaluator::evaluate_expr_to_expr(
          &collapse_identical_if_branches(&coef),
        )
        .ok()?;
        if !matches!(&coef, Expr::Integer(0))
          && !expr_contains_ident(&coef, "NDSolve$Uxx")
        {
          found = Some((i, p, coef));
          break;
        }
      }
      let (ui, p, coef) = found?;
      let c = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), coef].into(),
      })
      .ok()?;
      let at_x0 =
        |e: &Expr| crate::syntax::substitute_variable(e, &x_name, &x0_expr);
      let flux = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(if is_lo { -1 } else { 1 }),
          at_x0(&g),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![at_x0(&c), Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      })
      .ok()?;
      // The call arguments of the unknown, with the space slot fixed to x0.
      let t_name = doms.iter().find(|d| d.name != x_name)?.name.clone();
      let mut call_args = vec![Expr::Identifier(t_name); 2];
      call_args[p] = x0_expr.clone();
      let mut orders = vec![0i64; 2];
      orders[p] = 1;
      out.push(Expr::Comparison {
        operands: vec![
          pde_derivative_call(&u_names[ui], &orders, call_args),
          flux,
        ],
        operators: vec![ComparisonOp::Equal],
      });
      has_bc[ui] = true;
      x_info = Some((x_name, dom.min, dom.max));
    }
    out.push(Expr::Comparison {
      operands: vec![lhs.clone(), rest_expr],
      operators: vec![ComparisonOp::Equal],
    });
  }
  let (x_name, x_min, x_max) = x_info?;
  let t_name = doms.iter().find(|d| d.name != x_name)?.name.clone();
  for (i, u) in u_names.iter().enumerate() {
    if has_bc[i] {
      continue;
    }
    let p = out.iter().find_map(|e| pde_space_position(e, u, &x_name))?;
    // A spatially coupled unknown needs boundary conditions of its own.
    let mut orders = vec![0i64; 2];
    orders[p] = 1;
    let probe = Expr::Identifier("NDSolve$Ux".to_string());
    if out.iter().any(|e| {
      expr_contains_ident(
        &replace_pde_derivative(e, u, &orders, &probe),
        "NDSolve$Ux",
      )
    }) {
      return None;
    }
    for x0 in [x_min, x_max] {
      let mut call_args = vec![Expr::Identifier(t_name.clone()); 2];
      call_args[p] = Expr::Real(x0);
      out.push(Expr::Comparison {
        operands: vec![
          pde_derivative_call(u, &orders, call_args),
          Expr::Integer(0),
        ],
        operators: vec![ComparisonOp::Equal],
      });
    }
  }
  Some((out, x_name))
}

/// `NDSolve[eqns, u, {v1, v1min, v1max}, {v2, v2min, v2max}]` — one or
/// several coupled parabolic PDEs in one space dimension (a
/// diffusion/reaction/convection shape), solved by the method of lines:
/// the space-like variable is discretized onto a uniform grid, the spatial
/// derivatives become central finite-difference stencils, and the
/// resulting ODE system in the time-like variable is integrated with
/// classical RK4 subject to a diffusive stability bound on the time step.
/// Each unknown's evolution equation may carry a coefficient (constant or
/// depending on the space variable) on its time derivative, and may
/// reference every other unknown's plain value (not its derivatives) for
/// reaction coupling; each unknown needs exactly one initial condition and
/// one boundary condition — Dirichlet or Neumann — at each end of the space
/// domain. Which of the two declared domains plays the time-like
/// (integrated) role and which plays the space-like (gridded) role isn't
/// fixed by argument order, so both assignments are tried.
///
/// Returns `Ok(None)` when the equations aren't in that shape, so the
/// caller falls back to leaving the call unevaluated.
fn ndsolve_pde(args: &[Expr]) -> Result<Option<Expr>, InterpreterError> {
  let u_names: Vec<String> = match &args[1] {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::List(items) => {
      let mut names = Vec::with_capacity(items.len());
      for item in items {
        match item {
          Expr::Identifier(name) => names.push(name.clone()),
          _ => return Ok(None),
        }
      }
      names
    }
    _ => return Ok(None),
  };
  if u_names.is_empty() {
    return Ok(None);
  }

  let Some(dom_a) = parse_pde_domain(&args[2]) else {
    return Ok(None);
  };
  let Some(dom_b) = parse_pde_domain(&args[3]) else {
    return Ok(None);
  };

  let eq_items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  let eq_items = flatten_chained_equalities(&eq_items);
  // The finite-element form (`NeumannValue`) returns its solutions as
  // functions of the space variable first, then time, like the unknowns'
  // own `u[x, t]` calls, whatever order the domains were declared in.
  let (eq_items, dim0_dom) =
    match lower_neumann_values(&eq_items, &u_names, &[&dom_a, &dom_b]) {
      Some((lowered, x_name)) => {
        let dim0 = if dom_a.name == x_name { &dom_a } else { &dom_b };
        (lowered, dim0.clone())
      }
      None => (eq_items, dom_a.clone()),
    };
  if eq_items.len() != 4 * u_names.len() {
    return ndsolve_pde_components(args, &u_names, &eq_items, &dom_a, &dom_b);
  }

  for (t_dom, x_dom) in [(&dom_a, &dom_b), (&dom_b, &dom_a)] {
    for swap in [false, true] {
      if let Some(result) = try_solve_pde_system(
        &eq_items, &u_names, t_dom, x_dom, &dim0_dom, swap,
      )? {
        return Ok(Some(result));
      }
    }
  }
  Ok(None)
}

/// `NDSolve` PDE systems that mix unknown groups of different kinds — e.g.
/// a plug-flow model `c[x, y]`, `t[x, y]` that only has an `x` derivative
/// (so `y` is a mere parameter) next to diffusive models that also have
/// second `y` derivatives. The unknowns are partitioned into groups that
/// share equations; a group with four equations per unknown is a regular
/// evolution PDE system and goes to [`ndsolve_pde`], while a group with no
/// derivative in the other variable is an ODE in the evolution variable
/// ([`solve_parameter_ode_group`]). Returns `Ok(None)` if any group fits
/// neither.
fn ndsolve_pde_components(
  args: &[Expr],
  u_names: &[String],
  eq_items: &[Expr],
  dom_a: &PdeDomain,
  dom_b: &PdeDomain,
) -> Result<Option<Expr>, InterpreterError> {
  // Union-find over unknowns: equations mentioning several unknowns couple
  // them.
  let mut parent: Vec<usize> = (0..u_names.len()).collect();
  fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
      r = parent[r];
    }
    parent[i] = r;
    r
  }
  let mut eq_groups: Vec<Vec<usize>> = Vec::with_capacity(eq_items.len());
  for eq in eq_items {
    let used: Vec<usize> = (0..u_names.len())
      .filter(|&i| {
        mentions_head(eq, &u_names[i]) || expr_contains_ident(eq, &u_names[i])
      })
      .collect();
    if used.is_empty() {
      return Ok(None);
    }
    for &i in &used[1..] {
      let (ra, rb) = (find(&mut parent, used[0]), find(&mut parent, i));
      parent[rb] = ra;
    }
    eq_groups.push(used);
  }
  let mut roots: Vec<usize> = Vec::new();
  for i in 0..u_names.len() {
    let r = find(&mut parent, i);
    if !roots.contains(&r) {
      roots.push(r);
    }
  }
  let mut solved: Vec<Option<Expr>> = vec![None; u_names.len()];
  for root in roots {
    let members: Vec<usize> = (0..u_names.len())
      .filter(|&i| find(&mut parent, i) == root)
      .collect();
    let eqs: Vec<Expr> = eq_items
      .iter()
      .zip(&eq_groups)
      .filter(|(_, g)| find(&mut parent, g[0]) == root)
      .map(|(e, _)| e.clone())
      .collect();
    let names: Vec<String> =
      members.iter().map(|&i| u_names[i].clone()).collect();
    let rules = if eqs.len() == 4 * names.len() {
      let sub_args = [
        Expr::List(eqs.into()),
        Expr::List(names.iter().map(|n| Expr::Identifier(n.clone())).collect()),
        args[2].clone(),
        args[3].clone(),
      ];
      let Some(result) = ndsolve_pde(&sub_args)? else {
        return Ok(None);
      };
      pde_result_rules(&result)
    } else {
      solve_parameter_ode_group(args, &names, &eqs, dom_a, dom_b)?
    };
    let Some(rules) = rules else {
      return Ok(None);
    };
    for (name, interp) in rules {
      if let Some(idx) = u_names.iter().position(|n| *n == name) {
        solved[idx] = Some(interp);
      }
    }
  }
  let mut out = Vec::with_capacity(u_names.len());
  for (name, interp) in u_names.iter().zip(solved) {
    let Some(interp) = interp else {
      return Ok(None);
    };
    out.push(rule_expr(Expr::Identifier(name.clone()), interp));
  }
  Ok(Some(Expr::List(vec![Expr::List(out.into())].into())))
}

/// Unpack `{{name -> value, …}}` into `(name, value)` pairs.
fn pde_result_rules(result: &Expr) -> Option<Vec<(String, Expr)>> {
  let Expr::List(outer) = result else {
    return None;
  };
  let Expr::List(rules) = outer.first()? else {
    return None;
  };
  rules
    .iter()
    .map(|r| match r {
      Expr::Rule {
        pattern,
        replacement,
      } => match pattern.as_ref() {
        Expr::Identifier(n) => Some((n.clone(), replacement.as_ref().clone())),
        _ => None,
      },
      _ => None,
    })
    .collect()
}

/// Rewrite `u[t, p]` and `Derivative[k, 0][u][t, p]` into the ordinary
/// `u[t]` / `Derivative[k][u][t]` for every name in `names`, where `p` is
/// the parameter variable. Anything else involving `p` (a derivative in
/// `p`, `p` in a coefficient) clears `ok`.
fn drop_parameter_argument(
  expr: &Expr,
  names: &[String],
  param: &str,
  ok: &std::cell::Cell<bool>,
) -> Expr {
  let is_param = |e: &Expr| matches!(e, Expr::Identifier(n) if n == param);
  if let Expr::FunctionCall { name, args } = expr
    && names.contains(name)
    && args.len() == 2
    && is_param(&args[1])
  {
    return Expr::FunctionCall {
      name: name.clone(),
      args: vec![args[0].clone()].into(),
    };
  }
  if let Expr::CurriedCall { func, args } = expr
    && args.len() == 2
    && is_param(&args[1])
    && let Expr::CurriedCall {
      func: deriv_head,
      args: fname_args,
    } = func.as_ref()
    && fname_args.len() == 1
    && let Expr::Identifier(fname) = &fname_args[0]
    && names.contains(fname)
    && let Expr::FunctionCall {
      name: deriv_name,
      args: orders,
    } = deriv_head.as_ref()
    && deriv_name == "Derivative"
    && orders.len() == 2
    && let (Expr::Integer(k), Expr::Integer(0)) = (&orders[0], &orders[1])
  {
    let inner = args[0].clone();
    if *k == 0 {
      return Expr::FunctionCall {
        name: fname.clone(),
        args: vec![inner].into(),
      };
    }
    return Expr::CurriedCall {
      func: Box::new(Expr::CurriedCall {
        func: Box::new(call("Derivative", vec![Expr::Integer(*k)])),
        args: vec![Expr::Identifier(fname.clone())],
      }),
      args: vec![inner],
    };
  }
  if is_param(expr) {
    ok.set(false);
    return expr.clone();
  }
  map_children(expr, &|c| drop_parameter_argument(c, names, param, ok))
}

/// A group of unknowns `u[t, p]` whose equations only differentiate in `t`
/// (and never mention `p` otherwise): an ODE in `t` for each value of the
/// parameter `p`, identical for all of them. Solved once as an ODE and
/// returned as `InterpolatingFunction`s over both variables, constant in
/// `p`.
fn solve_parameter_ode_group(
  args: &[Expr],
  names: &[String],
  eqs: &[Expr],
  dom_a: &PdeDomain,
  dom_b: &PdeDomain,
) -> Result<Option<Vec<(String, Expr)>>, InterpreterError> {
  for (t_dom, p_dom, t_arg) in
    [(dom_a, dom_b, &args[2]), (dom_b, dom_a, &args[3])]
  {
    let ok = std::cell::Cell::new(true);
    let ode_eqs: Vec<Expr> = eqs
      .iter()
      .map(|e| drop_parameter_argument(e, names, &p_dom.name, &ok))
      .collect();
    if !ok.get() {
      continue;
    }
    let ode_args = [
      Expr::List(ode_eqs.into()),
      Expr::List(names.iter().map(|n| Expr::Identifier(n.clone())).collect()),
      t_arg.clone(),
    ];
    let Ok(ode_result) = ndsolve_ast_inner(&ode_args) else {
      continue;
    };
    let Some(result) = pde_result_rules(&ode_result) else {
      continue;
    };
    let t_is_dim0 = t_dom.name == dom_a.name;
    let mut out = Vec::with_capacity(result.len());
    for (name, interp) in result {
      let Expr::FunctionCall {
        name: head,
        args: if_args,
      } = &interp
      else {
        return Ok(None);
      };
      if head != "InterpolatingFunction" || if_args.len() < 2 {
        return Ok(None);
      }
      let Expr::List(points) = &if_args[1] else {
        return Ok(None);
      };
      let mut ts = Vec::with_capacity(points.len());
      let mut vals = Vec::with_capacity(points.len());
      for p in points {
        let Expr::List(pair) = p else {
          return Ok(None);
        };
        if pair.len() != 2 {
          return Ok(None);
        }
        ts.push(expr_to_f64(&pair[0])?);
        vals.push(expr_to_f64(&pair[1])?);
      }
      let p_coords = [p_dom.min, p_dom.max];
      let real_list =
        |v: &[f64]| Expr::List(v.iter().map(|x| Expr::Real(*x)).collect());
      let range = |lo: f64, hi: f64| {
        Expr::List(vec![Expr::Real(lo), Expr::Real(hi)].into())
      };
      // `[t][p]` grid: every parameter column equals the ODE value.
      let tp_grid: Vec<Vec<f64>> = vals.iter().map(|v| vec![*v, *v]).collect();
      let (grid, dom0, dom1, coords0, coords1) = if t_is_dim0 {
        (
          tp_grid,
          range(t_dom.min, t_dom.max),
          range(p_dom.min, p_dom.max),
          real_list(&ts),
          real_list(&p_coords),
        )
      } else {
        (
          transpose_grid(&tp_grid),
          range(p_dom.min, p_dom.max),
          range(t_dom.min, t_dom.max),
          real_list(&p_coords),
          real_list(&ts),
        )
      };
      let grid_expr =
        Expr::List(grid.iter().map(|row| real_list(row)).collect());
      out.push((
        name,
        call(
          "InterpolatingFunction",
          vec![
            Expr::List(vec![dom0, dom1].into()),
            grid_expr,
            Expr::List(vec![Expr::Integer(1), Expr::Integer(1)].into()),
            Expr::List(vec![coords0, coords1].into()),
          ],
        ),
      ));
    }
    return Ok(Some(out));
  }
  Ok(None)
}

/// Expand a chained equality `e0 == e1 == ... == e_{k-1}` (`k > 2`
/// operands, every operator `==`) into the `k - 1` pairwise equations
/// `e_i == e_{k-1}` for `i` from `0` to `k - 2`. `NDSolve` demonstrations
/// commonly use this shorthand — both for a PDE's initial/boundary
/// conditions (e.g. `u[x, t0] == u[xmax, t] == c`) and for an ODE system's
/// initial conditions (e.g. `x[0] == y[0] == z[0] == 0`). Transitivity
/// makes every pairwise equation a valid consequence of the chain, so this
/// expansion is lossless. Equations that aren't a longer chain pass through
/// unchanged.
fn flatten_chained_equalities(items: &[Expr]) -> Vec<Expr> {
  let mut out = Vec::with_capacity(items.len());
  for item in items {
    if let Expr::Comparison {
      operands,
      operators,
    } = item
      && operands.len() > 2
      && operators.iter().all(|op| *op == ComparisonOp::Equal)
    {
      let last = operands.len() - 1;
      for operand in &operands[..last] {
        out.push(Expr::Comparison {
          operands: vec![operand.clone(), operands[last].clone()],
          operators: vec![ComparisonOp::Equal],
        });
      }
    } else {
      out.push(item.clone());
    }
  }
  out
}

/// One unknown's compiled evolution/initial/boundary functions.
struct PdeUnknown {
  /// Coefficient on the evolution equation's time-derivative term,
  /// compiled over `[t_name, x_name]`.
  coeff_fn: NumFn,
  /// The evolution equation's right-hand side (before dividing by
  /// `coeff_fn`), compiled over `["NDSolve$U0", …, "NDSolve$U{n-1}",
  /// "NDSolve$UX", "NDSolve$UXX", t_name, x_name]`.
  rhs_fn: NumFn,
  /// Initial profile, compiled over `[x_name]`.
  ic_fn: NumFn,
  bc_lo: PdeBoundaryFn,
  bc_hi: PdeBoundaryFn,
}

enum PdeBoundaryFn {
  /// The grid point at this boundary is fixed to `NumFn`'s value each step.
  Dirichlet(NumFn),
  /// The grid point at this boundary is integrated like an interior point,
  /// using a ghost point built from `NumFn`'s flux value each step.
  Neumann(NumFn),
  /// This end wraps around to the other end of the domain, which carries
  /// the same value (`u[t, xmin] == u[t, xmax]`); only the hyperbolic
  /// solver produces this.
  Periodic,
}

/// Attempt to match and solve the coupled PDE system with `t_dom` as the
/// time-like (integrated) domain and `x_dom` as the space-like (gridded)
/// one. `dim0_dom` is whichever of the two domains was declared first in
/// the `NDSolve` call (`args[2]`) — the output `InterpolatingFunction`s'
/// first dimension must match it regardless of which domain plays which
/// solving role. Returns `Ok(None)` when the equations don't fit this role
/// assignment.
/// Whether a PDE solver keeps the current time row besides its regular
/// stride: once the state has moved by more than 1% of its initial scale
/// since the last kept row (bounded to 1000 rows), so a fast early
/// transient — a decaying mode — is sampled densely enough to interpolate.
fn pde_row_moved(
  state: &[f64],
  last_kept: &[f64],
  scale: f64,
  rows: usize,
) -> bool {
  rows < 1000
    && state
      .iter()
      .zip(last_kept)
      .any(|(a, b)| (a - b).abs() > 0.01 * scale)
}

fn try_solve_pde_system(
  eq_items: &[Expr],
  u_names: &[String],
  t_dom: &PdeDomain,
  x_dom: &PdeDomain,
  dim0_dom: &PdeDomain,
  swap: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let n = u_names.len();
  let mut claimed = vec![false; eq_items.len()];
  struct RawSpec {
    coeff: Expr,
    rhs: Expr,
    ic: Expr,
    bc_lo: PdeBc,
    bc_hi: PdeBc,
  }

  /// Try to match `eq` as a linear flux-conservation boundary condition
  /// shared between several unknowns at the same end of the space domain
  /// — e.g. `D[a[x, t], x] + dBA D[b[x, t], x] == 0` at `x == 0`, the
  /// no-accumulation condition tying two coupled species' fluxes at a
  /// reacting interface — rather than a single unknown's own condition.
  /// Requires one side of the equation to be identically zero, and every
  /// unknown referenced besides `target` (at index `target_idx`, not yet
  /// in `raw_specs`) to already have its own Neumann condition resolved at
  /// this same end (`raw_specs[j].bc_lo`/`bc_hi`, one entry per unknown
  /// processed so far). Solving the linear relation for `target`'s flux —
  /// substituting each other referenced unknown's already-known flux
  /// expression — yields `target`'s own (derived) Neumann condition.
  /// Returns `None` when the equation isn't in this shape, or references
  /// an unknown whose own condition at this end isn't resolved yet.
  fn try_coupled_flux_bc(
    eq: &Expr,
    target_idx: usize,
    u_names: &[String],
    raw_specs: &[RawSpec],
    t_name: &str,
    x0: f64,
    at_lo: bool,
    swap: bool,
  ) -> Option<PdeBc> {
    fn is_zero_literal(e: &Expr) -> bool {
      match e {
        Expr::Integer(0) => true,
        Expr::Real(v) => v.abs() < 1e-12,
        _ => false,
      }
    }
    fn is_one_literal(e: &Expr) -> bool {
      matches!(e, Expr::Integer(1))
    }
    fn neg_expr(e: Expr) -> Expr {
      if is_zero_literal(&e) {
        return e;
      }
      neg1(e)
    }
    fn add_expr(a: Expr, b: Expr) -> Expr {
      plus2(a, b)
    }
    fn mul_expr(coeff: Expr, e: Expr) -> Expr {
      if is_one_literal(&coeff) {
        return e;
      }
      times2(coeff, e)
    }
    fn flatten_additive_signed(
      expr: &Expr,
      sign: f64,
      out: &mut Vec<(f64, Expr)>,
    ) {
      match expr {
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left,
          right,
        } => {
          flatten_additive_signed(left, sign, out);
          flatten_additive_signed(right, sign, out);
        }
        Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left,
          right,
        } => {
          flatten_additive_signed(left, sign, out);
          flatten_additive_signed(right, -sign, out);
        }
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } => {
          flatten_additive_signed(operand, -sign, out);
        }
        Expr::FunctionCall { name, args } if name == "Plus" => {
          for a in args {
            flatten_additive_signed(a, sign, out);
          }
        }
        other => out.push((sign, other.clone())),
      }
    }
    /// Match a single additive term as `coeff * D[u_name, x]` (any
    /// factor order, coefficient 1 when the term is a bare derivative),
    /// returning `(coeff, t_arg, x_arg)`.
    fn match_flux_term(
      term: &Expr,
      u_name: &str,
      swap: bool,
    ) -> Option<(Expr, Expr, Expr)> {
      let factors = flatten_times(term);
      let mut deriv: Option<(Expr, Expr)> = None;
      let mut coeff_factors = Vec::new();
      for f in &factors {
        if deriv.is_none()
          && let Some((dt, dx, t_arg, x_arg)) =
            match_pde_term_roles(f, u_name, swap)
          && dt == 0
          && dx == 1
        {
          deriv = Some((t_arg, x_arg));
        } else {
          coeff_factors.push(f.clone());
        }
      }
      let (t_arg, x_arg) = deriv?;
      let coeff = match coeff_factors.len() {
        0 => Expr::Integer(1),
        1 => coeff_factors.into_iter().next().unwrap(),
        _ => times(coeff_factors),
      };
      Some((coeff, t_arg, x_arg))
    }

    let (lhs, rhs) = as_equal_pair(eq)?;
    let sum_side = if is_zero_literal(lhs) {
      rhs
    } else if is_zero_literal(rhs) {
      lhs
    } else {
      return None;
    };
    let mut terms = Vec::new();
    flatten_additive_signed(sum_side, 1.0, &mut terms);

    let mut self_coeff: Option<Expr> = None;
    let mut other_terms: Vec<Expr> = Vec::new();
    for (sign, term) in &terms {
      let mut matched = false;
      for (j, name) in u_names.iter().enumerate() {
        let Some((coeff, t_arg, x_arg)) = match_flux_term(term, name, swap)
        else {
          continue;
        };
        if !matches!(&t_arg, Expr::Identifier(n) if n == t_name)
          || !nval_to_f64(&x_arg)
            .is_some_and(|v| (v - x0).abs() <= 1e-9 * x0.abs().max(1.0))
        {
          return None;
        }
        let signed_coeff = if *sign > 0.0 { coeff } else { neg_expr(coeff) };
        if j == target_idx {
          if self_coeff.is_some() {
            return None; // the target's own flux appears more than once.
          }
          self_coeff = Some(signed_coeff);
        } else {
          if j >= raw_specs.len() {
            return None; // that unknown's own condition isn't resolved yet.
          }
          let other_bc = if at_lo {
            &raw_specs[j].bc_lo
          } else {
            &raw_specs[j].bc_hi
          };
          let PdeBc::Neumann(other_flux_expr) = other_bc else {
            return None; // a Dirichlet boundary has no flux to substitute.
          };
          other_terms.push(mul_expr(signed_coeff, other_flux_expr.clone()));
        }
        matched = true;
        break;
      }
      if !matched {
        return None; // an unrecognised remainder term — outside this shape.
      }
    }
    let self_coeff = self_coeff?;
    let numerator = other_terms
      .into_iter()
      .reduce(add_expr)
      .unwrap_or(Expr::Integer(0));
    let negated = neg_expr(numerator);
    let derived = if is_one_literal(&self_coeff) {
      negated
    } else {
      div2(negated, self_coeff)
    };
    Some(PdeBc::Neumann(derived))
  }

  let mut raw_specs: Vec<RawSpec> = Vec::with_capacity(n);
  for name in u_names {
    let mut coeff_rhs: Option<(Expr, Expr)> = None;
    let mut ic: Option<Expr> = None;
    let mut bc_lo: Option<PdeBc> = None;
    let mut bc_hi: Option<PdeBc> = None;
    for (idx, eq) in eq_items.iter().enumerate() {
      if claimed[idx] {
        continue;
      }
      if coeff_rhs.is_none()
        && let Some((coeff, rhs)) =
          try_pde_evolution_term(eq, name, &t_dom.name, &x_dom.name, swap)
        && !expr_references_any(&coeff, u_names)
      {
        coeff_rhs = Some((coeff, rhs));
        claimed[idx] = true;
        continue;
      }
      if ic.is_none()
        && let Some(rhs) =
          try_pde_initial_condition(eq, name, &x_dom.name, t_dom.min, swap)
      {
        ic = Some(rhs);
        claimed[idx] = true;
        continue;
      }
      if bc_lo.is_none()
        && let Some(bc) =
          try_pde_boundary_condition(eq, name, &t_dom.name, x_dom.min, swap)
      {
        bc_lo = Some(bc);
        claimed[idx] = true;
        continue;
      }
      if bc_hi.is_none()
        && let Some(bc) =
          try_pde_boundary_condition(eq, name, &t_dom.name, x_dom.max, swap)
      {
        bc_hi = Some(bc);
        claimed[idx] = true;
        continue;
      }
      if bc_lo.is_none()
        && let Some(bc) = try_coupled_flux_bc(
          eq,
          raw_specs.len(),
          u_names,
          &raw_specs,
          &t_dom.name,
          x_dom.min,
          true,
          swap,
        )
      {
        bc_lo = Some(bc);
        claimed[idx] = true;
        continue;
      }
      if bc_hi.is_none()
        && let Some(bc) = try_coupled_flux_bc(
          eq,
          raw_specs.len(),
          u_names,
          &raw_specs,
          &t_dom.name,
          x_dom.max,
          false,
          swap,
        )
      {
        bc_hi = Some(bc);
        claimed[idx] = true;
      }
    }
    let (Some((coeff, rhs)), Some(ic), Some(bc_lo), Some(bc_hi)) =
      (coeff_rhs, ic, bc_lo, bc_hi)
    else {
      return Ok(None);
    };
    raw_specs.push(RawSpec {
      coeff,
      rhs,
      ic,
      bc_lo,
      bc_hi,
    });
  }
  if claimed.iter().any(|&c| !c) {
    return Ok(None);
  }

  let rhs_vars: Vec<String> = (0..n)
    .map(|j| format!("NDSolve$U{j}"))
    .chain(["NDSolve$UX".to_string(), "NDSolve$UXX".to_string()])
    .chain([t_dom.name.clone(), x_dom.name.clone()])
    .collect();
  let bc_rhs_vars: Vec<String> = std::iter::once(t_dom.name.clone())
    .chain((0..n).map(|j| format!("NDSolve$U{j}")))
    .collect();

  let mut unknowns: Vec<PdeUnknown> = Vec::with_capacity(n);
  for (i, spec) in raw_specs.into_iter().enumerate() {
    let Some(rhs_placeholder) =
      rewrite_pde_rhs(&spec.rhs, u_names, i, &t_dom.name, &x_dom.name, swap)
    else {
      return Ok(None);
    };
    let coeff_fn =
      NumFn::new(spec.coeff, &[t_dom.name.clone(), x_dom.name.clone()]);
    let rhs_fn = NumFn::new(rhs_placeholder, &rhs_vars);
    let ic_fn = NumFn::new(spec.ic, std::slice::from_ref(&x_dom.name));
    let bc_lo = match spec.bc_lo {
      PdeBc::Dirichlet(rhs) => PdeBoundaryFn::Dirichlet(NumFn::new(
        rhs,
        std::slice::from_ref(&t_dom.name),
      )),
      PdeBc::Neumann(rhs) => {
        let Some(rewritten) =
          rewrite_pde_bc_rhs(&rhs, u_names, &t_dom.name, x_dom.min, swap)
        else {
          return Ok(None);
        };
        PdeBoundaryFn::Neumann(NumFn::new(rewritten, &bc_rhs_vars))
      }
      // The coupled-system solver doesn't attempt a periodic domain; only
      // the hyperbolic solver (`compile_hyperbolic_boundary`) does.
      PdeBc::Periodic => return Ok(None),
    };
    let bc_hi = match spec.bc_hi {
      PdeBc::Dirichlet(rhs) => PdeBoundaryFn::Dirichlet(NumFn::new(
        rhs,
        std::slice::from_ref(&t_dom.name),
      )),
      PdeBc::Neumann(rhs) => {
        let Some(rewritten) =
          rewrite_pde_bc_rhs(&rhs, u_names, &t_dom.name, x_dom.max, swap)
        else {
          return Ok(None);
        };
        PdeBoundaryFn::Neumann(NumFn::new(rewritten, &bc_rhs_vars))
      }
      PdeBc::Periodic => return Ok(None),
    };
    unknowns.push(PdeUnknown {
      coeff_fn,
      rhs_fn,
      ic_fn,
      bc_lo,
      bc_hi,
    });
  }

  const N_X: usize = 41;
  let dx = (x_dom.max - x_dom.min) / (N_X - 1) as f64;
  let xs: Vec<f64> = (0..N_X).map(|i| x_dom.min + i as f64 * dx).collect();

  // Each unknown's coefficient is floored relative to its own scale over
  // the domain (rather than to a fixed absolute constant), so a
  // coefficient that vanishes at a boundary — the no-slip velocity at a
  // channel wall, say — is regularized to a large but bounded rate there
  // instead of spiking arbitrarily close to a division by zero.
  // The coefficient's dominant sign, read off from whichever sampled point
  // has its largest magnitude — used below instead of a floored-near-zero
  // sample's own sign, which is floating-point noise (e.g. evaluating to
  // -1e-15 rather than the mathematically exact 0 at a coefficient's own
  // zero) and would otherwise flip that grid point's evolution equation
  // from stable diffusion to unstable anti-diffusion.
  let (coeff_floors, coeff_signs): (Vec<f64>, Vec<f64>) = unknowns
    .iter()
    .map(|u| {
      let samples: Vec<f64> = xs
        .iter()
        .filter_map(|&x| u.coeff_fn.eval(&[t_dom.min, x]).ok())
        .collect();
      let peak = samples
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
      let sign = samples
        .iter()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .map_or(1.0, |v| if *v < 0.0 { -1.0 } else { 1.0 });
      ((peak * 5e-1).max(1e-9), sign)
    })
    .collect();

  // The grid index range each unknown integrates directly: a Dirichlet
  // boundary is fixed via its `NumFn` each step, so it's excluded; a
  // Neumann boundary has no such formula and is integrated like an
  // interior point instead.
  let free_lo: Vec<usize> = unknowns
    .iter()
    .map(|u| usize::from(!matches!(u.bc_lo, PdeBoundaryFn::Neumann(_))))
    .collect();
  let free_hi: Vec<usize> = unknowns
    .iter()
    .map(|u| {
      if matches!(u.bc_hi, PdeBoundaryFn::Neumann(_)) {
        N_X - 1
      } else {
        N_X - 2
      }
    })
    .collect();
  let free_len: Vec<usize> =
    (0..n).map(|i| free_hi[i] + 1 - free_lo[i]).collect();
  let offset: Vec<usize> = {
    let mut acc = 0;
    (0..n)
      .map(|i| {
        let o = acc;
        acc += free_len[i];
        o
      })
      .collect()
  };
  let total_len: usize = free_len.iter().sum();

  // Divide a raw evolution RHS by its (possibly floored, see
  // `coeff_floors` above) coefficient.
  let scale = |raw: f64, coeff: f64, floor: f64, sign: f64| -> f64 {
    let denom = if coeff.abs() < floor {
      sign * floor
    } else {
      coeff
    };
    raw / denom
  };

  // Reconstruct unknown `i`'s full `N_X`-point profile from the combined
  // integrated state, filling any Dirichlet boundary from its `NumFn`.
  let full_of =
    |i: usize, state: &[f64], t: f64| -> Result<Vec<f64>, InterpreterError> {
      let mut full = vec![0.0; N_X];
      for (k, gi) in (free_lo[i]..=free_hi[i]).enumerate() {
        full[gi] = state[offset[i] + k];
      }
      if let PdeBoundaryFn::Dirichlet(f) = &unknowns[i].bc_lo {
        full[0] = f.eval(&[t])?;
      }
      if let PdeBoundaryFn::Dirichlet(f) = &unknowns[i].bc_hi {
        full[N_X - 1] = f.eval(&[t])?;
      }
      Ok(full)
    };

  let derivative =
    |t: f64, state: &[f64]| -> Result<Vec<f64>, InterpreterError> {
      let fulls: Vec<Vec<f64>> = (0..n)
        .map(|i| full_of(i, state, t))
        .collect::<Result<_, _>>()?;
      let mut out = vec![0.0; total_len];
      for i in 0..n {
        for (k, gi) in (free_lo[i]..=free_hi[i]).enumerate() {
          let (ux, uxx) = if gi == 0 {
            let PdeBoundaryFn::Neumann(f) = &unknowns[i].bc_lo else {
              unreachable!("a Dirichlet lo boundary is never a free index")
            };
            let bc_args: Vec<f64> = std::iter::once(t)
              .chain((0..n).map(|k2| fulls[k2][gi]))
              .collect();
            let flux = f.eval(&bc_args)?;
            let ghost = fulls[i][1] - 2.0 * dx * flux;
            (
              (fulls[i][1] - ghost) / (2.0 * dx),
              (fulls[i][1] - 2.0 * fulls[i][0] + ghost) / (dx * dx),
            )
          } else if gi == N_X - 1 {
            let PdeBoundaryFn::Neumann(f) = &unknowns[i].bc_hi else {
              unreachable!("a Dirichlet hi boundary is never a free index")
            };
            let bc_args: Vec<f64> = std::iter::once(t)
              .chain((0..n).map(|k2| fulls[k2][gi]))
              .collect();
            let flux = f.eval(&bc_args)?;
            let ghost = fulls[i][N_X - 2] + 2.0 * dx * flux;
            (
              (ghost - fulls[i][N_X - 2]) / (2.0 * dx),
              (ghost - 2.0 * fulls[i][N_X - 1] + fulls[i][N_X - 2]) / (dx * dx),
            )
          } else {
            (
              (fulls[i][gi + 1] - fulls[i][gi - 1]) / (2.0 * dx),
              (fulls[i][gi + 1] - 2.0 * fulls[i][gi] + fulls[i][gi - 1])
                / (dx * dx),
            )
          };
          let mut vars: Vec<f64> = (0..n).map(|k2| fulls[k2][gi]).collect();
          vars.push(ux);
          vars.push(uxx);
          vars.push(t);
          vars.push(xs[gi]);
          let raw = unknowns[i].rhs_fn.eval(&vars)?;
          let coeff = unknowns[i].coeff_fn.eval(&[t, xs[gi]])?;
          let scaled = scale(raw, coeff, coeff_floors[i], coeff_signs[i]);
          out[offset[i] + k] = scaled;
        }
      }
      Ok(out)
    };

  let mut full0: Vec<Vec<f64>> = Vec::with_capacity(n);
  for u in &unknowns {
    let mut row = Vec::with_capacity(N_X);
    for &x in &xs {
      row.push(u.ic_fn.eval(&[x])?);
    }
    if let PdeBoundaryFn::Dirichlet(f) = &u.bc_lo {
      row[0] = f.eval(&[t_dom.min])?;
    }
    if let PdeBoundaryFn::Dirichlet(f) = &u.bc_hi {
      row[N_X - 1] = f.eval(&[t_dom.min])?;
    }
    full0.push(row);
  }

  // Effective diffusivity per unknown, read off its RHS's dependence on
  // `NDSolve$UXX` (divided by its coefficient), to size a stable explicit
  // time step (a von Neumann/CFL-type bound). Sampled at every grid point
  // rather than just the midpoint: a coefficient that varies over the
  // domain (and especially one that's floored near a vanishing point, see
  // `coeff_floors` above) can need a far smaller step there than its
  // domain-average would suggest. Falls back to a small floor when an
  // unknown's PDE has (almost) no `NDSolve$UXX` dependence at all.
  let mut dt_stable = f64::INFINITY;
  for (i, u) in unknowns.iter().enumerate() {
    for (j, &x) in xs.iter().enumerate() {
      let mut probe_lo: Vec<f64> = (0..n).map(|k| full0[k][j]).collect();
      probe_lo.push(0.0);
      probe_lo.push(0.0);
      probe_lo.push(t_dom.min);
      probe_lo.push(x);
      let mut probe_hi = probe_lo.clone();
      probe_hi[n + 1] = 1.0;
      let raw_c_eff =
        (u.rhs_fn.eval(&probe_hi)? - u.rhs_fn.eval(&probe_lo)?).max(1e-9);
      let coeff = u.coeff_fn.eval(&[t_dom.min, x])?.abs().max(coeff_floors[i]);
      let c_eff = (raw_c_eff / coeff).max(1e-9);
      dt_stable = dt_stable.min(0.4 * dx * dx / c_eff);
    }
  }
  // The diffusive bound alone can leave the boundary conditions' own time
  // variation (a seasonal `Sin[2 Pi t/period]` forcing, say) badly
  // undersampled even though the scheme stays numerically stable, so the
  // step count is also floored well above what a smooth time-dependent
  // boundary needs to be resolved.
  const MIN_STEPS: usize = 200;
  let n_steps = (((t_dom.max - t_dom.min) / dt_stable).ceil() as usize)
    .clamp(MIN_STEPS, 200_000);
  let dt = (t_dom.max - t_dom.min) / n_steps as f64;

  // The fine step count above is sized for integration accuracy and
  // stability, not for how many rows an `InterpolatingFunction` needs to
  // represent a smooth solution — and every stored row is a separate
  // heap-allocated list that later gets cloned whole on every lookup (a
  // `ContourPlot` alone samples on the order of 10,000 points), so only
  // every `stride`-th step's row is kept, with the true endpoint always
  // included.
  const OUTPUT_ROWS: usize = 60;
  let stride = (n_steps / OUTPUT_ROWS).max(1);
  let mut grids: Vec<Vec<Vec<f64>>> =
    (0..n).map(|i| vec![full0[i].clone()]).collect();
  let mut ts: Vec<f64> = Vec::with_capacity(n_steps / stride + 2);
  ts.push(t_dom.min);
  let mut state: Vec<f64> = Vec::with_capacity(total_len);
  for (i, row) in full0.iter().enumerate() {
    state.extend(row[free_lo[i]..=free_hi[i]].iter().copied());
  }
  let scale = state
    .iter()
    .fold(0.0_f64, |m, v| m.max(v.abs()))
    .max(1e-300);
  let mut last_kept = state.clone();
  let mut t = t_dom.min;
  for step in 0..n_steps {
    let k1 = derivative(t, &state)?;
    let s2: Vec<f64> = state
      .iter()
      .zip(&k1)
      .map(|(y, k)| y + 0.5 * dt * k)
      .collect();
    let k2 = derivative(t + 0.5 * dt, &s2)?;
    let s3: Vec<f64> = state
      .iter()
      .zip(&k2)
      .map(|(y, k)| y + 0.5 * dt * k)
      .collect();
    let k3 = derivative(t + 0.5 * dt, &s3)?;
    let s4: Vec<f64> = state.iter().zip(&k3).map(|(y, k)| y + dt * k).collect();
    let k4 = derivative(t + dt, &s4)?;
    for idx in 0..state.len() {
      state[idx] +=
        dt / 6.0 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
    }
    // From the step index, not by accumulating `dt`, which ends a few
    // ulps short of the domain (59.9999999999998 instead of 60.).
    t = if step + 1 == n_steps {
      t_dom.max
    } else {
      t_dom.min + (step + 1) as f64 * dt
    };
    let is_last = step + 1 == n_steps;
    if (step + 1) % stride == 0
      || is_last
      || pde_row_moved(&state, &last_kept, scale, ts.len())
    {
      for (i, g) in grids.iter_mut().enumerate() {
        g.push(full_of(i, &state, t)?);
      }
      ts.push(t);
      last_kept.clone_from(&state);
    }
  }

  // The output `InterpolatingFunction`s' first dimension must match
  // `dim0_dom` — the domain declared first in the `NDSolve` call —
  // regardless of which domain played the time-like solving role above.
  let t_is_dim0 = t_dom.name == dim0_dom.name;
  let (dim0, dim1) = if t_is_dim0 {
    (t_dom, x_dom)
  } else {
    (x_dom, t_dom)
  };
  let (dim0_coords, dim1_coords) =
    if t_is_dim0 { (&ts, &xs) } else { (&xs, &ts) };
  let domain = Expr::List(
    vec![
      Expr::List(vec![Expr::Real(dim0.min), Expr::Real(dim0.max)].into()),
      Expr::List(vec![Expr::Real(dim1.min), Expr::Real(dim1.max)].into()),
    ]
    .into(),
  );
  let orders = Expr::List(vec![Expr::Integer(1), Expr::Integer(1)].into());
  let coords = Expr::List(
    vec![
      Expr::List(dim0_coords.iter().map(|v| Expr::Real(*v)).collect()),
      Expr::List(dim1_coords.iter().map(|v| Expr::Real(*v)).collect()),
    ]
    .into(),
  );

  let mut rules = Vec::with_capacity(n);
  for (i, name) in u_names.iter().enumerate() {
    let grid = if t_is_dim0 {
      grids[i].clone()
    } else {
      transpose_grid(&grids[i])
    };
    let grid_expr = Expr::List(
      grid
        .iter()
        .map(|row| Expr::List(row.iter().map(|v| Expr::Real(*v)).collect()))
        .collect(),
    );
    let interp = call(
      "InterpolatingFunction",
      vec![domain.clone(), grid_expr, orders.clone(), coords.clone()],
    );
    rules.push(rule_expr(Expr::Identifier(name.clone()), interp));
  }
  Ok(Some(Expr::List(vec![Expr::List(rules.into())].into())))
}

/// Transpose a rectangular grid of rows into columns — used to reorient a
/// `[time-index][space-index]` solve grid into the `[dim0][dim1]` layout an
/// `InterpolatingFunction` expects when the space-like variable is `dim0`.
fn transpose_grid(grid: &[Vec<f64>]) -> Vec<Vec<f64>> {
  if grid.is_empty() {
    return Vec::new();
  }
  let cols = grid[0].len();
  (0..cols)
    .map(|c| grid.iter().map(|row| row[c]).collect())
    .collect()
}

// ─── Hyperbolic PDE NDSolve (order reduction + implicit method of lines) ──

/// `NDSolve[eqns, u, {t domain}, {x domain}]` for a single hyperbolic
/// (second-order-in-time) 1-D PDE `Derivative[2, 0][u][t, x] == rhs`,
/// where `rhs` may reference `u` and its pure space derivatives (up to
/// second order), the velocity field `Derivative[1, 0][u]` and its own
/// pure space derivatives, and even the acceleration field itself —
/// `Derivative[2, 0][u]` — and *its* space derivatives, so long as the
/// dependence on the acceleration field is linear. That last shape is
/// what a wave-type equation with mixed space/time derivatives on the
/// right looks like once it's written out, e.g. the Wolfram Demonstration
/// "A Passive Cochlear Model"'s
/// `D[u[t,x],t,t] == a(x) (D[u,x,x]-…) + b(x) (D[u,x,x,t]-…) + c (D[u,x,x,t,t]-…)`.
///
/// The equation is reduced to the first-order system `D[u,t] == v`,
/// `D[v,t] == w`; substituting these into `rhs` turns every mixed
/// derivative with one time derivative into a pure space derivative of
/// `v`, and every one with two into a pure space derivative of `w`
/// itself — the very quantity being solved for. At each method-of-lines
/// step, `w`'s grid values are eliminated by solving the tridiagonal
/// linear relation the (linear-in-`w`) equation implies, via
/// [`solve_tridiagonal`], rather than by evaluating an explicit formula
/// as the first-order-in-time branch (`try_solve_pde_system`) does.
///
/// Needs exactly five equations: the evolution equation, an initial value
/// `u(t0, x) == f(x)`, an initial velocity `D[u, t](t0, x) == g(x)`, and a
/// Dirichlet or Neumann boundary condition at each end of the space
/// domain. Only a single dependent function is supported (no coupled
/// hyperbolic systems). Returns `Ok(None)` when the equations aren't in
/// that shape, or the right-hand side isn't linear in the acceleration
/// field, so the caller leaves the call unevaluated.
fn ndsolve_pde_hyperbolic(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Expr::Identifier(u_name) = &args[1] else {
    return Ok(None);
  };
  let Some(dom_a) = parse_pde_domain(&args[2]) else {
    return Ok(None);
  };
  let Some(dom_b) = parse_pde_domain(&args[3]) else {
    return Ok(None);
  };

  let eq_items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  let eq_items = flatten_chained_equalities(&eq_items);
  // Normally 5: the evolution equation, both initial conditions, and one
  // boundary condition per end. A periodic boundary (`u[t, xmin] == u[t,
  // xmax]`) ties both ends together in a single equation, so that shape
  // has only 4.
  if eq_items.len() != 5 && eq_items.len() != 4 {
    return Ok(None);
  }

  for (t_dom, x_dom) in [(&dom_a, &dom_b), (&dom_b, &dom_a)] {
    for swap in [false, true] {
      if let Some(result) =
        try_solve_hyperbolic_pde(&eq_items, u_name, t_dom, x_dom, &dom_a, swap)?
      {
        return Ok(Some(result));
      }
    }
  }
  Ok(None)
}

/// Rewrite a hyperbolic PDE's right-hand side into the placeholder
/// identifiers `try_solve_hyperbolic_pde` compiles numeric values into:
/// `u`'s own value and its first/second space derivative become
/// `NDSolve$U`/`NDSolve$UX`/`NDSolve$UXX`; the velocity field
/// `Derivative[1, 0][u]` and its space derivatives become
/// `NDSolve$V`/`NDSolve$VX`/`NDSolve$VXX`; and the acceleration field
/// `Derivative[2, 0][u]` and its space derivatives become
/// `NDSolve$W`/`NDSolve$WX`/`NDSolve$WXX`. Returns `None` for any other
/// derivative shape (a third time derivative, a derivative at a point
/// other than `(t, x)`, …) this solver doesn't attempt.
fn rewrite_pde_rhs_hyperbolic(
  expr: &Expr,
  u_name: &str,
  t_name: &str,
  x_name: &str,
  swap: bool,
) -> Option<Expr> {
  if let Some((dt, dx, t_arg, x_arg)) = match_pde_term_roles(expr, u_name, swap)
  {
    if !matches!(&t_arg, Expr::Identifier(n) if n == t_name)
      || !matches!(&x_arg, Expr::Identifier(n) if n == x_name)
    {
      return None;
    }
    let placeholder = match (dt, dx) {
      (0, 0) => "NDSolve$U",
      (0, 1) => "NDSolve$UX",
      (0, 2) => "NDSolve$UXX",
      (1, 0) => "NDSolve$V",
      (1, 1) => "NDSolve$VX",
      (1, 2) => "NDSolve$VXX",
      (2, 0) => "NDSolve$W",
      (2, 1) => "NDSolve$WX",
      (2, 2) => "NDSolve$WXX",
      _ => return None,
    };
    return Some(id_expr(placeholder));
  }
  let failed = std::cell::Cell::new(false);
  let mapped = map_children(expr, &|c| {
    if failed.get() {
      return c.clone();
    }
    if let Some(r) = rewrite_pde_rhs_hyperbolic(c, u_name, t_name, x_name, swap)
    {
      r
    } else {
      failed.set(true);
      c.clone()
    }
  });
  if failed.get() { None } else { Some(mapped) }
}

/// Whether `expr` contains the bare identifier `name` anywhere in its
/// tree — used to confirm a coefficient extracted by differentiating a
/// linear expression really did lose all dependence on the variable
/// differentiated against (see `try_solve_hyperbolic_pde`'s linearity
/// check).
/// The function a derivative unknown differentiates: `x` for `x'[t]` or
/// `x''`, which evaluate to `Derivative[n, x, t]` and `Derivative[n, x]`.
fn derivative_unknown_base(u: &Expr) -> Option<Expr> {
  match u {
    Expr::FunctionCall { name, args }
      if name == "Derivative"
        && args.len() >= 2
        && matches!(&args[0], Expr::Integer(_)) =>
    {
      match &args[1] {
        f @ (Expr::Identifier(_) | Expr::FunctionCall { .. }) => {
          Some(f.clone())
        }
        _ => None,
      }
    }
    // The parsed form `Derivative[k][f]` / `Derivative[k][f][x]`, whose
    // `f` may be a compound head such as `θ[1]`.
    Expr::CurriedCall { func, args } if args.len() == 1 => {
      match func.as_ref() {
        Expr::FunctionCall { name, args: orders }
          if name == "Derivative"
            && orders.len() == 1
            && matches!(&orders[0], Expr::Integer(_)) =>
        {
          match &args[0] {
            f @ (Expr::Identifier(_) | Expr::FunctionCall { .. }) => {
              Some(f.clone())
            }
            _ => None,
          }
        }
        inner @ Expr::CurriedCall { .. } => derivative_unknown_base(inner),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Whether `name` occurs in `expr` as a symbol or as the head of a call
/// (`w`, `w[t]`, `w'[t]`).
fn mentions_symbol(expr: &Expr, name: &str) -> bool {
  match expr {
    Expr::Identifier(n) => n == name,
    Expr::FunctionCall { name: head, .. } if head == name => true,
    _ => expr_children(expr).iter().any(|c| mentions_symbol(c, name)),
  }
}

fn expr_contains_ident(expr: &Expr, name: &str) -> bool {
  if matches!(expr, Expr::Identifier(n) if n == name) {
    return true;
  }
  expr_children(expr)
    .iter()
    .any(|c| expr_contains_ident(c, name))
}

/// Solve the tridiagonal system with sub-diagonal `a` (`a[0]` unused),
/// diagonal `b`, super-diagonal `c` (`c[last]` unused), and right-hand
/// side `d` — all the same length — via the Thomas algorithm.
fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
  let n = b.len();
  if n == 0 {
    return Vec::new();
  }
  let mut cp = vec![0.0; n];
  let mut dp = vec![0.0; n];
  cp[0] = c[0] / b[0];
  dp[0] = d[0] / b[0];
  for i in 1..n {
    let m = b[i] - a[i] * cp[i - 1];
    cp[i] = if i + 1 < n { c[i] / m } else { 0.0 };
    dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
  }
  let mut x = vec![0.0; n];
  x[n - 1] = dp[n - 1];
  for i in (0..n - 1).rev() {
    x[i] = dp[i] - cp[i] * x[i + 1];
  }
  x
}

/// Solve a periodic ("cyclic") tridiagonal system: like `solve_tridiagonal`,
/// but row `n - 1`'s sub-diagonal wraps around to column `0` with
/// coefficient `alpha`, and row `0`'s super-diagonal wraps around to
/// column `n - 1` with coefficient `beta`. Uses the Sherman-Morrison trick
/// (Numerical Recipes' "cyclic" routine): absorb the two corner entries
/// into a rank-1 update of an ordinary tridiagonal system, solve that twice
/// (once for the real right-hand side, once for the update vector), then
/// combine the two solutions to cancel the update back out.
fn solve_cyclic_tridiagonal(
  a: &[f64],
  b: &[f64],
  c: &[f64],
  d: &[f64],
  alpha: f64,
  beta: f64,
) -> Vec<f64> {
  let n = b.len();
  if n == 1 {
    return vec![d[0] / (b[0] + alpha + beta)];
  }
  let gamma = -b[0];
  let mut b2 = b.to_vec();
  b2[0] -= gamma;
  b2[n - 1] -= alpha * beta / gamma;
  let x = solve_tridiagonal(a, &b2, c, d);
  let mut u = vec![0.0; n];
  u[0] = gamma;
  u[n - 1] = alpha;
  let z = solve_tridiagonal(a, &b2, c, &u);
  let fact =
    (x[0] + beta * x[n - 1] / gamma) / (1.0 + z[0] + beta * z[n - 1] / gamma);
  x.iter().zip(&z).map(|(xi, zi)| xi - fact * zi).collect()
}

/// One field's (`u`, `v`, or `w`) compiled boundary condition at one end
/// of the space domain, mirroring `u`'s own condition kind: `v = D[u,t]`'s
/// right-hand side is `u`'s own boundary value differentiated once with
/// respect to time, and `w = D[u,t,t]`'s is differentiated twice — a
/// Dirichlet boundary stays Dirichlet, a Neumann flux stays Neumann.
struct HyperbolicBoundary {
  u: PdeBoundaryFn,
  v: PdeBoundaryFn,
  w: PdeBoundaryFn,
}

fn compile_hyperbolic_boundary(
  bc: PdeBc,
  u_name: &str,
  t_name: &str,
) -> Result<Option<HyperbolicBoundary>, InterpreterError> {
  let t_vars = vec![t_name.to_string()];
  match bc {
    PdeBc::Dirichlet(rhs) => {
      let d1 = differentiate_expr(&rhs, t_name)?;
      let d2 = differentiate_expr(&d1, t_name)?;
      Ok(Some(HyperbolicBoundary {
        u: PdeBoundaryFn::Dirichlet(NumFn::new(rhs, &t_vars)),
        v: PdeBoundaryFn::Dirichlet(NumFn::new(d1, &t_vars)),
        w: PdeBoundaryFn::Dirichlet(NumFn::new(d2, &t_vars)),
      }))
    }
    PdeBc::Neumann(rhs) => {
      // Single dependent function: a flux depending on the unknown itself
      // (a Robin-type condition) is outside this solver's scope.
      let u_names_single = [u_name.to_string()];
      if expr_references_any(&rhs, &u_names_single) {
        return Ok(None);
      }
      let d1 = differentiate_expr(&rhs, t_name)?;
      let d2 = differentiate_expr(&d1, t_name)?;
      Ok(Some(HyperbolicBoundary {
        u: PdeBoundaryFn::Neumann(NumFn::new(rhs, &t_vars)),
        v: PdeBoundaryFn::Neumann(NumFn::new(d1, &t_vars)),
        w: PdeBoundaryFn::Neumann(NumFn::new(d2, &t_vars)),
      }))
    }
    PdeBc::Periodic => Ok(Some(HyperbolicBoundary {
      u: PdeBoundaryFn::Periodic,
      v: PdeBoundaryFn::Periodic,
      w: PdeBoundaryFn::Periodic,
    })),
  }
}

/// Reconstruct a field's full `N_X`-point profile from its combined
/// integrated free values, filling a Dirichlet boundary from its own
/// `NumFn`. Mirrors `try_solve_pde_system`'s `full_of` closure, adapted
/// to a single field/domain pair.
fn reconstruct_pde_profile(
  free_vals: &[f64],
  free_lo: usize,
  free_hi: usize,
  n_x: usize,
  bc_lo: &PdeBoundaryFn,
  bc_hi: &PdeBoundaryFn,
  t: f64,
) -> Result<Vec<f64>, InterpreterError> {
  let mut full = vec![0.0; n_x];
  for (k, gi) in (free_lo..=free_hi).enumerate() {
    full[gi] = free_vals[k];
  }
  if let PdeBoundaryFn::Dirichlet(f) = bc_lo {
    full[0] = f.eval(&[t])?;
  }
  if let PdeBoundaryFn::Dirichlet(f) = bc_hi {
    full[n_x - 1] = f.eval(&[t])?;
  }
  // A periodic domain's last grid point is never itself an integrated/free
  // index (`free_hi` stops one short of it, at `n_x - 2`, to avoid solving
  // for the same point twice) — it just mirrors the first.
  if matches!(bc_lo, PdeBoundaryFn::Periodic) {
    full[n_x - 1] = full[0];
  }
  Ok(full)
}

/// First and second space-derivative of a field at grid index `gi`, via
/// central differences at an interior point or, at a boundary carrying a
/// Neumann condition, a ghost point built from the boundary's own flux
/// value — mirrors `try_solve_pde_system`'s `derivative` closure's own
/// boundary handling. `bc_lo`/`bc_hi` are never `Dirichlet` when `gi` is
/// `0`/`n_x - 1` respectively (those indices are excluded from the
/// free/solved range in that case).
fn pde_space_derivs(
  full: &[f64],
  gi: usize,
  dx: f64,
  bc_lo: &PdeBoundaryFn,
  bc_hi: &PdeBoundaryFn,
  t: f64,
) -> Result<(f64, f64), InterpreterError> {
  let n_x = full.len();
  if gi == 0 && matches!(bc_lo, PdeBoundaryFn::Periodic) {
    // The periodic "ghost" below index 0 is the domain's other end, which
    // `reconstruct_pde_profile` has already mirrored into `full[n_x - 1]`
    // — but that's the *duplicate* of index 0 itself, so the true wrapped
    // neighbor is one step further in, at `n_x - 2` (the last genuinely
    // free/solved index).
    let lo = full[n_x - 2];
    Ok((
      (full[1] - lo) / (2.0 * dx),
      (full[1] - 2.0 * full[0] + lo) / (dx * dx),
    ))
  } else if gi == 0 {
    let PdeBoundaryFn::Neumann(f) = bc_lo else {
      unreachable!("a Dirichlet lo boundary is never a free index")
    };
    let flux = f.eval(&[t])?;
    let ghost = full[1] - 2.0 * dx * flux;
    Ok((flux, (full[1] - 2.0 * full[0] + ghost) / (dx * dx)))
  } else if gi == n_x - 1 {
    let PdeBoundaryFn::Neumann(f) = bc_hi else {
      unreachable!("a Dirichlet hi boundary is never a free index")
    };
    let flux = f.eval(&[t])?;
    let ghost = full[n_x - 2] + 2.0 * dx * flux;
    Ok((
      flux,
      (ghost - 2.0 * full[n_x - 1] + full[n_x - 2]) / (dx * dx),
    ))
  } else {
    Ok((
      (full[gi + 1] - full[gi - 1]) / (2.0 * dx),
      (full[gi + 1] - 2.0 * full[gi] + full[gi - 1]) / (dx * dx),
    ))
  }
}

/// Attempt to match and solve the hyperbolic PDE with `t_dom` as the
/// time-like (integrated) domain and `x_dom` as the space-like (gridded)
/// one — see [`ndsolve_pde_hyperbolic`]. Returns `Ok(None)` when the
/// equations don't fit this role assignment.
fn try_solve_hyperbolic_pde(
  eq_items: &[Expr],
  u_name: &str,
  t_dom: &PdeDomain,
  x_dom: &PdeDomain,
  dim0_dom: &PdeDomain,
  swap: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let mut evolution_rhs: Option<Expr> = None;
  let mut ic_pos: Option<Expr> = None;
  let mut ic_vel: Option<Expr> = None;
  let mut bc_lo: Option<PdeBc> = None;
  let mut bc_hi: Option<PdeBc> = None;

  for eq in eq_items {
    if evolution_rhs.is_none()
      && let Some(rhs) = try_pde_evolution_term_order2(
        eq,
        u_name,
        &t_dom.name,
        &x_dom.name,
        swap,
      )
    {
      evolution_rhs = Some(rhs);
      continue;
    }
    if ic_pos.is_none()
      && let Some(rhs) =
        try_pde_initial_condition(eq, u_name, &x_dom.name, t_dom.min, swap)
    {
      ic_pos = Some(rhs);
      continue;
    }
    if ic_vel.is_none()
      && let Some(rhs) = try_pde_initial_condition_ord(
        eq,
        u_name,
        &x_dom.name,
        t_dom.min,
        1,
        swap,
      )
    {
      ic_vel = Some(rhs);
      continue;
    }
    if bc_lo.is_none()
      && bc_hi.is_none()
      && try_pde_periodic_boundary(
        eq,
        u_name,
        &t_dom.name,
        x_dom.min,
        x_dom.max,
        swap,
      )
    {
      bc_lo = Some(PdeBc::Periodic);
      bc_hi = Some(PdeBc::Periodic);
      continue;
    }
    if bc_lo.is_none()
      && let Some(bc) =
        try_pde_boundary_condition(eq, u_name, &t_dom.name, x_dom.min, swap)
    {
      bc_lo = Some(bc);
      continue;
    }
    if bc_hi.is_none()
      && let Some(bc) =
        try_pde_boundary_condition(eq, u_name, &t_dom.name, x_dom.max, swap)
    {
      bc_hi = Some(bc);
    }
  }
  let (Some(rhs), Some(ic_pos), Some(ic_vel), Some(bc_lo), Some(bc_hi)) =
    (evolution_rhs, ic_pos, ic_vel, bc_lo, bc_hi)
  else {
    return Ok(None);
  };

  let Some(rewritten) =
    rewrite_pde_rhs_hyperbolic(&rhs, u_name, &t_dom.name, &x_dom.name, swap)
  else {
    return Ok(None);
  };

  // Extract the acceleration field's linear coefficients (`c0` multiplies
  // `w` itself, `c1` its first space derivative, `c2` its second) via
  // symbolic differentiation, and verify the right-hand side really is
  // linear in them: a coefficient that still references one of the three
  // placeholders after differentiating it away means the dependence
  // wasn't linear, a shape this solver doesn't attempt.
  const W_VARS: [&str; 3] = ["NDSolve$W", "NDSolve$WX", "NDSolve$WXX"];
  let mut coeffs = Vec::with_capacity(3);
  for wv in W_VARS {
    let c = differentiate_expr(&rewritten, wv)?;
    if W_VARS.iter().any(|wv2| expr_contains_ident(&c, wv2)) {
      return Ok(None);
    }
    coeffs.push(c);
  }
  let explicit_expr = crate::syntax::substitute_variables(
    &rewritten,
    &[
      ("NDSolve$W", &Expr::Integer(0)),
      ("NDSolve$WX", &Expr::Integer(0)),
      ("NDSolve$WXX", &Expr::Integer(0)),
    ],
  );

  let full_vars: Vec<String> = [
    "NDSolve$U",
    "NDSolve$UX",
    "NDSolve$UXX",
    "NDSolve$V",
    "NDSolve$VX",
    "NDSolve$VXX",
  ]
  .into_iter()
  .map(String::from)
  .chain([t_dom.name.clone(), x_dom.name.clone()])
  .collect();

  let explicit_fn = NumFn::new(explicit_expr, &full_vars);
  let coeff0_fn = NumFn::new(coeffs[0].clone(), &full_vars);
  let coeff1_fn = NumFn::new(coeffs[1].clone(), &full_vars);
  let coeff2_fn = NumFn::new(coeffs[2].clone(), &full_vars);

  let x_vars = vec![x_dom.name.clone()];
  let ic_pos_fn = NumFn::new(ic_pos, &x_vars);
  let ic_vel_fn = NumFn::new(ic_vel, &x_vars);

  let Some(bnd_lo) = compile_hyperbolic_boundary(bc_lo, u_name, &t_dom.name)?
  else {
    return Ok(None);
  };
  let Some(bnd_hi) = compile_hyperbolic_boundary(bc_hi, u_name, &t_dom.name)?
  else {
    return Ok(None);
  };

  const N_X: usize = 41;
  let dx = (x_dom.max - x_dom.min) / (N_X - 1) as f64;
  let xs: Vec<f64> = (0..N_X).map(|i| x_dom.min + i as f64 * dx).collect();

  // A periodic domain's last grid point duplicates the first (see
  // `reconstruct_pde_profile`), so it's excluded from the free/solved
  // range just like a Dirichlet boundary, even though it isn't fixed from
  // an external `NumFn` the way a Dirichlet point is.
  let free_lo = usize::from(matches!(bnd_lo.u, PdeBoundaryFn::Dirichlet(_)));
  let free_hi = if matches!(bnd_hi.u, PdeBoundaryFn::Neumann(_)) {
    N_X - 1
  } else {
    N_X - 2
  };
  let periodic = matches!(bnd_lo.u, PdeBoundaryFn::Periodic);
  let free_len = free_hi + 1 - free_lo;
  let total_len = 2 * free_len;

  let derivative = |t: f64,
                    state: &[f64]|
   -> Result<Vec<f64>, InterpreterError> {
    let u_full = reconstruct_pde_profile(
      &state[..free_len],
      free_lo,
      free_hi,
      N_X,
      &bnd_lo.u,
      &bnd_hi.u,
      t,
    )?;
    let v_full = reconstruct_pde_profile(
      &state[free_len..],
      free_lo,
      free_hi,
      N_X,
      &bnd_lo.v,
      &bnd_hi.v,
      t,
    )?;

    // Only the free/solved indices ever feed into the tridiagonal system
    // below — a Dirichlet boundary's own index is fixed directly from its
    // `NumFn` each step and never appears as a row or a neighbor there —
    // so `pde_space_derivs` (which assumes index `0`/`N_X - 1` means a
    // *Neumann* boundary) is never asked to differentiate at one.
    let mut explicit = vec![0.0; N_X];
    let mut c0 = vec![0.0; N_X];
    let mut c1 = vec![0.0; N_X];
    let mut c2 = vec![0.0; N_X];
    for gi in free_lo..=free_hi {
      let (ux, uxx) =
        pde_space_derivs(&u_full, gi, dx, &bnd_lo.u, &bnd_hi.u, t)?;
      let (vx, vxx) =
        pde_space_derivs(&v_full, gi, dx, &bnd_lo.v, &bnd_hi.v, t)?;
      let vars = [u_full[gi], ux, uxx, v_full[gi], vx, vxx, t, xs[gi]];
      explicit[gi] = explicit_fn.eval(&vars)?;
      c0[gi] = coeff0_fn.eval(&vars)?;
      c1[gi] = coeff1_fn.eval(&vars)?;
      c2[gi] = coeff2_fn.eval(&vars)?;
    }

    // Eliminate `w`'s free grid values from the linear relation
    // `w == c0 w + c1 D[w,x] + c2 D[w,x,x] + explicit`, i.e.
    // `(1 - c0) w - c1 D[w,x] - c2 D[w,x,x] == explicit`, via a
    // tridiagonal solve.
    let mut a = vec![0.0; free_len];
    let mut b = vec![0.0; free_len];
    let mut c = vec![0.0; free_len];
    let mut d = vec![0.0; free_len];
    // For a periodic domain, row 0's "lower" coupling and the last free
    // row's "upper" coupling don't land on `a`/`c` at all — they wrap
    // around to the *other* end of the array, which `solve_cyclic_tridiagonal`
    // takes as separate corner coefficients instead.
    let mut corner_lo = 0.0;
    let mut corner_hi = 0.0;
    for (k, gi) in (free_lo..=free_hi).enumerate() {
      if gi == 0 && !periodic {
        let PdeBoundaryFn::Neumann(f) = &bnd_lo.w else {
          unreachable!("a Dirichlet lo boundary is never a free index")
        };
        let flux = f.eval(&[t])?;
        b[k] = (1.0 - c0[gi]) + 2.0 * c2[gi] / (dx * dx);
        c[k] = -2.0 * c2[gi] / (dx * dx);
        d[k] = explicit[gi] + c1[gi] * flux - 2.0 * c2[gi] * flux / dx;
      } else if gi == N_X - 1 && !periodic {
        let PdeBoundaryFn::Neumann(f) = &bnd_hi.w else {
          unreachable!("a Dirichlet hi boundary is never a free index")
        };
        let flux = f.eval(&[t])?;
        a[k] = -2.0 * c2[gi] / (dx * dx);
        b[k] = (1.0 - c0[gi]) + 2.0 * c2[gi] / (dx * dx);
        d[k] = explicit[gi] + c1[gi] * flux + 2.0 * c2[gi] * flux / dx;
      } else {
        let ai = c1[gi] / (2.0 * dx) - c2[gi] / (dx * dx);
        let bi = (1.0 - c0[gi]) + 2.0 * c2[gi] / (dx * dx);
        let ci = -c1[gi] / (2.0 * dx) - c2[gi] / (dx * dx);
        let mut di = explicit[gi];
        if gi == free_lo && free_lo > 0 {
          let PdeBoundaryFn::Dirichlet(f) = &bnd_lo.w else {
            unreachable!("gi == free_lo > 0 means the lo boundary is Dirichlet")
          };
          di -= ai * f.eval(&[t])?;
        } else if gi == 0 && periodic {
          corner_lo = ai;
        } else {
          a[k] = ai;
        }
        if gi == free_hi && free_hi < N_X - 1 && !periodic {
          let PdeBoundaryFn::Dirichlet(f) = &bnd_hi.w else {
            unreachable!(
              "gi == free_hi < N_X - 1 means the hi boundary is Dirichlet"
            )
          };
          di -= ci * f.eval(&[t])?;
        } else if gi == free_hi && periodic {
          corner_hi = ci;
        } else {
          c[k] = ci;
        }
        b[k] = bi;
        d[k] = di;
      }
    }
    let w_free = if periodic {
      solve_cyclic_tridiagonal(&a, &b, &c, &d, corner_hi, corner_lo)
    } else {
      solve_tridiagonal(&a, &b, &c, &d)
    };

    let mut out = vec![0.0; total_len];
    for (k, gi) in (free_lo..=free_hi).enumerate() {
      out[k] = v_full[gi];
      out[free_len + k] = w_free[k];
    }
    Ok(out)
  };

  let mut u0_full = vec![0.0; N_X];
  let mut v0_full = vec![0.0; N_X];
  for (gi, &x) in xs.iter().enumerate() {
    u0_full[gi] = ic_pos_fn.eval(&[x])?;
    v0_full[gi] = ic_vel_fn.eval(&[x])?;
  }
  if let PdeBoundaryFn::Dirichlet(f) = &bnd_lo.u {
    u0_full[0] = f.eval(&[t_dom.min])?;
  }
  if let PdeBoundaryFn::Dirichlet(f) = &bnd_hi.u {
    u0_full[N_X - 1] = f.eval(&[t_dom.min])?;
  }
  if let PdeBoundaryFn::Dirichlet(f) = &bnd_lo.v {
    v0_full[0] = f.eval(&[t_dom.min])?;
  }
  if let PdeBoundaryFn::Dirichlet(f) = &bnd_hi.v {
    v0_full[N_X - 1] = f.eval(&[t_dom.min])?;
  }

  let mut state0: Vec<f64> = Vec::with_capacity(total_len);
  state0.extend(u0_full[free_lo..=free_hi].iter().copied());
  state0.extend(v0_full[free_lo..=free_hi].iter().copied());

  // Effective diffusivity-like bound, read off the explicit right-hand
  // side's sensitivity to `u`'s and `v`'s own second space derivative —
  // mirrors `try_solve_pde_system`'s own CFL-type probe. The acceleration
  // field's *implicit* dependence (`c0`/`c1`/`c2`) is excluded: an
  // implicit elliptic solve relaxes the stability bound rather than
  // tightening it, so the explicit terms' own diffusivity remains the
  // binding constraint.
  let mut dt_stable = f64::INFINITY;
  for &x in &xs {
    let base =
      explicit_fn.eval(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, t_dom.min, x])?;
    let du =
      (explicit_fn.eval(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, t_dom.min, x])? - base)
        .abs();
    let dv =
      (explicit_fn.eval(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, t_dom.min, x])? - base)
        .abs();
    let c_eff = du.max(dv).max(1e-9);
    dt_stable = dt_stable.min(0.4 * dx * dx / c_eff);
  }
  const MIN_STEPS: usize = 200;
  let n_steps = (((t_dom.max - t_dom.min) / dt_stable).ceil() as usize)
    .clamp(MIN_STEPS, 200_000);
  let dt = (t_dom.max - t_dom.min) / n_steps as f64;

  const OUTPUT_ROWS: usize = 60;
  let stride = (n_steps / OUTPUT_ROWS).max(1);
  let mut ts: Vec<f64> = Vec::with_capacity(n_steps / stride + 2);
  ts.push(t_dom.min);
  let mut grid: Vec<Vec<f64>> = vec![reconstruct_pde_profile(
    &state0[..free_len],
    free_lo,
    free_hi,
    N_X,
    &bnd_lo.u,
    &bnd_hi.u,
    t_dom.min,
  )?];
  let mut state = state0;
  let scale = state
    .iter()
    .fold(0.0_f64, |m, v| m.max(v.abs()))
    .max(1e-300);
  let mut last_kept = state.clone();
  let mut t = t_dom.min;
  for step in 0..n_steps {
    let k1 = derivative(t, &state)?;
    let s2: Vec<f64> = state
      .iter()
      .zip(&k1)
      .map(|(y, k)| y + 0.5 * dt * k)
      .collect();
    let k2 = derivative(t + 0.5 * dt, &s2)?;
    let s3: Vec<f64> = state
      .iter()
      .zip(&k2)
      .map(|(y, k)| y + 0.5 * dt * k)
      .collect();
    let k3 = derivative(t + 0.5 * dt, &s3)?;
    let s4: Vec<f64> = state.iter().zip(&k3).map(|(y, k)| y + dt * k).collect();
    let k4 = derivative(t + dt, &s4)?;
    for idx in 0..state.len() {
      state[idx] +=
        dt / 6.0 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
    }
    // From the step index, not by accumulating `dt`, which ends a few
    // ulps short of the domain (59.9999999999998 instead of 60.).
    t = if step + 1 == n_steps {
      t_dom.max
    } else {
      t_dom.min + (step + 1) as f64 * dt
    };
    let is_last = step + 1 == n_steps;
    if (step + 1) % stride == 0
      || is_last
      || pde_row_moved(&state, &last_kept, scale, ts.len())
    {
      last_kept.clone_from(&state);
      grid.push(reconstruct_pde_profile(
        &state[..free_len],
        free_lo,
        free_hi,
        N_X,
        &bnd_lo.u,
        &bnd_hi.u,
        t,
      )?);
      ts.push(t);
    }
  }

  let t_is_dim0 = t_dom.name == dim0_dom.name;
  let (dim0, dim1) = if t_is_dim0 {
    (t_dom, x_dom)
  } else {
    (x_dom, t_dom)
  };
  let (dim0_coords, dim1_coords) =
    if t_is_dim0 { (&ts, &xs) } else { (&xs, &ts) };
  let domain = Expr::List(
    vec![
      Expr::List(vec![Expr::Real(dim0.min), Expr::Real(dim0.max)].into()),
      Expr::List(vec![Expr::Real(dim1.min), Expr::Real(dim1.max)].into()),
    ]
    .into(),
  );
  let orders = Expr::List(vec![Expr::Integer(1), Expr::Integer(1)].into());
  let coords = Expr::List(
    vec![
      Expr::List(dim0_coords.iter().map(|v| Expr::Real(*v)).collect()),
      Expr::List(dim1_coords.iter().map(|v| Expr::Real(*v)).collect()),
    ]
    .into(),
  );
  let grid = if t_is_dim0 {
    grid
  } else {
    transpose_grid(&grid)
  };
  let grid_expr = Expr::List(
    grid
      .iter()
      .map(|row| Expr::List(row.iter().map(|v| Expr::Real(*v)).collect()))
      .collect(),
  );
  let interp = call(
    "InterpolatingFunction",
    vec![domain, grid_expr, orders, coords],
  );
  let rule = rule_expr(id_expr(u_name), interp);
  Ok(Some(Expr::List(vec![Expr::List(vec![rule].into())].into())))
}

/// Dependent functions of an `NDSolve` system need not be bare symbols:
/// a spatially discretized transport equation is normally written
/// `NDSolve[…, Table[Subscript[c, i], {i, 1, n}], {t, 0, tmax}]`, where every
/// unknown is a `Subscript[…]` expression. The integrator keys each unknown
/// by a symbol name, so every distinct compound head is rewritten to a fresh
/// symbol before integrating and restored in the solution rules afterwards.
///
/// Returns the `(head expression, fresh symbol)` pairs, empty when every
/// unknown already is a symbol.
fn compound_head_renamings(vars: &Expr, domain: &Expr) -> Vec<(Expr, String)> {
  let x_name = match domain {
    Expr::List(items) => match items.first() {
      Some(Expr::Identifier(x)) => x.clone(),
      _ => return Vec::new(),
    },
    _ => return Vec::new(),
  };
  let entries: Vec<&Expr> = match vars {
    Expr::List(items) => items.iter().collect(),
    single => vec![single],
  };
  let mut renames: Vec<(Expr, String)> = Vec::new();
  for entry in entries {
    let head = match entry {
      // `Subscript[c, 1][t]` — applied form of a compound head.
      Expr::CurriedCall { func, args }
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(a) if *a == x_name) =>
      {
        func.as_ref().clone()
      }
      // `y[t]` — applied form of a plain symbol; nothing to rename.
      Expr::FunctionCall { args, .. }
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(a) if *a == x_name) =>
      {
        continue;
      }
      // `Derivative[k][f]` (bare) or `Derivative[k][f][x]` (applied) — a
      // request for `f`'s derivative, not a compound head itself unless
      // `f` is one (e.g. `Derivative[1][Subscript[c, 1]]`).
      Expr::FunctionCall { name, args }
        if name == "Derivative" && (args.len() == 2 || args.len() == 3) =>
      {
        match &args[1] {
          Expr::Identifier(_) => continue,
          inner => inner.clone(),
        }
      }
      // `Subscript[c, 1]` — bare compound head.
      Expr::FunctionCall { .. } => entry.clone(),
      // `y` — bare symbol.
      _ => continue,
    };
    if matches!(head, Expr::Identifier(_)) {
      continue;
    }
    let key = expr_to_string(&head);
    if renames.iter().any(|(h, _)| expr_to_string(h) == key) {
      continue;
    }
    let fresh = format!("NDSolve$fn${}", renames.len() + 1);
    renames.push((head, fresh));
  }
  renames
}

/// Rewrite every occurrence of a compound head to its fresh symbol, turning
/// `Subscript[c, 1][t]` into `NDSolve$fn$1[t]` and
/// `Derivative[1][Subscript[c, 1]][t]` into `Derivative[1][NDSolve$fn$1][t]`.
pub(crate) fn rename_compound_heads(
  expr: &Expr,
  renames: &[(Expr, String)],
) -> Expr {
  let keyed: Vec<(&Expr, String, &String)> = renames
    .iter()
    .map(|(head, fresh)| (head, expr_to_string(head), fresh))
    .collect();
  rename_keyed_heads(expr, &keyed, renames)
}

fn rename_keyed_heads(
  expr: &Expr,
  keyed: &[(&Expr, String, &String)],
  renames: &[(Expr, String)],
) -> Expr {
  for (head, key, fresh) in keyed {
    if crate::syntax::renders_as(expr, head, key) {
      return Expr::Identifier((*fresh).clone());
    }
  }
  let mapped = map_children(expr, &|c| rename_keyed_heads(c, keyed, renames));
  // `Subscript[c, 1][t]` is a curried call; once its head is a plain symbol
  // it must print and match as the ordinary call `NDSolve$fn$1[t]`.
  if let Expr::CurriedCall { func, args } = &mapped
    && let Expr::Identifier(name) = func.as_ref()
    && renames.iter().any(|(_, fresh)| fresh == name)
  {
    return Expr::FunctionCall {
      name: name.clone(),
      args: args.clone().into(),
    };
  }
  mapped
}

/// The inverse of [`rename_compound_heads`], applied to the solution rules:
/// `NDSolve$fn$1 -> InterpolatingFunction[…]` becomes
/// `Subscript[c, 1] -> InterpolatingFunction[…]`.
pub(crate) fn restore_compound_heads(
  expr: &Expr,
  renames: &[(Expr, String)],
) -> Expr {
  if renames.is_empty() {
    return expr.clone();
  }
  if let Expr::Identifier(name) = expr
    && let Some((head, _)) = renames.iter().find(|(_, fresh)| fresh == name)
  {
    return head.clone();
  }
  if let Expr::FunctionCall { name, args } = expr
    && let Some((head, _)) = renames.iter().find(|(_, fresh)| fresh == name)
  {
    return Expr::CurriedCall {
      func: Box::new(head.clone()),
      args: args
        .iter()
        .map(|a| restore_compound_heads(a, renames))
        .collect(),
    };
  }
  map_children(expr, &|c| restore_compound_heads(c, renames))
}

/// One member of a stage-indexed dependent-variable family: `head[index,
/// x_name]` (e.g. `x[3, t]`, the 3rd entry of `Table[x[i, t], {i, 1, n}]`)
/// is keyed by `fresh`, a synthesized single-argument symbol name.
struct IndexedRename {
  head: String,
  index: Expr,
  fresh: String,
}

/// Find every `head[index, x]` entry in the dependent-variable spec — the
/// idiom a discretized PDE or a staged process model (a distillation
/// column's per-tray composition, a chain of coupled oscillators, …) uses
/// to name its unknowns: `Table[x[i, t], {i, 1, n}]` rather than `n`
/// separately named functions. `ndsolve_system` below only understands
/// `f[t]`-shaped unknowns, so each concrete `head[index, x]` is assigned a
/// fresh 1-argument symbol; [`rename_indexed_vars`]/[`restore_indexed_vars`]
/// translate the system into and back out of that vocabulary.
fn indexed_family_renamings(vars: &Expr, domain: &Expr) -> Vec<IndexedRename> {
  let x_name = match domain {
    Expr::List(items) => match items.first() {
      Some(Expr::Identifier(x)) => x.clone(),
      _ => return Vec::new(),
    },
    _ => return Vec::new(),
  };
  let entries: Vec<&Expr> = match vars {
    Expr::List(items) => items.iter().collect(),
    single => vec![single],
  };
  let mut renames: Vec<IndexedRename> = Vec::new();
  for entry in entries {
    let Expr::FunctionCall { name, args } = entry else {
      continue;
    };
    if args.len() != 2 {
      continue;
    }
    let Some(Expr::Identifier(a)) = args.get(1) else {
      continue;
    };
    if *a != x_name {
      continue;
    }
    let index = args[0].clone();
    let index_key = expr_to_string(&index);
    if renames
      .iter()
      .any(|r| r.head == *name && expr_to_string(&r.index) == index_key)
    {
      continue;
    }
    let fresh = format!("NDSolve$idx${name}${}", renames.len() + 1);
    renames.push(IndexedRename {
      head: name.clone(),
      index,
      fresh,
    });
  }
  renames
}

/// Rewrite `head[index, other]` — bare, or under `Derivative[0, k][head]`
/// (the parsed form of `D[head[index, x], {x, k}]`) — to the matching
/// family member's fresh symbol applied to `other` alone: `fresh[other]`,
/// respectively `Derivative[k][fresh][other]`.
fn rename_indexed_vars(expr: &Expr, renames: &[IndexedRename]) -> Expr {
  if renames.is_empty() {
    return expr.clone();
  }
  // `Derivative[0, k][head][index, other]`, the parsed form of
  // `D[head[index, x], {x, k}]`: the leading `0` says the derivative isn't
  // taken with respect to the index slot, so it collapses away with it.
  if let Expr::CurriedCall { func, args } = expr
    && args.len() == 2
    && let Expr::CurriedCall {
      func: deriv_head,
      args: fname_args,
    } = func.as_ref()
    && fname_args.len() == 1
    && let Expr::Identifier(fname) = &fname_args[0]
    && let Expr::FunctionCall {
      name: deriv_name,
      args: orders,
    } = deriv_head.as_ref()
    && deriv_name == "Derivative"
    && orders.len() == 2
    && matches!(orders[0], Expr::Integer(0))
  {
    let index_key = expr_to_string(&args[0]);
    if let Some(r) = renames
      .iter()
      .find(|r| r.head == *fname && expr_to_string(&r.index) == index_key)
    {
      let other = rename_indexed_vars(&args[1], renames);
      return Expr::CurriedCall {
        func: Box::new(Expr::CurriedCall {
          func: Box::new(call1("Derivative", orders[1].clone())),
          args: vec![Expr::Identifier(r.fresh.clone())],
        }),
        args: vec![other],
      };
    }
  }
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 2
  {
    let index_key = expr_to_string(&args[0]);
    if let Some(r) = renames
      .iter()
      .find(|r| r.head == *name && expr_to_string(&r.index) == index_key)
    {
      let other = rename_indexed_vars(&args[1], renames);
      return Expr::FunctionCall {
        name: r.fresh.clone(),
        args: vec![other].into(),
      };
    }
  }
  map_children(expr, &|c| rename_indexed_vars(c, renames))
}

/// The inverse of [`rename_indexed_vars`], applied to the solution rules:
/// `NDSolve$idx$x$1 -> InterpolatingFunction[…][t]` becomes
/// `x[1, t] -> InterpolatingFunction[…][t]`.
fn restore_indexed_vars(expr: &Expr, renames: &[IndexedRename]) -> Expr {
  if renames.is_empty() {
    return expr.clone();
  }
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && let Some(r) = renames.iter().find(|r| r.fresh == *name)
  {
    let other = restore_indexed_vars(&args[0], renames);
    return Expr::FunctionCall {
      name: r.head.clone(),
      args: vec![r.index.clone(), other].into(),
    };
  }
  if let Expr::Identifier(name) = expr
    && let Some(r) = renames.iter().find(|r| r.fresh == *name)
  {
    return Expr::FunctionCall {
      name: r.head.clone(),
      args: vec![r.index.clone()].into(),
    };
  }
  map_children(expr, &|c| restore_indexed_vars(c, renames))
}

// ─── The NDSolve integrator ────────────────────────────────────────────
//
// Handles every shape NDSolve takes: one equation or several coupled
// dependent functions (`NDSolve[eqns, {θ, ϕ, ψ}, {t, t0, t1}]`), any
// order, linear or not, equations that are implicit in the highest
// derivatives (each equation may mix θ'', ϕ'', ψ''), initial conditions
// given at an interior point of the domain (integrating both
// directions), and the `Method -> {"EventLocator", …}` option that stops
// integration when an event function crosses zero.
//
// Each step evaluates the residuals numerically: with the state fixed,
// every residual is affine in the vector of highest derivatives, so one
// evaluation at h = 0 and one per unit vector recovers the linear system
// M·h = -c, which Gaussian elimination solves. Residuals are compiled to
// a small numeric closure tree (`NExpr`) for speed, with a fall-back to
// full symbolic evaluation for operations the compiler doesn't know.

/// The event specification of `Method -> {"EventLocator", "Event" -> g,
/// "EventAction" :> action}`: the (still symbolic) event function and the
/// held action.
struct EventSpec {
  event: Expr,
  action: Option<Expr>,
}

/// Extract an `EventLocator` event from NDSolve's option rules. Returns
/// `None` when no such method is given (any other `Method` is ignored —
/// the integrator itself stays the same).
fn parse_event_locator_option(opts: &[Expr]) -> Option<EventSpec> {
  for opt in opts {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = opt
    else {
      continue;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Method") {
      continue;
    }
    let Expr::List(items) = replacement.as_ref() else {
      continue;
    };
    let is_event_locator = items
      .first()
      .is_some_and(|i| matches!(i, Expr::String(s) | Expr::Identifier(s) if s == "EventLocator"));
    if !is_event_locator {
      continue;
    }
    let mut event = None;
    let mut action = None;
    for item in &items[1..] {
      if let Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } = item
      {
        match pattern.as_ref() {
          Expr::String(s) | Expr::Identifier(s) if s == "Event" => {
            event = Some(replacement.as_ref().clone());
          }
          Expr::String(s) | Expr::Identifier(s) if s == "EventAction" => {
            action = Some(replacement.as_ref().clone());
          }
          _ => {}
        }
      }
    }
    // A condition (`y[t] < 0`) fires where its two sides cross.
    let event = event.map(|e| match &e {
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2
        && matches!(
          operators[0],
          ComparisonOp::Less
            | ComparisonOp::LessEqual
            | ComparisonOp::Greater
            | ComparisonOp::GreaterEqual
        ) =>
      {
        minus2(operands[0].clone(), operands[1].clone())
      }
      _ => e,
    });
    if let Some(event) = event {
      return Some(EventSpec { event, action });
    }
  }
  None
}

/// One dependent function of the system.
struct SysFunc {
  name: String,
  /// Order of its highest derivative in the equations.
  order: usize,
  /// Whether the result rule should use the bare-symbol form
  /// (`θ -> InterpolatingFunction[…]`); `false` means the `θ[t]` form.
  function_form: bool,
  /// Initial values `[f(x0), f'(x0), …]`, filled during IC parsing.
  ics: Vec<Option<f64>>,
}

/// Solve a (possibly coupled) system. Returns `Ok(None)` when the input
/// doesn't have a shape this solver understands, so the caller can leave
/// the expression unevaluated.
/// The dependent function's name in an `NDSolve` second argument entry:
/// `f` or `f[x]`.
fn dep_name(item: &Expr) -> Option<String> {
  match item {
    Expr::Identifier(name) => Some(name.clone()),
    Expr::FunctionCall { name, .. } => Some(name.clone()),
    _ => None,
  }
}

/// Recognize a `Derivative[k][f]` (bare) or `Derivative[k][f][x]` (applied)
/// second-argument entry as a request for `f`'s `k`-th derivative,
/// returned alongside `f` itself — e.g.
/// `NDSolve[eqns, {y, y'}, {t, t0, t1}]` returns both `y ->
/// InterpolatingFunction[…]` and `Derivative[1][y] ->
/// InterpolatingFunction[…]`, sparing the caller from differentiating the
/// interpolant. Returns `(function name, order, function_form)`;
/// `function_form` is `false` when the entry was applied to `x_name`.
/// The parser builds `Derivative[k][f]` as nested `CurriedCall`s
/// (`Derivative[k]` called on `f`), so that raw shape is matched first;
/// the flattened `FunctionCall("Derivative", [k, f])` form some
/// normalization passes produce is matched too, defensively.
fn derivative_dep_item(
  item: &Expr,
  x_name: &str,
) -> Option<(String, usize, bool)> {
  if let Expr::CurriedCall { func, args } = item
    && args.len() == 1
  {
    if let Expr::Identifier(fname) = &args[0]
      && let Expr::FunctionCall {
        name: deriv_name,
        args: deriv_args,
      } = func.as_ref()
      && deriv_name == "Derivative"
      && deriv_args.len() == 1
      && let Expr::Integer(k) = &deriv_args[0]
    {
      return Some((fname.clone(), *k as usize, true));
    }
    if matches!(&args[0], Expr::Identifier(a) if a == x_name)
      && let Some((fname, k, _)) = derivative_dep_item(func.as_ref(), x_name)
    {
      return Some((fname, k, false));
    }
  }
  if let Expr::FunctionCall { name, args } = item
    && name == "Derivative"
  {
    if args.len() == 2
      && let Expr::Integer(k) = &args[0]
      && let Expr::Identifier(fname) = &args[1]
    {
      return Some((fname.clone(), *k as usize, true));
    }
    if args.len() == 3
      && let Expr::Integer(k) = &args[0]
      && let Expr::Identifier(fname) = &args[1]
      && matches!(&args[2], Expr::Identifier(a) if a == x_name)
    {
      return Some((fname.clone(), *k as usize, false));
    }
  }
  None
}

/// Flatten every nesting level of `List` in an `NDSolve`/`DSolve`
/// equations argument into individual equations. A single function's
/// initial conditions are often grouped into their own sublist —
/// `{ode, {ic1, ic2}}` — and that grouping can nest arbitrarily deep, so
/// each level is flattened, not just the outermost.
/// `DiracDelta[a t + b]` terms in an ODE are impulses: wolframscript drops
/// them from the equation and, where the root `r = -b/a` lies strictly inside
/// the domain, makes the next-lower derivative of the equation's unknown
/// jump by the term's coefficient (taken at `t = r`, from the state just
/// before) over `|a|` times the top derivative's coefficient. The impulses
/// become `WhenEvent[t == r, …]` resets, one per root; a root on either end
/// of the domain is ignored. Returns None when no equation has an impulse
/// this covers.
fn dirac_delta_events(
  args: &[Expr],
) -> Result<Option<Vec<Expr>>, InterpreterError> {
  if !mentions_head(&args[0], "DiracDelta") {
    return Ok(None);
  }
  let Expr::List(domain) = &args[2] else {
    return Ok(None);
  };
  let [Expr::Identifier(t), lo, hi] = domain.as_slice() else {
    return Ok(None);
  };
  let (Some(lo), Some(hi)) = (nval_to_f64(lo), nval_to_f64(hi)) else {
    return Ok(None);
  };
  let (lo, hi) = (lo.min(hi), lo.max(hi));
  let unknowns: Vec<String> = match &args[1] {
    Expr::List(items) => items.iter().collect::<Vec<_>>(),
    other => vec![other],
  }
  .into_iter()
  .filter_map(|u| match u {
    Expr::Identifier(n) => Some(n.clone()),
    Expr::FunctionCall { name, .. } => Some(name.clone()),
    _ => None,
  })
  .collect();
  fn deltas(expr: &Expr, out: &mut Vec<Expr>) {
    if let Expr::FunctionCall { name, args } = expr
      && name == "DiracDelta"
      && args.len() == 1
    {
      if !out.iter().any(|d| exprs_match(d, expr)) {
        out.push(expr.clone());
      }
      return;
    }
    for c in expr_children(expr) {
      deltas(c, out);
    }
  }
  let mut equations = Vec::new();
  flatten_eq_list(&args[0], &mut equations);
  // (root, jumping derivative, increment) per impulse.
  let mut jumps: Vec<(f64, Expr, Expr)> = Vec::new();
  let mut rewritten = Vec::with_capacity(equations.len());
  for eq in &equations {
    let Expr::Comparison {
      operands,
      operators,
    } = eq
    else {
      rewritten.push(eq.clone());
      continue;
    };
    let mut found = Vec::new();
    deltas(eq, &mut found);
    if found.is_empty() {
      rewritten.push(eq.clone());
      continue;
    }
    if operands.len() != 2 || operators[0] != ComparisonOp::Equal {
      return Ok(None);
    }
    // The single unknown the equation differentiates, at its top order.
    let ordered: Vec<(&String, usize)> = unknowns
      .iter()
      .map(|f| (f, max_derivative_order(eq, f)))
      .filter(|(_, k)| *k > 0)
      .collect();
    let [(f, k)] = ordered.as_slice() else {
      return Ok(None);
    };
    let derivative = |order: usize| -> Result<Expr, InterpreterError> {
      if order == 0 {
        Ok(Expr::FunctionCall {
          name: (*f).clone(),
          args: vec![Expr::Identifier(t.clone())].into(),
        })
      } else {
        crate::evaluator::evaluate_expr_to_expr(&crate::syntax::string_to_expr(
          &format!("Derivative[{order}][{f}][{t}]"),
        )?)
      }
    };
    let residual = crate::evaluator::evaluate_function_call_ast(
      "Expand",
      &[Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(operands[0].clone()),
        right: Box::new(operands[1].clone()),
      }],
    )?;
    let top = crate::evaluator::evaluate_function_call_ast(
      "Coefficient",
      &[residual.clone(), derivative(*k)?],
    )?;
    if matches!(top, Expr::Integer(0)) {
      return Ok(None);
    }
    for delta in &found {
      let Expr::FunctionCall { args: dargs, .. } = delta else {
        unreachable!()
      };
      // A linear argument a t + b.
      let slope = crate::evaluator::evaluate_function_call_ast(
        "D",
        &[dargs[0].clone(), Expr::Identifier(t.clone())],
      )?;
      let Some(a) = nval_to_f64(&slope) else {
        return Ok(None);
      };
      let Some(b) = nval_to_f64(&crate::syntax::substitute_variable(
        &dargs[0],
        t,
        &Expr::Integer(0),
      )) else {
        return Ok(None);
      };
      if a == 0.0 {
        return Ok(None);
      }
      let root = -b / a;
      if !(root > lo && root < hi) {
        continue;
      }
      let coefficient = crate::evaluator::evaluate_function_call_ast(
        "Coefficient",
        &[residual.clone(), delta.clone()],
      )?;
      if mentions_head(&coefficient, "DiracDelta") {
        return Ok(None);
      }
      // `t` takes the root's value, the unknowns' `f[t]` state stays.
      let keep_state: Vec<String> = unknowns
        .iter()
        .map(|u| format!("{u}[{t}] | Derivative[_][{u}][{t}]"))
        .collect();
      let at_root = crate::evaluator::evaluate_function_call_ast(
        "ReplaceAll",
        &[
          coefficient,
          crate::syntax::string_to_expr(&format!(
            "{{state$ : ({}) :> state$, {t} -> {root:?}}}",
            keep_state.join(" | ")
          ))?,
        ],
      )?;
      let increment = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::Real(-1.0 / a.abs()),
          at_root,
          crate::evaluator::evaluate_function_call_ast(
            "Power",
            &[top.clone(), Expr::Integer(-1)],
          )?,
        ],
      )?;
      jumps.push((root, derivative(k - 1)?, increment));
    }
    let zeroed = crate::evaluator::evaluate_function_call_ast(
      "ReplaceAll",
      &[
        eq.clone(),
        crate::syntax::string_to_expr("_DiracDelta -> 0")?,
      ],
    )?;
    rewritten.push(zeroed);
  }
  let mut roots: Vec<f64> = jumps.iter().map(|(r, _, _)| *r).collect();
  roots.sort_by(f64::total_cmp);
  roots.dedup();
  for root in roots {
    let rules: Vec<Expr> = jumps
      .iter()
      .filter(|(r, _, _)| *r == root)
      .map(|(_, target, increment)| Expr::Rule {
        pattern: Box::new(target.clone()),
        replacement: Box::new(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![target.clone(), increment.clone()].into(),
        }),
      })
      .collect();
    rewritten.push(Expr::FunctionCall {
      name: "WhenEvent".to_string(),
      args: vec![
        Expr::Comparison {
          operands: vec![Expr::Identifier(t.clone()), Expr::Real(root)],
          operators: vec![ComparisonOp::Equal],
        },
        Expr::List(rules.into()),
      ]
      .into(),
    });
  }
  let mut out = args.to_vec();
  out[0] = Expr::List(rewritten.into());
  Ok(Some(out))
}

fn flatten_eq_list(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::List(items) => {
      for item in items {
        flatten_eq_list(item, out);
      }
    }
    // `{x[0], y[0]} == {0, 1}` states one condition per component.
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 1
      && operators[0] == ComparisonOp::Equal
      && matches!(
        (&operands[0], &operands[1]),
        (Expr::List(a), Expr::List(b)) if a.len() == b.len()
      ) =>
    {
      if let (Expr::List(a), Expr::List(b)) = (&operands[0], &operands[1]) {
        for (l, r) in a.iter().zip(b.iter()) {
          flatten_eq_list(
            &Expr::Comparison {
              operands: vec![l.clone(), r.clone()],
              operators: vec![ComparisonOp::Equal],
            },
            out,
          );
        }
      }
    }
    other => out.push(other.clone()),
  }
}

/// The function a solution rule is for: `f -> …` or `f[x] -> …`.
fn rule_target_name(rule: &Expr) -> Option<&str> {
  let Expr::Rule { pattern, .. } = rule else {
    return None;
  };
  match pattern.as_ref() {
    Expr::Identifier(name) => Some(name.as_str()),
    Expr::FunctionCall { name, .. } => Some(name.as_str()),
    _ => None,
  }
}

/// Functions applied to the independent variable in `expr` — `g[x]` or a
/// derivative `g'[x]` — that are not among `funcs` and have no definition
/// of their own (a built-in like `Sin[x]` or a user function evaluates
/// away or is known), in order of first appearance.
fn collect_unlisted_unknowns(
  expr: &Expr,
  x_name: &str,
  funcs: &[SysFunc],
  out: &mut Vec<String>,
) {
  let is_x = |e: &Expr| matches!(e, Expr::Identifier(a) if a == x_name);
  let mut consider = |name: &str| {
    if !funcs.iter().any(|f| f.name == name)
      && !out.iter().any(|n| n == name)
      && !crate::evaluator::functions::is_builtin_symbol(name)
      && crate::FUNC_DEFS.with(|m| m.borrow().get(name).is_none())
    {
      out.push(name.to_string());
    }
  };
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 && is_x(&args[0]) => {
      consider(name);
    }
    Expr::FunctionCall { name, args }
      if name == "Derivative"
        && args.len() == 3
        && is_x(&args[2])
        && let Expr::Identifier(g) = &args[1] =>
    {
      consider(g);
    }
    _ => {}
  }
  // `Derivative[n][g][x]` in either curried shape.
  if let Expr::CurriedCall { func, args } = expr
    && args.len() == 1
    && is_x(&args[0])
  {
    match func.as_ref() {
      Expr::FunctionCall { name, args: d }
        if name == "Derivative"
          && d.len() == 2
          && let Expr::Identifier(g) = &d[1] =>
      {
        consider(g);
      }
      Expr::CurriedCall {
        func: inner,
        args: g,
      } if g.len() == 1
        && matches!(inner.as_ref(), Expr::FunctionCall { name, .. } if name == "Derivative")
        && let Expr::Identifier(g) = &g[0] =>
      {
        consider(g);
      }
      _ => {}
    }
  }
  match expr {
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_unlisted_unknowns(a, x_name, funcs, out);
      }
    }
    Expr::List(items) => {
      for a in items {
        collect_unlisted_unknowns(a, x_name, funcs, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_unlisted_unknowns(left, x_name, funcs, out);
      collect_unlisted_unknowns(right, x_name, funcs, out);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_unlisted_unknowns(operand, x_name, funcs, out);
    }
    Expr::Comparison { operands, .. } => {
      for a in operands {
        collect_unlisted_unknowns(a, x_name, funcs, out);
      }
    }
    Expr::CurriedCall { func, args } => {
      collect_unlisted_unknowns(func, x_name, funcs, out);
      for a in args {
        collect_unlisted_unknowns(a, x_name, funcs, out);
      }
    }
    _ => {}
  }
}

/// Find an algebraic constraint among `odes` that determines one of `funcs`
/// explicitly: an equation with no derivative of any unknown, in which
/// exactly one unknown appears and appears nowhere differentiated elsewhere.
/// Returns its index, that function's index, and the expression it solves to.
fn find_algebraic_constraint(
  odes: &[Expr],
  funcs: &[SysFunc],
) -> Option<(usize, usize, Expr)> {
  let orders: Vec<std::collections::HashMap<String, usize>> =
    odes.iter().map(derivative_orders).collect();
  let never_differentiated =
    |name: &str| orders.iter().all(|o| o.get(name).is_none_or(|&k| k == 0));
  for (ei, eq) in odes.iter().enumerate() {
    // An equation already solved for an unknown no equation differentiates
    // (`F0[t] == 1 + Kp (2 - h[t]) - Kp τd h'[t]`) is substituted as it
    // stands, even when its right side holds other unknowns' derivatives.
    if let Some((a, b)) = as_equal_pair(eq) {
      for (lhs, rhs) in [(a, b), (b, a)] {
        if let Expr::FunctionCall { name, args } = lhs
          && args.len() == 1
          && let Some(fi) = funcs.iter().position(|f| &f.name == name)
          && !expr_mentions_function(rhs, name)
          && never_differentiated(name)
        {
          return Some((ei, fi, rhs.clone()));
        }
      }
    }
    if funcs
      .iter()
      .any(|f| orders[ei].get(&f.name).is_some_and(|&k| k > 0))
    {
      continue;
    }
    // The unknown this constraint defines is one it mentions that no
    // equation ever differentiates — the others in it are dynamic
    // variables the solution is written in terms of.
    let Some(fi) = funcs.iter().enumerate().position(|(_, f)| {
      expr_mentions_function(eq, &f.name) && never_differentiated(&f.name)
    }) else {
      continue;
    };
    // A constraint transcendental in the unknown (`Exp[-b/T[t]]`) has no
    // closed form: `Solve` would only burn time before giving up, so it is
    // left to the numerical elimination in [`find_numeric_constraint`].
    if function_in_transcendental_context(eq, &funcs[fi].name, false) {
      continue;
    }
    let Some(solved) = solve_for_function(eq, &funcs[fi].name) else {
      continue;
    };
    return Some((ei, fi, solved));
  }
  None
}

/// Whether `fname[…]` occurs inside anything other than sums, products and
/// integer powers — an `Exp`, a `Log`, a symbolic exponent, … — i.e. whether
/// solving for it in closed form is hopeless in general.
fn function_in_transcendental_context(
  expr: &Expr,
  fname: &str,
  inside: bool,
) -> bool {
  if matches!(expr, Expr::FunctionCall { name, .. } if name == fname) {
    return inside;
  }
  let now_inside = inside
    || match expr {
      Expr::FunctionCall { name, .. } => {
        !matches!(name.as_str(), "Plus" | "Times" | "Subtract" | "Divide")
          && name != fname
          && expr_mentions_function(expr, fname)
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        right,
        ..
      } => !matches!(right.as_ref(), Expr::Integer(_)),
      _ => false,
    };
  expr_children(expr)
    .into_iter()
    .any(|c| function_in_transcendental_context(c, fname, now_inside))
}

/// Like [`find_algebraic_constraint`], but for constraints that cannot be
/// solved in closed form: returns the equations' and the unknowns' indices,
/// leaving the unknowns to be found numerically at every evaluation. When
/// the derivative-free equations and the never-differentiated unknowns they
/// mention are equally many they form one coupled system; otherwise the
/// first such equation is paired with the first such unknown.
fn find_numeric_constraints(
  odes: &[Expr],
  funcs: &[SysFunc],
) -> Option<(Vec<usize>, Vec<usize>)> {
  let orders: Vec<std::collections::HashMap<String, usize>> =
    odes.iter().map(derivative_orders).collect();
  let algebraic: Vec<usize> = (0..odes.len())
    .filter(|&i| {
      !funcs
        .iter()
        .any(|f| orders[i].get(&f.name).is_some_and(|&k| k > 0))
    })
    .collect();
  let free: Vec<usize> = funcs
    .iter()
    .enumerate()
    .filter(|(_, f)| {
      orders
        .iter()
        .all(|o| o.get(&f.name).is_none_or(|&k| k == 0))
        && algebraic
          .iter()
          .any(|&ei| expr_mentions_function(&odes[ei], &f.name))
    })
    .map(|(i, _)| i)
    .collect();
  if !algebraic.is_empty() && algebraic.len() == free.len() {
    return Some((algebraic, free));
  }
  for &ei in &algebraic {
    if let Some(&fi) = free
      .iter()
      .find(|&&fi| expr_mentions_function(&odes[ei], &funcs[fi].name))
    {
      return Some((vec![ei], vec![fi]));
    }
  }
  None
}

/// Whether `expr` contains `fname[…]` anywhere.
fn expr_mentions_function(expr: &Expr, fname: &str) -> bool {
  if matches!(expr, Expr::FunctionCall { name, .. } if name == fname) {
    return true;
  }
  expr_children(expr)
    .into_iter()
    .any(|c| expr_mentions_function(c, fname))
}

/// Whether `expr` is, at the top level, exactly a call to `fname`
/// (`fname[…]`) — not merely an expression that mentions one somewhere
/// inside it.
fn is_bare_call(expr: &Expr, fname: &str) -> bool {
  matches!(expr, Expr::FunctionCall { name, .. } if name == fname)
}

/// Solve an algebraic equation for `fname[x]`, returning the right-hand side
/// of the single solution. `None` when it does not solve uniquely.
fn solve_for_function(eq: &Expr, fname: &str) -> Option<Expr> {
  let unknown = find_function_call(eq, fname)?;

  // Fast path: the equation already reads `fname[x] == rhs` (or the
  // mirror image) — an explicit definition, not one that needs solving.
  // Elimination keeps substituting the *other* side of equations like this
  // as more unknowns fall out of the system, so their `rhs` can grow large
  // well before this one's own turn comes up; handing that growing
  // expression to the general `Solve[]` below on every candidate equation,
  // every elimination round, is what makes a system of this DAE's size
  // (dozens of unknowns) stop scaling. Recognizing the already-solved
  // shape directly is exact — not an approximation of what `Solve[]` would
  // return — since `rhs` not mentioning `fname` means it already is the
  // unique solution.
  if let Some((lhs, rhs)) = as_equal_pair(eq) {
    if is_bare_call(lhs, fname) && !expr_mentions_function(rhs, fname) {
      return Some(rhs.clone());
    }
    if is_bare_call(rhs, fname) && !expr_mentions_function(lhs, fname) {
      return Some(lhs.clone());
    }
  }

  // An equation linear in the unknown, `c u + r == 0` with `c` free of it,
  // gives `-r/c` straight away. `Solve` would get the same answer, but it
  // expands everything first, which is what makes the growing equations of
  // a large DAE (a distillation column's dozens of stages) stop scaling.
  let (lhs, rhs) = as_equal_pair(eq)?;
  let u = Expr::Identifier("\u{2620}dae\u{2620}".to_string());
  let residual = crate::functions::polynomial_ast::solve::substitute_expr(
    &crate::helpers::minus2(lhs.clone(), rhs.clone()),
    &unknown,
    &u,
  );
  let eval = crate::evaluator::evaluate_expr_to_expr;
  if let Ok(c) = eval(&call("D", vec![residual.clone(), u.clone()]))
    && crate::functions::calculus_ast::is_constant_wrt(
      &c,
      "\u{2620}dae\u{2620}",
    )
    && !matches!(c, Expr::Integer(0))
    && let Ok(r) =
      eval(&crate::functions::polynomial_ast::solve::substitute_expr(
        &residual,
        &u,
        &Expr::Integer(0),
      ))
  {
    return eval(&call(
      "Times",
      vec![
        Expr::Integer(-1),
        r,
        call("Power", vec![c, Expr::Integer(-1)]),
      ],
    ))
    .ok();
  }
  // A large equation that is not linear in the unknown is left to the
  // numerical elimination rather than to an expanding `Solve`.
  if expr_size_exceeds(eq, 400) {
    return None;
  }

  let solved = eval(&call("Solve", vec![eq.clone(), unknown])).ok()?;
  let Expr::List(ref sols) = solved else {
    return None;
  };
  let [Expr::List(rules)] = &sols[..] else {
    return None;
  };
  let [Expr::Rule { replacement, .. }] = &rules[..] else {
    return None;
  };
  Some(replacement.as_ref().clone())
}

/// Whether `expr` has more than `limit` nodes, counting no further than
/// that.
fn expr_size_exceeds(expr: &Expr, limit: usize) -> bool {
  fn count(e: &Expr, left: &mut usize) -> bool {
    if *left == 0 {
      return true;
    }
    *left -= 1;
    expr_children(e).into_iter().any(|c| count(c, left))
  }
  let mut left = limit;
  count(expr, &mut left)
}

/// The first `fname[…]` call inside `expr`.
fn find_function_call(expr: &Expr, fname: &str) -> Option<Expr> {
  if matches!(expr, Expr::FunctionCall { name, .. } if name == fname) {
    return Some(expr.clone());
  }
  expr_children(expr)
    .into_iter()
    .find_map(|c| find_function_call(c, fname))
}

/// The interpolation order NDSolve's InterpolatingFunctions carry, matching
/// the Wolfram Language's default. Reading the solution back between grid
/// points then has the accuracy of the integration itself; a linear
/// interpolation would throw most of it away — visibly so through `θ'`,
/// which differentiates the interpolating piece.
const NDSOLVE_INTERPOLATION_ORDER: i128 = 3;

/// Restrict integrated solution samples (ascending in x) to `[lo, hi]`.
///
/// A bound that falls between two grid points becomes a sample of its own,
/// interpolated from the neighbours at the same order the returned
/// `InterpolatingFunction` uses, so the reported domain is exactly the
/// requested range rather than the nearest grid point to it. A bound that
/// already coincides with a grid point keeps that point untouched.
fn clip_points_to_range(
  points: &[(f64, Vec<f64>)],
  lo: f64,
  hi: f64,
) -> Vec<(f64, Vec<f64>)> {
  let tol = (hi - lo).abs().max(1.0) * 1e-9;
  let xs: Vec<f64> = points.iter().map(|(x, _)| *x).collect();
  let bound = |target: f64| -> (f64, Vec<f64>) {
    if let Some(p) = points.iter().find(|(x, _)| (x - target).abs() <= tol) {
      return p.clone();
    }
    let state = (0..points[0].1.len())
      .map(|k| {
        let ys: Vec<f64> = points.iter().map(|(_, s)| s[k]).collect();
        interp_1d_xy_f64(&xs, &ys, target, NDSOLVE_INTERPOLATION_ORDER as usize)
      })
      .collect();
    (target, state)
  };
  let mut clipped = vec![bound(lo)];
  clipped.extend(
    points
      .iter()
      .filter(|(x, _)| *x > lo + tol && *x < hi - tol)
      .cloned(),
  );
  clipped.push(bound(hi));
  clipped
}

fn ndsolve_system(
  args: &[Expr],
  event: Option<&EventSpec>,
) -> Result<Option<Expr>, InterpreterError> {
  // Domain {x, xmin, xmax}, or the shorthand {x, xmax} that integrates
  // from the initial conditions' x-value out to xmax — the x_min side is
  // resolved below, once x0 is known from the equations.
  let Expr::List(domain_items) = &args[2] else {
    return Ok(None);
  };
  let Expr::Identifier(x_name) = &domain_items[0] else {
    return Ok(None);
  };
  let (x_min_given, x_max): (Option<f64>, f64) = match domain_items.len() {
    3 => {
      let Some(min) = nval_to_f64(&domain_items[1]) else {
        return Ok(None);
      };
      // `{t, t0, Infinity}` is only solvable when an event is present to
      // stop the integration — nothing else would ever tell a fixed-step
      // method when to quit, so without one it's left unevaluated like
      // any other pattern this solver doesn't cover.
      let max = if is_infinity(&domain_items[2]) {
        if event.is_none() && !mentions_head(&args[0], "WhenEvent") {
          return Ok(None);
        }
        f64::INFINITY
      } else {
        let Some(max) = nval_to_f64(&domain_items[2]) else {
          return Ok(None);
        };
        max
      };
      // A descending domain `{t, 0, -2}` integrates backwards and is
      // reported over the ascending interval `[-2, 0]`, so its bounds are
      // swapped. A NaN bound compares as neither and falls through to the
      // check below.
      let (min, max) = if max < min { (max, min) } else { (min, max) };
      // NaN bounds must bail out too, so compare via partial_cmp rather
      // than a negated float comparison. A zero-length domain
      // `{t, t0, t0}` is accepted (an animation clock starts there); the
      // initial-value path below answers it from the initial conditions.
      if !matches!(
        max.partial_cmp(&min),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
      ) {
        return Ok(None);
      }
      (Some(min), max)
    }
    2 => {
      let Some(target) = nval_to_f64(&domain_items[1]) else {
        return Ok(None);
      };
      (None, target)
    }
    _ => return Ok(None),
  };

  // Dependent functions: `{θ, ϕ, ψ}`, a single symbol, or `f[x]` forms.
  let dep_items: Vec<&Expr> = match &args[1] {
    Expr::List(items) => items.iter().collect(),
    other => vec![other],
  };
  let mut funcs: Vec<SysFunc> = Vec::with_capacity(dep_items.len());
  // A `Derivative[k][f]` entry doesn't introduce a new dependent function;
  // it asks for `f`'s k-th derivative alongside `f`'s own solution, so
  // it's collected separately and resolved once `f`'s order is known.
  let mut deriv_requests: Vec<(String, usize, bool)> = Vec::new();
  for item in dep_items {
    if let Some(req) = derivative_dep_item(item, x_name) {
      deriv_requests.push(req);
      continue;
    }
    match item {
      Expr::Identifier(name) => funcs.push(SysFunc {
        name: name.clone(),
        order: 0,
        function_form: true,
        ics: Vec::new(),
      }),
      Expr::FunctionCall { name, args: fargs }
        if fargs.len() == 1
          && matches!(&fargs[0], Expr::Identifier(a) if a == x_name) =>
      {
        funcs.push(SysFunc {
          name: name.clone(),
          order: 0,
          function_form: false,
          ics: Vec::new(),
        });
      }
      _ => return Ok(None),
    }
  }
  if funcs.is_empty() {
    return Ok(None);
  }

  // Split equations into ODEs and initial conditions. The equations
  // argument nests arbitrarily — a common idiom groups a function's
  // initial conditions into their own sublist, e.g. `{ode, {ic1, ic2}}` —
  // so every level of `List` is flattened rather than just the outermost.
  let mut eq_items: Vec<Expr> = Vec::new();
  flatten_eq_list(&args[0], &mut eq_items);
  let eq_items: Vec<Expr> = flatten_chained_equalities(&eq_items)
    .into_iter()
    .map(|eq| history_condition_to_ic(&eq, x_name).unwrap_or(eq))
    .collect();
  // `WhenEvent[Mod[x, p] == 0, action]` entries are periodic actions, not
  // equations: split them off before the equations are classified.
  let mut raw_time_events: Vec<RawTimeEvent> = Vec::new();
  let mut raw_state_events: Vec<RawStateEvent> = Vec::new();
  let mut eq_items_no_events: Vec<Expr> = Vec::with_capacity(eq_items.len());
  for eq in eq_items {
    if let Expr::FunctionCall { name, args: wargs } = &eq
      && name == "WhenEvent"
    {
      // `WhenEvent[{c1, c2, …}, action]` fires on any of the conditions:
      // one event per condition, all with the same action.
      if let [Expr::List(conds), action] = wargs.as_slice() {
        for c in conds {
          raw_state_events.push(RawStateEvent {
            condition: c.clone(),
            action: action.clone(),
          });
        }
        continue;
      }
      match parse_periodic_when_event(wargs, x_name) {
        Some(ev) => raw_time_events.push(ev),
        None if wargs.len() == 2 => raw_state_events.push(RawStateEvent {
          condition: wargs[0].clone(),
          action: wargs[1].clone(),
        }),
        None => return Ok(None),
      }
    } else {
      eq_items_no_events.push(eq);
    }
  }
  let eq_items = eq_items_no_events;
  let time_events_empty =
    raw_time_events.is_empty() && raw_state_events.is_empty();
  let mut odes: Vec<Expr> = Vec::new();
  // (function name, derivative order, evaluation point, value) — by name,
  // not index: an eliminated constraint variable shifts the positions in
  // `funcs`. Conditions are no longer required to share one point here —
  // a two-point boundary value problem states one condition at each end
  // of the domain — so the point travels with each condition and the
  // shape (one shared point vs. exactly two) is decided once every
  // condition has been collected, below.
  let mut raw_ics: Vec<(String, usize, f64, f64)> = Vec::new();
  // Conditions whose value refers to another initial value
  // (`PD[0] == 1 - PB[0]`) wait until those values are known.
  let mut pending: Vec<(Expr, String)> = Vec::new();
  // Conditions that tie several values at one point together, as `(point,
  // residual)` with each `f^(k)[point]` replaced by its state placeholder.
  let mut general_conds: Vec<(f64, Expr)> = Vec::new();
  // `f[x0] -> value` for every condition read so far.
  let mut known: Vec<Expr> = Vec::new();
  let record_known = |eq: &Expr, value: f64, known: &mut Vec<Expr>| {
    if let Some((a, b)) = as_equal_pair(eq) {
      let lhs = if nval_to_f64(b).is_some() { a } else { b };
      known.push(Expr::Rule {
        pattern: Box::new(lhs.clone()),
        replacement: Box::new(Expr::Real(value)),
      });
    }
  };
  for eq in &eq_items {
    let mut is_ic = false;
    for f in &funcs {
      if is_initial_condition(eq, &f.name, x_name) {
        match parse_numeric_initial_condition(eq, &f.name) {
          Some((order, x_val, y_val)) => {
            raw_ics.push((f.name.clone(), order, x_val, y_val));
            record_known(eq, y_val, &mut known);
          }
          None => pending.push((eq.clone(), f.name.clone())),
        }
        is_ic = true;
        break;
      }
    }
    if !is_ic {
      // An equation over the unknowns at one numeric point only
      // (`y'[1] + y[1] == 2`) is a condition, not an equation to integrate.
      match general_condition(eq, &funcs, x_name) {
        Some(cond) => general_conds.push(cond),
        None => odes.push(eq.clone()),
      }
    }
  }
  while !pending.is_empty() {
    let before = pending.len();
    let mut still = Vec::new();
    for (eq, fname) in pending {
      let substituted = match as_equal_pair(&eq) {
        Some((a, b)) => {
          let rules = Expr::List(known.clone().into());
          let side = |e: &Expr| -> Expr {
            crate::evaluator::evaluate_expr_to_expr(&Expr::ReplaceAll {
              expr: Box::new(e.clone()),
              rules: Box::new(rules.clone()),
            })
            .unwrap_or_else(|_| e.clone())
          };
          // Only the value side is rewritten; the side naming the unknown
          // stays as it is.
          let (a2, b2) = if is_initial_condition(
            &Expr::Comparison {
              operands: vec![a.clone(), Expr::Integer(0)],
              operators: vec![ComparisonOp::Equal],
            },
            &fname,
            x_name,
          ) {
            (a.clone(), side(b))
          } else {
            (side(a), b.clone())
          };
          Expr::Comparison {
            operands: vec![a2, b2],
            operators: vec![ComparisonOp::Equal],
          }
        }
        None => eq.clone(),
      };
      match parse_numeric_initial_condition(&substituted, &fname) {
        Some((order, x_val, y_val)) => {
          raw_ics.push((fname.clone(), order, x_val, y_val));
          record_known(&substituted, y_val, &mut known);
        }
        None => still.push((eq, fname)),
      }
    }
    if still.len() == before {
      // What is left relates several values at one point (`1 == f[0] -
      // f'[0]`, a Robin condition): kept as an equation for the
      // boundary-value solver below.
      for (eq, _) in &still {
        match general_condition(eq, &funcs, x_name) {
          Some(cond) => general_conds.push(cond),
          None => return Ok(None),
        }
      }
      break;
    }
    pending = still;
  }
  // An unknown the equations use but the caller did not list (a Lagrange
  // multiplier `λ[t]` in a constrained-motion model) is solved for with
  // the others when the system would otherwise be short of unknowns;
  // Wolfram does the same and reports only the functions it was asked for.
  let mut hidden: Vec<String> = Vec::new();
  if odes.len() > funcs.len() {
    let mut extra: Vec<String> = Vec::new();
    for ode in &odes {
      collect_unlisted_unknowns(ode, x_name, &funcs, &mut extra);
    }
    if funcs.len() + extra.len() == odes.len() {
      for name in extra {
        funcs.push(SysFunc {
          name: name.clone(),
          order: 0,
          function_form: true,
          ics: Vec::new(),
        });
        hidden.push(name);
      }
    }
  }

  // Index reduction: an algebraic equation whose unknowns are all
  // differentiated elsewhere (`y[t] == x[t]^2` next to `2 x'[t] + y'[t] ==
  // -x[t]`) determines no unknown on its own; its time derivative
  // (`y'[t] == 2 x[t] x'[t]`) is the differential equation the integrator
  // needs. The initial conditions already satisfy the constraint.
  let differentiated: Vec<String> = {
    let orders: Vec<std::collections::HashMap<String, usize>> =
      odes.iter().map(derivative_orders).collect();
    funcs
      .iter()
      .filter(|f| {
        orders
          .iter()
          .any(|o| o.get(&f.name).is_some_and(|&k| k > 0))
      })
      .map(|f| f.name.clone())
      .collect()
  };
  for ode in &mut odes {
    let mentioned: Vec<&SysFunc> = funcs
      .iter()
      .filter(|f| expr_mentions_function(ode, &f.name))
      .collect();
    // Only a constraint along the solution: an initial condition
    // (`x[0] == 1`) mentions its unknown at a point instead of at `t`.
    let at_variable = |f: &&SysFunc| {
      let call_at_x = Expr::FunctionCall {
        name: f.name.clone(),
        args: vec![Expr::Identifier(x_name.clone())].into(),
      };
      crate::functions::polynomial_ast::solve::expr_contains_expr(
        ode, &call_at_x,
      )
    };
    // An equation with any derivative in it (of a mentioned unknown or not:
    // `x'[t] == y[t]` mentions only `y` as a call) is no constraint.
    if mentioned.is_empty()
      || !mentioned.iter().all(at_variable)
      || {
        let own = derivative_orders(ode);
        funcs
          .iter()
          .any(|f| own.get(&f.name).is_some_and(|&k| k > 0))
      }
      || mentioned.iter().any(|f| !differentiated.contains(&f.name))
    {
      continue;
    }
    let Some((lhs, rhs)) = as_equal_pair(ode) else {
      continue;
    };
    let d = |side: &Expr| {
      crate::evaluator::evaluate_expr_to_expr(&call(
        "D",
        vec![side.clone(), Expr::Identifier(x_name.clone())],
      ))
    };
    if let (Ok(dl), Ok(dr)) = (d(lhs), d(rhs)) {
      *ode = Expr::Comparison {
        operands: vec![dl, dr],
        operators: vec![ComparisonOp::Equal],
      };
    }
  }

  // An algebraic constraint — an equation with no derivative in it — that
  // determines one unknown explicitly is eliminated: the constraint is
  // solved for that unknown, the solution substituted into the remaining
  // equations, and the unknown dropped from the system. This is how a
  // Lagrangian model states a rigid link (`l[t] + d - y[t] == L`), and it
  // turns an index-1 DAE the integrator cannot take into the ODE system it
  // can. The eliminated function is rebuilt from the solution afterwards.
  let mut eliminated: Vec<(SysFunc, Expr)> = Vec::new();
  loop {
    let Some((eq_idx, fi, solved)) = find_algebraic_constraint(&odes, &funcs)
    else {
      break;
    };
    // Substituting one solution into the next makes the expressions swell
    // (a distillation column's stage-by-stage equilibria); past a modest
    // size the remaining constraints are cheaper to solve numerically.
    if expr_size_exceeds(&solved, 300) {
      break;
    }
    let f = funcs.remove(fi);
    odes.remove(eq_idx);
    let target = Expr::FunctionCall {
      name: f.name.clone(),
      args: vec![Expr::Identifier(x_name.clone())].into(),
    };
    for ode in &mut odes {
      *ode = crate::functions::polynomial_ast::solve::substitute_expr(
        ode, &target, &solved,
      );
    }
    // A later constraint may refer to an already-eliminated function.
    for (_, prev) in &mut eliminated {
      *prev = crate::functions::polynomial_ast::solve::substitute_expr(
        prev, &target, &solved,
      );
    }
    eliminated.push((f, solved));
  }

  // A constraint with no closed-form solution (index-1 DAE): its unknown is
  // found by Newton iteration whenever the residuals are evaluated.
  let mut implicit: Vec<(SysFunc, Expr)> = Vec::new();
  if let Some((mut eq_idxs, mut fis)) = find_numeric_constraints(&odes, &funcs)
  {
    if event.is_some() {
      return Ok(None);
    }
    eq_idxs.sort_unstable_by(|a, b| b.cmp(a));
    fis.sort_unstable_by(|a, b| b.cmp(a));
    let eqs: Vec<Expr> = eq_idxs.iter().map(|&i| odes.remove(i)).collect();
    let fs: Vec<SysFunc> = fis.iter().map(|&i| funcs.remove(i)).collect();
    // Back in the order they were given, each paired with its equation.
    implicit = fs.into_iter().rev().zip(eqs.into_iter().rev()).collect();
  }

  if odes.len() != funcs.len() {
    return Ok(None);
  }

  // Determine each function's order from the equations — needed both to
  // classify the conditions below and to lay out the state vector.
  for ode in &odes {
    for f in &mut funcs {
      let order = max_derivative_order(ode, &f.name);
      f.order = f.order.max(order);
    }
  }
  // An unknown that is never differentiated is algebraic: it has no state
  // of its own and is solved for at every step together with the highest
  // derivatives it is coupled with (`λ[t] f'[x[t]] == m x''[t]`). A system
  // of nothing but such unknowns is no differential equation at all.
  if funcs.iter().all(|f| f.order == 0) {
    return Ok(None);
  }
  let algebraic_unknowns = funcs.iter().any(|f| f.order == 0);
  // Every requested derivative must be of a function actually being
  // solved for, and of an order already carried in its state vector
  // (0..order-1); the highest derivative itself isn't stored there.
  for (name, order, _) in &deriv_requests {
    let Some(f) = funcs.iter().find(|f| &f.name == name) else {
      return Ok(None);
    };
    if *order >= f.order {
      return Ok(None);
    }
  }

  // Classify the conditions by the distinct points they're given at. One
  // shared point is an initial value problem (the ordinary case, handled
  // exactly as before); a single second-order equation with one condition
  // at each of two distinct points — matching the solved domain's own
  // endpoints — is a two-point boundary value problem, solved below by
  // shooting: integrating the very same IVP stepper from a guessed value
  // at one end and adjusting that guess until the far condition is met.
  // Anything else (three or more distinct points, a coupled system, …)
  // isn't a shape this solver understands.
  let mut distinct_points: Vec<f64> = Vec::new();
  for x_val in raw_ics
    .iter()
    .map(|(_, _, x, _)| *x)
    .chain(general_conds.iter().map(|(x, _)| *x))
  {
    if !distinct_points.iter().any(|p| (p - x_val).abs() <= 1e-9) {
      distinct_points.push(x_val);
    }
  }
  #[derive(Clone, Copy)]
  enum ProblemKind {
    Initial { x0: f64 },
    Boundary { x_lo: f64, x_hi: f64 },
  }
  let boundary_shape = eliminated.is_empty()
    && implicit.is_empty()
    && raw_ics.len() + general_conds.len()
      == funcs.iter().map(|f| f.order).sum::<usize>()
    && raw_ics.iter().all(|(name, order, _, _)| {
      funcs.iter().any(|f| &f.name == name && *order < f.order)
    });
  let kind = match distinct_points.len() {
    1 if general_conds.is_empty() => ProblemKind::Initial {
      x0: distinct_points[0],
    },
    // Conditions at several points, or relating several values at one: a
    // boundary value problem, solved by shooting. The points may lie
    // anywhere in the domain, not only at its ends.
    n if n >= 1 && boundary_shape => {
      let Some(x_min_given) = x_min_given else {
        return Ok(None);
      };
      let tol = 1e-9 * (1.0 + x_min_given.abs().max(x_max.abs()));
      if distinct_points
        .iter()
        .any(|p| *p < x_min_given - tol || *p > x_max + tol)
      {
        return Ok(None);
      }
      ProblemKind::Boundary {
        x_lo: x_min_given,
        x_hi: x_max,
      }
    }
    _ => return Ok(None),
  };

  // For an initial value problem: the range that was *asked* for, before
  // the integration range below is widened to reach the initial
  // condition. The solution is reported over this range (see the
  // clipping step after the integration/shooting).
  let mut requested_range: Option<(f64, f64)> = None;
  let mut x_min = 0.0;
  let mut x_max_resolved = 0.0;
  let mut x0 = 0.0;
  if let ProblemKind::Initial { x0: x0_val } = kind {
    x0 = x0_val;
    requested_range = x_min_given.map(|min| (min, x_max));
    let (lo, hi) = if let Some(min) = x_min_given {
      // The initial condition may sit just outside the requested output
      // range — a Demonstration commonly states `y[0] == n0` at the
      // natural reference point but plots from a tiny epsilon
      // (`{t, 0.00001, 200}`) to dodge a singularity (a fractional power
      // of `y`, a `LogPlot`, …) exactly at that point. The integrator
      // still has to start at the initial condition, so the solved range
      // extends to include it rather than rejecting the system outright —
      // matching the `{x, xmax}` shorthand below, which already
      // integrates from x0 unconditionally.
      (min.min(x0), x_max.max(x0))
    } else {
      // {x, xmax} shorthand: integrate from x0 to xmax, in whichever
      // direction that is — x0 always lands on one edge of the range. An
      // absolute epsilon here would wrongly reject ranges that are tiny
      // by scale (e.g. femtosecond time constants) rather than
      // degenerate, so only bit-identical endpoints count as degenerate.
      let target = x_max;
      if target == x0 {
        return Ok(None);
      }
      (x0.min(target), x0.max(target))
    };
    x_min = lo;
    x_max_resolved = hi;
  } else if let ProblemKind::Boundary { x_lo, x_hi, .. } = kind {
    requested_range = Some((x_lo, x_hi));
  }

  // Validate and store the initial conditions (initial value problems
  // only — a boundary value problem's two conditions are consumed
  // directly by the shooting solver below instead of seeding `f.ics`).
  for f in &mut funcs {
    f.ics = vec![None; f.order];
  }
  if matches!(kind, ProblemKind::Initial { .. }) {
    for (name, order, _, val) in &raw_ics {
      // An initial condition for a function the constraint eliminated is
      // redundant — its value follows from the others — so it is dropped.
      let Some(f) = funcs.iter_mut().find(|f| &f.name == name) else {
        continue;
      };
      // A value given for an algebraic unknown (`L[0] == 5` for an `L[t]`
      // no equation differentiates) is fixed by the constraints anyway;
      // Wolfram accepts it as the consistent starting value.
      if f.order == 0 && *order == 0 {
        continue;
      }
      if *order >= f.order || f.ics[*order].is_some() {
        return Ok(None);
      }
      f.ics[*order] = Some(*val);
    }
    if funcs
      .iter()
      .any(|f| f.ics.iter().any(std::option::Option::is_none))
    {
      return Ok(None);
    }
  }

  // Variable vector layout: [x, state…, h…] where `state` holds
  // (f, f', …, f^(order-1)) for each function in turn and `h` holds the
  // highest derivative of each function.
  let mut var_names: Vec<String> = vec![x_name.clone()];
  let mut state_offset: Vec<usize> = Vec::with_capacity(funcs.len());
  for f in &funcs {
    state_offset.push(var_names.len() - 1);
    for k in 0..f.order {
      var_names.push(placeholder_name(&f.name, k));
    }
  }
  let n_state = var_names.len() - 1;
  for f in &funcs {
    var_names.push(placeholder_name(&f.name, f.order));
  }

  // Rewrite the equations over placeholder variables and wrap them into
  // numeric residual functions.
  let mut residuals: Vec<NumFn> = Vec::with_capacity(odes.len());
  let delay_found = std::cell::Cell::new(false);
  // The numerically-eliminated unknowns live in extra slots past the
  // placeholders; each residual solves their constraints for those first.
  let implicit_slot = var_names.len();
  let mut ext_names = var_names.clone();
  let mut implicit_guess: Vec<f64> = Vec::new();
  for (zf, _) in &implicit {
    ext_names.push(placeholder_name(&zf.name, 0));
    implicit_guess.push(
      raw_ics
        .iter()
        .find(|(n, o, _, _)| n == &zf.name && *o == 0)
        .map_or(1.0, |(_, _, _, v)| *v),
    );
  }
  // Every function the residuals read, the numerically-eliminated unknowns
  // (order 0, in their extra slots) first.
  let implicit_funcs: Vec<SysFunc> = implicit
    .iter()
    .map(|(f, _)| f)
    .chain(funcs.iter())
    .map(|f| SysFunc {
      name: f.name.clone(),
      order: f.order,
      function_form: f.function_form,
      ics: Vec::new(),
    })
    .collect();
  // The compiled constraints of the numerically-eliminated unknowns.
  let implicit_memo = std::rc::Rc::new(std::cell::RefCell::new(None));
  let compile_implicit = || -> Option<Vec<NExpr>> {
    implicit
      .iter()
      .map(|(_, eq)| {
        let norm = normalize_equation(eq).ok()?;
        let r = substitute_function_values(&norm, &implicit_funcs, x_name);
        compile_numeric(&r, &ext_names)
      })
      .collect()
  };
  if !implicit.is_empty() && compile_implicit().is_none() {
    return Ok(None);
  }
  for ode in &odes {
    let normalized = normalize_equation(ode).ok();
    let Some(normalized) = normalized else {
      return Ok(None);
    };
    let normalized = replace_delayed_values(
      &normalized,
      &funcs,
      &state_offset,
      x_name,
      &delay_found,
    );
    if !implicit.is_empty() {
      let rewritten =
        substitute_function_values(&normalized, &implicit_funcs, x_name);
      let (Some(body), Some(constraints)) =
        (compile_numeric(&rewritten, &ext_names), compile_implicit())
      else {
        return Ok(None);
      };
      residuals.push(NumFn::Compiled(NExpr::Implicit {
        constraints,
        slot: implicit_slot,
        guess: std::cell::RefCell::new(implicit_guess.clone()),
        init: implicit_guess.clone(),
        memo: implicit_memo.clone(),
        key_len: 1 + n_state,
        body: Box::new(body),
      }));
      continue;
    }
    let rewritten = substitute_function_values(&normalized, &funcs, x_name);
    residuals.push(NumFn::new(rewritten, &var_names));
  }
  // A delay equation reads earlier values of the solution: before the
  // initial point those are the (constant) history conditions, afterwards
  // the steps the integrator has accepted so far.
  let _dde_guard = DdeGuard;
  if delay_found.get() {
    if !matches!(kind, ProblemKind::Initial { .. }) || !time_events_empty {
      return Ok(None);
    }
    let mut past = vec![0.0; var_names.len()];
    for (fi, f) in funcs.iter().enumerate() {
      if f.order > 0 {
        past[1 + state_offset[fi]] = f.ics[0].unwrap_or(0.0);
      }
    }
    DDE_HISTORY.with(|h| {
      *h.borrow_mut() = Some(DdeHistory {
        x0,
        past,
        points: Vec::new(),
      });
    });
  }
  let event_fn = match event {
    Some(spec) => {
      let rewritten = substitute_function_values(&spec.event, &funcs, x_name);
      Some(NumFn::new(rewritten, &var_names))
    }
    None => None,
  };

  let time_events: Vec<TimeEvent> = raw_time_events
    .into_iter()
    .map(|ev| TimeEvent {
      period: ev.period,
      offset: ev.offset,
      action: substitute_function_values(&ev.action, &funcs, x_name),
      state_names: var_names[1..=n_state].to_vec(),
      x_name: x_name.clone(),
    })
    .collect();

  // An event on a highest derivative (`WhenEvent[x'[t] == 0, …]` for a
  // first-order system) reads it from its equation, `x'[t] == rhs`: it is no
  // state component the integrator tracks.
  let mut highest: Vec<(String, usize, Expr)> = Vec::new();
  for f in &funcs {
    if f.order == 0 {
      continue;
    }
    for ode in &odes {
      if let Some((a, b)) = as_equal_pair(ode) {
        for (lhs, rhs) in [(a, b), (b, a)] {
          if let Some((k, Expr::Identifier(ref p))) =
            extract_derivative_order_and_point(lhs, &f.name)
            && k == f.order
            && p == x_name
            && max_derivative_order(rhs, &f.name) < f.order
          {
            highest.push((f.name.clone(), k, rhs.clone()));
          }
        }
      }
    }
    // An equation that holds the highest derivative only implicitly
    // (`5 g Sin[θ[t]] + 7 r θ''[t] == 0`) is solved for it — only when an
    // event asks for that derivative, since solving a large equation is
    // costly.
    fn find_node(e: &Expr, name: &str, order: usize, x: &str) -> Option<Expr> {
      if let Some((k, Expr::Identifier(ref p))) =
        extract_derivative_order_and_point(e, name)
        && k == order
        && p == x
      {
        return Some(e.clone());
      }
      expr_children(e)
        .into_iter()
        .find_map(|c| find_node(c, name, order, x))
    }
    if !highest.iter().any(|(name, _, _)| *name == f.name)
      && raw_state_events
        .iter()
        .any(|ev| find_node(&ev.condition, &f.name, f.order, x_name).is_some())
    {
      for ode in &odes {
        let Some(node) = find_node(ode, &f.name, f.order, x_name) else {
          continue;
        };
        let solved = crate::functions::polynomial_ast::solve_ast(&[
          ode.clone(),
          node.clone(),
        ]);
        if let Ok(Expr::List(ref sols)) = solved
          && let [Expr::List(rules)] = sols.as_slice()
          && let [Expr::Rule { replacement, .. }] = rules.as_slice()
          && max_derivative_order(replacement, &f.name) < f.order
        {
          highest.push((f.name.clone(), f.order, (**replacement).clone()));
          break;
        }
      }
    }
  }
  let explicit_highest = |e: &Expr| -> Expr {
    crate::functions::string_ast::map_expr_tree(e, &|node: &Expr| {
      highest.iter().find_map(|(name, k, rhs)| {
        match extract_derivative_order_and_point(node, name) {
          Some((order, Expr::Identifier(ref p)))
            if order == *k && p == x_name =>
          {
            Some(rhs.clone())
          }
          _ => None,
        }
      })
    })
  };
  let state_events: Vec<StateEvent> = raw_state_events
    .into_iter()
    .map(|raw| {
      let raw = RawStateEvent {
        condition: explicit_highest(&raw.condition),
        action: raw.action,
      };
      StateEvent::new(raw, &funcs, x_name, &var_names)
    })
    .collect();
  let event_ctx = StateEventCtx {
    events: &state_events,
    var_names: &var_names,
  };

  // 1000 nominal steps, or more when the caller asked for a step size
  // smaller than that grid's.
  let n_steps = match STEP_HINT.with(std::cell::Cell::get) {
    Some(h) => {
      let span = (x_max_resolved - x_min).abs();
      let wanted = if span.is_finite() {
        (span / h).ceil()
      } else {
        0.0
      };
      (wanted.min(2_000_000.0) as usize).max(1000)
    }
    None => 1000usize,
  };
  if algebraic_unknowns {
    crate::emit_message(
      "NDSolve::pdord: Some of the functions have zero differential order, \
       so the equations will be solved as a system of differential-algebraic \
       equations.",
    );
  }
  let mut points: Vec<(f64, Vec<f64>)> = match kind {
    ProblemKind::Initial { .. } => {
      // Initial state.
      let mut init_state: Vec<f64> = vec![0.0; n_state];
      for (fi, f) in funcs.iter().enumerate() {
        for (k, ic) in f.ics.iter().enumerate() {
          init_state[state_offset[fi] + k] = ic.unwrap_or(0.0);
        }
      }

      // A zero-length domain `{t, t0, t0}` has nothing to integrate: the
      // solution is just the initial state at the single point `t0`.
      if x_max_resolved == x_min {
        vec![(x0, init_state.clone()), (x0, init_state)]
      } else {
        // When the mass matrix (the highest-derivative coefficients) turns
        // out not to depend on `x` or the state — common for a
        // holdup/mass-balance system, whose coefficients are plain
        // numerals — building it once here instead of at every one of the
        // run's stage evaluations is a large constant-factor speedup with
        // no change to the result: the linear solve at each stage is
        // unaffected, only where its matrix comes from.
        let cached_mass_matrix = probe_constant_mass_matrix(
          &residuals,
          funcs.len(),
          x0,
          &init_state,
          x_min,
          x_max_resolved,
        );
        // Integrate (if x0 is interior) backward from x0 to x_min, then
        // forward from x0 to x_max, locating events on both legs in that
        // order — wolframscript sows `{-Pi, -2 Pi, Pi, 2 Pi}` for the turning
        // points of a cosine on {s, -7, 7}.
        //
        // An infinite `x_max_resolved` (`{t, t0, Infinity}`) only reaches
        // here with an event present (the domain parsing above rejects it
        // otherwise), so there's always something to stop on; a fixed step
        // size sized to the whole domain, the way the finite case picks
        // one, has no domain length to size itself from, so it's searched
        // for instead.
        let backward = if x0 - x_min > 1e-12 {
          // An infinite `x_max_resolved` leaves no whole-domain length to
          // size a step from (unlike the finite case, which reuses the
          // forward leg's own step); the backward leg's own span, from
          // `x0` down to `x_min`, is always finite, so it's sized off that
          // instead.
          let h_back = if x_max_resolved.is_finite() {
            (x_max_resolved - x_min) / n_steps as f64
          } else {
            (x0 - x_min) / n_steps as f64
          };
          let leg = integrate_leg(
            &residuals,
            &funcs,
            &state_offset,
            init_state.clone(),
            x0,
            x_min,
            -h_back,
            event_fn.as_ref(),
            event.and_then(|e| e.action.as_ref()),
            &time_events,
            event_ctx,
            x_name,
            cached_mass_matrix.as_ref(),
          )?;
          let Some(leg) = leg else {
            return Ok(None);
          };
          leg
        } else {
          Vec::new()
        };

        let forward = if x_max_resolved.is_finite() {
          let h = (x_max_resolved - x_min) / n_steps as f64;
          integrate_leg(
            &residuals,
            &funcs,
            &state_offset,
            init_state.clone(),
            x0,
            x_max_resolved,
            h,
            event_fn.as_ref(),
            event.and_then(|e| e.action.as_ref()),
            &time_events,
            event_ctx,
            x_name,
            cached_mass_matrix.as_ref(),
          )?
        } else {
          integrate_leg_until_event(
            &residuals,
            &funcs,
            &state_offset,
            &init_state,
            x0,
            event_fn.as_ref(),
            event.and_then(|e| e.action.as_ref()),
            event_ctx,
            x_name,
            cached_mass_matrix.as_ref(),
          )?
        };
        let Some(forward) = forward else {
          return Ok(None);
        };
        // Combined, ascending in x. The backward leg is (x0, x0-h, …);
        // reverse it and drop its first point (x0, present in the forward
        // leg too).
        let mut points: Vec<(f64, Vec<f64>)> = backward;
        points.reverse();
        points.pop();
        points.extend(forward);
        points
      }
    }
    ProblemKind::Boundary { x_lo, x_hi } => {
      // Each simple condition fixes one state component at its point; the
      // general ones are residual equations over the state there.
      let state_index = |name: &str, order: usize| {
        funcs
          .iter()
          .position(|f| f.name == name)
          .map(|fi| state_offset[fi] + order)
      };
      let mut conds: Vec<(f64, BvpCondition)> = Vec::new();
      for (name, order, x_val, val) in &raw_ics {
        let Some(idx) = state_index(name, *order) else {
          return Ok(None);
        };
        conds.push((*x_val, BvpCondition::Fixed(idx, *val)));
      }
      let placeholders: Vec<(String, usize)> = funcs
        .iter()
        .enumerate()
        .flat_map(|(fi, f)| {
          let offset = state_offset[fi];
          (0..f.order).map(move |k| (placeholder_name(&f.name, k), offset + k))
        })
        .collect();
      for (x_val, residual) in &general_conds {
        conds.push((*x_val, BvpCondition::Residual(residual.clone())));
      }
      let Some(points) = solve_bvp_shooting(
        &residuals,
        &funcs,
        &state_offset,
        n_state,
        (x_lo, x_hi),
        &conds,
        &placeholders,
        n_steps,
        x_name,
      )?
      else {
        return Ok(None);
      };
      points
    }
  };
  if points.len() < 2 {
    return Ok(None);
  }
  // Integrating had to start at the initial condition even when that sits
  // outside the requested range, but the solution NDSolve hands back only
  // covers the range that was asked for:
  // `NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 50, 200}]` reports a
  // domain of `{{50., 200.}}` and warns (`ifun::dmval`) about a query at
  // t == 0, rather than quietly carrying the run-up along.
  //
  // Only ever *narrows*: an integration that stalled before the requested
  // endpoint (a singularity, a step size driven to zero) covers less than
  // was asked for, and that truncated domain is the answer.
  if let Some((req_lo, req_hi)) = requested_range {
    let lo = req_lo.max(points.first().unwrap().0);
    let hi = req_hi.min(points.last().unwrap().0);
    if hi > lo {
      points = clip_points_to_range(&points, lo, hi);
    }
    if points.len() < 2 {
      return Ok(None);
    }
  }
  let x_lo = points.first().unwrap().0;
  let x_hi = points.last().unwrap().0;

  // Build one InterpolatingFunction rule per dependent function.
  let mut rules: Vec<Expr> = Vec::with_capacity(funcs.len() + eliminated.len());
  for (fi, f) in funcs.iter().enumerate() {
    if hidden.contains(&f.name) {
      continue;
    }
    let domain = Expr::List(
      vec![Expr::List(vec![Expr::Real(x_lo), Expr::Real(x_hi)].into())].into(),
    );
    // An algebraic unknown is read from the highest-derivative solve at
    // each point; every other function from its state.
    let value_at = |x: f64, s: &[f64]| -> f64 {
      if f.order > 0 {
        return s[state_offset[fi]];
      }
      solve_highest_derivatives_cached(
        &residuals,
        funcs.len(),
        x,
        &s[..n_state],
        None,
      )
      .map_or(f64::NAN, |h| h[fi])
    };
    let data = Expr::List(
      points
        .iter()
        .map(|(x, s)| {
          Expr::List(vec![Expr::Real(*x), Expr::Real(value_at(*x, s))].into())
        })
        .collect(),
    );
    let interp = call(
      "InterpolatingFunction",
      vec![domain, data, Expr::Integer(NDSOLVE_INTERPOLATION_ORDER)],
    );
    rules.push(if f.function_form {
      rule_expr(Expr::Identifier(f.name.clone()), interp)
    } else {
      rule_expr(
        Expr::FunctionCall {
          name: f.name.clone(),
          args: vec![Expr::Identifier(x_name.clone())].into(),
        },
        Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        },
      )
    });
  }

  // Extra rules for `Derivative[k][f]` entries requested alongside `f`:
  // the same points, just read from the state slot the derivative already
  // occupies (`f`'s state runs f, f', …, f^(order-1) from `state_offset`).
  for (name, order, function_form) in &deriv_requests {
    let fi = funcs.iter().position(|f| &f.name == name).unwrap();
    let domain = Expr::List(
      vec![Expr::List(vec![Expr::Real(x_lo), Expr::Real(x_hi)].into())].into(),
    );
    let data = Expr::List(
      points
        .iter()
        .map(|(x, s)| {
          Expr::List(
            vec![Expr::Real(*x), Expr::Real(s[state_offset[fi] + order])]
              .into(),
          )
        })
        .collect(),
    );
    let interp = call(
      "InterpolatingFunction",
      vec![domain, data, Expr::Integer(NDSOLVE_INTERPOLATION_ORDER)],
    );
    // Mirrors the parser's own shape for `Derivative[k][f]` (nested
    // `CurriedCall`s) so `/.` matches the same expression the user wrote.
    let deriv_pattern = Expr::CurriedCall {
      func: Box::new(call("Derivative", vec![Expr::Integer(*order as i128)])),
      args: vec![Expr::Identifier(name.clone())],
    };
    rules.push(if *function_form {
      rule_expr(deriv_pattern, interp)
    } else {
      rule_expr(
        Expr::CurriedCall {
          func: Box::new(deriv_pattern),
          args: vec![Expr::Identifier(x_name.clone())],
        },
        Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        },
      )
    });
  }

  // An eliminated function is sampled from its constraint solution at the
  // same grid, so it comes back as an InterpolatingFunction like the rest.
  for (f, solved) in &eliminated {
    // The solution may be written with the numerically-eliminated unknowns,
    // which are read from the extra slots past the placeholders.
    let rewritten = substitute_function_values(solved, &implicit_funcs, x_name);
    let value_fn = NumFn::new(rewritten, &ext_names);
    let mut vars = vec![0.0; ext_names.len()];
    // The solve for the highest derivatives probes every residual once per
    // unknown, so it is skipped at each grid point unless the solution
    // actually reads one of those slots.
    let reads_highest_derivatives = value_fn
      .vars_from(1 + n_state)
      .is_none_or(|used| used.iter().any(|&slot| slot < implicit_slot));
    let root_fn = (!implicit.is_empty()).then(|| {
      compile_implicit().map(|constraints| NExpr::Implicit {
        constraints,
        slot: implicit_slot,
        guess: std::cell::RefCell::new(implicit_guess.clone()),
        init: implicit_guess.clone(),
        memo: implicit_memo.clone(),
        key_len: 1 + n_state,
        body: Box::new(NExpr::Const(0.0)),
      })
    });
    let data = Expr::List(
      points
        .iter()
        .map(|(x, state)| {
          vars[0] = *x;
          vars[1..=n_state].copy_from_slice(&state[..n_state]);
          if let Some(Some(NExpr::Implicit {
            constraints,
            guess,
            init,
            ..
          })) = &root_fn
          {
            let start = guess.borrow().clone();
            if let Some(root) =
              newton_system(constraints, &mut vars, implicit_slot, &start)
                .or_else(|| {
                  newton_system(constraints, &mut vars, implicit_slot, init)
                })
            {
              vars[implicit_slot..implicit_slot + root.len()]
                .copy_from_slice(&root);
              *guess.borrow_mut() = root;
            }
          }
          // A solution written with a highest derivative (`F0 == … - h'`)
          // reads it from the solve for the highest derivatives.
          if reads_highest_derivatives
            && let Some(high) = solve_highest_derivatives_cached(
              &residuals,
              funcs.len(),
              *x,
              &state[..n_state],
              None,
            )
          {
            vars[1 + n_state..1 + n_state + high.len()].copy_from_slice(&high);
          }
          Expr::List(
            vec![
              Expr::Real(*x),
              Expr::Real(value_fn.eval(&vars).unwrap_or(f64::NAN)),
            ]
            .into(),
          )
        })
        .collect(),
    );
    let domain = Expr::List(
      vec![Expr::List(vec![Expr::Real(x_lo), Expr::Real(x_hi)].into())].into(),
    );
    let interp = call(
      "InterpolatingFunction",
      vec![domain, data, Expr::Integer(NDSOLVE_INTERPOLATION_ORDER)],
    );
    rules.push(if f.function_form {
      rule_expr(Expr::Identifier(f.name.clone()), interp)
    } else {
      rule_expr(
        Expr::FunctionCall {
          name: f.name.clone(),
          args: vec![Expr::Identifier(x_name.clone())].into(),
        },
        Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        },
      )
    });
  }
  // The numerically-eliminated unknowns are recovered at the same grid.
  for (k, (zf, _)) in implicit.iter().enumerate() {
    let Some(constraints) = compile_implicit() else {
      return Ok(None);
    };
    let value_fn = NExpr::Implicit {
      constraints,
      slot: implicit_slot,
      guess: std::cell::RefCell::new(implicit_guess.clone()),
      init: implicit_guess.clone(),
      memo: implicit_memo.clone(),
      key_len: 1 + n_state,
      body: Box::new(NExpr::Var(implicit_slot + k)),
    };
    let mut vars = vec![0.0; var_names.len()];
    let data = Expr::List(
      points
        .iter()
        .map(|(x, state)| {
          vars[0] = *x;
          vars[1..=n_state].copy_from_slice(&state[..n_state]);
          Expr::List(
            vec![Expr::Real(*x), Expr::Real(value_fn.eval(&vars))].into(),
          )
        })
        .collect(),
    );
    let domain = Expr::List(
      vec![Expr::List(vec![Expr::Real(x_lo), Expr::Real(x_hi)].into())].into(),
    );
    let interp = call(
      "InterpolatingFunction",
      vec![domain, data, Expr::Integer(NDSOLVE_INTERPOLATION_ORDER)],
    );
    rules.push(if zf.function_form {
      rule_expr(Expr::Identifier(zf.name.clone()), interp)
    } else {
      rule_expr(
        Expr::FunctionCall {
          name: zf.name.clone(),
          args: vec![Expr::Identifier(x_name.clone())].into(),
        },
        Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        },
      )
    });
  }
  // Return the rules in the order the functions were asked for: an
  // eliminated one was appended last, but `NDSolveValue` hands back a list
  // of values positionally.
  let requested: Vec<String> = match &args[1] {
    Expr::List(items) => items.iter().filter_map(dep_name).collect(),
    other => dep_name(other).into_iter().collect(),
  };
  let mut ordered: Vec<Expr> = Vec::with_capacity(rules.len());
  for name in &requested {
    if let Some(pos) = rules
      .iter()
      .position(|r| rule_target_name(r) == Some(name.as_str()))
    {
      ordered.push(rules.remove(pos));
    }
  }
  ordered.append(&mut rules);
  Ok(Some(Expr::List(vec![Expr::List(ordered.into())].into())))
}

/// How many times [`integrate_leg_until_event`] doubles its search window
/// before giving up. `2^40` units past `x0` is far past any plausible
/// event time for a problem whose own scale is of order 1 — the common
/// case, since a Demonstration's controls are themselves usually of that
/// order — while still costing only `MAX_UNBOUNDED_DOUBLINGS * 1000` RK4
/// evaluations in the worst case.
const MAX_UNBOUNDED_DOUBLINGS: u32 = 40;

/// Forward-integrate from `x0` with no fixed upper bound, for
/// `NDSolve[…, {t, t0, Infinity}, Method -> {"EventLocator", …}]`. There's
/// no domain length to size a step from, so the unknown span is searched
/// geometrically — windows of 1, 2, 4, … units past `x0`, each with its
/// own 1000-step grid — until `event_fn` fires inside one of them. Gives
/// up after [`MAX_UNBOUNDED_DOUBLINGS`] windows and reports the last
/// (event-less) one, the way wolframscript's `NDSolve::mxst` reports a
/// truncated solution after an ordinary step-count overrun.
#[allow(clippy::too_many_arguments)]
fn integrate_leg_until_event(
  residuals: &[NumFn],
  funcs: &[SysFunc],
  state_offset: &[usize],
  init_state: &[f64],
  x0: f64,
  event_fn: Option<&NumFn>,
  event_action: Option<&Expr>,
  state_events: StateEventCtx,
  x_name: &str,
  cached_m: Option<&MassLu>,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  let mut window = 1.0f64;
  for attempt in 0..MAX_UNBOUNDED_DOUBLINGS {
    let x_to = x0 + window;
    let h = window / 1000.0;
    let Some(leg) = integrate_leg(
      residuals,
      funcs,
      state_offset,
      init_state.to_vec(),
      x0,
      x_to,
      h,
      event_fn,
      event_action,
      &[],
      state_events,
      x_name,
      cached_m,
    )?
    else {
      return Ok(None);
    };
    // The leg stops short of `x_to` exactly when the event fired (or a
    // step failed) before the window's end; a leg that reaches it ran
    // clean to the requested window with no event, so the search widens.
    let reached_end =
      leg.last().is_some_and(|(x, _)| (x_to - x).abs() < h * 0.5);
    if !reached_end || attempt + 1 == MAX_UNBOUNDED_DOUBLINGS {
      if reached_end {
        crate::emit_message(
          "NDSolve::mxst: Maximum number of 40000 steps reached before the \
           event was located; returning the solution found so far.",
        );
      }
      return Ok(Some(leg));
    }
    window *= 2.0;
  }
  Ok(None)
}

/// A condition `lhs == rhs` relating values and derivatives of the unknowns
/// at one numeric point (`1 == f[0] - f'[0]`): that point, and `lhs - rhs`
/// with every `f[point]` / `Derivative[k][f][point]` replaced by the
/// placeholder of state component `(f, k)`. `None` when the unknowns are
/// taken at different points or at a non-numeric one.
fn general_condition(
  eq: &Expr,
  funcs: &[SysFunc],
  x_name: &str,
) -> Option<(f64, Expr)> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let point: std::cell::Cell<Option<f64>> = std::cell::Cell::new(None);
  let ok = std::cell::Cell::new(true);
  fn walk(
    e: &Expr,
    funcs: &[SysFunc],
    x_name: &str,
    point: &std::cell::Cell<Option<f64>>,
    ok: &std::cell::Cell<bool>,
  ) -> Expr {
    // `f[p]` or `Derivative[k][f][p]`
    let unknown_at = |name: &str, k: usize, arg: &Expr| -> Option<Expr> {
      // Orders are not known yet here; a derivative beyond the state is
      // left for the solver, which cannot evaluate it and gives up.
      funcs.iter().find(|f| f.name == name)?;
      // The independent variable is never a point, even when an outer
      // `Block` gives it a value: `y'[t] == 1` is an equation.
      if crate::functions::plot::expr_mentions_var(arg, x_name) {
        ok.set(false);
        return None;
      }
      let at = nval_to_f64(arg)?;
      match point.get() {
        Some(p) if (p - at).abs() > 1e-12 => ok.set(false),
        _ => point.set(Some(at)),
      }
      Some(Expr::Identifier(placeholder_name(name, k)))
    };
    if let Expr::FunctionCall { name, args } = e
      && args.len() == 1
      && let Some(r) = unknown_at(name, 0, &args[0])
    {
      return r;
    }
    for f in funcs {
      if let Some((k, at)) = extract_derivative_order_and_point(e, &f.name)
        && let Some(r) = unknown_at(&f.name, k, &at)
      {
        return r;
      }
    }
    crate::syntax::map_children(e, &|c| walk(c, funcs, x_name, point, ok))
  }
  let residual = walk(
    &crate::helpers::minus2(lhs.clone(), rhs.clone()),
    funcs,
    x_name,
    &point,
    &ok,
  );
  let p = point.get()?;
  // Every mention of an unknown must have been replaced
  let leftover = funcs.iter().any(|f| {
    crate::functions::calculus_ast::is_constant_wrt(&residual, &f.name)
      .then_some(())
      .is_none()
  });
  (ok.get() && !leftover).then_some((p, residual))
}

/// One condition of a boundary value problem, at its point.
enum BvpCondition {
  /// State component `index` equals `value` (`f[a] == v`, `f'[b] == v`).
  Fixed(usize, f64),
  /// This residual over the state placeholders is zero
  /// (`1 == f[0] - f'[0]` as `1 - (f[0] - f'[0])`).
  Residual(Expr),
}

/// Solve a boundary value problem on `domain` by shooting. The conditions
/// may sit at any points of the domain. Shooting starts at the point with
/// the most fixed state components: those are set, the others guessed, and
/// the *same* IVP stepper (`integrate_leg`) integrates from there through
/// every other condition point out to both ends of the domain. Newton's
/// method (finite-difference Jacobian) refines the guesses until every
/// remaining condition holds, so linear and nonlinear systems alike are
/// handled without inspecting the equations. Returns the trajectory over
/// the whole domain, ascending in x.
#[allow(clippy::too_many_arguments)]
fn solve_bvp_shooting(
  residuals: &[NumFn],
  funcs: &[SysFunc],
  state_offset: &[usize],
  n_state: usize,
  domain: (f64, f64),
  conds: &[(f64, BvpCondition)],
  placeholders: &[(String, usize)],
  n_steps: usize,
  x_name: &str,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  let same = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + a.abs());
  let fixed_at = |p: f64| {
    conds
      .iter()
      .filter(|(x, c)| same(*x, p) && matches!(c, BvpCondition::Fixed(..)))
      .count()
  };
  // The start point: most fixed components, then the leftmost
  let Some(x0) = conds
    .iter()
    .map(|(x, _)| *x)
    .max_by(|a, b| fixed_at(*a).cmp(&fixed_at(*b)).then(b.total_cmp(a)))
  else {
    return Ok(None);
  };
  let mut probe_state = vec![0.0; n_state];
  let mut is_fixed = vec![false; n_state];
  let mut remaining: Vec<&(f64, BvpCondition)> = Vec::new();
  for cond in conds {
    match cond {
      (x, BvpCondition::Fixed(i, v)) if same(*x, x0) && !is_fixed[*i] => {
        probe_state[*i] = *v;
        is_fixed[*i] = true;
      }
      other => remaining.push(other),
    }
  }
  let unknowns: Vec<usize> = (0..n_state).filter(|i| !is_fixed[*i]).collect();
  if unknowns.len() != remaining.len() {
    return Ok(None);
  }
  let (lo, hi) = domain;
  let h = (hi - lo) / n_steps as f64;
  let cached_mass_matrix = probe_constant_mass_matrix(
    residuals,
    funcs.len(),
    x0,
    &probe_state,
    lo,
    hi,
  );

  // Breakpoints on either side of x0: every condition point, then the end
  let mut ahead: Vec<f64> = conds
    .iter()
    .map(|(x, _)| *x)
    .filter(|x| *x > x0 && !same(*x, x0))
    .collect();
  ahead.push(hi);
  ahead.sort_by(f64::total_cmp);
  ahead.dedup_by(|a, b| same(*a, *b));
  let mut behind: Vec<f64> = conds
    .iter()
    .map(|(x, _)| *x)
    .filter(|x| *x < x0 && !same(*x, x0))
    .collect();
  behind.push(lo);
  behind.sort_by(|a, b| b.total_cmp(a));
  behind.dedup_by(|a, b| same(*a, *b));

  // Integrate from x0 with `init` to both ends; the ascending trajectory.
  let shoot =
    |init: &[f64]| -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
      let legs =
        |targets: &[f64],
         step: f64|
         -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
          let mut path: Vec<(f64, Vec<f64>)> = vec![(x0, init.to_vec())];
          for &target in targets {
            let (from, state) = path.last().unwrap().clone();
            if same(from, target) {
              continue;
            }
            let Some(leg) = integrate_leg(
              residuals,
              funcs,
              state_offset,
              state,
              from,
              target,
              step,
              None,
              None,
              &[],
              StateEventCtx::none(),
              x_name,
              cached_mass_matrix.as_ref(),
            )?
            else {
              return Ok(None);
            };
            let Some((x_end, _)) = leg.last() else {
              return Ok(None);
            };
            if !same(*x_end, target)
              && (x_end - target).abs() > 1e-6 * (1.0 + target.abs())
            {
              return Ok(None);
            }
            path.extend(leg.into_iter().skip(1));
          }
          Ok(Some(path))
        };
      let Some(forward) = legs(&ahead, h)? else {
        return Ok(None);
      };
      let Some(mut backward) = legs(&behind, -h)? else {
        return Ok(None);
      };
      backward.reverse();
      backward.pop();
      backward.extend(forward);
      Ok(Some(backward))
    };
  let state_at = |path: &[(f64, Vec<f64>)], p: f64| -> Option<Vec<f64>> {
    path
      .iter()
      .min_by(|(a, _), (b, _)| (a - p).abs().total_cmp(&(b - p).abs()))
      .map(|(_, s)| s.clone())
  };
  let misses = |path: &[(f64, Vec<f64>)]| -> Option<Vec<f64>> {
    remaining
      .iter()
      .map(|(p, cond)| {
        let state = state_at(path, *p)?;
        match cond {
          BvpCondition::Fixed(i, v) => Some(state[*i] - v),
          BvpCondition::Residual(r) => {
            // Substituted directly: a placeholder name starting with `__`
            // would read as a sequence blank in a replacement rule.
            let mut filled = r.clone();
            for (name, i) in placeholders {
              filled = crate::syntax::substitute_variable(
                &filled,
                name,
                &Expr::Real(state[*i]),
              );
            }
            let value =
              crate::evaluator::evaluate_expr_to_expr(&filled).ok()?;
            nval_to_f64(&value)
          }
        }
      })
      .collect()
  };
  let with_guess = |guess: &[f64]| {
    let mut init = probe_state.clone();
    for (i, g) in unknowns.iter().zip(guess) {
      init[*i] = *g;
    }
    init
  };

  let k = unknowns.len();
  let mut guess = vec![0.0; k];
  for _ in 0..60 {
    let Some(path) = shoot(&with_guess(&guess))? else {
      return Ok(None);
    };
    let Some(miss) = misses(&path) else {
      return Ok(None);
    };
    let scale = 1.0
      + path
        .iter()
        .flat_map(|(_, s)| s.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()).min(1e12));
    if miss.iter().all(|m| m.abs() <= 1e-9 * scale) || k == 0 {
      return Ok(Some(path));
    }
    // Jacobian of the misses w.r.t. the guesses, by forward differences
    let mut jac = vec![vec![0.0; k]; k];
    for j in 0..k {
      let delta = 1e-6 * (1.0 + guess[j].abs());
      let mut bumped = guess.clone();
      bumped[j] += delta;
      let Some(bumped_path) = shoot(&with_guess(&bumped))? else {
        return Ok(None);
      };
      let Some(miss_j) = misses(&bumped_path) else {
        return Ok(None);
      };
      for i in 0..k {
        jac[i][j] = (miss_j[i] - miss[i]) / delta;
      }
    }
    let Some(step) = solve_dense_linear(jac, miss) else {
      return Ok(None);
    };
    for (g, d) in guess.iter_mut().zip(step) {
      *g -= d;
    }
  }
  Ok(None)
}

/// Solve the dense system `a x = b` by Gaussian elimination with partial
/// pivoting; `None` when `a` is singular.
fn solve_dense_linear(
  mut a: Vec<Vec<f64>>,
  mut b: Vec<f64>,
) -> Option<Vec<f64>> {
  let n = b.len();
  for col in 0..n {
    let pivot = (col..n).max_by(|&i, &j| {
      a[i][col]
        .abs()
        .partial_cmp(&a[j][col].abs())
        .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let magnitude = a[pivot][col].abs();
    if magnitude.is_nan() || magnitude <= 1e-300 {
      return None;
    }
    a.swap(col, pivot);
    b.swap(col, pivot);
    for row in col + 1..n {
      let f = a[row][col] / a[col][col];
      for c in col..n {
        a[row][c] -= f * a[col][c];
      }
      b[row] -= f * b[col];
    }
  }
  let mut x = vec![0.0; n];
  for row in (0..n).rev() {
    let tail: f64 = (row + 1..n).map(|c| a[row][c] * x[c]).sum();
    x[row] = (b[row] - tail) / a[row][row];
  }
  Some(x)
}

/// A `WhenEvent[Mod[x, period, offset] == 0, action]` as parsed from the
/// equation list (action still over the original `f[x]` calls).
struct RawTimeEvent {
  period: f64,
  offset: f64,
  action: Expr,
}

/// A periodic time event ready for integration: `action` is rewritten over
/// the state placeholders, which [`TimeEvent::run`] replaces by the numeric
/// state at the event time before evaluating it.
struct TimeEvent {
  period: f64,
  offset: f64,
  action: Expr,
  state_names: Vec<String>,
  x_name: String,
}

impl TimeEvent {
  /// Calls `f` with every event time `offset + k*period` in `(lo, hi]`.
  fn times_in(&self, lo: f64, hi: f64, f: &mut dyn FnMut(f64)) {
    let slack = 1e-12 * (1.0 + hi.abs());
    let mut k = ((lo - self.offset) / self.period + 1e-12).floor() as i64 + 1;
    loop {
      let t = self.offset + k as f64 * self.period;
      if t > hi + slack {
        break;
      }
      if t > lo + slack {
        f(t);
      }
      k += 1;
    }
  }

  /// Evaluate the action at time `t` with the given state. Returns
  /// `Ok(false)` when the action asked to stop the integration.
  fn run(&self, t: f64, state: &[f64]) -> Result<bool, InterpreterError> {
    let mut action = self.action.clone();
    for (name, v) in self.state_names.iter().zip(state) {
      action =
        crate::syntax::substitute_variable(&action, name, &Expr::Real(*v));
    }
    let action =
      crate::syntax::substitute_variable(&action, &self.x_name, &Expr::Real(t));
    match crate::evaluator::evaluate_expr_to_expr(&action) {
      Ok(_) => Ok(true),
      Err(InterpreterError::ThrowValue(_, tag))
        if matches!(
          tag.as_deref(),
          Some(Expr::String(s)) if s == "StopIntegration"
        ) =>
      {
        Ok(false)
      }
      Err(e) => Err(e),
    }
  }
}

/// A `WhenEvent[condition, action]` on the solution's state, as parsed
/// from the equation list (both still over the original `f[x]` calls).
struct RawStateEvent {
  condition: Expr,
  action: Expr,
}

/// How a state event's condition is watched while integrating.
enum StateTrigger {
  /// `lhs == rhs`: fires whenever `lhs - rhs` changes sign.
  Crossing(NumFn),
  /// Any other condition: fires when it turns from False to True.
  Condition(Expr),
}

/// A state event ready for integration. Its trigger is rewritten over the
/// state placeholders; the action is kept as written, since a rule
/// `f[x] -> value` in it names the state component it resets.
struct StateEvent {
  trigger: StateTrigger,
  /// The other conjuncts of `lhs == rhs && cond`, checked when the
  /// crossing happens (over the state placeholders).
  guard: Option<Expr>,
  action: Expr,
  active: std::cell::Cell<bool>,
}

/// What running a state event's action asks the integrator to do.
struct EventOutcome {
  stop: bool,
  state: Vec<f64>,
}

impl StateEvent {
  fn new(
    raw: RawStateEvent,
    funcs: &[SysFunc],
    x_name: &str,
    var_names: &[String],
  ) -> Self {
    // `lhs == rhs && cond`: a crossing of the equation, taken only where the
    // rest of the conjunction holds (an equation alone is true on a set of
    // measure zero, so as a plain condition it would never fire).
    let conjuncts: Vec<Expr> = match &raw.condition {
      Expr::FunctionCall { name, args } if name == "And" => args.to_vec(),
      Expr::BinaryOp {
        op: BinaryOperator::And,
        left,
        right,
      } => vec![(**left).clone(), (**right).clone()],
      _ => Vec::new(),
    };
    let equations: Vec<usize> = conjuncts
      .iter()
      .enumerate()
      .filter(|(_, c)| as_equal_pair(c).is_some())
      .map(|(i, _)| i)
      .collect();
    let (condition, guard) = if let [ei] = equations.as_slice() {
      let rest: Vec<Expr> = conjuncts
        .iter()
        .enumerate()
        .filter(|(i, _)| i != ei)
        .map(|(_, c)| c.clone())
        .collect();
      let guard = if rest.len() == 1 {
        rest.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "And".to_string(),
          args: rest.into(),
        }
      };
      (
        conjuncts[*ei].clone(),
        Some(substitute_function_values(&guard, funcs, x_name)),
      )
    } else {
      (raw.condition.clone(), None)
    };
    let trigger = match as_equal_pair(&condition) {
      Some((lhs, rhs)) => {
        let diff = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            lhs.clone(),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), rhs.clone()].into(),
            },
          ]
          .into(),
        };
        let rewritten = substitute_function_values(&diff, funcs, x_name);
        StateTrigger::Crossing(NumFn::new(rewritten, var_names))
      }
      None => StateTrigger::Condition(substitute_function_values(
        &condition, funcs, x_name,
      )),
    };
    StateEvent {
      trigger,
      guard,
      action: raw.action,
      active: std::cell::Cell::new(true),
    }
  }

  /// The trigger's value at `(x, state)`: the signed residual of a
  /// crossing, or ±1 for a condition that is True/False. `None` when it
  /// does not evaluate to a number or truth value.
  fn value(
    &self,
    x: f64,
    state: &[f64],
    funcs: &[SysFunc],
    var_names: &[String],
  ) -> Option<f64> {
    match &self.trigger {
      StateTrigger::Crossing(f) => f
        .eval_state(x, state, funcs.len())
        .ok()
        .filter(|g| g.is_finite()),
      StateTrigger::Condition(cond) => {
        let cond = substitute_state(cond, var_names, x, state);
        match &crate::evaluator::evaluate_expr_to_expr(&cond).ok()? {
          Expr::Identifier(b) if b == "True" => Some(1.0),
          Expr::Identifier(b) if b == "False" => Some(-1.0),
          _ => None,
        }
      }
    }
  }

  /// Whether the event's guard (if any) holds at `(x, state)`.
  fn guard_holds(&self, x: f64, state: &[f64], var_names: &[String]) -> bool {
    let Some(guard) = &self.guard else {
      return true;
    };
    let g = substitute_state(guard, var_names, x, state);
    matches!(
      crate::evaluator::evaluate_expr_to_expr(&g),
      Ok(Expr::Identifier(ref b)) if b == "True"
    )
  }

  /// Whether moving from trigger value `prev` to `next` fires the event.
  fn fires(&self, prev: f64, next: f64) -> bool {
    match self.trigger {
      StateTrigger::Crossing(_) => {
        (prev < 0.0 && next >= 0.0) || (prev > 0.0 && next <= 0.0)
      }
      StateTrigger::Condition(_) => prev < 0.0 && next > 0.0,
    }
  }

  /// Run the action at event time `t`. A rule `f[x] -> v` (or
  /// `f'[x] -> v`) resets that state component; any other part is
  /// evaluated for its effect, and one that yields `"StopIntegration"`
  /// ends the integration, `"RemoveEvent"` disarms the event.
  fn run(
    &self,
    t: f64,
    state: &[f64],
    funcs: &[SysFunc],
    state_offset: &[usize],
    var_names: &[String],
    x_name: &str,
  ) -> Result<EventOutcome, InterpreterError> {
    let items: Vec<Expr> = match &self.action {
      Expr::List(items) => items.to_vec(),
      other => vec![other.clone()],
    };
    let mut new_state = state.to_vec();
    let mut stop = false;
    let in_state = |e: &Expr| {
      let rewritten = substitute_function_values(e, funcs, x_name);
      substitute_state(&rewritten, var_names, t, state)
    };
    for item in &items {
      if let Expr::Rule {
        pattern,
        replacement,
      } = item
        && let Some(slot) = state_slot(pattern, funcs, state_offset, x_name)
      {
        let value =
          crate::evaluator::evaluate_expr_to_expr(&in_state(replacement))?;
        if let Some(v) = try_eval_to_f64(&value) {
          new_state[slot] = v;
        }
        continue;
      }
      match crate::evaluator::evaluate_expr_to_expr(&in_state(item)) {
        Ok(value) => match &value {
          Expr::String(s) if s == "StopIntegration" => stop = true,
          Expr::String(s) if s == "RemoveEvent" => self.active.set(false),
          _ => {}
        },
        Err(InterpreterError::ThrowValue(_, tag))
          if matches!(
            tag.as_deref(),
            Some(Expr::String(s)) if s == "StopIntegration"
          ) =>
        {
          stop = true;
        }
        Err(e) => return Err(e),
      }
    }
    Ok(EventOutcome {
      stop,
      state: new_state,
    })
  }
}

/// The state events of one solve, with the variable names their triggers
/// are written over.
#[derive(Clone, Copy)]
struct StateEventCtx<'a> {
  events: &'a [StateEvent],
  var_names: &'a [String],
}

impl StateEventCtx<'_> {
  fn none() -> StateEventCtx<'static> {
    StateEventCtx {
      events: &[],
      var_names: &[],
    }
  }
}

/// Replace the independent variable and the state placeholders in `expr`
/// by their numeric values.
fn substitute_state(
  expr: &Expr,
  var_names: &[String],
  x: f64,
  state: &[f64],
) -> Expr {
  let mut out =
    crate::syntax::substitute_variable(expr, &var_names[0], &Expr::Real(x));
  for (name, v) in var_names[1..].iter().zip(state) {
    out = crate::syntax::substitute_variable(&out, name, &Expr::Real(*v));
  }
  out
}

/// The state-vector index that `f[x]` or `Derivative[k][f][x]` names, when
/// `f` is one of the system's functions and `k` is below its order.
fn state_slot(
  expr: &Expr,
  funcs: &[SysFunc],
  state_offset: &[usize],
  x_name: &str,
) -> Option<usize> {
  for (fi, f) in funcs.iter().enumerate() {
    let k = match expr {
      Expr::FunctionCall { name, args }
        if name == &f.name
          && args.len() == 1
          && matches!(&args[0], Expr::Identifier(a) if a == x_name) =>
      {
        0
      }
      _ => match extract_derivative_order_and_point(expr, &f.name) {
        Some((k, Expr::Identifier(ref a))) if a == x_name => k,
        _ => continue,
      },
    };
    if k < f.order.max(1) {
      return Some(state_offset[fi] + k);
    }
  }
  None
}

/// Parse `WhenEvent[Mod[x, p] == 0, action]`.
fn parse_periodic_when_event(
  args: &[Expr],
  x_name: &str,
) -> Option<RawTimeEvent> {
  if args.len() != 2 {
    return None;
  }
  let (lhs, rhs) = as_equal_pair(&args[0])?;
  if !matches!(rhs, Expr::Integer(0)) {
    return None;
  }
  let Expr::FunctionCall { name, args: margs } = lhs else {
    return None;
  };
  if name != "Mod" || margs.len() != 2 {
    return None;
  }
  if !matches!(&margs[0], Expr::Identifier(a) if a == x_name) {
    return None;
  }
  let period = nval_to_f64(&margs[1])?;
  if !period.is_finite() || period <= 0.0 {
    return None;
  }
  let offset = 0.0;
  Some(RawTimeEvent {
    period,
    offset,
    action: args[1].clone(),
  })
}

/// Integrate one leg with RK4. Returns the points in integration order
/// (starting at `x_from`), or `None` if a derivative evaluation failed.
/// State events on a leg integrated with adaptive steps: each crossing is
/// located between two output points by bisecting the straight line
/// between their states, the action runs there, and a reset state (or a
/// stop) restarts (or ends) the integration from the event.
#[allow(clippy::type_complexity)]
fn state_events_on_points(
  mut leg: Vec<(f64, Vec<f64>)>,
  state_events: &StateEventCtx,
  funcs: &[SysFunc],
  state_offset: &[usize],
  x_name: &str,
  x_to: f64,
  integrate: &dyn Fn(
    &[f64],
    f64,
    f64,
  )
    -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError>,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  let value = |ev: &StateEvent, x: f64, s: &[f64]| {
    ev.value(x, s, funcs, state_events.var_names)
  };
  let lerp = |a: &(f64, Vec<f64>), b: &(f64, Vec<f64>), u: f64| {
    (
      a.0 + u * (b.0 - a.0),
      a.1
        .iter()
        .zip(&b.1)
        .map(|(p, q)| p + u * (q - p))
        .collect::<Vec<f64>>(),
    )
  };
  let mut out: Vec<(f64, Vec<f64>)> = vec![leg[0].clone()];
  let mut start = 0usize;
  loop {
    let mut prev: Vec<Option<f64>> = state_events
      .events
      .iter()
      .map(|ev| value(ev, leg[start].0, &leg[start].1))
      .collect();
    let mut hit: Option<(usize, (f64, Vec<f64>), usize)> = None;
    'scan: for i in start..leg.len().saturating_sub(1) {
      for (ei, ev) in state_events.events.iter().enumerate() {
        if !ev.active.get() {
          continue;
        }
        let (Some(g0), Some(g1)) =
          (prev[ei], value(ev, leg[i + 1].0, &leg[i + 1].1))
        else {
          continue;
        };
        if !ev.fires(g0, g1) {
          continue;
        }
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..60 {
          let mid = f64::midpoint(lo, hi);
          let (xm, sm) = lerp(&leg[i], &leg[i + 1], mid);
          match value(ev, xm, &sm) {
            Some(g) if ev.fires(g0, g) => hi = mid,
            _ => lo = mid,
          }
        }
        let at = lerp(&leg[i], &leg[i + 1], hi);
        if ev.guard_holds(at.0, &at.1, state_events.var_names) {
          hit = Some((i, at, ei));
          break 'scan;
        }
      }
      for (ei, ev) in state_events.events.iter().enumerate() {
        prev[ei] = value(ev, leg[i + 1].0, &leg[i + 1].1);
      }
    }
    let Some((i, (te, state_at), ei)) = hit else {
      out.extend(leg[start + 1..].iter().cloned());
      return Ok(Some(out));
    };
    out.extend(leg[start + 1..=i].iter().cloned());
    let outcome = state_events.events[ei].run(
      te,
      &state_at,
      funcs,
      state_offset,
      state_events.var_names,
      x_name,
    )?;
    out.push((te, outcome.state.clone()));
    if outcome.stop {
      return Ok(Some(out));
    }
    if outcome.state == state_at {
      // Carry on along the same points, from the event.
      let mut rest = vec![(te, state_at)];
      rest.extend(leg[i + 1..].iter().cloned());
      leg = rest;
      start = 0;
    } else {
      // A reset: integrate on from the event with the new state.
      match integrate(&outcome.state, te, x_to)? {
        Some(rest) => {
          leg = rest;
          start = 0;
        }
        None => return Ok(Some(out)),
      }
    }
  }
}

/// When an event function is given, integration stops at the located
/// event point after running the (held) event action; an action that
/// throws the `"StopIntegration"` tag simply stops the leg.
#[allow(clippy::too_many_arguments)]
fn integrate_leg(
  residuals: &[NumFn],
  funcs: &[SysFunc],
  state_offset: &[usize],
  init_state: Vec<f64>,
  x_from: f64,
  x_to: f64,
  h: f64,
  event_fn: Option<&NumFn>,
  event_action: Option<&Expr>,
  time_events: &[TimeEvent],
  state_events: StateEventCtx,
  x_name: &str,
  cached_m: Option<&MassLu>,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  // At least one step for a leg of any length: an initial point just inside
  // the domain leaves a backward leg shorter than half a step, which would
  // otherwise round to none and cut the domain short.
  let n_steps = ((x_to - x_from) / h).round() as usize;
  let n_steps = if n_steps == 0 && x_to != x_from {
    1
  } else {
    n_steps
  };
  // Each grid point is computed from the leg's ends rather than by adding
  // `h` to the previous one, and the last one *is* `x_to`: accumulating
  // the step leaves the domain a few ulps short of where it was asked to
  // end (`{{0., 4.999999999999916}}` instead of `{{0., 5.}}`).
  let step = if n_steps > 0 {
    (x_to - x_from) / n_steps as f64
  } else {
    h
  };
  let grid = |i: usize| {
    if i >= n_steps {
      x_to
    } else {
      x_from + i as f64 * step
    }
  };
  let mut points: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n_steps + 1);
  let mut state = init_state;
  let mut x = x_from;
  points.push((x, state.clone()));
  if dde_active() {
    dde_record(&points);
  }

  let eval_event = |x: f64, state: &[f64]| -> Option<f64> {
    event_fn.and_then(|f| f.eval_state(x, state, funcs.len()).ok())
  };
  // An event function that starts at zero (to within roundoff — a bounce
  // restarted on the surface it bounced off) has no side yet, so the
  // first step cannot cross it; its sign is taken after that step.
  let mut prev_event = eval_event(x, &state).filter(|g| g.abs() > 1e-8);
  let state_value = |ev: &StateEvent, x: f64, state: &[f64]| {
    ev.value(x, state, funcs, state_events.var_names)
  };
  let mut prev_state_values: Vec<Option<f64>> = state_events
    .events
    .iter()
    .map(|ev| state_value(ev, x, &state))
    .collect();

  // A stiff system or a transient faster than the grid makes RK4
  // diverge: without event functions or periodic actions to locate, the
  // rest of the leg is integrated again with adaptive steps.
  // A right-hand side that solves a system at every evaluation (the
  // implicit constraints of a DAE) makes an adaptive leg crawling towards a
  // singular point prohibitively slow: such a leg only gets a bounded
  // number of stiff steps, and otherwise stops where RK4 did.
  let costly = residuals.iter().any(NumFn::is_costly);
  let can_go_adaptive = event_fn.is_none()
    && time_events.is_empty()
    && !dde_active()
    && !funcs.iter().any(|f| f.order == 0);
  let rhs = |xv: f64, s: &[f64]| {
    Ok(system_derivative(
      residuals,
      funcs,
      state_offset,
      s,
      xv,
      cached_m,
    ))
  };
  // An adaptive explicit step first (a fast transient the grid missed),
  // handing a stiff system to the stiff method as soon as the explicit
  // steps detect it; when the stiff method fails too (fast spikes rather
  // than stiffness), the explicit steps run to the end after all.
  let adaptive = |start: &[f64],
                  a: f64,
                  b: f64|
   -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
    if costly {
      return stiff::rosenbrock_leg_limited(
        &rhs,
        start,
        a,
        b,
        COSTLY_STIFF_STEPS,
        COSTLY_STIFF_RTOL,
        COSTLY_STIFF_ATOL,
      );
    }
    if let Some(p) = stiff::dormand_prince_leg(&rhs, start, a, b, true)? {
      return Ok(Some(p));
    }
    if let Some(p) = stiff::rosenbrock_leg(&rhs, start, a, b)? {
      return Ok(Some(p));
    }
    stiff::dormand_prince_leg(&rhs, start, a, b, false)
  };
  // With state events the leg goes on adaptively from where the fixed
  // steps stopped (the events found up to there have run their actions),
  // and the events of the rest are located on the adaptive points.
  let adaptive_with_events =
    |points: &mut Vec<(f64, Vec<f64>)>,
     from_state: &[f64],
     from_x: f64|
     -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
      let Some(leg) = adaptive(from_state, from_x, x_to)? else {
        return Ok(None);
      };
      let mut done = std::mem::take(points);
      if let Some(rest) = state_events_on_points(
        leg,
        &state_events,
        funcs,
        state_offset,
        x_name,
        x_to,
        &adaptive,
      )? {
        done.extend(rest.into_iter().skip(1));
      }
      Ok(Some(done))
    };
  for i in 0..n_steps {
    let new_x = grid(i + 1);
    let h = new_x - x;
    let mut refined: Vec<(f64, Vec<f64>)> = Vec::new();
    let Some(mut new_state) = rk4_step_refined(
      residuals,
      funcs,
      state_offset,
      &state,
      x,
      h,
      Refinement::START,
      &mut refined,
      cached_m,
    )?
    else {
      // The step failed even after refinement — typically the trajectory
      // itself diverges to infinity partway through the domain (a shooting
      // guess that isn't yet the true boundary-value solution commonly
      // does this well short of `x_to`). Rather than discarding every
      // point integrated so far, hand back the leg truncated at the last
      // point that was actually reached: the caller already builds the
      // returned `InterpolatingFunction`'s domain from the first/last
      // collected point rather than assuming it spans the requested
      // range, so a truncated leg becomes a truncated (rather than a
      // missing) solution — matching wolframscript, which likewise
      // returns an `InterpolatingFunction` valid up to where it could
      // integrate instead of failing the whole call (alongside the
      // `NDSolve::ndsz` message it prints at the point stepping stalled).
      // A stiff system or a transient faster than the grid makes RK4
      // diverge: without events to locate, the whole leg is integrated
      // again with adaptive steps.
      // State events are found afterwards on the adaptive points.
      if can_go_adaptive {
        if state_events.events.is_empty() {
          if let Some(leg) = adaptive(&points[0].1, x_from, x_to)? {
            return Ok(Some(leg));
          }
        } else if let Some(done) = adaptive_with_events(&mut points, &state, x)?
        {
          return Ok(Some(done));
        }
      }
      if points.len() > 1 {
        // Named by the independent variable (`At t == 1.002`), as Wolfram
        // reports it.
        crate::emit_message(&format!(
          "NDSolve::ndsz: At {x_name} == {}, step size is effectively zero; \
           singularity or stiff system suspected.",
          crate::syntax::expr_to_string(&Expr::Real(x))
        ));
      }
      return Ok(if points.len() > 1 { Some(points) } else { None });
    };

    if !state_events.events.is_empty() {
      // The earliest state event that fires inside (x, new_x], located by
      // bisecting the step: a partial RK4 step from `x` gives the state at
      // any time inside it.
      let mut earliest: Option<(f64, Vec<f64>, usize)> = None;
      for (ei, ev) in state_events.events.iter().enumerate() {
        if !ev.active.get() {
          continue;
        }
        let (Some(g0), Some(g1)) =
          (prev_state_values[ei], state_value(ev, new_x, &new_state))
        else {
          continue;
        };
        if !ev.fires(g0, g1) {
          continue;
        }
        let (mut lo, mut hi) = (0.0_f64, h);
        let mut hi_state = new_state.clone();
        for _ in 0..60 {
          let mid = f64::midpoint(lo, hi);
          if mid == lo || mid == hi {
            break;
          }
          let Some(s_mid) = rk4_system_step(
            residuals,
            funcs,
            state_offset,
            &state,
            x,
            mid,
            cached_m,
          )?
          else {
            break;
          };
          match state_value(ev, x + mid, &s_mid) {
            Some(g) if ev.fires(g0, g) => {
              hi = mid;
              hi_state = s_mid;
            }
            _ => lo = mid,
          }
        }
        if !ev.guard_holds(x + hi, &hi_state, state_events.var_names) {
          continue;
        }
        // "Earliest" along the leg's direction (a backward leg runs down).
        let dir = h.signum();
        if earliest
          .as_ref()
          .is_none_or(|(t, _, _)| dir * (x + hi) < dir * *t)
        {
          earliest = Some((x + hi, hi_state, ei));
        }
      }
      if let Some((te, state_at, ei)) = earliest {
        let dir = h.signum();
        points.extend(refined.drain(..).filter(|(xr, _)| dir * *xr < dir * te));
        let outcome = state_events.events[ei].run(
          te,
          &state_at,
          funcs,
          state_offset,
          state_events.var_names,
          x_name,
        )?;
        points.push((te, outcome.state.clone()));
        if outcome.stop {
          return Ok(Some(points));
        }
        // Carry on from the event with the (possibly reset) state.
        new_state = if dir * (new_x - te) > f64::EPSILON * new_x.abs().max(1.0)
        {
          let mut after: Vec<(f64, Vec<f64>)> = Vec::new();
          match rk4_step_refined(
            residuals,
            funcs,
            state_offset,
            &outcome.state,
            te,
            new_x - te,
            Refinement::START,
            &mut after,
            cached_m,
          )? {
            Some(s) => s,
            None if can_go_adaptive => {
              return match adaptive_with_events(
                &mut points,
                &outcome.state,
                te,
              )? {
                Some(done) => Ok(Some(done)),
                None => Ok(Some(points)),
              };
            }
            None => return Ok(Some(points)),
          }
        } else {
          outcome.state.clone()
        };
        refined.clear();
        refined.push((new_x, new_state.clone()));
        // A condition that is already true after the event must turn
        // false before it can fire again.
        for (ej, ev) in state_events.events.iter().enumerate() {
          prev_state_values[ej] = state_value(ev, te, &outcome.state);
        }
      }
      for (ej, ev) in state_events.events.iter().enumerate() {
        if let Some(g) = state_value(ev, new_x, &new_state) {
          prev_state_values[ej] = Some(g);
        }
      }
    }

    if !time_events.is_empty() {
      // Fire every periodic action whose time falls in (x, new_x], in
      // time order, with the state interpolated to that time.
      let mut due: Vec<(f64, usize)> = Vec::new();
      if new_x >= x {
        for (ei, ev) in time_events.iter().enumerate() {
          ev.times_in(x, new_x, &mut |t| due.push((t, ei)));
        }
        due.sort_by(|a, b| a.0.total_cmp(&b.0));
      } else {
        // A backward leg meets the times in [new_x, x), latest first.
        let slack = 1e-12 * (1.0 + x.abs());
        for (ei, ev) in time_events.iter().enumerate() {
          ev.times_in(new_x - 2.0 * slack, x, &mut |t| {
            if t < x - slack {
              due.push((t, ei));
            }
          });
        }
        due.sort_by(|a, b| b.0.total_cmp(&a.0));
      }
      for (te, ei) in due {
        let state_at = if (new_x - te).abs() <= f64::EPSILON * new_x.abs() {
          new_state.clone()
        } else {
          match rk4_system_step(
            residuals,
            funcs,
            state_offset,
            &state,
            x,
            te - x,
            cached_m,
          )? {
            Some(s) => s,
            None => new_state.clone(),
          }
        };
        if !time_events[ei].run(te, &state_at)? {
          points.push((te, state_at));
          return Ok(Some(points));
        }
      }
    }

    if let Some(prev_g) = prev_event
      && let Some(new_g) = eval_event(new_x, &new_state)
    {
      if prev_g.signum() != new_g.signum() && prev_g.is_finite() {
        // Event crossing inside this step: locate it by linear
        // interpolation of the event function, then take a partial RK4
        // step to land on it.
        let frac = if (prev_g - new_g).abs() > f64::EPSILON {
          (prev_g / (prev_g - new_g)).clamp(0.0, 1.0)
        } else {
          1.0
        };
        let h_star = h * frac;
        let (x_ev, state_ev) = if h_star.abs() > f64::EPSILON {
          match rk4_system_step(
            residuals,
            funcs,
            state_offset,
            &state,
            x,
            h_star,
            cached_m,
          )? {
            Some(s) => (x + h_star, s),
            None => (new_x, new_state.clone()),
          }
        } else {
          (x, state.clone())
        };
        points.push((x_ev, state_ev));
        if let Some(action) = event_action {
          let action = crate::syntax::substitute_variable(
            action,
            x_name,
            &Expr::Real(x_ev),
          );
          match crate::evaluator::evaluate_expr_to_expr(&action) {
            Ok(_) => {}
            Err(InterpreterError::ThrowValue(_, tag))
              if matches!(
                tag.as_deref(),
                Some(Expr::String(s)) if s == "StopIntegration"
              ) => {}
            Err(e) => return Err(e),
          }
        }
        return Ok(Some(points));
      }
      prev_event = Some(new_g);
    } else {
      prev_event = eval_event(new_x, &new_state);
    }

    state = new_state;
    x = new_x;
    if dde_active() {
      dde_record(&refined);
    }
    points.append(&mut refined);
  }
  Ok(Some(points))
}

/// Local-error tolerances for the step-doubling refinement below, applied
/// to the Richardson estimate `|full - two halves| / 15` of the two-half-step
/// result. RK4's local error at the nominal step size is far below them for
/// a smooth problem, so an ordinary integration keeps exactly the grid — and
/// the values — the fixed-step integrator produced, while a close encounter
/// (an eccentric orbit's pericentre) is still subdivided.
const STEP_REFINE_RTOL: f64 = 1e-7;
const STEP_REFINE_ATOL: f64 = 1e-9;

/// How far one nominal step may be bisected. A tracer pulse 10^-6 wide in a
/// domain of length 20 needs ~15 bisections of the 1/1000 nominal step to be
/// seen at all, and another dozen for its edges to stop costing accuracy.
const MAX_STEP_BISECTIONS: u32 = 32;

/// How many successive bisections may fail to shrink the error estimate
/// at RK4's rate before the step is taken as it is.
const MAX_STALLED_BISECTIONS: u32 = 4;

/// Where a step-doubling refinement stands: its bisection depth, the error
/// estimate of the enclosing step (relative to the tolerance) and how many
/// bisections in a row barely reduced it.
#[derive(Clone, Copy)]
struct Refinement {
  depth: u32,
  error: f64,
  stalls: u32,
}

impl Refinement {
  const START: Self = Self {
    depth: 0,
    error: f64::INFINITY,
    stalls: 0,
  };
}

/// Upper bound on the points one leg may accumulate, so a pathological
/// right-hand side cannot make the refinement run away.
const MAX_REFINED_POINTS: usize = 200_000;

/// Take one nominal step, bisecting it when comparing a whole step against
/// two half steps says the step is too coarse. A forcing term that is
/// nonzero only on a very narrow interval — an injected tracer pulse, say —
/// is otherwise integrated as though it lasted the whole step, which
/// inflates the solution by the ratio of the two widths.
///
/// Every point the refinement visits is appended to `out`, so the
/// interpolating solution resolves the feature too. Returns the state at
/// `x + h`.
#[allow(clippy::too_many_arguments)]
fn rk4_step_refined(
  residuals: &[NumFn],
  funcs: &[SysFunc],
  state_offset: &[usize],
  state: &[f64],
  x: f64,
  h: f64,
  level: Refinement,
  out: &mut Vec<(f64, Vec<f64>)>,
  cached_m: Option<&MassLu>,
) -> Result<Option<Vec<f64>>, InterpreterError> {
  let Some(full) =
    rk4_system_step(residuals, funcs, state_offset, state, x, h, cached_m)?
  else {
    return Ok(None);
  };
  let accept = |s: Vec<f64>, out: &mut Vec<(f64, Vec<f64>)>| {
    out.push((x + h, s.clone()));
    Ok(Some(s))
  };
  if level.depth >= MAX_STEP_BISECTIONS || out.len() >= MAX_REFINED_POINTS {
    return accept(full, out);
  }

  // Two half steps. If either fails the whole step stands — the coarse
  // result is still the best available.
  let half = h / 2.0;
  // A half step too small to move `x` cannot be refined any further.
  if x + half == x {
    return accept(full, out);
  }
  let Some(mid) =
    rk4_system_step(residuals, funcs, state_offset, state, x, half, cached_m)?
  else {
    return accept(full, out);
  };
  let Some(two) = rk4_system_step(
    residuals,
    funcs,
    state_offset,
    &mid,
    x + half,
    half,
    cached_m,
  )?
  else {
    return accept(full, out);
  };

  // A non-finite difference means the problem is singular here, not that a
  // smaller step would help, so it counts as converged.
  let error = full
    .iter()
    .zip(&two)
    .map(|(a, b)| {
      let diff = (a - b).abs() / 15.0;
      let scale = STEP_REFINE_ATOL + STEP_REFINE_RTOL * a.abs().max(b.abs());
      if diff.is_finite() { diff / scale } else { 0.0 }
    })
    .fold(0.0_f64, f64::max);
  if error <= 1.0 {
    return accept(full, out);
  }
  // RK4's local error falls 32-fold per halving on a smooth stretch; where
  // it barely falls the right-hand side jumps inside the step (a friction
  // `Sign[v0 - x'[t]]` sticking at v0) and bisecting only piles up points.
  let stalls = if error > level.error / 4.0 {
    level.stalls + 1
  } else {
    0
  };
  if stalls >= MAX_STALLED_BISECTIONS {
    return accept(full, out);
  }
  let next = Refinement {
    depth: level.depth + 1,
    error,
    stalls,
  };

  let Some(mid) = rk4_step_refined(
    residuals,
    funcs,
    state_offset,
    state,
    x,
    half,
    next,
    out,
    cached_m,
  )?
  else {
    return Ok(None);
  };
  rk4_step_refined(
    residuals,
    funcs,
    state_offset,
    &mid,
    x + half,
    half,
    next,
    out,
    cached_m,
  )
}

/// The derivative of the first-order system's state at `x`: each unknown's
/// lower derivatives shift down and its highest one is solved for. `None`
/// when a residual can't be evaluated numerically or the linear system for
/// the highest derivatives is singular.
fn system_derivative(
  residuals: &[NumFn],
  funcs: &[SysFunc],
  state_offset: &[usize],
  state: &[f64],
  x: f64,
  cached_m: Option<&MassLu>,
) -> Option<Vec<f64>> {
  let high = solve_highest_derivatives_cached(
    residuals,
    funcs.len(),
    x,
    state,
    cached_m,
  )?;
  let mut d = vec![0.0; state.len()];
  for (fi, f) in funcs.iter().enumerate() {
    // An algebraic unknown (order 0) carries no state.
    if f.order == 0 {
      continue;
    }
    let off = state_offset[fi];
    for k in 0..f.order - 1 {
      d[off + k] = state[off + k + 1];
    }
    d[off + f.order - 1] = high[fi];
  }
  Some(d)
}

/// One RK4 step of the first-order system equivalent. Returns `None`
/// when a residual can't be evaluated numerically or the linear system
/// for the highest derivatives is singular. `cached_m` is the mass matrix
/// `probe_constant_mass_matrix` proved constant for this system, if any —
/// passing it in skips the `O(n^2)` rebuild at every stage.
/// Stiff steps an adaptive leg with a costly right-hand side may take, and
/// its tolerances: Rodas4's embedded estimate grows with the stiffness, so
/// at the default 1e-8/1e-10 a smooth stiff solution (`x' == -1000 (x -
/// z)`) takes twenty times the steps of wolframscript's grid; these still
/// agree with it to about eight digits.
const COSTLY_STIFF_STEPS: usize = 20_000;
const COSTLY_STIFF_RTOL: f64 = 1e-7;
const COSTLY_STIFF_ATOL: f64 = 1e-8;

fn rk4_system_step(
  residuals: &[NumFn],
  funcs: &[SysFunc],
  state_offset: &[usize],
  state: &[f64],
  x: f64,
  h: f64,
  cached_m: Option<&MassLu>,
) -> Result<Option<Vec<f64>>, InterpreterError> {
  let deriv =
    |s: &[f64], xv: f64| -> Result<Option<Vec<f64>>, InterpreterError> {
      Ok(system_derivative(
        residuals,
        funcs,
        state_offset,
        s,
        xv,
        cached_m,
      ))
    };

  let add_scaled = |s: &[f64], d: &[f64], c: f64| -> Vec<f64> {
    s.iter().zip(d).map(|(a, b)| a + b * c).collect()
  };

  let Some(k1) = deriv(state, x)? else {
    return Ok(None);
  };
  let Some(k2) = deriv(&add_scaled(state, &k1, h / 2.0), x + h / 2.0)? else {
    return Ok(None);
  };
  let Some(k3) = deriv(&add_scaled(state, &k2, h / 2.0), x + h / 2.0)? else {
    return Ok(None);
  };
  let Some(k4) = deriv(&add_scaled(state, &k3, h), x + h)? else {
    return Ok(None);
  };

  Ok(Some(
    state
      .iter()
      .enumerate()
      .map(|(i, s)| s + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
      .collect(),
  ))
}

/// Solve for the highest derivatives at one point. The residuals are
/// affine in the highest-derivative vector `hvec`: `r(hvec) = M·hvec + c`,
/// so `c = r(0)` and the columns of `M` follow from unit vectors; the
/// result solves `M·hvec = -c`.
/// The `c` half of `r(hvec) = M·hvec + c`: `r` evaluated with every
/// highest-derivative placeholder at zero. Cheap (one pass over the
/// residuals) — the expensive part is `mass_matrix`, below.
fn residual_constant_term(
  residuals: &[NumFn],
  n: usize,
  x: f64,
  state: &[f64],
) -> Option<Vec<f64>> {
  let mut vars = Vec::with_capacity(1 + state.len() + n);
  vars.push(x);
  vars.extend_from_slice(state);
  vars.extend(std::iter::repeat_n(0.0, n));

  let mut c = vec![0.0; n];
  for (i, r) in residuals.iter().enumerate() {
    let v = r.eval(&vars).ok()?;
    if !v.is_finite() {
      return None;
    }
    c[i] = v;
  }
  Some(c)
}

/// The `M` half of `r(hvec) = M·hvec + c`, recovered a column at a time via
/// `r(delta·e_j) − r(0) = M·(delta·e_j)` — exact since the residuals are
/// affine in the highest-derivative vector `hvec`. This is the expensive
/// part of solving for the highest derivatives: `n` extra residual
/// evaluations per column, `n` columns. When `M` is independent of `x` and
/// `state` (e.g. the highest-derivative coefficients are plain numerals,
/// as in a constant-holdup mass/energy balance), the caller can compute it
/// once and reuse it — see `probe_constant_mass_matrix`.
fn mass_matrix(
  residuals: &[NumFn],
  n: usize,
  x: f64,
  state: &[f64],
  c: &[f64],
) -> Option<Vec<Vec<f64>>> {
  let mut vars = Vec::with_capacity(1 + state.len() + n);
  vars.push(x);
  vars.extend_from_slice(state);
  vars.extend(std::iter::repeat_n(0.0, n));

  // A unit-sized `delta` is lost entirely to rounding when the other terms
  // in the residual dwarf it — e.g. a coefficient of 1e16 (from squaring a
  // large angular frequency) makes `c[i] + 1.0` round straight back down
  // to `c[i]` in `f64`, which reads as an exactly-zero, falsely singular
  // column. Scaling `delta` up to the residuals' own magnitude keeps the
  // perturbation's contribution above the rounding floor at that scale, so
  // it survives the subtraction.
  let delta = c.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
  let first_slot = 1 + state.len();
  // A residual that never mentions a highest derivative has a zero
  // coefficient for it; skipping those evaluations keeps a sparse system
  // (a chain of coupled stages) at `O(n)` rather than `O(n^2)` passes.
  let used: Vec<Option<Vec<usize>>> =
    residuals.iter().map(|r| r.vars_from(first_slot)).collect();
  let mut m = vec![vec![0.0; n]; n];
  for j in 0..n {
    let slot = first_slot + j;
    vars[slot] = delta;
    for (i, r) in residuals.iter().enumerate() {
      if used[i].as_ref().is_some_and(|u| !u.contains(&slot)) {
        continue;
      }
      let v = r.eval(&vars).ok()?;
      if !v.is_finite() {
        return None;
      }
      m[i][j] = (v - c[i]) / delta;
    }
    vars[slot] = 0.0;
  }
  Some(m)
}

/// Solve `M·hvec = -c` by Gaussian elimination with partial pivoting.
/// `m` is consumed (the elimination overwrites it) since every caller
/// either owns a fresh matrix or is working from a clone of a cached one.
fn solve_mass_matrix_system(
  mut m: Vec<Vec<f64>>,
  c: &[f64],
) -> Option<Vec<f64>> {
  let n = c.len();
  let mut rhs: Vec<f64> = c.iter().map(|v| -v).collect();
  for col in 0..n {
    let (pivot_row, pivot_abs) = (col..n)
      .map(|r| (r, m[r][col].abs()))
      .max_by(|a, b| a.1.total_cmp(&b.1))
      .unwrap();
    if pivot_abs < 1e-14 {
      return None;
    }
    m.swap(col, pivot_row);
    rhs.swap(col, pivot_row);
    for row in col + 1..n {
      let factor = m[row][col] / m[col][col];
      for k in col..n {
        m[row][k] -= factor * m[col][k];
      }
      rhs[row] -= factor * rhs[col];
    }
  }
  let mut sol = vec![0.0; n];
  for row in (0..n).rev() {
    let mut acc = rhs[row];
    for k in row + 1..n {
      acc -= m[row][k] * sol[k];
    }
    sol[row] = acc / m[row][row];
  }
  Some(sol)
}

/// Solve for the highest derivatives at one point, reusing `cached_m` in
/// place of rebuilding the mass matrix when the caller has established
/// (via `probe_constant_mass_matrix`) that it doesn't vary with `x` or
/// `state`. Recomputing `c` (an `O(n)` pass) still happens every call,
/// since that half genuinely does depend on the current point.
fn solve_highest_derivatives_cached(
  residuals: &[NumFn],
  n_funcs: usize,
  x: f64,
  state: &[f64],
  cached_m: Option<&MassLu>,
) -> Option<Vec<f64>> {
  let n = n_funcs;
  let c = residual_constant_term(residuals, n, x, state)?;
  if let Some(lu) = cached_m {
    Some(lu.solve(&c))
  } else {
    let m = mass_matrix(residuals, n, x, state, &c)?;
    solve_mass_matrix_system(m, &c)
  }
}

/// Local-error tolerances for judging two mass-matrix samples "the same
/// point" — loose enough to absorb the finite-difference rounding a
/// different `delta` scale (from a different `c` at each probe point)
/// introduces, tight enough that a genuinely state-dependent mass matrix
/// still fails the check.
const MASS_MATRIX_RTOL: f64 = 1e-8;
const MASS_MATRIX_ATOL: f64 = 1e-10;

fn mass_matrices_close(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
  a.iter().zip(b).all(|(ra, rb)| {
    ra.iter().zip(rb).all(|(x, y)| {
      let diff = (x - y).abs();
      diff.is_finite()
        && diff <= MASS_MATRIX_ATOL + MASS_MATRIX_RTOL * x.abs().max(y.abs())
    })
  })
}

/// Whether the DAE's mass matrix — the highest-derivative coefficients in
/// `r(hvec) = M·hvec + c` — is the same everywhere, so integrating can
/// build it once instead of at every one of the thousands of stage
/// evaluations a run takes. Many physical models are exactly this way: a
/// holdup/mass balance's highest-derivative coefficients are plain
/// numerals (a tank's volume, a stage's molar holdup), not expressions of
/// the state.
///
/// Checked numerically rather than by inspecting the symbolic equations:
/// `M` is recomputed at two points away from `(x0, state0)` and compared.
/// Since the finite-difference recovery of `M` is exact for any point at
/// which the residuals evaluate (affine-in-`hvec` is already required for
/// this solver to apply at all), a truly constant `M` reproduces to near
/// machine precision regardless of where it's sampled — so this can't
/// mistake a slowly-varying `M` for a constant one, only a genuinely
/// constant one for itself. Any evaluation failure (e.g. a probe point
/// landing outside a `Log`/`Sqrt` argument's domain) is treated as "not
/// safely cacheable" and falls back to the exact per-call rebuild.
fn probe_constant_mass_matrix(
  residuals: &[NumFn],
  n: usize,
  x0: f64,
  state0: &[f64],
  x_min: f64,
  x_max: f64,
) -> Option<MassLu> {
  // The probe points are the solver's own, not the user's: a singular
  // right-hand side there (`1/(x - x0)` guarded by `If[t == 0, …]`) must
  // not print messages for evaluations the kernel would never make.
  let snapshot = crate::snapshot_warnings();
  crate::push_quiet();
  let result =
    probe_constant_mass_matrix_inner(residuals, n, x0, state0, x_min, x_max);
  crate::pop_quiet();
  crate::restore_warnings(snapshot);
  result.and_then(MassLu::factor)
}

/// A constant mass matrix factored once: the elimination
/// `solve_mass_matrix_system` performs, with each row's multipliers kept
/// below the diagonal and the pivot row of each column, so every stage
/// solve costs O(n²) instead of O(n³).
pub(crate) struct MassLu {
  lu: Vec<Vec<f64>>,
  pivots: Vec<usize>,
}

impl MassLu {
  fn factor(mut m: Vec<Vec<f64>>) -> Option<Self> {
    let n = m.len();
    let mut pivots = Vec::with_capacity(n);
    for col in 0..n {
      let (pivot_row, pivot_abs) = (col..n)
        .map(|r| (r, m[r][col].abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
      if pivot_abs < 1e-14 {
        return None;
      }
      m.swap(col, pivot_row);
      pivots.push(pivot_row);
      for row in col + 1..n {
        let factor = m[row][col] / m[col][col];
        for k in col + 1..n {
          m[row][k] -= factor * m[col][k];
        }
        m[row][col] = factor;
      }
    }
    Some(Self { lu: m, pivots })
  }

  /// `hvec` with `M·hvec = -c`.
  fn solve(&self, c: &[f64]) -> Vec<f64> {
    let n = c.len();
    let m = &self.lu;
    let mut rhs: Vec<f64> = c.iter().map(|v| -v).collect();
    // The factorization swapped whole rows, multipliers included, so `L`
    // is in the final row order: every interchange applies to the
    // right-hand side before the forward substitution.
    for (col, &pivot) in self.pivots.iter().enumerate() {
      rhs.swap(col, pivot);
    }
    for col in 0..n {
      for row in col + 1..n {
        rhs[row] -= m[row][col] * rhs[col];
      }
    }
    let mut sol = vec![0.0; n];
    for row in (0..n).rev() {
      let mut acc = rhs[row];
      for k in row + 1..n {
        acc -= m[row][k] * sol[k];
      }
      sol[row] = acc / m[row][row];
    }
    sol
  }
}

fn probe_constant_mass_matrix_inner(
  residuals: &[NumFn],
  n: usize,
  x0: f64,
  state0: &[f64],
  x_min: f64,
  x_max: f64,
) -> Option<Vec<Vec<f64>>> {
  let c0 = residual_constant_term(residuals, n, x0, state0)?;
  let m0 = mass_matrix(residuals, n, x0, state0, &c0)?;

  // Probe 1: a different point in the integration domain.
  let span = x_max - x_min;
  let x_probe = if span.is_finite() && span > 0.0 {
    x_min + span * 0.618
  } else {
    x0 + 1.0
  };
  if x_probe != x0 {
    let c1 = residual_constant_term(residuals, n, x_probe, state0)?;
    let m1 = mass_matrix(residuals, n, x_probe, state0, &c1)?;
    if !mass_matrices_close(&m0, &m1) {
      return None;
    }
  }

  // Probe 2: a perturbed state vector (relative offset, plus a small
  // absolute one so an exactly-zero component still moves).
  let state_probe: Vec<f64> = state0
    .iter()
    .map(|v| v * 1.0009 + v.signum() * 1e-6 + 1e-9)
    .collect();
  let c2 = residual_constant_term(residuals, n, x0, &state_probe)?;
  let m2 = mass_matrix(residuals, n, x0, &state_probe, &c2)?;
  if !mass_matrices_close(&m0, &m2) {
    return None;
  }

  Some(m0)
}

/// The synthesized variable name standing for the k-th derivative of `f`
/// in a rewritten residual.
fn placeholder_name(fname: &str, k: usize) -> String {
  format!("__ndsolve${fname}${k}")
}

/// Highest derivative order of `fname` appearing anywhere in `expr`
/// (`fname[x]` counts as order 0 only if a derivative also appears —
/// order is the max over `Derivative[k][fname][…]` occurrences).
/// The highest derivative order of every function differentiated in
/// `expr` (`Derivative[k][f][x]` in its flat and curried forms), in one walk
/// — asking `max_derivative_order` per unknown and equation made the
/// constraint search cubic in the size of a large system.
fn derivative_orders(expr: &Expr) -> std::collections::HashMap<String, usize> {
  fn go(e: &Expr, out: &mut std::collections::HashMap<String, usize>) {
    let mut note = |name: &str, n: &i128| {
      let order = (*n).max(0) as usize;
      let slot = out.entry(name.to_string()).or_insert(0);
      *slot = (*slot).max(order);
    };
    match e {
      Expr::FunctionCall { name, args }
        if name == "Derivative" && args.len() == 3 =>
      {
        if let (Expr::Integer(n), Expr::Identifier(f)) = (&args[0], &args[1]) {
          note(f, n);
        }
      }
      Expr::CurriedCall { func, args } if args.len() == 1 => {
        match func.as_ref() {
          Expr::FunctionCall { name, args: d }
            if name == "Derivative" && d.len() == 2 =>
          {
            if let (Expr::Integer(n), Expr::Identifier(f)) = (&d[0], &d[1]) {
              note(f, n);
            }
          }
          Expr::CurriedCall {
            func: inner,
            args: inner_args,
          } if inner_args.len() == 1 => {
            if let (Expr::FunctionCall { name, args: d }, Expr::Identifier(f)) =
              (inner.as_ref(), &inner_args[0])
              && name == "Derivative"
              && d.len() == 1
              && let Expr::Integer(n) = &d[0]
            {
              note(f, n);
            }
          }
          _ => {}
        }
      }
      _ => {}
    }
    for child in expr_children(e) {
      go(child, out);
    }
  }
  let mut out = std::collections::HashMap::new();
  go(expr, &mut out);
  out
}

fn max_derivative_order(expr: &Expr, fname: &str) -> usize {
  let mut max = 0;
  if let Some((order, _)) = extract_derivative_order_and_point(expr, fname) {
    max = max.max(order);
  }
  for child in expr_children(expr) {
    max = max.max(max_derivative_order(child, fname));
  }
  max
}

/// Replace every `f[x]` and `Derivative[k][f][x]` occurrence with its
/// placeholder identifier, for every function of the system.
fn substitute_function_values(
  expr: &Expr,
  funcs: &[SysFunc],
  x_name: &str,
) -> Expr {
  // Match this node against `f[x]` / `Derivative[k][f][x]`.
  for f in funcs {
    if let Expr::FunctionCall { name, args } = expr
      && name == &f.name
      && args.len() == 1
      && matches!(&args[0], Expr::Identifier(a) if a == x_name)
    {
      return Expr::Identifier(placeholder_name(&f.name, 0));
    }
    if let Some((order, point)) =
      extract_derivative_order_and_point(expr, &f.name)
      && matches!(&point, Expr::Identifier(a) if a == x_name)
    {
      return Expr::Identifier(placeholder_name(&f.name, order));
    }
  }
  map_children(expr, &|child| {
    substitute_function_values(child, funcs, x_name)
  })
}

// ─── Delay differential equations ──────────────────────────────────────

/// Internal head marking a delayed dependent-function value
/// `DdeDelayed$[variableIndex, lag]` between the rewrite and the compiler.
const DELAYED_HEAD: &str = "DdeDelayed$";

/// Solution history of the delay equation being integrated: the constant
/// pre-initial-time values (indexed like the variable vector) plus every
/// accepted step so far.
struct DdeHistory {
  x0: f64,
  past: Vec<f64>,
  points: Vec<(f64, Vec<f64>)>,
}

thread_local! {
  static DDE_HISTORY: std::cell::RefCell<Option<DdeHistory>> =
    const { std::cell::RefCell::new(None) };
}

/// Value of variable `idx` (a slot of the `[x, state…]` vector) at `x`.
/// Before the initial time it is the history function; afterwards it is
/// linearly interpolated between accepted steps, and a lookup beyond the
/// newest step (a lag shorter than the step being taken) uses that step.
fn dde_history_value(idx: usize, x: f64) -> f64 {
  DDE_HISTORY.with(|h| {
    let h = h.borrow();
    let Some(h) = h.as_ref() else {
      return f64::NAN;
    };
    if x <= h.x0 || h.points.is_empty() {
      return h.past.get(idx).copied().unwrap_or(f64::NAN);
    }
    let slot = idx - 1;
    let i = h.points.partition_point(|(px, _)| *px < x);
    if i == 0 {
      return h.points[0].1[slot];
    }
    if i >= h.points.len() {
      return h.points[h.points.len() - 1].1[slot];
    }
    let (xa, ya) = (&h.points[i - 1].0, &h.points[i - 1].1);
    let (xb, yb) = (&h.points[i].0, &h.points[i].1);
    if xb == xa {
      return yb[slot];
    }
    ya[slot] + (yb[slot] - ya[slot]) * (x - xa) / (xb - xa)
  })
}

fn dde_record(points: &[(f64, Vec<f64>)]) {
  DDE_HISTORY.with(|h| {
    if let Some(h) = h.borrow_mut().as_mut() {
      h.points.extend(points.iter().cloned());
    }
  });
}

fn dde_active() -> bool {
  DDE_HISTORY.with(|h| h.borrow().is_some())
}

/// Clears the delay history when the solve ends, however it ends.
struct DdeGuard;
impl Drop for DdeGuard {
  fn drop(&mut self) {
    DDE_HISTORY.with(|h| *h.borrow_mut() = None);
  }
}

/// `f[t /; t <= t0] == c` (or `<`, `t_` pattern spelling) states the
/// constant history of `f` up to `t0`. Rewrite it into the ordinary
/// initial condition `f[t0] == c`, which also seeds the integration.
fn history_condition_to_ic(eq: &Expr, x_name: &str) -> Option<Expr> {
  let Expr::Comparison {
    operands,
    operators,
  } = eq
  else {
    return None;
  };
  if operands.len() != 2 || operators != &vec![ComparisonOp::Equal] {
    return None;
  }
  let Expr::FunctionCall { name, args } = &operands[0] else {
    return None;
  };
  let [
    Expr::FunctionCall {
      name: cname,
      args: cargs,
    },
  ] = args.as_ref()
  else {
    return None;
  };
  if cname != "Condition" || cargs.len() != 2 {
    return None;
  }
  let is_var = |e: &Expr| match e {
    Expr::Identifier(n) => n == x_name,
    Expr::Pattern { name, .. } => name == x_name,
    _ => false,
  };
  if !is_var(&cargs[0]) {
    return None;
  }
  let Expr::Comparison {
    operands: cops,
    operators: cos,
  } = &cargs[1]
  else {
    return None;
  };
  if cops.len() != 2 || cos.len() != 1 {
    return None;
  }
  let bound = match cos[0] {
    ComparisonOp::LessEqual | ComparisonOp::Less if is_var(&cops[0]) => {
      &cops[1]
    }
    ComparisonOp::GreaterEqual | ComparisonOp::Greater if is_var(&cops[1]) => {
      &cops[0]
    }
    _ => return None,
  };
  Some(Expr::Comparison {
    operands: vec![
      Expr::FunctionCall {
        name: name.clone(),
        args: vec![bound.clone()].into(),
      },
      operands[1].clone(),
    ],
    operators: vec![ComparisonOp::Equal],
  })
}

/// Replace every delayed value `f[t - τ]` (τ a positive constant) of a
/// dependent function by `DdeDelayed$[variableIndex, τ]`; returns whether
/// any was found. A zero lag is simply `f[t]`.
fn replace_delayed_values(
  expr: &Expr,
  funcs: &[SysFunc],
  state_offset: &[usize],
  x_name: &str,
  found: &std::cell::Cell<bool>,
) -> Expr {
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && let Some(fi) = funcs.iter().position(|f| &f.name == name)
    && !matches!(&args[0], Expr::Identifier(a) if a == x_name)
  {
    let at = |v: f64| {
      let sub =
        crate::syntax::substitute_variable(&args[0], x_name, &Expr::Real(v));
      crate::evaluator::evaluate_expr_to_expr(&sub)
        .ok()
        .and_then(|r| expr_to_f64(&r).ok())
    };
    if let (Some(a0), Some(a1), Some(a2)) = (at(0.0), at(1.0), at(2.5))
      && (a1 - a0 - 1.0).abs() < 1e-12
      && (a2 - a0 - 2.5).abs() < 1e-12
    {
      let lag = -a0;
      if lag.abs() < 1e-14 {
        return Expr::FunctionCall {
          name: name.clone(),
          args: vec![id_expr(x_name)].into(),
        };
      }
      if lag > 0.0 {
        found.set(true);
        return Expr::FunctionCall {
          name: DELAYED_HEAD.to_string(),
          args: vec![
            Expr::Integer((1 + state_offset[fi]) as i128),
            Expr::Real(lag),
          ]
          .into(),
        };
      }
    }
  }
  map_children(expr, &|c| {
    replace_delayed_values(c, funcs, state_offset, x_name, found)
  })
}

// ─── Compiled numeric expressions ─────────────────────────────────────

/// A residual/event function over the numeric variable vector, compiled
/// to a closure tree when possible and falling back to per-call symbolic
/// substitution otherwise.
enum NumFn {
  Compiled(NExpr),
  Symbolic { expr: Expr, names: Vec<String> },
}

impl NumFn {
  /// The variable indices `>= from` the value depends on, or `None` when
  /// it must be assumed to depend on all of them.
  fn vars_from(&self, from: usize) -> Option<Vec<usize>> {
    match self {
      Self::Compiled(e) => {
        let mut out = Vec::new();
        e.collect_vars_from(from, &mut out).then_some(out)
      }
      Self::Symbolic { .. } => None,
    }
  }

  /// Whether each evaluation solves an algebraic system by Newton
  /// iteration (an index-1 DAE's constraints) or goes through the symbolic
  /// evaluator — too costly for the thousands of evaluations an adaptive
  /// leg spends near a singular point.
  fn is_costly(&self) -> bool {
    match self {
      Self::Compiled(e) => e.has_implicit(),
      Self::Symbolic { .. } => true,
    }
  }

  fn new(expr: Expr, var_names: &[String]) -> Self {
    match compile_numeric(&expr, var_names) {
      Some(compiled) => Self::Compiled(compiled),
      None => Self::Symbolic {
        expr,
        names: var_names.to_vec(),
      },
    }
  }

  fn eval(&self, vars: &[f64]) -> Result<f64, InterpreterError> {
    match self {
      Self::Compiled(nexpr) => Ok(nexpr.eval(vars)),
      Self::Symbolic { expr, names } => {
        let mut substituted = expr.clone();
        for (name, value) in names.iter().zip(vars) {
          substituted = crate::syntax::substitute_variable(
            &substituted,
            name,
            &Expr::Real(*value),
          );
        }
        let result = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
        expr_to_f64(&result)
      }
    }
  }

  /// Evaluate with the highest-derivative slots zeroed (they may
  /// legitimately appear in an event function; zero is the best guess
  /// short of re-solving, and events rarely reference them).
  fn eval_state(
    &self,
    x: f64,
    state: &[f64],
    n_funcs: usize,
  ) -> Result<f64, InterpreterError> {
    let mut vars = Vec::with_capacity(1 + state.len() + n_funcs);
    vars.push(x);
    vars.extend_from_slice(state);
    vars.extend(std::iter::repeat_n(0.0, n_funcs));
    self.eval(&vars)
  }
}

/// A numeric expression tree over a variable vector — the compiled form
/// of a residual, evaluated tens of thousands of times per solve.
enum NExpr {
  Const(f64),
  Var(usize),
  Add(Vec<Self>),
  Mul(Vec<Self>),
  Pow(Box<Self>, Box<Self>),
  Neg(Box<Self>),
  Fn1(fn(f64) -> f64, Box<Self>),
  Fn2(fn(f64, f64) -> f64, Box<Self>, Box<Self>),
  /// `If[lhs op rhs, then, otherwise]` on a single numeric comparison — a
  /// right-hand side that switches branch on the independent variable
  /// (`If[t == 0, limit, general]`, the usual way to dodge a removable
  /// singularity at the start). Only the chosen branch is evaluated, so a
  /// branch that is singular off its own region never poisons the other.
  If {
    op: ComparisonOp,
    lhs: Box<Self>,
    rhs: Box<Self>,
    then: Box<Self>,
    otherwise: Box<Self>,
  },
  /// A leaf the compiler can't reduce to closed-form arithmetic — e.g. a
  /// call to a function defined elsewhere in the session (a boundary value
  /// pinned via `f[n][t_] := …`, say) rather than one of `compile_numeric`'s
  /// known operators. Falls back to full symbolic evaluation, but only for
  /// this leaf: the surrounding arithmetic stays compiled, so one
  /// unresolved call amid an otherwise-numeric right-hand side doesn't
  /// force the *entire* residual through the slow path on every step.
  External {
    template: Expr,
    refs: Vec<(String, usize)>,
  },
  /// A call `InterpolatingFunction[domain, data, order…][arg]` — e.g. a
  /// lookup table built with `Interpolation[…]` and referenced from an
  /// ODE's right-hand side. Goes straight to
  /// `evaluate_interpolating_function` instead of through `External`'s
  /// full symbolic round trip (re-resolving the call generically on every
  /// one of the tens of thousands of RK4 stages a solve takes), while
  /// still calling that same function so the interpolation math itself is
  /// not duplicated.
  Interp1D {
    func_args: Vec<Expr>,
    arg: Box<Self>,
    /// The last `(argument bits, value)` looked up. Building the mass
    /// matrix re-evaluates every residual once per unknown with the same
    /// state, so the same argument recurs many times in a row.
    last: std::cell::Cell<Option<(u64, f64)>>,
  },
  /// A right-hand side written with complex constants (`I`, `Complex[a, b]`)
  /// whose value along the trajectory is real — e.g. a closed-form soliton
  /// velocity field assembled from `E^(I …)` and `Sqrt[… I …]` pieces. It
  /// is evaluated in complex arithmetic and reduced to its real part;
  /// a value with a significant imaginary part is not a real ODE, and
  /// yields NaN so the solve is rejected instead of silently wrong.
  ComplexReal(Box<CExpr>),
  /// The value of state variable `Var(idx)` `lag` earlier than the current
  /// independent-variable value (`vars[0]`) — a delayed argument `f[t - τ]`
  /// of a delay differential equation, read from the solution history.
  Delayed {
    idx: usize,
    lag: f64,
  },
  /// `body`, evaluated with `vars[slot..slot + n]` set to the root of the
  /// square system `constraints == 0` — the unknowns of algebraic equations
  /// that have no closed-form solution (index-1 DAE). `slot` is the first
  /// index past the caller's variable vector. Newton iteration starts from
  /// the last root found (`guess`) and falls back to the initial values
  /// (`init`).
  Implicit {
    constraints: Vec<Self>,
    slot: usize,
    guess: std::cell::RefCell<Vec<f64>>,
    init: Vec<f64>,
    /// The last solve: the first `key_len` variables it was solved at and
    /// the root found, or `None` where Newton failed. Shared by every
    /// residual of a system, which are evaluated at the same point one
    /// after another — a failure is remembered too, since retrying it costs
    /// a full failing Newton run per residual.
    memo: std::rc::Rc<std::cell::RefCell<Option<(Vec<f64>, Option<Vec<f64>>)>>>,
    key_len: usize,
    body: Box<Self>,
  },
}

/// Root of the square system `constraints(vars with vars[slot..] = z)` near
/// `z0` by damped Newton iteration with a finite-difference Jacobian; `None`
/// when it fails to converge.
fn newton_system(
  constraints: &[NExpr],
  vars: &mut [f64],
  slot: usize,
  z0: &[f64],
) -> Option<Vec<f64>> {
  let n = constraints.len();
  let mut z = z0.to_vec();
  let residual = |z: &[f64], vars: &mut [f64]| -> Vec<f64> {
    vars[slot..slot + n].copy_from_slice(z);
    constraints.iter().map(|c| c.eval(vars)).collect()
  };
  let norm = |f: &[f64]| f.iter().map(|v| v * v).sum::<f64>().sqrt();
  let mut f = residual(&z, vars);
  let mut last_step = f64::INFINITY;
  for _ in 0..100 {
    if f.iter().any(|v| !v.is_finite()) {
      return None;
    }
    let fnorm = norm(&f);
    if fnorm == 0.0 {
      return Some(z);
    }
    // Jacobian by forward differences, one column per unknown.
    let mut jac = vec![vec![0.0; n]; n];
    for j in 0..n {
      let h = 1e-7 * z[j].abs().max(1e-3);
      let mut zp = z.clone();
      zp[j] += h;
      let fp = residual(&zp, vars);
      for i in 0..n {
        jac[i][j] = (fp[i] - f[i]) / h;
      }
    }
    let step = solve_dense(jac, f.clone())?;
    // A full Newton step that no longer moves `z` means it is already the
    // root to working precision — a guess carried over from the previous
    // solve often is, with a residual of a few ulps that no step lowers.
    let full_step = z
      .iter()
      .zip(&step)
      .map(|(zi, di)| di.abs() / zi.abs().max(1e-3))
      .fold(0.0, f64::max);
    if full_step <= 1e-12 {
      // That last step is still taken: it brings `z` to the last bit, which
      // a stiff right-hand side multiplying the residual needs.
      let zn: Vec<f64> = z.iter().zip(&step).map(|(zi, di)| zi - di).collect();
      let fnew = residual(&zn, vars);
      return Some(
        if fnew.iter().all(|v| v.is_finite()) && norm(&fnew) <= fnorm {
          zn
        } else {
          z
        },
      );
    }
    let mut lambda = 1.0;
    let mut accepted = false;
    for _ in 0..40 {
      let zn: Vec<f64> = z
        .iter()
        .zip(&step)
        .map(|(zi, di)| zi - lambda * di)
        .collect();
      let fnew = residual(&zn, vars);
      if fnew.iter().all(|v| v.is_finite()) && norm(&fnew) < fnorm {
        z = zn;
        f = fnew;
        accepted = true;
        break;
      }
      lambda *= 0.5;
    }
    if !accepted {
      // At the noise floor no step lowers the residual any further.
      return (last_step.min(full_step) <= 1e-8).then_some(z);
    }
    let rel_step = z
      .iter()
      .zip(&step)
      .map(|(zi, di)| lambda * di.abs() / zi.abs().max(1e-3))
      .fold(0.0, f64::max);
    last_step = rel_step;
    if rel_step <= 1e-13 {
      return Some(z);
    }
  }
  Some(z)
}

/// Solution of the dense linear system `a x = b` by Gaussian elimination
/// with partial pivoting; `None` when `a` is singular.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
  let n = b.len();
  for col in 0..n {
    let pivot = (col..n).max_by(|&i, &j| {
      a[i][col]
        .abs()
        .partial_cmp(&a[j][col].abs())
        .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    if !a[pivot][col].is_finite() || a[pivot][col] == 0.0 {
      return None;
    }
    a.swap(col, pivot);
    b.swap(col, pivot);
    for row in col + 1..n {
      let factor = a[row][col] / a[col][col];
      for k in col..n {
        a[row][k] -= factor * a[col][k];
      }
      b[row] -= factor * b[col];
    }
  }
  let mut x = vec![0.0; n];
  for row in (0..n).rev() {
    let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
    x[row] = (b[row] - tail) / a[row][row];
    if !x[row].is_finite() {
      return None;
    }
  }
  Some(x)
}

impl NExpr {
  /// Push every variable index `>= from` the value can depend on. Returns
  /// `false` when that cannot be told — a node whose inputs are not
  /// visible (an `Implicit` solve, a complex-valued subtree) — in which
  /// case the value must be assumed to depend on every variable.
  fn collect_vars_from(&self, from: usize, out: &mut Vec<usize>) -> bool {
    match self {
      Self::Const(_) | Self::Delayed { .. } => true,
      Self::Var(i) => {
        if *i >= from {
          out.push(*i);
        }
        true
      }
      Self::Add(items) | Self::Mul(items) => {
        items.iter().all(|e| e.collect_vars_from(from, out))
      }
      Self::Pow(a, b) | Self::Fn2(_, a, b) => {
        a.collect_vars_from(from, out) && b.collect_vars_from(from, out)
      }
      Self::Neg(a) | Self::Fn1(_, a) => a.collect_vars_from(from, out),
      Self::If {
        lhs,
        rhs,
        then,
        otherwise,
        ..
      } => [lhs, rhs, then, otherwise]
        .iter()
        .all(|e| e.collect_vars_from(from, out)),
      Self::Interp1D { arg, .. } => arg.collect_vars_from(from, out),
      Self::External { refs, .. } => {
        out.extend(refs.iter().map(|(_, i)| *i).filter(|i| *i >= from));
        true
      }
      Self::ComplexReal(_) | Self::Implicit { .. } => false,
    }
  }

  /// Whether an `Implicit` (Newton-solved) node appears anywhere inside.
  fn has_implicit(&self) -> bool {
    match self {
      Self::Implicit { .. } => true,
      Self::Add(items) | Self::Mul(items) => {
        items.iter().any(Self::has_implicit)
      }
      Self::Pow(a, b) | Self::Fn2(_, a, b) => {
        a.has_implicit() || b.has_implicit()
      }
      Self::Neg(a) | Self::Fn1(_, a) => a.has_implicit(),
      Self::If {
        lhs,
        rhs,
        then,
        otherwise,
        ..
      } => [lhs, rhs, then, otherwise].iter().any(|e| e.has_implicit()),
      Self::Interp1D { arg, .. } => arg.has_implicit(),
      Self::Const(_)
      | Self::Var(_)
      | Self::External { .. }
      | Self::ComplexReal(_)
      | Self::Delayed { .. } => false,
    }
  }

  fn eval(&self, vars: &[f64]) -> f64 {
    match self {
      Self::Implicit {
        constraints,
        slot,
        guess,
        init,
        memo,
        key_len,
        body,
      } => {
        let mut local = vars.to_vec();
        local.resize(slot + constraints.len(), 0.0);
        let key = &vars[..(*key_len).min(vars.len())];
        let cached = memo
          .borrow()
          .as_ref()
          .filter(|(k, _)| k.as_slice() == key)
          .map(|(_, root)| root.clone());
        let root = if let Some(root) = cached {
          root
        } else {
          let start = guess.borrow().clone();
          let root = newton_system(constraints, &mut local, *slot, &start)
            .or_else(|| newton_system(constraints, &mut local, *slot, init));
          if let Some(ref root) = root {
            guess.borrow_mut().clone_from(root);
          }
          *memo.borrow_mut() = Some((key.to_vec(), root.clone()));
          root
        };
        let Some(root) = root else {
          return f64::NAN;
        };
        local[*slot..*slot + root.len()].copy_from_slice(&root);
        body.eval(&local)
      }
      Self::Const(v) => *v,
      Self::Var(i) => vars[*i],
      Self::Add(items) => items.iter().map(|e| e.eval(vars)).sum(),
      Self::Mul(items) => items.iter().map(|e| e.eval(vars)).product(),
      Self::Pow(base, exp) => {
        let b = base.eval(vars);
        let e = exp.eval(vars);
        // Integer exponents use powi to keep negative bases exact.
        if e.fract() == 0.0 && e.abs() < i32::MAX as f64 {
          b.powi(e as i32)
        } else {
          b.powf(e)
        }
      }
      Self::Neg(e) => -e.eval(vars),
      Self::Fn1(f, a) => f(a.eval(vars)),
      Self::Fn2(f, a, b) => f(a.eval(vars), b.eval(vars)),
      Self::If {
        op,
        lhs,
        rhs,
        then,
        otherwise,
      } => {
        let (l, r) = (lhs.eval(vars), rhs.eval(vars));
        // `Equal` on machine reals tolerates the last few bits.
        let close = (l - r).abs() <= l.abs().max(r.abs()) * 2f64.powi(-47);
        let holds = match op {
          ComparisonOp::Equal | ComparisonOp::SameQ => close,
          ComparisonOp::NotEqual | ComparisonOp::UnsameQ => !close,
          ComparisonOp::Less => l < r && !close,
          ComparisonOp::LessEqual => l < r || close,
          ComparisonOp::Greater => l > r && !close,
          ComparisonOp::GreaterEqual => l > r || close,
        };
        if holds {
          then.eval(vars)
        } else {
          otherwise.eval(vars)
        }
      }
      Self::External { template, refs } => {
        let mut substituted = template.clone();
        for (name, idx) in refs {
          substituted = crate::syntax::substitute_variable(
            &substituted,
            name,
            &Expr::Real(vars[*idx]),
          );
        }
        crate::evaluator::evaluate_expr_to_expr(&substituted)
          .ok()
          .and_then(|r| expr_to_f64(&r).ok())
          .unwrap_or(f64::NAN)
      }
      Self::ComplexReal(c) => {
        let z = c.eval(vars);
        if z.im.abs() <= 1e-6 * z.re.abs().max(1.0) {
          z.re
        } else {
          f64::NAN
        }
      }
      Self::Delayed { idx, lag } => dde_history_value(*idx, vars[0] - lag),
      Self::Interp1D {
        func_args,
        arg,
        last,
      } => {
        let x = arg.eval(vars);
        if let Some((bits, value)) = last.get()
          && bits == x.to_bits()
        {
          return value;
        }
        let value =
          evaluate_interpolating_function(func_args, &[Expr::Real(x)])
            .ok()
            .and_then(|r| expr_to_f64(&r).ok())
            .unwrap_or(f64::NAN);
        last.set(Some((x.to_bits(), value)));
        value
      }
    }
  }
}

/// Every `var_names` identifier referenced anywhere inside `expr`, paired
/// with its index — the substitutions an `NExpr::External` leaf needs to
/// apply before handing the template to the full evaluator.
fn collect_var_refs(
  expr: &Expr,
  var_names: &[String],
  out: &mut Vec<(String, usize)>,
) {
  if let Expr::Identifier(name) = expr {
    if let Some(idx) = var_names.iter().position(|n| n == name)
      && !out.iter().any(|(n, _)| n == name)
    {
      out.push((name.clone(), idx));
    }
    return;
  }
  for child in expr_children(expr) {
    collect_var_refs(child, var_names, out);
  }
}

/// Wrap a construct `compile_numeric` doesn't otherwise reduce to
/// closed-form arithmetic as an `NExpr::External` leaf.
fn compile_external_leaf(expr: &Expr, var_names: &[String]) -> NExpr {
  let mut refs = Vec::new();
  collect_var_refs(expr, var_names, &mut refs);
  NExpr::External {
    template: expr.clone(),
    refs,
  }
}

/// If `expr` is a call `InterpolatingFunction[domain, data, …][arg]` — a
/// lookup table built by `Interpolation[…]`/`ListInterpolation[…]` or an
/// NDSolve result, already resolved to its value by the time equations
/// reach this compiler — compile it as an `NExpr::Interp1D` leaf so a
/// residual that looks the table up stays out of `External`'s full
/// symbolic dispatch on every RK4 stage. Returns `None` for anything else:
/// a 2-D grid interpolation (its call takes a `{x, y}` pair, not this
/// leaf's single scalar) or a call whose head isn't `InterpolatingFunction`.
fn compile_interp1d_call(
  func: &Expr,
  args: &[Expr],
  var_names: &[String],
) -> Option<NExpr> {
  if args.len() != 1 {
    return None;
  }
  let Expr::FunctionCall {
    name: interp_name,
    args: func_args,
  } = func
  else {
    return None;
  };
  if interp_name != "InterpolatingFunction"
    || !(2..=4).contains(&func_args.len())
  {
    return None;
  }
  // The 2-D grid form carries `{orderRow, orderCol}` as its third
  // argument; a plain 1-D table has a single integer order (or none).
  if matches!(func_args.get(2), Some(Expr::List(_))) {
    return None;
  }
  let arg = compile_numeric(&args[0], var_names)?;
  Some(NExpr::Interp1D {
    func_args: func_args.to_vec(),
    arg: Box::new(arg),
    last: std::cell::Cell::new(None),
  })
}

/// A real-valued expression in some variables compiled once for repeated
/// numeric evaluation — the closure tree NDSolve evaluates its residuals
/// with, also used to sample plots. `eval` gives NaN where the value is not
/// a real number.
pub(crate) struct CompiledReal(NExpr);

impl CompiledReal {
  pub(crate) fn new(expr: &Expr, var_names: &[String]) -> Option<Self> {
    compile_numeric(expr, var_names).map(Self)
  }

  pub(crate) fn eval(&self, values: &[f64]) -> f64 {
    self.0.eval(values)
  }
}

/// Compile an expression over the given variable names to an `NExpr`.
/// Returns `None` for any construct outside the supported numeric subset
/// (the caller then falls back to symbolic evaluation).
fn compile_numeric(expr: &Expr, var_names: &[String]) -> Option<NExpr> {
  if contains_imaginary(expr) {
    return compile_complex(expr, var_names)
      .map(|c| NExpr::ComplexReal(Box::new(c)));
  }
  compile_numeric_real(expr, var_names)
}

fn compile_numeric_real(expr: &Expr, var_names: &[String]) -> Option<NExpr> {
  let comp = |e: &Expr| compile_numeric(e, var_names);
  match expr {
    Expr::Integer(n) => Some(NExpr::Const(*n as f64)),
    Expr::Real(v) => Some(NExpr::Const(*v)),
    Expr::Identifier(name) => {
      if let Some(idx) = var_names.iter().position(|n| n == name) {
        return Some(NExpr::Var(idx));
      }
      match name.as_str() {
        "Pi" => Some(NExpr::Const(std::f64::consts::PI)),
        "E" => Some(NExpr::Const(std::f64::consts::E)),
        "Degree" => Some(NExpr::Const(std::f64::consts::PI / 180.0)),
        "GoldenRatio" => Some(NExpr::Const(f64::midpoint(1.0, 5.0_f64.sqrt()))),
        // A parameter left symbolic in a held branch (`If[t <= 100, 2.5,
        // 2.5 + delta/40]`) takes the value it has where NDSolve runs,
        // e.g. from an enclosing Block.
        _ => crate::evaluator::evaluate_expr_to_expr(expr)
          .ok()
          .filter(|v| !matches!(v, Expr::Identifier(_)))
          .and_then(|v| crate::functions::math_ast::try_eval_to_f64(&v))
          .map(NExpr::Const),
      }
    }
    Expr::BinaryOp { op, left, right } => {
      let l = comp(left)?;
      let r = comp(right)?;
      Some(match op {
        BinaryOperator::Plus => NExpr::Add(vec![l, r]),
        BinaryOperator::Minus => NExpr::Add(vec![l, NExpr::Neg(Box::new(r))]),
        BinaryOperator::Times => NExpr::Mul(vec![l, r]),
        BinaryOperator::Divide => NExpr::Mul(vec![
          l,
          NExpr::Pow(Box::new(r), Box::new(NExpr::Const(-1.0))),
        ]),
        BinaryOperator::Power => NExpr::Pow(Box::new(l), Box::new(r)),
        _ => return None,
      })
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(NExpr::Neg(Box::new(comp(operand)?))),
    Expr::FunctionCall { name, args } => {
      if name == DELAYED_HEAD
        && let [Expr::Integer(idx), Expr::Real(lag)] = args.as_ref()
      {
        return Some(NExpr::Delayed {
          idx: *idx as usize,
          lag: *lag,
        });
      }
      let unary: Option<fn(f64) -> f64> = match name.as_str() {
        "Sin" => Some(f64::sin),
        "Cos" => Some(f64::cos),
        "Tan" => Some(f64::tan),
        "Sec" => Some(|v: f64| 1.0 / v.cos()),
        "Csc" => Some(|v: f64| 1.0 / v.sin()),
        "Cot" => Some(|v: f64| 1.0 / v.tan()),
        "Sinh" => Some(f64::sinh),
        "Cosh" => Some(f64::cosh),
        "Tanh" => Some(f64::tanh),
        "ArcSin" => Some(f64::asin),
        "ArcCos" => Some(f64::acos),
        "ArcTan" if args.len() == 1 => Some(f64::atan),
        "Exp" => Some(f64::exp),
        "Log" if args.len() == 1 => Some(f64::ln),
        "Sqrt" => Some(f64::sqrt),
        "Abs" => Some(f64::abs),
        "Sign" => Some(|v: f64| if v == 0.0 { 0.0 } else { v.signum() }),
        // Step functions make RK4 stages land exactly on the jump
        // (`HeavisideTheta[0.]` stays symbolic in the evaluator), so they
        // are compiled with the midpoint value there.
        "HeavisideTheta" => Some(|v: f64| {
          if v > 0.0 {
            1.0
          } else if v < 0.0 {
            0.0
          } else {
            0.5
          }
        }),
        "UnitStep" => Some(|v: f64| if v < 0.0 { 0.0 } else { 1.0 }),
        "Floor" if args.len() == 1 => Some(f64::floor),
        "Ceiling" if args.len() == 1 => Some(f64::ceil),
        _ => None,
      };
      if let Some(f) = unary
        && args.len() == 1
      {
        return Some(NExpr::Fn1(f, Box::new(comp(&args[0])?)));
      }
      match name.as_str() {
        "Plus" => Some(NExpr::Add(
          args.iter().map(comp).collect::<Option<Vec<_>>>()?,
        )),
        "Times" => Some(NExpr::Mul(
          args.iter().map(comp).collect::<Option<Vec<_>>>()?,
        )),
        "Subtract" if args.len() == 2 => Some(NExpr::Add(vec![
          comp(&args[0])?,
          NExpr::Neg(Box::new(comp(&args[1])?)),
        ])),
        "Divide" if args.len() == 2 => Some(NExpr::Mul(vec![
          comp(&args[0])?,
          NExpr::Pow(Box::new(comp(&args[1])?), Box::new(NExpr::Const(-1.0))),
        ])),
        "Power" if args.len() == 2 => Some(NExpr::Pow(
          Box::new(comp(&args[0])?),
          Box::new(comp(&args[1])?),
        )),
        "Rational" if args.len() == 2 => match (&args[0], &args[1]) {
          (Expr::Integer(a), Expr::Integer(b)) if *b != 0 => {
            Some(NExpr::Const(*a as f64 / *b as f64))
          }
          _ => None,
        },
        // Wolfram's `ArcTan[x, y]` is the angle of the point (x, y).
        "ArcTan" if args.len() == 2 => Some(NExpr::Fn2(
          |x, y| y.atan2(x),
          Box::new(comp(&args[0])?),
          Box::new(comp(&args[1])?),
        )),
        "Log" if args.len() == 2 => Some(NExpr::Fn2(
          |b, v| v.ln() / b.ln(),
          Box::new(comp(&args[0])?),
          Box::new(comp(&args[1])?),
        )),
        "Mod" if args.len() == 2 => Some(NExpr::Fn2(
          f64::rem_euclid,
          Box::new(comp(&args[0])?),
          Box::new(comp(&args[1])?),
        )),
        "Max" if !args.is_empty() => {
          let compiled = args.iter().map(comp).collect::<Option<Vec<_>>>()?;
          compiled
            .into_iter()
            .reduce(|acc, e| NExpr::Fn2(f64::max, Box::new(acc), Box::new(e)))
        }
        "Min" if !args.is_empty() => {
          let compiled = args.iter().map(comp).collect::<Option<Vec<_>>>()?;
          compiled
            .into_iter()
            .reduce(|acc, e| NExpr::Fn2(f64::min, Box::new(acc), Box::new(e)))
        }
        "If" if args.len() == 3 => {
          if let Expr::Comparison {
            operands,
            operators,
          } = &args[0]
            && let ([lhs, rhs], [op]) =
              (operands.as_slice(), operators.as_slice())
          {
            return Some(NExpr::If {
              op: *op,
              lhs: Box::new(comp(lhs)?),
              rhs: Box::new(comp(rhs)?),
              then: Box::new(comp(&args[1])?),
              otherwise: Box::new(comp(&args[2])?),
            });
          }
          Some(compile_external_leaf(expr, var_names))
        }
        // Not one of the closed-form operators above — most commonly a
        // call to a function the session itself defined (a fixed boundary
        // value, a helper via `f[n_] := …`), which can only be resolved by
        // the full evaluator. Compiled as an External leaf rather than
        // bailing the whole residual out to the symbolic path.
        _ => Some(compile_external_leaf(expr, var_names)),
      }
    }
    // `xy = Interpolation[…]; …xy[y[t]]…` — by the time an ODE's
    // right-hand side reaches this compiler, the symbol `xy` has already
    // been resolved to its value, leaving `InterpolatingFunction[…][arg]`
    // as a `CurriedCall` rather than a plain named call. Recognizing it
    // here keeps a table lookup out of `External`'s full generic dispatch
    // on every one of the tens of thousands of RK4 stages a solve takes.
    Expr::CurriedCall { func, args } => {
      compile_interp1d_call(func, args, var_names)
        .or_else(|| Some(compile_external_leaf(expr, var_names)))
    }
    other => Some(compile_external_leaf(other, var_names)),
  }
}

// ─── Complex-valued right-hand sides ───────────────────────────────────

#[derive(Clone, Copy)]
struct Cx {
  re: f64,
  im: f64,
}

impl Cx {
  const fn real(re: f64) -> Self {
    Self { re, im: 0.0 }
  }
  fn add(self, o: Self) -> Self {
    Self {
      re: self.re + o.re,
      im: self.im + o.im,
    }
  }
  fn mul(self, o: Self) -> Self {
    Self {
      re: self.re * o.re - self.im * o.im,
      im: self.re * o.im + self.im * o.re,
    }
  }
  fn neg(self) -> Self {
    Self {
      re: -self.re,
      im: -self.im,
    }
  }
  fn recip(self) -> Self {
    let d = self.re * self.re + self.im * self.im;
    Self {
      re: self.re / d,
      im: -self.im / d,
    }
  }
  fn exp(self) -> Self {
    let m = self.re.exp();
    Self {
      re: m * self.im.cos(),
      im: m * self.im.sin(),
    }
  }
  fn ln(self) -> Self {
    Self {
      re: self.re.hypot(self.im).ln(),
      im: self.im.atan2(self.re),
    }
  }
  fn sqrt(self) -> Self {
    let r = self.re.hypot(self.im);
    let re = f64::midpoint(r, self.re).sqrt();
    let im = ((r - self.re) / 2.0).sqrt();
    Self {
      re,
      im: if self.im < 0.0 { -im } else { im },
    }
  }
  fn sin(self) -> Self {
    Self {
      re: self.re.sin() * self.im.cosh(),
      im: self.re.cos() * self.im.sinh(),
    }
  }
  fn cos(self) -> Self {
    Self {
      re: self.re.cos() * self.im.cosh(),
      im: -self.re.sin() * self.im.sinh(),
    }
  }
  fn sinh(self) -> Self {
    Self {
      re: self.re.sinh() * self.im.cos(),
      im: self.re.cosh() * self.im.sin(),
    }
  }
  fn cosh(self) -> Self {
    Self {
      re: self.re.cosh() * self.im.cos(),
      im: self.re.sinh() * self.im.sin(),
    }
  }
  fn powc(self, e: Self) -> Self {
    if e.im == 0.0 && e.re.fract() == 0.0 && e.re.abs() <= 64.0 {
      let mut acc = Self::real(1.0);
      for _ in 0..(e.re.abs() as u32) {
        acc = acc.mul(self);
      }
      return if e.re < 0.0 { acc.recip() } else { acc };
    }
    if self.re == 0.0 && self.im == 0.0 {
      return Self::real(0.0);
    }
    e.mul(self.ln()).exp()
  }
}

/// Complex counterpart of `NExpr`, for right-hand sides that mention `I`.
enum CExpr {
  Const(Cx),
  Var(usize),
  Add(Vec<Self>),
  Mul(Vec<Self>),
  Pow(Box<Self>, Box<Self>),
  Neg(Box<Self>),
  Fn1(fn(Cx) -> Cx, Box<Self>),
}

impl CExpr {
  fn eval(&self, vars: &[f64]) -> Cx {
    match self {
      Self::Const(c) => *c,
      Self::Var(i) => Cx::real(vars[*i]),
      Self::Add(items) => items
        .iter()
        .fold(Cx::real(0.0), |acc, e| acc.add(e.eval(vars))),
      Self::Mul(items) => items
        .iter()
        .fold(Cx::real(1.0), |acc, e| acc.mul(e.eval(vars))),
      Self::Pow(b, e) => b.eval(vars).powc(e.eval(vars)),
      Self::Neg(e) => e.eval(vars).neg(),
      Self::Fn1(f, a) => f(a.eval(vars)),
    }
  }
}

/// Does `expr` mention the imaginary unit or a `Complex[…]` literal?
fn contains_imaginary(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(n) | Expr::Constant(n) => n == "I",
    Expr::FunctionCall { name, .. } if name == "Complex" => true,
    _ => expr_children(expr).into_iter().any(contains_imaginary),
  }
}

fn compile_complex(expr: &Expr, var_names: &[String]) -> Option<CExpr> {
  let comp = |e: &Expr| compile_complex(e, var_names);
  let all = |args: &[Expr]| args.iter().map(comp).collect::<Option<Vec<_>>>();
  let inv =
    |e: CExpr| CExpr::Pow(Box::new(e), Box::new(CExpr::Const(Cx::real(-1.0))));
  match expr {
    Expr::Integer(n) => Some(CExpr::Const(Cx::real(*n as f64))),
    Expr::Real(v) => Some(CExpr::Const(Cx::real(*v))),
    Expr::Identifier(name) | Expr::Constant(name) => {
      if let Some(idx) = var_names.iter().position(|n| n == name) {
        return Some(CExpr::Var(idx));
      }
      let c = match name.as_str() {
        "I" => Cx { re: 0.0, im: 1.0 },
        "Pi" => Cx::real(std::f64::consts::PI),
        "E" => Cx::real(std::f64::consts::E),
        "Degree" => Cx::real(std::f64::consts::PI / 180.0),
        _ => return None,
      };
      Some(CExpr::Const(c))
    }
    Expr::BinaryOp { op, left, right } => {
      let (l, r) = (comp(left)?, comp(right)?);
      Some(match op {
        BinaryOperator::Plus => CExpr::Add(vec![l, r]),
        BinaryOperator::Minus => CExpr::Add(vec![l, CExpr::Neg(Box::new(r))]),
        BinaryOperator::Times => CExpr::Mul(vec![l, r]),
        BinaryOperator::Divide => CExpr::Mul(vec![l, inv(r)]),
        BinaryOperator::Power => CExpr::Pow(Box::new(l), Box::new(r)),
        _ => return None,
      })
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(CExpr::Neg(Box::new(comp(operand)?))),
    Expr::FunctionCall { name, args } => {
      let unary: Option<fn(Cx) -> Cx> = match name.as_str() {
        "Exp" => Some(Cx::exp),
        "Log" if args.len() == 1 => Some(Cx::ln),
        "Sqrt" => Some(Cx::sqrt),
        "Sin" => Some(Cx::sin),
        "Cos" => Some(Cx::cos),
        "Sinh" => Some(Cx::sinh),
        "Cosh" => Some(Cx::cosh),
        _ => None,
      };
      if let Some(f) = unary
        && args.len() == 1
      {
        return Some(CExpr::Fn1(f, Box::new(comp(&args[0])?)));
      }
      match name.as_str() {
        "Plus" => Some(CExpr::Add(all(args)?)),
        "Times" => Some(CExpr::Mul(all(args)?)),
        "Subtract" if args.len() == 2 => Some(CExpr::Add(vec![
          comp(&args[0])?,
          CExpr::Neg(Box::new(comp(&args[1])?)),
        ])),
        "Divide" if args.len() == 2 => {
          Some(CExpr::Mul(vec![comp(&args[0])?, inv(comp(&args[1])?)]))
        }
        "Power" if args.len() == 2 => Some(CExpr::Pow(
          Box::new(comp(&args[0])?),
          Box::new(comp(&args[1])?),
        )),
        "Rational" if args.len() == 2 => match (&args[0], &args[1]) {
          (Expr::Integer(a), Expr::Integer(b)) if *b != 0 => {
            Some(CExpr::Const(Cx::real(*a as f64 / *b as f64)))
          }
          _ => None,
        },
        "Complex" if args.len() == 2 => Some(CExpr::Const(Cx {
          re: nval_to_f64(&args[0])?,
          im: nval_to_f64(&args[1])?,
        })),
        _ => None,
      }
    }
    _ => None,
  }
}

// ─── ODE Term Structures ───────────────────────────────────────────────

/// Represents a term in the ODE: coefficient * y^(order)[x]
/// order == -1 means it's a forcing term (no y dependence)
#[derive(Debug, Clone)]
struct OdeTerm {
  /// Derivative order: 0 for y[x], 1 for y'[x], 2 for y''[x], etc.
  /// -1 for forcing terms (free of y)
  order: i32,
  /// The coefficient multiplying this term
  coefficient: Expr,
}

// ─── ODE Parsing Helpers ───────────────────────────────────────────────

/// Normalize equation: lhs == rhs → lhs - rhs (everything on left side)
/// Does `expr` contain any derivative anywhere — a `Derivative[...]` head
/// (i.e. `y'[x]`, `y''[x]`, … which parse to `Derivative[n][y][x]`) or a `D[…]`
/// operator? Used to distinguish a genuine ODE/PDE from a purely algebraic
/// equation in the dependent function.
fn contains_derivative(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == "Derivative"
        || name == "D"
        || args.iter().any(contains_derivative)
    }
    // Derivative[n][y][x] parses to a nested CurriedCall whose innermost head
    // is FunctionCall("Derivative", …); traverse both func and args.
    Expr::CurriedCall { func, args } => {
      contains_derivative(func) || args.iter().any(contains_derivative)
    }
    Expr::List(items) => items.iter().any(contains_derivative),
    Expr::BinaryOp { left, right, .. } => {
      contains_derivative(left) || contains_derivative(right)
    }
    Expr::UnaryOp { operand, .. } => contains_derivative(operand),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(contains_derivative)
    }
    _ => false,
  }
}

fn normalize_equation(eq: &Expr) -> Result<Expr, InterpreterError> {
  match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal =>
    {
      let lhs = &operands[0];
      let rhs = &operands[1];
      // lhs - rhs
      Ok(minus2(lhs.clone(), rhs.clone()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "DSolve expects an equation (lhs == rhs)".into(),
    )),
  }
}

/// Check if an expression is an initial condition like `y[0] == 1` or
/// `y'[0] == 0`. The condition's point must NOT be the independent variable
/// `x`; otherwise `y[x] == …` / `y'[x] == …` (i.e. the ODE itself) would be
/// misclassified as an initial condition.
fn is_initial_condition(expr: &Expr, y_name: &str, x_name: &str) -> bool {
  let is_ode_point = |p: &Expr| matches!(p, Expr::Identifier(s) if s == x_name);
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && operands.len() == 2
    && operators.len() == 1
    && operators[0] == ComparisonOp::Equal
  {
    let lhs = &operands[0];
    // y[val]
    if let Expr::FunctionCall { name, args } = lhs
      && name == y_name
      && args.len() == 1
    {
      return !is_ode_point(&args[0]);
    }
    // Derivative[n][y][val] — curried/flattened form
    if let Some((_, point)) = extract_derivative_order_and_point(lhs, y_name) {
      return !is_ode_point(&point);
    }
  }
  false
}

/// Extract derivative order and evaluation point from y^(n)[val]
/// Returns (order, val_expr) if matched
fn extract_derivative_order_and_point(
  expr: &Expr,
  y_name: &str,
) -> Option<(usize, Expr)> {
  // Flattened form: FunctionCall("Derivative", [n, y, val])
  if let Expr::FunctionCall { name, args } = expr
    && name == "Derivative"
    && args.len() == 3
    && let Expr::Integer(n) = &args[0]
    && let Expr::Identifier(fname) = &args[1]
    && fname == y_name
  {
    return Some((*n as usize, args[2].clone()));
  }

  // CurriedCall form: Derivative[n][y][val]
  if let Expr::CurriedCall { func, args } = expr
    && args.len() == 1
  {
    // CurriedCall { func: FunctionCall("Derivative", [n, y]), args: [val] }
    if let Expr::FunctionCall {
      name: deriv_name,
      args: deriv_args,
    } = func.as_ref()
      && deriv_name == "Derivative"
      && deriv_args.len() == 2
      && let Expr::Integer(n) = &deriv_args[0]
      && let Expr::Identifier(fname) = &deriv_args[1]
      && fname == y_name
    {
      return Some((*n as usize, args[0].clone()));
    }

    // CurriedCall { func: CurriedCall { func: FunctionCall("Derivative", [n]), args: [Id(y)] }, args: [val] }
    if let Expr::CurriedCall {
      func: inner_func,
      args: inner_args,
    } = func.as_ref()
      && inner_args.len() == 1
      && let Expr::Identifier(name) = &inner_args[0]
      && name == y_name
      && let Expr::FunctionCall {
        name: deriv_name,
        args: deriv_args,
      } = inner_func.as_ref()
      && deriv_name == "Derivative"
      && deriv_args.len() == 1
      && let Expr::Integer(n) = &deriv_args[0]
    {
      return Some((*n as usize, args[0].clone()));
    }
  }
  None
}

/// Evaluate an initial-condition side (or domain bound) numerically.
/// Exact symbolic values (`-ArcCos[31/40]`, `Pi/6`, …) numericise via the
/// `N` fallback of `interp_value_to_f64` — NDSolve is a numeric solver,
/// so this loses nothing.
fn nval_to_f64(expr: &Expr) -> Option<f64> {
  interp_value_to_f64(expr).ok()
}

/// Is `expr` the `Infinity` symbol (as opposed to a finite numeric bound)?
fn is_infinity(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
}

/// Parse a numeric initial condition: y[x0] == y0 or y'[x0] == y0
/// Returns (derivative_order, x_val, y_val)
fn parse_numeric_initial_condition(
  expr: &Expr,
  y_name: &str,
) -> std::option::Option<(usize, f64, f64)> {
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && operands.len() == 2
    && operators.len() == 1
    && operators[0] == ComparisonOp::Equal
  {
    let lhs = &operands[0];

    // y[x0] == val
    if let Expr::FunctionCall { name, args } = lhs
      && name == y_name
      && args.len() == 1
    {
      // Try to evaluate both sides numerically
      if let (Some(x_val), Some(rhs_val)) =
        (nval_to_f64(&args[0]), nval_to_f64(&operands[1]))
      {
        return Some((0, x_val, rhs_val));
      }
    }

    // Derivative[n, y, x0] == val or Derivative[n][y][x0] == val
    if let Some((order, val_expr)) =
      extract_derivative_order_and_point(lhs, y_name)
      && let (Some(x_val), Some(rhs_val)) =
        (nval_to_f64(&val_expr), nval_to_f64(&operands[1]))
    {
      return Some((order, x_val, rhs_val));
    }
  }
  None
}

/// Collect all additive terms from the normalized ODE expression,
/// classifying each by derivative order of y.
fn collect_ode_terms(
  expr: &Expr,
  y_name: &str,
  x_name: &str,
) -> Result<Vec<OdeTerm>, InterpreterError> {
  let mut terms = Vec::new();
  collect_additive_terms(expr, y_name, x_name, false, &mut terms)?;
  Ok(terms)
}

/// Recursively collect additive terms, handling Plus, Minus, UnaryMinus
fn collect_additive_terms(
  expr: &Expr,
  y_name: &str,
  x_name: &str,
  negated: bool,
  terms: &mut Vec<OdeTerm>,
) -> Result<(), InterpreterError> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_additive_terms(left, y_name, x_name, negated, terms)?;
      collect_additive_terms(right, y_name, x_name, negated, terms)?;
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      collect_additive_terms(left, y_name, x_name, negated, terms)?;
      collect_additive_terms(right, y_name, x_name, !negated, terms)?;
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      collect_additive_terms(operand, y_name, x_name, !negated, terms)?;
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      for arg in args {
        collect_additive_terms(arg, y_name, x_name, negated, terms)?;
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Times[a, b, ...] — check which factor is y-related
      classify_product_term(args, y_name, x_name, negated, terms)?;
    }
    _ => {
      // Single term: classify it
      let term = classify_single_term(expr, y_name, x_name)?;
      let coeff = if negated {
        negate_expr(&term.coefficient)
      } else {
        term.coefficient
      };
      terms.push(OdeTerm {
        order: term.order,
        coefficient: coeff,
      });
    }
  }
  Ok(())
}

/// Classify a single expression as an ODE term
fn classify_single_term(
  expr: &Expr,
  y_name: &str,
  _x_name: &str,
) -> Result<OdeTerm, InterpreterError> {
  // Check if expr is y[x] — order 0
  if let Expr::FunctionCall { name, args } = expr
    && name == y_name
    && args.len() == 1
  {
    return Ok(OdeTerm {
      order: 0,
      coefficient: Expr::Integer(1),
    });
  }

  // Check if expr is Derivative[n][y][x] — order n
  if let Some(order) = extract_derivative_order(expr, y_name) {
    return Ok(OdeTerm {
      order: order as i32,
      coefficient: Expr::Integer(1),
    });
  }

  // Check if it's a product: coeff * y^(n)[x]
  if let Expr::BinaryOp {
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    // Check left * y^(n)[x]
    if let Some(order) = extract_derivative_order(right, y_name) {
      return Ok(OdeTerm {
        order: order as i32,
        coefficient: *left.clone(),
      });
    }
    if let Expr::FunctionCall { name, args } = right.as_ref()
      && name == y_name
      && args.len() == 1
    {
      return Ok(OdeTerm {
        order: 0,
        coefficient: *left.clone(),
      });
    }
    // Check y^(n)[x] * right
    if let Some(order) = extract_derivative_order(left, y_name) {
      return Ok(OdeTerm {
        order: order as i32,
        coefficient: *right.clone(),
      });
    }
    if let Expr::FunctionCall { name, args } = left.as_ref()
      && name == y_name
      && args.len() == 1
    {
      return Ok(OdeTerm {
        order: 0,
        coefficient: *right.clone(),
      });
    }
  }

  // Not y-related: it's a forcing term
  if is_free_of_y(expr, y_name) {
    return Ok(OdeTerm {
      order: -1,
      coefficient: expr.clone(),
    });
  }

  // Complex y-dependent term we can't handle
  Err(InterpreterError::EvaluationError(format!(
    "DSolve: cannot classify term involving {y_name}"
  )))
}

/// Classify a product (Times[...]) as an ODE term
fn classify_product_term(
  factors: &[Expr],
  y_name: &str,
  _x_name: &str,
  negated: bool,
  terms: &mut Vec<OdeTerm>,
) -> Result<(), InterpreterError> {
  // Find the y-dependent factor
  let mut y_factor_idx = None;
  let mut y_order = -1i32;

  for (i, factor) in factors.iter().enumerate() {
    if let Some(order) = extract_derivative_order(factor, y_name) {
      y_factor_idx = Some(i);
      y_order = order as i32;
      break;
    }
    if let Expr::FunctionCall { name, args } = factor
      && name == y_name
      && args.len() == 1
    {
      y_factor_idx = Some(i);
      y_order = 0;
      break;
    }
  }

  let order;
  let coefficient;

  if let Some(idx) = y_factor_idx {
    order = y_order;
    // Coefficient is product of all other factors
    let other_factors: Vec<&Expr> = factors
      .iter()
      .enumerate()
      .filter(|(i, _)| *i != idx)
      .map(|(_, f)| f)
      .collect();
    coefficient = if other_factors.is_empty() {
      Expr::Integer(1)
    } else if other_factors.len() == 1 {
      other_factors[0].clone()
    } else {
      times(other_factors.into_iter().cloned().collect())
    };
  } else {
    // No recognized linear y factor. The product is only a forcing term if it
    // is genuinely free of y; otherwise it depends on y nonlinearly (e.g.
    // x*y[x]^2) and the linear solver cannot handle it — bail out so DSolve
    // returns unevaluated rather than fabricating a circular
    // `C[1] + Integrate[x*y[x]^2, x]` "solution". Mirrors classify_single_term.
    let product = if factors.len() == 1 {
      factors[0].clone()
    } else {
      unevaluated("Times", factors)
    };
    if !is_free_of_y(&product, y_name) {
      return Err(InterpreterError::EvaluationError(format!(
        "DSolve: cannot classify term involving {y_name}"
      )));
    }
    order = -1;
    coefficient = product;
  }

  let coeff = if negated {
    negate_expr(&coefficient)
  } else {
    coefficient
  };

  terms.push(OdeTerm {
    order,
    coefficient: coeff,
  });
  Ok(())
}

/// Extract derivative order from Derivative[n][y][x] pattern
/// After evaluation, this can appear as:
///   - FunctionCall("Derivative", [n, y, x]) — fully flattened form
///   - CurriedCall { func: CurriedCall { func: FunctionCall("Derivative", [n]), args: [Id(y)] }, args: [x] }
fn extract_derivative_order(expr: &Expr, y_name: &str) -> Option<usize> {
  // Flattened form: FunctionCall("Derivative", [n, y, x])
  if let Expr::FunctionCall { name, args } = expr
    && name == "Derivative"
    && args.len() == 3
    && let Expr::Integer(n) = &args[0]
    && let Expr::Identifier(fname) = &args[1]
    && fname == y_name
  {
    return Some(*n as usize);
  }

  // CurriedCall form: Derivative[n][y][x]
  if let Expr::CurriedCall { func, args: _ } = expr {
    if let Expr::CurriedCall {
      func: inner_func,
      args: inner_args,
    } = func.as_ref()
      && inner_args.len() == 1
      && let Expr::Identifier(name) = &inner_args[0]
      && name == y_name
      && let Expr::FunctionCall {
        name: deriv_name,
        args: deriv_args,
      } = inner_func.as_ref()
      && deriv_name == "Derivative"
      && deriv_args.len() == 1
      && let Expr::Integer(n) = &deriv_args[0]
    {
      return Some(*n as usize);
    }
    // Also handle FunctionCall("Derivative", [n, y])[x]
    if let Expr::FunctionCall {
      name: deriv_name,
      args: deriv_args,
    } = func.as_ref()
      && deriv_name == "Derivative"
      && deriv_args.len() == 2
      && let Expr::Integer(n) = &deriv_args[0]
      && let Expr::Identifier(fname) = &deriv_args[1]
      && fname == y_name
    {
      return Some(*n as usize);
    }
  }
  None
}

/// Check if expression is free of the dependent variable y
fn is_free_of_y(expr: &Expr, y_name: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name != y_name,
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      true
    }
    Expr::FunctionCall { name, args } => {
      if name == y_name {
        return false;
      }
      // Derivative[n, y, x] contains y
      if name == "Derivative"
        && args.len() >= 2
        && matches!(&args[1], Expr::Identifier(n) if n == y_name)
      {
        return false;
      }
      args.iter().all(|a| is_free_of_y(a, y_name))
    }
    Expr::BinaryOp { left, right, .. } => {
      is_free_of_y(left, y_name) && is_free_of_y(right, y_name)
    }
    Expr::UnaryOp { operand, .. } => is_free_of_y(operand, y_name),
    Expr::List(items) => items.iter().all(|e| is_free_of_y(e, y_name)),
    Expr::CurriedCall { func, args } => {
      is_free_of_y(func, y_name) && args.iter().all(|a| is_free_of_y(a, y_name))
    }
    _ => false,
  }
}

/// Negate an expression
fn negate_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::Real(f) => Expr::Real(-f),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    _ => times2(Expr::Integer(-1), expr.clone()),
  }
}

// ─── Constant Coefficient ODE Solver ───────────────────────────────────

/// Solve constant-coefficient linear ODE
/// a_n * y^(n) + ... + a_1 * y' + a_0 * y = f(x)
fn solve_constant_coefficient_ode(
  terms: &[OdeTerm],
  max_order: usize,
  x_name: &str,
) -> Result<Expr, InterpreterError> {
  // Symbolic coefficients (`y'' + k^2 y == 0`) have symbolic roots.
  if max_order == 2
    && terms
      .iter()
      .filter(|t| t.order >= 0)
      .any(|t| eval_to_f64(&t.coefficient).is_err())
    && let Some(solution) = solve_symbolic_second_order(terms, x_name)?
  {
    return Ok(solution);
  }
  // Extract numeric coefficients for the characteristic equation
  let mut coeffs: Vec<f64> = vec![0.0; max_order + 1];
  let mut forcing: Option<Expr> = None;

  for term in terms {
    if term.order >= 0 {
      let idx = term.order as usize;
      let val = eval_to_f64(&term.coefficient)?;
      coeffs[idx] += val;
    } else {
      // Forcing term
      let evaluated =
        crate::evaluator::evaluate_expr_to_expr(&term.coefficient)
          .unwrap_or(term.coefficient.clone());
      match &forcing {
        None => forcing = Some(evaluated),
        Some(existing) => {
          forcing = Some(plus2(existing.clone(), evaluated));
        }
      }
    }
  }

  // Solve characteristic equation: a_n*r^n + ... + a_1*r + a_0 = 0
  let roots = find_characteristic_roots(&coeffs, max_order)?;

  // Build the general homogeneous solution
  let homogeneous = build_homogeneous_solution(&roots, x_name);

  // If there's no forcing term, we're done
  if forcing.is_none() || matches!(&forcing, Some(Expr::Integer(0))) {
    return Ok(homogeneous);
  }

  // For non-homogeneous: return homogeneous part + particular solution
  // For now, handle simple forcing terms
  if let Some(forcing_expr) = &forcing {
    let particular = find_particular_solution(
      &coeffs,
      max_order,
      forcing_expr,
      &roots,
      x_name,
    )
    .or_else(|| variation_of_parameters(&coeffs, &roots, forcing_expr, x_name));
    if let Some(part) = particular {
      return Ok(crate::functions::calculus_ast::simplify(plus2(
        homogeneous,
        part,
      )));
    }
  }

  // A non-constant forcing term we cannot integrate: returning just the
  // homogeneous part would silently drop the forcing and produce a wrong
  // answer, so leave DSolve unevaluated instead.
  Err(InterpreterError::EvaluationError(
    "DSolve: cannot find a particular solution for the forcing term".into(),
  ))
}

/// Particular solution of a second-order constant-coefficient ODE
/// a2 y'' + a1 y' + a0 y = g(x) by variation of parameters:
///
///   y_p = -y1 ∫ y2 g / (a2 W) dx + y2 ∫ y1 g / (a2 W) dx
///
/// with {y1, y2} the fundamental pair from the characteristic roots and W
/// their Wronskian (computed in closed form per root configuration). The
/// `terms` convention is Σ term = 0 with the forcing collected on the left,
/// so g = -forcing. Returns None when the quadratures don't close.
fn variation_of_parameters(
  coeffs: &[f64],
  roots: &[(f64, f64, usize)],
  forcing: &Expr,
  x_name: &str,
) -> Option<Expr> {
  if coeffs.len() != 3 {
    return None;
  }
  let x = || id_expr(x_name);
  // E^(r x), reduced to 1 for r == 0.
  let exp_rx = |r: f64| -> Expr {
    if r.abs() < 1e-10 {
      Expr::Integer(1)
    } else {
      make_exp_term(r, x_name)
    }
  };
  let mul = |a: Expr, b: Expr| -> Expr {
    match (&a, &b) {
      (Expr::Integer(1), _) => b,
      (_, Expr::Integer(1)) => a,
      _ => times2(a, b),
    }
  };

  // Fundamental pair and Wronskian y1 y2' - y1' y2 per root configuration.
  let (y1, y2, wronskian) = match roots {
    [(r, im, 2)] if im.abs() < 1e-10 => {
      // Double root r: {E^(r x), x E^(r x)}, W = E^(2 r x)
      (exp_rx(*r), mul(x(), exp_rx(*r)), exp_rx(2.0 * r))
    }
    [(r1, im1, 1), (r2, im2, 1)] if im1.abs() < 1e-10 && im2.abs() < 1e-10 => {
      // Distinct real roots: {E^(r1 x), E^(r2 x)}, W = (r2 - r1) E^((r1+r2) x)
      (
        exp_rx(*r1),
        exp_rx(*r2),
        mul(f64_to_nice_expr(r2 - r1), exp_rx(r1 + r2)),
      )
    }
    [(alpha, beta, 1)] if *beta > 1e-10 => {
      // Complex pair α ± iβ: {E^(α x) Cos[β x], E^(α x) Sin[β x]},
      // W = β E^(2 α x)
      let cos_t = make_trig_term("Cos", *beta, x_name);
      let sin_t = make_trig_term("Sin", *beta, x_name);
      (
        mul(exp_rx(*alpha), cos_t),
        mul(exp_rx(*alpha), sin_t),
        mul(f64_to_nice_expr(*beta), exp_rx(2.0 * alpha)),
      )
    }
    _ => return None,
  };

  // g = -forcing (terms sum to zero), scaled by the leading coefficient and
  // the Wronskian: integrands are y_i * g / (a2 W).
  let g = negate_expr(forcing);
  let a2 = f64_to_nice_expr(coeffs[2]);
  let integrate_with = |y: &Expr| -> Option<Expr> {
    let integrand = div2(
      times2(y.clone(), g.clone()),
      mul(a2.clone(), wronskian.clone()),
    );
    // Canonicalize first so the integrator sees simplified products
    // (e.g. Cos[x]*Tan[x] -> Sin[x]).
    let integrand =
      crate::evaluator::evaluate_expr_to_expr(&integrand).unwrap_or(integrand);
    let result = crate::functions::calculus_ast::integrate_ast(&[
      integrand,
      id_expr(x_name),
    ])
    .ok()?;
    if contains_function_call(&result, "Integrate") {
      return None;
    }
    Some(result)
  };
  let int1 = integrate_with(&y2)?;
  let int2 = integrate_with(&y1)?;

  // y_p = -y1 ∫ y2 g/(a2 W) + y2 ∫ y1 g/(a2 W); expand so products like
  // -Cos (ArcTanh[Sin] - Sin) + Sin (-Cos) cancel across the two halves.
  let y_p = plus2(times2(negate_expr(&y1), int1), times2(y2, int2));
  let y_p = crate::evaluator::evaluate_function_call_ast(
    "Expand",
    std::slice::from_ref(&y_p),
  )
  .unwrap_or(y_p);
  // Sums of sinusoidal forcing terms leave products of trig functions behind;
  // keep whichever of the plain, simplified and trig-reduced forms is shortest.
  let candidates = [y_p.clone(), call1("TrigReduce", y_p.clone())]
    .into_iter()
    .filter_map(|e| crate::evaluator::evaluate_expr_to_expr(&e).ok())
    .flat_map(|e| {
      let simplified = crate::evaluator::evaluate_function_call_ast(
        "Simplify",
        std::slice::from_ref(&e),
      )
      .unwrap_or_else(|_| e.clone());
      [e, simplified]
    });
  candidates.min_by_key(|e| expr_to_string(e).len())
}

/// Whether the expression contains a call to the named function anywhere.
fn contains_function_call(expr: &Expr, fname: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == fname || args.iter().any(|a| contains_function_call(a, fname))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_function_call(left, fname)
        || contains_function_call(right, fname)
    }
    Expr::UnaryOp { operand, .. } => contains_function_call(operand, fname),
    Expr::List(items) => items.iter().any(|a| contains_function_call(a, fname)),
    Expr::CurriedCall { func, args } => {
      contains_function_call(func, fname)
        || args.iter().any(|a| contains_function_call(a, fname))
    }
    _ => false,
  }
}

/// Find roots of characteristic polynomial
fn find_characteristic_roots(
  coeffs: &[f64],
  max_order: usize,
) -> Result<Vec<(f64, f64, usize)>, InterpreterError> {
  // Returns (real_part, imag_part, multiplicity)
  let leading = coeffs[max_order];
  if leading.abs() < 1e-15 {
    return Err(InterpreterError::EvaluationError(
      "DSolve: leading coefficient is zero".into(),
    ));
  }

  match max_order {
    1 => {
      // a_1*r + a_0 = 0 → r = -a_0/a_1
      let r = -coeffs[0] / coeffs[1];
      Ok(vec![(r, 0.0, 1)])
    }
    2 => {
      // a_2*r^2 + a_1*r + a_0 = 0
      let a = coeffs[2];
      let b = coeffs[1];
      let c = coeffs[0];
      let disc = b * b - 4.0 * a * c;
      if disc > 1e-10 {
        let ra = (-b + disc.sqrt()) / (2.0 * a);
        let rb = (-b - disc.sqrt()) / (2.0 * a);
        // Match wolframscript's constant labeling: the fundamental solutions
        // (and hence C[1], C[2]) are ordered by ascending root, except for a
        // pair symmetric about zero (r, -r), where the positive root leads.
        let (first, second) = if (ra + rb).abs() < 1e-10 {
          if ra >= rb { (ra, rb) } else { (rb, ra) }
        } else if ra <= rb {
          (ra, rb)
        } else {
          (rb, ra)
        };
        Ok(vec![(first, 0.0, 1), (second, 0.0, 1)])
      } else if disc.abs() <= 1e-10 {
        let r = -b / (2.0 * a);
        Ok(vec![(r, 0.0, 2)])
      } else {
        let real = -b / (2.0 * a);
        let imag = (-disc).sqrt() / (2.0 * a);
        Ok(vec![(real, imag, 1)])
      }
    }
    3 => Ok(solve_cubic_characteristic(coeffs)),
    4 => Ok(solve_quartic_characteristic(coeffs)),
    _ => {
      // For higher orders, try numerical root finding
      Err(InterpreterError::EvaluationError(format!(
        "DSolve: order {max_order} constant-coefficient ODEs not supported"
      )))
    }
  }
}

/// Solve cubic characteristic polynomial
fn solve_cubic_characteristic(
  coeffs: &[f64],
) -> std::vec::Vec<(f64, f64, usize)> {
  let a = coeffs[3];
  let b = coeffs[2];
  let c = coeffs[1];
  let d = coeffs[0];

  // Normalize: r^3 + pr^2 + qr + s = 0
  let p = b / a;
  let q = c / a;
  let s = d / a;

  // Depressed cubic: t^3 + pt2 + q2 = 0 where r = t - p/3
  let p2 = q - p * p / 3.0;
  let q2 = 2.0 * p * p * p / 27.0 - p * q / 3.0 + s;

  let disc = q2 * q2 / 4.0 + p2 * p2 * p2 / 27.0;

  let mut roots = Vec::new();

  if disc > 1e-10 {
    // One real root, two complex conjugates
    let sqrt_disc = disc.sqrt();
    let u = (-q2 / 2.0 + sqrt_disc).cbrt();
    let v = (-q2 / 2.0 - sqrt_disc).cbrt();
    let r1 = u + v - p / 3.0;
    roots.push((r1, 0.0, 1));

    let real_part = -(u + v) / 2.0 - p / 3.0;
    let imag_part = (u - v) * 3.0_f64.sqrt() / 2.0;
    if imag_part.abs() > 1e-10 {
      roots.push((real_part, imag_part.abs(), 1));
    } else {
      roots.push((real_part, 0.0, 1));
      roots.push((real_part, 0.0, 1));
    }
  } else if disc.abs() <= 1e-10 {
    // All real, at least two equal
    if p2.abs() < 1e-10 && q2.abs() < 1e-10 {
      roots.push((-p / 3.0, 0.0, 3));
    } else {
      let u = if q2 > 0.0 {
        -(q2 / 2.0).cbrt()
      } else {
        (-q2 / 2.0).cbrt()
      };
      roots.push((2.0 * u - p / 3.0, 0.0, 1));
      roots.push((-u - p / 3.0, 0.0, 2));
    }
  } else {
    // Three distinct real roots (casus irreducibilis)
    let r = (-p2 * p2 * p2 / 27.0).sqrt();
    let theta = (-q2 / (2.0 * r)).acos();
    let m = 2.0 * (r.cbrt());
    roots.push((m * (theta / 3.0).cos() - p / 3.0, 0.0, 1));
    roots.push((
      m * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - p / 3.0,
      0.0,
      1,
    ));
    roots.push((
      m * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos() - p / 3.0,
      0.0,
      1,
    ));
  }

  roots
}

/// Solve quartic characteristic polynomial
/// Roots of the depressed biquadratic `y^4 + p*y^2 + r = 0` (`x = y +
/// shift`), via the quadratic `u^2 + p*u + r = 0` in `u = y^2` followed by
/// a complex square root of each `u` root. See the comment at its call
/// site in [`solve_quartic_characteristic`] for why this is needed instead
/// of always going through the general resolvent-cubic method.
fn solve_biquadratic(
  p: f64,
  r: f64,
  shift: f64,
) -> std::vec::Vec<(f64, f64, usize)> {
  let disc = p * p - 4.0 * r;
  let u_roots: [(f64, f64); 2] = if disc >= 0.0 {
    let s = disc.sqrt();
    [(f64::midpoint(-p, s), 0.0), ((-p - s) / 2.0, 0.0)]
  } else {
    let s = (-disc).sqrt();
    [(-p / 2.0, s / 2.0), (-p / 2.0, -s / 2.0)]
  };

  let mut roots = Vec::with_capacity(4);
  for &(ure, uim) in &u_roots {
    if uim.abs() < 1e-12 {
      // Real u: y = ±sqrt(u), real if u >= 0, else a conjugate pair on the
      // imaginary axis.
      if ure >= 0.0 {
        let s = ure.sqrt();
        roots.push((s + shift, 0.0, 1));
        roots.push((-s + shift, 0.0, 1));
      } else {
        let s = (-ure).sqrt();
        roots.push((shift, s, 1));
      }
    } else if uim > 0.0 {
      // Complex-conjugate u pair: y = ±sqrt(u) for the positive-imaginary
      // twin gives the two conjugate-pair representatives directly (see
      // the call site comment for the derivation).
      let (sre, sim) = complex_sqrt(ure, uim);
      roots.push((sre + shift, sim, 1));
      roots.push((-sre + shift, sim, 1));
    }
    // uim < 0 is the conjugate of the pair already handled above; skip.
  }
  roots
}

/// Principal square root of a complex number, matching the sign of `im` in
/// the result (so `im < 0` gives a result with negative imaginary part).
fn complex_sqrt(re: f64, im: f64) -> (f64, f64) {
  let modulus = re.hypot(im);
  let sre = f64::midpoint(modulus, re).max(0.0).sqrt();
  let mut sim = ((modulus - re) / 2.0).max(0.0).sqrt();
  if im < 0.0 {
    sim = -sim;
  }
  (sre, sim)
}

fn solve_quartic_characteristic(
  coeffs: &[f64],
) -> std::vec::Vec<(f64, f64, usize)> {
  let a = coeffs[4];
  let b = coeffs[3] / a;
  let c = coeffs[2] / a;
  let d = coeffs[1] / a;
  let e = coeffs[0] / a;

  // Depressed quartic: y^4 + py^2 + qy + r = 0 where x = y - b/4
  let p = c - 3.0 * b * b / 8.0;
  let q = b * b * b / 8.0 - b * c / 2.0 + d;
  let r = -3.0 * b * b * b * b / 256.0 + b * b * c / 16.0 - b * d / 4.0 + e;
  let quartic_shift = -b / 4.0;

  // Biquadratic special case (q == 0, e.g. any quartic with no odd-power
  // terms — the common shape a real coupled linear ODE system's
  // characteristic polynomial takes): the general resolvent-cubic method
  // below is numerically fragile right where it matters most here, because
  // a biquadratic's resolvent cubic is itself degenerate (a repeated or
  // near-repeated root), which the cubic solver's branch selection can
  // land on the wrong side of by a hair — turning what should be a purely
  // imaginary conjugate quadruple into a spuriously real one. Solving
  // y^4 + py^2 + r = 0 directly as a quadratic in u = y^2 sidesteps that
  // entirely and is exact.
  if q.abs() < 1e-9 * (p.abs() + r.abs() + 1.0) {
    return solve_biquadratic(p, r, quartic_shift);
  }

  // Solve resolvent cubic: m^3 - p/2 * m^2 - r*m + (p*r/2 - q^2/8) = 0
  let resolvent_coeffs = vec![p * r / 2.0 - q * q / 8.0, -r, -p / 2.0, 1.0];

  let cubic_roots = solve_cubic_characteristic(&resolvent_coeffs);
  // Pick a real root
  let m = cubic_roots
    .iter()
    .find(|(_, im, _)| im.abs() < 1e-10)
    .map_or(cubic_roots[0].0, |(re, _, _)| *re);

  let disc1 = 2.0 * m - p;
  let mut roots = Vec::new();
  let shift = -b / 4.0;

  if disc1 > 1e-10 {
    let sqrt_disc1 = disc1.sqrt();
    // Two quadratics
    let disc2a = -(2.0 * m + p + q / sqrt_disc1);
    let disc2b = -(2.0 * m + p - q / sqrt_disc1);

    if disc2a >= -1e-10 {
      let s = disc2a.max(0.0).sqrt();
      roots.push((f64::midpoint(sqrt_disc1, s) + shift, 0.0, 1));
      roots.push(((sqrt_disc1 - s) / 2.0 + shift, 0.0, 1));
    } else {
      let s = (-disc2a).sqrt();
      roots.push((sqrt_disc1 / 2.0 + shift, s / 2.0, 1));
    }

    if disc2b >= -1e-10 {
      let s = disc2b.max(0.0).sqrt();
      roots.push((f64::midpoint(-sqrt_disc1, s) + shift, 0.0, 1));
      roots.push(((-sqrt_disc1 - s) / 2.0 + shift, 0.0, 1));
    } else {
      let s = (-disc2b).sqrt();
      roots.push((-sqrt_disc1 / 2.0 + shift, s / 2.0, 1));
    }
  } else if disc1.abs() <= 1e-10 {
    // m is a double root of the resolvent
    let disc2 = m * m - r;
    if disc2 >= -1e-10 {
      let s = disc2.max(0.0).sqrt();
      roots.push(((m + s).sqrt() + shift, 0.0, 1));
      roots.push((-(m + s).sqrt() + shift, 0.0, 1));
      roots.push(((m - s).sqrt() + shift, 0.0, 1));
      roots.push((-(m - s).sqrt() + shift, 0.0, 1));
    } else {
      // Complex roots
      let s = (-disc2).sqrt();
      let mod_val = (m * m + disc2.abs()).sqrt().sqrt();
      let angle = s.atan2(m) / 2.0;
      roots.push((mod_val * angle.cos() + shift, mod_val * angle.sin(), 1));
      roots.push((-mod_val * angle.cos() + shift, -mod_val * angle.sin(), 1));
    }
  } else {
    // disc1 < 0: complex scenario
    let sqrt_disc1 = (-disc1).sqrt();
    roots.push((shift, sqrt_disc1 / 2.0, 1));
    roots.push((shift, -sqrt_disc1 / 2.0, 1));
  }

  roots
}

/// Build homogeneous solution from characteristic roots
/// `a2 y'' + a1 y' + a0 y + f == 0` with symbolic constant coefficients
/// (and a constant forcing `f`), in wolframscript's forms. The roots come
/// from the quadratic formula, a square root taken with `PowerExpand`
/// (`Sqrt[k^2]` is `k`) and a negative-looking radicand turned into `I`
/// times the root of its negation. Without a first-order term a pair `±I s`
/// gives `C[1] Cos[s x] + C[2] Sin[s x]` and a pair `±s` gives
/// `E^(s x) C[1] + E^(-s x) C[2]`; otherwise `E^(r1 x) C[1] + E^(r2 x) C[2]`
/// with the minus-sign root first, or `x E^(r x)` for a double root.
fn solve_symbolic_second_order(
  terms: &[OdeTerm],
  x_name: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let coefficient = |order: i32| -> Result<Expr, InterpreterError> {
    eval(&plus(
      terms
        .iter()
        .filter(|t| t.order == order)
        .map(|t| t.coefficient.clone())
        .chain(std::iter::once(Expr::Integer(0)))
        .collect(),
    ))
  };
  let (a2, a1, a0, f) = (
    coefficient(2)?,
    coefficient(1)?,
    coefficient(0)?,
    coefficient(-1)?,
  );
  let x = id_expr(x_name);
  let c = |i: i128| call1("C", Expr::Integer(i));
  let i_unit = id_expr("I");
  let simplify = |e: Expr| eval(&call1("Simplify", e));
  let power_expand_sqrt =
    |e: Expr| eval(&call1("PowerExpand", call1("Sqrt", e)));
  // `Some(p)` when `e` reads as `-p`.
  let negated = |e: &Expr| crate::functions::math_ast::strip_negation(e);
  let exp_of = |root: &Expr| call("Exp", vec![times2(root.clone(), x.clone())]);
  let disc = simplify(plus2(
    pow2(a1.clone(), Expr::Integer(2)),
    times(vec![Expr::Integer(-4), a2.clone(), a0.clone()]),
  ))?;
  let homogeneous = if matches!(disc, Expr::Integer(0)) {
    let root = simplify(div2(
      times2(Expr::Integer(-1), a1),
      times2(Expr::Integer(2), a2),
    ))?;
    plus2(
      times2(exp_of(&root), c(1)),
      times(vec![x.clone(), exp_of(&root), c(2)]),
    )
  } else if matches!(a1, Expr::Integer(0)) {
    // r^2 == -a0/a2.
    let n = simplify(div2(a0.clone(), a2.clone()))?;
    if let Some(p) = negated(&n) {
      let s_real = power_expand_sqrt(p)?;
      plus2(
        times2(exp_of(&s_real), c(1)),
        times2(exp_of(&times2(Expr::Integer(-1), s_real)), c(2)),
      )
    } else {
      let s_im = power_expand_sqrt(n)?;
      plus2(
        times2(c(1), call1("Cos", times2(s_im.clone(), x.clone()))),
        times2(c(2), call1("Sin", times2(s_im, x.clone()))),
      )
    }
  } else {
    let sq = match negated(&disc) {
      Some(p) => times2(i_unit, power_expand_sqrt(p)?),
      None => call1("Sqrt", disc),
    };
    let two_a2 = times2(Expr::Integer(2), a2);
    let minus_a1 = times2(Expr::Integer(-1), a1);
    let r1 = simplify(div2(
      plus2(minus_a1.clone(), times2(Expr::Integer(-1), sq.clone())),
      two_a2.clone(),
    ))?;
    let r2 = simplify(div2(plus2(minus_a1, sq), two_a2))?;
    plus2(times2(exp_of(&r1), c(1)), times2(exp_of(&r2), c(2)))
  };
  // A constant forcing: the particular solution -f/a0.
  let solution = if matches!(f, Expr::Integer(0)) {
    homogeneous
  } else if matches!(a0, Expr::Integer(0)) {
    return Ok(None);
  } else {
    plus2(times2(Expr::Integer(-1), div2(f, a0)), homogeneous)
  };
  Ok(Some(eval(&solution)?))
}

fn build_homogeneous_solution(
  roots: &[(f64, f64, usize)],
  x_name: &str,
) -> Expr {
  let x = id_expr(x_name);
  let mut terms: Vec<Expr> = Vec::new();
  let mut c_idx = 1usize;

  for (real, imag, mult) in roots {
    if imag.abs() < 1e-10 {
      // Real root with multiplicity
      for k in 0..*mult {
        let c_k = make_c(c_idx);
        c_idx += 1;

        let mut term = c_k;

        // Multiply by x^k for repeated roots
        if k > 0 {
          let x_power = if k == 1 {
            x.clone()
          } else {
            pow2(x.clone(), Expr::Integer(k as i128))
          };
          term = times2(x_power, term);
        }

        // Multiply by E^(r*x) if r != 0
        if real.abs() > 1e-10 {
          let exp_term = make_exp_term(*real, x_name);
          term = times2(exp_term, term);
        }

        terms.push(term);
      }
    } else if *imag > 0.0 {
      // Complex roots α ± iβ
      // E^(α*x) * (C[n]*Cos[β*x] + C[n+1]*Sin[β*x])
      let c1 = make_c(c_idx);
      c_idx += 1;
      let c2 = make_c(c_idx);
      c_idx += 1;

      let cos_term = times2(c1, make_trig_term("Cos", *imag, x_name));
      let sin_term = times2(c2, make_trig_term("Sin", *imag, x_name));

      let trig_sum = plus2(cos_term, sin_term);

      let term = if real.abs() > 1e-10 {
        let exp_term = make_exp_term(*real, x_name);
        times2(exp_term, trig_sum)
      } else {
        trig_sum
      };

      terms.push(term);
    }
    // Skip negative imaginary parts (conjugate pairs handled together)
  }

  if terms.is_empty() {
    return Expr::Integer(0);
  }
  if terms.len() == 1 {
    return terms.into_iter().next().unwrap();
  }

  // Sum all terms
  let mut result = terms.remove(0);
  for term in terms {
    result = plus2(result, term);
  }
  result
}

/// Create C[n] constant expression
fn make_c(n: usize) -> Expr {
  call1("C", Expr::Integer(n as i128))
}

/// Create E^(r*x) expression, simplifying for special values
fn make_exp_term(r: f64, x_name: &str) -> Expr {
  let x = id_expr(x_name);
  let r_expr = f64_to_nice_expr(r);
  let exponent = if matches!(&r_expr, Expr::Integer(1)) {
    x
  } else {
    times2(r_expr, x)
  };
  pow2(const_expr("E"), exponent)
}

/// Create Cos[β*x] or Sin[β*x] expression
fn make_trig_term(func: &str, beta: f64, x_name: &str) -> Expr {
  let x = id_expr(x_name);
  let beta_expr = f64_to_nice_expr(beta);
  let arg = if matches!(&beta_expr, Expr::Integer(1)) {
    x
  } else {
    times2(beta_expr, x)
  };
  call1(func, arg)
}

/// Convert f64 to a nice Expr: integer if whole, fraction if rational, otherwise Real
fn f64_to_nice_expr(f: f64) -> Expr {
  if f == f.round() && f.abs() < 1e15 {
    return Expr::Integer(f as i128);
  }
  // Try simple fractions
  for denom in 2..=12 {
    let numer = f * denom as f64;
    if (numer - numer.round()).abs() < 1e-10 {
      let (n, d) = (numer.round() as i128, denom as i128);
      let (nn, dd) = rat_reduce(n, d);
      if dd == 1 {
        return Expr::Integer(nn);
      }
      return div2(Expr::Integer(nn), Expr::Integer(dd));
    }
  }
  // Try sqrt expressions: check if f^2 is a nice rational. A negative f is
  // the negated square root of the same rational (-Sqrt[3], not the raw
  // -1.7320508075688772).
  let f2 = f * f;
  for denom in 1..=12 {
    let numer = f2 * denom as f64;
    if (numer - numer.round()).abs() < 1e-10 {
      let n = numer.round() as i128;
      let d = denom as i128;
      // |f| = Sqrt[n/d]
      let magnitude = if d == 1 {
        make_sqrt(Expr::Integer(n))
      } else {
        div2(make_sqrt(Expr::Integer(n)), make_sqrt(Expr::Integer(d)))
      };
      return if f < 0.0 {
        times2(Expr::Integer(-1), magnitude)
      } else {
        magnitude
      };
    }
  }
  Expr::Real(f)
}

// ─── First-order Linear ODE Solver ─────────────────────────────────────

/// Solve first-order linear ODE: y' + P(x)*y = Q(x)
fn solve_first_order_linear(
  terms: &[OdeTerm],
  x_name: &str,
) -> Result<Expr, InterpreterError> {
  // Collect coefficients: a1*y' + a0*y + forcing = 0
  let mut a1 = Expr::Integer(0);
  let mut a0 = Expr::Integer(0);
  let mut forcing = Expr::Integer(0);

  for term in terms {
    match term.order {
      1 => {
        a1 = plus2(a1, term.coefficient.clone());
      }
      0 => {
        a0 = plus2(a0, term.coefficient.clone());
      }
      -1 => {
        forcing = plus2(forcing, term.coefficient.clone());
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "DSolve: unexpected term order in first-order ODE".into(),
        ));
      }
    }
  }

  let a1 = crate::evaluator::evaluate_expr_to_expr(&a1).unwrap_or(a1);
  let a0 = crate::evaluator::evaluate_expr_to_expr(&a0).unwrap_or(a0);
  let forcing =
    crate::evaluator::evaluate_expr_to_expr(&forcing).unwrap_or(forcing);

  // Normalize: y' + P(x)*y = Q(x)
  // P(x) = a0/a1, Q(x) = -forcing/a1
  let p_expr = crate::functions::calculus_ast::simplify(div2(a0, a1.clone()));
  let q_expr =
    crate::functions::calculus_ast::simplify(div2(negate_expr(&forcing), a1));

  // Check for special cases

  // Case 1: y' = f(x) (P=0, Q=f(x))
  let p_is_zero = matches!(&p_expr, Expr::Integer(0))
    || matches!(&p_expr, Expr::Real(f) if f.abs() < 1e-15);
  if p_is_zero {
    // y = ∫Q(x)dx + C[1]
    let integral = crate::functions::calculus_ast::integrate_ast(&[
      q_expr,
      id_expr(x_name),
    ])?;
    return Ok(plus2(integral, make_c(1)));
  }

  // Case 2: y' + a*y = 0 (constant coefficient, homogeneous)
  let q_is_zero = matches!(&q_expr, Expr::Integer(0))
    || matches!(&q_expr, Expr::Real(f) if f.abs() < 1e-15);
  if q_is_zero
    && crate::functions::calculus_ast::is_constant_wrt(&p_expr, x_name)
  {
    // y = E^(-a*x)*C[1]
    let neg_p = negate_expr(&p_expr);
    let exp_term = make_exp_term_expr(&neg_p, x_name);
    return Ok(times2(exp_term, make_c(1)));
  }

  // Case 3: General integrating factor method
  // μ(x) = E^(∫P(x)dx)
  // y = (1/μ) * (∫μ*Q(x)dx + C[1])
  let p_integral = crate::functions::calculus_ast::integrate_ast(&[
    p_expr.clone(),
    id_expr(x_name),
  ])?;

  let mu = pow2(const_expr("E"), p_integral.clone());

  let mu_q =
    crate::functions::calculus_ast::simplify(times2(mu.clone(), q_expr));

  let mu_q_integral =
    crate::functions::calculus_ast::integrate_ast(&[mu_q, id_expr(x_name)])?;

  // y = E^(-∫P dx) * (∫(μ*Q)dx + C[1])
  let neg_p_integral = negate_expr(&p_integral);
  let inv_mu = pow2(const_expr("E"), neg_p_integral);

  // Distribute the integrating factor over the particular part and the
  // constant separately, matching wolframscript's form, e.g.
  // (E^(3x)/3 + C[1])/E^(2x) -> E^x/3 + C[1]/E^(2x). Each product is simplified
  // (not fully expanded), so a grouped particular like (-Cos[x]+3Sin[x])/10
  // stays grouped rather than splitting into separate terms.
  let mu_q_integral = crate::evaluator::evaluate_expr_to_expr(&mu_q_integral)
    .unwrap_or(mu_q_integral);
  let particular = match &mu_q_integral {
    // A sum of several antiderivative terms: scale each one so the
    // exponentials cancel term by term instead of staying nested.
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let scaled: Vec<Expr> = args
        .iter()
        .map(|t| {
          crate::functions::calculus_ast::simplify(times2(
            inv_mu.clone(),
            t.clone(),
          ))
        })
        .collect();
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Plus".to_string(),
        args: scaled.into(),
      })?
    }
    _ => crate::functions::calculus_ast::simplify(times2(
      inv_mu.clone(),
      mu_q_integral,
    )),
  };
  let homogeneous =
    crate::functions::calculus_ast::simplify(times2(inv_mu, make_c(1)));
  Ok(plus2(particular, homogeneous))
}

/// Reduction of order for `a2(x) y'' + a1(x) y' + f(x) == 0`: solve the first
/// order linear equation for `v = y'` (introducing `C[1]`), then integrate
/// (introducing `C[2]`). `None` when the integral stays unevaluated.
fn solve_second_order_without_unknown(
  terms: &[OdeTerm],
  x_name: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let reduced: Vec<OdeTerm> = terms
    .iter()
    .map(|t| OdeTerm {
      order: if t.order >= 1 { t.order - 1 } else { t.order },
      coefficient: t.coefficient.clone(),
    })
    .collect();
  let v = solve_first_order_linear(&reduced, x_name)?;
  let integral =
    crate::functions::calculus_ast::integrate_ast(&[v, id_expr(x_name)])?;
  if matches!(&integral, Expr::FunctionCall { name, .. } if name == "Integrate")
  {
    return Ok(None);
  }
  Ok(Some(plus2(integral, make_c(2))))
}

// ─── Separable First-Order ODEs ────────────────────────────────────────

/// Solve a separable first-order ODE `y'[x] == g(x) h(y[x])` by quadrature.
///
/// Only the initial-value problem is answered: the general solution needs an
/// arbitrary constant, and where wolframscript places it inside the solved
/// form is not something Woxi reproduces yet, so `DSolve` keeps returning the
/// equation unevaluated when no initial condition pins the constant.
fn solve_separable_first_order(
  ode: &Expr,
  y_name: &str,
  x_name: &str,
  initial_conditions: &[Expr],
) -> Option<Expr> {
  let rhs = separable_derivative_rhs(ode, y_name, x_name)?;
  let (x0, y0) = separable_initial_value(initial_conditions, y_name)?;

  let u_name = fresh_symbol_name(ode, "u")?;
  let rhs_u =
    replace_y_call(&rhs, y_name, x_name, &Expr::Identifier(u_name.clone()));
  if !is_free_of_y(&rhs_u, y_name) {
    return None;
  }
  let (g, h) = split_separable_factors(&rhs_u, &u_name, x_name)?;

  // Separating gives ∫ du/h(u) == ∫ g(x) dx + C.
  let y_integral = closed_form_integral(&div2(Expr::Integer(1), h), &u_name)?;
  let x_integral = closed_form_integral(&g, x_name)?;

  let constant = simplify(minus2(
    crate::syntax::substitute_variable(&y_integral, &u_name, &y0),
    crate::syntax::substitute_variable(&x_integral, x_name, &x0),
  ));
  let relation = Expr::Comparison {
    operands: vec![y_integral, plus2(x_integral, constant)],
    operators: vec![ComparisonOp::Equal],
  };

  let solved =
    crate::functions::solve_ast(&[relation, Expr::Identifier(u_name.clone())])
      .ok()?;

  // Separating loses branch information, so more than one root can come back
  // (`y^2` on the right gives a ± pair). Keep the one the initial condition
  // actually selects.
  let Expr::List(solution_sets) = &solved else {
    return None;
  };
  for set in solution_sets {
    let Expr::List(rules) = set else { continue };
    let Some(Expr::Rule {
      pattern,
      replacement,
    }) = rules.get(0)
    else {
      continue;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(n) if n == &u_name) {
      continue;
    }
    let candidate = simplify(replacement.as_ref().clone());
    let at_x0 = crate::evaluator::evaluate_expr_to_expr(
      &crate::syntax::substitute_variable(&candidate, x_name, &x0),
    )
    .ok()?;
    if values_agree(&at_x0, &y0) {
      return Some(candidate);
    }
  }
  None
}

/// Read `y'[x] == rhs` (in either orientation) and return the right-hand side.
fn separable_derivative_rhs(
  ode: &Expr,
  y_name: &str,
  x_name: &str,
) -> Option<Expr> {
  let (lhs, rhs) = as_equal_pair(ode)?;
  for (derivative, other) in [(lhs, rhs), (rhs, lhs)] {
    if let Some((1, point)) =
      extract_derivative_order_and_point(derivative, y_name)
      && matches!(&point, Expr::Identifier(n) if n == x_name)
      && !contains_derivative(other)
    {
      return Some(other.clone());
    }
  }
  None
}

/// Find an initial condition `y[x0] == y0`, returning the pair `(x0, y0)`.
fn separable_initial_value(
  initial_conditions: &[Expr],
  y_name: &str,
) -> Option<(Expr, Expr)> {
  for condition in initial_conditions {
    let Some((lhs, rhs)) = as_equal_pair(condition) else {
      continue;
    };
    if let Expr::FunctionCall { name, args } = lhs
      && name == y_name
      && args.len() == 1
    {
      return Some((args[0].clone(), rhs.clone()));
    }
  }
  None
}

/// Pick a symbol name that does not occur in `expr`, so the dependent
/// variable can be stood in for by a plain symbol while integrating.
fn fresh_symbol_name(expr: &Expr, base: &str) -> Option<String> {
  (0..64).find_map(|n| {
    let candidate = if n == 0 {
      base.to_string()
    } else {
      format!("{base}{n}")
    };
    crate::functions::calculus_ast::is_constant_wrt(expr, &candidate)
      .then_some(candidate)
  })
}

/// Replace every `y[x]` in `expr` with `value`.
fn replace_y_call(
  expr: &Expr,
  y_name: &str,
  x_name: &str,
  value: &Expr,
) -> Expr {
  if let Expr::FunctionCall { name, args } = expr
    && name == y_name
    && args.len() == 1
    && matches!(&args[0], Expr::Identifier(n) if n == x_name)
  {
    return value.clone();
  }
  map_children(expr, &|child| replace_y_call(child, y_name, x_name, value))
}

/// Split a product into its `x`-only and `u`-only halves. `None` when some
/// factor mixes the two, which means the equation is not separable this way.
fn split_separable_factors(
  rhs: &Expr,
  u_name: &str,
  x_name: &str,
) -> Option<(Expr, Expr)> {
  let mut factors = Vec::new();
  collect_multiplicative_factors(rhs, &mut factors);

  let mut x_part = Vec::new();
  let mut u_part = Vec::new();
  for factor in factors {
    if crate::functions::calculus_ast::is_constant_wrt(&factor, u_name) {
      x_part.push(factor);
    } else if crate::functions::calculus_ast::is_constant_wrt(&factor, x_name) {
      u_part.push(factor);
    } else {
      return None;
    }
  }
  Some((product_of(x_part), product_of(u_part)))
}

fn collect_multiplicative_factors(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_multiplicative_factors(left, out);
      collect_multiplicative_factors(right, out);
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      collect_multiplicative_factors(left, out);
      out.push(pow2(right.as_ref().clone(), Expr::Integer(-1)));
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      out.push(Expr::Integer(-1));
      collect_multiplicative_factors(operand, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for arg in args {
        collect_multiplicative_factors(arg, out);
      }
    }
    _ => out.push(expr.clone()),
  }
}

fn product_of(factors: Vec<Expr>) -> Expr {
  factors
    .into_iter()
    .reduce(times2)
    .unwrap_or(Expr::Integer(1))
}

/// Integrate, accepting only a result that actually closed — a residual
/// `Integrate[…]` head means the antiderivative is not available and the
/// separable route cannot finish.
fn closed_form_integral(integrand: &Expr, var: &str) -> Option<Expr> {
  let result = crate::functions::calculus_ast::integrate_ast(&[
    simplify(integrand.clone()),
    id_expr(var),
  ])
  .ok()?;
  (!mentions_head(&result, "Integrate")).then_some(result)
}

/// Whether `expr` calls a function the user defined (has `DownValues`).
fn calls_user_function(expr: &Expr) -> bool {
  if let Expr::FunctionCall { name, .. } = expr
    && crate::FUNC_DEFS.with(|m| m.borrow().get(name).is_some())
  {
    return true;
  }
  expr_children(expr).iter().any(|c| calls_user_function(c))
}

fn mentions_head(expr: &Expr, head: &str) -> bool {
  if matches!(expr, Expr::FunctionCall { name, .. } if name == head) {
    return true;
  }
  expr_children(expr).iter().any(|c| mentions_head(c, head))
}

/// Compare two solution values, numerically where both sides evaluate to a
/// number and structurally otherwise.
fn values_agree(a: &Expr, b: &Expr) -> bool {
  match (expr_to_f64(a), expr_to_f64(b)) {
    (Ok(x), Ok(y)) => (x - y).abs() <= 1e-9 * y.abs().max(1.0),
    _ => exprs_match(a, b),
  }
}

/// Create E^(expr*x) for symbolic expressions
fn make_exp_term_expr(coeff: &Expr, x_name: &str) -> Expr {
  let x = id_expr(x_name);
  let exponent = match coeff {
    Expr::Integer(1) => x,
    Expr::Integer(-1) => neg1(x),
    _ => times2(coeff.clone(), x),
  };
  pow2(const_expr("E"), exponent)
}

// ─── Particular Solution (Undetermined Coefficients) ───────────────────

/// Find a particular solution for constant-coefficient ODE with forcing term
fn find_particular_solution(
  coeffs: &[f64],
  _max_order: usize,
  forcing: &Expr,
  _roots: &[(f64, f64, usize)],
  x_name: &str,
) -> Option<Expr> {
  // Try to evaluate forcing as a constant
  if let Ok(val) = eval_to_f64(forcing) {
    if val.abs() < 1e-15 {
      return Some(Expr::Integer(0));
    }
    // Constant forcing: particular solution is constant c where a_0 * c = -val
    if coeffs[0].abs() > 1e-15 {
      let c = -val / coeffs[0];
      return Some(f64_to_nice_expr(c));
    }
    // If a_0 = 0 but a_1 != 0, try y_p = c*x
    if coeffs.len() > 1 && coeffs[1].abs() > 1e-15 {
      let c = -val / coeffs[1];
      return Some(times2(f64_to_nice_expr(c), id_expr(x_name)));
    }
  }
  None
}

// ─── Initial Condition Application ─────────────────────────────────────

/// Apply initial conditions to determine constants C[1], C[2], ...
fn apply_initial_conditions(
  general_solution: &Expr,
  ics: &[Expr],
  y_name: &str,
  x_name: &str,
  max_order: usize,
) -> Result<Expr, InterpreterError> {
  // Replace C[i] with placeholder identifiers for Solve compatibility
  let placeholders: Vec<String> =
    (1..=max_order).map(|i| format!("__C{i}")).collect();

  // Substitute C[i] -> __Ci in the general solution
  let mut sol_with_placeholders = general_solution.clone();
  for i in 1..=max_order {
    sol_with_placeholders = substitute_c_constant(
      &sol_with_placeholders,
      &make_c(i),
      &Expr::Identifier(placeholders[i - 1].clone()),
    );
  }

  // Build equations from initial conditions
  let mut equations = Vec::new();

  for ic in ics {
    if let Expr::Comparison {
      operands,
      operators,
    } = ic
      && operands.len() == 2
      && operators.len() == 1
      && operators[0] == ComparisonOp::Equal
    {
      let lhs = &operands[0];
      let rhs = &operands[1];

      // Determine order and point
      let (order, point) = if let Expr::FunctionCall { name, args } = lhs {
        if name == y_name && args.len() == 1 {
          (0usize, args[0].clone())
        } else if name == "Derivative" && args.len() == 3 {
          if let Expr::Integer(n) = &args[0] {
            if let Expr::Identifier(fname) = &args[1] {
              if fname == y_name {
                (*n as usize, args[2].clone())
              } else {
                continue;
              }
            } else {
              continue;
            }
          } else {
            continue;
          }
        } else {
          continue;
        }
      } else if let Some((ord, pt)) =
        extract_derivative_order_and_point(lhs, y_name)
      {
        (ord, pt)
      } else {
        continue;
      };

      // Differentiate general solution `order` times
      let mut deriv_solution = sol_with_placeholders.clone();
      for _ in 0..order {
        deriv_solution = crate::functions::calculus_ast::differentiate_expr(
          &deriv_solution,
          x_name,
        )?;
        deriv_solution =
          crate::functions::calculus_ast::simplify(deriv_solution);
        deriv_solution =
          crate::evaluator::evaluate_expr_to_expr(&deriv_solution)
            .unwrap_or(deriv_solution);
      }

      // Substitute x = point
      let substituted =
        crate::syntax::substitute_variable(&deriv_solution, x_name, &point);
      let evaluated = crate::evaluator::evaluate_expr_to_expr(&substituted)
        .unwrap_or(substituted);

      // Create equation: evaluated == rhs
      let equation = Expr::Comparison {
        operands: vec![evaluated, rhs.clone()],
        operators: vec![ComparisonOp::Equal],
      };
      equations.push(equation);
    }
  }

  if equations.len() != max_order {
    // Not enough initial conditions — return general solution
    return Ok(general_solution.clone());
  }

  // Solve for __C1, __C2, ...
  let c_id_exprs: Vec<Expr> = placeholders
    .iter()
    .map(|name| Expr::Identifier(name.clone()))
    .collect();

  let eqs_list = Expr::List(equations.into());
  let vars_list = Expr::List(c_id_exprs.into());

  let solve_result = crate::functions::solve_ast(&[eqs_list, vars_list])?;

  // Extract solutions: {{__C1 -> val1, __C2 -> val2, ...}}
  if let Expr::List(outer) = &solve_result
    && let Some(Expr::List(rules)) = outer.first()
  {
    let mut result = sol_with_placeholders.clone();
    for rule in rules {
      if let Expr::Rule {
        pattern,
        replacement,
      } = rule
        && let Expr::Identifier(var_name) = pattern.as_ref()
      {
        result =
          crate::syntax::substitute_variable(&result, var_name, replacement);
      }
    }
    // Simplify the result
    let result =
      crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result);
    return Ok(result);
  }

  // Solve failed — return general solution
  Ok(general_solution.clone())
}

/// Substitute a C[n] constant in an expression
fn substitute_c_constant(
  expr: &Expr,
  pattern: &Expr,
  replacement: &Expr,
) -> Expr {
  // Pattern is C[n], need to find and replace matching FunctionCall
  if exprs_match(expr, pattern) {
    return replacement.clone();
  }

  match expr {
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      substitute_c_constant(left, pattern, replacement),
      substitute_c_constant(right, pattern, replacement),
    ),
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_c_constant(operand, pattern, replacement)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_c_constant(a, pattern, replacement))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| substitute_c_constant(a, pattern, replacement))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// Check if two expressions are structurally equal
fn exprs_match(a: &Expr, b: &Expr) -> bool {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => x == y,
    (Expr::Real(x), Expr::Real(y)) => (x - y).abs() < 1e-15,
    (Expr::Identifier(x), Expr::Identifier(y)) => x == y,
    (Expr::Constant(x), Expr::Constant(y)) => x == y,
    (
      Expr::FunctionCall { name: n1, args: a1 },
      Expr::FunctionCall { name: n2, args: a2 },
    ) => {
      n1 == n2
        && a1.len() == a2.len()
        && a1.iter().zip(a2.iter()).all(|(x, y)| exprs_match(x, y))
    }
    _ => false,
  }
}

// ─── NDSolve RK4 Helpers ──────────────────────────────────────────────

/// Convert Expr to f64.
///
/// Shared by `NDSolve` and every other caller in this module that needs a
/// plain numeric value from an expression (`Interpolation`, `DSolve`'s
/// numeric fallbacks, …) — the error text below must stay caller-agnostic
/// rather than naming any one of them.
fn expr_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(f) => Ok(*f),
    Expr::Constant(name) if name == "E" => Ok(std::f64::consts::E),
    Expr::Constant(name) if name == "Pi" => Ok(std::f64::consts::PI),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Ok(-expr_to_f64(operand)?),
    Expr::BinaryOp { op, left, right } => {
      let l = expr_to_f64(left)?;
      let r = expr_to_f64(right)?;
      match op {
        BinaryOperator::Plus => Ok(l + r),
        BinaryOperator::Minus => Ok(l - r),
        BinaryOperator::Times => Ok(l * r),
        BinaryOperator::Divide => Ok(l / r),
        BinaryOperator::Power => Ok(l.powf(r)),
        _ => Err(InterpreterError::EvaluationError(
          "cannot convert expression to a numeric value".into(),
        )),
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let n = expr_to_f64(&args[0])?;
      let d = expr_to_f64(&args[1])?;
      Ok(n / d)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut result = 1.0;
      for arg in args {
        result *= expr_to_f64(arg)?;
      }
      Ok(result)
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut result = 0.0;
      for arg in args {
        result += expr_to_f64(arg)?;
      }
      Ok(result)
    }
    // Any other real numeric value — `-Sin[2]` from a `Piecewise` branch,
    // say — through the general numericization.
    _ => crate::functions::math_ast::try_eval_to_f64(expr).ok_or_else(|| {
      InterpreterError::EvaluationError(format!(
        "cannot convert {} to a numeric value",
        expr_to_string(expr)
      ))
    }),
  }
}

/// The value of an `{x, y}` interpolation point when it is not a number
/// (nor a list or complex number, which have their own paths).
fn symbolic_interpolation_value(row: &Expr) -> Option<&Expr> {
  match row {
    Expr::List(p) if p.len() == 2 => {
      let y = &p[1];
      // Literal numbers and lists, by far the common case (every NDSolve
      // solution), are told apart without evaluating anything.
      if matches!(
        y,
        Expr::Integer(_)
          | Expr::Real(_)
          | Expr::BigInteger(_)
          | Expr::BigFloat(..)
          | Expr::List(_)
      ) {
        return None;
      }
      let numeric = crate::functions::math_ast::try_eval_to_f64(y).is_some()
        || matches!(y, Expr::List(_))
        || matches!(
          crate::evaluator::evaluate_expr_to_expr(&call1("NumericQ", y.clone())),
          Ok(Expr::Identifier(ref t)) if t == "True"
        );
      (!numeric).then_some(y)
    }
    _ => None,
  }
}

/// `Interpolation[{{x1, y1}, …}]` whose values are not all numbers: the
/// interpolating function built for zero values (so the coordinates are
/// checked and sorted the same way) with the values put back.
fn symbolic_value_interpolation(
  args: &[Expr],
  head: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let Expr::List(given) = &args[0] else {
    return Ok(None);
  };
  // A list of values alone stands at the coordinates 1, 2, ….
  let indexed: Vec<Expr>;
  let rows: &[Expr] = if given.iter().all(|v| !matches!(v, Expr::List(_))) {
    indexed = given
      .iter()
      .enumerate()
      .map(|(i, v)| {
        Expr::List(vec![Expr::Integer(i as i128 + 1), v.clone()].into())
      })
      .collect();
    &indexed
  } else {
    given
  };
  if !rows
    .iter()
    .any(|r| symbolic_interpolation_value(r).is_some())
    || !rows.iter().all(|r| {
      matches!(r, Expr::List(p) if p.len() == 2
        && crate::functions::math_ast::try_eval_to_f64(&p[0]).is_some())
    })
  {
    return Ok(None);
  }
  let zeros: Vec<Expr> = rows
    .iter()
    .map(|r| match r {
      Expr::List(p) => Expr::List(vec![p[0].clone(), Expr::Integer(0)].into()),
      other => other.clone(),
    })
    .collect();
  let mut zero_args = args.to_vec();
  zero_args[0] = Expr::List(zeros.into());
  let function = interpolation_ast(&zero_args, head)?;
  let Expr::FunctionCall { name, args: f_args } = &function else {
    return Ok(Some(function));
  };
  let Some(Expr::List(sorted)) = f_args.get(1) else {
    return Ok(None);
  };
  // Each point gets its own value back, found by its coordinate.
  let value_at = |x: &Expr| {
    let xf = crate::functions::math_ast::try_eval_to_f64(x);
    rows.iter().find_map(|r| match r {
      Expr::List(p)
        if crate::functions::math_ast::try_eval_to_f64(&p[0]) == xf =>
      {
        Some(p[1].clone())
      }
      _ => None,
    })
  };
  let restored: Option<Vec<Expr>> = sorted
    .iter()
    .map(|r| match r {
      Expr::List(p) if p.len() == 2 => {
        value_at(&p[0]).map(|y| Expr::List(vec![p[0].clone(), y].into()))
      }
      _ => None,
    })
    .collect();
  let Some(restored) = restored else {
    return Ok(None);
  };
  let mut new_args = f_args.to_vec();
  new_args[1] = Expr::List(restored.into());
  Ok(Some(Expr::FunctionCall {
    name: name.clone(),
    args: new_args.into(),
  }))
}

/// Evaluate an expression to f64, with simplification
fn eval_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  let evaluated =
    crate::evaluator::evaluate_expr_to_expr(expr).unwrap_or(expr.clone());
  expr_to_f64(&evaluated)
}

/// Convert an interpolation data value to f64, numericising exact symbolic
/// entries (e.g. `Sin[1]`, `Pi/2`) via `N` when a direct conversion fails.
/// wolframscript's Interpolation accepts exact symbolic data by numericising
/// it, so `Interpolation[Table[{x, Sin[x]}, {x, 0, 10}]]` must work.
fn interp_value_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Real(v) => return Ok(*v),
    Expr::Integer(n) => return Ok(*n as f64),
    _ => {}
  }
  let evaluated = crate::evaluator::evaluate_expr_to_expr(expr)
    .unwrap_or_else(|_| expr.clone());
  if let Ok(v) = expr_to_f64(&evaluated) {
    return Ok(v);
  }
  // Fall back to N[...] for exact symbolic values that do not reduce to a
  // number on their own (Sin[1] stays symbolic until numericised).
  let n_expr = call1("N", evaluated);
  let numericized = crate::evaluator::evaluate_expr_to_expr(&n_expr)
    .unwrap_or_else(|_| expr.clone());
  expr_to_f64(&numericized)
}

/// Real and imaginary parts of an interpolation data value. Complex-valued
/// data (e.g. a wave function sampled on a grid) interpolates its real and
/// imaginary parts independently.
fn interp_complex_to_parts(expr: &Expr) -> Option<(f64, f64)> {
  if let Ok(v) = interp_value_to_f64(expr) {
    return Some((v, 0.0));
  }
  if let Some(parts) =
    crate::functions::math_ast::numeric_utils::try_extract_complex_f64(expr)
  {
    return Some(parts);
  }
  let evaluated = crate::evaluator::evaluate_expr_to_expr(expr)
    .unwrap_or_else(|_| expr.clone());
  let numericized =
    crate::evaluator::evaluate_expr_to_expr(&call1("N", evaluated.clone()))
      .unwrap_or(evaluated);
  crate::functions::math_ast::numeric_utils::try_extract_complex_f64(
    &numericized,
  )
}

// ─── Interpolation ─────────────────────────────────────────────────────

/// Interpolation[{y1, y2, ...}] or Interpolation[{{x1,y1}, {x2,y2}, ...}]
/// Returns InterpolatingFunction[domain, data]
pub fn interpolation_ast(
  args: &[Expr],
  head: &str,
) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(unevaluated(head, args));
  }
  if let Some(function) = symbolic_value_interpolation(args, head)? {
    return Ok(function);
  }

  // Extract InterpolationOrder option (default 3). A 2-D grid may give one
  // order per axis (`InterpolationOrder -> {orderX, orderY}`); a bare
  // integer applies to both.
  let mut interp_order: i128 = 3;
  let mut interp_orders: Option<Vec<i128>> = None;
  // `Method -> "Spline"`: an interpolating B-spline instead of local
  // polynomial pieces (1-D scalar data).
  let spline = args.iter().skip(1).any(|opt| {
    matches!(opt, Expr::Rule { pattern, replacement }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Method")
        && matches!(replacement.as_ref(), Expr::String(m) if m == "Spline"))
  });
  let data_arg = &args[0];

  let mut apply_interpolation_order = |replacement: &Expr| {
    if let Some(n) = crate::functions::math_ast::expr_to_i128(replacement) {
      interp_order = n;
    } else if let Expr::List(items) = replacement
      && let Some(list) = items
        .iter()
        .map(crate::functions::math_ast::expr_to_i128)
        .collect::<Option<Vec<i128>>>()
    {
      interp_orders = Some(list);
    }
  };
  for opt in args.iter().skip(1) {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "InterpolationOrder"
        {
          apply_interpolation_order(replacement);
        }
      }
      Expr::FunctionCall {
        name,
        args: rule_args,
      } if name == "Rule" && rule_args.len() == 2 => {
        if let Expr::Identifier(opt_name) = &rule_args[0]
          && opt_name == "InterpolationOrder"
        {
          apply_interpolation_order(&rule_args[1]);
        }
      }
      _ => {}
    }
  }

  // A positional `{{xmin, xmax}}` argument gives the coordinate domain for the
  // (uniformly spaced) 1-D value list, overriding the default 1, 2, 3, … grid.
  let mut domain_spec: Option<(f64, f64)> = None;
  for opt in args.iter().skip(1) {
    if let Expr::List(dims) = opt
      && dims.len() == 1
      && let Expr::List(pair) = &dims[0]
      && pair.len() == 2
      && let (Ok(a), Ok(b)) =
        (interp_value_to_f64(&pair[0]), interp_value_to_f64(&pair[1]))
    {
      domain_spec = Some((a, b));
    }
  }

  // Evaluate the data argument
  let data_evaluated = crate::evaluator::evaluate_expr_to_expr(data_arg)?;

  // Complex values: the function is built from the real parts and keeps the
  // complex `{x, z}` rows, which `evaluate_interpolating_function` splits.
  // A spline needs machine reals, so `Method -> "Spline"` falls back to the
  // default method with `Interpolation::mspl`, as in wolframscript.
  if let Expr::List(items) = &data_evaluated {
    let pairs: Vec<Expr> = if items
      .iter()
      .all(|i| matches!(i, Expr::List(p) if p.len() == 2))
    {
      items.to_vec()
    } else {
      items
        .iter()
        .enumerate()
        .map(|(k, v)| {
          Expr::List(vec![Expr::Integer(k as i128 + 1), v.clone()].into())
        })
        .collect()
    };
    let pairs_expr = Expr::List(pairs.into());
    if domain_spec.is_none()
      && let Some((re_rows, _)) =
        complex_interpolation_parts(&pairs_expr, false)
    {
      if spline {
        crate::emit_message(
          "Interpolation::mspl: The Spline method could not be used because the data could not be coerced to machine real numbers.",
        );
      }
      let mut real_args = vec![Expr::List(re_rows.into())];
      real_args.extend(
        args
          .iter()
          .skip(1)
          .filter(|opt| {
            !matches!(opt, Expr::Rule { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Method"))
          })
          .cloned(),
      );
      let built = interpolation_ast(&real_args, head)?;
      if let Expr::FunctionCall { name, args: fargs } = &built
        && name == "InterpolatingFunction"
        && (2..=4).contains(&fargs.len())
      {
        // Every value in complex form, real ones included, which is what
        // marks the function as complex for evaluation
        let complex_rows: Vec<Expr> = match &pairs_expr {
          Expr::List(rows) => rows
            .iter()
            .map(|row| match row {
              Expr::List(p) if p.len() == 2 => {
                let (re, im) =
                  crate::functions::math_ast::numeric_utils::try_extract_complex_f64(&p[1])
                    .unwrap_or((f64::NAN, 0.0));
                Expr::List(
                  vec![
                    p[0].clone(),
                    plus2(Expr::Real(re), times2(Expr::Real(im), id_expr("I"))),
                  ]
                  .into(),
                )
              }
              other => other.clone(),
            })
            .collect(),
          _ => Vec::new(),
        };
        let mut fargs = fargs.to_vec();
        fargs[1] = Expr::List(complex_rows.into());
        return Ok(Expr::FunctionCall {
          name: name.clone(),
          args: fargs.into(),
        });
      }
      return Ok(unevaluated(head, args));
    }
  }

  let Expr::List(data_list) = &data_evaluated else {
    // A non-list first argument is not interpolatable. wolframscript emits
    // this message tagged `Interpolation` (regardless of the actual head)
    // and keeps the call unevaluated.
    crate::emit_message(&format!(
      "Interpolation::innd: First argument in {} does not contain a list of data and coordinates.",
      crate::syntax::format_expr(
        &data_evaluated,
        crate::syntax::ExprForm::Output
      )
    ));
    return Ok(unevaluated(head, args));
  };

  if data_list.is_empty() {
    // An empty data list is not interpolatable: emit innd and stay
    // unevaluated (wolframscript parity) rather than raising a hard error.
    crate::emit_message(&format!(
      "Interpolation::innd: First argument in {} does not contain a list of data and coordinates.",
      crate::syntax::format_expr(
        &data_evaluated,
        crate::syntax::ExprForm::Output
      )
    ));
    return Ok(unevaluated(head, args));
  }

  // `{{x}, f, f', …}` rows: with any derivative given, Hermite interpolation;
  // with values only, the same as `{x, f}` pairs.
  let normalized_rows: Vec<Expr>;
  let mut data_list: &[Expr] = data_list;
  if head != "ListInterpolation"
    && let Some(rows) =
      crate::functions::hermite_interpolation::parse_rows(data_list)
  {
    if crate::functions::hermite_interpolation::has_derivatives(&rows) {
      return Ok(crate::functions::hermite_interpolation::build(
        &rows,
        interp_order.max(1) as usize,
      ));
    }
    normalized_rows = data_list
      .iter()
      .filter_map(|r| match r {
        Expr::List(parts) => match parts.as_slice() {
          [Expr::List(c), v] => {
            Some(Expr::List(vec![c[0].clone(), v.clone()].into()))
          }
          _ => None,
        },
        _ => None,
      })
      .collect();
    data_list = &normalized_rows;
  }

  // ListInterpolation of a numeric array of depth two or more is a grid
  // with uniformly spaced coordinates on each axis (1, 2, …, n by default,
  // or spanning an explicit `{{xmin, xmax}, {ymin, ymax}, …}` domain); the
  // coordinates are machine numbers when any value is.
  if head == "ListInterpolation"
    && let Some(dims) = numeric_tensor_dims(&data_evaluated)
    && dims.len() >= 2
  {
    let domains: Option<Vec<(Expr, Expr)>> =
      args.iter().skip(1).find_map(|a| {
        let Expr::List(pairs) = a else { return None };
        if pairs.len() != dims.len() {
          return None;
        }
        pairs
          .iter()
          .map(|p| match p {
            Expr::List(lh) if lh.len() == 2 => {
              interp_value_to_f64(&lh[0]).ok()?;
              interp_value_to_f64(&lh[1]).ok()?;
              Some((lh[0].clone(), lh[1].clone()))
            }
            _ => None,
          })
          .collect()
      });
    {
      let inexact = crate::syntax::contains_inexact(&data_evaluated);
      let axis = |k: usize, i: usize| -> Expr {
        let exact = match &domains {
          None => Expr::Integer(i as i128 + 1),
          Some(d) => {
            let (lo, hi) = &d[k];
            let n = dims[k].max(2) - 1;
            crate::evaluator::evaluate_expr_to_expr(&plus2(
              lo.clone(),
              times2(
                minus2(hi.clone(), lo.clone()),
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![
                    Expr::Integer(i as i128),
                    Expr::Integer(n as i128),
                  ]
                  .into(),
                },
              ),
            ))
            .unwrap_or(Expr::Integer(i as i128 + 1))
          }
        };
        if inexact {
          interp_value_to_f64(&exact).map_or(exact, Expr::Real)
        } else {
          exact
        }
      };
      let mut samples: Vec<(Vec<Expr>, Expr)> = Vec::new();
      fn walk(
        e: &Expr,
        index: &mut Vec<usize>,
        axis: &dyn Fn(usize, usize) -> Expr,
        out: &mut Vec<(Vec<Expr>, Expr)>,
      ) {
        match e {
          Expr::List(items) => {
            for (i, item) in items.iter().enumerate() {
              index.push(i);
              walk(item, index, axis, out);
              index.pop();
            }
          }
          leaf => out.push((
            index.iter().enumerate().map(|(k, &i)| axis(k, i)).collect(),
            leaf.clone(),
          )),
        }
      }
      walk(&data_evaluated, &mut Vec::new(), &axis, &mut samples);
      if let Some(result) = try_grid_interpolation(
        &samples,
        interp_orders.as_deref().unwrap_or(&[interp_order]),
        head,
      ) {
        return Ok(result);
      }
    }
  }

  // Multidimensional data on a complete grid: `{{x1, …, xn}, f}` pairs (the
  // documented form; `f` may be a vector) or flat `{x1, …, xn, f}` rows, as
  // `Flatten[Table[{x, y, f[x, y]}, {x, xs}, {y, ys}], 1]` produces.
  // (`ListInterpolation`'s rows are grid rows, handled above.)
  if head != "ListInterpolation" {
    let samples: Option<Vec<(Vec<Expr>, Expr)>> = data_list
      .iter()
      .map(|item| {
        let Expr::List(parts) = item else {
          return None;
        };
        match parts.as_slice() {
          [Expr::List(coords), value] if coords.len() >= 2 => {
            Some((coords.to_vec(), value.clone()))
          }
          [coords @ .., value]
            if coords.len() >= 2
              && !parts.iter().any(|p| matches!(p, Expr::List(_))) =>
          {
            Some((coords.to_vec(), value.clone()))
          }
          _ => None,
        }
      })
      .collect();
    if let Some(samples) = samples
      && let Some(result) = try_grid_interpolation(
        &samples,
        interp_orders.as_deref().unwrap_or(&[interp_order]),
        head,
      )
    {
      return Ok(result);
    }
  }

  // Vector- or tensor-valued data `{{x1, {…}}, {x2, {…}}, …}` interpolates
  // every component separately; see `evaluate_tensor_interpolating_function`.
  if let Some(tensor_fn) = build_tensor_interpolation(data_list, interp_order) {
    return Ok(tensor_fn);
  }

  // Determine format: list of values or list of {x, y} pairs
  let mut points: Vec<(f64, f64)> = Vec::new();
  // The exact coordinate expressions alongside their f64 values, so that
  // interpolating at an exact point can stay exact.
  let mut exact_coords: Vec<(Expr, Expr)> = Vec::new();

  let first = &data_list[0];
  let is_pair_format = matches!(first, Expr::List(items) if items.len() == 2);

  if is_pair_format {
    // {{x1, y1}, {x2, y2}, ...}
    for item in data_list {
      let (x, y) = extract_point(item)?;
      points.push((x, y));
      let Expr::List(pair) = item else {
        return Err(InterpreterError::EvaluationError(
          "InterpolatingFunction: invalid data point format".into(),
        ));
      };
      exact_coords
        .push((exact_coordinate(&pair[0], x), exact_coordinate(&pair[1], y)));
    }
  } else {
    // {y1, y2, ...} — x values are 1, 2, 3, ... by default, or uniformly
    // spaced across [xmin, xmax] when a domain was supplied.
    let count = data_list.len();
    for (i, item) in data_list.iter().enumerate() {
      // A one-element row `{v}` (e.g. from `Accumulate[{f[#]} & /@ ...]`)
      // carries the single value `v`.
      let item = match item {
        Expr::List(inner) if inner.len() == 1 => &inner[0],
        other => other,
      };
      let y = interp_value_to_f64(item)?;
      let (x, x_expr) = match domain_spec {
        Some((xmin, xmax)) if count > 1 => {
          let x = xmin + (i as f64) * (xmax - xmin) / ((count - 1) as f64);
          (x, Expr::Real(x))
        }
        _ => ((i + 1) as f64, Expr::Integer(i as i128 + 1)),
      };
      points.push((x, y));
      exact_coords.push((x_expr, exact_coordinate(item, y)));
    }
  }

  // Sort by x value, keeping each exact coordinate pair with its point.
  let mut order_index: Vec<usize> = (0..points.len()).collect();
  order_index.sort_by(|a, b| points[*a].0.partial_cmp(&points[*b].0).unwrap());
  points = order_index.iter().map(|i| points[*i]).collect();
  exact_coords = order_index
    .iter()
    .map(|i| exact_coords[*i].clone())
    .collect();

  let n = points.len();

  // Clamp order to valid range. A single data point is allowed: the order is
  // reduced to 0 (with inhr) and the result is a constant interpolation,
  // matching wolframscript.
  let mut order = interp_order.max(1) as usize;
  if order >= n {
    let reduced = n - 1;
    crate::emit_message(&format!(
      "{head}::inhr: Requested order is too high; order has been reduced to {{{reduced}}}."
    ));
    order = reduced;
  }

  let x_min = points[0].0;
  let x_max = points[n - 1].0;

  let domain = Expr::List(
    vec![Expr::List(
      vec![Expr::Real(x_min), Expr::Real(x_max)].into(),
    )]
    .into(),
  );

  if spline
    && let Some(sp) =
      crate::functions::spline_interpolation::Spline::interpolate(
        points.iter().map(|p| p.0).collect(),
        points.iter().map(|p| p.1).collect(),
        order.max(1),
      )
  {
    let degree = Expr::Integer(sp.degree as i128);
    return Ok(call(
      "InterpolatingFunction",
      vec![domain, sp.to_data(), degree],
    ));
  }

  // Store data as list of {x, y} pairs, keeping exact coordinates exact (an
  // Integer grid, rational values) so that interpolating at an exact point
  // gives an exact result, and evaluation at a grid point returns the
  // original value unchanged.
  let data_expr = Expr::List(
    exact_coords
      .iter()
      .map(|(x, y)| Expr::List(vec![x.clone(), y.clone()].into()))
      .collect(),
  );

  // Store the interpolation order as a third argument
  let interp_func = call(
    "InterpolatingFunction",
    vec![domain, data_expr, Expr::Integer(order as i128)],
  );

  Ok(interp_func)
}

/// The shape (nested list lengths) of a tensor value, or `None` when it is a
/// ragged list or has a non-numeric leaf.
fn numeric_tensor_dims(e: &Expr) -> Option<Vec<usize>> {
  match e {
    Expr::List(items) => {
      let first = numeric_tensor_dims(items.first()?)?;
      for it in items.iter().skip(1) {
        if numeric_tensor_dims(it)? != first {
          return None;
        }
      }
      let mut dims = vec![items.len()];
      dims.extend(first);
      Some(dims)
    }
    other => interp_value_to_f64(other).ok().map(|_| Vec::new()),
  }
}

/// `Interpolation[{{x1, t1}, {x2, t2}, …}]` where every `ti` is a list (or
/// nested list) of numbers of one common shape. The data is stored as given
/// (sorted by `x`); evaluation maps the scalar interpolation over the leaves.
fn build_tensor_interpolation(
  data_list: &[Expr],
  interp_order: i128,
) -> Option<Expr> {
  let mut rows: Vec<(f64, Expr, Expr)> = Vec::with_capacity(data_list.len());
  let mut dims: Option<Vec<usize>> = None;
  for item in data_list {
    let Expr::List(pair) = item else {
      return None;
    };
    if pair.len() != 2 || !matches!(&pair[1], Expr::List(_)) {
      return None;
    }
    let x = interp_value_to_f64(&pair[0]).ok()?;
    let d = numeric_tensor_dims(&pair[1])?;
    if *dims.get_or_insert_with(|| d.clone()) != d {
      return None;
    }
    rows.push((x, exact_coordinate(&pair[0], x), pair[1].clone()));
  }
  rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
  let n = rows.len();
  let mut order = interp_order.max(1) as usize;
  if order >= n {
    order = n.saturating_sub(1);
    crate::emit_message(&format!(
      "Interpolation::inhr: Requested order is too high; order has been reduced to {{{order}}}."
    ));
  }
  let domain = Expr::List(
    vec![Expr::List(
      vec![Expr::Real(rows[0].0), Expr::Real(rows[n - 1].0)].into(),
    )]
    .into(),
  );
  let data_expr = Expr::List(
    rows
      .into_iter()
      .map(|(_, x, t)| Expr::List(vec![x, t].into()))
      .collect(),
  );
  Some(call(
    "InterpolatingFunction",
    vec![domain, data_expr, Expr::Integer(order as i128)],
  ))
}

/// Evaluate a tensor-valued `InterpolatingFunction` by interpolating each
/// leaf of the stored tensors through the scalar code path.
fn evaluate_tensor_interpolating_function(
  func_args: &[Expr],
  rows: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  fn map_leaves(
    func_args: &[Expr],
    rows: &[Expr],
    path: &mut Vec<usize>,
    call_args: &[Expr],
  ) -> Result<Expr, InterpreterError> {
    let mut leaf_rows = Vec::with_capacity(rows.len());
    let mut sample: &Expr = &Expr::Integer(0);
    for row in rows {
      let Expr::List(pair) = row else {
        return Err(InterpreterError::EvaluationError(
          "InterpolatingFunction: invalid data point format".into(),
        ));
      };
      let mut v = &pair[1];
      for &i in path.iter() {
        let Expr::List(items) = v else {
          unreachable!("tensor shapes are validated at construction");
        };
        v = &items[i];
      }
      sample = v;
      leaf_rows.push(Expr::List(vec![pair[0].clone(), v.clone()].into()));
    }
    if let Expr::List(items) = sample {
      let mut out = Vec::with_capacity(items.len());
      for i in 0..items.len() {
        path.push(i);
        out.push(map_leaves(func_args, rows, path, call_args)?);
        path.pop();
      }
      return Ok(Expr::List(out.into()));
    }
    let mut scalar_args = func_args.to_vec();
    scalar_args[1] = Expr::List(leaf_rows.into());
    evaluate_interpolating_function(&scalar_args, call_args)
  }
  map_leaves(func_args, rows, &mut Vec::new(), call_args)
}

/// The relative tolerance two coordinates must fall within to count as "the
/// same" grid line — generous enough for the roundoff two different `Table`
/// passes over the same arithmetic can leave, tight enough not to merge
/// genuinely distinct samples.
fn same_coordinate(a: f64, b: f64) -> bool {
  (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

/// Build the `InterpolatingFunction` for `{coords, value}` samples that tile
/// a complete grid in two or more dimensions: `InterpolatingFunction[{{x1min,
/// x1max}, …}, values, {order1, …}, {x1coords, …}]`, where `values` is nested
/// one level per axis. Values may be real, complex or vectors (each
/// component interpolates on its own).
///
/// Returns `None` when the samples are not such a grid (a non-numeric
/// coordinate, a missing or repeated grid point) so the caller falls back to
/// its other formats. Genuinely scattered data, which Wolfram interpolates
/// over a Delaunay triangulation, is not supported here.
fn try_grid_interpolation(
  samples: &[(Vec<Expr>, Expr)],
  orders: &[i128],
  head: &str,
) -> Option<Expr> {
  let n = samples.first()?.0.len();
  if n < 2 || samples.iter().any(|(c, _)| c.len() != n) {
    return None;
  }
  // Distinct coordinates per axis, sorted, each keeping the first expression
  // seen for it (so an exact coordinate stays exact).
  let mut axes: Vec<Vec<(f64, Expr)>> = vec![Vec::new(); n];
  let mut numeric: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
  for (coords, value) in samples {
    grid_value(value)?;
    let mut point = Vec::with_capacity(n);
    for (k, c) in coords.iter().enumerate() {
      let v = interp_value_to_f64(c).ok()?;
      if !axes[k].iter().any(|(a, _)| same_coordinate(*a, v)) {
        axes[k].push((v, c.clone()));
      }
      point.push(v);
    }
    numeric.push(point);
  }
  for axis in &mut axes {
    axis.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
  }
  let sizes: Vec<usize> = axes.iter().map(Vec::len).collect();
  // An axis with a single coordinate is interpolated with order 0.
  if sizes.iter().product::<usize>() != samples.len() {
    return None;
  }
  let mut cells: Vec<Option<Expr>> = vec![None; samples.len()];
  for ((_, value), point) in samples.iter().zip(&numeric) {
    let mut flat = 0usize;
    for (k, v) in point.iter().enumerate() {
      let i = axes[k].iter().position(|(a, _)| same_coordinate(*a, *v))?;
      flat = flat * sizes[k] + i;
    }
    if cells[flat].is_some() {
      return None;
    }
    cells[flat] = Some(stored_grid_value(value));
  }
  let cells: Vec<Expr> = cells.into_iter().collect::<Option<_>>()?;
  fn nest(cells: &[Expr], sizes: &[usize]) -> Expr {
    match sizes.split_first() {
      Some((&m, rest)) if !rest.is_empty() => {
        let chunk = cells.len() / m;
        Expr::List(
          (0..m)
            .map(|i| nest(&cells[i * chunk..(i + 1) * chunk], rest))
            .collect(),
        )
      }
      _ => Expr::List(cells.to_vec().into()),
    }
  }
  let values = nest(&cells, &sizes);

  let wanted: Vec<usize> = (0..n)
    .map(|k| {
      orders.get(k).or(orders.last()).copied().unwrap_or(3).max(1) as usize
    })
    .collect();
  let used: Vec<usize> = wanted
    .iter()
    .zip(&sizes)
    .map(|(&w, &m)| w.min(m - 1))
    .collect();
  if used != wanted {
    let list: Vec<String> = used.iter().map(usize::to_string).collect();
    crate::emit_message(&format!(
      "{head}::inhr: Requested order is too high; order has been reduced to {{{}}}.",
      list.join(", ")
    ));
  }
  let coord_exprs: Vec<Vec<Expr>> = axes
    .iter()
    .map(|axis| axis.iter().map(|(v, e)| exact_coordinate(e, *v)).collect())
    .collect();
  let domain = Expr::List(
    coord_exprs
      .iter()
      .map(|c| Expr::List(vec![c[0].clone(), c[c.len() - 1].clone()].into()))
      .collect(),
  );
  Some(call(
    "InterpolatingFunction",
    vec![
      domain,
      values,
      Expr::List(used.iter().map(|&o| Expr::Integer(o as i128)).collect()),
      Expr::List(
        coord_exprs
          .into_iter()
          .map(|c| Expr::List(c.into()))
          .collect(),
      ),
    ],
  ))
}

/// A grid sample: a (possibly complex) number, or a list of them for
/// vector- and tensor-valued data.
enum GridValue {
  Num(f64, f64),
  List(Vec<GridValue>),
}

fn grid_value(e: &Expr) -> Option<GridValue> {
  match e {
    Expr::List(items) if !items.is_empty() => Some(GridValue::List(
      items.iter().map(grid_value).collect::<Option<_>>()?,
    )),
    Expr::List(_) => None,
    other => {
      interp_complex_to_parts(other).map(|(re, im)| GridValue::Num(re, im))
    }
  }
}

/// How a sample is kept in the grid: exact numbers stay exact, complex ones
/// become machine complexes.
fn stored_grid_value(e: &Expr) -> Expr {
  match e {
    Expr::List(items) => {
      Expr::List(items.iter().map(stored_grid_value).collect())
    }
    other => match interp_complex_to_parts(other) {
      Some((re, im)) if im != 0.0 => {
        crate::functions::math_ast::numeric_utils::build_complex_float_expr(
          re, im,
        )
      }
      Some((re, _)) => exact_coordinate(other, re),
      None => other.clone(),
    },
  }
}

fn grid_components(v: &GridValue, out: &mut Vec<(f64, f64)>) {
  match v {
    GridValue::Num(re, im) => out.push((*re, *im)),
    GridValue::List(items) => {
      for i in items {
        grid_components(i, out);
      }
    }
  }
}

fn grid_rebuild(
  template: &GridValue,
  comps: &mut impl Iterator<Item = (f64, f64)>,
) -> Expr {
  match template {
    GridValue::Num(..) => {
      let (re, im) = comps.next().unwrap_or((f64::NAN, 0.0));
      real_or_machine_complex(re, im)
    }
    GridValue::List(items) => {
      Expr::List(items.iter().map(|t| grid_rebuild(t, comps)).collect())
    }
  }
}

/// Evaluate a grid `InterpolatingFunction` with explicit coordinates in two
/// or more dimensions (`try_grid_interpolation`'s output, or a stored
/// kernel-layout one): `InterpolatingFunction[domain, values, orders,
/// coords]`, plus a fifth `{d1, …}` argument for a partial derivative.
///
/// The value is a tensor product of the 1-D local Lagrange pieces: along the
/// last axis first, for every grid line in the other axes' stencils, then
/// axis by axis outward. Only the `(order + 1)`-point stencil on each axis is
/// read, so a query never touches the rest of a large grid (`NDSolve`'s PDE
/// grids are resampled at every plot point).
fn evaluate_grid_interpolating_function(
  domain: &Expr,
  values: &Expr,
  orders: &[Expr],
  coords: &[Expr],
  derivative: &[usize],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let n = coords.len();
  let unevaluated = || {
    let mut parts = vec![
      domain.clone(),
      values.clone(),
      Expr::List(orders.to_vec().into()),
      Expr::List(coords.to_vec().into()),
    ];
    if derivative.iter().any(|&d| d > 0) {
      parts.push(Expr::List(
        derivative
          .iter()
          .map(|&d| Expr::Integer(d as i128))
          .collect(),
      ));
    }
    Expr::CurriedCall {
      func: Box::new(call("InterpolatingFunction", parts)),
      args: call_args.to_vec(),
    }
  };
  let coord_exprs: Vec<&[Expr]> = coords
    .iter()
    .map(|c| match c {
      Expr::List(items) => Some(items.as_slice()),
      _ => None,
    })
    .collect::<Option<_>>()
    .ok_or_else(|| {
      InterpreterError::EvaluationError(
        "InterpolatingFunction: invalid coordinates".into(),
      )
    })?;
  let axes: Vec<Vec<f64>> = coord_exprs
    .iter()
    .map(|c| {
      c.iter()
        .map(|e| interp_value_to_f64(e).unwrap_or(f64::NAN))
        .collect()
    })
    .collect();
  if axes.iter().any(Vec::is_empty) {
    return Ok(unevaluated());
  }

  if let [Expr::String(prop)] = call_args {
    match prop.as_str() {
      "Domain" => return Ok(domain.clone()),
      "Coordinates" => return Ok(Expr::List(coords.to_vec().into())),
      "Grid" => {
        fn points(axes: &[&[Expr]], prefix: &mut Vec<Expr>) -> Expr {
          match axes.split_first() {
            None => Expr::List(prefix.clone().into()),
            Some((axis, rest)) => Expr::List(
              axis
                .iter()
                .map(|c| {
                  prefix.push(c.clone());
                  let p = points(rest, prefix);
                  prefix.pop();
                  p
                })
                .collect(),
            ),
          }
        }
        return Ok(points(&coord_exprs, &mut Vec::new()));
      }
      "ValuesOnGrid" => return Ok(values.clone()),
      "InterpolationOrder" => return Ok(Expr::List(orders.to_vec().into())),
      _ => {}
    }
  }

  let query: Vec<Expr> = match call_args {
    [Expr::List(point)] if point.len() == n => point.to_vec(),
    other => other.to_vec(),
  };
  if query.len() != n {
    return Ok(unevaluated());
  }
  let query: Vec<Expr> = query
    .iter()
    .map(|a| {
      crate::evaluator::evaluate_expr_to_expr(a).unwrap_or_else(|_| a.clone())
    })
    .collect();
  let Some(x): Option<Vec<f64>> = query
    .iter()
    .map(|e| match e {
      Expr::Integer(k) => Some(*k as f64),
      Expr::Real(f) => Some(*f),
      _ if is_exact_number(e) => interp_value_to_f64(e).ok(),
      _ => None,
    })
    .collect()
  else {
    return Ok(unevaluated());
  };

  let cell = |index: &[usize]| -> Result<&Expr, InterpreterError> {
    let mut v = values;
    for &i in index {
      let Expr::List(items) = v else {
        return Err(InterpreterError::EvaluationError(
          "InterpolatingFunction: invalid grid".into(),
        ));
      };
      v = items.get(i).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "InterpolatingFunction: invalid grid".into(),
        )
      })?;
    }
    Ok(v)
  };
  let template = grid_value(cell(&vec![0; n])?).ok_or_else(|| {
    InterpreterError::EvaluationError(
      "InterpolatingFunction: invalid grid value".into(),
    )
  })?;

  // A grid point itself gives the stored sample, kept as written when the
  // query coordinates are exact integers.
  if derivative.iter().all(|&d| d == 0)
    && let Some(index) = axes
      .iter()
      .zip(&x)
      .map(|(axis, &v)| axis.iter().position(|&a| same_coordinate(a, v)))
      .collect::<Option<Vec<usize>>>()
  {
    let stored = cell(&index)?;
    if query.iter().all(|a| matches!(a, Expr::Integer(_))) {
      return Ok(stored.clone());
    }
    if let Some(v) = grid_value(stored) {
      let mut comps = Vec::new();
      grid_components(&v, &mut comps);
      return Ok(grid_rebuild(&template, &mut comps.into_iter()));
    }
  }

  if axes.iter().zip(&x).any(|(axis, &v)| {
    let (lo, hi) = (
      axis[0].min(axis[axis.len() - 1]),
      axis[0].max(axis[axis.len() - 1]),
    );
    outside_interpolation_domain(v, lo, hi)
  }) {
    emit_dmval(&query);
  }

  let windows: Vec<(usize, usize)> = axes
    .iter()
    .zip(&x)
    .enumerate()
    .map(|(k, (axis, &v))| {
      let order = match orders.get(k) {
        Some(Expr::Integer(o)) => (*o).max(1) as usize,
        _ => 1,
      };
      let m = axis.len();
      if m == 1 {
        (0, 1)
      } else {
        lagrange_window(m, bracket_index_f64(axis, v), order.min(m - 1))
      }
    })
    .collect();
  // Exact coordinates, values and query point: the tensor-product Lagrange
  // interpolant in exact arithmetic (`ListInterpolation[{{1, 2}, {3, 4}}][
  // 3/2, 3/2]` is 5/2).
  if derivative.iter().all(|&d| d == 0)
    && query.iter().all(is_exact_number)
    && coord_exprs.iter().all(|c| c.iter().all(is_exact_number))
  {
    let weights: Vec<Vec<Expr>> = windows
      .iter()
      .enumerate()
      .map(|(k, &(start, end))| {
        (start..end)
          .map(|i| {
            let mut factors = Vec::new();
            for j in start..end {
              if j != i {
                factors.push(div2(
                  minus2(query[k].clone(), coord_exprs[k][j].clone()),
                  minus2(coord_exprs[k][i].clone(), coord_exprs[k][j].clone()),
                ));
              }
            }
            times(factors)
          })
          .collect()
      })
      .collect();
    let mut terms = Vec::new();
    let mut exact = true;
    let mut index = vec![0usize; n];
    let mut counters = vec![0usize; n];
    'cells: loop {
      for k in 0..n {
        index[k] = windows[k].0 + counters[k];
      }
      let value = cell(&index)?;
      if !is_exact_number(value) {
        exact = false;
        break;
      }
      let mut factors: Vec<Expr> =
        (0..n).map(|k| weights[k][counters[k]].clone()).collect();
      factors.push(value.clone());
      terms.push(times(factors));
      for k in (0..n).rev() {
        counters[k] += 1;
        if windows[k].0 + counters[k] < windows[k].1 {
          continue 'cells;
        }
        counters[k] = 0;
      }
      break;
    }
    if exact {
      return crate::evaluator::evaluate_expr_to_expr(&plus(terms));
    }
  }

  // Only the cells of the interpolation window take part: reading the whole
  // grid into machine numbers on every call made each evaluation cost as
  // much as the grid is large. A window spanning its whole sub-axis is the
  // one the full grid would pick, so the value is unchanged.
  let sub_axes: Vec<Vec<f64>> = axes
    .iter()
    .zip(&windows)
    .map(|(axis, &(start, end))| axis[start..end].to_vec())
    .collect();
  fn slice_values(e: &Expr, windows: &[(usize, usize)]) -> Option<Expr> {
    let Some((&(start, end), rest)) = windows.split_first() else {
      return Some(e.clone());
    };
    let Expr::List(items) = e else { return None };
    if end > items.len() {
      return None;
    }
    Some(Expr::List(
      (start..end)
        .map(|i| slice_values(&items[i], rest))
        .collect::<Option<Vec<_>>>()?
        .into(),
    ))
  }
  let derivative_free = derivative.iter().all(|&d| d == 0);
  let window_values = if derivative_free {
    slice_values(values, &windows)
  } else {
    None
  };
  let (grid_axes, grid_values) = match &window_values {
    Some(v) => (sub_axes, v),
    None => (axes, values),
  };
  let grid = NumericGrid::new(
    grid_axes,
    orders
      .iter()
      .map(|o| match o {
        Expr::Integer(o) => (*o).max(1) as usize,
        _ => 1,
      })
      .collect(),
    derivative.to_vec(),
    grid_values,
  )
  .ok_or_else(|| {
    InterpreterError::EvaluationError(
      "InterpolatingFunction: inconsistent grid values".into(),
    )
  })?;
  let comps = grid.eval(&x);
  Ok(grid_rebuild(&template, &mut comps.into_iter()))
}

/// A grid `InterpolatingFunction` in machine numbers: the coordinates of
/// each axis, the interpolation order and partial-derivative order along
/// it, and every grid point's components (real and imaginary part of each
/// number of a scalar, vector or tensor value) in row-major order.
pub(crate) struct NumericGrid {
  axes: Vec<Vec<f64>>,
  orders: Vec<usize>,
  derivative: Vec<usize>,
  width: usize,
  cells: Vec<(f64, f64)>,
}

impl NumericGrid {
  fn new(
    axes: Vec<Vec<f64>>,
    orders: Vec<usize>,
    derivative: Vec<usize>,
    values: &Expr,
  ) -> Option<Self> {
    fn walk(
      e: &Expr,
      depth: usize,
      out: &mut Vec<Vec<(f64, f64)>>,
    ) -> Option<()> {
      if depth == 0 {
        let mut comps = Vec::new();
        grid_components(&grid_value(e)?, &mut comps);
        out.push(comps);
        return Some(());
      }
      let Expr::List(items) = e else { return None };
      for item in items {
        walk(item, depth - 1, out)?;
      }
      Some(())
    }
    let mut leaves = Vec::new();
    walk(values, axes.len(), &mut leaves)?;
    if leaves.len() != axes.iter().map(Vec::len).product::<usize>() {
      return None;
    }
    let width = leaves.first()?.len();
    if leaves.iter().any(|l| l.len() != width) {
      return None;
    }
    Some(Self {
      axes,
      orders,
      derivative,
      width,
      cells: leaves.into_iter().flatten().collect(),
    })
  }

  /// The function stored as `InterpolatingFunction[domain, values,
  /// orders, coords]` (with a fifth argument of partial-derivative orders
  /// as `Derivative[…][f]` leaves it).
  pub(crate) fn from_function(func_args: &[Expr]) -> Option<Self> {
    if !(4..=5).contains(&func_args.len()) {
      return None;
    }
    let (Expr::List(orders), Expr::List(coords)) =
      (&func_args[2], &func_args[3])
    else {
      return None;
    };
    if orders.len() < 2 || coords.len() != orders.len() {
      return None;
    }
    let axes: Vec<Vec<f64>> = coords
      .iter()
      .map(|c| match c {
        Expr::List(items) if !items.is_empty() => {
          items.iter().map(|e| interp_value_to_f64(e).ok()).collect()
        }
        _ => None,
      })
      .collect::<Option<_>>()?;
    let orders: Vec<usize> = orders
      .iter()
      .map(|o| match o {
        Expr::Integer(o) if *o >= 0 => Some((*o).max(1) as usize),
        _ => None,
      })
      .collect::<Option<_>>()?;
    let derivative: Vec<usize> = match func_args.get(4) {
      None => vec![0; orders.len()],
      Some(Expr::List(d)) if d.len() == orders.len() => d
        .iter()
        .map(|e| match e {
          Expr::Integer(k) if *k >= 0 => Some(*k as usize),
          _ => None,
        })
        .collect::<Option<_>>()?,
      _ => return None,
    };
    Self::new(axes, orders, derivative, &func_args[1])
  }

  /// Whether `x` lies outside the grid (by more than rounding).
  pub(crate) fn outside(&self, x: &[f64]) -> bool {
    self.axes.iter().zip(x).any(|(axis, &v)| {
      let (a, b) = (axis[0], axis[axis.len() - 1]);
      outside_interpolation_domain(v, a.min(b), a.max(b))
    })
  }

  /// The value at `x` of a real scalar function (NaN otherwise).
  pub(crate) fn eval_real(&self, x: &[f64]) -> f64 {
    match self.eval(x).as_slice() {
      [(re, im)] if *im == 0.0 => *re,
      _ => f64::NAN,
    }
  }

  /// Every component of the value at `x`: tensor-product local Lagrange
  /// interpolation, one axis at a time.
  pub(crate) fn eval(&self, x: &[f64]) -> Vec<(f64, f64)> {
    let n = self.axes.len();
    let windows: Vec<(usize, usize)> = (0..n)
      .map(|k| {
        let axis = &self.axes[k];
        let m = axis.len();
        if m == 1 {
          (0, 1)
        } else {
          lagrange_window(
            m,
            bracket_index_f64(axis, x[k]),
            self.orders[k].min(m - 1),
          )
        }
      })
      .collect();
    let mut strides = vec![self.width; n];
    for k in (0..n.saturating_sub(1)).rev() {
      strides[k] = strides[k + 1] * self.axes[k + 1].len();
    }
    self.along(0, 0, &windows, &strides, x)
  }

  fn along(
    &self,
    axis: usize,
    offset: usize,
    windows: &[(usize, usize)],
    strides: &[usize],
    x: &[f64],
  ) -> Vec<(f64, f64)> {
    if axis == windows.len() {
      return self.cells[offset..offset + self.width].to_vec();
    }
    let (start, end) = windows[axis];
    let samples: Vec<Vec<(f64, f64)>> = (start..end)
      .map(|i| {
        self.along(axis + 1, offset + i * strides[axis], windows, strides, x)
      })
      .collect();
    let xs = &self.axes[axis][start..end];
    let d = self.derivative.get(axis).copied().unwrap_or(0);
    let mut re = vec![0.0; samples.len()];
    let mut im = vec![0.0; samples.len()];
    (0..self.width)
      .map(|c| {
        for (k, s) in samples.iter().enumerate() {
          re[k] = s[c].0;
          im[k] = s[c].1;
        }
        (
          interp_1d_xy_deriv_f64(xs, &re, x[axis], d),
          if im.iter().all(|v| *v == 0.0) {
            0.0
          } else {
            interp_1d_xy_deriv_f64(xs, &im, x[axis], d)
          },
        )
      })
      .collect()
  }
}

/// Whether `x` lies outside `[lo, hi]` by more than wolframscript lets
/// pass without `InterpolatingFunction::dmval`: a query a few millionths
/// of the domain's width past an end (a grid built by repeated `t += dt`
/// overshoots by rounding) is silent; from about 10^-5 of the width on it
/// warns.
pub(crate) fn outside_interpolation_domain(x: f64, lo: f64, hi: f64) -> bool {
  let slack = 7.5e-6 * (hi - lo).abs();
  x < lo - slack || x > hi + slack
}

/// `InterpolatingFunction::dmval` for a query point outside the data.
pub(crate) fn emit_dmval(point: &[Expr]) {
  // Built lazily: an ODE right-hand side that strays outside its lookup
  // table queries it tens of thousands of times, and wolframscript stops
  // reporting after three.
  crate::emit_message_with("InterpolatingFunction::dmval", || {
    let coords: Vec<String> = point
      .iter()
      .map(|c| crate::syntax::format_expr(c, crate::syntax::ExprForm::Output))
      .collect();
    format!(
      "InterpolatingFunction::dmval: Input value {{{}}} lies outside the range of data in the interpolating function. Extrapolation will be used.",
      coords.join(", ")
    )
  });
}

/// A machine real when the imaginary part vanishes, else the machine complex.
fn real_or_machine_complex(re: f64, im: f64) -> Expr {
  if im == 0.0 {
    Expr::Real(re)
  } else {
    crate::functions::math_ast::numeric_utils::build_complex_float_expr(re, im)
  }
}

/// Binary search for the interval bracketing `x_val` in an ascending or
/// descending coordinate array: returns `lo` such that `[lo, lo + 1]`
/// brackets it (clamped to the array's ends for an out-of-range query).
fn bracket_index_f64(xs: &[f64], x_val: f64) -> usize {
  let n = xs.len();
  if n < 2 {
    return 0;
  }
  let ascending = xs[n - 1] >= xs[0];
  let mut lo = 0usize;
  let mut hi = n - 1;
  while lo < hi - 1 {
    let mid = usize::midpoint(lo, hi);
    let before = if ascending {
      x_val < xs[mid]
    } else {
      x_val > xs[mid]
    };
    if before {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  lo
}

/// 1-D local Lagrange interpolation of `(xs[i], ys[i])` samples at
/// arbitrary (not necessarily unit-spaced) coordinates, evaluated at
/// `x_val`. The explicit-coordinate counterpart of `interp_1d_f64`, which
/// assumes samples at integer positions `1..=n`; shares its window
/// selection (`lagrange_window`) so results agree wherever the spacing
/// happens to be uniform.
fn interp_1d_xy_f64(xs: &[f64], ys: &[f64], x_val: f64, order: usize) -> f64 {
  let n = xs.len();
  if n == 1 {
    return ys[0];
  }
  // Binary search for the interval containing x_val (or nearest, for a
  // point outside the range, so extrapolation uses the boundary stencil).
  let mut lo = 0usize;
  let mut hi = n - 1;
  while lo < hi - 1 {
    let mid = usize::midpoint(lo, hi);
    if x_val < xs[mid] {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  let eff_order = order.min(n - 1).max(1);
  let (start, end) = lagrange_window(n, lo, eff_order);
  let m = end - start;
  let mut result = 0.0;
  for i in 0..m {
    let mut basis = 1.0;
    for j in 0..m {
      if j != i {
        basis *= (x_val - xs[start + j]) / (xs[start + i] - xs[start + j]);
      }
    }
    result += ys[start + i] * basis;
  }
  result
}

/// `d`-th derivative at `x_val` of the Lagrange polynomial through the whole
/// sample window `(xs[i], ys[i])` (the window `lagrange_window` selected).
/// The derivative of each basis polynomial is the sum, over every choice of
/// `d` distinct factors to differentiate, of `d!` times the product of the
/// remaining factors.
fn interp_1d_xy_deriv_f64(xs: &[f64], ys: &[f64], x_val: f64, d: usize) -> f64 {
  let m = xs.len();
  if m == 1 {
    return if d == 0 { ys[0] } else { 0.0 };
  }
  if d >= m {
    return 0.0;
  }
  if d == 0 {
    // The value itself: no subsets to enumerate.
    return (0..m)
      .map(|i| {
        let mut term = ys[i];
        for j in 0..m {
          if j != i {
            term *= (x_val - xs[j]) / (xs[i] - xs[j]);
          }
        }
        term
      })
      .sum();
  }
  let factorial: f64 = (1..=d).map(|k| k as f64).product();
  let mut total = 0.0;
  for i in 0..m {
    let others: Vec<usize> = (0..m).filter(|&j| j != i).collect();
    // Sum over the size-`d` subsets of `others` that get differentiated.
    let mut basis_deriv = 0.0;
    let mut chosen = vec![false; others.len()];
    fn walk(
      pos: usize,
      left: usize,
      chosen: &mut Vec<bool>,
      others: &[usize],
      xs: &[f64],
      i: usize,
      x_val: f64,
      acc: &mut f64,
    ) {
      if left > others.len() - pos {
        return;
      }
      if pos == others.len() {
        let mut term = 1.0;
        for (k, &j) in others.iter().enumerate() {
          let denom = xs[i] - xs[j];
          term *= if chosen[k] {
            1.0 / denom
          } else {
            (x_val - xs[j]) / denom
          };
        }
        *acc += term;
        return;
      }
      if left > 0 {
        chosen[pos] = true;
        walk(pos + 1, left - 1, chosen, others, xs, i, x_val, acc);
        chosen[pos] = false;
      }
      walk(pos + 1, left, chosen, others, xs, i, x_val, acc);
    }
    walk(0, d, &mut chosen, &others, xs, i, x_val, &mut basis_deriv);
    total += ys[i] * basis_deriv * factorial;
  }
  total
}

// ─── InterpolatingFunction evaluation ──────────────────────────────────

/// InterpolatingFunction returns machine-precision reals for interpolated values.
fn real_or_integer(v: f64) -> Expr {
  Expr::Real(v)
}

/// The stored form of one interpolation coordinate: the evaluated source
/// expression when it is an exact number (so `1/2` stays a rational rather
/// than becoming `0.5`), and the machine value otherwise.
fn exact_coordinate(source: &Expr, value: f64) -> Expr {
  let evaluated = crate::evaluator::evaluate_expr_to_expr(source)
    .unwrap_or_else(|_| source.clone());
  match &evaluated {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => evaluated,
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && args
          .iter()
          .all(|a| matches!(a, Expr::Integer(_) | Expr::BigInteger(_))) =>
    {
      evaluated
    }
    _ => Expr::Real(value),
  }
}

/// Whether an interpolation coordinate is an exact number, so that exact
/// arithmetic through it is worthwhile.
pub(crate) fn is_exact_number(e: &Expr) -> bool {
  match e {
    Expr::Integer(_) | Expr::BigInteger(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      args
        .iter()
        .all(|a| matches!(a, Expr::Integer(_) | Expr::BigInteger(_)))
    }
    _ => false,
  }
}

/// Answer an `InterpolatingFunction[…]["property"]` query from the stored
/// `{x, y}` grid data. Returns `None` for unrecognized properties so the call
/// stays unevaluated.
fn interpolating_function_property(
  data: &Expr,
  order: usize,
  derivative_order: usize,
  prop: &str,
) -> Option<Expr> {
  let Expr::List(pairs) = data else {
    return None;
  };
  let mut xs: Vec<Expr> = Vec::with_capacity(pairs.len());
  let mut ys: Vec<Expr> = Vec::with_capacity(pairs.len());
  for p in pairs {
    if let Expr::List(pair) = p
      && pair.len() == 2
    {
      xs.push(pair[0].clone());
      ys.push(pair[1].clone());
    } else {
      return None;
    }
  }
  if xs.is_empty() {
    return None;
  }
  let list1 = |items: Vec<Expr>| Expr::List(items.into());
  match prop {
    // The interpolation domain, as a list of {min, max} per dimension.
    "Domain" => Some(list1(vec![list1(vec![
      xs.first().unwrap().clone(),
      xs.last().unwrap().clone(),
    ])])),
    // Each grid coordinate wrapped in a one-element list.
    "Grid" => Some(list1(xs.into_iter().map(|x| list1(vec![x])).collect())),
    // The grid coordinates of each dimension (here, one dimension).
    "Coordinates" => Some(list1(vec![list1(xs)])),
    // The sampled values at the grid points.
    "ValuesOnGrid" => Some(list1(ys)),
    "InterpolationOrder" => Some(list1(vec![Expr::Integer(order as i128)])),
    // wolframscript reports a differentiated interpolating function's order
    // as the Derivative operator itself, and a plain one's as 0.
    "DerivativeOrder" => Some(if derivative_order == 0 {
      Expr::Integer(0)
    } else {
      call("Derivative", vec![Expr::Integer(derivative_order as i128)])
    }),
    _ => None,
  }
}

/// Evaluate InterpolatingFunction[domain, data][x_val]
/// or InterpolatingFunction[domain, data, order][x_val]
/// Rewrites the kernel's stored 1-D form `InterpolatingFunction[domain,
/// {version, …, {npoints}, {order + 1}, …}, {grid}, {values}, {methods}]`
/// (what `Save` / a notebook's saved definitions hold) into the
/// `InterpolatingFunction[domain, {{x, y}, …}, order]` form evaluated here.
/// `values` is either a plain list or `{Developer`PackedArrayForm, offsets,
/// {y, …}}`. Returns `None` for anything that is not that exact shape.
/// A kernel-layout grid `InterpolatingFunction` in two or more dimensions in
/// Woxi's `{domain, values, orders, coords}` form. Each point stores its
/// value (possibly a vector) first, then any derivatives, which the grid
/// evaluator does not use.
pub(crate) fn kernel_grid_interpolating_function(
  func_args: &[Expr],
) -> Option<Vec<Expr>> {
  let [domain, Expr::List(info), Expr::List(grids), values, _] = func_args
  else {
    return None;
  };
  let n = grids.len();
  if n < 2 || !matches!(domain, Expr::List(d) if d.len() == n) {
    return None;
  }
  let Some(Expr::List(order_plus_one)) = info.get(4) else {
    return None;
  };
  if order_plus_one.len() != n {
    return None;
  }
  let orders: Vec<Expr> = order_plus_one
    .iter()
    .map(|o| match o {
      Expr::Integer(k) if *k >= 2 => Some(Expr::Integer(*k - 1)),
      _ => None,
    })
    .collect::<Option<_>>()?;
  let sizes: Vec<usize> = grids
    .iter()
    .map(|g| match g {
      Expr::List(items) if !items.is_empty() => Some(items.len()),
      _ => None,
    })
    .collect::<Option<_>>()?;
  fn strip(v: &Expr, depth: usize) -> Option<Expr> {
    let Expr::List(items) = v else {
      return None;
    };
    if depth == 0 {
      return items.first().cloned();
    }
    Some(Expr::List(
      items
        .iter()
        .map(|i| strip(i, depth - 1))
        .collect::<Option<_>>()?,
    ))
  }
  let values = match values {
    Expr::List(items) if matches!(items.first(), Some(Expr::Identifier(t)) if t == "Developer`PackedArrayForm") =>
    {
      let [_, Expr::List(offsets), Expr::List(flat)] = items.as_slice() else {
        return None;
      };
      let offs: Vec<usize> = offsets
        .iter()
        .map(|o| match o {
          Expr::Integer(k) if *k >= 0 => Some(*k as usize),
          _ => None,
        })
        .collect::<Option<_>>()?;
      let stored = match info.get(2) {
        Some(Expr::Integer(d)) if *d >= 0 => *d as usize + 1,
        _ => 1,
      };
      let points: Vec<Expr> = offs
        .windows(2)
        .map(|w| {
          let width = w[1].checked_sub(w[0])?;
          let dim = (width / stored).max(1);
          let part: Vec<Expr> =
            flat.iter().skip(w[0]).take(dim).cloned().collect();
          (part.len() == dim).then(|| {
            if dim == 1 {
              part[0].clone()
            } else {
              Expr::List(part.into())
            }
          })
        })
        .collect::<Option<_>>()?;
      if points.len() != sizes.iter().product::<usize>() {
        return None;
      }
      fn nest(points: &[Expr], sizes: &[usize]) -> Expr {
        match sizes.split_first() {
          Some((&m, rest)) if !rest.is_empty() => {
            let chunk = points.len() / m;
            Expr::List(
              (0..m)
                .map(|i| nest(&points[i * chunk..(i + 1) * chunk], rest))
                .collect(),
            )
          }
          _ => Expr::List(points.to_vec().into()),
        }
      }
      nest(&points, &sizes)
    }
    other => strip(other, n)?,
  };
  Some(vec![
    domain.clone(),
    values,
    Expr::List(orders.into()),
    Expr::List(grids.clone()),
  ])
}

/// A kernel-layout spline `InterpolatingFunction` (`Method -> "Spline"`,
/// whose values are a `BSplineFunction`) in Woxi's spline form.
fn kernel_spline_interpolating_function(
  func_args: &[Expr],
) -> Option<Vec<Expr>> {
  let [domain, _, Expr::List(grids), values, _] = func_args else {
    return None;
  };
  let [Expr::List(xs)] = grids.as_slice() else {
    return None;
  };
  let sp =
    crate::functions::spline_interpolation::Spline::from_kernel(values, xs)?;
  Some(vec![
    domain.clone(),
    sp.to_data(),
    Expr::Integer(sp.degree as i128),
  ])
}

/// A kernel-layout `InterpolatingFunction` (as wolframscript stores it) in
/// Woxi's own `{domain, data, order}` form, when it is one Woxi reads.
pub(crate) fn kernel_interpolating_function_to_native(
  func_args: &[Expr],
) -> Option<Vec<Expr>> {
  if let Some(native) = kernel_spline_interpolating_function(func_args) {
    return Some(native);
  }
  let mut pairs = kernel_interpolating_function_to_pairs(func_args)?;
  let order = pairs.pop()?;
  Some(vec![func_args[0].clone(), Expr::List(pairs.into()), order])
}

/// A kernel InterpolatingFunction fitted to complex data stores every value
/// as `x + 0. I` when the data is in fact real; those entries read as the
/// machine real `x`, anything with a non-zero imaginary part is kept.
fn demote_zero_imaginary(e: &Expr) -> Expr {
  match crate::functions::math_ast::numeric_utils::machine_complex_shape(e) {
    Some((re, 0.0)) => Expr::Real(re),
    _ => e.clone(),
  }
}

fn kernel_interpolating_function_to_pairs(
  func_args: &[Expr],
) -> Option<Vec<Expr>> {
  let [domain, Expr::List(info), Expr::List(grids), values, _] = func_args
  else {
    return None;
  };
  let Expr::List(domain_ranges) = domain else {
    return None;
  };
  let [Expr::List(xs)] = grids.as_slice() else {
    return None;
  };
  if domain_ranges.len() != 1 {
    return None;
  }
  // Each grid point stores its value followed by `info[[3]]` derivatives:
  // `{v, v', …}` per point, or packed as `{Developer`PackedArrayForm,
  // offsets, flat}` where point i owns flat[[offsets[[i]] + 1 ;;
  // offsets[[i + 1]]]] (a vector value's components come first).
  let stored = match info.get(2) {
    Some(Expr::Integer(d)) if *d >= 0 => *d as usize + 1,
    _ => 1,
  };
  let ys: Vec<Expr> = match values {
    Expr::List(items) => match items.as_slice() {
      [Expr::Identifier(tag), Expr::List(offsets), Expr::List(flat)]
        if tag == "Developer`PackedArrayForm" =>
      {
        let offsets: Vec<usize> = offsets
          .iter()
          .map(|o| match o {
            Expr::Integer(o) if *o >= 0 => Some(*o as usize),
            _ => None,
          })
          .collect::<Option<_>>()?;
        offsets
          .windows(2)
          .map(|w| {
            let width = w[1].checked_sub(w[0])?;
            let dim = (width / stored).max(1);
            let part: Vec<Expr> = (0..dim)
              .map(|j| flat.get(w[0] + j).map(demote_zero_imaginary))
              .collect::<Option<_>>()?;
            Some(if dim == 1 {
              part[0].clone()
            } else {
              Expr::List(part.into())
            })
          })
          .collect::<Option<_>>()?
      }
      _ => items
        .iter()
        .map(|item| match item {
          Expr::List(per_point) => per_point.first().map(demote_zero_imaginary),
          other => Some(demote_zero_imaginary(other)),
        })
        .collect::<Option<_>>()?,
    },
    _ => return None,
  };
  if xs.len() != ys.len() || xs.len() < 2 {
    return None;
  }
  // Scalar values stored with their derivatives become Hermite rows.
  let hermite_rows: Option<Vec<Expr>> = (stored > 1)
    .then(|| hermite_blocks(values, xs.len(), stored))
    .flatten()
    .map(|blocks| {
      xs.iter()
        .zip(blocks)
        .map(|(x, block)| {
          let mut parts = vec![Expr::List(vec![x.clone()].into())];
          parts.extend(block);
          Expr::List(parts.into())
        })
        .collect()
    });
  let order = match info.get(4) {
    Some(Expr::List(o)) => match o.as_slice() {
      [Expr::Integer(k)] if *k >= 2 => *k - 1,
      _ => return None,
    },
    _ => return None,
  };
  let mut out: Vec<Expr> = match hermite_rows {
    Some(rows) => rows,
    None => xs
      .iter()
      .zip(ys.iter())
      .map(|(x, y)| Expr::List(vec![x.clone(), y.clone()].into()))
      .collect(),
  };
  out.push(Expr::Integer(order));
  Some(out)
}

/// The per-point `{f, f', …}` blocks of a kernel InterpolatingFunction's
/// stored values, when every point holds exactly `stored` scalars.
fn hermite_blocks(
  values: &Expr,
  n: usize,
  stored: usize,
) -> Option<Vec<Vec<Expr>>> {
  let Expr::List(items) = values else {
    return None;
  };
  let blocks: Vec<Vec<Expr>> = match items.as_slice() {
    [Expr::Identifier(tag), Expr::List(offsets), Expr::List(flat)]
      if tag == "Developer`PackedArrayForm" =>
    {
      let offsets: Vec<usize> = offsets
        .iter()
        .map(|o| match o {
          Expr::Integer(o) if *o >= 0 => Some(*o as usize),
          _ => None,
        })
        .collect::<Option<_>>()?;
      offsets
        .windows(2)
        .map(|w| {
          (w[0] <= w[1] && w[1] <= flat.len()).then(|| {
            flat
              .iter()
              .skip(w[0])
              .take(w[1] - w[0])
              .map(demote_zero_imaginary)
              .collect()
          })
        })
        .collect::<Option<_>>()?
    }
    _ => items
      .iter()
      .map(|item| match item {
        Expr::List(block) => Some(block.to_vec()),
        _ => None,
      })
      .collect::<Option<_>>()?,
  };
  (blocks.len() == n
    && blocks.iter().all(|b| {
      b.len() == stored && !b.iter().any(|v| matches!(v, Expr::List(_)))
    }))
  .then_some(blocks)
}

/// For `{{x, z}, …}` interpolation data in which some `z` is a complex
/// number, the same rows with only the real parts and with only the
/// imaginary parts. `None` for real data or any other shape.
fn complex_interpolation_parts(
  data: &Expr,
  stored: bool,
) -> Option<(Vec<Expr>, Vec<Expr>)> {
  use crate::functions::math_ast::numeric_utils::try_extract_complex_f64;
  let Expr::List(rows) = data else {
    return None;
  };
  // A complex interpolant stores every value in complex form (see
  // `interpolation_ast`), so for `stored` data the first row decides, at no
  // cost to the real data of an NDSolve solution evaluated at every plot
  // point; the raw data `Interpolation` is given is scanned in full
  let first_is_complex = matches!(rows.first(), Some(Expr::List(p))
    if p.len() == 2 && crate::functions::math_ast::contains_imaginary_unit(&p[1]));
  if stored && !first_is_complex {
    return None;
  }
  let mut any_complex = false;
  let mut re_rows = Vec::with_capacity(rows.len());
  let mut im_rows = Vec::with_capacity(rows.len());
  for row in rows {
    let Expr::List(pair) = row else {
      return None;
    };
    let [x, z] = pair.as_slice() else {
      return None;
    };
    let (re, im) = try_extract_complex_f64(z)?;
    any_complex |= crate::functions::math_ast::contains_imaginary_unit(z);
    re_rows.push(Expr::List(vec![x.clone(), Expr::Real(re)].into()));
    im_rows.push(Expr::List(vec![x.clone(), Expr::Real(im)].into()));
  }
  any_complex.then_some((re_rows, im_rows))
}

pub fn evaluate_interpolating_function(
  func_args: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if crate::functions::fem2d::is_mesh_interpolation(func_args) {
    return Ok(crate::functions::fem2d::evaluate_mesh_interpolation(
      func_args, call_args,
    ));
  }
  // A one-dimensional function threads over a list argument, at any depth.
  if let [Expr::List(points)] = call_args
    && matches!(func_args.first(), Some(Expr::List(domain)) if domain.len() == 1)
  {
    return points
      .iter()
      .map(|p| {
        evaluate_interpolating_function(func_args, std::slice::from_ref(p))
      })
      .collect::<Result<Vec<_>, _>>()
      .map(|values| Expr::List(values.into()));
  }
  // 2-D explicit-coordinate grid carrying partial-derivative orders, as left
  // by `Derivative[a, b][f]`: InterpolatingFunction[domain, grid, orders,
  // coords, {a, b}].
  // Grid form with explicit coordinates in two or more dimensions:
  // InterpolatingFunction[domain, values, {orders}, {coords}] and, as left by
  // `Derivative[d1, …][f]`, a fifth {d1, …} argument.
  if (func_args.len() == 4 || func_args.len() == 5)
    && let Expr::List(orders) = &func_args[2]
    && let Expr::List(coords) = &func_args[3]
    && orders.len() >= 2
    && coords.len() == orders.len()
    && coords.iter().all(|c| matches!(c, Expr::List(_)))
    && orders.iter().all(|o| matches!(o, Expr::Integer(_)))
  {
    let derivative: Option<Vec<usize>> = match func_args.get(4) {
      None => Some(vec![0; orders.len()]),
      Some(Expr::List(d)) if d.len() == orders.len() => d
        .iter()
        .map(|e| match e {
          Expr::Integer(k) if *k >= 0 => Some(*k as usize),
          _ => None,
        })
        .collect(),
      _ => None,
    };
    if let Some(derivative) = derivative {
      return evaluate_grid_interpolating_function(
        &func_args[0],
        &func_args[1],
        orders,
        coords,
        &derivative,
        call_args,
      );
    }
  }
  // A kernel-layout spline over two variables.
  if func_args.len() == 5
    && let [a, b] = call_args
    && let (Some(x), Some(y)) = (
      crate::functions::math_ast::try_eval_to_f64(a),
      crate::functions::math_ast::try_eval_to_f64(b),
    )
    && let Some(v) =
      crate::functions::spline_interpolation::Spline::eval_kernel_2d(
        &func_args[3],
        x,
        y,
      )
  {
    return Ok(Expr::Real(v));
  }
  // A kernel-layout grid in two or more dimensions.
  if func_args.len() == 5
    && let Some(native) = kernel_grid_interpolating_function(func_args)
  {
    return evaluate_interpolating_function(&native, call_args);
  }
  if func_args.len() == 5
    && let Some(native) = kernel_spline_interpolating_function(func_args)
  {
    return evaluate_interpolating_function(&native, call_args);
  }
  if func_args.len() == 5
    && let Some(mut pairs) = kernel_interpolating_function_to_pairs(func_args)
  {
    let order = pairs.pop().unwrap_or(Expr::Integer(3));
    return evaluate_interpolating_function(
      &[func_args[0].clone(), Expr::List(pairs.into()), order],
      call_args,
    );
  }

  if !(2..=4).contains(&func_args.len()) || call_args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InterpolatingFunction expects domain and data, called with one argument"
        .into(),
    ));
  }

  // Tensor-valued data: `{{x, {…}}, …}` interpolates component-wise.
  if let Expr::List(rows) = &func_args[1]
    && matches!(rows.first(), Some(Expr::List(p)) if p.len() == 2 && matches!(&p[1], Expr::List(_)))
    && !matches!(&call_args[0], Expr::String(_))
  {
    return evaluate_tensor_interpolating_function(func_args, rows, call_args);
  }

  // Complex-valued `{{x, z}, …}` data interpolates the real and imaginary
  // parts on their own (interpolation is linear in the values) and
  // recombines them, derivatives included.
  if let Some((re_rows, im_rows)) =
    complex_interpolation_parts(&func_args[1], true)
    && !matches!(&call_args[0], Expr::String(_))
  {
    let with_rows = |rows: Vec<Expr>| {
      let mut args = func_args.to_vec();
      args[1] = Expr::List(rows.into());
      evaluate_interpolating_function(&args, call_args)
    };
    let re = with_rows(re_rows)?;
    let im = with_rows(im_rows)?;
    return crate::evaluator::evaluate_expr_to_expr(&plus2(
      re,
      times2(id_expr("I"), im),
    ));
  }

  // Values that are not numbers (`a`, `0.5 ± 0.05`): interpolation is
  // linear in the values, so the result is each value times the
  // interpolant of the data that is 1 at its point and 0 elsewhere.
  //
  // The numeric path is tried first, so that the long numeric tables of
  // NDSolve solutions are never scanned for such values.
  let numeric = evaluate_interpolating_function_values(func_args, call_args);
  if numeric.is_ok() {
    return numeric;
  }
  if let Expr::List(rows) = &func_args[1]
    && !matches!(&call_args[0], Expr::String(_))
    && rows
      .iter()
      .any(|r| symbolic_interpolation_value(r).is_some())
  {
    let mut terms = Vec::new();
    for i in 0..rows.len() {
      let unit: Vec<Expr> = rows
        .iter()
        .enumerate()
        .map(|(j, r)| match r {
          Expr::List(p) if p.len() == 2 => Expr::List(
            vec![p[0].clone(), Expr::Integer(i128::from(i == j))].into(),
          ),
          other => other.clone(),
        })
        .collect();
      let mut unit_args = func_args.to_vec();
      unit_args[1] = Expr::List(unit.into());
      let weight = evaluate_interpolating_function(&unit_args, call_args)?;
      if matches!(weight, Expr::Integer(0))
        || matches!(weight, Expr::Real(w) if w == 0.0)
      {
        continue;
      }
      let Expr::List(p) = &rows[i] else {
        continue;
      };
      terms.push(times2(weight, p[1].clone()));
    }
    return crate::evaluator::evaluate_expr_to_expr(&plus(terms));
  }

  numeric
}

/// A one-dimensional InterpolatingFunction with numeric values at a point.
fn evaluate_interpolating_function_values(
  func_args: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let data = &func_args[1];
  let order = if func_args.len() >= 3 {
    match &func_args[2] {
      Expr::Integer(n) => *n as usize,
      _ => 3,
    }
  } else {
    1 // Default for NDSolve-generated (backwards compat)
  };
  // A fourth argument is the derivative order, as left by `f'`.
  let derivative_order = match func_args.get(3) {
    Some(Expr::Integer(n)) if *n >= 0 => *n as usize,
    _ => 0,
  };

  // An interpolating B-spline (`Method -> "Spline"`).
  if let Some(sp) =
    crate::functions::spline_interpolation::Spline::from_data(data, order)
  {
    if let Expr::String(prop) = &call_args[0] {
      if let Some(result) = sp.property(prop) {
        return Ok(result);
      }
    } else {
      let x_expr = crate::evaluator::evaluate_expr_to_expr(&call_args[0])?;
      // A spline extrapolates outside the data without a message
      // (wolframscript gives `InterpolatingFunction::dmval` only for the
      // local-polynomial methods).
      if let Ok(x) = interp_value_to_f64(&x_expr) {
        return Ok(Expr::Real(sp.eval(x, derivative_order)));
      }
    }
    return Ok(Expr::CurriedCall {
      func: Box::new(unevaluated("InterpolatingFunction", func_args)),
      args: call_args.to_vec(),
    });
  }

  // Values with derivatives: `{{x}, f, f', …}` rows.
  if let Expr::List(items) = data
    && let Some(rows) =
      crate::functions::hermite_interpolation::parse_rows(items)
  {
    use crate::functions::hermite_interpolation as hermite;
    if let Expr::String(prop) = &call_args[0] {
      if let Some(result) = hermite::property(&rows, order, prop) {
        return Ok(result);
      }
    } else {
      let x = crate::evaluator::evaluate_expr_to_expr(&call_args[0])?;
      if let Some(v) = hermite::evaluate(&rows, order, derivative_order, &x)? {
        return Ok(v);
      }
    }
    return Ok(Expr::CurriedCall {
      func: Box::new(unevaluated("InterpolatingFunction", func_args)),
      args: call_args.to_vec(),
    });
  }

  // Property access: InterpolatingFunction[…]["Domain"], ["Grid"], etc.
  if let Expr::String(prop) = &call_args[0]
    && let Some(result) =
      interpolating_function_property(data, order, derivative_order, prop)
  {
    return Ok(result);
  }

  let x_val_expr = crate::evaluator::evaluate_expr_to_expr(&call_args[0])?;
  // Rationals and other exact numbers count as arguments too, not just
  // machine numbers: Interpolation[{1, 4, 9}][5/2] is 25/4.
  let Ok(x_val) = interp_value_to_f64(&x_val_expr) else {
    // Can't evaluate symbolically — return unevaluated
    return Ok(Expr::CurriedCall {
      func: Box::new(unevaluated("InterpolatingFunction", func_args)),
      args: call_args.to_vec(),
    });
  };

  // Data is a list of {x, y} pairs
  let Expr::List(data_points) = data else {
    return Err(InterpreterError::EvaluationError(
      "InterpolatingFunction: invalid data format".into(),
    ));
  };

  let n = data_points.len();
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "InterpolatingFunction: not enough data points".into(),
    ));
  }

  // Outside the data range the boundary piece is extended rather than the
  // value clamped, which is what wolframscript does after warning about it.
  if let (Ok((x_lo, _)), Ok((x_hi, _))) = (
    extract_point(&data_points[0]),
    extract_point(&data_points[n - 1]),
  ) && outside_interpolation_domain(x_val, x_lo, x_hi)
  {
    emit_dmval(std::slice::from_ref(&x_val_expr));
  }

  if n == 1 {
    // A single data point is a constant interpolation: return the stored y
    // (preserving its Integer/Real type) for any input; its derivative is 0.
    if derivative_order > 0 {
      return Ok(Expr::Integer(0));
    }
    if let Expr::List(pair) = &data_points[0]
      && pair.len() == 2
    {
      return Ok(pair[1].clone());
    }
    let (_, y) = extract_point(&data_points[0])?;
    return Ok(real_or_integer(y));
  }

  // Extract all points for interpolation
  let (x_first, _) = extract_point(&data_points[0])?;
  let (x_last, _) = extract_point(&data_points[n - 1])?;

  // Binary search for the interval containing x; a point outside the data
  // range picks up the nearest interval and extrapolates along it.
  let idx = find_interval(data_points, x_val.max(x_first).min(x_last), n)?;

  // Check for exact grid point match — return the stored y-value directly
  // to preserve original types (e.g. Integer for ListInterpolation with integer data).
  // The data is sorted, so only the two ends of the located interval can match.
  if derivative_order == 0 {
    for pt in &data_points[idx..=(idx + 1).min(n - 1)] {
      if let Expr::List(pair) = pt
        && pair.len() == 2
        && let Ok(xp) = interp_value_to_f64(&pair[0])
        && (xp - x_val).abs() < 1e-15
      {
        return Ok(pair[1].clone());
      }
    }
  }

  let eff_order = if order == 1 || n <= 2 {
    1
  } else {
    order.min(n - 1)
  };
  let (start, end) = lagrange_window(n, idx, eff_order);

  // `f'` differentiates the local polynomial piece, so the derivative is
  // exact wherever the interpolation itself is. Exact data and an exact
  // query point take the symbolic route to preserve exactness; anything
  // else (machine-precision NDSolve output, the common case) takes the
  // fast numeric route so sampling a phase portrait's velocity stays cheap.
  if derivative_order > 0 {
    if is_exact_number(&x_val_expr)
      && let Some(exact) = interpolating_derivative_value_exact(
        data_points,
        &x_val_expr,
        start,
        end,
        derivative_order,
      )?
    {
      return Ok(exact);
    }
    return interpolating_derivative_value_numeric(
      data_points,
      x_val,
      start,
      end,
      derivative_order,
    );
  }

  // An exact query point over exact data interpolates exactly, the way
  // wolframscript reports Interpolation[{1, 4, 9}][5/2] as 25/4.
  if is_exact_number(&x_val_expr)
    && let Some(exact) =
      lagrange_interpolate_exact(data_points, &x_val_expr, start, end)?
  {
    return Ok(exact);
  }

  if eff_order == 1 {
    // Linear interpolation
    let (x0, y0) = extract_point(&data_points[idx])?;
    let (x1, y1) = extract_point(&data_points[idx + 1])?;
    let t = if (x1 - x0).abs() > 1e-15 {
      (x_val - x0) / (x1 - x0)
    } else {
      0.0
    };
    let y_val = y0 + t * (y1 - y0);
    Ok(real_or_integer(y_val))
  } else {
    // Order >= 2: piecewise local polynomial interpolation through the
    // nearest (order + 1) points. This matches wolframscript's default
    // `Interpolation`, which reproduces any polynomial of degree <= order
    // exactly — unlike a natural cubic spline, whose zero-curvature
    // boundary conditions distort the fit (e.g. x^2 data would not yield
    // exact values).
    let y_val = lagrange_interpolate(data_points, x_val, n, idx, eff_order)?;
    Ok(real_or_integer(y_val))
  }
}

/// `InterpolatingFunctionDomain[interp]`, from the
/// `DifferentialEquations`InterpolatingFunctionAnatomy`` package. Reports the
/// same domain as `interp["Domain"]` (a list of `{min, max}` pairs, one per
/// independent variable), so it delegates to `evaluate_interpolating_function`
/// rather than re-deriving the domain.
pub fn interpolating_function_domain_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let interp = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  if let Expr::FunctionCall {
    name,
    args: func_args,
  } = &interp
    && name == "InterpolatingFunction"
  {
    return evaluate_interpolating_function(func_args, &[str_expr("Domain")]);
  }
  Ok(unevaluated("InterpolatingFunctionDomain", &[interp]))
}

/// Find the interval index for x_val using binary search.
/// Returns idx such that x[idx] <= x_val <= x[idx+1].
fn find_interval(
  data_points: &[Expr],
  x_val: f64,
  n: usize,
) -> Result<usize, InterpreterError> {
  let mut lo = 0usize;
  let mut hi = n - 1;
  while lo < hi - 1 {
    let mid = usize::midpoint(lo, hi);
    let (x_mid, _) = extract_point(&data_points[mid])?;
    if x_val < x_mid {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  Ok(lo)
}

/// The (order+1)-point stencil for the interval starting at `idx`, as centered
/// as possible on it; it must contain both interval endpoints or the local
/// polynomial would extrapolate and miss the next grid value.
fn lagrange_window(n: usize, idx: usize, order: usize) -> (usize, usize) {
  let needed = order + 1;
  let start = idx
    .saturating_sub((order.max(1) - 1) / 2)
    .min(n.saturating_sub(needed));
  (start, (start + needed).min(n))
}

/// The local Lagrange polynomial over the stencil `data_points[start..end]`,
/// written in `var`. `require_exact` rejects machine numbers, which is what
/// the exact-value path wants; the derivative path takes them.
fn lagrange_polynomial(
  data_points: &[Expr],
  var: &Expr,
  start: usize,
  end: usize,
  require_exact: bool,
) -> Option<Expr> {
  let usable = |e: &Expr| {
    is_exact_number(e) || (!require_exact && matches!(e, Expr::Real(_)))
  };
  let mut xs: Vec<Expr> = Vec::with_capacity(end - start);
  let mut ys: Vec<Expr> = Vec::with_capacity(end - start);
  for pt in &data_points[start..end] {
    let Expr::List(pair) = pt else {
      return None;
    };
    if pair.len() != 2 || !usable(&pair[0]) || !usable(&pair[1]) {
      return None;
    }
    xs.push(pair[0].clone());
    ys.push(pair[1].clone());
  }

  let minus = |a: &Expr, b: &Expr| minus2(a.clone(), b.clone());
  let m = xs.len();
  let mut terms: Vec<Expr> = Vec::with_capacity(m);
  for i in 0..m {
    let mut factors: Vec<Expr> = vec![ys[i].clone()];
    for j in 0..m {
      if j != i {
        factors.push(div2(minus(var, &xs[j]), minus(&xs[i], &xs[j])));
      }
    }
    factors.retain(|f| !matches!(f, Expr::Integer(1)));
    terms.push(times(factors));
  }
  Some(plus(terms))
}

/// Lagrange interpolation over the stencil carried out in exact arithmetic.
/// Returns `None` when a coordinate in the stencil is not an exact number,
/// leaving the machine-precision path to handle it.
fn lagrange_interpolate_exact(
  data_points: &[Expr],
  x: &Expr,
  start: usize,
  end: usize,
) -> Result<Option<Expr>, InterpreterError> {
  match lagrange_polynomial(data_points, x, start, end, true) {
    Some(poly) => Ok(Some(crate::evaluator::evaluate_expr_to_expr(&poly)?)),
    None => Ok(None),
  }
}

/// The `derivative_order`-th derivative of the local polynomial piece,
/// evaluated at `x`, carried out in exact arithmetic. Returns `None` when a
/// coordinate in the stencil (or `x` itself) is not an exact number, leaving
/// [`interpolating_derivative_value_numeric`] to handle it.
fn interpolating_derivative_value_exact(
  data_points: &[Expr],
  x: &Expr,
  start: usize,
  end: usize,
  derivative_order: usize,
) -> Result<Option<Expr>, InterpreterError> {
  let var = id_expr("\u{2620}ifderiv\u{2620}");
  let Some(poly) = lagrange_polynomial(data_points, &var, start, end, true)
  else {
    return Ok(None);
  };
  let mut expr = crate::evaluator::evaluate_expr_to_expr(&poly)?;
  for _ in 0..derivative_order {
    expr = crate::evaluator::evaluate_expr_to_expr(&call(
      "D",
      vec![expr, var.clone()],
    ))?;
  }
  let substituted =
    crate::syntax::substitute_variable(&expr, "\u{2620}ifderiv\u{2620}", x);
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&substituted)?))
}

/// The `derivative_order`-th derivative of the local polynomial piece,
/// evaluated at `x_val`, computed directly in machine arithmetic (Newton
/// divided differences expanded to monomial coefficients, then
/// differentiated and evaluated by Horner's method).
///
/// This is the hot path for `f'[t]`/`f''[t]` on an `NDSolve`-produced
/// `InterpolatingFunction` — e.g. sampling a phase portrait's velocity
/// alongside its position. Building and simplifying a symbolic Lagrange
/// polynomial through the general evaluator for every sample point (the
/// exact-arithmetic path above) is thousands of times slower and turns a
/// `Manipulate` slider drag into a multi-second stall, so machine-precision
/// data — the overwhelmingly common case — never takes that path.
fn interpolating_derivative_value_numeric(
  data_points: &[Expr],
  x_val: f64,
  start: usize,
  end: usize,
  derivative_order: usize,
) -> Result<Expr, InterpreterError> {
  let mut xs = Vec::with_capacity(end - start);
  let mut coeffs = Vec::with_capacity(end - start);
  for pt in &data_points[start..end] {
    let (x, y) = extract_point(pt)?;
    xs.push(x);
    coeffs.push(y);
  }
  let m = xs.len();

  // Newton divided differences: coeffs[k] becomes f[x_0, ..., x_k].
  for j in 1..m {
    for i in (j..m).rev() {
      coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (xs[i] - xs[i - j]);
    }
  }

  // Evaluate the nested Newton form c[0] + (x-x0)(c[1] + (x-x1)(c[2] + ...))
  // as a truncated Taylor series in h = x - x_val, so the derivative comes
  // out without expanding monomial coefficients (which cancels
  // catastrophically for high `InterpolationOrder`).
  let d = derivative_order;
  if d >= m {
    return Ok(Expr::Real(0.0));
  }
  let mut taylor = vec![0.0; d + 1];
  taylor[0] = coeffs[m - 1];
  for k in (0..m - 1).rev() {
    let shift = x_val - xs[k];
    // taylor = taylor * (shift + h) + coeffs[k]
    for i in (0..=d).rev() {
      let carried = if i > 0 { taylor[i - 1] } else { 0.0 };
      taylor[i] = taylor[i] * shift + carried;
    }
    taylor[0] += coeffs[k];
  }
  let factorial: f64 = (1..=d).map(|i| i as f64).product();
  Ok(Expr::Real(taylor[d] * factorial))
}

/// Lagrange polynomial interpolation using (order+1) nearest points.
fn lagrange_interpolate(
  data_points: &[Expr],
  x_val: f64,
  n: usize,
  idx: usize,
  order: usize,
) -> Result<f64, InterpreterError> {
  let (start, end) = lagrange_window(n, idx, order);

  let mut xs = Vec::with_capacity(end - start);
  let mut ys = Vec::with_capacity(end - start);
  for pt in &data_points[start..end] {
    let (x, y) = extract_point(pt)?;
    xs.push(x);
    ys.push(y);
  }

  // Lagrange basis polynomials
  let m = xs.len();
  let mut result = 0.0;
  for i in 0..m {
    let mut basis = 1.0;
    for j in 0..m {
      if j != i {
        basis *= (x_val - xs[j]) / (xs[i] - xs[j]);
      }
    }
    result += ys[i] * basis;
  }
  Ok(result)
}

/// Extract (x, y) from a List[x, y] expression
fn extract_point(expr: &Expr) -> Result<(f64, f64), InterpreterError> {
  if let Expr::List(items) = expr
    && items.len() == 2
  {
    let x = interp_value_to_f64(&items[0])?;
    let y = interp_value_to_f64(&items[1])?;
    return Ok((x, y));
  }
  Err(InterpreterError::EvaluationError(
    "InterpolatingFunction: invalid data point format".into(),
  ))
}

// ─── First-order linear PDE in two variables ──────────────────────────

/// Recognise `a*D[f[x,y], x] + b*D[f[x,y], y] == c*f[x,y]` (and its
/// equivalent forms) and return the closed-form solution
/// `{{f -> Function[{x, y}, E^((c/a)*x) * C[1][y - (b/a)*x]]}}`.
///
/// Accepts the input in the form Wolfram emits after evaluating
/// `D[f[x,y], x]/f[x,y] + 3 D[f[x,y], y]/f[x,y] == 2` — i.e. each
/// derivative term divided by `f[x,y]` and the RHS being the constant
/// `c`. The implementation collects all top-level Plus terms on both
/// sides of the equation, classifies each as either a constant, a
/// derivative-over-`f[x,y]` term, or a multiple of `f[x,y]`, and
/// solves for the (a, b, c) triple.
fn try_linear_first_order_pde_body(
  eqn: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<Expr> {
  let (lhs, rhs) = pde_split_equation(eqn)?;
  // Move everything to the LHS: lhs - rhs == 0. Each term then
  // contributes a signed coefficient.
  let mut a = 0i128; // coefficient of D[f[x,y], x] / f[x,y]
  let mut b = 0i128; // coefficient of D[f[x,y], y] / f[x,y]
  let mut c = 0i128; // coefficient of constant (negated; we want a*fx + b*fy = c*f)
  collect_pde_terms(&lhs, fname, xn, yn, 1, &mut a, &mut b, &mut c)?;
  collect_pde_terms(&rhs, fname, xn, yn, -1, &mut a, &mut b, &mut c)?;
  // Equation as gathered: a*fx/f + b*fy/f - c == 0  ⇒  a*fx + b*fy == c*f.
  let c_eff = -c; // -c was accumulated; flip sign back.
  if a == 0 || a != 1 {
    // The closed form below assumes a == 1. Restrict to that shape;
    // generalised rationals require a Rational arithmetic path.
    return None;
  }
  // Build the body: E^(c*x) * C[1][y - b*x]
  let n_var = |s: &str| id_expr(s);
  let exp_part = if c_eff == 0 {
    Expr::Integer(1)
  } else {
    let exponent = if c_eff == 1 {
      n_var(xn)
    } else {
      times2(Expr::Integer(c_eff), n_var(xn))
    };
    pow2(const_expr("E"), exponent)
  };
  // Argument to C[1]: y - b*x  (or just y when b == 0).
  let c1_arg = if b == 0 {
    n_var(yn)
  } else {
    let bx = if b == 1 {
      n_var(xn)
    } else {
      times2(Expr::Integer(-b), n_var(xn))
    };
    plus2(bx, n_var(yn))
  };
  let c1_applied = Expr::CurriedCall {
    func: Box::new(call1("C", Expr::Integer(1))),
    args: vec![c1_arg],
  };
  let body = if matches!(&exp_part, Expr::Integer(1)) {
    c1_applied
  } else {
    times2(exp_part, c1_applied)
  };
  Some(body)
}

/// Recognise the Euler-type PDE `x*D[f[x,y], x] + y*D[f[x,y], y] == c`
/// (constant c) and return the body
/// `c*Log[x] + C[1][y/x]` of the closed-form solution.
/// Recognise `a*D[f[x,y], x] + b*D[f[x,y], y] == c` (constant integer
/// coefficients on bare derivatives, integer RHS) and return the body
/// `(c/a)*x + C[1][y - (b/a)*x]`. Inhomogeneous (c ≠ 0) and homogeneous
/// (c = 0) cases are both handled.
fn try_direct_linear_pde_body(
  eqn: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<Expr> {
  let (lhs, rhs) = pde_split_equation(eqn)?;
  // LHS: collect coefficients of Fx and Fy via Plus walking. Reject
  // shapes that don't fit (e.g. divided-by-f, mixed parameters, etc.).
  let mut a = 0i128;
  let mut b = 0i128;
  collect_direct_pde_terms(&lhs, fname, xn, yn, 1, &mut a, &mut b)?;
  // RHS must be an integer constant for the closed form below.
  let mut c = match &rhs {
    Expr::Integer(n) => *n,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) => -*n,
      _ => return None,
    },
    _ => return None,
  };
  // If we accumulated any constant terms on the LHS, fold them into
  // -c on the RHS by negating: a*Fx + b*Fy + k == c  ⇒  a*Fx + b*Fy
  // == c - k. We don't currently parse a separate `k` slot, so any
  // non-derivative LHS term aborts the recognition above.
  let _ = &mut c;
  if a == 0 {
    return None; // No Fx term means no characteristic to integrate along.
  }
  let n_var = |s: &str| id_expr(s);
  // Argument to C[1]: y - (b/a)*x.
  let c1_arg = if b == 0 {
    n_var(yn)
  } else {
    let coeff = make_neg_b_over_a(b, a);
    let bx = match &coeff {
      Expr::Integer(1) => n_var(xn),
      Expr::Integer(n) => times2(Expr::Integer(*n), n_var(xn)),
      other => times2(other.clone(), n_var(xn)),
    };
    plus2(n_var(yn), bx)
  };
  let c1_applied = Expr::CurriedCall {
    func: Box::new(call1("C", Expr::Integer(1))),
    args: vec![c1_arg],
  };
  // Inhomogeneous head term: (c/a)*x.
  if c == 0 {
    return Some(c1_applied);
  }
  let coeff = make_c_over_a(c, a);
  let head_term = match &coeff {
    Expr::Integer(1) => n_var(xn),
    Expr::Integer(n) => times2(Expr::Integer(*n), n_var(xn)),
    other => times2(other.clone(), n_var(xn)),
  };
  Some(plus2(head_term, c1_applied))
}

/// `-b/a` reduced to either an `Integer` (if `a` divides `b`) or a
/// `Rational` literal. Sign is folded into the numerator.
fn make_neg_b_over_a(b: i128, a: i128) -> Expr {
  use crate::functions::math_ast::make_rational;
  let (num, den) = rat_reduce(-b, a);
  if den == 1 {
    Expr::Integer(num)
  } else {
    make_rational(num, den)
  }
}

fn make_c_over_a(c: i128, a: i128) -> Expr {
  use crate::functions::math_ast::make_rational;
  let (num, den) = rat_reduce(c, a);
  if den == 1 {
    Expr::Integer(num)
  } else {
    make_rational(num, den)
  }
}

/// Walk a Plus chain and accumulate signed integer coefficients of
/// bare `Derivative[1,0][f][x,y]` and `Derivative[0,1][f][x,y]` terms.
/// Anything else (constants, divided-by-f terms, mixed factors) bails.
fn collect_direct_pde_terms(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
  sign: i128,
  a: &mut i128,
  b: &mut i128,
) -> Option<()> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for arg in args {
        collect_direct_pde_terms(arg, fname, xn, yn, sign, a, b)?;
      }
      Some(())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_direct_pde_terms(left, fname, xn, yn, sign, a, b)?;
      collect_direct_pde_terms(right, fname, xn, yn, sign, a, b)
    }
    _ => {
      let (coeff, kind) = classify_direct_pde_term(expr, fname, xn, yn)?;
      let signed = sign * coeff;
      match kind {
        PdeTerm::Fx => *a += signed,
        PdeTerm::Fy => *b += signed,
        PdeTerm::Const => return None, // not allowed in this branch
      }
      Some(())
    }
  }
}

fn classify_direct_pde_term(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<(i128, PdeTerm)> {
  // Bare derivative call (coefficient 1).
  if let Some(kind) = classify_derivative_call(expr, fname, xn, yn) {
    return Some((1, kind));
  }
  // Times[coeff, Derivative…] with integer coefficient.
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
  {
    let mut coeff = 1i128;
    let mut deriv_kind: Option<PdeTerm> = None;
    for factor in args {
      if let Expr::Integer(n) = factor {
        coeff *= *n;
        continue;
      }
      if let Some(kind) = classify_derivative_call(factor, fname, xn, yn) {
        if deriv_kind.is_some() {
          return None;
        }
        deriv_kind = Some(kind);
        continue;
      }
      return None;
    }
    if let Some(kind) = deriv_kind {
      return Some((coeff, kind));
    }
  }
  None
}

fn try_euler_pde_body(
  eqn: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<Expr> {
  let (lhs, rhs) = pde_split_equation(eqn)?;
  // Walk the LHS Plus chain: expect exactly two terms — `x * Fx` and
  // `y * Fy`, in either order. RHS must be an integer constant.
  let mut saw_fx = false;
  let mut saw_fy = false;
  walk_euler_terms(&lhs, fname, xn, yn, &mut saw_fx, &mut saw_fy)?;
  if !(saw_fx && saw_fy) {
    return None;
  }
  let c = match &rhs {
    Expr::Integer(n) => *n,
    _ => return None,
  };
  let n_var = |s: &str| id_expr(s);
  // Build c*Log[x] + C[1][y/x].
  let log_x = call1("Log", n_var(xn));
  let y_over_x = div2(n_var(yn), n_var(xn));
  let c1_applied = Expr::CurriedCall {
    func: Box::new(call1("C", Expr::Integer(1))),
    args: vec![y_over_x],
  };
  // `c == 0` drops the logarithm entirely; keeping `0*Log[x]` would leak an
  // unfolded zero term into the solution.
  if c == 0 {
    return Some(c1_applied);
  }
  let log_term = if c == 1 {
    log_x
  } else {
    times2(Expr::Integer(c), log_x)
  };
  let body = plus2(log_term, c1_applied);
  Some(body)
}

fn walk_euler_terms(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
  saw_fx: &mut bool,
  saw_fy: &mut bool,
) -> Option<()> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for arg in args {
        walk_euler_terms(arg, fname, xn, yn, saw_fx, saw_fy)?;
      }
      Some(())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      walk_euler_terms(left, fname, xn, yn, saw_fx, saw_fy)?;
      walk_euler_terms(right, fname, xn, yn, saw_fx, saw_fy)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Look for exactly one Identifier matching xn or yn and exactly
      // one Derivative call on f.
      let mut coord: Option<&str> = None;
      let mut deriv_kind: Option<PdeTerm> = None;
      for factor in args {
        if let Expr::Identifier(s) = factor {
          if s == xn || s == yn {
            if coord.is_some() {
              return None;
            }
            coord = Some(if s == xn { xn } else { yn });
            continue;
          }
          return None;
        }
        if let Some(kind) = classify_derivative_call(factor, fname, xn, yn) {
          if deriv_kind.is_some() {
            return None;
          }
          deriv_kind = Some(kind);
          continue;
        }
        return None;
      }
      match (coord, deriv_kind) {
        (Some(c), Some(PdeTerm::Fx)) if c == xn => {
          if *saw_fx {
            return None;
          }
          *saw_fx = true;
          Some(())
        }
        (Some(c), Some(PdeTerm::Fy)) if c == yn => {
          if *saw_fy {
            return None;
          }
          *saw_fy = true;
          Some(())
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Wrap a PDE body expression as the rule the test expects.
/// `return_call_form` selects between `f[x, y] -> body` and
/// `f -> Function[{x, y}, body]`.
fn wrap_pde_solution(
  body: Expr,
  fname: &str,
  xn: &str,
  yn: &str,
  return_call_form: bool,
) -> Expr {
  let n_var = |s: &str| id_expr(s);
  let rule = if return_call_form {
    rule_expr(call(fname, vec![n_var(xn), n_var(yn)]), body)
  } else {
    rule_expr(
      id_expr(fname),
      call(
        "Function",
        vec![Expr::List(vec![n_var(xn), n_var(yn)].into()), body],
      ),
    )
  };
  Expr::List(vec![Expr::List(vec![rule].into())].into())
}

/// Pull `(lhs, rhs)` out of an `Equal` expression, accepting either the
/// `Comparison` AST node or a literal `Equal[…]` FunctionCall.
fn pde_split_equation(eqn: &Expr) -> Option<(Expr, Expr)> {
  match eqn {
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

/// Walk a Plus chain in `expr`, classifying each term by its shape and
/// adding its (signed) integer coefficient to the appropriate slot.
/// Returns `None` if any term doesn't fit the recognised PDE shape.
fn collect_pde_terms(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
  sign: i128,
  a: &mut i128,
  b: &mut i128,
  c: &mut i128,
) -> Option<()> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for arg in args {
        collect_pde_terms(arg, fname, xn, yn, sign, a, b, c)?;
      }
      Some(())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_pde_terms(left, fname, xn, yn, sign, a, b, c)?;
      collect_pde_terms(right, fname, xn, yn, sign, a, b, c)
    }
    _ => {
      let (coeff, kind) = classify_pde_term(expr, fname, xn, yn)?;
      let signed = sign * coeff;
      match kind {
        PdeTerm::Fx => *a += signed,
        PdeTerm::Fy => *b += signed,
        PdeTerm::Const => *c += signed,
      }
      Some(())
    }
  }
}

#[derive(Clone, Copy)]
enum PdeTerm {
  Fx,
  Fy,
  Const,
}

/// Decompose a single Plus term into (integer coefficient, kind). The
/// recognised shapes are:
///   * integer constant                        -> Const
///   * c * f[x,y]                              -> Const (consumes the f)
///   * c * Derivative[1,0][f][x,y] / f[x,y]    -> Fx
///   * c * Derivative[0,1][f][x,y] / f[x,y]    -> Fy
///   * Derivative[…][f][x,y] / f[x,y]          -> Fx/Fy with c = 1
fn classify_pde_term(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<(i128, PdeTerm)> {
  // Plain integer constant (RHS of the PDE).
  if let Expr::Integer(n) = expr {
    return Some((*n, PdeTerm::Const));
  }
  // `c * f[x,y]` (rare on input but possible after rearrangement).
  if is_f_at_xy(expr, fname, xn, yn) {
    return Some((1, PdeTerm::Const));
  }
  // Single `Derivative[…][f][x,y] / f[x,y]` (coefficient 1).
  if let Some(kind) = classify_derivative_over_f(expr, fname, xn, yn) {
    return Some((1, kind));
  }
  // `c * Derivative[…][f][x,y] / f[x,y]` represented as
  // `Times[c, Derivative…, Power[f[x,y], -1]]` (or any factor order).
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
  {
    let mut coeff = 1i128;
    let mut deriv_kind: Option<PdeTerm> = None;
    let mut saw_inverse_f = false;
    for factor in args {
      if let Expr::Integer(n) = factor {
        coeff *= *n;
        continue;
      }
      if let Some(kind) = classify_derivative_call(factor, fname, xn, yn) {
        if deriv_kind.is_some() {
          return None; // Two derivatives in one term — not a linear PDE term.
        }
        deriv_kind = Some(kind);
        continue;
      }
      if is_inverse_f(factor, fname, xn, yn) {
        if saw_inverse_f {
          return None;
        }
        saw_inverse_f = true;
        continue;
      }
      // Unknown factor — give up.
      return None;
    }
    if saw_inverse_f && let Some(kind) = deriv_kind {
      return Some((coeff, kind));
    }
  }
  None
}

/// The derivative order `Derivative[i, j][f][x, y]` names, for a call on
/// exactly `f[x, y]`. `None` for anything else.
fn derivative_order_of(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<(i128, i128)> {
  if let Expr::CurriedCall {
    func,
    args: call_args,
  } = expr
    && call_args.len() == 2
    && let Expr::Identifier(x_arg) = &call_args[0]
    && let Expr::Identifier(y_arg) = &call_args[1]
    && x_arg == xn
    && y_arg == yn
    && let Expr::CurriedCall {
      func: deriv,
      args: f_args,
    } = func.as_ref()
    && f_args.len() == 1
    && let Expr::Identifier(fa) = &f_args[0]
    && fa == fname
    && let Expr::FunctionCall {
      name: dn,
      args: dargs,
    } = deriv.as_ref()
    && dn == "Derivative"
    && dargs.len() == 2
    && let (Expr::Integer(di), Expr::Integer(dj)) = (&dargs[0], &dargs[1])
  {
    return Some((*di, *dj));
  }
  None
}

/// Recognise the homogeneous second-order PDE with constant coefficients
/// `a*D[f, x, x] + b*D[f, x, y] + c*D[f, y, y] == 0` and return the body of
/// its general solution.
///
/// A solution `f = C[λ*x + y]` turns the equation into `a λ² + b λ + c = 0`
/// for the characteristic slope λ, so the general solution is one arbitrary
/// function per root. Wolfram reports the two in the order their
/// *reciprocals* μ = 1/λ take in canonical order — the roots of the
/// reversed polynomial `c μ² + b μ + a` — which is what decides whether
/// `C[1]` gets `I*x + y` or `(-I)*x + y`:
///
/// ```wolfram
/// DSolveValue[D[u[x, y], x, x] + D[u[x, y], y, y] == 0, u, {x, y}]
/// (* Function[{x, y}, C[1][I*x + y] + C[2][(-I)*x + y]] *)
/// ```
///
/// A repeated root gives `C[1][λ*x + y] + x*C[2][λ*x + y]`. Equations with
/// no `D[f, x, x]` or no `D[f, y, y]` term have a characteristic along an
/// axis, which Wolfram writes in a different (unnormalised) shape; those are
/// left to the caller's other recognisers.
fn try_second_order_constant_pde_body(
  eqn: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<Expr> {
  let (lhs, rhs) = pde_split_equation(eqn)?;
  // Everything moves to the left: each derivative contributes its signed
  // coefficient, and anything else at all disqualifies the equation.
  let mut coeffs: [Expr; 3] =
    [Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)];
  collect_second_order_pde_terms(&lhs, fname, xn, yn, false, &mut coeffs)?;
  collect_second_order_pde_terms(&rhs, fname, xn, yn, true, &mut coeffs)?;
  let [a, b, c] = coeffs;
  let is_zero = |e: &Expr| matches!(e, Expr::Integer(0));
  if is_zero(&a) || is_zero(&c) {
    return None;
  }

  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e);
  let disc = eval(minus2(
    pow2(b.clone(), Expr::Integer(2)),
    times2(Expr::Integer(4), times2(c.clone(), a.clone())),
  ));
  let sqrt_disc = eval(call1("Sqrt", disc.clone()));
  // The quadratic formula, over the leading coefficient of whichever
  // polynomial is asked for: `a λ² + b λ + c` for the slopes themselves,
  // `c μ² + b μ + a` for their reciprocals.
  let root = |lead: &Expr, sign: i128| {
    eval(div2(
      plus2(
        times2(Expr::Integer(-1), b.clone()),
        times2(Expr::Integer(sign), sqrt_disc.clone()),
      ),
      times2(Expr::Integer(2), lead.clone()),
    ))
  };
  // `λ*x + y`, with the coefficient folded in by the evaluator so that
  // λ = 1 leaves a bare `x`.
  let arg_of =
    |lambda: Expr| eval(plus2(times2(lambda, id_expr(xn)), id_expr(yn)));
  let c_of = |k: i128, arg: Expr| Expr::CurriedCall {
    func: Box::new(call1("C", Expr::Integer(k))),
    args: vec![arg],
  };

  if is_zero(&disc) {
    // A repeated root: the second solution picks up a factor of x.
    let arg = arg_of(root(&a, 1));
    return Some(plus2(
      c_of(1, arg.clone()),
      times2(id_expr(xn), c_of(2, arg)),
    ));
  }
  // Taking the reciprocal turns the `+` root into the `-` one, so the
  // canonical order of the reciprocals decides which slope `C[1]` gets.
  let mu_plus_first = crate::functions::list_helpers_ast::compare_exprs(
    &root(&c, 1),
    &root(&c, -1),
  ) > 0;
  let (first, second) = if mu_plus_first { (-1, 1) } else { (1, -1) };
  Some(plus2(
    c_of(1, arg_of(root(&a, first))),
    c_of(2, arg_of(root(&a, second))),
  ))
}

/// Walk a Plus chain, adding each term's coefficient to the `xx`, `xy` and
/// `yy` slots of `coeffs`. `negate` flips the sign (for the right-hand side
/// of the equation). `None` for a term that is not a constant multiple of
/// one of those three second derivatives.
fn collect_second_order_pde_terms(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
  negate: bool,
  coeffs: &mut [Expr; 3],
) -> Option<()> {
  match expr {
    Expr::Integer(0) => Some(()),
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for arg in args {
        collect_second_order_pde_terms(arg, fname, xn, yn, negate, coeffs)?;
      }
      Some(())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_second_order_pde_terms(left, fname, xn, yn, negate, coeffs)?;
      collect_second_order_pde_terms(right, fname, xn, yn, negate, coeffs)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      collect_second_order_pde_terms(left, fname, xn, yn, negate, coeffs)?;
      collect_second_order_pde_terms(right, fname, xn, yn, !negate, coeffs)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      collect_second_order_pde_terms(operand, fname, xn, yn, !negate, coeffs)
    }
    _ => {
      // A bare derivative, or a product of one with constant factors.
      let mut factors: Vec<&Expr> = Vec::new();
      match expr {
        Expr::FunctionCall { name, args } if name == "Times" => {
          factors.extend(args.iter());
        }
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left,
          right,
        } => {
          factors.push(left);
          factors.push(right);
        }
        other => factors.push(other),
      }
      let mut order: Option<(i128, i128)> = None;
      let mut coeff = Expr::Integer(1);
      for factor in factors {
        if let Some(o) = derivative_order_of(factor, fname, xn, yn) {
          if order.is_some() {
            return None; // Two derivatives multiplied — not linear.
          }
          order = Some(o);
          continue;
        }
        // Every other factor has to be free of the unknown function and of
        // both variables, or the coefficients are not constant.
        if expr_mentions(factor, fname)
          || expr_mentions(factor, xn)
          || expr_mentions(factor, yn)
        {
          return None;
        }
        coeff = times2(coeff, factor.clone());
      }
      let slot = match order? {
        (2, 0) => 0,
        (1, 1) => 1,
        (0, 2) => 2,
        _ => return None,
      };
      if negate {
        coeff = times2(Expr::Integer(-1), coeff);
      }
      let sum = plus2(coeffs[slot].clone(), coeff);
      coeffs[slot] =
        crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum);
      Some(())
    }
  }
}

/// Whether `name` occurs anywhere in `expr` as a symbol.
fn expr_mentions(expr: &Expr, name: &str) -> bool {
  expr_to_string(expr)
    .split(|c: char| !(c.is_alphanumeric() || c == '$'))
    .any(|tok| tok == name)
}

/// Match `Derivative[1,0][f][x, y] / f[x, y]` (and the (0,1) variant)
/// expressed as a Times of the derivative call and Power[f[x,y], -1].
fn classify_derivative_over_f(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<PdeTerm> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
  {
    for (i, j) in [(0usize, 1usize), (1, 0)] {
      if let Some(kind) = classify_derivative_call(&args[i], fname, xn, yn)
        && is_inverse_f(&args[j], fname, xn, yn)
      {
        return Some(kind);
      }
    }
  }
  None
}

/// Match `Derivative[1,0][f][x, y]` -> Fx, `Derivative[0,1][f][x, y]` -> Fy.
fn classify_derivative_call(
  expr: &Expr,
  fname: &str,
  xn: &str,
  yn: &str,
) -> Option<PdeTerm> {
  // Shape: CurriedCall { func: FunctionCall { name: "Derivative", args: [i, j] },
  //                     args: [Identifier f] } applied to [x, y].
  if let Expr::CurriedCall {
    func,
    args: call_args,
  } = expr
    && call_args.len() == 2
    && let Expr::Identifier(x_arg) = &call_args[0]
    && let Expr::Identifier(y_arg) = &call_args[1]
    && x_arg == xn
    && y_arg == yn
    && let Expr::CurriedCall {
      func: deriv,
      args: f_args,
    } = func.as_ref()
    && f_args.len() == 1
    && let Expr::Identifier(fa) = &f_args[0]
    && fa == fname
    && let Expr::FunctionCall {
      name: dn,
      args: dargs,
    } = deriv.as_ref()
    && dn == "Derivative"
    && dargs.len() == 2
    && let (Expr::Integer(di), Expr::Integer(dj)) = (&dargs[0], &dargs[1])
  {
    return match (*di, *dj) {
      (1, 0) => Some(PdeTerm::Fx),
      (0, 1) => Some(PdeTerm::Fy),
      _ => None,
    };
  }
  None
}

/// Match `Power[f[x, y], -1]` in either FunctionCall or BinaryOp form.
fn is_inverse_f(expr: &Expr, fname: &str, xn: &str, yn: &str) -> bool {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Power"
    && args.len() == 2
    && is_f_at_xy(&args[0], fname, xn, yn)
    && matches!(&args[1], Expr::Integer(-1))
  {
    return true;
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    right,
  } = expr
    && is_f_at_xy(left, fname, xn, yn)
    && matches!(right.as_ref(), Expr::Integer(-1))
  {
    return true;
  }
  false
}

/// Match `f[x, y]`.
fn is_f_at_xy(expr: &Expr, fname: &str, xn: &str, yn: &str) -> bool {
  matches!(
    expr,
    Expr::FunctionCall { name, args }
      if name == fname
        && args.len() == 2
        && matches!(&args[0], Expr::Identifier(s) if s == xn)
        && matches!(&args[1], Expr::Identifier(s) if s == yn)
  )
}

// ─── Coupled Linear Constant-Coefficient ODE Systems ───────────────────
//
// `DSolve[{eq1, eq2, …, ic1, ic2, …}, {y1[t], y2[t], …}, t]` for a linear,
// constant-coefficient system (the classic "two coupled oscillators" shape
// a Demonstrations-Project notebook tends to use, e.g. a Foucault pendulum
// or coupled pendulums). The single-variable branch above only recognises
// one dependent function; this extends the same closed-form machinery to a
// small system via operator-determinant elimination:
//
//   1. Each equation is linear in the y_i and their t-derivatives with
//      constant coefficients, so it can be written as a row of the
//      differential-operator matrix M(D), D = d/dt.
//   2. Every y_i solves the SAME scalar ODE Δ(D) y_i = (cofactor terms),
//      Δ = det(M(D)) — Cramer's rule for a linear operator system. Only a
//      constant forcing term is supported (enough for a homogeneous
//      physical system in equilibrium coordinates); Δ has degree ≤ 4 to
//      reuse the existing closed-form quadratic/cubic/quartic root finder.
//   3. For each simple root λ of Δ, the amplitude ratios between the y_i
//      at that mode are the null vector of M(λ) (found by Gaussian
//      elimination in complex arithmetic — n is small, typically 2–4).
//   4. The shared arbitrary constants C[k] are pinned down by evaluating
//      the symbolic general solution (and its derivatives) at the initial
//      conditions and calling the existing linear `Solve`.
//
// Repeated roots (defective systems needing a `t*E^(λt)` secular term) and
// non-constant forcing are out of scope and fall back to unevaluated,
// exactly like the single-variable solver's own unsupported cases.

/// One classified additive term of a coupled system's equation: which
/// dependent variable (by index into the system's `y_names`) and
/// derivative order it belongs to, or `None` for a term free of every
/// dependent variable (the forcing side).
struct MultiOdeTerm {
  var_index: Option<usize>,
  order: i32,
  coefficient: Expr,
}

/// Is `expr` free of every name in `y_names` (as `Identifier` uses and as
/// `Derivative[…, name, …]` heads)? Delegates to the single-variable
/// [`is_free_of_y`] once per name.
fn is_free_of_any_y(expr: &Expr, y_names: &[String]) -> bool {
  y_names.iter().all(|yn| is_free_of_y(expr, yn))
}

/// Like [`extract_derivative_order`], but against a whole set of dependent
/// variable names. Returns `(index into y_names, order)`.
fn extract_multivar_derivative(
  expr: &Expr,
  y_names: &[String],
) -> Option<(usize, usize)> {
  for (i, yn) in y_names.iter().enumerate() {
    if let Some(order) = extract_derivative_order(expr, yn) {
      return Some((i, order));
    }
  }
  None
}

/// Multi-variable counterpart of [`collect_ode_terms`]: collect the
/// additive terms of a normalized (lhs − rhs) equation, classifying each
/// by which dependent variable (if any) it involves.
fn collect_multivar_terms(
  expr: &Expr,
  y_names: &[String],
) -> Result<Vec<MultiOdeTerm>, InterpreterError> {
  let mut terms = Vec::new();
  collect_multivar_additive(expr, y_names, false, &mut terms)?;
  Ok(terms)
}

fn collect_multivar_additive(
  expr: &Expr,
  y_names: &[String],
  negated: bool,
  terms: &mut Vec<MultiOdeTerm>,
) -> Result<(), InterpreterError> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_multivar_additive(left, y_names, negated, terms)?;
      collect_multivar_additive(right, y_names, negated, terms)?;
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      collect_multivar_additive(left, y_names, negated, terms)?;
      collect_multivar_additive(right, y_names, !negated, terms)?;
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      collect_multivar_additive(operand, y_names, !negated, terms)?;
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      for arg in args {
        collect_multivar_additive(arg, y_names, negated, terms)?;
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      classify_multivar_product(args, y_names, negated, terms)?;
    }
    _ => {
      let term = classify_multivar_single(expr, y_names)?;
      let coeff = if negated {
        negate_expr(&term.coefficient)
      } else {
        term.coefficient
      };
      terms.push(MultiOdeTerm {
        var_index: term.var_index,
        order: term.order,
        coefficient: coeff,
      });
    }
  }
  Ok(())
}

fn classify_multivar_product(
  factors: &[Expr],
  y_names: &[String],
  negated: bool,
  terms: &mut Vec<MultiOdeTerm>,
) -> Result<(), InterpreterError> {
  let mut y_factor_idx = None;
  let mut y_var = 0usize;
  let mut y_order = -1i32;

  for (i, factor) in factors.iter().enumerate() {
    if let Some((vi, order)) = extract_multivar_derivative(factor, y_names) {
      y_factor_idx = Some(i);
      y_var = vi;
      y_order = order as i32;
      break;
    }
    if let Expr::FunctionCall { name, args } = factor
      && args.len() == 1
      && let Some(vi) = y_names.iter().position(|n| n == name)
    {
      y_factor_idx = Some(i);
      y_var = vi;
      y_order = 0;
      break;
    }
  }

  let (order, var_index, coefficient) = if let Some(idx) = y_factor_idx {
    let other_factors: Vec<&Expr> = factors
      .iter()
      .enumerate()
      .filter(|(i, _)| *i != idx)
      .map(|(_, f)| f)
      .collect();
    let coeff = if other_factors.is_empty() {
      Expr::Integer(1)
    } else if other_factors.len() == 1 {
      other_factors[0].clone()
    } else {
      times(other_factors.into_iter().cloned().collect())
    };
    (y_order, Some(y_var), coeff)
  } else {
    let product = if factors.len() == 1 {
      factors[0].clone()
    } else {
      unevaluated("Times", factors)
    };
    if !is_free_of_any_y(&product, y_names) {
      return Err(InterpreterError::EvaluationError(
        "DSolve: cannot classify term in coupled system".into(),
      ));
    }
    (-1, None, product)
  };

  let coeff = if negated {
    negate_expr(&coefficient)
  } else {
    coefficient
  };
  terms.push(MultiOdeTerm {
    var_index,
    order,
    coefficient: coeff,
  });
  Ok(())
}

fn classify_multivar_single(
  expr: &Expr,
  y_names: &[String],
) -> Result<MultiOdeTerm, InterpreterError> {
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && let Some(i) = y_names.iter().position(|n| n == name)
  {
    return Ok(MultiOdeTerm {
      var_index: Some(i),
      order: 0,
      coefficient: Expr::Integer(1),
    });
  }
  if let Some((i, order)) = extract_multivar_derivative(expr, y_names) {
    return Ok(MultiOdeTerm {
      var_index: Some(i),
      order: order as i32,
      coefficient: Expr::Integer(1),
    });
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    if let Some((i, order)) = extract_multivar_derivative(right, y_names) {
      return Ok(MultiOdeTerm {
        var_index: Some(i),
        order: order as i32,
        coefficient: *left.clone(),
      });
    }
    if let Expr::FunctionCall { name, args } = right.as_ref()
      && args.len() == 1
      && let Some(i) = y_names.iter().position(|n| n == name)
    {
      return Ok(MultiOdeTerm {
        var_index: Some(i),
        order: 0,
        coefficient: *left.clone(),
      });
    }
    if let Some((i, order)) = extract_multivar_derivative(left, y_names) {
      return Ok(MultiOdeTerm {
        var_index: Some(i),
        order: order as i32,
        coefficient: *right.clone(),
      });
    }
    if let Expr::FunctionCall { name, args } = left.as_ref()
      && args.len() == 1
      && let Some(i) = y_names.iter().position(|n| n == name)
    {
      return Ok(MultiOdeTerm {
        var_index: Some(i),
        order: 0,
        coefficient: *right.clone(),
      });
    }
  }
  if is_free_of_any_y(expr, y_names) {
    return Ok(MultiOdeTerm {
      var_index: None,
      order: -1,
      coefficient: expr.clone(),
    });
  }
  Err(InterpreterError::EvaluationError(
    "DSolve: cannot classify term in coupled system".into(),
  ))
}

// ─── Small numeric linear algebra for the companion system ─────────────

/// A complex number as a plain `(re, im)` pair — the system's dimension is
/// always tiny (a handful of coupled equations), so a dependency on a
/// complex-number crate isn't worth it for this one local computation.
type Cplx = (f64, f64);

fn cadd(a: Cplx, b: Cplx) -> Cplx {
  (a.0 + b.0, a.1 + b.1)
}
fn csub(a: Cplx, b: Cplx) -> Cplx {
  (a.0 - b.0, a.1 - b.1)
}
fn cmul(a: Cplx, b: Cplx) -> Cplx {
  (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cdiv(a: Cplx, b: Cplx) -> Cplx {
  let d = b.0 * b.0 + b.1 * b.1;
  ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn cabs(a: Cplx) -> f64 {
  a.0.hypot(a.1)
}

/// A polynomial in the differential operator `D`, `coeffs[k]` is the
/// coefficient of `D^k` (constant, since the ODE system has constant
/// coefficients).
type OpPoly = Vec<f64>;

fn poly_add(a: &[f64], b: &[f64]) -> OpPoly {
  let n = a.len().max(b.len());
  let mut out = vec![0.0; n];
  for (i, v) in a.iter().enumerate() {
    out[i] += v;
  }
  for (i, v) in b.iter().enumerate() {
    out[i] += v;
  }
  out
}

fn poly_scale(a: &[f64], s: f64) -> OpPoly {
  a.iter().map(|v| v * s).collect()
}

fn poly_mul(a: &[f64], b: &[f64]) -> OpPoly {
  if a.is_empty() || b.is_empty() {
    return Vec::new();
  }
  let mut out = vec![0.0; a.len() + b.len() - 1];
  for (i, &av) in a.iter().enumerate() {
    if av == 0.0 {
      continue;
    }
    for (j, &bv) in b.iter().enumerate() {
      out[i + j] += av * bv;
    }
  }
  out
}

/// Determinant of an `n×n` matrix of operator polynomials, via cofactor
/// expansion along the first row. `n` is always small (the number of
/// coupled dependent variables), so the `O(n!)` recursion is fine.
fn poly_det(m: &[Vec<OpPoly>]) -> OpPoly {
  let n = m.len();
  if n == 1 {
    return m[0][0].clone();
  }
  let mut result: OpPoly = vec![0.0];
  let mut sign = 1.0;
  for col in 0..n {
    let minor: Vec<Vec<OpPoly>> = (1..n)
      .map(|r| {
        (0..n)
          .filter(|&c| c != col)
          .map(|c| m[r][c].clone())
          .collect()
      })
      .collect();
    let cofactor = poly_det(&minor);
    let term = poly_scale(&poly_mul(&m[0][col], &cofactor), sign);
    result = poly_add(&result, &term);
    sign = -sign;
  }
  result
}

/// Drop numerically-negligible trailing high-order coefficients so the
/// polynomial's true degree can be read off from its length.
fn poly_trim(p: OpPoly) -> OpPoly {
  let mut p = p;
  let scale = p.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
  while p.len() > 1 && p.last().is_some_and(|v| v.abs() < scale * 1e-9) {
    p.pop();
  }
  if p.is_empty() {
    p.push(0.0);
  }
  p
}

fn eval_poly_c(p: &[f64], x: Cplx) -> Cplx {
  let mut acc: Cplx = (0.0, 0.0);
  let mut xp: Cplx = (1.0, 0.0);
  for &c in p {
    acc = cadd(acc, (c * xp.0, c * xp.1));
    xp = cmul(xp, x);
  }
  acc
}

/// A basis of the null space of a (numerically) rank-deficient complex
/// matrix with `cols` columns, via Gauss-Jordan elimination with partial
/// pivoting to reduced row-echelon form. One basis vector per free column,
/// each setting its own free column to 1 and the other free columns to 0,
/// so the basis follows the column order of the unknowns.
fn null_space(mut mat: Vec<Vec<Cplx>>, cols: usize) -> Vec<Vec<Cplx>> {
  let rows = mat.len();
  let scale = mat
    .iter()
    .flat_map(|r| r.iter())
    .fold(1.0_f64, |acc, c| acc.max(cabs(*c)));
  let tol = 1e-7 * scale;
  let mut pivot_cols: Vec<usize> = Vec::new();
  let mut row = 0usize;

  for col in 0..cols {
    if row >= rows {
      break;
    }
    let mut best_row = row;
    let mut best_val = cabs(mat[row][col]);
    for r in (row + 1)..rows {
      let v = cabs(mat[r][col]);
      if v > best_val {
        best_val = v;
        best_row = r;
      }
    }
    if best_val < tol {
      continue;
    }
    mat.swap(row, best_row);
    let piv = mat[row][col];
    for cell in mat[row].iter_mut().skip(col) {
      *cell = cdiv(*cell, piv);
    }
    for r in 0..rows {
      if r == row {
        continue;
      }
      let factor = mat[r][col];
      if cabs(factor) > 0.0 {
        for c in col..cols {
          let sub = cmul(factor, mat[row][c]);
          mat[r][c] = csub(mat[r][c], sub);
        }
      }
    }
    pivot_cols.push(col);
    row += 1;
  }

  let free_cols: Vec<usize> =
    (0..cols).filter(|c| !pivot_cols.contains(c)).collect();
  free_cols
    .iter()
    .map(|&free_col| {
      let mut v = vec![(0.0, 0.0); cols];
      v[free_col] = (1.0, 0.0);
      for (i, &pc) in pivot_cols.iter().enumerate() {
        v[pc] = (-mat[i][free_col].0, -mat[i][free_col].1);
      }
      v
    })
    .collect()
}

/// One term of a basis solution of a homogeneous linear system:
/// `coeff · x^power · E^(rate·x) · Cos[freq·x]`, or the same with `Sin`
/// when `sine` is set. A zero `freq` always uses the cosine form, whose
/// trigonometric factor is 1.
#[derive(Clone, Copy, Debug)]
struct SolTerm {
  rate: f64,
  freq: f64,
  power: usize,
  sine: bool,
  coeff: f64,
}

/// `P^(order)(λ)` — an operator polynomial's `order`-th derivative
/// evaluated at a (possibly complex) root.
fn poly_deriv_at(p: &OpPoly, order: usize, lambda: Cplx) -> Cplx {
  if order >= p.len() {
    return (0.0, 0.0);
  }
  let derived: OpPoly = p
    .iter()
    .enumerate()
    .skip(order)
    .map(|(k, c)| c * (0..order).map(|s| (k - s) as f64).product::<f64>())
    .collect();
  eval_poly_c(&derived, lambda)
}

/// Add a term to a solution, merging it into an existing term with the same
/// shape rather than repeating that shape.
fn add_sol_term(terms: &mut Vec<SolTerm>, term: SolTerm) {
  if let Some(existing) = terms.iter_mut().find(|t| {
    t.power == term.power
      && t.sine == term.sine
      && (t.rate - term.rate).abs() < 1e-9
      && (t.freq - term.freq).abs() < 1e-9
  }) {
    existing.coeff += term.coeff;
  } else {
    terms.push(term);
  }
}

/// The solutions of `M(D)·Y = 0` belonging to one root λ of multiplicity
/// `m` of `Det M(D)`. They have the form `Y_i = Σ_j a_ij·x^j·E^(λx)` with
/// `j < m`, and since `P(D)[x^j·E^(λx)] = E^(λx)·P(D+λ)[x^j]`, row `r` of
/// the system becomes one linear constraint per power of `x`:
///
/// `Σ_ij a_ij · P_ri^(j−p)(λ)/(j−p)! · j!/p! = 0` for every `p < m`.
///
/// The null space of that `n·m × n·m` matrix is `m`-dimensional — that is
/// what multiplicity `m` means — and each of its vectors is one solution,
/// or two real ones (the real and imaginary parts) when λ is complex.
fn root_solutions(
  poly_matrix: &[Vec<OpPoly>],
  lambda: Cplx,
  m: usize,
  n: usize,
) -> Option<Vec<Vec<Vec<SolTerm>>>> {
  let cols = n * m;
  let mut rows: Vec<Vec<Cplx>> = Vec::with_capacity(cols);
  for r in 0..n {
    for p in 0..m {
      let mut row = vec![(0.0, 0.0); cols];
      for i in 0..n {
        for j in p..m {
          let s = j - p;
          // j!/p! / s!, all of them tiny factorials (m ≤ 4).
          let factor = ((p + 1)..=j).map(|v| v as f64).product::<f64>()
            / (1..=s).map(|v| v as f64).product::<f64>();
          let value = poly_deriv_at(&poly_matrix[r][i], s, lambda);
          row[i * m + j] = (value.0 * factor, value.1 * factor);
        }
      }
      rows.push(row);
    }
  }

  let space = null_space(rows, cols);
  if space.len() != m {
    return None;
  }

  let mut solutions: Vec<Vec<Vec<SolTerm>>> = Vec::new();
  for v in &space {
    if lambda.1.abs() < 1e-9 {
      let mut solution: Vec<Vec<SolTerm>> = vec![Vec::new(); n];
      for (i, terms) in solution.iter_mut().enumerate() {
        for j in 0..m {
          let coeff = v[i * m + j].0;
          if coeff.abs() > 1e-12 {
            add_sol_term(
              terms,
              SolTerm {
                rate: lambda.0,
                freq: 0.0,
                power: j,
                sine: false,
                coeff,
              },
            );
          }
        }
      }
      solutions.push(solution);
    } else {
      // `a·x^j·E^((α+iβ)x)` contributes the real solution
      // `x^j·E^(αx)·(Re a·Cos[βx] − Im a·Sin[βx])` and the imaginary one
      // `x^j·E^(αx)·(Im a·Cos[βx] + Re a·Sin[βx])`.
      let mut real_part: Vec<Vec<SolTerm>> = vec![Vec::new(); n];
      let mut imag_part: Vec<Vec<SolTerm>> = vec![Vec::new(); n];
      for i in 0..n {
        for j in 0..m {
          let (p, q) = v[i * m + j];
          for (terms, cos_coeff, sin_coeff) in
            [(&mut real_part[i], p, -q), (&mut imag_part[i], q, p)]
          {
            for (sine, coeff) in [(false, cos_coeff), (true, sin_coeff)] {
              if coeff.abs() > 1e-12 {
                add_sol_term(
                  terms,
                  SolTerm {
                    rate: lambda.0,
                    freq: lambda.1,
                    power: j,
                    sine,
                    coeff,
                  },
                );
              }
            }
          }
        }
      }
      solutions.push(real_part);
      solutions.push(imag_part);
    }
  }
  Some(solutions)
}

/// A fundamental basis of the homogeneous system's solution space: the
/// solutions of every characteristic root, in root order.
fn solution_basis(
  poly_matrix: &[Vec<OpPoly>],
  roots: &[(f64, f64, usize)],
  n: usize,
) -> Option<Vec<Vec<Vec<SolTerm>>>> {
  let mut basis: Vec<Vec<Vec<SolTerm>>> = Vec::new();
  for (re, im, mult) in roots {
    basis.extend(root_solutions(poly_matrix, (*re, *im), *mult, n)?);
  }
  Some(basis)
}

/// Re-parametrize a basis of a *first-order* system so that its `k`-th
/// solution is the one starting at the `k`-th unit vector — i.e. so the
/// fundamental matrix is the identity at the origin and the arbitrary
/// constants are the initial values, `Y(x) = MatrixExp[A·x]·Y(0)`. That is
/// the basis wolframscript reports, and unlike the eigenvector basis it is
/// unique.
fn normalize_basis_at_origin(
  basis: &[Vec<Vec<SolTerm>>],
  n: usize,
) -> Option<Vec<Vec<Vec<SolTerm>>>> {
  // Φ(0): only a constant term (no `x` factor, no sine) survives at x = 0.
  let phi0: Vec<Vec<f64>> = (0..n)
    .map(|i| {
      (0..n)
        .map(|j| {
          basis[j][i]
            .iter()
            .filter(|t| t.power == 0 && !t.sine)
            .map(|t| t.coeff)
            .sum()
        })
        .collect()
    })
    .collect();

  // Column k of Φ(0)⁻¹ solves Φ(0)·b = e_k.
  let mut inverse: Vec<Vec<f64>> = Vec::with_capacity(n);
  for k in 0..n {
    let mut unit = vec![0.0; n];
    unit[k] = 1.0;
    let column = solve_real_linear_system(&phi0, &unit)?;
    if column.iter().any(|v| !v.is_finite()) {
      return None;
    }
    inverse.push(column);
  }
  // Guard against a near-singular Φ(0) that Gaussian elimination solved
  // with nonsense instead of failing outright.
  for i in 0..n {
    for k in 0..n {
      let entry: f64 = (0..n).map(|j| phi0[i][j] * inverse[k][j]).sum();
      let expected = if i == k { 1.0 } else { 0.0 };
      if (entry - expected).abs() > 1e-6 {
        return None;
      }
    }
  }

  let mut normalized: Vec<Vec<Vec<SolTerm>>> = Vec::with_capacity(n);
  for k in 0..n {
    let mut solution: Vec<Vec<SolTerm>> = vec![Vec::new(); n];
    for j in 0..n {
      let weight = inverse[k][j];
      if weight.abs() <= 1e-12 {
        continue;
      }
      for (i, terms) in solution.iter_mut().enumerate() {
        for term in &basis[j][i] {
          add_sol_term(
            terms,
            SolTerm {
              coeff: term.coeff * weight,
              ..*term
            },
          );
        }
      }
    }
    for terms in &mut solution {
      terms.retain(|t| t.coeff.abs() > 1e-12);
    }
    normalized.push(solution);
  }
  Some(normalized)
}

/// One basis solution's terms for a single variable, as an expression.
fn sol_terms_expr(terms: &[SolTerm], x_name: &str) -> Expr {
  let mut sum: Option<Expr> = None;
  for term in terms {
    let mut factors: Vec<Expr> = Vec::new();
    let coeff = f64_to_nice_expr(term.coeff);
    if !matches!(&coeff, Expr::Integer(1)) {
      factors.push(coeff);
    }
    if term.power > 0 {
      let x = id_expr(x_name);
      factors.push(if term.power == 1 {
        x
      } else {
        pow2(x, Expr::Integer(term.power as i128))
      });
    }
    if term.rate.abs() > 1e-10 {
      factors.push(make_exp_term(term.rate, x_name));
    }
    if term.freq.abs() > 1e-10 {
      factors.push(make_trig_term(
        if term.sine { "Sin" } else { "Cos" },
        term.freq,
        x_name,
      ));
    }
    let product = factors
      .into_iter()
      .reduce(times2)
      .unwrap_or(Expr::Integer(1));
    sum = Some(match sum {
      Some(acc) => plus2(acc, product),
      None => product,
    });
  }
  sum.unwrap_or(Expr::Integer(0))
}

/// Solve a real `n×n` linear system by Gaussian elimination with partial
/// pivoting. Used only for the constant particular solution of a system
/// with nonzero (constant) forcing.
fn solve_real_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
  let n = a.len();
  let mut aug: Vec<Vec<f64>> = a
    .iter()
    .zip(b)
    .map(|(row, &bi)| {
      let mut r = row.clone();
      r.push(bi);
      r
    })
    .collect();

  for col in 0..n {
    let mut best_row = col;
    let mut best_val = aug[col][col].abs();
    for r in (col + 1)..n {
      if aug[r][col].abs() > best_val {
        best_val = aug[r][col].abs();
        best_row = r;
      }
    }
    if best_val < 1e-12 {
      return None;
    }
    aug.swap(col, best_row);
    let pivot = aug[col][col];
    for r in (col + 1)..n {
      let factor = aug[r][col] / pivot;
      for c in col..=n {
        aug[r][c] -= factor * aug[col][c];
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

/// A polynomial particular solution of `M(D)·Y = f` for a constant forcing
/// `f`: each `Y_i` a polynomial of degree at most `zero_multiplicity` (the
/// multiplicity of the characteristic root 0; degree 0 when `M(0)` is
/// invertible). Matching the coefficients of every power of `x` gives a
/// linear system in the polynomial coefficients; free unknowns are set to
/// zero, since the homogeneous solution absorbs them.
fn polynomial_particular_solution(
  poly_matrix: &[Vec<OpPoly>],
  f: &[f64],
  zero_multiplicity: usize,
) -> Option<Vec<Vec<f64>>> {
  let n = f.len();
  let degree = zero_multiplicity;
  let unknowns = n * (degree + 1);
  // Row (r, p): the coefficient of x^p in row r of M(D)·Y.
  let mut rows: Vec<Vec<f64>> = Vec::new();
  for (r, f_r) in f.iter().enumerate() {
    for p in 0..=degree {
      let mut row = vec![0.0; unknowns + 1];
      for i in 0..n {
        for (s, &c) in poly_matrix[r][i].iter().enumerate() {
          // D^s x^j = j!/(j-s)! x^(j-s), which lands on x^p for j = p + s.
          let j = p + s;
          if c == 0.0 || j > degree {
            continue;
          }
          let falling: f64 = ((p + 1)..=j).map(|v| v as f64).product();
          row[i * (degree + 1) + j] += c * falling;
        }
      }
      row[unknowns] = if p == 0 { *f_r } else { 0.0 };
      rows.push(row);
    }
  }
  // Gauss-Jordan elimination with partial pivoting; free unknowns stay 0.
  let scale = rows
    .iter()
    .flat_map(|r| r.iter())
    .fold(0.0_f64, |m, v| m.max(v.abs()))
    .max(1.0);
  let mut pivots: Vec<(usize, usize)> = Vec::new();
  let mut next_row = 0;
  for col in 0..unknowns {
    let Some(best) = (next_row..rows.len())
      .max_by(|&a, &b| rows[a][col].abs().total_cmp(&rows[b][col].abs()))
    else {
      break;
    };
    if rows[best][col].abs() <= 1e-10 * scale {
      continue;
    }
    rows.swap(next_row, best);
    let pivot = rows[next_row][col];
    for v in &mut rows[next_row] {
      *v /= pivot;
    }
    for r in 0..rows.len() {
      if r != next_row && rows[r][col] != 0.0 {
        let factor = rows[r][col];
        for c in 0..=unknowns {
          rows[r][c] -= factor * rows[next_row][c];
        }
      }
    }
    pivots.push((next_row, col));
    next_row += 1;
  }
  // An inconsistent remaining row means no polynomial solution exists.
  if rows[next_row..]
    .iter()
    .any(|r| r[unknowns].abs() > 1e-9 * scale)
  {
    return None;
  }
  let mut solution = vec![vec![0.0; degree + 1]; n];
  for (r, col) in pivots {
    solution[col / (degree + 1)][col % (degree + 1)] = rows[r][unknowns];
  }
  Some(solution)
}

/// Placeholder identifier standing in for the shared arbitrary constant
/// `C[k]` while the general solution is still being solved against the
/// initial conditions (mirrors [`apply_initial_conditions`]'s own `__Ci`
/// placeholders, generalized to the whole system rather than one `y`).
fn sys_const(k: usize) -> Expr {
  Expr::Identifier(format!("$DSolveSystemC{k}"))
}

/// `DSolve[{eq1, …, ic1, …}, {y1[t], y2[t], …}, t]` for a linear,
/// constant-coefficient system — see the module-level comment above this
/// section for the method. Returns an `Err` with a `"DSolve:"`-prefixed
/// message for any shape outside this scope, which
/// [`dsolve_ast_with_head`] turns into an unevaluated `DSolve[…]` rather
/// than leaking the internal error.
fn dsolve_linear_system(
  eqns_arg: &Expr,
  dep_items: &[Expr],
  x_name: &str,
) -> Result<Expr, InterpreterError> {
  let bail = || {
    InterpreterError::EvaluationError(
      "DSolve: coupled system not supported".to_string(),
    )
  };

  let mut y_names: Vec<String> = Vec::with_capacity(dep_items.len());
  for item in dep_items {
    match item {
      Expr::FunctionCall { name, args }
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(s) if s == x_name) =>
      {
        y_names.push(name.clone());
      }
      _ => return Err(bail()),
    }
  }
  let n = y_names.len();
  if n < 2 {
    return Err(bail());
  }

  let items: Vec<Expr> = match eqns_arg {
    Expr::List(items) => items.iter().cloned().collect(),
    other => vec![other.clone()],
  };

  let mut odes: Vec<Expr> = Vec::new();
  let mut ics: Vec<(usize, Expr)> = Vec::new();
  for item in &items {
    let matched_var = y_names
      .iter()
      .position(|yn| is_initial_condition(item, yn, x_name));
    match matched_var {
      Some(vi) => ics.push((vi, item.clone())),
      None => odes.push(item.clone()),
    }
  }
  if odes.len() != n {
    return Err(bail());
  }

  // Build the n×n matrix of operator polynomials and the constant forcing
  // vector from each normalized (lhs − rhs) equation.
  let mut poly_matrix: Vec<Vec<OpPoly>> = vec![vec![Vec::new(); n]; n];
  let mut forcing_sum: Vec<f64> = vec![0.0; n];

  for (row, ode) in odes.iter().enumerate() {
    // Expanded, so a coefficient written as a product of a sum
    // (`(4 (x[t] - y[t]) + 4 x[t])/2`) splits into its terms
    let normalized = crate::evaluator::evaluate_function_call_ast(
      "Expand",
      &[normalize_equation(ode)?],
    )?;
    let terms = collect_multivar_terms(&normalized, &y_names)?;
    for term in terms {
      match term.var_index {
        Some(col) => {
          let coeff = eval_to_f64(&term.coefficient)?;
          let order = usize::try_from(term.order).map_err(|_| bail())?;
          let cell = &mut poly_matrix[row][col];
          if cell.len() <= order {
            cell.resize(order + 1, 0.0);
          }
          cell[order] += coeff;
        }
        None => {
          forcing_sum[row] += eval_to_f64(&term.coefficient)?;
        }
      }
    }
  }
  let f_vec: Vec<f64> = forcing_sum.iter().map(|v| -v).collect();

  let delta = poly_trim(poly_det(&poly_matrix));
  let degree = delta.len() - 1;
  if degree == 0 || delta.iter().all(|c| c.abs() < 1e-9) {
    return Err(bail());
  }
  // The root 0 is factored out first (`s^4 (9 s^2 + 100)` for a system
  // with a decoupled `z'' = c`), leaving at most a quartic to solve.
  let zeros = delta.iter().take_while(|c| c.abs() < 1e-12).count();
  let reduced = &delta[zeros..];
  if reduced.len() - 1 > 4 {
    return Err(bail());
  }
  let mut found: Vec<(f64, f64, usize)> = Vec::new();
  if zeros > 0 {
    found.push((0.0, 0.0, zeros));
  }
  if reduced.len() > 1 {
    found.extend(find_characteristic_roots(reduced, reduced.len() - 1)?);
  }

  // The root finders can list a repeated root once per occurrence
  // (`s^4 + s^2` gives 0 twice); the basis needs it once with its
  // multiplicity.
  let mut roots: Vec<(f64, f64, usize)> = Vec::new();
  for (re, im, mult) in found {
    match roots.iter_mut().find(|(r, i, _)| {
      (r - re).abs() <= 1e-7 * re.abs().max(1.0)
        && (i - im).abs() <= 1e-7 * im.abs().max(1.0)
    }) {
      Some(existing) => existing.2 += mult,
      None => roots.push((if re == 0.0 { 0.0 } else { re }, im, mult)),
    }
  }

  // The multiplicity of the root 0: a constant forcing needs a particular
  // solution polynomial of that degree when M(0) is singular.
  let zero_multiplicity = roots
    .iter()
    .find(|(re, im, _)| re.abs() <= 1e-9 && im.abs() <= 1e-9)
    .map_or(0, |r| r.2);

  // Constant particular solution for nonzero constant forcing: M(0) Yp = f.
  // With M(0) singular it is a polynomial in x instead: `particular[i][j]`
  // is the coefficient of x^j in Yp_i.
  let particular: Vec<Vec<f64>> = if f_vec.iter().any(|v| v.abs() > 1e-12) {
    polynomial_particular_solution(&poly_matrix, &f_vec, zero_multiplicity)
      .ok_or_else(bail)?
  } else {
    vec![vec![0.0]; n]
  };

  // A fundamental basis of the homogeneous solution space: one entry per
  // degree of freedom, each listing every variable's terms.
  let mut basis = solution_basis(&poly_matrix, &roots, n).ok_or_else(bail)?;
  if basis.len() != degree {
    return Err(bail());
  }
  let total_dof = basis.len();

  // For a first-order homogeneous system, wolframscript's constants are the
  // initial values themselves — its solution is `Y(x) = MatrixExp[A·x]·C`,
  // i.e. the fundamental matrix normalized to the identity at the origin.
  // Re-parametrize to that basis, which is unique (unlike the eigenvector
  // basis above, whose scaling and ordering are arbitrary).
  let is_first_order_homogeneous =
    total_dof == n && f_vec.iter().all(|v| v.abs() <= 1e-12);
  if is_first_order_homogeneous {
    basis = normalize_basis_at_origin(&basis, n).ok_or_else(bail)?;
  }

  let mut y_exprs: Vec<Expr> = (0..n)
    .map(|i| {
      let terms: Vec<SolTerm> = particular[i]
        .iter()
        .enumerate()
        .filter(|(_, c)| c.abs() > 1e-12)
        .map(|(power, &coeff)| SolTerm {
          rate: 0.0,
          freq: 0.0,
          power,
          sine: false,
          coeff,
        })
        .collect();
      if terms.is_empty() {
        Expr::Integer(0)
      } else {
        sol_terms_expr(&terms, x_name)
      }
    })
    .collect();

  for (j, solution) in basis.iter().enumerate() {
    let c = sys_const(j + 1);
    for k in 0..n {
      let mut coefficient = sol_terms_expr(&solution[k], x_name);
      if matches!(&coefficient, Expr::Integer(0)) {
        continue;
      }
      coefficient = crate::evaluator::evaluate_expr_to_expr(&coefficient)
        .unwrap_or(coefficient);
      if is_first_order_homogeneous {
        // wolframscript presents each constant's coefficient over a common
        // denominator: `(1 + E^(2*x))/(2*E^x)`, not `1/(2*E^x) + E^x/2`.
        coefficient = crate::functions::polynomial_ast::together::together_ast(
          std::slice::from_ref(&coefficient),
        )?;
      }
      y_exprs[k] = plus2(y_exprs[k].clone(), times2(coefficient, c.clone()));
    }
  }

  // Apply initial conditions (if a full set was given) by evaluating each
  // y_i and its derivatives at the IC point and solving for the shared
  // constants — the same technique as the single-variable
  // `apply_initial_conditions`, generalized across variables.
  let mut equations: Vec<Expr> = Vec::new();
  for (vi, ic) in &ics {
    let Expr::Comparison {
      operands,
      operators,
    } = ic
    else {
      continue;
    };
    if operands.len() != 2
      || operators.len() != 1
      || operators[0] != ComparisonOp::Equal
    {
      continue;
    }
    let lhs = &operands[0];
    let rhs = &operands[1];
    let yn = &y_names[*vi];
    let point_order = if let Expr::FunctionCall { name, args } = lhs
      && name == yn
      && args.len() == 1
    {
      Some((0usize, args[0].clone()))
    } else {
      extract_derivative_order_and_point(lhs, yn)
    };
    let Some((order, point)) = point_order else {
      continue;
    };

    let mut deriv = y_exprs[*vi].clone();
    for _ in 0..order {
      deriv =
        crate::functions::calculus_ast::differentiate_expr(&deriv, x_name)?;
      deriv = crate::functions::calculus_ast::simplify(deriv);
      deriv = crate::evaluator::evaluate_expr_to_expr(&deriv).unwrap_or(deriv);
    }
    let substituted =
      crate::syntax::substitute_variable(&deriv, x_name, &point);
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&substituted)
      .unwrap_or(substituted);
    equations.push(Expr::Comparison {
      operands: vec![evaluated, rhs.clone()],
      operators: vec![ComparisonOp::Equal],
    });
  }

  if equations.len() == total_dof {
    let var_exprs: Vec<Expr> = (1..=total_dof).map(sys_const).collect();
    let solve_result = crate::functions::solve_ast(&[
      Expr::List(equations.into()),
      Expr::List(var_exprs.into()),
    ])?;
    if let Expr::List(outer) = &solve_result
      && let Some(Expr::List(rules)) = outer.first()
    {
      for y_expr in &mut y_exprs {
        for rule in rules {
          if let Expr::Rule {
            pattern,
            replacement,
          } = rule
            && let Expr::Identifier(var_name) = pattern.as_ref()
          {
            *y_expr =
              crate::syntax::substitute_variable(y_expr, var_name, replacement);
          }
        }
        // A fully determined component comes back over one denominator,
        // as wolframscript writes it: `(Cos[t] + Cos[Sqrt[3] t])/2`
        *y_expr = crate::evaluator::evaluate_function_call_ast(
          "Together",
          &[crate::evaluator::evaluate_expr_to_expr(y_expr)
            .unwrap_or_else(|_| y_expr.clone())],
        )
        .unwrap_or_else(|_| y_expr.clone());
      }
    }
  } else {
    // Not enough initial conditions: present the general solution with
    // Wolfram's `C[k]` constants instead of the internal placeholders.
    for y_expr in &mut y_exprs {
      for k in 1..=total_dof {
        *y_expr = crate::syntax::substitute_variable(
          y_expr,
          &format!("$DSolveSystemC{k}"),
          &make_c(k),
        );
      }
      *y_expr = crate::evaluator::evaluate_expr_to_expr(y_expr)
        .unwrap_or_else(|_| y_expr.clone());
    }
  }

  let rules: Vec<Expr> = y_names
    .iter()
    .zip(y_exprs)
    .map(|(yn, sol)| {
      rule_expr(
        Expr::FunctionCall {
          name: yn.clone(),
          args: vec![id_expr(x_name)].into(),
        },
        sol,
      )
    })
    .collect();

  Ok(Expr::List(vec![Expr::List(rules.into())].into()))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A hand-built 4x4 periodic tridiagonal system whose two corners
  /// (`alpha` at the bottom-left, `beta` at the top-right — see
  /// `solve_cyclic_tridiagonal`'s doc comment) are different values,
  /// instead of the coincidentally-zero, symmetric corners every PDE
  /// integration test exercises this solver through. A caller that swaps
  /// `alpha`/`beta` (as `try_solve_hyperbolic_pde`'s call site did — see
  /// PR #960's review) transposes the solved matrix's two corners; with
  /// `alpha == beta` that transposition is invisible, so it takes a
  /// genuinely asymmetric system like this one to catch it. Verified by
  /// reconstructing the dense matrix and checking `A x == d` directly,
  /// independent of `solve_cyclic_tridiagonal`'s own algorithm.
  #[test]
  fn cyclic_tridiagonal_solves_the_periodic_system_with_asymmetric_corners() {
    let a = [0.0, 1.0, 1.0, 1.0];
    let b = [4.0, 4.0, 4.0, 4.0];
    let c = [1.0, 1.0, 1.0, 0.0];
    let alpha = 2.0; // bottom-left corner: A[3][0]
    let beta = 3.0; // top-right corner: A[0][3]
    let d = [10.0, 11.0, 12.0, 13.0];

    let x = solve_cyclic_tridiagonal(&a, &b, &c, &d, alpha, beta);

    let n = 4;
    let dense = [
      [b[0], c[0], 0.0, beta],
      [a[1], b[1], c[1], 0.0],
      [0.0, a[2], b[2], c[2]],
      [alpha, 0.0, a[3], b[3]],
    ];
    for i in 0..n {
      let row_dot: f64 = (0..n).map(|j| dense[i][j] * x[j]).sum();
      assert!(
        (row_dot - d[i]).abs() < 1e-9,
        "row {i}: A x = {row_dot}, expected {}",
        d[i]
      );
    }
  }
}
