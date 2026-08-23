//! AST-native ODE solving functions (DSolve, NDSolve).
//!
//! DSolve solves ordinary differential equations symbolically.
//! NDSolve solves initial-value problems numerically using RK4.

#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::{make_sqrt, rat_reduce};

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
  let result = match dsolve_ast_inner(args) {
    Err(InterpreterError::EvaluationError(msg))
      if msg.starts_with("DSolve:") =>
    {
      Ok(unevaluated("DSolve", args))
    }
    other => other,
  };
  result.map(|expr| retag_unevaluated(expr, "DSolve", head))
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

fn dsolve_ast_inner(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("DSolve", args));
  }

  let eqns_arg = &args[0];
  let dep_arg = &args[1];
  let indep_var = &args[2];

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

  // Collect terms: classify each additive term by derivative order
  let terms = match collect_ode_terms(&ode_normalized, &y_name, &x_name) {
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
  if max_order == 0 {
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

/// Wrap a solved right-hand side as `{{y[x] -> sol}}`, or
/// `{{y -> Function[{x}, sol]}}` when the caller asked for `y` rather than
/// `y[x]`.
fn build_dsolve_result(
  solution: Expr,
  y_name: String,
  x_name: String,
  function_form: bool,
) -> Expr {
  let rule = if function_form {
    Expr::Rule {
      pattern: Box::new(Expr::Identifier(y_name)),
      replacement: Box::new(Expr::NamedFunction {
        params: vec![x_name],
        body: Box::new(solution),
        bracketed: true,
      }),
    }
  } else {
    Expr::Rule {
      pattern: Box::new(Expr::FunctionCall {
        name: y_name,
        args: vec![Expr::Identifier(x_name)].into(),
      }),
      replacement: Box::new(solution),
    }
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
  ndsolve_ast_inner(args).map(|expr| retag_unevaluated(expr, "NDSolve", head))
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
    return match ndsolve_pde(&args[..4]) {
      Ok(Some(result)) => Ok(result),
      Ok(None) => Ok(unevaluated("NDSolve", args)),
      Err(e) => Err(e),
    };
  }
  if n_pos != 3 {
    return Ok(unevaluated("NDSolve", args));
  }
  let opts = &args[n_pos..];
  let event = parse_event_locator_option(opts);

  // Unknowns written as compound expressions (`Subscript[c, 1]`) are keyed
  // by a fresh symbol while integrating, then restored in the result.
  let renames = compound_head_renamings(&args[1], &args[2]);
  let positional: Vec<Expr> = if renames.is_empty() {
    args[..3].to_vec()
  } else {
    args[..3]
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
  match ndsolve_system(&positional, event.as_ref()) {
    Ok(Some(result)) => Ok(restore_compound_heads(&result, &renames)),
    Ok(None) => Ok(unevaluated("NDSolve", args)),
    Err(e) => Err(e),
  }
}

// ─── PDE NDSolve (method of lines) ─────────────────────────────────────

/// A domain spec `{name, min, max}` for one PDE independent variable.
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

/// Recognise `Derivative[1, 0][u][t, x] == rhs` (or the reverse) — the
/// evolution equation of a parabolic PDE. Returns the RHS.
fn try_pde_evolution_rhs(
  eq: &Expr,
  u_name: &str,
  t_name: &str,
  x_name: &str,
) -> Option<Expr> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let is_dudt = |e: &Expr| {
    matches!(
      match_pde_term(e, u_name),
      Some((1, 0, t_arg, x_arg))
        if matches!(&t_arg, Expr::Identifier(n) if n == t_name)
          && matches!(&x_arg, Expr::Identifier(n) if n == x_name)
    )
  };
  if is_dudt(lhs) {
    return Some(rhs.clone());
  }
  if is_dudt(rhs) {
    return Some(lhs.clone());
  }
  None
}

/// Recognise `u[t0, x] == rhs(x)` (or reversed) — the initial condition at
/// `t == t0`. Returns the RHS.
fn try_pde_initial_condition(
  eq: &Expr,
  u_name: &str,
  x_name: &str,
  t0: f64,
) -> Option<Expr> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let is_ic = |e: &Expr| {
    let Expr::FunctionCall { name, args } = e else {
      return false;
    };
    name == u_name
      && args.len() == 2
      && matches!(&args[1], Expr::Identifier(n) if n == x_name)
      && nval_to_f64(&args[0])
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

/// Recognise `u[t, x0] == rhs(t)` (or reversed) — a Dirichlet boundary
/// condition at `x == x0`. Returns the RHS.
fn try_pde_boundary_condition(
  eq: &Expr,
  u_name: &str,
  t_name: &str,
  x0: f64,
) -> Option<Expr> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let is_bc = |e: &Expr| {
    let Expr::FunctionCall { name, args } = e else {
      return false;
    };
    name == u_name
      && args.len() == 2
      && matches!(&args[0], Expr::Identifier(n) if n == t_name)
      && nval_to_f64(&args[1])
        .is_some_and(|v| (v - x0).abs() <= 1e-9 * x0.abs().max(1.0))
  };
  if is_bc(lhs) {
    return Some(rhs.clone());
  }
  if is_bc(rhs) {
    return Some(lhs.clone());
  }
  None
}

/// Rewrite `u[t, x]`, `Derivative[0, 1][u][t, x]` and
/// `Derivative[0, 2][u][t, x]` in a PDE's evolution right-hand side into the
/// placeholder identifiers that `NumFn` compiles the finite-difference
/// stencil's numeric values into (see `ndsolve_pde`).
fn rewrite_pde_rhs(
  expr: &Expr,
  u_name: &str,
  t_name: &str,
  x_name: &str,
) -> Expr {
  if let Some((dt, dx, t_arg, x_arg)) = match_pde_term(expr, u_name)
    && matches!(&t_arg, Expr::Identifier(n) if n == t_name)
    && matches!(&x_arg, Expr::Identifier(n) if n == x_name)
  {
    let placeholder = match (dt, dx) {
      (0, 0) => Some("NDSolve$U"),
      (0, 1) => Some("NDSolve$UX"),
      (0, 2) => Some("NDSolve$UXX"),
      _ => None,
    };
    if let Some(name) = placeholder {
      return Expr::Identifier(name.to_string());
    }
  }
  map_children(expr, &|child| {
    rewrite_pde_rhs(child, u_name, t_name, x_name)
  })
}

/// `NDSolve[eqns, u, {t, t0, t1}, {x, x0, x1}]` — a scalar parabolic PDE in
/// one space dimension (a diffusion/heat-equation shape), solved by the
/// method of lines: `x` is discretized onto a uniform grid, the spatial
/// derivatives `D[u[t, x], x]` and `D[u[t, x], {x, 2}]` become central
/// finite-difference stencils, and the resulting ODE system in `t` is
/// integrated with classical RK4 subject to a diffusive stability bound on
/// the time step. `eqns` must contain exactly one evolution equation, one
/// initial condition at `t == t0`, and Dirichlet conditions at both space
/// boundaries.
///
/// Returns `Ok(None)` when the equations aren't in that shape, so the
/// caller falls back to leaving the call unevaluated.
fn ndsolve_pde(args: &[Expr]) -> Result<Option<Expr>, InterpreterError> {
  let u_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    Expr::List(items) if items.len() == 1 => match &items[0] {
      Expr::Identifier(name) => name.clone(),
      _ => return Ok(None),
    },
    _ => return Ok(None),
  };
  let Some(t_dom) = parse_pde_domain(&args[2]) else {
    return Ok(None);
  };
  let Some(x_dom) = parse_pde_domain(&args[3]) else {
    return Ok(None);
  };

  let eq_items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  if eq_items.len() != 4 {
    return Ok(None);
  }

  let mut evolution_rhs: Option<Expr> = None;
  let mut ic_rhs: Option<Expr> = None;
  let mut bc_lo_rhs: Option<Expr> = None;
  let mut bc_hi_rhs: Option<Expr> = None;
  for eq in &eq_items {
    if evolution_rhs.is_none()
      && let Some(rhs) =
        try_pde_evolution_rhs(eq, &u_name, &t_dom.name, &x_dom.name)
    {
      evolution_rhs = Some(rhs);
      continue;
    }
    if ic_rhs.is_none()
      && let Some(rhs) =
        try_pde_initial_condition(eq, &u_name, &x_dom.name, t_dom.min)
    {
      ic_rhs = Some(rhs);
      continue;
    }
    if bc_lo_rhs.is_none()
      && let Some(rhs) =
        try_pde_boundary_condition(eq, &u_name, &t_dom.name, x_dom.min)
    {
      bc_lo_rhs = Some(rhs);
      continue;
    }
    if bc_hi_rhs.is_none()
      && let Some(rhs) =
        try_pde_boundary_condition(eq, &u_name, &t_dom.name, x_dom.max)
    {
      bc_hi_rhs = Some(rhs);
      continue;
    }
    return Ok(None);
  }
  let (Some(evolution_rhs), Some(ic_rhs), Some(bc_lo_rhs), Some(bc_hi_rhs)) =
    (evolution_rhs, ic_rhs, bc_lo_rhs, bc_hi_rhs)
  else {
    return Ok(None);
  };

  let rhs_placeholder =
    rewrite_pde_rhs(&evolution_rhs, &u_name, &t_dom.name, &x_dom.name);
  let rhs_vars = [
    "NDSolve$U".to_string(),
    "NDSolve$UX".to_string(),
    "NDSolve$UXX".to_string(),
    t_dom.name.clone(),
    x_dom.name.clone(),
  ];
  let rhs_fn = NumFn::new(rhs_placeholder, &rhs_vars);
  let ic_fn = NumFn::new(ic_rhs, std::slice::from_ref(&x_dom.name));
  let bc_lo_fn = NumFn::new(bc_lo_rhs, std::slice::from_ref(&t_dom.name));
  let bc_hi_fn = NumFn::new(bc_hi_rhs, std::slice::from_ref(&t_dom.name));

  const N_X: usize = 41;
  let dx = (x_dom.max - x_dom.min) / (N_X - 1) as f64;
  let xs: Vec<f64> = (0..N_X).map(|i| x_dom.min + i as f64 * dx).collect();

  let mut u: Vec<f64> = Vec::with_capacity(N_X);
  for &x in &xs {
    u.push(ic_fn.eval(&[x])?);
  }
  u[0] = bc_lo_fn.eval(&[t_dom.min])?;
  u[N_X - 1] = bc_hi_fn.eval(&[t_dom.min])?;

  // Effective diffusivity, read off the RHS's dependence on `NDSolve$UXX`,
  // to size a stable explicit time step (a von Neumann/CFL-type bound).
  // Falls back to a small floor when the PDE has (almost) no such
  // dependence, since a step must still be chosen.
  let probe_lo = [u[N_X / 2], 0.0, 0.0, t_dom.min, xs[N_X / 2]];
  let mut probe_hi = probe_lo;
  probe_hi[2] = 1.0;
  let c_eff = (rhs_fn.eval(&probe_hi)? - rhs_fn.eval(&probe_lo)?).max(1e-9);
  let dt_stable = 0.4 * dx * dx / c_eff;
  // The diffusive bound alone can leave the boundary conditions' own time
  // variation (a seasonal `Sin[2 Pi t/period]` forcing, say) badly
  // undersampled even though the scheme stays numerically stable, so the
  // step count is also floored well above what a smooth time-dependent
  // boundary needs to be resolved.
  const MIN_STEPS: usize = 200;
  let n_steps = (((t_dom.max - t_dom.min) / dt_stable).ceil() as usize)
    .clamp(MIN_STEPS, 20_000);
  let dt = (t_dom.max - t_dom.min) / n_steps as f64;

  let derivative =
    |t: f64, interior: &[f64]| -> Result<Vec<f64>, InterpreterError> {
      let mut full = vec![0.0; N_X];
      full[0] = bc_lo_fn.eval(&[t])?;
      full[N_X - 1] = bc_hi_fn.eval(&[t])?;
      full[1..N_X - 1].copy_from_slice(interior);
      let mut out = Vec::with_capacity(N_X - 2);
      for i in 1..N_X - 1 {
        let ux = (full[i + 1] - full[i - 1]) / (2.0 * dx);
        let uxx = (full[i + 1] - 2.0 * full[i] + full[i - 1]) / (dx * dx);
        out.push(rhs_fn.eval(&[full[i], ux, uxx, t, xs[i]])?);
      }
      Ok(out)
    };

  // The fine step count above is sized for integration accuracy and
  // stability, not for how many rows an `InterpolatingFunction` needs to
  // represent a smooth solution — and every stored row is a separate
  // heap-allocated list that later gets cloned whole on every lookup (a
  // `ContourPlot` alone samples on the order of 10,000 points), so only
  // every `stride`-th step's row is kept, with the true endpoint always
  // included.
  const OUTPUT_ROWS: usize = 60;
  let stride = (n_steps / OUTPUT_ROWS).max(1);
  let mut grid: Vec<Vec<f64>> = Vec::with_capacity(n_steps / stride + 2);
  let mut ts: Vec<f64> = Vec::with_capacity(n_steps / stride + 2);
  grid.push(u.clone());
  ts.push(t_dom.min);
  let mut interior: Vec<f64> = u[1..N_X - 1].to_vec();
  let mut t = t_dom.min;
  for step in 0..n_steps {
    let k1 = derivative(t, &interior)?;
    let y2: Vec<f64> = interior
      .iter()
      .zip(&k1)
      .map(|(y, k)| y + 0.5 * dt * k)
      .collect();
    let k2 = derivative(t + 0.5 * dt, &y2)?;
    let y3: Vec<f64> = interior
      .iter()
      .zip(&k2)
      .map(|(y, k)| y + 0.5 * dt * k)
      .collect();
    let k3 = derivative(t + 0.5 * dt, &y3)?;
    let y4: Vec<f64> =
      interior.iter().zip(&k3).map(|(y, k)| y + dt * k).collect();
    let k4 = derivative(t + dt, &y4)?;
    for i in 0..interior.len() {
      interior[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    t += dt;
    let is_last = step + 1 == n_steps;
    if (step + 1) % stride == 0 || is_last {
      let mut row = vec![0.0; N_X];
      row[0] = bc_lo_fn.eval(&[t])?;
      row[N_X - 1] = bc_hi_fn.eval(&[t])?;
      row[1..N_X - 1].copy_from_slice(&interior);
      grid.push(row);
      ts.push(t);
    }
  }

  let domain = Expr::List(
    vec![
      Expr::List(vec![Expr::Real(t_dom.min), Expr::Real(t_dom.max)].into()),
      Expr::List(vec![Expr::Real(x_dom.min), Expr::Real(x_dom.max)].into()),
    ]
    .into(),
  );
  let grid_expr = Expr::List(
    grid
      .iter()
      .map(|row| Expr::List(row.iter().map(|v| Expr::Real(*v)).collect()))
      .collect(),
  );
  let orders = Expr::List(vec![Expr::Integer(1), Expr::Integer(1)].into());
  let coords = Expr::List(
    vec![
      Expr::List(ts.iter().map(|v| Expr::Real(*v)).collect()),
      Expr::List(xs.iter().map(|v| Expr::Real(*v)).collect()),
    ]
    .into(),
  );
  let interp = call(
    "InterpolatingFunction",
    vec![domain, grid_expr, orders, coords],
  );
  let rule = Expr::Rule {
    pattern: Box::new(Expr::Identifier(u_name)),
    replacement: Box::new(interp),
  };
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
    let key = crate::syntax::expr_to_string(&head);
    if renames
      .iter()
      .any(|(h, _)| crate::syntax::expr_to_string(h) == key)
    {
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
fn rename_compound_heads(expr: &Expr, renames: &[(Expr, String)]) -> Expr {
  let key = crate::syntax::expr_to_string(expr);
  for (head, fresh) in renames {
    if crate::syntax::expr_to_string(head) == key {
      return Expr::Identifier(fresh.clone());
    }
  }
  let mapped = map_children(expr, &|c| rename_compound_heads(c, renames));
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
fn restore_compound_heads(expr: &Expr, renames: &[(Expr, String)]) -> Expr {
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
      .is_some_and(|i| matches!(i, Expr::String(s) if s == "EventLocator"));
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
          Expr::String(s) if s == "Event" => {
            event = Some(replacement.as_ref().clone());
          }
          Expr::String(s) if s == "EventAction" => {
            action = Some(replacement.as_ref().clone());
          }
          _ => {}
        }
      }
    }
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
fn flatten_eq_list(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::List(items) => {
      for item in items {
        flatten_eq_list(item, out);
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

/// Find an algebraic constraint among `odes` that determines one of `funcs`
/// explicitly: an equation with no derivative of any unknown, in which
/// exactly one unknown appears and appears nowhere differentiated elsewhere.
/// Returns its index, that function's index, and the expression it solves to.
fn find_algebraic_constraint(
  odes: &[Expr],
  funcs: &[SysFunc],
) -> Option<(usize, usize, Expr)> {
  for (ei, eq) in odes.iter().enumerate() {
    if funcs.iter().any(|f| max_derivative_order(eq, &f.name) > 0) {
      continue;
    }
    // The unknown this constraint defines is one it mentions that no
    // equation ever differentiates — the others in it are dynamic
    // variables the solution is written in terms of.
    let Some(fi) = funcs.iter().enumerate().position(|(_, f)| {
      expr_mentions_function(eq, &f.name)
        && odes.iter().all(|o| max_derivative_order(o, &f.name) == 0)
    }) else {
      continue;
    };
    let Some(solved) = solve_for_function(eq, &funcs[fi].name) else {
      continue;
    };
    return Some((ei, fi, solved));
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

/// Solve an algebraic equation for `fname[x]`, returning the right-hand side
/// of the single solution. `None` when it does not solve uniquely.
fn solve_for_function(eq: &Expr, fname: &str) -> Option<Expr> {
  let unknown = find_function_call(eq, fname)?;
  let solved = crate::evaluator::evaluate_expr_to_expr(&call(
    "Solve",
    vec![eq.clone(), unknown],
  ))
  .ok()?;
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
      let Some(max) = nval_to_f64(&domain_items[2]) else {
        return Ok(None);
      };
      // NaN bounds must bail out too, so compare via partial_cmp rather
      // than a negated float comparison.
      if max.partial_cmp(&min) != Some(std::cmp::Ordering::Greater) {
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
  let mut odes: Vec<Expr> = Vec::new();
  let mut x0: Option<f64> = None;
  // (function name, derivative order, value) — by name, not index: an
  // eliminated constraint variable shifts the positions in `funcs`.
  let mut ics: Vec<(String, usize, f64)> = Vec::new();
  for eq in &eq_items {
    let mut is_ic = false;
    for f in &funcs {
      if is_initial_condition(eq, &f.name, x_name) {
        let Some((order, x_val, y_val)) =
          parse_numeric_initial_condition(eq, &f.name)
        else {
          return Ok(None);
        };
        match x0 {
          None => x0 = Some(x_val),
          // All ICs must be given at the same point.
          Some(prev) if (prev - x_val).abs() > 1e-12 => return Ok(None),
          Some(_) => {}
        }
        ics.push((f.name.clone(), order, y_val));
        is_ic = true;
        break;
      }
    }
    if !is_ic {
      odes.push(eq.clone());
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

  if odes.len() != funcs.len() {
    return Ok(None);
  }
  let Some(x0) = x0 else { return Ok(None) };
  let (x_min, x_max) = if let Some(min) = x_min_given {
    (min, x_max)
  } else {
    // {x, xmax} shorthand: integrate from x0 to xmax, in whichever
    // direction that is — x0 always lands on one edge of the range. An
    // absolute epsilon here would wrongly reject ranges that are tiny by
    // scale (e.g. femtosecond time constants) rather than degenerate, so
    // only bit-identical endpoints count as degenerate.
    let target = x_max;
    if target == x0 {
      return Ok(None);
    }
    (x0.min(target), x0.max(target))
  };
  if x0 < x_min - 1e-12 || x0 > x_max + 1e-12 {
    return Ok(None);
  }

  // Determine each function's order from the equations.
  for ode in &odes {
    for f in &mut funcs {
      let order = max_derivative_order(ode, &f.name);
      f.order = f.order.max(order);
    }
  }
  if funcs.iter().any(|f| f.order == 0) {
    return Ok(None);
  }
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

  // Validate and store the initial conditions.
  for f in &mut funcs {
    f.ics = vec![None; f.order];
  }
  for (name, order, val) in ics {
    // An initial condition for a function the constraint eliminated is
    // redundant — its value follows from the others — so it is dropped.
    let Some(f) = funcs.iter_mut().find(|f| f.name == name) else {
      continue;
    };
    if order >= f.order || f.ics[order].is_some() {
      return Ok(None);
    }
    f.ics[order] = Some(val);
  }
  if funcs
    .iter()
    .any(|f| f.ics.iter().any(std::option::Option::is_none))
  {
    return Ok(None);
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
  for ode in &odes {
    let normalized = normalize_equation(ode).ok();
    let Some(normalized) = normalized else {
      return Ok(None);
    };
    let rewritten = substitute_function_values(&normalized, &funcs, x_name);
    residuals.push(NumFn::new(rewritten, &var_names));
  }
  let event_fn = match event {
    Some(spec) => {
      let rewritten = substitute_function_values(&spec.event, &funcs, x_name);
      Some(NumFn::new(rewritten, &var_names))
    }
    None => None,
  };

  // Initial state.
  let mut init_state: Vec<f64> = vec![0.0; n_state];
  for (fi, f) in funcs.iter().enumerate() {
    for (k, ic) in f.ics.iter().enumerate() {
      init_state[state_offset[fi] + k] = ic.unwrap_or(0.0);
    }
  }

  let n_steps = 1000usize;
  let h = (x_max - x_min) / n_steps as f64;

  // Integrate forward from x0 to x_max, then (if x0 is interior)
  // backward from x0 to x_min; events are only located on the forward
  // leg, matching the direction NDSolve integrates first.
  let forward = integrate_leg(
    &residuals,
    &funcs,
    &state_offset,
    init_state.clone(),
    x0,
    x_max,
    h,
    event_fn.as_ref(),
    event.and_then(|e| e.action.as_ref()),
    x_name,
  )?;
  let Some(forward) = forward else {
    return Ok(None);
  };
  let backward = if x0 - x_min > 1e-12 {
    let leg = integrate_leg(
      &residuals,
      &funcs,
      &state_offset,
      init_state,
      x0,
      x_min,
      -h,
      None,
      None,
      x_name,
    )?;
    let Some(leg) = leg else {
      return Ok(None);
    };
    leg
  } else {
    Vec::new()
  };

  // Combined, ascending in x. The backward leg is (x0, x0-h, …); reverse
  // it and drop its first point (x0, present in the forward leg too).
  let mut points: Vec<(f64, Vec<f64>)> = backward;
  points.reverse();
  points.pop();
  points.extend(forward);
  if points.len() < 2 {
    return Ok(None);
  }
  let x_lo = points.first().unwrap().0;
  let x_hi = points.last().unwrap().0;

  // Build one InterpolatingFunction rule per dependent function.
  let mut rules: Vec<Expr> = Vec::with_capacity(funcs.len() + eliminated.len());
  for (fi, f) in funcs.iter().enumerate() {
    let domain = Expr::List(
      vec![Expr::List(vec![Expr::Real(x_lo), Expr::Real(x_hi)].into())].into(),
    );
    let data = Expr::List(
      points
        .iter()
        .map(|(x, s)| {
          Expr::List(
            vec![Expr::Real(*x), Expr::Real(s[state_offset[fi]])].into(),
          )
        })
        .collect(),
    );
    let interp = call(
      "InterpolatingFunction",
      vec![domain, data, Expr::Integer(NDSOLVE_INTERPOLATION_ORDER)],
    );
    rules.push(if f.function_form {
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(f.name.clone())),
        replacement: Box::new(interp),
      }
    } else {
      Expr::Rule {
        pattern: Box::new(Expr::FunctionCall {
          name: f.name.clone(),
          args: vec![Expr::Identifier(x_name.clone())].into(),
        }),
        replacement: Box::new(Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        }),
      }
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
      func: Box::new(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: vec![Expr::Integer(*order as i128)].into(),
      }),
      args: vec![Expr::Identifier(name.clone())],
    };
    rules.push(if *function_form {
      Expr::Rule {
        pattern: Box::new(deriv_pattern),
        replacement: Box::new(interp),
      }
    } else {
      Expr::Rule {
        pattern: Box::new(Expr::CurriedCall {
          func: Box::new(deriv_pattern),
          args: vec![Expr::Identifier(x_name.clone())],
        }),
        replacement: Box::new(Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        }),
      }
    });
  }

  // An eliminated function is sampled from its constraint solution at the
  // same grid, so it comes back as an InterpolatingFunction like the rest.
  for (f, solved) in &eliminated {
    let rewritten = substitute_function_values(solved, &funcs, x_name);
    let value_fn = NumFn::new(rewritten, &var_names);
    let mut vars = vec![0.0; var_names.len()];
    let data = Expr::List(
      points
        .iter()
        .map(|(x, state)| {
          vars[0] = *x;
          vars[1..=n_state].copy_from_slice(&state[..n_state]);
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
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(f.name.clone())),
        replacement: Box::new(interp),
      }
    } else {
      Expr::Rule {
        pattern: Box::new(Expr::FunctionCall {
          name: f.name.clone(),
          args: vec![Expr::Identifier(x_name.clone())].into(),
        }),
        replacement: Box::new(Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        }),
      }
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

/// Integrate one leg with RK4. Returns the points in integration order
/// (starting at `x_from`), or `None` if a derivative evaluation failed.
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
  x_name: &str,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  let n_steps = ((x_to - x_from) / h).round() as usize;
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

  let eval_event = |x: f64, state: &[f64]| -> Option<f64> {
    event_fn.and_then(|f| f.eval_state(x, state, funcs.len()).ok())
  };
  let mut prev_event = eval_event(x, &state);

  for i in 0..n_steps {
    let new_x = grid(i + 1);
    let h = new_x - x;
    let mut refined: Vec<(f64, Vec<f64>)> = Vec::new();
    let Some(new_state) = rk4_step_refined(
      residuals,
      funcs,
      state_offset,
      &state,
      x,
      h,
      0,
      &mut refined,
    )?
    else {
      return Ok(None);
    };

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
    points.append(&mut refined);
  }
  Ok(Some(points))
}

/// Local-error tolerances for the step-doubling refinement below. They are
/// deliberately loose: RK4's local error at the nominal step size is orders
/// of magnitude smaller for any smooth problem, so an ordinary integration
/// never subdivides and keeps exactly the grid — and the values — the
/// fixed-step integrator produced.
const STEP_REFINE_RTOL: f64 = 1e-4;
const STEP_REFINE_ATOL: f64 = 1e-8;

/// How far one nominal step may be bisected. A tracer pulse 10^-6 wide in a
/// domain of length 20 needs ~15 bisections of the 1/1000 nominal step to be
/// seen at all, and another dozen for its edges to stop costing accuracy.
const MAX_STEP_BISECTIONS: u32 = 32;

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
  depth: u32,
  out: &mut Vec<(f64, Vec<f64>)>,
) -> Result<Option<Vec<f64>>, InterpreterError> {
  let Some(full) =
    rk4_system_step(residuals, funcs, state_offset, state, x, h)?
  else {
    return Ok(None);
  };
  let accept = |s: Vec<f64>, out: &mut Vec<(f64, Vec<f64>)>| {
    out.push((x + h, s.clone()));
    Ok(Some(s))
  };
  if depth >= MAX_STEP_BISECTIONS || out.len() >= MAX_REFINED_POINTS {
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
    rk4_system_step(residuals, funcs, state_offset, state, x, half)?
  else {
    return accept(full, out);
  };
  let Some(two) =
    rk4_system_step(residuals, funcs, state_offset, &mid, x + half, half)?
  else {
    return accept(full, out);
  };

  // A non-finite difference means the problem is singular here, not that a
  // smaller step would help, so it counts as converged.
  let converged = full.iter().zip(&two).all(|(a, b)| {
    let diff = (a - b).abs();
    let scale = STEP_REFINE_ATOL + STEP_REFINE_RTOL * a.abs().max(b.abs());
    !diff.is_finite() || diff <= scale
  });
  if converged {
    return accept(full, out);
  }

  let Some(mid) = rk4_step_refined(
    residuals,
    funcs,
    state_offset,
    state,
    x,
    half,
    depth + 1,
    out,
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
    depth + 1,
    out,
  )
}

/// One RK4 step of the first-order system equivalent. Returns `None`
/// when a residual can't be evaluated numerically or the linear system
/// for the highest derivatives is singular.
fn rk4_system_step(
  residuals: &[NumFn],
  funcs: &[SysFunc],
  state_offset: &[usize],
  state: &[f64],
  x: f64,
  h: f64,
) -> Result<Option<Vec<f64>>, InterpreterError> {
  let deriv =
    |s: &[f64], xv: f64| -> Result<Option<Vec<f64>>, InterpreterError> {
      let Some(high) = solve_highest_derivatives(residuals, funcs.len(), xv, s)
      else {
        return Ok(None);
      };
      let mut d = vec![0.0; s.len()];
      for (fi, f) in funcs.iter().enumerate() {
        let off = state_offset[fi];
        for k in 0..f.order - 1 {
          d[off + k] = s[off + k + 1];
        }
        d[off + f.order - 1] = high[fi];
      }
      Ok(Some(d))
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
fn solve_highest_derivatives(
  residuals: &[NumFn],
  n_funcs: usize,
  x: f64,
  state: &[f64],
) -> std::option::Option<std::vec::Vec<f64>> {
  let n = n_funcs;
  let mut vars = Vec::with_capacity(1 + state.len() + n);
  vars.push(x);
  vars.extend_from_slice(state);
  vars.extend(std::iter::repeat_n(0.0, n));

  let mut c = vec![0.0; n];
  for (i, r) in residuals.iter().enumerate() {
    let Ok(v) = r.eval(&vars) else {
      return None;
    };
    if !v.is_finite() {
      return None;
    }
    c[i] = v;
  }
  // Each column is recovered from r(delta·e_j) − r(0) = M·(delta·e_j), a
  // finite difference that is *exact* since the residuals are affine in
  // the highest-derivative vector. A unit-sized `delta` is lost entirely to
  // rounding when the other terms in the residual dwarf it — e.g. a
  // coefficient of 1e16 (from squaring a large angular frequency) makes
  // `c[i] + 1.0` round straight back down to `c[i]` in `f64`, which reads
  // as an exactly-zero, falsely singular column. Scaling `delta` up to the
  // residuals' own magnitude keeps the perturbation's contribution above
  // the rounding floor at that scale, so it survives the subtraction.
  let delta = c.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
  let mut m = vec![vec![0.0; n]; n];
  for j in 0..n {
    vars[1 + state.len() + j] = delta;
    for (i, r) in residuals.iter().enumerate() {
      let Ok(v) = r.eval(&vars) else {
        return None;
      };
      if !v.is_finite() {
        return None;
      }
      m[i][j] = (v - c[i]) / delta;
    }
    vars[1 + state.len() + j] = 0.0;
  }

  // Gaussian elimination with partial pivoting on M·hvec = -c.
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

/// The synthesized variable name standing for the k-th derivative of `f`
/// in a rewritten residual.
fn placeholder_name(fname: &str, k: usize) -> String {
  format!("__ndsolve${fname}${k}")
}

/// Highest derivative order of `fname` appearing anywhere in `expr`
/// (`fname[x]` counts as order 0 only if a derivative also appears —
/// order is the max over `Derivative[k][fname][…]` occurrences).
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

/// Immutable children of an expression node, for the traversals above.
fn expr_children(expr: &Expr) -> Vec<&Expr> {
  match expr {
    Expr::List(items) => items.iter().collect(),
    Expr::FunctionCall { args, .. } => args.iter().collect(),
    Expr::BinaryOp { left, right, .. } => vec![left, right],
    Expr::UnaryOp { operand, .. } => vec![operand],
    Expr::Comparison { operands, .. } => operands.iter().collect(),
    Expr::CurriedCall { func, args } => {
      let mut v: Vec<&Expr> = vec![func];
      v.extend(args.iter());
      v
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => vec![pattern, replacement],
    _ => Vec::new(),
  }
}

/// Rebuild an expression with `f` applied to each direct child. Nodes
/// whose children aren't covered by `expr_children` are returned as-is.
fn map_children(expr: &Expr, f: &dyn Fn(&Expr) -> Expr) -> Expr {
  match expr {
    Expr::List(items) => Expr::List(items.iter().map(f).collect()),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(f).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(f(left)),
      right: Box::new(f(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(f(operand)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands.iter().map(f).collect(),
      operators: operators.clone(),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(f(func)),
      args: args.iter().map(f).collect(),
    },
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(f(pattern)),
      replacement: Box::new(f(replacement)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(f(pattern)),
      replacement: Box::new(f(replacement)),
    },
    other => other.clone(),
  }
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
}

impl NExpr {
  fn eval(&self, vars: &[f64]) -> f64 {
    match self {
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

/// Compile an expression over the given variable names to an `NExpr`.
/// Returns `None` for any construct outside the supported numeric subset
/// (the caller then falls back to symbolic evaluation).
fn compile_numeric(expr: &Expr, var_names: &[String]) -> Option<NExpr> {
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
        _ => None,
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
        "Sign" => Some(f64::signum),
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
        // Not one of the closed-form operators above — most commonly a
        // call to a function the session itself defined (a fixed boundary
        // value, a helper via `f[n_] := …`), which can only be resolved by
        // the full evaluator. Compiled as an External leaf rather than
        // bailing the whole residual out to the symbolic path.
        _ => Some(compile_external_leaf(expr, var_names)),
      }
    }
    other => Some(compile_external_leaf(other, var_names)),
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
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: other_factors.into_iter().cloned().collect(),
      }
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
  let x = || Expr::Identifier(x_name.to_string());
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
      Expr::Identifier(x_name.to_string()),
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
  let y_p = crate::evaluator::evaluate_function_call_ast(
    "Simplify",
    std::slice::from_ref(&y_p),
  )
  .unwrap_or(y_p);
  Some(y_p)
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
fn build_homogeneous_solution(
  roots: &[(f64, f64, usize)],
  x_name: &str,
) -> Expr {
  let x = Expr::Identifier(x_name.to_string());
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
  let x = Expr::Identifier(x_name.to_string());
  let r_expr = f64_to_nice_expr(r);
  let exponent = if matches!(&r_expr, Expr::Integer(1)) {
    x
  } else {
    times2(r_expr, x)
  };
  pow2(Expr::Constant("E".to_string()), exponent)
}

/// Create Cos[β*x] or Sin[β*x] expression
fn make_trig_term(func: &str, beta: f64, x_name: &str) -> Expr {
  let x = Expr::Identifier(x_name.to_string());
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
  // Try sqrt expressions: check if f^2 is a nice rational
  let f2 = f * f;
  if f > 0.0 {
    for denom in 1..=12 {
      let numer = f2 * denom as f64;
      if (numer - numer.round()).abs() < 1e-10 {
        let n = numer.round() as i128;
        let d = denom as i128;
        // f = Sqrt[n/d]
        if d == 1 {
          return make_sqrt(Expr::Integer(n));
        }
        return div2(make_sqrt(Expr::Integer(n)), make_sqrt(Expr::Integer(d)));
      }
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
      Expr::Identifier(x_name.to_string()),
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
    Expr::Identifier(x_name.to_string()),
  ])?;

  let mu = pow2(Expr::Constant("E".to_string()), p_integral.clone());

  let mu_q =
    crate::functions::calculus_ast::simplify(times2(mu.clone(), q_expr));

  let mu_q_integral = crate::functions::calculus_ast::integrate_ast(&[
    mu_q,
    Expr::Identifier(x_name.to_string()),
  ])?;

  // y = E^(-∫P dx) * (∫(μ*Q)dx + C[1])
  let neg_p_integral = negate_expr(&p_integral);
  let inv_mu = pow2(Expr::Constant("E".to_string()), neg_p_integral);

  // Distribute the integrating factor over the particular part and the
  // constant separately, matching wolframscript's form, e.g.
  // (E^(3x)/3 + C[1])/E^(2x) -> E^x/3 + C[1]/E^(2x). Each product is simplified
  // (not fully expanded), so a grouped particular like (-Cos[x]+3Sin[x])/10
  // stays grouped rather than splitting into separate terms.
  let particular = crate::functions::calculus_ast::simplify(times2(
    inv_mu.clone(),
    mu_q_integral,
  ));
  let homogeneous =
    crate::functions::calculus_ast::simplify(times2(inv_mu, make_c(1)));
  Ok(plus2(particular, homogeneous))
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
    Expr::Identifier(var.to_string()),
  ])
  .ok()?;
  (!mentions_head(&result, "Integrate")).then_some(result)
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
  let x = Expr::Identifier(x_name.to_string());
  let exponent = match coeff {
    Expr::Integer(1) => x,
    Expr::Integer(-1) => neg1(x),
    _ => times2(coeff.clone(), x),
  };
  pow2(Expr::Constant("E".to_string()), exponent)
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
      return Some(times2(
        f64_to_nice_expr(c),
        Expr::Identifier(x_name.to_string()),
      ));
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
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_c_constant(left, pattern, replacement)),
      right: Box::new(substitute_c_constant(right, pattern, replacement)),
    },
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
    _ => Err(InterpreterError::EvaluationError(format!(
      "cannot convert {} to a numeric value",
      crate::syntax::expr_to_string(expr)
    ))),
  }
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

  // Extract InterpolationOrder option (default 3). A 2-D grid may give one
  // order per axis (`InterpolationOrder -> {orderX, orderY}`); a bare
  // integer applies to both.
  let mut interp_order: i128 = 3;
  let mut interp_order_xy: Option<(i128, i128)> = None;
  let data_arg = &args[0];

  let mut apply_interpolation_order = |replacement: &Expr| {
    if let Some(n) = crate::functions::math_ast::expr_to_i128(replacement) {
      interp_order = n;
    } else if let Expr::List(items) = replacement
      && items.len() == 2
      && let (Some(a), Some(b)) = (
        crate::functions::math_ast::expr_to_i128(&items[0]),
        crate::functions::math_ast::expr_to_i128(&items[1]),
      )
    {
      interp_order_xy = Some((a, b));
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

  // ListInterpolation of a rectangular numeric matrix is a 2-D grid (values on
  // the integer grid 1..rows × 1..cols), not a list of {x, y} points.
  if head == "ListInterpolation"
    && let Some(grid) = as_2d_scalar_grid(data_list)
  {
    return Ok(build_2d_list_interpolation(&grid, interp_order, head));
  }

  // `Interpolation` (not `ListInterpolation`, whose triples are raw grid
  // rows — see above) of a flat `{x, y, z}` list — the shape
  // `Flatten[Table[Table[{x, y, f[x, y]}, {y, ys}], {x, xs}], 1]` (or the
  // equivalent built with `Join`) produces — is 2-D scattered data. It
  // interpolates like `ListInterpolation`'s grid once the distinct x/y
  // coordinates recover the grid structure; see `try_2d_scattered_interpolation`.
  if head != "ListInterpolation"
    && matches!(&data_list[0], Expr::List(items) if items.len() == 3)
    && let Some(result) = try_2d_scattered_interpolation(
      data_list,
      interp_order_xy.map_or(interp_order, |(a, _)| a),
      interp_order_xy.map_or(interp_order, |(_, b)| b),
      head,
    )
  {
    return Ok(result);
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
  let mut order = interp_order.clamp(1, 3) as usize;
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

/// A rectangular numeric matrix (≥2 rows, ≥2 columns, all rows the same length,
/// every entry a number). Returns the original entries (preserving Integer /
/// Real types) alongside their `f64` values. Returns `None` for ragged matrices,
/// 1-D lists, or any non-numeric entry — so 1-D `ListInterpolation` and
/// `Interpolation` of `{x, y}` pairs keep the existing path.
type Grid = (Vec<Vec<Expr>>, Vec<Vec<f64>>);
fn as_2d_scalar_grid(rows: &[Expr]) -> Option<Grid> {
  if rows.len() < 2 {
    return None;
  }
  let mut exprs: Vec<Vec<Expr>> = Vec::with_capacity(rows.len());
  let mut vals: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
  let mut width: Option<usize> = None;
  for row in rows {
    let Expr::List(cells) = row else {
      return None;
    };
    if cells.len() < 2 || *width.get_or_insert(cells.len()) != cells.len() {
      return None;
    }
    let mut row_e = Vec::with_capacity(cells.len());
    let mut row_v = Vec::with_capacity(cells.len());
    for c in cells {
      let ce = crate::evaluator::evaluate_expr_to_expr(c).unwrap_or(c.clone());
      let v = match &ce {
        Expr::Integer(n) => *n as f64,
        Expr::Real(f) => *f,
        _ => return None,
      };
      row_e.push(ce);
      row_v.push(v);
    }
    exprs.push(row_e);
    vals.push(row_v);
  }
  Some((exprs, vals))
}

/// Build the `InterpolatingFunction` for a 2-D grid. Stored form is
/// `InterpolatingFunction[{{1, nr}, {1, nc}}, gridExprs, {orderR, orderC}]`;
/// the `{orderR, orderC}` list (rather than an integer) marks it as 2-D for the
/// evaluator. Orders are clamped per dimension to `min(requested, dim - 1, 3)`.
fn build_2d_list_interpolation(
  grid: &Grid,
  interp_order: i128,
  head: &str,
) -> crate::syntax::Expr {
  let (exprs, _vals) = &grid;
  let nr = exprs.len();
  let nc = exprs[0].len();
  let want = interp_order.clamp(1, 3) as usize;
  let order_r = want.min(nr - 1);
  let order_c = want.min(nc - 1);
  if order_r < want || order_c < want {
    crate::emit_message(&format!(
      "{head}::inhr: Requested order is too high; order has been reduced to {{{order_r}, {order_c}}}."
    ));
  }
  let domain = Expr::List(
    vec![
      Expr::List(vec![Expr::Integer(1), Expr::Integer(nr as i128)].into()),
      Expr::List(vec![Expr::Integer(1), Expr::Integer(nc as i128)].into()),
    ]
    .into(),
  );
  let grid_expr = Expr::List(
    exprs
      .iter()
      .map(|row| Expr::List(row.clone().into()))
      .collect(),
  );
  let orders = Expr::List(
    vec![
      Expr::Integer(order_r as i128),
      Expr::Integer(order_c as i128),
    ]
    .into(),
  );
  call("InterpolatingFunction", vec![domain, grid_expr, orders])
}

/// The relative tolerance two coordinates must fall within to count as "the
/// same" grid line — generous enough for the roundoff two different `Table`
/// passes over the same arithmetic can leave, tight enough not to merge
/// genuinely distinct samples.
fn same_coordinate(a: f64, b: f64) -> bool {
  (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

/// Detect a flat `{x, y, z}` triple list that tiles a complete rectangular
/// grid over its distinct x and y coordinates — the shape
/// `Flatten[Table[Table[{x, y, f[x, y]}, {y, ys}], {x, xs}], 1]` (or the
/// equivalent built with `Join`) produces — and build the 2-D
/// `InterpolatingFunction` for it: `InterpolatingFunction[{{xmin, xmax},
/// {ymin, ymax}}, grid, {orderX, orderY}, {xcoords, ycoords}]`, the explicit-
/// coordinate counterpart of `build_2d_list_interpolation`'s implicit
/// integer grid.
///
/// Returns `Ok(None)` when the data isn't in this shape (a non-triple entry,
/// a non-numeric coordinate, or coordinates that don't tile a full grid) so
/// the caller falls back to its other formats. Genuinely scattered, non-grid
/// 2-D data — which Wolfram interpolates via Delaunay triangulation — is not
/// supported here.
fn try_2d_scattered_interpolation(
  data_list: &[Expr],
  order_x: i128,
  order_y: i128,
  head: &str,
) -> Option<Expr> {
  // At least a 2x2 grid is needed for either axis to interpolate at all.
  if data_list.len() < 4 {
    return None;
  }
  let mut triples: Vec<(Expr, f64, Expr, f64, Expr, f64)> =
    Vec::with_capacity(data_list.len());
  for item in data_list {
    let Expr::List(parts) = item else {
      return None;
    };
    if parts.len() != 3 {
      return None;
    }
    let (Ok(x), Ok(y), Ok(z)) = (
      interp_value_to_f64(&parts[0]),
      interp_value_to_f64(&parts[1]),
      interp_value_to_f64(&parts[2]),
    ) else {
      return None;
    };
    triples.push((
      parts[0].clone(),
      x,
      parts[1].clone(),
      y,
      parts[2].clone(),
      z,
    ));
  }

  // Distinct x/y values, sorted, each keeping the first expression seen for
  // it (so an exact/integer coordinate stays exact).
  let mut xs_pairs: Vec<(f64, Expr)> = Vec::new();
  let mut ys_pairs: Vec<(f64, Expr)> = Vec::new();
  for (xe, x, ye, y, _ze, _z) in &triples {
    if !xs_pairs.iter().any(|(xv, _)| same_coordinate(*xv, *x)) {
      xs_pairs.push((*x, xe.clone()));
    }
    if !ys_pairs.iter().any(|(yv, _)| same_coordinate(*yv, *y)) {
      ys_pairs.push((*y, ye.clone()));
    }
  }
  xs_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
  ys_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
  let nr = xs_pairs.len();
  let nc = ys_pairs.len();
  // Fewer than 2 distinct values along an axis, or a point count that isn't
  // the full nr*nc product, means this isn't a complete rectangular grid.
  if nr < 2 || nc < 2 || nr * nc != triples.len() {
    return None;
  }

  let mut grid: Vec<Vec<Option<Expr>>> = vec![vec![None; nc]; nr];
  for (_xe, x, _ye, y, ze, z) in &triples {
    let i = xs_pairs
      .iter()
      .position(|(xv, _)| same_coordinate(*xv, *x))?;
    let j = ys_pairs
      .iter()
      .position(|(yv, _)| same_coordinate(*yv, *y))?;
    if grid[i][j].is_some() {
      // Duplicate (x, y) — not a well-formed grid.
      return None;
    }
    grid[i][j] = Some(exact_coordinate(ze, *z));
  }
  if grid.iter().flatten().any(Option::is_none) {
    // A combination of some x with some y is missing — a partial grid,
    // which is genuinely scattered data this path does not support.
    return None;
  }

  let x_exprs: Vec<Expr> = xs_pairs
    .iter()
    .map(|(v, e)| exact_coordinate(e, *v))
    .collect();
  let y_exprs: Vec<Expr> = ys_pairs
    .iter()
    .map(|(v, e)| exact_coordinate(e, *v))
    .collect();
  let grid_exprs: Vec<Vec<Expr>> = grid
    .into_iter()
    .map(|row| row.into_iter().map(Option::unwrap).collect())
    .collect();

  let want_r = order_x.clamp(1, 3) as usize;
  let want_c = order_y.clamp(1, 3) as usize;
  let order_r = want_r.min(nr - 1);
  let order_c = want_c.min(nc - 1);
  if order_r < want_r || order_c < want_c {
    crate::emit_message(&format!(
      "{head}::inhr: Requested order is too high; order has been reduced to {{{order_r}, {order_c}}}."
    ));
  }

  let domain = Expr::List(
    vec![
      Expr::List(vec![x_exprs[0].clone(), x_exprs[nr - 1].clone()].into()),
      Expr::List(vec![y_exprs[0].clone(), y_exprs[nc - 1].clone()].into()),
    ]
    .into(),
  );
  let grid_expr = Expr::List(
    grid_exprs
      .into_iter()
      .map(|row| Expr::List(row.into()))
      .collect(),
  );
  let orders = Expr::List(
    vec![
      Expr::Integer(order_r as i128),
      Expr::Integer(order_c as i128),
    ]
    .into(),
  );
  let coords = Expr::List(
    vec![Expr::List(x_exprs.into()), Expr::List(y_exprs.into())].into(),
  );

  Some(call(
    "InterpolatingFunction",
    vec![domain, grid_expr, orders, coords],
  ))
}

/// Evaluate a 2-D grid `InterpolatingFunction` whose coordinates are
/// explicit (rather than the implicit integer grid
/// `evaluate_interpolating_function_2d` assumes) — the form
/// `try_2d_scattered_interpolation` builds from a flat `{x, y, z}` triple
/// list. Otherwise identical: tensor-product local Lagrange interpolation,
/// exact at grid points.
fn evaluate_interpolating_function_2d_explicit(
  domain: &Expr,
  grid_rows: &[Expr],
  orders: &[Expr],
  coords: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let order_r = match &orders[0] {
    Expr::Integer(n) => *n as usize,
    _ => 1,
  };
  let order_c = match &orders[1] {
    Expr::Integer(n) => *n as usize,
    _ => 1,
  };

  let (Expr::List(x_coord_exprs), Expr::List(y_coord_exprs)) =
    (&coords[0], &coords[1])
  else {
    return Err(InterpreterError::EvaluationError(
      "InterpolatingFunction: invalid coordinates".into(),
    ));
  };
  let xs: Vec<f64> = x_coord_exprs
    .iter()
    .map(|e| interp_value_to_f64(e).unwrap_or(f64::NAN))
    .collect();
  let ys: Vec<f64> = y_coord_exprs
    .iter()
    .map(|e| interp_value_to_f64(e).unwrap_or(f64::NAN))
    .collect();
  let nr = xs.len();
  let nc = ys.len();

  // A single grid cell, decoded lazily: a query only ever needs an
  // `(order + 1) x (order + 1)` window of cells, not the full `nr x nc`
  // grid — materializing the whole thing up front made every call
  // `O(nr * nc)` regardless of interpolation order, which is ruinous for
  // the grids `NDSolve`'s PDE branch produces (hundreds of time steps by
  // tens of space steps, resampled at every `ContourPlot` point).
  let cell = |i: usize, j: usize| -> Result<&Expr, InterpreterError> {
    let Expr::List(cells) = &grid_rows[i] else {
      return Err(InterpreterError::EvaluationError(
        "InterpolatingFunction: invalid 2-D grid".into(),
      ));
    };
    cells.get(j).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "InterpolatingFunction: invalid 2-D grid".into(),
      )
    })
  };
  let cell_f64 = |i: usize, j: usize| -> Result<f64, InterpreterError> {
    Ok(match cell(i, j)? {
      Expr::Integer(n) => *n as f64,
      Expr::Real(f) => *f,
      other => interp_value_to_f64(other).unwrap_or(f64::NAN),
    })
  };

  let unevaluated = || Expr::CurriedCall {
    func: Box::new(call(
      "InterpolatingFunction",
      vec![
        domain.clone(),
        Expr::List(grid_rows.to_vec().into()),
        Expr::List(orders.to_vec().into()),
        Expr::List(coords.to_vec().into()),
      ],
    )),
    args: call_args.to_vec(),
  };

  if nr == 0 || nc == 0 {
    return Ok(unevaluated());
  }

  if call_args.len() == 1
    && let Expr::String(prop) = &call_args[0]
  {
    match prop.as_str() {
      "Domain" => return Ok(domain.clone()),
      "Coordinates" => {
        return Ok(Expr::List(
          vec![
            Expr::List(x_coord_exprs.clone()),
            Expr::List(y_coord_exprs.clone()),
          ]
          .into(),
        ));
      }
      "Grid" => {
        let mut pts = Vec::with_capacity(nr * nc);
        for xe in x_coord_exprs {
          for ye in y_coord_exprs {
            pts.push(Expr::List(vec![xe.clone(), ye.clone()].into()));
          }
        }
        return Ok(Expr::List(pts.into()));
      }
      "ValuesOnGrid" => return Ok(Expr::List(grid_rows.to_vec().into())),
      "InterpolationOrder" => return Ok(Expr::List(orders.to_vec().into())),
      _ => {}
    }
  }

  let coord_args: Vec<Expr> = match call_args {
    [Expr::List(pair)] if pair.len() == 2 => pair.to_vec(),
    other => other.to_vec(),
  };
  if coord_args.len() != 2 {
    return Ok(unevaluated());
  }
  let coord_exprs: Vec<Expr> = coord_args
    .iter()
    .map(|a| {
      crate::evaluator::evaluate_expr_to_expr(a).unwrap_or_else(|_| a.clone())
    })
    .collect();
  let coord = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(f) => Some(*f),
      _ if is_exact_number(e) => interp_value_to_f64(e).ok(),
      _ => None,
    }
  };
  let (Some(x), Some(y)) = (coord(&coord_exprs[0]), coord(&coord_exprs[1]))
  else {
    return Ok(unevaluated());
  };

  // Exact grid point: return the stored entry, preserving its type only
  // when both query coordinates were given as exact integers (matching
  // `evaluate_interpolating_function_2d`).
  let int_coords = coord_exprs.iter().all(|a| matches!(a, Expr::Integer(_)));
  if let (Some(i), Some(j)) = (
    xs.iter().position(|&xv| same_coordinate(xv, x)),
    ys.iter().position(|&yv| same_coordinate(yv, y)),
  ) {
    if int_coords {
      return Ok(cell(i, j)?.clone());
    }
    return Ok(Expr::Real(cell_f64(i, j)?));
  }

  let (x_lo, x_hi) = (xs[0].min(xs[nr - 1]), xs[0].max(xs[nr - 1]));
  let (y_lo, y_hi) = (ys[0].min(ys[nc - 1]), ys[0].max(ys[nc - 1]));
  if x < x_lo || x > x_hi || y < y_lo || y > y_hi {
    crate::emit_message(&format!(
      "InterpolatingFunction::dmval: Input value {{{}, {}}} lies outside the range of data in the interpolating function. Extrapolation will be used.",
      crate::syntax::format_expr(
        &coord_exprs[0],
        crate::syntax::ExprForm::Output
      ),
      crate::syntax::format_expr(
        &coord_exprs[1],
        crate::syntax::ExprForm::Output
      )
    ));
  }

  // Only the local `(order + 1)`-point stencil on each axis feeds the
  // result, so the cells outside that window never need decoding.
  let eff_order_r = order_r.min(nr - 1).max(1);
  let eff_order_c = order_c.min(nc - 1).max(1);
  let (r_start, r_end) =
    lagrange_window(nr, bracket_index_f64(&xs, x), eff_order_r);
  let (c_start, c_end) =
    lagrange_window(nc, bracket_index_f64(&ys, y), eff_order_c);
  let ys_window = &ys[c_start..c_end];
  let mut col_interp: Vec<f64> = Vec::with_capacity(r_end - r_start);
  for i in r_start..r_end {
    let mut row_vals: Vec<f64> = Vec::with_capacity(c_end - c_start);
    for j in c_start..c_end {
      row_vals.push(cell_f64(i, j)?);
    }
    col_interp.push(interp_1d_xy_f64(ys_window, &row_vals, y, eff_order_c));
  }
  let result =
    interp_1d_xy_f64(&xs[r_start..r_end], &col_interp, x, eff_order_r);
  Ok(Expr::Real(result))
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

/// 1-D local Lagrange interpolation of `values` (sampled at positions
/// `1..=values.len()`) at `coord`, using `order + 1` nearest samples. Mirrors
/// the point-selection of `lagrange_interpolate` so 2-D results match the 1-D
/// engine (and wolframscript).
fn interp_1d_f64(values: &[f64], coord: f64, order: usize) -> f64 {
  let n = values.len();
  let coord = coord.max(1.0).min(n as f64);
  // Interval index: largest i with (i+1) <= coord.
  let mut idx = (coord.floor() as usize).saturating_sub(1);
  if idx >= n - 1 {
    idx = n - 2;
  }
  let needed = (order + 1).min(n);
  // Order 1 uses the bracketing interval [idx, idx+1] (linear interpolation
  // within the cell), matching the 1-D engine; higher orders center the
  // (order+1)-point stencil on that interval, always containing both
  // interval endpoints.
  let start = if needed <= 2 {
    idx.min(n - needed)
  } else {
    idx.saturating_sub((needed - 2) / 2).min(n - needed)
  };
  let mut acc = 0.0;
  for i in 0..needed {
    let xi = (start + i + 1) as f64;
    let yi = values[start + i];
    let mut term = yi;
    for j in 0..needed {
      if j != i {
        let xj = (start + j + 1) as f64;
        term *= (coord - xj) / (xi - xj);
      }
    }
    acc += term;
  }
  acc
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
fn is_exact_number(e: &Expr) -> bool {
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
pub fn evaluate_interpolating_function(
  func_args: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // 2-D grid form: InterpolatingFunction[{{1,nr},{1,nc}}, grid, {orderR, orderC}].
  // The `{orderR, orderC}` list marks it as 2-D.
  if func_args.len() == 3
    && let Expr::List(orders) = &func_args[2]
    && orders.len() == 2
    && let Expr::List(grid_rows) = &func_args[1]
  {
    return evaluate_interpolating_function_2d(grid_rows, orders, call_args);
  }

  // 2-D grid form with explicit coordinates: InterpolatingFunction[{{xmin,
  // xmax}, {ymin, ymax}}, grid, {orderX, orderY}, {xcoords, ycoords}] —
  // `try_2d_scattered_interpolation`'s output.
  if func_args.len() == 4
    && let Expr::List(orders) = &func_args[2]
    && orders.len() == 2
    && let Expr::List(grid_rows) = &func_args[1]
    && let Expr::List(coords) = &func_args[3]
    && coords.len() == 2
    && matches!(&coords[0], Expr::List(_))
    && matches!(&coords[1], Expr::List(_))
  {
    return evaluate_interpolating_function_2d_explicit(
      &func_args[0],
      grid_rows,
      orders,
      coords,
      call_args,
    );
  }

  if !(2..=4).contains(&func_args.len()) || call_args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InterpolatingFunction expects domain and data, called with one argument"
        .into(),
    ));
  }

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
  ) && (x_val < x_lo || x_val > x_hi)
  {
    crate::emit_message(&format!(
      "InterpolatingFunction::dmval: Input value {{{}}} lies outside the range of data in the interpolating function. Extrapolation will be used.",
      crate::syntax::format_expr(&x_val_expr, crate::syntax::ExprForm::Output)
    ));
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

  // Check for exact grid point match — return the stored y-value directly
  // to preserve original types (e.g. Integer for ListInterpolation with integer data).
  if derivative_order == 0 {
    for pt in data_points {
      if let Expr::List(pair) = pt
        && pair.len() == 2
        && let Ok(xp) = interp_value_to_f64(&pair[0])
        && (xp - x_val).abs() < 1e-15
      {
        return Ok(pair[1].clone());
      }
    }
  }

  // Binary search for the interval containing x; a point outside the data
  // range picks up the nearest interval and extrapolates along it.
  let idx = find_interval(data_points, x_val.max(x_first).min(x_last), n)?;
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

/// Evaluate a 2-D grid InterpolatingFunction at `[x, y]` via tensor-product
/// local Lagrange interpolation: interpolate each row along the column
/// direction at `y`, then interpolate those values along the row direction at
/// `x`. Exact integer grid points return the stored entry (preserving type).
fn evaluate_interpolating_function_2d(
  grid_rows: &[Expr],
  orders: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let nr = grid_rows.len();
  let order_r = match &orders[0] {
    Expr::Integer(n) => *n as usize,
    _ => 1,
  };
  let order_c = match &orders[1] {
    Expr::Integer(n) => *n as usize,
    _ => 1,
  };

  // Materialize the grid as f64 values and as original expressions.
  let mut vals: Vec<Vec<f64>> = Vec::with_capacity(nr);
  let mut exprs: Vec<Vec<Expr>> = Vec::with_capacity(nr);
  for row in grid_rows {
    let Expr::List(cells) = row else {
      return Err(InterpreterError::EvaluationError(
        "InterpolatingFunction: invalid 2-D grid".into(),
      ));
    };
    let mut rv = Vec::with_capacity(cells.len());
    let mut re = Vec::with_capacity(cells.len());
    for c in cells {
      rv.push(match c {
        Expr::Integer(n) => *n as f64,
        Expr::Real(f) => *f,
        _ => 0.0,
      });
      re.push(c.clone());
    }
    vals.push(rv);
    exprs.push(re);
  }
  let nc = vals[0].len();

  let domain2d = || {
    Expr::List(
      vec![
        Expr::List(vec![Expr::Integer(1), Expr::Integer(nr as i128)].into()),
        Expr::List(vec![Expr::Integer(1), Expr::Integer(nc as i128)].into()),
      ]
      .into(),
    )
  };
  let unevaluated = || Expr::CurriedCall {
    func: Box::new(call(
      "InterpolatingFunction",
      vec![
        domain2d(),
        Expr::List(grid_rows.to_vec().into()),
        Expr::List(orders.to_vec().into()),
      ],
    )),
    args: call_args.to_vec(),
  };

  // Property access (e.g. ["Domain"]) — return the rectangular domain.
  if call_args.len() == 1
    && let Expr::String(prop) = &call_args[0]
    && prop == "Domain"
  {
    return Ok(domain2d());
  }
  // Other properties: leave unevaluated.

  // The coordinates may come either as two arguments or as one list.
  let coord_args: Vec<Expr> = match call_args {
    [Expr::List(pair)] if pair.len() == 2 => pair.to_vec(),
    other => other.to_vec(),
  };
  if coord_args.len() != 2 {
    return Ok(unevaluated());
  }

  let coord_exprs: Vec<Expr> = coord_args
    .iter()
    .map(|a| {
      crate::evaluator::evaluate_expr_to_expr(a).unwrap_or_else(|_| a.clone())
    })
    .collect();
  let coord = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(f) => Some(*f),
      _ if is_exact_number(e) => interp_value_to_f64(e).ok(),
      _ => None,
    }
  };
  let (Some(x), Some(y)) = (coord(&coord_exprs[0]), coord(&coord_exprs[1]))
  else {
    // Non-numeric coordinate: stay symbolic.
    return Ok(unevaluated());
  };

  // Exact grid point: return the stored entry, but only keep its (Integer)
  // type when both coordinates were given as integers — a real coordinate
  // forces a real result (matching wolframscript: `[1, 3]` → 9, `[1., 3.]` → 9.).
  let int_coords = coord_exprs.iter().all(|a| matches!(a, Expr::Integer(_)));
  let xi = x.round();
  let yi = y.round();
  if (x - xi).abs() < 1e-12
    && (y - yi).abs() < 1e-12
    && xi >= 1.0
    && xi <= nr as f64
    && yi >= 1.0
    && yi <= nc as f64
  {
    let entry = &exprs[xi as usize - 1][yi as usize - 1];
    if int_coords {
      return Ok(entry.clone());
    }
    return Ok(Expr::Real(vals[xi as usize - 1][yi as usize - 1]));
  }

  // Exact coordinates over an exact grid interpolate exactly, the way
  // wolframscript reports ListInterpolation[{{1, 2}, {3, 4}}][{3/2, 3/2}]
  // as 5/2.
  if coord_exprs.iter().all(is_exact_number)
    && exprs.iter().flatten().all(is_exact_number)
    && let Some(rows) = exprs
      .iter()
      .map(|row| interp_1d_exact(row, &coord_exprs[1], y, order_c))
      .collect::<Option<Vec<Expr>>>()
    && let Some(result) = interp_1d_exact(&rows, &coord_exprs[0], x, order_r)
  {
    return Ok(result);
  }

  // Interpolate each row along columns at y, then along rows at x.
  let col_interp: Vec<f64> = vals
    .iter()
    .map(|row| interp_1d_f64(row, y, order_c))
    .collect();
  let result = interp_1d_f64(&col_interp, x, order_r);
  Ok(Expr::Real(result))
}

/// Exact counterpart of `interp_1d_f64`: interpolate `values` sitting on the
/// integer grid 1..n at the exact coordinate `coord`, using the same stencil.
fn interp_1d_exact(
  values: &[Expr],
  coord: &Expr,
  coord_f64: f64,
  order: usize,
) -> Option<Expr> {
  let n = values.len();
  if n == 0 {
    return None;
  }
  let clamped = coord_f64.max(1.0).min(n as f64);
  let mut idx = (clamped.floor() as usize).saturating_sub(1);
  if idx >= n - 1 {
    idx = n - 2;
  }
  let needed = (order + 1).min(n);
  let start = if needed <= 2 {
    idx.min(n - needed)
  } else {
    idx.saturating_sub((needed - 2) / 2).min(n - needed)
  };
  let points: Vec<Expr> = values
    .iter()
    .enumerate()
    .map(|(i, v)| {
      Expr::List(vec![Expr::Integer(i as i128 + 1), v.clone()].into())
    })
    .collect();
  let poly = lagrange_polynomial(&points, coord, start, start + needed, true)?;
  crate::evaluator::evaluate_expr_to_expr(&poly).ok()
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
        factors.push(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(minus(var, &xs[j])),
          right: Box::new(minus(&xs[i], &xs[j])),
        });
      }
    }
    factors.retain(|f| !matches!(f, Expr::Integer(1)));
    terms.push(call("Times", factors));
  }
  Some(call("Plus", terms))
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
  let var = Expr::Identifier("\u{2620}ifderiv\u{2620}".to_string());
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

  // Expand the nested Newton form
  // c[0] + (x-x0)(c[1] + (x-x1)(c[2] + ...))
  // into ascending-power monomial coefficients, via repeated polynomial
  // multiplication by (x - x_k) — cheap since m is a handful of points.
  let mut monomial = vec![coeffs[m - 1]];
  for k in (0..m - 1).rev() {
    // monomial *= (x - xs[k])
    let mut shifted = vec![0.0; monomial.len() + 1];
    for (i, c) in monomial.iter().enumerate() {
      shifted[i + 1] += c;
      shifted[i] -= c * xs[k];
    }
    monomial = shifted;
    // monomial += coeffs[k] (constant term)
    monomial[0] += coeffs[k];
  }

  if derivative_order >= monomial.len() {
    return Ok(Expr::Real(0.0));
  }
  // Differentiate `derivative_order` times: term c*x^p -> c*p*x^(p-1).
  for _ in 0..derivative_order {
    monomial = monomial
      .iter()
      .enumerate()
      .skip(1)
      .map(|(p, c)| c * p as f64)
      .collect();
  }

  // Horner evaluation of the (now differentiated) monomial polynomial.
  let value = monomial.iter().rev().fold(0.0, |acc, c| acc * x_val + c);
  Ok(Expr::Real(value))
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
  let n_var = |s: &str| Expr::Identifier(s.to_string());
  let exp_part = if c_eff == 0 {
    Expr::Integer(1)
  } else {
    let exponent = if c_eff == 1 {
      n_var(xn)
    } else {
      times2(Expr::Integer(c_eff), n_var(xn))
    };
    pow2(Expr::Constant("E".to_string()), exponent)
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
  let n_var = |s: &str| Expr::Identifier(s.to_string());
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
  let n_var = |s: &str| Expr::Identifier(s.to_string());
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
  let n_var = |s: &str| Expr::Identifier(s.to_string());
  let rule = if return_call_form {
    Expr::Rule {
      pattern: Box::new(call(fname, vec![n_var(xn), n_var(yn)])),
      replacement: Box::new(body),
    }
  } else {
    Expr::Rule {
      pattern: Box::new(Expr::Identifier(fname.to_string())),
      replacement: Box::new(call(
        "Function",
        vec![Expr::List(vec![n_var(xn), n_var(yn)].into()), body],
      )),
    }
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
  let arg_of = |lambda: Expr| {
    eval(plus2(
      times2(lambda, Expr::Identifier(xn.to_string())),
      Expr::Identifier(yn.to_string()),
    ))
  };
  let c_of = |k: i128, arg: Expr| Expr::CurriedCall {
    func: Box::new(call1("C", Expr::Integer(k))),
    args: vec![arg],
  };

  if is_zero(&disc) {
    // A repeated root: the second solution picks up a factor of x.
    let arg = arg_of(root(&a, 1));
    return Some(plus2(
      c_of(1, arg.clone()),
      times2(Expr::Identifier(xn.to_string()), c_of(2, arg)),
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
  crate::syntax::expr_to_string(expr)
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
