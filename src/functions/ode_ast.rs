//! AST-native ODE solving functions (DSolve, NDSolve).
//!
//! DSolve solves ordinary differential equations symbolically.
//! NDSolve solves initial-value problems numerically using RK4.

use crate::InterpreterError;
use crate::functions::math_ast::make_sqrt;
use crate::syntax::{BinaryOperator, Expr};

// ─── DSolve ────────────────────────────────────────────────────────────

/// DSolve[eqn, y[x], x] or DSolve[{eqn, ic1, ...}, y[x], x]
/// Also DSolve[eqn, y, x] (returns Function form)
pub fn dsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // An ODE Woxi can't classify/solve should stay unevaluated (like
  // wolframscript for genuinely unsolvable equations) rather than leaking an
  // internal "DSolve: …" error to the user.
  match dsolve_ast_inner(args) {
    Err(InterpreterError::EvaluationError(msg)) if msg.starts_with("DSolve:") => {
      Ok(Expr::FunctionCall {
        name: "DSolve".to_string(),
        args: args.to_vec().into(),
      })
    }
    other => other,
  }
}

fn dsolve_ast_inner(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "DSolve".to_string(),
      args: args.to_vec().into(),
    });
  }

  let eqns_arg = &args[0];
  let dep_arg = &args[1];
  let indep_var = &args[2];

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
    }
  }

  // Extract independent variable name
  let x_name = match indep_var {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DSolve".to_string(),
        args: args.to_vec().into(),
      });
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
          return Ok(Expr::FunctionCall {
            name: "DSolve".to_string(),
            args: args.to_vec().into(),
          });
        }
      } else {
        return Ok(Expr::FunctionCall {
          name: "DSolve".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
    // y form → return y -> Function[{x}, expr]
    Expr::Identifier(name) => (name.clone(), true),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DSolve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Separate equations and initial conditions
  let (ode_expr, initial_conditions) = match eqns_arg {
    Expr::List(items) => {
      // First item should be the ODE, rest are initial conditions
      if items.is_empty() {
        return Ok(Expr::FunctionCall {
          name: "DSolve".to_string(),
          args: args.to_vec().into(),
        });
      }
      let mut ics = Vec::new();
      let mut ode = None;
      for item in items {
        if is_initial_condition(item, &y_name, &x_name) {
          ics.push(item.clone());
        } else {
          if ode.is_some() {
            // Multiple ODEs not supported
            return Ok(Expr::FunctionCall {
              name: "DSolve".to_string(),
              args: args.to_vec().into(),
            });
          }
          ode = Some(item.clone());
        }
      }
      match ode {
        Some(o) => (o, ics),
        None => {
          return Ok(Expr::FunctionCall {
            name: "DSolve".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
    }
    // Single equation, no ICs
    _ => (eqns_arg.clone(), Vec::new()),
  };

  // Parse the ODE: extract lhs == rhs, move everything to lhs - rhs = 0
  let ode_normalized = normalize_equation(&ode_expr)?;

  // Collect terms: classify each additive term by derivative order
  let terms = collect_ode_terms(&ode_normalized, &y_name, &x_name)?;

  // Determine max order
  let max_order = terms.iter().map(|t| t.order).max().unwrap_or(0);
  if max_order == 0 {
    // Not actually an ODE
    return Ok(Expr::FunctionCall {
      name: "DSolve".to_string(),
      args: args.to_vec().into(),
    });
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
    return Ok(Expr::FunctionCall {
      name: "DSolve".to_string(),
      args: args.to_vec().into(),
    });
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

  // Build result: {{y[x] -> solution}} or {{y -> Function[{x}, solution]}}
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

  Ok(Expr::List(vec![Expr::List(vec![rule].into())].into()))
}

// ─── NDSolve ───────────────────────────────────────────────────────────

/// NDSolve[{eqn, ic1, ...}, y[x], {x, xmin, xmax}]
/// Also NDSolve[{eqn, ic1, ...}, y, {x, xmin, xmax}]
pub fn ndsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "NDSolve".to_string(),
      args: args.to_vec().into(),
    });
  }

  let eqns_arg = &args[0];
  let dep_arg = &args[1];
  let domain_arg = &args[2];

  // Extract domain {x, xmin, xmax}
  let (x_name, x_min, x_max) = match domain_arg {
    Expr::List(items) if items.len() == 3 => {
      let xn = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "NDSolve".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      let xmin =
        expr_to_f64(&crate::evaluator::evaluate_expr_to_expr(&items[1])?)?;
      let xmax =
        expr_to_f64(&crate::evaluator::evaluate_expr_to_expr(&items[2])?)?;
      (xn, xmin, xmax)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NDSolve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Extract dependent variable name
  let (y_name, function_form) = match dep_arg {
    Expr::FunctionCall { name, args: fargs } if fargs.len() == 1 => {
      if let Expr::Identifier(xn) = &fargs[0] {
        if xn == &x_name {
          (name.clone(), false)
        } else {
          return Ok(Expr::FunctionCall {
            name: "NDSolve".to_string(),
            args: args.to_vec().into(),
          });
        }
      } else {
        return Ok(Expr::FunctionCall {
          name: "NDSolve".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
    Expr::Identifier(name) => (name.clone(), true),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NDSolve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Extract equations and initial conditions from list
  let items = match eqns_arg {
    Expr::List(items) => items.clone(),
    _ => vec![eqns_arg.clone()].into(),
  };

  // Separate ODE from initial conditions
  let mut ode_expr = None;
  let mut initial_conditions: Vec<(usize, f64, f64)> = Vec::new(); // (deriv_order, x_val, y_val)

  for item in &items {
    if let Some((order, x_val, y_val)) =
      parse_numeric_initial_condition(item, &y_name)?
    {
      initial_conditions.push((order, x_val, y_val));
    } else {
      if ode_expr.is_some() {
        return Ok(Expr::FunctionCall {
          name: "NDSolve".to_string(),
          args: args.to_vec().into(),
        });
      }
      ode_expr = Some(item.clone());
    }
  }

  let ode = match ode_expr {
    Some(e) => e,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NDSolve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Parse the ODE and determine order
  let ode_normalized = normalize_equation(&ode)?;
  let terms = collect_ode_terms(&ode_normalized, &y_name, &x_name)?;
  let max_order = terms.iter().map(|t| t.order).max().unwrap_or(0) as usize;

  if max_order == 0 || initial_conditions.len() != max_order {
    return Ok(Expr::FunctionCall {
      name: "NDSolve".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Build the RHS function: solve for highest derivative
  // Sum of terms = 0, so y^(n) = -(sum of other terms) / coeff_of_y^(n)
  let rhs_expr =
    build_rhs_for_highest_derivative(&terms, max_order, &y_name, &x_name)?;

  // Sort ICs by order
  let mut ics_by_order: Vec<Option<(f64, f64)>> = vec![None; max_order];
  for (order, xv, yv) in &initial_conditions {
    if *order < max_order {
      ics_by_order[*order] = Some((*xv, *yv));
    }
  }

  // Verify all ICs present and at same x
  let x0 = match ics_by_order[0] {
    Some((xv, _)) => xv,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NDSolve".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut y0: Vec<f64> = Vec::new();
  for ic in &ics_by_order {
    match ic {
      Some((_, yv)) => y0.push(*yv),
      None => {
        return Ok(Expr::FunctionCall {
          name: "NDSolve".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  }

  // Perform RK4 integration
  let n_steps = 1000;
  let h = (x_max - x_min) / n_steps as f64;
  let mut data_points: Vec<(f64, f64)> = Vec::with_capacity(n_steps + 1);

  // state = [y, y', y'', ...] (values of y and its derivatives)
  let mut state = y0;
  let mut x = x_min;

  // We need to handle the case where x0 != x_min
  // For simplicity, we require x0 == x_min (or very close)
  if (x0 - x_min).abs() > 1e-10 {
    return Ok(Expr::FunctionCall {
      name: "NDSolve".to_string(),
      args: args.to_vec().into(),
    });
  }

  data_points.push((x, state[0]));

  for _ in 0..n_steps {
    let new_state =
      rk4_step(&rhs_expr, &state, x, h, max_order, &y_name, &x_name)?;
    state = new_state;
    x += h;
    data_points.push((x, state[0]));
  }

  // Build InterpolatingFunction
  let domain = Expr::List(
    vec![Expr::List(
      vec![Expr::Real(x_min), Expr::Real(x_max)].into(),
    )]
    .into(),
  );

  // Store data as a list of {x, y} pairs
  let data_expr = Expr::List(
    data_points
      .iter()
      .map(|(xv, yv)| Expr::List(vec![Expr::Real(*xv), Expr::Real(*yv)].into()))
      .collect(),
  );

  let interp_func = Expr::FunctionCall {
    name: "InterpolatingFunction".to_string(),
    args: vec![domain, data_expr].into(),
  };

  // Build result: {{y -> InterpolatingFunction[...]}} or {{y[x] -> InterpolatingFunction[...][x]}}
  let rule = if function_form {
    Expr::Rule {
      pattern: Box::new(Expr::Identifier(y_name)),
      replacement: Box::new(interp_func),
    }
  } else {
    Expr::Rule {
      pattern: Box::new(Expr::Identifier(y_name)),
      replacement: Box::new(interp_func),
    }
  };

  Ok(Expr::List(vec![Expr::List(vec![rule].into())].into()))
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
fn normalize_equation(eq: &Expr) -> Result<Expr, InterpreterError> {
  match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
      let lhs = &operands[0];
      let rhs = &operands[1];
      // lhs - rhs
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(lhs.clone()),
        right: Box::new(rhs.clone()),
      })
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
    && operators[0] == crate::syntax::ComparisonOp::Equal
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

/// Parse a numeric initial condition: y[x0] == y0 or y'[x0] == y0
/// Returns (derivative_order, x_val, y_val)
fn parse_numeric_initial_condition(
  expr: &Expr,
  y_name: &str,
) -> Result<Option<(usize, f64, f64)>, InterpreterError> {
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && operands.len() == 2
    && operators.len() == 1
    && operators[0] == crate::syntax::ComparisonOp::Equal
  {
    let lhs = &operands[0];

    // y[x0] == val
    if let Expr::FunctionCall { name, args } = lhs
      && name == y_name
      && args.len() == 1
    {
      // Try to evaluate both sides numerically
      let x_eval = crate::evaluator::evaluate_expr_to_expr(&args[0]).ok();
      let rhs_eval = crate::evaluator::evaluate_expr_to_expr(&operands[1]).ok();
      if let (Some(x_e), Some(rhs_e)) = (x_eval, rhs_eval)
        && let (Ok(x_val), Ok(rhs_val)) =
          (expr_to_f64(&x_e), expr_to_f64(&rhs_e))
      {
        return Ok(Some((0, x_val, rhs_val)));
      }
    }

    // Derivative[n, y, x0] == val or Derivative[n][y][x0] == val
    if let Some((order, val_expr)) =
      extract_derivative_order_and_point(lhs, y_name)
    {
      let x_eval = crate::evaluator::evaluate_expr_to_expr(&val_expr).ok();
      let rhs_eval = crate::evaluator::evaluate_expr_to_expr(&operands[1]).ok();
      if let (Some(x_e), Some(rhs_e)) = (x_eval, rhs_eval)
        && let (Ok(x_val), Ok(rhs_val)) =
          (expr_to_f64(&x_e), expr_to_f64(&rhs_e))
      {
        return Ok(Some((order, x_val, rhs_val)));
      }
    }
  }
  Ok(None)
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
      op: crate::syntax::UnaryOperator::Minus,
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
    "DSolve: cannot classify term involving {}",
    y_name
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
    // No y factor — forcing term
    order = -1;
    coefficient = if factors.len() == 1 {
      factors[0].clone()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors.to_vec().into(),
      }
    };
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
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(expr.clone()),
    },
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
          forcing = Some(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(existing.clone()),
            right: Box::new(evaluated),
          });
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
    );
    if let Some(part) = particular {
      return Ok(crate::functions::calculus_ast::simplify(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(homogeneous),
        right: Box::new(part),
      }));
    }
  }

  Ok(homogeneous)
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
        let r1 = (-b + disc.sqrt()) / (2.0 * a);
        let r2 = (-b - disc.sqrt()) / (2.0 * a);
        Ok(vec![(r1, 0.0, 1), (r2, 0.0, 1)])
      } else if disc.abs() <= 1e-10 {
        let r = -b / (2.0 * a);
        Ok(vec![(r, 0.0, 2)])
      } else {
        let real = -b / (2.0 * a);
        let imag = (-disc).sqrt() / (2.0 * a);
        Ok(vec![(real, imag, 1)])
      }
    }
    3 => solve_cubic_characteristic(coeffs),
    4 => solve_quartic_characteristic(coeffs),
    _ => {
      // For higher orders, try numerical root finding
      Err(InterpreterError::EvaluationError(format!(
        "DSolve: order {} constant-coefficient ODEs not supported",
        max_order
      )))
    }
  }
}

/// Solve cubic characteristic polynomial
fn solve_cubic_characteristic(
  coeffs: &[f64],
) -> Result<Vec<(f64, f64, usize)>, InterpreterError> {
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

  Ok(roots)
}

/// Solve quartic characteristic polynomial
fn solve_quartic_characteristic(
  coeffs: &[f64],
) -> Result<Vec<(f64, f64, usize)>, InterpreterError> {
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

  let cubic_roots = solve_cubic_characteristic(&resolvent_coeffs)?;
  // Pick a real root
  let m = cubic_roots
    .iter()
    .find(|(_, im, _)| im.abs() < 1e-10)
    .map(|(re, _, _)| *re)
    .unwrap_or(cubic_roots[0].0);

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
      roots.push(((sqrt_disc1 + s) / 2.0 + shift, 0.0, 1));
      roots.push(((sqrt_disc1 - s) / 2.0 + shift, 0.0, 1));
    } else {
      let s = (-disc2a).sqrt();
      roots.push((sqrt_disc1 / 2.0 + shift, s / 2.0, 1));
    }

    if disc2b >= -1e-10 {
      let s = disc2b.max(0.0).sqrt();
      roots.push(((-sqrt_disc1 + s) / 2.0 + shift, 0.0, 1));
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

  Ok(roots)
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
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(x.clone()),
              right: Box::new(Expr::Integer(k as i128)),
            }
          };
          term = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(x_power),
            right: Box::new(term),
          };
        }

        // Multiply by E^(r*x) if r != 0
        if real.abs() > 1e-10 {
          let exp_term = make_exp_term(*real, x_name);
          term = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(exp_term),
            right: Box::new(term),
          };
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

      let cos_term = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(c1),
        right: Box::new(make_trig_term("Cos", *imag, x_name)),
      };
      let sin_term = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(c2),
        right: Box::new(make_trig_term("Sin", *imag, x_name)),
      };

      let trig_sum = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(cos_term),
        right: Box::new(sin_term),
      };

      let term = if real.abs() > 1e-10 {
        let exp_term = make_exp_term(*real, x_name);
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(exp_term),
          right: Box::new(trig_sum),
        }
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
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(term),
    };
  }
  result
}

/// Create C[n] constant expression
fn make_c(n: usize) -> Expr {
  Expr::FunctionCall {
    name: "C".to_string(),
    args: vec![Expr::Integer(n as i128)].into(),
  }
}

/// Create E^(r*x) expression, simplifying for special values
fn make_exp_term(r: f64, x_name: &str) -> Expr {
  let x = Expr::Identifier(x_name.to_string());
  let r_expr = f64_to_nice_expr(r);
  let exponent = if matches!(&r_expr, Expr::Integer(1)) {
    x
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(r_expr),
      right: Box::new(x),
    }
  };
  Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Constant("E".to_string())),
    right: Box::new(exponent),
  }
}

/// Create Cos[β*x] or Sin[β*x] expression
fn make_trig_term(func: &str, beta: f64, x_name: &str) -> Expr {
  let x = Expr::Identifier(x_name.to_string());
  let beta_expr = f64_to_nice_expr(beta);
  let arg = if matches!(&beta_expr, Expr::Integer(1)) {
    x
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(beta_expr),
      right: Box::new(x),
    }
  };
  Expr::FunctionCall {
    name: func.to_string(),
    args: vec![arg].into(),
  }
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
      let n = numer.round() as i128;
      let d = denom as i128;
      let g = gcd(n.unsigned_abs(), d as u128) as i128;
      let nn = n / g;
      let dd = d / g;
      if dd == 1 {
        return Expr::Integer(nn);
      }
      return Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(nn)),
        right: Box::new(Expr::Integer(dd)),
      };
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
        return Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(make_sqrt(Expr::Integer(n))),
          right: Box::new(make_sqrt(Expr::Integer(d))),
        };
      }
    }
  }
  Expr::Real(f)
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
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
        a1 = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(a1),
          right: Box::new(term.coefficient.clone()),
        };
      }
      0 => {
        a0 = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(a0),
          right: Box::new(term.coefficient.clone()),
        };
      }
      -1 => {
        forcing = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(forcing),
          right: Box::new(term.coefficient.clone()),
        };
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
  let p_expr = crate::functions::calculus_ast::simplify(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(a0),
    right: Box::new(a1.clone()),
  });
  let q_expr = crate::functions::calculus_ast::simplify(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(negate_expr(&forcing)),
    right: Box::new(a1),
  });

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
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(integral),
      right: Box::new(make_c(1)),
    });
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
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(exp_term),
      right: Box::new(make_c(1)),
    });
  }

  // Case 3: General integrating factor method
  // μ(x) = E^(∫P(x)dx)
  // y = (1/μ) * (∫μ*Q(x)dx + C[1])
  let p_integral = crate::functions::calculus_ast::integrate_ast(&[
    p_expr.clone(),
    Expr::Identifier(x_name.to_string()),
  ])?;

  let mu = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Constant("E".to_string())),
    right: Box::new(p_integral.clone()),
  };

  let mu_q = crate::functions::calculus_ast::simplify(Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(mu.clone()),
    right: Box::new(q_expr),
  });

  let mu_q_integral = crate::functions::calculus_ast::integrate_ast(&[
    mu_q,
    Expr::Identifier(x_name.to_string()),
  ])?;

  // y = E^(-∫P dx) * (∫(μ*Q)dx + C[1])
  let neg_p_integral = negate_expr(&p_integral);
  let inv_mu = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Constant("E".to_string())),
    right: Box::new(neg_p_integral),
  };

  let inner = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(mu_q_integral),
    right: Box::new(make_c(1)),
  };

  Ok(Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(inv_mu),
    right: Box::new(inner),
  })
}

/// Create E^(expr*x) for symbolic expressions
fn make_exp_term_expr(coeff: &Expr, x_name: &str) -> Expr {
  let x = Expr::Identifier(x_name.to_string());
  let exponent = match coeff {
    Expr::Integer(1) => x,
    Expr::Integer(-1) => Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(x),
    },
    _ => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(coeff.clone()),
      right: Box::new(x),
    },
  };
  Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Constant("E".to_string())),
    right: Box::new(exponent),
  }
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
      return Some(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(f64_to_nice_expr(c)),
        right: Box::new(Expr::Identifier(x_name.to_string())),
      });
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
    (1..=max_order).map(|i| format!("__C{}", i)).collect();

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
      && operators[0] == crate::syntax::ComparisonOp::Equal
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
        operators: vec![crate::syntax::ComparisonOp::Equal],
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

/// Build the RHS expression for the highest derivative:
/// y^(n) = -(sum of lower-order terms + forcing) / leading_coeff
fn build_rhs_for_highest_derivative(
  terms: &[OdeTerm],
  max_order: usize,
  _y_name: &str,
  _x_name: &str,
) -> Result<Expr, InterpreterError> {
  let mut leading_coeff = None;
  let mut other_terms: Vec<OdeTerm> = Vec::new();

  for term in terms {
    if term.order == max_order as i32 {
      leading_coeff = Some(term.coefficient.clone());
    } else {
      other_terms.push(term.clone());
    }
  }

  let leading = match leading_coeff {
    Some(c) => c,
    None => {
      return Err(InterpreterError::EvaluationError(
        "NDSolve: no highest-order term found".into(),
      ));
    }
  };

  // Store coefficients for each order and forcing
  // We'll evaluate numerically during RK4 steps
  // Store as metadata in a special expression
  // Instead of building a symbolic expression, we'll store the term data
  // For NDSolve we need a representation we can evaluate numerically
  // Use a FunctionCall to wrap the data

  // Actually, let's build the symbolic RHS and evaluate it numerically in RK4
  // rhs = -(other terms evaluated with y and derivatives substituted) / leading

  // Build: -(a_{n-1}*y^(n-1) + ... + a_0*y + forcing) / a_n
  let mut numerator_terms: Vec<Expr> = Vec::new();
  for term in terms {
    if term.order == max_order as i32 {
      continue;
    }
    let coeff = term.coefficient.clone();
    if term.order >= 0 {
      // This will have y^(order) replaced by state variables during RK4
      let var_name = if term.order == 0 {
        "__y_0".to_string()
      } else {
        format!("__y_{}", term.order)
      };
      numerator_terms.push(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(coeff),
        right: Box::new(Expr::Identifier(var_name)),
      });
    } else {
      numerator_terms.push(coeff);
    }
  }

  if numerator_terms.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let mut sum = numerator_terms.remove(0);
  for t in numerator_terms {
    sum = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(sum),
      right: Box::new(t),
    };
  }

  // RHS = -sum / leading
  Ok(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(negate_expr(&sum)),
    right: Box::new(leading),
  })
}

/// Perform one step of RK4 for an ODE system
fn rk4_step(
  rhs_expr: &Expr,
  state: &[f64],
  x: f64,
  h: f64,
  max_order: usize,
  _y_name: &str,
  x_name: &str,
) -> Result<Vec<f64>, InterpreterError> {
  let n = max_order;

  // Evaluate the derivative of the system at a given state
  let eval_system = |s: &[f64],
                     xv: f64|
   -> Result<Vec<f64>, InterpreterError> {
    let mut derivs = vec![0.0; n];
    // d/dx y_0 = y_1, d/dx y_1 = y_2, ..., d/dx y_{n-2} = y_{n-1}
    for i in 0..n - 1 {
      derivs[i] = s[i + 1];
    }
    // d/dx y_{n-1} = rhs_expr evaluated with substitutions
    let mut expr = rhs_expr.clone();
    // Substitute x
    expr = crate::syntax::substitute_variable(&expr, x_name, &Expr::Real(xv));
    // Substitute __y_i
    for i in 0..n {
      let var_name = format!("__y_{}", i);
      expr =
        crate::syntax::substitute_variable(&expr, &var_name, &Expr::Real(s[i]));
    }
    let result =
      crate::evaluator::evaluate_expr_to_expr(&expr).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "NDSolve RK4: evaluation failed for expr '{}': {}",
          crate::syntax::expr_to_string(&expr),
          e
        ))
      })?;
    derivs[n - 1] = expr_to_f64(&result).map_err(|e| {
      InterpreterError::EvaluationError(format!(
        "NDSolve RK4: cannot convert result '{}' to f64: {}",
        crate::syntax::expr_to_string(&result),
        e
      ))
    })?;
    Ok(derivs)
  };

  // k1
  let k1 = eval_system(state, x)?;

  // k2
  let s2: Vec<f64> = state
    .iter()
    .zip(k1.iter())
    .map(|(s, k)| s + h / 2.0 * k)
    .collect();
  let k2 = eval_system(&s2, x + h / 2.0)?;

  // k3
  let s3: Vec<f64> = state
    .iter()
    .zip(k2.iter())
    .map(|(s, k)| s + h / 2.0 * k)
    .collect();
  let k3 = eval_system(&s3, x + h / 2.0)?;

  // k4
  let s4: Vec<f64> = state
    .iter()
    .zip(k3.iter())
    .map(|(s, k)| s + h * k)
    .collect();
  let k4 = eval_system(&s4, x + h)?;

  // Update state
  let new_state: Vec<f64> = (0..n)
    .map(|i| state[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    .collect();

  Ok(new_state)
}

/// Convert Expr to f64
fn expr_to_f64(expr: &Expr) -> Result<f64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as f64),
    Expr::Real(f) => Ok(*f),
    Expr::Constant(name) if name == "E" => Ok(std::f64::consts::E),
    Expr::Constant(name) if name == "Pi" => Ok(std::f64::consts::PI),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
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
          "NDSolve: cannot convert expression to numeric value".into(),
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
      "NDSolve: cannot convert {} to numeric value",
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

// ─── Interpolation ─────────────────────────────────────────────────────

/// Interpolation[{y1, y2, ...}] or Interpolation[{{x1,y1}, {x2,y2}, ...}]
/// Returns InterpolatingFunction[domain, data]
pub fn interpolation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Interpolation".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Extract InterpolationOrder option (default 3)
  let mut interp_order: i128 = 3;
  let data_arg = &args[0];

  for opt in args.iter().skip(1) {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "InterpolationOrder"
          && let Some(n) = crate::functions::math_ast::expr_to_i128(replacement)
        {
          interp_order = n;
        }
      }
      Expr::FunctionCall {
        name,
        args: rule_args,
      } if name == "Rule" && rule_args.len() == 2 => {
        if let Expr::Identifier(opt_name) = &rule_args[0]
          && opt_name == "InterpolationOrder"
          && let Some(n) =
            crate::functions::math_ast::expr_to_i128(&rule_args[1])
        {
          interp_order = n;
        }
      }
      _ => {}
    }
  }

  // Evaluate the data argument
  let data_evaluated = crate::evaluator::evaluate_expr_to_expr(data_arg)?;

  let data_list = match &data_evaluated {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Interpolation".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if data_list.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Interpolation: need at least one data point".into(),
    ));
  }

  // Determine format: list of values or list of {x, y} pairs
  let mut points: Vec<(f64, f64)> = Vec::new();

  let first = &data_list[0];
  let is_pair_format = matches!(first, Expr::List(items) if items.len() == 2);

  if is_pair_format {
    // {{x1, y1}, {x2, y2}, ...}
    for item in data_list {
      let (x, y) = extract_point(item)?;
      points.push((x, y));
    }
  } else {
    // {y1, y2, ...} — x values are 1, 2, 3, ...
    for (i, item) in data_list.iter().enumerate() {
      let y = expr_to_f64(
        &crate::evaluator::evaluate_expr_to_expr(item).unwrap_or(item.clone()),
      )?;
      points.push(((i + 1) as f64, y));
    }
  }

  // Sort by x value
  points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

  let n = points.len();
  if n < 2 {
    return Err(InterpreterError::EvaluationError(
      "Interpolation: need at least 2 data points".into(),
    ));
  }

  // Clamp order to valid range
  let mut order = interp_order.max(1).min(3) as usize;
  if order >= n {
    let reduced = n - 1;
    crate::emit_message(&format!(
      "Interpolation::inhr: Requested order is too high; order has been reduced to {{{}}}.",
      reduced
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

  // Store data as list of {x, y} pairs, preserving original y-value types
  // (e.g. Integer for integer inputs) so that evaluation at exact grid points
  // returns the original type, matching Wolfram behavior.
  let data_expr = Expr::List(
    points
      .iter()
      .enumerate()
      .map(|(i, (x, _y))| {
        // Use original y-expression when available (non-pair format)
        let y_expr = if !is_pair_format {
          let orig = &data_list[i];
          let evaluated = crate::evaluator::evaluate_expr_to_expr(orig)
            .unwrap_or(orig.clone());
          match &evaluated {
            Expr::Integer(_) | Expr::Real(_) => evaluated,
            _ => Expr::Real(*_y),
          }
        } else {
          // For pair format, extract y from original {x, y} pair
          if let Expr::List(pair) = &data_list[i]
            && pair.len() == 2
          {
            let y_eval = crate::evaluator::evaluate_expr_to_expr(&pair[1])
              .unwrap_or(pair[1].clone());
            match &y_eval {
              Expr::Integer(_) | Expr::Real(_) => y_eval,
              _ => Expr::Real(*_y),
            }
          } else {
            Expr::Real(*_y)
          }
        };
        Expr::List(vec![Expr::Real(*x), y_expr].into())
      })
      .collect(),
  );

  // Store the interpolation order as a third argument
  let interp_func = Expr::FunctionCall {
    name: "InterpolatingFunction".to_string(),
    args: vec![domain, data_expr, Expr::Integer(order as i128)].into(),
  };

  Ok(interp_func)
}

// ─── InterpolatingFunction evaluation ──────────────────────────────────

/// InterpolatingFunction returns machine-precision reals for interpolated values.
fn real_or_integer(v: f64) -> Expr {
  Expr::Real(v)
}

/// Convert a whole-number `Real` to an `Integer` (the grid coordinates of an
/// implicit-grid interpolation are integers in wolframscript even though they
/// are stored as reals); other values pass through unchanged.
fn whole_real_to_int(e: &Expr) -> Expr {
  match e {
    Expr::Real(f) if f.fract() == 0.0 && f.abs() < 9e15 => {
      Expr::Integer(*f as i128)
    }
    other => other.clone(),
  }
}

/// Answer an `InterpolatingFunction[…]["property"]` query from the stored
/// `{x, y}` grid data. Returns `None` for unrecognized properties so the call
/// stays unevaluated.
fn interpolating_function_property(
  data: &Expr,
  order: usize,
  prop: &str,
) -> Option<Expr> {
  let Expr::List(pairs) = data else {
    return None;
  };
  let mut xs: Vec<Expr> = Vec::with_capacity(pairs.len());
  let mut ys: Vec<Expr> = Vec::with_capacity(pairs.len());
  for p in pairs.iter() {
    if let Expr::List(pair) = p
      && pair.len() == 2
    {
      xs.push(whole_real_to_int(&pair[0]));
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
    "DerivativeOrder" => Some(Expr::Integer(0)),
    _ => None,
  }
}

/// Evaluate InterpolatingFunction[domain, data][x_val]
/// or InterpolatingFunction[domain, data, order][x_val]
pub fn evaluate_interpolating_function(
  func_args: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if (func_args.len() != 2 && func_args.len() != 3) || call_args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InterpolatingFunction expects domain and data, called with one argument"
        .into(),
    ));
  }

  let data = &func_args[1];
  let order = if func_args.len() == 3 {
    match &func_args[2] {
      Expr::Integer(n) => *n as usize,
      _ => 3,
    }
  } else {
    1 // Default for NDSolve-generated (backwards compat)
  };

  // Property access: InterpolatingFunction[…]["Domain"], ["Grid"], etc.
  if let Expr::String(prop) = &call_args[0]
    && let Some(result) = interpolating_function_property(data, order, prop)
  {
    return Ok(result);
  }

  let x_val_expr = crate::evaluator::evaluate_expr_to_expr(&call_args[0])?;
  let x_val = match &x_val_expr {
    Expr::Integer(n) => *n as f64,
    Expr::Real(f) => *f,
    _ => {
      // Can't evaluate symbolically — return unevaluated
      return Ok(Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "InterpolatingFunction".to_string(),
          args: func_args.to_vec().into(),
        }),
        args: call_args.to_vec(),
      });
    }
  };

  // Data is a list of {x, y} pairs
  let data_points = match data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "InterpolatingFunction: invalid data format".into(),
      ));
    }
  };

  let n = data_points.len();
  if n < 2 {
    return Err(InterpreterError::EvaluationError(
      "InterpolatingFunction: not enough data points".into(),
    ));
  }

  // Extract all points for interpolation
  let (x_first, _) = extract_point(&data_points[0])?;
  let (x_last, _) = extract_point(&data_points[n - 1])?;

  // Clamp to domain
  let x_clamped = x_val.max(x_first).min(x_last);

  // Check for exact grid point match — return the stored y-value directly
  // to preserve original types (e.g. Integer for ListInterpolation with integer data).
  for pt in data_points {
    if let Expr::List(pair) = pt
      && pair.len() == 2
      && let Some(xp) = match &pair[0] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => None,
      }
      && (xp - x_clamped).abs() < 1e-15
    {
      return Ok(pair[1].clone());
    }
  }

  // Binary search for the interval containing x_clamped
  let idx = find_interval(data_points, x_clamped, n)?;

  if order == 1 || n <= 2 {
    // Linear interpolation
    let (x0, y0) = extract_point(&data_points[idx])?;
    let (x1, y1) = extract_point(&data_points[idx + 1])?;
    let t = if (x1 - x0).abs() > 1e-15 {
      (x_clamped - x0) / (x1 - x0)
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
    let eff_order = order.min(n - 1);
    let y_val =
      lagrange_interpolate(data_points, x_clamped, n, idx, eff_order)?;
    Ok(real_or_integer(y_val))
  }
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
    let mid = (lo + hi) / 2;
    let (x_mid, _) = extract_point(&data_points[mid])?;
    if x_val < x_mid {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  Ok(lo)
}

/// Lagrange polynomial interpolation using (order+1) nearest points.
fn lagrange_interpolate(
  data_points: &[Expr],
  x_val: f64,
  n: usize,
  idx: usize,
  order: usize,
) -> Result<f64, InterpreterError> {
  // Select (order+1) points centered around idx
  let needed = order + 1;
  let start = if idx + 1 >= needed {
    (idx + 1)
      .saturating_sub(needed)
      .min(n.saturating_sub(needed))
  } else {
    0
  };
  let end = (start + needed).min(n);

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
    let x = expr_to_f64(&items[0])?;
    let y = expr_to_f64(&items[1])?;
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
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(c_eff)),
        right: Box::new(n_var(xn)),
      }
    };
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Constant("E".to_string())),
      right: Box::new(exponent),
    }
  };
  // Argument to C[1]: y - b*x  (or just y when b == 0).
  let c1_arg = if b == 0 {
    n_var(yn)
  } else {
    let bx = if b == 1 {
      n_var(xn)
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-b)),
        right: Box::new(n_var(xn)),
      }
    };
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(bx),
      right: Box::new(n_var(yn)),
    }
  };
  let c1_applied = Expr::CurriedCall {
    func: Box::new(Expr::FunctionCall {
      name: "C".to_string(),
      args: vec![Expr::Integer(1)].into(),
    }),
    args: vec![c1_arg],
  };
  let body = if matches!(&exp_part, Expr::Integer(1)) {
    c1_applied
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(exp_part),
      right: Box::new(c1_applied),
    }
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
      op: crate::syntax::UnaryOperator::Minus,
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
      Expr::Integer(n) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(*n)),
        right: Box::new(n_var(xn)),
      },
      other => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(other.clone()),
        right: Box::new(n_var(xn)),
      },
    };
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(n_var(yn)),
      right: Box::new(bx),
    }
  };
  let c1_applied = Expr::CurriedCall {
    func: Box::new(Expr::FunctionCall {
      name: "C".to_string(),
      args: vec![Expr::Integer(1)].into(),
    }),
    args: vec![c1_arg],
  };
  // Inhomogeneous head term: (c/a)*x.
  if c == 0 {
    return Some(c1_applied);
  }
  let coeff = make_c_over_a(c, a);
  let head_term = match &coeff {
    Expr::Integer(1) => n_var(xn),
    Expr::Integer(n) => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(*n)),
      right: Box::new(n_var(xn)),
    },
    other => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(other.clone()),
      right: Box::new(n_var(xn)),
    },
  };
  Some(Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(head_term),
    right: Box::new(c1_applied),
  })
}

/// `-b/a` reduced to either an `Integer` (if `a` divides `b`) or a
/// `Rational` literal. Sign is folded into the numerator.
fn make_neg_b_over_a(b: i128, a: i128) -> Expr {
  use crate::functions::math_ast::make_rational_pub;
  let g = gcd_i128(b.abs(), a.abs());
  let g = if g == 0 { 1 } else { g };
  let (num, den) = (-b / g, a / g);
  let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
  if den == 1 {
    Expr::Integer(num)
  } else {
    make_rational_pub(num, den)
  }
}

fn make_c_over_a(c: i128, a: i128) -> Expr {
  use crate::functions::math_ast::make_rational_pub;
  let g = gcd_i128(c.abs(), a.abs());
  let g = if g == 0 { 1 } else { g };
  let (num, den) = (c / g, a / g);
  let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
  if den == 1 {
    Expr::Integer(num)
  } else {
    make_rational_pub(num, den)
  }
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a, b);
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
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
  let log_x = Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![n_var(xn)].into(),
  };
  let log_term = if c == 1 {
    log_x
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(c)),
      right: Box::new(log_x),
    }
  };
  let y_over_x = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(n_var(yn)),
    right: Box::new(n_var(xn)),
  };
  let c1_applied = Expr::CurriedCall {
    func: Box::new(Expr::FunctionCall {
      name: "C".to_string(),
      args: vec![Expr::Integer(1)].into(),
    }),
    args: vec![y_over_x],
  };
  let body = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(log_term),
    right: Box::new(c1_applied),
  };
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
      pattern: Box::new(Expr::FunctionCall {
        name: fname.to_string(),
        args: vec![n_var(xn), n_var(yn)].into(),
      }),
      replacement: Box::new(body),
    }
  } else {
    Expr::Rule {
      pattern: Box::new(Expr::Identifier(fname.to_string())),
      replacement: Box::new(Expr::FunctionCall {
        name: "Function".to_string(),
        args: vec![Expr::List(vec![n_var(xn), n_var(yn)].into()), body].into(),
      }),
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
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
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
