//! The step-wise `NDSolve` framework: `NDSolve`ProcessEquations` sets a
//! problem up as an `NDSolve`StateData` object, `NDSolve`Reinitialize` gives
//! a copy with new initial conditions, `NDSolve`Iterate` advances it to a
//! later (or earlier) time and `NDSolve`ProcessSolutions` reads the solution
//! computed so far.
//!
//! As in the Wolfram Language, a state is a mutable object: `Iterate`
//! changes the state it is given in place, so every variable holding the
//! same `NDSolve`StateData[…]` sees the advanced solution. The objects live
//! in a per-thread registry and the expression only carries their index.

use super::*;
use std::cell::RefCell;

pub(crate) const STATE_DATA: &str = "NDSolve`StateData";

#[derive(Clone)]
struct StateRecord {
  /// The differential equations (those mentioning the independent variable).
  equations: Vec<Expr>,
  /// The initial conditions.
  conditions: Vec<Expr>,
  /// The unknown functions, as given.
  vars: Expr,
  /// The independent variable.
  var: Expr,
  /// The trailing `NDSolve` options.
  options: Vec<Expr>,
  /// The time the initial conditions are given at.
  start: Expr,
  /// The solution rules computed by the last `Iterate`.
  solution: Option<Expr>,
}

thread_local! {
  static STATES: RefCell<Vec<StateRecord>> = const { RefCell::new(Vec::new()) };
}

fn register(record: StateRecord) -> Expr {
  let index = STATES.with(|s| {
    let mut states = s.borrow_mut();
    states.push(record);
    states.len()
  });
  Expr::List(vec![call(STATE_DATA, vec![Expr::Integer(index as i128)])].into())
}

/// The registry index of an `NDSolve`StateData[i]` expression.
fn state_index(expr: &Expr) -> Option<usize> {
  match expr {
    Expr::FunctionCall { name, args } if name == STATE_DATA => {
      match args.as_slice() {
        [Expr::Integer(i)] if *i >= 1 => {
          let i = *i as usize;
          STATES.with(|s| (i <= s.borrow().len()).then_some(i - 1))
        }
        _ => None,
      }
    }
    _ => None,
  }
}

fn state(index: usize) -> StateRecord {
  STATES.with(|s| s.borrow()[index].clone())
}

fn as_items(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  }
}

fn mentions(expr: &Expr, name: &str) -> bool {
  if let Expr::Identifier(s) = expr {
    s == name
  } else {
    let mut found = false;
    for child in crate::syntax::expr_children(expr) {
      if mentions(child, name) {
        found = true;
        break;
      }
    }
    found
  }
}

/// The point an initial condition is given at: the argument of the first
/// `u[t0]` or `Derivative[n][u][t0]` it contains.
fn condition_point(expr: &Expr, unknowns: &[String]) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args }
      if args.len() == 1 && unknowns.contains(name) =>
    {
      Some(args[0].clone())
    }
    Expr::CurriedCall { func, args } if args.len() == 1 => {
      if let Expr::CurriedCall { func: inner, .. } = func.as_ref()
        && matches!(inner.as_ref(), Expr::FunctionCall { name, .. } if name == "Derivative")
      {
        return Some(args[0].clone());
      }
      crate::syntax::expr_children(expr)
        .into_iter()
        .find_map(|c| condition_point(c, unknowns))
    }
    _ => crate::syntax::expr_children(expr)
      .into_iter()
      .find_map(|c| condition_point(c, unknowns)),
  }
}

fn unknown_names(vars: &Expr) -> Vec<String> {
  as_items(vars)
    .iter()
    .filter_map(|v| match v {
      Expr::Identifier(s) => Some(s.clone()),
      Expr::FunctionCall { name, .. } => Some(name.clone()),
      _ => None,
    })
    .collect()
}

fn split_equations(eqns: &[Expr], var: &str) -> (Vec<Expr>, Vec<Expr>) {
  eqns.iter().cloned().partition(|e| mentions(e, var))
}

/// `NDSolve`ProcessEquations[eqns, vars, t, opts…]`.
pub fn process_equations_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let head = "NDSolve`ProcessEquations";
  if args.len() < 3 {
    return Ok(unevaluated(head, args));
  }
  let var = match &args[2] {
    Expr::List(items) if !items.is_empty() => items[0].clone(),
    other => other.clone(),
  };
  let Expr::Identifier(var_name) = &var else {
    return Ok(unevaluated(head, args));
  };
  let eqns = as_items(&args[0]);
  let (equations, conditions) = split_equations(&eqns, var_name);
  let unknowns = unknown_names(&args[1]);
  let Some(start) = conditions
    .iter()
    .find_map(|c| condition_point(c, &unknowns))
  else {
    return Ok(unevaluated(head, args));
  };
  Ok(register(StateRecord {
    equations,
    conditions,
    vars: args[1].clone(),
    var,
    options: args[3..].to_vec(),
    start,
    solution: None,
  }))
}

/// `NDSolve`Reinitialize[state, conditions]`: a new state for the same
/// equations, started from the given initial conditions.
pub fn reinitialize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let head = "NDSolve`Reinitialize";
  let [state_expr, conds] = args else {
    return Ok(unevaluated(head, args));
  };
  let Some(index) = state_index(state_expr) else {
    return Ok(unevaluated(head, args));
  };
  let old = state(index);
  let conditions = as_items(conds);
  let Some(start) = conditions
    .iter()
    .find_map(|c| condition_point(c, &unknown_names(&old.vars)))
  else {
    return Ok(unevaluated(head, args));
  };
  Ok(register(StateRecord {
    conditions,
    start,
    solution: None,
    ..old
  }))
}

/// `NDSolve`Iterate[state, t1]`: integrate the state's problem from its
/// initial point to `t1`, keeping the solution in the state.
pub fn iterate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let head = "NDSolve`Iterate";
  let [state_expr, end] = args else {
    return Ok(unevaluated(head, args));
  };
  let Some(index) = state_index(state_expr) else {
    return Ok(unevaluated(head, args));
  };
  let record = state(index);
  let mut eqns = record.equations.clone();
  eqns.extend(record.conditions.iter().cloned());
  let mut solve_args = vec![
    Expr::List(eqns.into()),
    record.vars.clone(),
    Expr::List(
      vec![record.var.clone(), record.start.clone(), end.clone()].into(),
    ),
  ];
  solve_args.extend(record.options.iter().cloned());
  let solved = crate::functions::ode_ast::ndsolve_ast(&solve_args)?;
  if let Expr::List(branches) = &solved
    && let Some(rules @ Expr::List(_)) = branches.first()
  {
    let rules = rules.clone();
    STATES.with(|s| s.borrow_mut()[index].solution = Some(rules));
  }
  Ok(Expr::Identifier("Null".to_string()))
}

/// `NDSolve`ProcessSolutions[state]`: the solution rules computed so far.
pub fn process_solutions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let head = "NDSolve`ProcessSolutions";
  if let [state_expr] = args
    && let Some(index) = state_index(state_expr)
    && let Some(solution) = state(index).solution
  {
    return Ok(solution);
  }
  Ok(unevaluated(head, args))
}
