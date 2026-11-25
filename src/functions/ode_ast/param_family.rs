//! `NDSolve[eqns, u, {t, …}, {p, …}]` whose equations differentiate the
//! unknowns in `t` only: the second variable `p` is a parameter, an ODE in
//! `t` for each of its values. Unlike a family that never mentions `p`
//! ([`solve_parameter_ode_group`]), the parameter may enter the equations
//! and the initial conditions (`u[0, p] == p`, a coefficient `Sin[p]`).
//!
//! The parameter is sampled on a uniform grid — as the method of lines
//! would sample a space variable — sized so the initial profiles are
//! resolved and honoring `"MinPoints"`/`"MaxPoints"`. The ODEs of all grid
//! values are integrated as one system, so they share their time steps, and
//! the solution is an `InterpolatingFunction` over both variables.

use super::pde_mol::{MolOptions, choose_points};
use super::*;

/// The call `u[a, b]` or `Derivative[i, j][u][a, b]` of one of `names`:
/// `(name, [i, j], [a, b])`.
fn two_argument_call<'a>(
  expr: &'a Expr,
  names: &[String],
) -> Option<(&'a String, [i128; 2], [&'a Expr; 2])> {
  match expr {
    Expr::FunctionCall { name, args }
      if names.contains(name) && args.len() == 2 =>
    {
      Some((name, [0, 0], [&args[0], &args[1]]))
    }
    Expr::CurriedCall { func, args } if args.len() == 2 => {
      let Expr::CurriedCall {
        func: deriv_head,
        args: fname_args,
      } = func.as_ref()
      else {
        return None;
      };
      let [Expr::Identifier(fname)] = fname_args.as_slice() else {
        return None;
      };
      let Expr::FunctionCall {
        name: deriv_name,
        args: orders,
      } = deriv_head.as_ref()
      else {
        return None;
      };
      if deriv_name != "Derivative"
        || orders.len() != 2
        || !names.contains(fname)
      {
        return None;
      }
      let (Expr::Integer(i), Expr::Integer(j)) = (&orders[0], &orders[1])
      else {
        return None;
      };
      Some((fname, [*i, *j], [&args[0], &args[1]]))
    }
    _ => None,
  }
}

/// What rewriting the equations for one role assignment found.
struct Shape {
  /// No unknown is used in a way an ODE family cannot express.
  ok: Cell<bool>,
  /// The argument position of the parameter in the unknowns' calls.
  param_pos: Cell<Option<usize>>,
  /// Some unknown is differentiated in the evolution variable.
  differentiated: Cell<bool>,
}

/// Rewrite `u[t, p]` and `Derivative[k, 0][u][t, p]` (the parameter in
/// either argument position, the same one throughout) into `u[t]` and
/// `Derivative[k][u][t]`. Other occurrences of `p` stay; any other use of
/// an unknown clears `shape.ok`.
fn drop_parameter(
  expr: &Expr,
  names: &[String],
  param: &str,
  shape: &Shape,
) -> Expr {
  if let Some((name, orders, call_args)) = two_argument_call(expr, names) {
    let is_param = |e: &Expr| matches!(e, Expr::Identifier(n) if n == param);
    let pos = match (is_param(call_args[0]), is_param(call_args[1])) {
      (false, true) => 1,
      (true, false) => 0,
      _ => {
        shape.ok.set(false);
        return expr.clone();
      }
    };
    let other = call_args[1 - pos];
    if orders[pos] != 0
      || orders[1 - pos] < 0
      || expr_contains_ident(other, param)
      || shape.param_pos.get().is_some_and(|p| p != pos)
    {
      shape.ok.set(false);
      return expr.clone();
    }
    shape.param_pos.set(Some(pos));
    let k = orders[1 - pos];
    if k == 0 {
      return Expr::FunctionCall {
        name: name.clone(),
        args: vec![other.clone()].into(),
      };
    }
    shape.differentiated.set(true);
    return Expr::CurriedCall {
      func: Box::new(Expr::CurriedCall {
        func: Box::new(call("Derivative", vec![Expr::Integer(k)])),
        args: vec![Expr::Identifier(name.clone())],
      }),
      args: vec![other.clone()],
    };
  }
  match expr {
    Expr::Identifier(n) if names.contains(n) => {
      shape.ok.set(false);
      expr.clone()
    }
    Expr::FunctionCall { name, .. } if names.contains(name) => {
      shape.ok.set(false);
      expr.clone()
    }
    _ => map_children(expr, &|c| drop_parameter(c, names, param, shape)),
  }
}

/// The member of the family at one parameter value: the parameter replaced
/// by `value` and every unknown `u` by its own symbol `rename(u)`.
fn family_member(
  expr: &Expr,
  names: &[String],
  param: &str,
  value: f64,
  rename: &dyn Fn(&str) -> String,
) -> Expr {
  match expr {
    Expr::Identifier(n) if n == param => Expr::Real(value),
    Expr::Identifier(n) if names.contains(n) => Expr::Identifier(rename(n)),
    Expr::FunctionCall { name, args } if names.contains(name) => {
      Expr::FunctionCall {
        name: rename(name),
        args: args
          .iter()
          .map(|a| family_member(a, names, param, value, rename))
          .collect(),
      }
    }
    _ => map_children(expr, &|c| family_member(c, names, param, value, rename)),
  }
}

/// The initial profile an equation `u[t0] == g` or `u'[t0] == g` states
/// (either way around), as an expression in the parameter: the side
/// without an unknown.
fn initial_profile(eq: &Expr, names: &[String], t_name: &str) -> Option<Expr> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  if expr_contains_ident(eq, t_name) {
    return None;
  }
  let mentions = |e: &Expr| {
    names
      .iter()
      .any(|n| mentions_head(e, n) || expr_contains_ident(e, n))
  };
  match (mentions(lhs), mentions(rhs)) {
    (true, false) => Some(rhs.clone()),
    (false, true) => Some(lhs.clone()),
    _ => None,
  }
}

/// Solve the family; `args` are `NDSolve`'s four positional arguments and
/// `opts` its trailing option rules. `Ok(None)` when the equations do not
/// have this shape or the stacked ODE system is not solved.
pub(super) fn ndsolve_parameter_family(
  args: &[Expr],
  opts: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let names: Vec<String> = match &args[1] {
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
  if names.is_empty() {
    return Ok(None);
  }
  let (Some(dom_a), Some(dom_b)) =
    (parse_pde_domain(&args[2]), parse_pde_domain(&args[3]))
  else {
    return Ok(None);
  };
  let eq_items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  let eq_items = flatten_chained_equalities(&eq_items);
  let mol_options = MolOptions::from_rules(opts);

  for (t_dom, p_dom, t_arg) in
    [(&dom_a, &dom_b, &args[2]), (&dom_b, &dom_a, &args[3])]
  {
    let shape = Shape {
      ok: Cell::new(true),
      param_pos: Cell::new(None),
      differentiated: Cell::new(false),
    };
    let ode_eqs: Vec<Expr> = eq_items
      .iter()
      .map(|e| drop_parameter(e, &names, &p_dom.name, &shape))
      .collect();
    let Some(param_pos) = shape.param_pos.get() else {
      continue;
    };
    if !shape.ok.get() || !shape.differentiated.get() {
      continue;
    }

    // The parameter grid.
    let profiles: Vec<NumFn> = ode_eqs
      .iter()
      .filter_map(|e| initial_profile(e, &names, &t_dom.name))
      .map(|e| NumFn::new(e, std::slice::from_ref(&p_dom.name)))
      .collect();
    let n_pts =
      choose_points(&profiles, p_dom.min, p_dom.max, false, &mol_options);
    let step = (p_dom.max - p_dom.min) / (n_pts - 1) as f64;
    let ps: Vec<f64> = (0..n_pts)
      .map(|g| {
        if g + 1 == n_pts {
          p_dom.max
        } else {
          p_dom.min + g as f64 * step
        }
      })
      .collect();

    // One ODE system holding every grid value's copy of the equations.
    let member_name = |name: &str, g: usize| format!("NDSolve$P{g}${name}");
    let mut stacked_eqs = Vec::with_capacity(ode_eqs.len() * n_pts);
    let mut stacked_names = Vec::with_capacity(names.len() * n_pts);
    for (g, &p) in ps.iter().enumerate() {
      let rename = |name: &str| member_name(name, g);
      for eq in &ode_eqs {
        stacked_eqs.push(family_member(eq, &names, &p_dom.name, p, &rename));
      }
      for name in &names {
        stacked_names.push(Expr::Identifier(rename(name)));
      }
    }
    let mut ode_args = vec![
      Expr::List(stacked_eqs.into()),
      Expr::List(stacked_names.into()),
      t_arg.clone(),
    ];
    ode_args.extend(opts.iter().cloned());
    let solved = ndsolve_ast_inner(&ode_args)?;
    let Some(rules) = pde_result_rules(&solved) else {
      return Ok(None);
    };

    // The samples `[unknown][parameter][time]` and their common times.
    let mut ts: Vec<f64> = Vec::new();
    let mut samples: Vec<Vec<Vec<f64>>> = vec![Vec::new(); names.len()];
    for g in 0..n_pts {
      for (u, name) in names.iter().enumerate() {
        let key = member_name(name, g);
        let Some((_, interp)) = rules.iter().find(|(n, _)| *n == key) else {
          return Ok(None);
        };
        let Expr::FunctionCall {
          name: head,
          args: if_args,
        } = interp
        else {
          return Ok(None);
        };
        if head != "InterpolatingFunction" || if_args.len() < 2 {
          return Ok(None);
        }
        let Expr::List(points) = &if_args[1] else {
          return Ok(None);
        };
        let mut times = Vec::with_capacity(points.len());
        let mut values = Vec::with_capacity(points.len());
        for point in points {
          let Expr::List(pair) = point else {
            return Ok(None);
          };
          if pair.len() != 2 {
            return Ok(None);
          }
          times.push(expr_to_f64(&pair[0])?);
          values.push(expr_to_f64(&pair[1])?);
        }
        if ts.is_empty() {
          ts = times;
        } else if ts != times {
          return Ok(None);
        }
        samples[u].push(values);
      }
    }
    if ts.len() < 2 {
      return Ok(None);
    }

    let reals =
      |v: &[f64]| Expr::List(v.iter().map(|x| Expr::Real(*x)).collect());
    let range = |lo: f64, hi: f64| {
      Expr::List(vec![Expr::Real(lo), Expr::Real(hi)].into())
    };
    let t_range = range(ts[0], ts[ts.len() - 1]);
    let p_range = range(p_dom.min, p_dom.max);
    let t_first = param_pos == 1;
    let (domain, coords) = if t_first {
      (vec![t_range, p_range], vec![reals(&ts), reals(&ps)])
    } else {
      (vec![p_range, t_range], vec![reals(&ps), reals(&ts)])
    };
    let orders = Expr::List(vec![Expr::Integer(3), Expr::Integer(3)].into());
    let mut out = Vec::with_capacity(names.len());
    for (name, by_param) in names.iter().zip(&samples) {
      let grid = if t_first {
        transpose_grid(by_param)
      } else {
        by_param.clone()
      };
      out.push(rule_expr(
        Expr::Identifier(name.clone()),
        call(
          "InterpolatingFunction",
          vec![
            Expr::List(domain.clone().into()),
            Expr::List(grid.iter().map(|row| reals(row)).collect()),
            orders.clone(),
            Expr::List(coords.clone().into()),
          ],
        ),
      ));
    }
    return Ok(Some(Expr::List(vec![Expr::List(out.into())].into())));
  }
  Ok(None)
}
