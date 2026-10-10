//! `RulePlot[CellularAutomaton[spec]]` — the picture of a cellular
//! automaton's rule: one small diagram per neighborhood, the cells of the
//! neighborhood in a row with the cell they produce centred underneath.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::helpers::{call, call1};
use crate::syntax::Expr;

/// Largest number of neighborhoods drawn; beyond this the picture would be
/// unreadably wide (and costly to compute), so the call stays a placeholder.
const MAX_CASES: usize = 4096;

/// `(rule, colors, radius)` of a numeric `CellularAutomaton[spec]` rule.
/// `spec` is `n`, `{n, k, r}`, or `{n, {k, …}, r}`; `k` defaults to 2 and
/// `r` to 1.
fn parse_rule_spec(spec: &Expr) -> Option<(Expr, usize, usize)> {
  let small = |e: &Expr| match e {
    Expr::Integer(n) if (1..=64).contains(n) => Some(*n as usize),
    _ => None,
  };
  match spec {
    Expr::Integer(_) => Some((spec.clone(), 2, 1)),
    Expr::List(items) if (1..=3).contains(&items.len()) => {
      let k = match items.get(1) {
        None => 2,
        Some(Expr::List(inner)) => small(inner.first()?)?,
        Some(e) => small(e)?,
      };
      let r = match items.get(2) {
        None => 1,
        Some(Expr::List(inner)) if inner.len() == 1 => small(&inner[0])?,
        Some(e) => small(e)?,
      };
      matches!(items[0], Expr::Integer(_)).then(|| (items[0].clone(), k, r))
    }
    _ => None,
  }
}

/// The graphics for a `RulePlot` of a cellular automaton, or `None` when
/// the first argument is not a numeric `CellularAutomaton[spec]` rule.
pub fn rule_plot_graphics(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::FunctionCall { name, args: ca } = &args[0] else {
    return None;
  };
  if name != "CellularAutomaton" || ca.len() != 1 {
    return None;
  }
  let (_, k, r) = parse_rule_spec(&ca[0])?;
  let width = 2 * r + 1;
  let cases = k.checked_pow(width as u32).filter(|c| *c <= MAX_CASES)?;

  let mut color_rules: Option<Expr> = None;
  let mut graphics_opts: Vec<Expr> = Vec::new();
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(n) = pattern.as_ref()
    {
      match n.as_str() {
        "ColorRules" => {
          color_rules = evaluate_expr_to_expr(replacement).ok();
        }
        "ImageSize" => graphics_opts.push(opt.clone()),
        _ => {}
      }
    }
  }

  let color_of = |v: usize| -> Expr {
    if let Some(rules) = &color_rules
      && let Ok(c) = crate::evaluator::pattern_matching::apply_replace_all_ast(
        &Expr::Integer(v as i128),
        rules,
      )
      && crate::functions::graphics::parse_color(&c).is_some()
    {
      return c;
    }
    let level = 1.0 - v as f64 / (k - 1).max(1) as f64;
    call1("GrayLevel", Expr::Real(level))
  };
  let pt =
    |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  let cell = |x: f64, y: f64, v: usize| -> Vec<Expr> {
    vec![
      color_of(v),
      call("Rectangle", vec![pt(x, y), pt(x + 1.0, y + 1.0)]),
    ]
  };

  let mut prims: Vec<Expr> =
    vec![call1("EdgeForm", call1("GrayLevel", Expr::Real(0.6)))];
  let case_pitch = width as f64 + 1.0;
  for (slot, code) in (0..cases).rev().enumerate() {
    // Neighborhood digits, most significant (leftmost cell) first.
    let mut digits = vec![0usize; width];
    let mut rest = code;
    for d in digits.iter_mut().rev() {
      *d = rest % k;
      rest /= k;
    }
    // The canonical `CellularAutomaton` evaluates the rule on a cyclic
    // row exactly one neighborhood wide, so its centre cell is the result.
    let init = Expr::List(
      digits
        .iter()
        .map(|d| Expr::Integer(*d as i128))
        .collect::<Vec<_>>()
        .into(),
    );
    let stepped = evaluate_expr_to_expr(&call(
      "CellularAutomaton",
      vec![ca[0].clone(), init, Expr::Integer(1)],
    ))
    .ok()?;
    let result = match &stepped {
      Expr::List(rows) => match rows.last() {
        Some(Expr::List(cells)) => match cells.get(r) {
          Some(Expr::Integer(v)) if (0..k as i128).contains(v) => *v as usize,
          _ => return None,
        },
        _ => return None,
      },
      _ => return None,
    };
    let x0 = slot as f64 * case_pitch;
    for (i, d) in digits.iter().enumerate() {
      prims.extend(cell(x0 + i as f64, 1.0, *d));
    }
    prims.extend(cell(x0 + r as f64, 0.0, result));
  }

  let mut gargs = vec![Expr::List(prims.into())];
  gargs.extend(graphics_opts);
  Some(crate::functions::graphics::show_ast(&[call(
    "Graphics", gargs,
  )]))
}
