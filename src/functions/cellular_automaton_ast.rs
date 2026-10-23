#[allow(unused_imports)]
use super::*;

/// CellularAutomaton[rule, init, steps]
///
/// Every supported rule specification reduces to a weighted rule: the new
/// value of a cell is digit `s` (base `k`) of the rule number, where `s` is
/// the weighted sum of the cell values in the neighborhood.
///
/// Rule specifications:
///   n                                 elementary rule (= {n, {2, {4, 2, 1}}})
///   {n, k}, {n, k, r}                 general k-color rule of range r
///   {n, {k, w}}, {n, {k, w}, r}       uniform weight w over the neighborhood
///   {n, {k, {w1, w2, ...}}, r}        1D rule with explicit weights
///   {n, {k, wmatrix}, {r1, r2}}       2D rule with a weight matrix
///
/// A range `r` is a non-negative integer or half-integer: `2r + 1` cells,
/// from offset `-Ceiling[r]` to `Floor[r]` (so `1/2` is the cell and its
/// left neighbor). An explicit offset list `{{dx1}, {dx2}, ...}` (or
/// `{{dy1, dx1}, ...}` in 2D) names the cells directly; positional and
/// explicit weights follow the list order.
///
/// Init forms: {c1, c2, ...} (cyclic) or {{c1, c2, ...}, bg} (infinite
/// background) for 1D rules; a matrix or {matrix, bg} for 2D rules.
///
/// Step specifications: the third argument is a bare `t` or the list
/// `{tspec, xspec[, yspec]}`, whose leading element is always the tspec —
/// `{1, 3}` is "steps 0..1, cells 0..3", not "steps 1 through 3".
///
/// tspec forms: `t` (steps 0 through t), `{t}` (step t alone, still in a
/// list), `{{t}}` (step t alone, returned bare), `{t1, t2}` (steps t1
/// through t2) and `{t1, t2, dt}` (every dt-th of those).
///
/// A 1D rule takes at most one space window (the cells); a 2D rule takes the
/// rows and then the columns. Each is `All` (every cell the evolution could
/// have affected), an offset `n` (from 0 or -n to n), or `{from, to[, dx]}`.
/// An axis without a window keeps only the cells that ever differ from the
/// background.
///
/// The legacy `CellularAutomaton[rule, init, t, {offt, offx, ...}]` form is
/// rewritten to the three-argument one (see `legacy_offset_form`).
pub fn cellular_automaton_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CellularAutomaton", args));

  // A SparseArray init stands for the dense list of its cells, also as the
  // cell part of a `{cells, background}` init.
  if args.len() >= 2 {
    let is_sparse = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "SparseArray");
    let densify = |e: &Expr| {
      crate::evaluator::evaluate_function_call_ast(
        "Normal",
        std::slice::from_ref(e),
      )
    };
    let dense_init = match &args[1] {
      e if is_sparse(e) => Some(densify(e)?),
      Expr::List(items) if items.len() == 2 && is_sparse(&items[0]) => Some(
        Expr::List(vec![densify(&items[0])?, items[1].clone()].into()),
      ),
      _ => None,
    };
    if let Some(dense) = dense_init {
      let still_sparse = match &dense {
        Expr::List(items) if items.len() == 2 && is_sparse(&items[0]) => true,
        e => is_sparse(e) || !matches!(e, Expr::List(_)),
      };
      if still_sparse {
        return unevaluated();
      }
      let mut new_args = args.to_vec();
      new_args[1] = dense;
      return cellular_automaton_ast(&new_args);
    }
  }

  // CellularAutomaton[rule, init] — one step. The result is the new state on
  // its own: a bare list for a cyclic init, or the `{cells, {background}}`
  // pair that can be fed straight back in for a background init.
  if args.len() == 2 {
    let stepped = cellular_automaton_ast(&[
      args[0].clone(),
      args[1].clone(),
      Expr::List(vec![Expr::List(vec![Expr::Integer(1)].into())].into()),
    ])?;
    let Expr::List(states) = &stepped else {
      return unevaluated();
    };
    let Some(Expr::List(cells)) = states.iter().next() else {
      return unevaluated();
    };
    // A `{cells, bg}` init keeps its background in the answer.
    if let Expr::List(init) = &args[1]
      && init.len() == 2
      && matches!(&init[0], Expr::List(_))
      && !matches!(&init[1], Expr::List(_))
    {
      return Ok(Expr::List(
        vec![
          Expr::List(cells.clone()),
          Expr::List(vec![init[1].clone()].into()),
        ]
        .into(),
      ));
    }
    return Ok(Expr::List(cells.clone()));
  }

  if args.len() == 4 {
    return legacy_offset_form(args);
  }

  if args.len() != 3 {
    return unevaluated();
  }

  if let Some(spec) = replacement_rule_spec(&args[0]) {
    let mut rewritten = args.to_vec();
    rewritten[0] = spec;
    return cellular_automaton_ast(&rewritten);
  }

  if let Some(result) = function_rule_automaton(args) {
    return result;
  }

  let rule = match parse_rule(&args[0]) {
    Ok(rule) => rule,
    Err(msg) => {
      crate::emit_message(&msg);
      return unevaluated();
    }
  };

  let Some(steps) = parse_step_spec(&args[2]) else {
    return unevaluated();
  };

  let Some((init, background)) = parse_init(&args[1], rule.two_d) else {
    return unevaluated();
  };

  // There is one space window per space dimension: a 1D rule takes at most
  // an `xspec` (the cells), a 2D rule an `xspec` and a `yspec` (the rows and
  // the columns). Naming more is the wrong shape for the rule, even when
  // every one of them is `All`.
  let (cols, rows) = match (steps.windows.as_slice(), rule.two_d) {
    ([], _) => (Window::Auto, Window::Auto),
    ([x], false) => (*x, Window::Auto),
    ([x], true) => (Window::Auto, *x),
    ([x, y], true) => (*y, *x),
    _ => return unevaluated(),
  };
  let Some(states) = evolve(&rule, &init, background, &steps.times, cols, rows)
  else {
    return unevaluated();
  };

  let mut exprs: Vec<Expr> = states
    .into_iter()
    .map(|state| {
      if rule.two_d {
        Expr::List(
          state
            .into_iter()
            .map(|row| {
              Expr::List(
                row.into_iter().map(|c| Expr::Integer(c as i128)).collect(),
              )
            })
            .collect(),
        )
      } else {
        // 1D states carry exactly one row.
        Expr::List(state[0].iter().map(|&c| Expr::Integer(c as i128)).collect())
      }
    })
    .collect();

  // The explicit single-state tspec `{{t}}` returns that state bare, not
  // wrapped in a length-1 list — e.g.
  // `ArrayPlot[CellularAutomaton[rule, init, {{{t}}, All, All}]]` needs the
  // matrix itself, not a singleton list holding it. Whether a space window
  // follows makes no difference: `{{{t}}}` is bare too. Every other tspec
  // form keeps returning a list even when it resolves to one state, since
  // those are documented as producing "a list of the selected states".
  if steps.single_state {
    debug_assert_eq!(exprs.len(), 1);
    return Ok(exprs.remove(0));
  }

  Ok(Expr::List(exprs.into()))
}

/// `CellularAutomaton[{lhs1 -> rhs1, …}, init, steps]` — replacement rules
/// applied to each cell's neighborhood, a list of `2r + 1` cells (or a matrix
/// of them in 2D). It is the function rule `{Replace[#, rules] &, {}, r}`
/// with the range read off the left-hand sides; a neighborhood no rule
/// matches is left as it is, exactly like `Replace`.
fn replacement_rule_spec(rule: &Expr) -> Option<Expr> {
  let Expr::List(rules) = rule else {
    return None;
  };
  let lhs = |r: &Expr| match r {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
      match pattern.as_ref() {
        Expr::List(items) => Some(items.clone()),
        _ => None,
      }
    }
    _ => None,
  };
  let first = lhs(rules.first()?)?;
  if first.is_empty() || !rules.iter().all(|r| lhs(r).is_some()) {
    return None;
  }
  // `m` cells span the range (m - 1)/2.
  let range = |m: usize| {
    if m % 2 == 1 {
      Expr::Integer((m as i128 - 1) / 2)
    } else {
      crate::functions::math_ast::make_rational(m as i128 - 1, 2)
    }
  };
  let two_d = first.iter().all(|row| matches!(row, Expr::List(_)));
  let (body, radius) = if two_d {
    let Expr::List(row) = &first[0] else {
      return None;
    };
    // The neighbors arrive as one flat list; the rules see them as rows.
    let shaped = Expr::FunctionCall {
      name: "Partition".into(),
      args: vec![Expr::Slot(1), Expr::Integer(row.len() as i128)].into(),
    };
    (
      shaped,
      Expr::List(vec![range(first.len()), range(row.len())].into()),
    )
  } else {
    (Expr::Slot(1), range(first.len()))
  };
  let func = Expr::Function {
    body: Box::new(Expr::FunctionCall {
      name: "Replace".into(),
      args: vec![body, rule.clone()].into(),
    }),
  };
  Some(Expr::List(
    vec![func, Expr::List(vec![].into()), radius].into(),
  ))
}

/// `CellularAutomaton[{f, {}, neighborhood}, init, steps]` — a rule given as
/// a function of the list of neighbor values (in offset order). The cells are
/// arbitrary expressions and the init is cyclic (a flat list for a 1D
/// neighborhood, a matrix for a 2D one) or `{cells, background}`. `None` when
/// the rule is not of this form.
fn function_rule_automaton(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::List(spec) = &args[0] else {
    return None;
  };
  let [func, Expr::List(order), nbhd_spec] = spec.as_ref() else {
    return None;
  };
  if !order.is_empty() || matches!(func, Expr::Integer(_) | Expr::List(_)) {
    return None;
  }
  let keep = || Some(Ok(unevaluated("CellularAutomaton", args)));
  let Ok(nbhd) = parse_neighborhood(nbhd_spec, &args[0]) else {
    return keep();
  };
  let steps = parse_step_spec(&args[2])?;
  let Expr::List(rows_in) = &args[1] else {
    return keep();
  };
  // `{cells, bg}` — the cells sit on an infinite background that evolves
  // too (every background cell sees an all-background neighborhood).
  let (cells_in, background): (&[Expr], Option<Expr>) = match rows_in.as_ref() {
    [cells @ Expr::List(_), bg] if !matches!(bg, Expr::List(_)) => {
      let Expr::List(cells) = cells else {
        unreachable!()
      };
      (cells.as_ref(), Some(bg.clone()))
    }
    _ => (rows_in.as_ref(), None),
  };
  let matrix_init = cells_in.iter().all(|r| matches!(r, Expr::List(_)));
  if cells_in.is_empty() || matrix_init != nbhd.two_d {
    return keep();
  }
  let init_rows: Vec<Vec<Expr>> = if nbhd.two_d {
    cells_in
      .iter()
      .map(|r| match r {
        Expr::List(cells) => cells.to_vec(),
        _ => unreachable!(),
      })
      .collect()
  } else {
    vec![cells_in.to_vec()]
  };
  let (init_h, init_w) = (init_rows.len(), init_rows[0].len());
  if init_w == 0 || init_rows.iter().any(|r| r.len() != init_w) {
    return keep();
  }
  let t_max = *steps.times.last()?;
  let has_bg = background.is_some();
  let row_axis =
    axis_layout(has_bg, init_h, nbhd.offsets.iter().map(|o| o.0), t_max)?;
  let col_axis =
    axis_layout(has_bg, init_w, nbhd.offsets.iter().map(|o| o.1), t_max)?;
  let (height, width) = (row_axis.len, col_axis.len);
  if height
    .checked_mul(width)?
    .checked_mul(t_max.max(1))?
    .checked_mul(nbhd.offsets.len().max(1))?
    > 10_000_000
  {
    return keep();
  }
  let (rows_win, cols_win) = match (steps.windows.as_slice(), nbhd.two_d) {
    ([], _) => (Window::Auto, Window::Auto),
    ([x], false) => (Window::Auto, *x),
    ([x], true) => (*x, Window::Auto),
    ([x, y], true) => (*x, *y),
    _ => return keep(),
  };

  let mut bg = background.clone().unwrap_or(Expr::Integer(0));
  let mut grid: Vec<Vec<Expr>> = if has_bg {
    let mut grid = vec![vec![bg.clone(); width]; height];
    for (i, row) in init_rows.iter().enumerate() {
      for (j, cell) in row.iter().enumerate() {
        grid[row_axis.origin as usize + i][col_axis.origin as usize + j] =
          cell.clone();
      }
    }
    grid
  } else {
    init_rows
  };

  let mut snapshots: Vec<(Vec<Vec<Expr>>, Expr)> = Vec::new();
  let mut next_time = 0;
  for t in 0..=t_max {
    if steps.times.get(next_time) == Some(&t) {
      snapshots.push((grid.clone(), bg.clone()));
      next_time += 1;
    }
    if t == t_max {
      break;
    }
    let mut next = Vec::with_capacity(height);
    for x in 0..height as i64 {
      let mut row = Vec::with_capacity(width);
      for y in 0..width as i64 {
        let neighbors: Vec<Expr> = nbhd
          .offsets
          .iter()
          .map(|&(dr, dc)| {
            let (r, c) = (x + dr, y + dc);
            if has_bg {
              if r < 0 || c < 0 || r >= height as i64 || c >= width as i64 {
                bg.clone()
              } else {
                grid[r as usize][c as usize].clone()
              }
            } else {
              grid[r.rem_euclid(height as i64) as usize]
                [c.rem_euclid(width as i64) as usize]
                .clone()
            }
          })
          .collect();
        match crate::evaluator::apply_function_to_arg(
          func,
          &Expr::List(neighbors.into()),
        ) {
          Ok(v) => row.push(v),
          Err(e) => return Some(Err(e)),
        }
      }
      next.push(row);
    }
    grid = next;
    if has_bg {
      let all_bg = Expr::List(vec![bg.clone(); nbhd.offsets.len()].into());
      bg = match crate::evaluator::apply_function_to_arg(func, &all_bg) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
    }
  }

  // Auto windows keep the rows/columns holding a non-background cell at a
  // returned step; cyclic inits keep their full size.
  let differs = |cell: &Expr, bg: &Expr| {
    crate::syntax::expr_to_string(cell) != crate::syntax::expr_to_string(bg)
  };
  let row_active = |r: usize| {
    snapshots
      .iter()
      .any(|(g, bg)| g[r].iter().any(|c| differs(c, bg)))
  };
  let col_active = |c: usize| {
    snapshots
      .iter()
      .any(|(g, bg)| g.iter().any(|row| differs(&row[c], bg)))
  };
  let indices = |window: Window,
                 axis: &AxisLayout,
                 active: &dyn Fn(usize) -> bool|
   -> Vec<i64> {
    match window {
      Window::Range(from, to, step) => (from..=to)
        .step_by(step)
        .map(|off| axis.origin + off)
        .collect(),
      Window::Auto if has_bg => {
        let first = (0..axis.len).find(|&i| active(i)).unwrap_or(0);
        let last = (0..axis.len).rfind(|&i| active(i)).unwrap_or(0);
        (first..=last).map(|i| i as i64).collect()
      }
      Window::Auto | Window::All => (0..axis.len as i64).collect(),
    }
  };
  let row_idx = indices(rows_win, &row_axis, &row_active);
  let col_idx = indices(cols_win, &col_axis, &col_active);
  let mut states: Vec<Expr> = snapshots
    .iter()
    .map(|(g, bg)| {
      let cell = |r: i64, c: i64| -> Expr {
        if has_bg {
          if r < 0 || r >= height as i64 || c < 0 || c >= width as i64 {
            bg.clone()
          } else {
            g[r as usize][c as usize].clone()
          }
        } else {
          g[r.rem_euclid(height as i64) as usize]
            [c.rem_euclid(width as i64) as usize]
            .clone()
        }
      };
      let pick = |r: i64| -> Expr {
        Expr::List(col_idx.iter().map(|&c| cell(r, c)).collect())
      };
      if nbhd.two_d {
        Expr::List(row_idx.iter().map(|&r| pick(r)).collect())
      } else {
        pick(0)
      }
    })
    .collect();
  if steps.single_state {
    return Some(Ok(states.remove(0)));
  }
  Some(Ok(Expr::List(states.into())))
}

/// CellularAutomaton[rule, init, t, offs] — the discontinued four-argument
/// form. `t` must be a non-negative integer and `offs` an integer, `All` or
/// an `{offt, offx, ...}` list with at most one `offx` per space dimension.
/// It is the three-argument form with step spec `{offt, offx, ...}`, except
/// that `offt` is read relative to `t`: `All` is steps 0 through t, and a
/// negative step `-n` counts back from t (`-1` is step t). A bare `-n`
/// selects the last n steps.
fn legacy_offset_form(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CellularAutomaton", args));
  let rule = match parse_rule(&args[0]) {
    Ok(rule) => rule,
    Err(msg) => {
      crate::emit_message(&msg);
      return unevaluated();
    }
  };
  let space_dims = if rule.two_d { 2 } else { 1 };
  let t = match &args[2] {
    Expr::Integer(t) if *t >= 0 => i64::try_from(*t).ok(),
    _ => None,
  };
  let offs: Option<Vec<Expr>> = match &args[3] {
    Expr::Integer(_) => Some(vec![args[3].clone()]),
    Expr::Identifier(s) if s == "All" => Some(vec![args[3].clone()]),
    Expr::List(items) if !items.is_empty() && items.len() <= space_dims + 1 => {
      Some(items.to_vec())
    }
    _ => None,
  };
  let (Some(t), Some(mut offs)) = (t, offs) else {
    crate::emit_message(&format!(
      "CellularAutomaton::offg4: Since four arguments are used, the third argument {} must be a non-negative machine integer, and the fourth argument {} must be an integer or {{offt, offx1, offx2, ... offxk}}, where offt is present and k does not exceed {}, the number of spatial dimensions. Note that the four-argument form of CellularAutomaton is being discontinued.",
      expr_to_string(&args[2]),
      expr_to_string(&args[3]),
      space_dims
    ));
    return unevaluated();
  };
  match resolve_legacy_tspec(&offs[0], t) {
    Ok(tspec) => offs[0] = tspec,
    Err(step) => {
      crate::emit_message(&format!(
        "CellularAutomaton::offtm4: Step size {} and temporal offset specification {} request step {}, but the lowest step is 0. Note that the four-argument form of CellularAutomaton is being discontinued.",
        t,
        expr_to_string(&offs[0]),
        step
      ));
      return unevaluated();
    }
  }
  cellular_automaton_ast(&[
    args[0].clone(),
    args[1].clone(),
    Expr::List(offs.into()),
  ])
}

/// Rewrite a legacy `offt` relative to the step count `t` into a modern
/// tspec. `Err` carries a step that resolves to before step 0.
fn resolve_legacy_tspec(offt: &Expr, t: i64) -> Result<Expr, i128> {
  let step = |e: &Expr| -> Result<Expr, i128> {
    match e {
      Expr::Integer(n) if *n < 0 => {
        let s = t as i128 + 1 + n;
        if s < 0 { Err(s) } else { Ok(Expr::Integer(s)) }
      }
      other => Ok(other.clone()),
    }
  };
  match offt {
    Expr::Identifier(s) if s == "All" => Ok(Expr::Integer(t as i128)),
    // The last n steps.
    Expr::Integer(n) if *n < 0 => Ok(Expr::List(
      vec![step(offt)?, Expr::Integer(t as i128)].into(),
    )),
    Expr::List(items) => {
      let resolved: Result<Vec<Expr>, i128> = match items.as_ref() {
        [Expr::List(inner)] if inner.len() == 1 => {
          Ok(vec![Expr::List(vec![step(&inner[0])?].into())])
        }
        [t1] => Ok(vec![step(t1)?]),
        [t1, t2] => Ok(vec![step(t1)?, step(t2)?]),
        [t1, t2, dt] => Ok(vec![step(t1)?, step(t2)?, dt.clone()]),
        _ => Ok(items.to_vec()),
      };
      Ok(Expr::List(resolved?.into()))
    }
    other => Ok(other.clone()),
  }
}

struct RuleSpec {
  /// The rule number's base-`k` digits, least significant first: digit `s`
  /// is the new value of a cell whose weighted neighborhood sum is `s`.
  digits: Vec<u128>,
  /// The neighborhood: the cell `(drow, dcol)` away from the updated cell
  /// adds `weight * value` to the weighted sum. 1D rules use `drow = 0`.
  neighbors: Vec<(i64, i64, u128)>,
  two_d: bool,
}

/// How one space axis of the result is chosen.
#[derive(Clone, Copy)]
enum Window {
  /// No window named: the cells that ever differ from the background.
  Auto,
  /// `All`: every cell the evolution could have affected.
  All,
  /// `{from, to, step}` offsets relative to the init's first cell.
  Range(i64, i64, usize),
}

struct StepSpec {
  /// The (ascending) time steps whose states are returned.
  times: Vec<usize>,
  /// The space windows named after `tspec`, in order. A 1D rule takes at
  /// most one (the cells); a 2D rule takes at most two (rows then columns).
  windows: Vec<Window>,
  /// Whether `tspec` was written as the explicit single-state `{{t}}`
  /// form, as opposed to a form documented as always yielding "a list of
  /// the selected states" (`t`, `{t}`, `{t1, t2}`, `{t1, t2, dt}`) that
  /// merely happens to resolve to one step.
  single_state: bool,
}

fn as_nonneg_int(expr: &Expr) -> Option<u128> {
  match expr {
    Expr::Integer(n) if *n >= 0 => Some(*n as u128),
    _ => None,
  }
}

/// The cells a rule reads, before weights are attached.
struct Neighborhood {
  /// `(drow, dcol)` offsets in weight order (row-major for a range).
  offsets: Vec<(i64, i64)>,
  /// The cell counts `{d}` or `{d1, d2}` of a range neighborhood, which an
  /// explicit weight array must match. `None` for an offset list, whose
  /// weights are a flat list in offset order.
  dims: Option<Vec<usize>>,
  two_d: bool,
}

/// Per-dimension neighborhood radius cap. Real rules use single-digit
/// ranges; this only guards against absurd specs allocating huge weight
/// grids.
const MAX_RANGE: usize = 256;

/// A range `r` (an integer or half-integer) as its cell count `2r + 1` and
/// the offset of its first cell, `-Ceiling[r]`.
fn parse_radius(expr: &Expr) -> Option<(usize, i64)> {
  match expr {
    Expr::Integer(r) if *r >= 0 && *r <= MAX_RANGE as i128 => {
      Some((2 * *r as usize + 1, -(*r as i64)))
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(2))
          if *p > 0 && *p <= 2 * MAX_RANGE as i128 =>
        {
          let cells = *p as usize + 1;
          Some((cells, -((cells / 2) as i64)))
        }
        _ => None,
      }
    }
    _ => None,
  }
}

fn range_neighborhood(radii: &[(usize, i64)]) -> Neighborhood {
  match radii {
    [(d, lo)] => Neighborhood {
      offsets: (0..*d as i64).map(|j| (0, lo + j)).collect(),
      dims: Some(vec![*d]),
      two_d: false,
    },
    [(d1, lo1), (d2, lo2)] => Neighborhood {
      offsets: (0..*d1 as i64)
        .flat_map(|i| (0..*d2 as i64).map(move |j| (lo1 + i, lo2 + j)))
        .collect(),
      dims: Some(vec![*d1, *d2]),
      two_d: true,
    },
    _ => unreachable!("a range has one or two radii"),
  }
}

/// The neighbor spec at position 3 of a rule: a range `r`, a 2D range
/// `{r1, r2}` or an offset list. `Err(None)` marks a well-formed spec Woxi
/// does not support (e.g. offsets in three dimensions).
fn parse_neighborhood(
  expr: &Expr,
  spec: &Expr,
) -> Result<Neighborhood, Option<String>> {
  let rspec = || {
    Some(format!(
      "CellularAutomaton::rspec: The neighbor specification at position 3 of rule specification {} should be a radius, a nonempty list of radii or a nonempty list of offsets. A radius must be (d - 1)/2 for some positive machine integer d. The offsets are nonempty lists of machine integers.",
      expr_to_string(spec)
    ))
  };
  if let Some(radius) = parse_radius(expr) {
    return Ok(range_neighborhood(&[radius]));
  }
  let Expr::List(items) = expr else {
    return Err(rspec());
  };
  if items.is_empty() {
    return Err(rspec());
  }
  if items.iter().all(|e| matches!(e, Expr::List(_))) {
    // An explicit offset list.
    let offsets: Option<Vec<Vec<i64>>> = items
      .iter()
      .map(|item| match item {
        Expr::List(o) if !o.is_empty() => o
          .iter()
          .map(|c| match c {
            Expr::Integer(v) => i64::try_from(*v).ok(),
            _ => None,
          })
          .collect(),
        _ => None,
      })
      .collect();
    let offsets = offsets.ok_or_else(rspec)?;
    let dim = offsets[0].len();
    if offsets.iter().any(|o| o.len() != dim) {
      return Err(Some(format!(
        "CellularAutomaton::offfo: Because rule specification {} has no specified order, the offsets at position 3 must have the same length.",
        expr_to_string(spec)
      )));
    }
    if dim > 2
      || offsets
        .iter()
        .flatten()
        .any(|c| c.unsigned_abs() > MAX_RANGE as u64)
    {
      return Err(None);
    }
    for (i, o) in offsets.iter().enumerate() {
      if offsets[..i].contains(o) {
        return Err(Some(format!(
          "CellularAutomaton::offrep: The offsets at position 3 in rule specification {} (after adding any missing 0 time coordinates in the higher-order case) must be distinct.",
          expr_to_string(spec)
        )));
      }
    }
    return Ok(Neighborhood {
      offsets: offsets
        .iter()
        .map(|o| if dim == 1 { (0, o[0]) } else { (o[0], o[1]) })
        .collect(),
      dims: None,
      two_d: dim == 2,
    });
  }
  match items.as_ref() {
    // A one-dimensional list of radii: `{1}` is range 1.
    [r] => parse_radius(r)
      .map(|a| range_neighborhood(&[a]))
      .ok_or_else(rspec),
    [r1, r2] => match (parse_radius(r1), parse_radius(r2)) {
      (Some(a), Some(b)) => Ok(range_neighborhood(&[a, b])),
      _ => Err(rspec()),
    },
    // Ranges in three or more dimensions.
    _ if items.iter().all(|e| parse_radius(e).is_some()) => Err(None),
    _ => Err(rspec()),
  }
}

/// Parse a rule spec. `Err` carries the message to emit.
fn parse_rule(expr: &Expr) -> Result<RuleSpec, String> {
  let nspecnl = || {
    format!(
      "CellularAutomaton::nspecnl: Rule specification {} should be an Integer, a List, a pure Boolean function, a String or an Association.",
      expr_to_string(expr)
    )
  };
  let (n, k, neighbors, two_d) = match expr {
    Expr::Integer(n) if *n >= 0 => (
      BigInt::from(*n),
      2,
      vec![(0, -1, 4), (0, 0, 2), (0, 1, 1)],
      false,
    ),
    Expr::List(items) if items.len() == 2 || items.len() == 3 => {
      // The rule number may be arbitrarily large: a 2-color range-1 2D
      // outer-totalistic rule already needs some 2^512.
      let n = match &items[0] {
        Expr::Integer(n) if *n >= 0 => BigInt::from(*n),
        Expr::BigInteger(n) if n.sign() != num_bigint::Sign::Minus => n.clone(),
        _ => return Err(nspecnl()),
      };
      let nbhd = match items.get(2) {
        Some(r) => parse_neighborhood(r, expr)
          .map_err(|msg| msg.unwrap_or_else(nspecnl))?,
        // The default range is 1.
        None => range_neighborhood(&[(3, -1)]),
      };
      let (k, weights) = match &items[1] {
        // {n, k[, r]} — general rule: the neighborhood read as a base-k
        // number, i.e. positional weights k^(cells-1), ..., k, 1.
        Expr::Integer(k) if *k >= 1 => {
          let k = *k as u128;
          let cells = nbhd.offsets.len();
          let weights: Option<Vec<u128>> = (0..cells)
            .map(|i| k.checked_pow(u32::try_from(cells - 1 - i).ok()?))
            .collect();
          (k, weights.ok_or_else(nspecnl)?)
        }
        // {n, {k, wspec}[, r]}
        Expr::List(kw) if kw.len() == 2 => {
          let k = as_nonneg_int(&kw[0])
            .filter(|&k| k > 0)
            .ok_or_else(nspecnl)?;
          (k, parse_weights(&kw[1], &nbhd, expr)?)
        }
        _ => return Err(nspecnl()),
      };
      let neighbors = nbhd
        .offsets
        .iter()
        .zip(weights)
        .map(|(&(dr, dc), w)| (dr, dc, w))
        .collect();
      (n, k, neighbors, nbhd.two_d)
    }
    _ => return Err(nspecnl()),
  };

  // The weighted sum runs from 0 to (k - 1) * (total weight), so a rule has
  // that many plus one base-k digits.
  let digit_count = neighbors
    .iter()
    .try_fold(0u128, |acc, &(_, _, w)| acc.checked_add(w))
    .and_then(|total| total.checked_mul(k - 1))
    .and_then(|max_sum| max_sum.checked_add(1));
  if let Some(count) = digit_count
    && count <= 1_000_000
  {
    let largest =
      num_traits::pow::pow(BigInt::from(k), count as usize) - BigInt::from(1);
    if n > largest {
      return Err(format!(
        "CellularAutomaton::rsize: The specified rule number {n} is greater than the largest possible rule number ({largest})."
      ));
    }
  }
  let mut digits: Vec<u128> = Vec::new();
  if k > 1 {
    use num_traits::{ToPrimitive, Zero};
    let base = BigInt::from(k);
    let mut rest = n;
    while !rest.is_zero() {
      digits.push((&rest % &base).to_u128().unwrap_or(0));
      rest /= &base;
    }
  }

  Ok(RuleSpec {
    digits,
    neighbors,
    two_d,
  })
}

/// Weight part of a rule spec, in neighborhood order: a uniform weight, or
/// an array matching the range's dimensions (a flat list matching the
/// number of cells for an offset list).
fn parse_weights(
  wspec: &Expr,
  nbhd: &Neighborhood,
  spec: &Expr,
) -> Result<Vec<u128>, String> {
  if let Some(w) = as_nonneg_int(wspec) {
    return Ok(vec![w; nbhd.offsets.len()]);
  }
  let flat_list = |e: &Expr, len: usize| -> Option<Vec<u128>> {
    match e {
      Expr::List(items) if items.len() == len => {
        items.iter().map(as_nonneg_int).collect()
      }
      _ => None,
    }
  };
  match &nbhd.dims {
    Some(dims) => {
      let weights = match dims.as_slice() {
        [d] => flat_list(wspec, *d),
        [d1, d2] => flat_list_of_rows(wspec, *d1, *d2, &flat_list),
        _ => None,
      };
      weights.ok_or_else(|| {
        format!(
          "CellularAutomaton::wtspec: The weight specification at position {{2, 2}} of rule specification {} should be 1 or an array of non-negative machine integers with dimensions {{{}}}.",
          expr_to_string(spec),
          dims
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
        )
      })
    }
    None => flat_list(wspec, nbhd.offsets.len()).ok_or_else(|| {
      let offsets = match spec {
        Expr::List(items) => expr_to_string(&items[2]),
        _ => String::new(),
      };
      format!(
        "CellularAutomaton::wtlen: The length of the weight list {} does not match the number of cells in the neighborhood {}.",
        expr_to_string(wspec),
        offsets
      )
    }),
  }
}

/// A `rows` x `cols` weight matrix, flattened row-major.
fn flat_list_of_rows(
  wspec: &Expr,
  rows: usize,
  cols: usize,
  flat_list: &dyn Fn(&Expr, usize) -> Option<Vec<u128>>,
) -> Option<Vec<u128>> {
  match wspec {
    Expr::List(items) if items.len() == rows => {
      let mut weights = Vec::with_capacity(rows * cols);
      for row in items {
        weights.extend(flat_list(row, cols)?);
      }
      Some(weights)
    }
    _ => None,
  }
}

/// Upper bound on the number of returned states — far beyond any sensible
/// use, but keeps a typo like 10^12 steps from exhausting memory.
const MAX_STATES: usize = 100_000;

/// Parse a `tspec` — the first element of the `{tspec, xspec, …}` step spec
/// (or the whole third argument when that is a bare integer). Returns the
/// resolved times and whether it was written as the explicit single-state
/// `{{t}}` form (see `StepSpec::single_state`).
fn parse_tspec(expr: &Expr) -> Option<(Vec<usize>, bool)> {
  let step = |t: &Expr| -> Option<usize> {
    let t = usize::try_from(as_nonneg_int(t)?).ok()?;
    (t < MAX_STATES).then_some(t)
  };
  match expr {
    // t — all steps 0 through t.
    Expr::Integer(t) if *t >= 0 && (*t as u128) < MAX_STATES as u128 => {
      Some(((0..=(*t as usize)).collect(), false))
    }
    Expr::List(items) => match items.as_ref() {
      // {{t}} — step t alone, returned bare rather than in a length-1 list.
      // Tried before `{t}` so the inner list is not read as a step number.
      [Expr::List(inner)] if inner.len() == 1 => {
        Some((vec![step(&inner[0])?], true))
      }
      // {t} — step t alone, still returned inside a list.
      [t] => Some((vec![step(t)?], false)),
      // {t1, t2} and {t1, t2, dt} — the selected states, as a list.
      [t1, t2] | [t1, t2, _] => {
        let from = step(t1)?;
        let to = step(t2)?;
        let dt = match items.as_ref() {
          [_, _, dt] => usize::try_from(as_nonneg_int(dt)?).ok()?,
          _ => 1,
        };
        if to < from || dt == 0 || (to - from) / dt >= MAX_STATES {
          return None;
        }
        Some(((from..=to).step_by(dt).collect(), false))
      }
      _ => None,
    },
    _ => None,
  }
}

/// The third argument is either a bare `t` or a `{tspec, xspec, …}` list —
/// the leading element is always the tspec, never a space window. That is
/// what separates `{1, 3}` (steps 0..1 restricted to cells 0..3) from
/// `{{1, 3}}` (steps 1 through 3 over every cell).
fn parse_step_spec(expr: &Expr) -> Option<StepSpec> {
  let (tspec, window_specs): (&Expr, &[Expr]) = match expr {
    Expr::List(items) if !items.is_empty() => (&items[0], &items.as_ref()[1..]),
    other => (other, &[]),
  };
  let (times, single_state) = parse_tspec(tspec)?;
  let windows: Option<Vec<_>> =
    window_specs.iter().map(parse_cell_spec).collect();
  Some(StepSpec {
    times,
    windows: windows?,
    single_state,
  })
}

/// The cells an `xspec` names; a spec that cannot be read yields `None`.
fn parse_cell_spec(expr: &Expr) -> Option<Window> {
  let as_int = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      _ => None,
    }
  };
  match expr {
    Expr::Identifier(s) if s == "All" => Some(Window::All),
    // A bare `n` runs from the origin out to offset n (either direction).
    Expr::Integer(_) => {
      let n = as_int(expr)?;
      Some(if n >= 0 {
        Window::Range(0, n, 1)
      } else {
        Window::Range(n, 0, 1)
      })
    }
    Expr::List(items) if items.len() == 2 || items.len() == 3 => {
      let from = as_int(&items[0])?;
      let to = as_int(&items[1])?;
      let step = match items.get(2) {
        Some(d) => usize::try_from(as_int(d)?).ok().filter(|&d| d > 0)?,
        None => 1,
      };
      (to >= from).then_some(Window::Range(from, to, step))
    }
    _ => None,
  }
}

/// Parse the initial condition. Returns the initial grid (1D inits become a
/// single-row grid) and `Some(background)` for infinite-background inits or
/// `None` for cyclic ones.
#[allow(clippy::type_complexity)]
fn parse_init(
  expr: &Expr,
  two_d: bool,
) -> Option<(Vec<Vec<u128>>, Option<u128>)> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.is_empty() {
    return None;
  }

  let parse_row = |e: &Expr| -> Option<Vec<u128>> {
    match e {
      Expr::List(cells) if !cells.is_empty() => {
        cells.iter().map(as_nonneg_int).collect()
      }
      _ => None,
    }
  };
  let parse_matrix = |e: &Expr| -> Option<Vec<Vec<u128>>> {
    match e {
      Expr::List(rows) if !rows.is_empty() => {
        let matrix: Option<Vec<Vec<u128>>> =
          rows.iter().map(parse_row).collect();
        let matrix = matrix?;
        let cols = matrix[0].len();
        if matrix.iter().any(|r| r.len() != cols) {
          return None;
        }
        Some(matrix)
      }
      _ => None,
    }
  };

  if two_d {
    // {matrix, bg} — infinite background.
    if items.len() == 2
      && let Some(bg) = as_nonneg_int(&items[1])
      && let Some(matrix) = parse_matrix(&items[0])
    {
      return Some((matrix, Some(bg)));
    }
    // Bare matrix — cyclic in both directions.
    let matrix = parse_matrix(expr)?;
    Some((matrix, None))
  } else {
    // {{cells}, bg} — infinite background.
    if items.len() == 2
      && let Some(bg) = as_nonneg_int(&items[1])
      && let Some(row) = parse_row(&items[0])
    {
      return Some((vec![row], Some(bg)));
    }
    // {cells} — cyclic.
    let row = parse_row(expr)?;
    Some((vec![row], None))
  }
}

/// Digit `s` (base `k`) of the rule number — the new value of a cell whose
/// neighborhood has weighted sum `s`.
fn rule_digit(rule: &RuleSpec, s: u128) -> u128 {
  usize::try_from(s)
    .ok()
    .and_then(|i| rule.digits.get(i).copied())
    .unwrap_or(0)
}

/// Where one axis of the evolution grid sits: `origin` is the grid index of
/// offset 0 (the init's first cell) and `len` the grid's size.
struct AxisLayout {
  origin: i64,
  len: usize,
}

/// A background init grows by the neighborhood's reach each step: a cell
/// reading offset `d > 0` pulls information `d` cells toward lower indices,
/// and one reading `d < 0` pushes it `-d` cells toward higher ones. The
/// grid holds exactly the cells that could differ from the background by
/// step `t_max`; every cell outside it equals the (evolving) background.
/// A cyclic init keeps its size.
fn axis_layout(
  background: bool,
  init_len: usize,
  offsets: impl Iterator<Item = i64> + Clone,
  t_max: usize,
) -> Option<AxisLayout> {
  if !background {
    return Some(AxisLayout {
      origin: 0,
      len: init_len,
    });
  }
  let t_max = i64::try_from(t_max).ok()?;
  let hi = offsets.clone().max().unwrap_or(0).max(0);
  let lo = offsets.min().unwrap_or(0).min(0);
  let before = hi.checked_mul(t_max)?;
  let after = (-lo).checked_mul(t_max)?;
  let len = usize::try_from(before.checked_add(after)?)
    .ok()?
    .checked_add(init_len)?;
  Some(AxisLayout {
    origin: before,
    len,
  })
}

/// Evolve `init` under `rule`, returning the states at the requested `times`
/// (which must be ascending). `cols` windows the 1D cells (or, for a 2D
/// rule, the columns) and `rows` the rows of a 2D rule.
fn evolve(
  rule: &RuleSpec,
  init: &[Vec<u128>],
  background: Option<u128>,
  times: &[usize],
  cols: Window,
  rows: Window,
) -> Option<Vec<Vec<Vec<u128>>>> {
  let t_max = *times.last()?;
  let row_axis = axis_layout(
    background.is_some(),
    init.len(),
    rule.neighbors.iter().map(|n| n.0),
    t_max,
  )?;
  let col_axis = axis_layout(
    background.is_some(),
    init[0].len(),
    rule.neighbors.iter().map(|n| n.1),
    t_max,
  )?;
  let (height, width) = (row_axis.len, col_axis.len);

  // Refuse absurd evolutions instead of hanging (grids beyond any plotting
  // use). The work bound covers cells x steps x neighborhood size; the
  // result bound covers the memory held by the returned states.
  let cells = height.checked_mul(width)?;
  let work = cells
    .checked_mul(t_max.max(1))?
    .checked_mul(rule.neighbors.len())?;
  let result_cells = cells.checked_mul(times.len())?;
  // The work bound is generous: on a background init the evolution only
  // computes the region the pattern can have reached (see below).
  if cells > 4_000_000 || work > 10_000_000_000 || result_cells > 64_000_000 {
    return None;
  }

  // Place the init on a background-filled grid (background inits) or use
  // it verbatim (cyclic inits).
  let mut grid = match background {
    Some(bg) => {
      let mut grid = vec![vec![bg; width]; height];
      for (i, row) in init.iter().enumerate() {
        for (j, &cell) in row.iter().enumerate() {
          grid[row_axis.origin as usize + i][col_axis.origin as usize + j] =
            cell;
        }
      }
      grid
    }
    None => init.to_vec(),
  };

  // The infinite background evolves too: every background cell sees an
  // all-background neighborhood.
  let mut bg = background.unwrap_or(0);
  let weight_total: u128 = rule
    .neighbors
    .iter()
    .fold(0u128, |acc, &(_, _, w)| acc.saturating_add(w));

  let mut snapshots: Vec<(Vec<Vec<u128>>, u128)> = Vec::new();
  let mut next_time = 0;
  // Rows r0..=r1 and columns c0..=c1 that can hold a non-background cell
  // (background inits only; None once the background itself evolves).
  let radius = (
    rule
      .neighbors
      .iter()
      .map(|n| n.0.unsigned_abs() as usize)
      .max()
      .unwrap_or(0),
    rule
      .neighbors
      .iter()
      .map(|n| n.1.unsigned_abs() as usize)
      .max()
      .unwrap_or(0),
  );
  let mut active: Option<(usize, usize, usize, usize)> = background.map(|_| {
    let (r0, c0) = (row_axis.origin as usize, col_axis.origin as usize);
    (
      r0,
      r0 + init.len().max(1) - 1,
      c0,
      c0 + init.first().map_or(1, |row| row.len().max(1)) - 1,
    )
  });
  for t in 0..=t_max {
    if next_time < times.len() && times[next_time] == t {
      snapshots.push((grid.clone(), bg));
      next_time += 1;
    }
    if t == t_max {
      break;
    }

    let next_bg = rule_digit(rule, bg.saturating_mul(weight_total));
    // With a background that stays put, only cells within reach of a
    // non-background cell can change: the init's box grown by the
    // neighborhood radius once per step. Everything else stays background.
    if let (Some(_), Some(reach)) = (background, active.as_mut())
      && next_bg == bg
    {
      reach.0 = reach.0.saturating_sub(radius.0);
      reach.1 = (reach.1 + radius.0).min(height - 1);
      reach.2 = reach.2.saturating_sub(radius.1);
      reach.3 = (reach.3 + radius.1).min(width - 1);
      let (r0, r1, c0, c1) = *reach;
      let mut next = grid.clone();
      for x in r0..=r1 {
        for y in c0..=c1 {
          let mut s: u128 = 0;
          for &(dr, dc, w) in &rule.neighbors {
            let r = x as i64 + dr;
            let c = y as i64 + dc;
            let value =
              if r < 0 || c < 0 || r >= height as i64 || c >= width as i64 {
                bg
              } else {
                grid[r as usize][c as usize]
              };
            s = s.saturating_add(w.saturating_mul(value));
          }
          next[x][y] = rule_digit(rule, s);
        }
      }
      grid = next;
      continue;
    }
    active = None;

    let mut next = vec![vec![0u128; width]; height];
    for (x, next_row) in next.iter_mut().enumerate() {
      for (y, next_cell) in next_row.iter_mut().enumerate() {
        let mut s: u128 = 0;
        for &(dr, dc, w) in &rule.neighbors {
          let r = x as i64 + dr;
          let c = y as i64 + dc;
          let value = if background.is_some() {
            if r < 0 || c < 0 || r >= height as i64 || c >= width as i64 {
              bg
            } else {
              grid[r as usize][c as usize]
            }
          } else {
            grid[r.rem_euclid(height as i64) as usize]
              [c.rem_euclid(width as i64) as usize]
          };
          s = s.saturating_add(w.saturating_mul(value));
        }
        *next_cell = rule_digit(rule, s);
      }
    }
    grid = next;
    bg = rule_digit(rule, bg.saturating_mul(weight_total));
  }

  // Each axis independently keeps its explicit window (wrapping around for
  // a cyclic init, reading the evolving background outside a background
  // grid), the whole grid for `All`, or, with no window named, every
  // row/column that holds a non-background cell at a returned step (a
  // cyclic init keeps its full size, since it has no background to trim
  // against).
  let row_active = |r: usize| {
    snapshots
      .iter()
      .any(|(g, bg)| g[r].iter().any(|&c| c != *bg))
  };
  let col_active = |c: usize| {
    snapshots
      .iter()
      .any(|(g, bg)| g.iter().any(|row| row[c] != *bg))
  };

  // Bound the output size arithmetically before collecting either index
  // list: an explicit window's length isn't capped by `height`/`width`
  // (those stay at the small cyclic-init size regardless of window size),
  // so collecting first could allocate gigabytes before this check ever
  // ran.
  let axis_len = |window: Window, grid_len: usize| -> Option<usize> {
    match window {
      Window::Range(from, to, step) => {
        let span = usize::try_from(to.checked_sub(from)?).ok()?;
        span.checked_div(step)?.checked_add(1)
      }
      Window::All | Window::Auto => Some(grid_len),
    }
  };
  let windowed_cells = axis_len(rows, height)?
    .checked_mul(axis_len(cols, width)?)?
    .checked_mul(times.len())?;
  if windowed_cells > 64_000_000 {
    return None;
  }

  let indices = |window: Window,
                 axis: &AxisLayout,
                 active: &dyn Fn(usize) -> bool|
   -> Vec<i64> {
    match window {
      Window::Range(from, to, step) => (from..=to)
        .step_by(step)
        .map(|off| axis.origin + off)
        .collect(),
      Window::Auto if background.is_some() => {
        let first = (0..axis.len).find(|&i| active(i)).unwrap_or(0);
        let last = (0..axis.len).rfind(|&i| active(i)).unwrap_or(0);
        (first..=last).map(|i| i as i64).collect()
      }
      Window::Auto | Window::All => (0..axis.len as i64).collect(),
    }
  };
  let row_indices = indices(rows, &row_axis, &row_active);
  let col_indices = indices(cols, &col_axis, &col_active);

  Some(
    snapshots
      .iter()
      .map(|(g, bg)| {
        row_indices
          .iter()
          .map(|&r| {
            col_indices
              .iter()
              .map(|&c| {
                if background.is_some() {
                  if r < 0 || r >= height as i64 || c < 0 || c >= width as i64 {
                    *bg
                  } else {
                    g[r as usize][c as usize]
                  }
                } else {
                  let r = r.rem_euclid(height as i64) as usize;
                  let c = c.rem_euclid(width as i64) as usize;
                  g[r][c]
                }
              })
              .collect()
          })
          .collect()
      })
      .collect(),
  )
}
