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
/// Init forms: {c1, c2, ...} (cyclic) or {{c1, c2, ...}, bg} (infinite
/// background) for 1D rules; a matrix or {matrix, bg} for 2D rules.
///
/// Step specifications: t and {t} (both steps 0..t), {{t}}, {{t1, t2}} and
/// {{t1, t2, dt}} (a list of the selected states). {tspec, xspec} restricts
/// a 1D rule's returned cells to xspec; {tspec, xspec, yspec} restricts a
/// 2D rule's returned rows/columns to xspec/yspec. Each of xspec/yspec is
/// `All`, an offset `n` (from 0 or -n to n), or `{from, to[, dx]}`. A
/// windowed spec whose tspec is the explicit single-state `{{t}}` form
/// returns that state bare rather than in a length-1 list; every other
/// tspec form keeps returning a list even when it resolves to one state
/// (e.g. bare `0`, or a `{{t1, t2}}` range that collapses to one step).
pub fn cellular_automaton_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CellularAutomaton", args));

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

  if args.len() != 3 {
    return unevaluated();
  }

  let Some(rule) = parse_rule(&args[0]) else {
    crate::emit_message(&format!(
      "CellularAutomaton::nspecnl: Rule specification {} should be an Integer, a List, a pure Boolean function, a String or an Association.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return unevaluated();
  };

  let Some(steps) = parse_step_spec(&args[2]) else {
    return unevaluated();
  };

  let Some((init, background)) = parse_init(&args[1], rule.two_d) else {
    return unevaluated();
  };

  // A `{tspec, xspec}` window (one axis) only makes sense on a 1D rule; a
  // `{tspec, xspec, yspec}` window (two axes) only makes sense on a 2D
  // rule. `window_axes` tracks how many axes were named (0, 1 or 2)
  // regardless of whether they resolved to `All` (`None`) or an explicit
  // range, since even an `All` xspec/yspec is the wrong shape for the
  // other dimensionality.
  match (steps.window_axes, rule.two_d) {
    (0, _) | (1, false) | (2, true) => {}
    _ => return unevaluated(),
  }
  let Some(states) = evolve(
    &rule,
    &init,
    background,
    &steps.times,
    steps.cells.or(steps.cols),
    steps.rows,
  ) else {
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

  // A step spec with an explicit cell/row/column window (`{tspec, xspec}` or
  // `{tspec, xspec, yspec}`) whose tspec is the explicit single-state
  // `{{t}}` form returns that state bare, not wrapped in a length-1 list —
  // e.g. `ArrayPlot[CellularAutomaton[rule, init, {{{t}}, All, All}]]`
  // needs the matrix itself, not a singleton list holding it. Every other
  // tspec form (bare `t`/`{t}`, or a `{{t1, t2[, dt]}}` range) keeps
  // returning a list even when it happens to resolve to one state, since
  // those forms are documented as always producing "a list of the
  // selected states".
  if steps.window_axes > 0 && steps.single_state {
    debug_assert_eq!(exprs.len(), 1);
    return Ok(exprs.remove(0));
  }

  Ok(Expr::List(exprs.into()))
}

struct RuleSpec {
  n: u128,
  k: u128,
  /// Neighborhood weights, (2*r1 + 1) rows x (2*r2 + 1) columns.
  /// 1D rules use a single row (r1 = 0).
  weights: Vec<Vec<u128>>,
  two_d: bool,
}

struct StepSpec {
  /// The (ascending) time steps whose states are returned.
  times: Vec<usize>,
  /// 1D cell offsets to keep, as `(from, to, step)` relative to the first
  /// cell of the initial condition. `None` keeps the whole affected region.
  cells: Option<(i64, i64, usize)>,
  /// 2D row offsets to keep (the `xspec` of a `{tspec, xspec, yspec}`
  /// step spec), relative to the first row of the initial condition.
  /// `None` keeps the whole affected region.
  rows: Option<(i64, i64, usize)>,
  /// 2D column offsets to keep (the `yspec`), relative to the first column
  /// of the initial condition. `None` keeps the whole affected region.
  cols: Option<(i64, i64, usize)>,
  /// How many window axes were named: 0 for a bare `tspec`, 1 for
  /// `{tspec, xspec}` (1D rules), 2 for `{tspec, xspec, yspec}` (2D
  /// rules). A windowed spec (> 0) whose tspec was the explicit
  /// single-state `{{t}}` form (see `single_state`) returns that state
  /// bare rather than wrapped in a list.
  window_axes: u8,
  /// Whether `tspec` was written as the explicit single-state `{{t}}`
  /// form, as opposed to a form documented as always yielding "a list of
  /// the selected states" (bare `t`/`{t}`, or a `{{t1, t2[, dt]}}` range)
  /// that merely happens to resolve to one step.
  single_state: bool,
}

fn as_nonneg_int(expr: &Expr) -> Option<u128> {
  match expr {
    Expr::Integer(n) if *n >= 0 => Some(*n as u128),
    _ => None,
  }
}

/// Range part of a rule spec: r (1D) or {r1, r2} (2D).
enum RangeSpec {
  One(usize),
  Two(usize, usize),
}

/// Per-dimension neighborhood radius cap. Real rules use single-digit
/// ranges; this only guards against absurd specs allocating huge weight
/// grids.
const MAX_RANGE: usize = 256;

fn parse_range(expr: &Expr) -> Option<RangeSpec> {
  let radius = |e: &Expr| -> Option<usize> {
    let r = usize::try_from(as_nonneg_int(e)?).ok()?;
    (r <= MAX_RANGE).then_some(r)
  };
  match expr {
    Expr::Integer(_) => Some(RangeSpec::One(radius(expr)?)),
    Expr::List(items) if items.len() == 2 => {
      Some(RangeSpec::Two(radius(&items[0])?, radius(&items[1])?))
    }
    _ => None,
  }
}

fn parse_rule(expr: &Expr) -> Option<RuleSpec> {
  match expr {
    Expr::Integer(n) if *n >= 0 => Some(RuleSpec {
      n: *n as u128,
      k: 2,
      weights: vec![vec![4, 2, 1]],
      two_d: false,
    }),
    Expr::List(items) if items.len() == 2 || items.len() == 3 => {
      let n = as_nonneg_int(&items[0])?;
      let range = match items.get(2) {
        Some(r) => Some(parse_range(r)?),
        None => None,
      };
      match &items[1] {
        // {n, k[, r]} — general rule: the neighborhood read as a base-k
        // number, i.e. positional weights k^(cells-1), ..., k, 1.
        Expr::Integer(k) if *k >= 1 => {
          let k = *k as u128;
          let range = range.unwrap_or(RangeSpec::One(1));
          let two_d = matches!(range, RangeSpec::Two(..));
          let weights = positional_weights(k, &range)?;
          Some(RuleSpec {
            n,
            k,
            weights,
            two_d,
          })
        }
        // {n, {k, wspec}[, r]}
        Expr::List(kw) if kw.len() == 2 => {
          let k = as_nonneg_int(&kw[0])?;
          if k == 0 {
            return None;
          }
          let (weights, two_d) = parse_weights(&kw[1], range)?;
          Some(RuleSpec {
            n,
            k,
            weights,
            two_d,
          })
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Positional weights for a general (non-totalistic) rule: cell values read
/// row-major as a base-k number.
fn positional_weights(k: u128, range: &RangeSpec) -> Option<Vec<Vec<u128>>> {
  let (rows, cols) = match range {
    RangeSpec::One(r) => (1, 2 * r + 1),
    RangeSpec::Two(r1, r2) => (2 * r1 + 1, 2 * r2 + 1),
  };
  let cells = rows * cols;
  let mut weights = Vec::with_capacity(rows);
  for i in 0..rows {
    let mut row = Vec::with_capacity(cols);
    for j in 0..cols {
      let exp = u32::try_from(cells - 1 - (i * cols + j)).ok()?;
      row.push(k.checked_pow(exp)?);
    }
    weights.push(row);
  }
  Some(weights)
}

/// Weight part of a rule spec: a uniform weight, a 1D list, or a matrix.
/// Returns the weight grid and whether the rule is two-dimensional.
fn parse_weights(
  wspec: &Expr,
  range: Option<RangeSpec>,
) -> Option<(Vec<Vec<u128>>, bool)> {
  match wspec {
    // Uniform weight over the whole neighborhood (e.g. totalistic rules).
    Expr::Integer(w) if *w >= 0 => {
      let w = *w as u128;
      match range.unwrap_or(RangeSpec::One(1)) {
        RangeSpec::One(r) => Some((vec![vec![w; 2 * r + 1]], false)),
        RangeSpec::Two(r1, r2) => {
          Some((vec![vec![w; 2 * r2 + 1]; 2 * r1 + 1], true))
        }
      }
    }
    Expr::List(rows) if !rows.is_empty() => {
      if rows.iter().all(|e| matches!(e, Expr::Integer(_))) {
        // 1D weight list; must be odd-length and match an explicit range.
        let weights: Option<Vec<u128>> =
          rows.iter().map(as_nonneg_int).collect();
        let weights = weights?;
        if weights.len() % 2 == 0 {
          return None;
        }
        match range {
          None | Some(RangeSpec::One(_)) => {
            if let Some(RangeSpec::One(r)) = range
              && weights.len() != 2 * r + 1
            {
              return None;
            }
            Some((vec![weights], false))
          }
          Some(RangeSpec::Two(..)) => None,
        }
      } else {
        // 2D weight matrix; odd dimensions, matching an explicit range.
        let matrix: Option<Vec<Vec<u128>>> = rows
          .iter()
          .map(|row| match row {
            Expr::List(cells) if !cells.is_empty() => {
              cells.iter().map(as_nonneg_int).collect()
            }
            _ => None,
          })
          .collect();
        let matrix = matrix?;
        let cols = matrix[0].len();
        if matrix.iter().any(|r| r.len() != cols)
          || matrix.len() % 2 == 0
          || cols % 2 == 0
        {
          return None;
        }
        match range {
          None => Some((matrix, true)),
          Some(RangeSpec::Two(r1, r2))
            if matrix.len() == 2 * r1 + 1 && cols == 2 * r2 + 1 =>
          {
            Some((matrix, true))
          }
          _ => None,
        }
      }
    }
    _ => None,
  }
}

/// Upper bound on the number of returned states — far beyond any sensible
/// use, but keeps a typo like 10^12 steps from exhausting memory.
const MAX_STATES: usize = 100_000;

/// Parse a bare `tspec` — never a `{tspec, xspec[, yspec]}` window, which
/// only the top level of `parse_step_spec` accepts. Returns the resolved
/// times and whether `tspec` was written as the explicit single-state
/// `{{t}}` form (see `StepSpec::single_state`).
fn parse_tspec(expr: &Expr) -> Option<(Vec<usize>, bool)> {
  match expr {
    // t — all steps 0 through t.
    Expr::Integer(t) if *t >= 0 && (*t as u128) < MAX_STATES as u128 => {
      Some(((0..=(*t as usize)).collect(), false))
    }
    Expr::List(items) if items.len() == 1 => match &items[0] {
      // {t} — all steps 0 through t, identical to the bare `t` form.
      Expr::Integer(t) if *t >= 0 && (*t as u128) < MAX_STATES as u128 => {
        Some(((0..=(*t as usize)).collect(), false))
      }
      // {{t}} — the explicit single state t.
      Expr::List(ts) if ts.len() == 1 => {
        let t = usize::try_from(as_nonneg_int(&ts[0])?).ok()?;
        (t < MAX_STATES).then_some((vec![t], true))
      }
      // {{t1, t2}}, {{t1, t2, dt}} — a list of the selected states.
      Expr::List(ts) if ts.len() == 2 || ts.len() == 3 => {
        let vals: Option<Vec<u128>> = ts.iter().map(as_nonneg_int).collect();
        let vals = vals?;
        let t1 = usize::try_from(vals[0]).ok()?;
        let t2 = usize::try_from(*vals.get(1).unwrap_or(&vals[0])).ok()?;
        let dt = usize::try_from(*vals.get(2).unwrap_or(&1)).ok()?;
        if t2 < t1 || dt == 0 || (t2 - t1) / dt >= MAX_STATES {
          return None;
        }
        Some(((t1..=t2).step_by(dt).collect(), false))
      }
      _ => None,
    },
    _ => None,
  }
}

fn parse_step_spec(expr: &Expr) -> Option<StepSpec> {
  match expr {
    // {tspec, xspec} — the time steps of `tspec`, restricted to the cells
    // `xspec` names (1D rules only). `All` keeps every cell that could be
    // affected.
    Expr::List(items) if items.len() == 2 => {
      let (times, single_state) = parse_tspec(&items[0])?;
      Some(StepSpec {
        times,
        cells: parse_cell_spec(&items[1])?,
        rows: None,
        cols: None,
        window_axes: 1,
        single_state,
      })
    }
    // {tspec, xspec, yspec} — the time steps of `tspec`, restricted to the
    // rows `xspec` and columns `yspec` name (2D rules only).
    Expr::List(items) if items.len() == 3 => {
      let (times, single_state) = parse_tspec(&items[0])?;
      Some(StepSpec {
        times,
        cells: None,
        rows: parse_cell_spec(&items[1])?,
        cols: parse_cell_spec(&items[2])?,
        window_axes: 2,
        single_state,
      })
    }
    // A bare tspec, with no xspec/yspec window.
    _ => {
      let (times, single_state) = parse_tspec(expr)?;
      Some(StepSpec {
        times,
        cells: None,
        rows: None,
        cols: None,
        window_axes: 0,
        single_state,
      })
    }
  }
}

/// The cell offsets an `xspec` names, as `(from, to, step)`. `Ok(None)` means
/// "every cell" (`All`); a spec that cannot be read yields `None`.
#[allow(clippy::type_complexity)]
fn parse_cell_spec(expr: &Expr) -> Option<Option<(i64, i64, usize)>> {
  let as_int = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      _ => None,
    }
  };
  match expr {
    Expr::Identifier(s) if s == "All" => Some(None),
    // A bare `n` runs from the origin out to offset n (either direction).
    Expr::Integer(_) => {
      let n = as_int(expr)?;
      Some(Some(if n >= 0 { (0, n, 1) } else { (n, 0, 1) }))
    }
    Expr::List(items) if items.len() == 2 || items.len() == 3 => {
      let from = as_int(&items[0])?;
      let to = as_int(&items[1])?;
      let step = match items.get(2) {
        Some(d) => usize::try_from(as_int(d)?).ok().filter(|&d| d > 0)?,
        None => 1,
      };
      (to >= from).then_some(Some((from, to, step)))
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
fn rule_digit(n: u128, k: u128, s: u128) -> u128 {
  if k == 1 {
    return 0;
  }
  let Ok(exp) = u32::try_from(s) else {
    return 0;
  };
  match k.checked_pow(exp) {
    // k^s overflowing u128 implies k^s > n, so the digit is 0.
    Some(p) => (n / p) % k,
    None => 0,
  }
}

/// Extra background cells an axis needs beyond the rule's automatic growth
/// so that an explicit window reaching past the affected region still
/// evolves correctly. `base_origin` is where offset 0 of that axis sits on
/// the automatically-grown grid; `init_len` is the init's size along it.
fn axis_margin(
  background: Option<u128>,
  window: Option<(i64, i64, usize)>,
  base_origin: i64,
  init_len: usize,
) -> usize {
  match (background, window) {
    (Some(_), Some((from, to, _))) => {
      let far_edge = base_origin + init_len as i64 - 1 + base_origin;
      let need_near = (base_origin - from).max(0) - base_origin;
      let need_far = to - far_edge;
      need_near.max(need_far).max(0) as usize
    }
    _ => 0,
  }
}

/// Evolve `init` under `rule`, returning the states at the requested `times`
/// (which must be ascending). Background inits grow by the rule's range per
/// step and are jointly trimmed afterwards; cyclic inits keep their size.
/// `window` names the 1D cells (or, for a 2D rule, the columns) to keep;
/// `row_window` names the rows to keep for a 2D rule. `None` means every
/// cell that could be affected (automatic trimming for a background init,
/// the whole grid for a cyclic one).
fn evolve(
  rule: &RuleSpec,
  init: &[Vec<u128>],
  background: Option<u128>,
  times: &[usize],
  window: Option<(i64, i64, usize)>,
  row_window: Option<(i64, i64, usize)>,
) -> Option<Vec<Vec<Vec<u128>>>> {
  let r1 = (rule.weights.len() - 1) / 2;
  let r2 = (rule.weights[0].len() - 1) / 2;
  let t_max = *times.last()?;

  // Offset 0 sits at row/column `r * t_max` of a background grid (the init
  // is centered there) or at row/column 0 of a cyclic one.
  let base_row_origin = match background {
    Some(_) => (r1.checked_mul(t_max)?) as i64,
    None => 0,
  };
  let base_col_origin = match background {
    Some(_) => (r2.checked_mul(t_max)?) as i64,
    None => 0,
  };
  let row_margin =
    axis_margin(background, row_window, base_row_origin, init.len());
  let col_margin =
    axis_margin(background, window, base_col_origin, init[0].len());
  let row_origin = base_row_origin + row_margin as i64;
  let col_origin = base_col_origin + col_margin as i64;

  let (height, width) = match background {
    Some(_) => (
      init
        .len()
        .checked_add((2 * r1).checked_mul(t_max)?)?
        .checked_add(2usize.checked_mul(row_margin)?)?,
      init[0]
        .len()
        .checked_add((2 * r2).checked_mul(t_max)?)?
        .checked_add(2usize.checked_mul(col_margin)?)?,
    ),
    None => (init.len(), init[0].len()),
  };
  // Refuse absurd evolutions instead of hanging (grids beyond any plotting
  // use). The work bound covers cells x steps x neighborhood size; the
  // result bound covers the memory held by the returned states.
  let nbhd = rule.weights.len() * rule.weights[0].len();
  let cells = height.checked_mul(width)?;
  let work = cells.checked_mul(t_max.max(1))?.checked_mul(nbhd)?;
  let result_cells = cells.checked_mul(times.len())?;
  if cells > 4_000_000 || work > 1_000_000_000 || result_cells > 64_000_000 {
    return None;
  }

  // Place the init centered on a background-filled grid (background inits)
  // or use it verbatim (cyclic inits).
  let mut grid = match background {
    Some(bg) => {
      let mut grid = vec![vec![bg; width]; height];
      for (i, row) in init.iter().enumerate() {
        for (j, &cell) in row.iter().enumerate() {
          grid[row_origin as usize + i][col_origin as usize + j] = cell;
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
    .weights
    .iter()
    .flatten()
    .fold(0u128, |acc, &w| acc.saturating_add(w));

  let mut snapshots: Vec<(Vec<Vec<u128>>, u128)> = Vec::new();
  let mut next_time = 0;
  for t in 0..=t_max {
    if next_time < times.len() && times[next_time] == t {
      snapshots.push((grid.clone(), bg));
      next_time += 1;
    }
    if t == t_max {
      break;
    }

    let mut next = vec![vec![0u128; width]; height];
    for (x, next_row) in next.iter_mut().enumerate() {
      for (y, next_cell) in next_row.iter_mut().enumerate() {
        let mut s: u128 = 0;
        for (i, wrow) in rule.weights.iter().enumerate() {
          for (j, &w) in wrow.iter().enumerate() {
            let dx = x as i64 + i as i64 - r1 as i64;
            let dy = y as i64 + j as i64 - r2 as i64;
            let value = if background.is_some() {
              if dx < 0 || dy < 0 || dx >= height as i64 || dy >= width as i64 {
                bg
              } else {
                grid[dx as usize][dy as usize]
              }
            } else {
              let dx = dx.rem_euclid(height as i64) as usize;
              let dy = dy.rem_euclid(width as i64) as usize;
              grid[dx][dy]
            };
            s = s.saturating_add(w.saturating_mul(value));
          }
        }
        *next_cell = rule_digit(rule.n, rule.k, s);
      }
    }
    grid = next;
    bg = rule_digit(rule.n, rule.k, bg.saturating_mul(weight_total));
  }

  // Each axis independently keeps either its explicit window (wrapping
  // around for a cyclic init, reading the evolving background outside a
  // background grid) or, with no window named, every row/column that
  // could hold a non-background cell (a cyclic init keeps its full size,
  // since it has no background to trim against).
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
  let auto_axis = |len: usize, active: &dyn Fn(usize) -> bool| -> Vec<i64> {
    match background {
      Some(_) => {
        let top = (0..len).find(|&i| active(i)).unwrap_or(0);
        let bottom = (0..len).rfind(|&i| active(i)).unwrap_or(0);
        (top..=bottom).map(|i| i as i64).collect()
      }
      None => (0..len).map(|i| i as i64).collect(),
    }
  };

  // Bound the output size arithmetically before collecting either index
  // list: an explicit window's length isn't capped by `height`/`width`
  // (those stay at the small cyclic-init size regardless of window size),
  // so collecting first could allocate gigabytes before this check ever
  // ran.
  let axis_len =
    |window: Option<(i64, i64, usize)>, auto_len: usize| -> Option<usize> {
      match window {
        Some((from, to, step)) => {
          let span = usize::try_from(to.checked_sub(from)?).ok()?;
          span.checked_div(step)?.checked_add(1)
        }
        None => Some(auto_len),
      }
    };
  let windowed_cells = axis_len(row_window, height)?
    .checked_mul(axis_len(window, width)?)?
    .checked_mul(times.len())?;
  if windowed_cells > 64_000_000 {
    return None;
  }

  let row_indices: Vec<i64> = match row_window {
    Some((from, to, step)) => (from..=to)
      .step_by(step)
      .map(|off| row_origin + off)
      .collect(),
    None => auto_axis(height, &row_active),
  };
  let col_indices: Vec<i64> = match window {
    Some((from, to, step)) => (from..=to)
      .step_by(step)
      .map(|off| col_origin + off)
      .collect(),
    None => auto_axis(width, &col_active),
  };

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
