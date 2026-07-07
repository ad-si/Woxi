use crate::InterpreterError;
use crate::syntax::Expr;

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
/// Step specifications: t (steps 0..t), {t} (the step-t state alone),
/// {{t}}, {{t1, t2}} and {{t1, t2, dt}} (a list of the selected states).
pub fn cellular_automaton_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "CellularAutomaton".to_string(),
      args: args.to_vec().into(),
    })
  };

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

  let Some(states) = evolve(&rule, &init, background, &steps.times) else {
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

  if steps.single {
    Ok(exprs.remove(0))
  } else {
    Ok(Expr::List(exprs.into()))
  }
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
  /// True for the `{t}` form, which returns the single state unwrapped.
  single: bool,
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
          let weights = positional_weights(k, range)?;
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
fn positional_weights(k: u128, range: RangeSpec) -> Option<Vec<Vec<u128>>> {
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

fn parse_step_spec(expr: &Expr) -> Option<StepSpec> {
  match expr {
    // t — all steps 0 through t.
    Expr::Integer(t) if *t >= 0 && (*t as u128) < MAX_STATES as u128 => {
      Some(StepSpec {
        times: (0..=(*t as usize)).collect(),
        single: false,
      })
    }
    Expr::List(items) if items.len() == 1 => match &items[0] {
      // {t} — the state at step t alone.
      Expr::Integer(t) if *t >= 0 => Some(StepSpec {
        times: vec![usize::try_from(*t).ok()?],
        single: true,
      }),
      // {{t}}, {{t1, t2}}, {{t1, t2, dt}} — a list of the selected states.
      Expr::List(ts) if !ts.is_empty() && ts.len() <= 3 => {
        let vals: Option<Vec<u128>> = ts.iter().map(as_nonneg_int).collect();
        let vals = vals?;
        let t1 = usize::try_from(vals[0]).ok()?;
        let t2 = usize::try_from(*vals.get(1).unwrap_or(&vals[0])).ok()?;
        let dt = usize::try_from(*vals.get(2).unwrap_or(&1)).ok()?;
        if t2 < t1 || dt == 0 || (t2 - t1) / dt >= MAX_STATES {
          return None;
        }
        Some(StepSpec {
          times: (t1..=t2).step_by(dt).collect(),
          single: false,
        })
      }
      _ => None,
    },
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

/// Evolve `init` under `rule`, returning the states at the requested `times`
/// (which must be ascending). Background inits grow by the rule's range per
/// step and are jointly trimmed afterwards; cyclic inits keep their size.
fn evolve(
  rule: &RuleSpec,
  init: &[Vec<u128>],
  background: Option<u128>,
  times: &[usize],
) -> Option<Vec<Vec<Vec<u128>>>> {
  let r1 = (rule.weights.len() - 1) / 2;
  let r2 = (rule.weights[0].len() - 1) / 2;
  let t_max = *times.last()?;

  let (height, width) = match background {
    Some(_) => (
      init.len().checked_add((2 * r1).checked_mul(t_max)?)?,
      init[0].len().checked_add((2 * r2).checked_mul(t_max)?)?,
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
          grid[r1 * t_max + i][r2 * t_max + j] = cell;
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
            let value = match background {
              Some(_) => {
                if dx < 0 || dy < 0 || dx >= height as i64 || dy >= width as i64
                {
                  bg
                } else {
                  grid[dx as usize][dy as usize]
                }
              }
              None => {
                let dx = dx.rem_euclid(height as i64) as usize;
                let dy = dy.rem_euclid(width as i64) as usize;
                grid[dx][dy]
              }
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

  if background.is_some() {
    Some(trim_background(&snapshots))
  } else {
    Some(snapshots.into_iter().map(|(g, _)| g).collect())
  }
}

/// Trim rows and columns that hold only background cells in every returned
/// state, keeping all states the same (jointly trimmed) size.
fn trim_background(
  snapshots: &[(Vec<Vec<u128>>, u128)],
) -> Vec<Vec<Vec<u128>>> {
  let height = snapshots[0].0.len();
  let width = snapshots[0].0[0].len();

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

  let top = (0..height).find(|&r| row_active(r)).unwrap_or(0);
  let bottom = (0..height).rfind(|&r| row_active(r)).unwrap_or(0);
  let left = (0..width).find(|&c| col_active(c)).unwrap_or(0);
  let right = (0..width).rfind(|&c| col_active(c)).unwrap_or(0);

  snapshots
    .iter()
    .map(|(g, _)| {
      g[top..=bottom]
        .iter()
        .map(|row| row[left..=right].to_vec())
        .collect()
    })
    .collect()
}
