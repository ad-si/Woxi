//! Evenly spaced streamlines of a 2D vector field, as `StreamPlot` draws
//! them: the field is sampled once on a grid and interpolated, and
//! streamlines are grown from seeds placed a fixed distance beside the ones
//! already drawn (Jobard & Lefer, "Creating evenly-spaced streamlines of
//! arbitrary density", 1997), so they fill the plot at an even spacing
//! instead of bunching up where the flow converges.

/// A vector field sampled at `n × n` nodes spanning the plot domain and
/// bilinearly interpolated between them. A node outside the field's domain
/// (or its `RegionFunction`) holds `None`, and so does any point next to it.
pub(crate) struct SampledField {
  x_range: (f64, f64),
  y_range: (f64, f64),
  n: usize,
  /// `values[i * n + j]` is the field at the `i`-th x and `j`-th y node.
  values: Vec<Option<(f64, f64)>>,
}

impl SampledField {
  pub(crate) fn sample(
    field: impl Fn(f64, f64) -> Option<(f64, f64)>,
    x_range: (f64, f64),
    y_range: (f64, f64),
    n: usize,
  ) -> Self {
    let n = n.max(2);
    let node = |range: (f64, f64), k: usize| {
      range.0 + (range.1 - range.0) * k as f64 / (n - 1) as f64
    };
    let values = (0..n)
      .flat_map(|i| (0..n).map(move |j| (i, j)))
      .map(|(i, j)| {
        field(node(x_range, i), node(y_range, j))
          .filter(|(vx, vy)| vx.is_finite() && vy.is_finite())
      })
      .collect();
    Self {
      x_range,
      y_range,
      n,
      values,
    }
  }

  /// The field at `(x, y)`, interpolated from the four surrounding nodes.
  pub(crate) fn at(&self, x: f64, y: f64) -> Option<(f64, f64)> {
    let cell = |v: f64, (lo, hi): (f64, f64)| {
      let t = (v - lo) / (hi - lo) * (self.n - 1) as f64;
      if !(-1e-9..=(self.n - 1) as f64 + 1e-9).contains(&t) {
        return None;
      }
      let k = (t.floor() as usize).min(self.n - 2);
      Some((k, t - k as f64))
    };
    let (i, fx) = cell(x, self.x_range)?;
    let (j, fy) = cell(y, self.y_range)?;
    let node = |a: usize, b: usize| self.values[a * self.n + b];
    let (v00, v10, v01, v11) = (
      node(i, j)?,
      node(i + 1, j)?,
      node(i, j + 1)?,
      node(i + 1, j + 1)?,
    );
    let mix = |a: f64, b: f64, c: f64, d: f64| {
      a * (1.0 - fx) * (1.0 - fy)
        + b * fx * (1.0 - fy)
        + c * (1.0 - fx) * fy
        + d * fx * fy
    };
    Some((
      mix(v00.0, v10.0, v01.0, v11.0),
      mix(v00.1, v10.1, v01.1, v11.1),
    ))
  }
}

/// A point already on a streamline: its display position, its streamline,
/// and its index along that streamline.
#[derive(Clone, Copy)]
struct Placed {
  u: f64,
  v: f64,
  line: usize,
  index: i64,
}

/// Placed points bucketed into square cells of the separation distance, so
/// a proximity test only looks at the neighbouring cells.
struct Occupancy {
  cell: f64,
  cols: usize,
  rows: usize,
  cells: Vec<Vec<Placed>>,
}

impl Occupancy {
  fn new(width: f64, height: f64, cell: f64) -> Self {
    let cols = (width / cell).ceil() as usize + 1;
    let rows = (height / cell).ceil() as usize + 1;
    Self {
      cell,
      cols,
      rows,
      cells: vec![Vec::new(); cols * rows],
    }
  }

  fn key(&self, u: f64, v: f64) -> (usize, usize) {
    (
      ((u / self.cell).max(0.0) as usize).min(self.cols - 1),
      ((v / self.cell).max(0.0) as usize).min(self.rows - 1),
    )
  }

  fn insert(&mut self, p: Placed) {
    let (c, r) = self.key(p.u, p.v);
    self.cells[r * self.cols + c].push(p);
  }

  /// Whether a placed point lies within `dist` of `(u, v)`, ignoring the
  /// points of streamline `own` within `skip` steps of `index` (the stretch
  /// of the line being traced that leads up to this point).
  fn crowded(
    &self,
    u: f64,
    v: f64,
    dist: f64,
    own: Option<(usize, i64, i64)>,
  ) -> bool {
    let (c, r) = self.key(u, v);
    let span = (dist / self.cell).ceil() as usize;
    for rr in r.saturating_sub(span)..=(r + span).min(self.rows - 1) {
      for cc in c.saturating_sub(span)..=(c + span).min(self.cols - 1) {
        for p in &self.cells[rr * self.cols + cc] {
          if let Some((line, index, skip)) = own
            && p.line == line
            && (p.index - index).abs() <= skip
          {
            continue;
          }
          if (p.u - u).powi(2) + (p.v - v).powi(2) < dist * dist {
            return true;
          }
        }
      }
    }
    false
  }
}

/// Evenly spaced streamlines of `field`, as point lists in data coordinates.
///
/// Spacing is measured as Wolfram measures it, in the plot domain scaled
/// to the unit square: `separation` is the distance between neighbouring
/// streamlines as a fraction of each axis' range.
pub(crate) fn even_streamlines(
  field: &SampledField,
  separation: f64,
) -> Vec<Vec<(f64, f64)>> {
  let (x_min, x_max) = field.x_range;
  let (y_min, y_max) = field.y_range;
  let (xr, yr) = (x_max - x_min, y_max - y_min);
  // Normalized coordinates: u and v run over [0, 1] across the x and y
  // ranges.
  let to_data = |u: f64, v: f64| (x_min + u * xr, y_min + v * yr);
  let in_bounds =
    |u: f64, v: f64| (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v);
  // The unit direction of the flow at a normalized point.
  let direction = |u: f64, v: f64| -> Option<(f64, f64)> {
    let (x, y) = to_data(u, v);
    let (vx, vy) = field.at(x, y)?;
    let (du, dv) = (vx / xr, vy / yr);
    let speed = (du * du + dv * dv).sqrt();
    (speed > 1e-12).then(|| (du / speed, dv / speed))
  };

  let d_sep = separation;
  let d_test = 0.5 * d_sep;
  let step = 0.1 * d_sep;
  // A streamline may come back within `d_test` of itself only further along
  // than this many steps — closer is the stretch it has just drawn.
  let skip = (2.0 * d_sep / step).ceil() as i64;
  let max_steps = (16.0 / step) as usize;
  let mut grid = Occupancy::new(1.0, 1.0, d_sep);

  // Grow streamline `line` from `seed` in one direction (`sign` = ±1),
  // placing each point as it goes; stops at the border, where the flow
  // stalls, or on coming too close to another streamline.
  let trace =
    |grid: &mut Occupancy, line: usize, seed: (f64, f64), sign: f64| {
      let mut points = Vec::new();
      let (mut u, mut v) = seed;
      for k in 1..=max_steps {
        let Some((du, dv)) = direction(u, v) else {
          break;
        };
        let (mu, mv) = (u + sign * du * step / 2.0, v + sign * dv * step / 2.0);
        let Some((du, dv)) = direction(mu, mv) else {
          break;
        };
        let (nu, nv) = (u + sign * du * step, v + sign * dv * step);
        let index = sign as i64 * k as i64;
        if !in_bounds(nu, nv)
          || grid.crowded(nu, nv, d_test, Some((line, index, skip)))
        {
          break;
        }
        grid.insert(Placed {
          u: nu,
          v: nv,
          line,
          index,
        });
        points.push((nu, nv));
        (u, v) = (nu, nv);
      }
      points
    };

  let mut lines: Vec<Vec<(f64, f64)>> = Vec::new();
  // Try to start a streamline at `seed`; returns its index when one was
  // drawn. Seeds are placed exactly one separation from a streamline, so
  // the test allows a hair less than that to survive rounding.
  let grow = |grid: &mut Occupancy,
              lines: &mut Vec<Vec<(f64, f64)>>,
              seed: (f64, f64)| {
    if !in_bounds(seed.0, seed.1)
      || direction(seed.0, seed.1).is_none()
      || grid.crowded(seed.0, seed.1, 0.99 * d_sep, None)
    {
      return None;
    }
    let line = lines.len();
    grid.insert(Placed {
      u: seed.0,
      v: seed.1,
      line,
      index: 0,
    });
    let forward = trace(grid, line, seed, 1.0);
    let backward = trace(grid, line, seed, -1.0);
    let mut points: Vec<(f64, f64)> = backward.into_iter().rev().collect();
    points.push(seed);
    points.extend(forward);
    lines.push(points);
    Some(line)
  };

  // New seeds sit one separation to either side of a drawn streamline;
  // once those run out, a scan of the whole plot picks up any region the
  // drawn streamlines never reach.
  let every = ((d_sep / step).round() as usize).max(1);
  let mut queue = std::collections::VecDeque::new();
  let scan_cols = (1.0 / d_sep).ceil() as usize;
  let scan_rows = scan_cols;
  let mut scan = (0..scan_rows)
    .flat_map(|r| (0..scan_cols).map(move |c| (c, r)))
    .map(|(c, r)| ((c as f64 + 0.5) * d_sep, (r as f64 + 0.5) * d_sep));
  let mut next_seed = Some((0.5, 0.5));
  loop {
    if let Some(seed) = next_seed.take()
      && let Some(line) = grow(&mut grid, &mut lines, seed)
    {
      queue.push_back(line);
    }
    if let Some(line) = queue.pop_front() {
      let points = lines[line].clone();
      for k in (0..points.len()).step_by(every) {
        let (u, v) = points[k];
        let (a, b) = (
          points[k.saturating_sub(1)],
          points[(k + 1).min(points.len() - 1)],
        );
        let (tu, tv) = (b.0 - a.0, b.1 - a.1);
        let len = (tu * tu + tv * tv).sqrt();
        if len < 1e-12 {
          continue;
        }
        let (nu, nv) = (-tv / len * d_sep, tu / len * d_sep);
        for seed in [(u + nu, v + nv), (u - nu, v - nv)] {
          if let Some(new_line) = grow(&mut grid, &mut lines, seed) {
            queue.push_back(new_line);
          }
        }
      }
      continue;
    }
    match scan.next() {
      Some(seed) => next_seed = Some(seed),
      None => break,
    }
  }

  lines
    .into_iter()
    .filter(|points| points.len() > 1)
    .map(|points| points.into_iter().map(|(u, v)| to_data(u, v)).collect())
    .collect()
}
