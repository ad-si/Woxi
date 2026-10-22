//! Complex-valued evolution PDEs with periodic boundaries for `NDSolve`.
//!
//! An equation first order in time such as the nonlinear Schrödinger
//! equation
//!
//! ```text
//! I D[u[t, x], t] + D[u[t, x], x, x] + 2 Abs[u[t, x]]^2 u[t, x] == g[x]
//! ```
//!
//! with a (possibly complex) initial profile and `u[t, xmin] == u[t, xmax]`
//! is solved by the method of lines: a Fourier pseudospectral
//! discretisation in space (FFT on a power-of-two grid) and classical RK4
//! in time. The right-hand side `u_t = R(u, u_x, u_xx, t, x)` is split into
//! real and imaginary parts once, symbolically, so the time loop only
//! evaluates compiled real expressions of `Re[u]` and `Im[u]`.

use super::*;

const VARS: [&str; 6] = [
  "NDSolve$P",
  "NDSolve$Q",
  "NDSolve$PX",
  "NDSolve$QX",
  "NDSolve$PXX",
  "NDSolve$QXX",
];

/// Finest spatial spacing the grid aims for (the default `MinStepSize`
/// the notebook-style option set requests is of this order).
const TARGET_SPACING: f64 = 0.25;
const MIN_POINTS: usize = 32;
const MAX_POINTS: usize = 4096;
/// Number of time rows kept in the output `InterpolatingFunction`.
const OUTPUT_ROWS: usize = 60;

pub(super) fn ndsolve_pde_complex(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Expr::Identifier(u_name) = &args[1] else {
    return Ok(None);
  };
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
  if eq_items.len() != 3 {
    return Ok(None);
  }
  for (t_dom, x_dom) in [(&dom_a, &dom_b), (&dom_b, &dom_a)] {
    for swap in [false, true] {
      if let Some(result) =
        try_solve(&eq_items, u_name, t_dom, x_dom, &dom_a, swap)?
      {
        return Ok(Some(result));
      }
    }
  }
  Ok(None)
}

/// `ComplexExpand[part[expr]]` for `part` = `Re` or `Im`, with every
/// symbol treated as real.
fn real_part_of(part: &str, expr: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&call(
    "ComplexExpand",
    vec![call(part, vec![expr.clone()])],
  ))
}

fn try_solve(
  eq_items: &[Expr],
  u_name: &str,
  t_dom: &PdeDomain,
  x_dom: &PdeDomain,
  dim0_dom: &PdeDomain,
  swap: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let mut ic: Option<Expr> = None;
  let mut periodic = false;
  let mut pde: Option<Expr> = None;
  for eq in eq_items {
    if !periodic
      && try_pde_periodic_boundary(
        eq,
        u_name,
        &t_dom.name,
        x_dom.min,
        x_dom.max,
        swap,
      )
    {
      periodic = true;
    } else if ic.is_none()
      && let Some(rhs) =
        try_pde_initial_condition(eq, u_name, &x_dom.name, t_dom.min, swap)
    {
      ic = Some(rhs);
    } else if pde.is_none() {
      pde = Some(eq.clone());
    } else {
      return Ok(None);
    }
  }
  let (true, Some(ic), Some(pde)) = (periodic, ic, pde) else {
    return Ok(None);
  };
  let Some((lhs, rhs)) = pde_split_equation(&pde) else {
    return Ok(None);
  };

  // Residual `lhs - rhs` with the unknown's values and derivatives replaced
  // by placeholders; only `u`, `u_x`, `u_xx` and `u_t` may occur.
  let residual = call(
    "Plus",
    vec![lhs, call("Times", vec![Expr::Integer(-1), rhs])],
  );
  let Some(rewritten) = rewrite_pde_rhs_hyperbolic(
    &residual,
    u_name,
    &t_dom.name,
    &x_dom.name,
    swap,
  ) else {
    return Ok(None);
  };
  const ALL: [&str; 9] = [
    "NDSolve$U",
    "NDSolve$UX",
    "NDSolve$UXX",
    "NDSolve$V",
    "NDSolve$VX",
    "NDSolve$VXX",
    "NDSolve$W",
    "NDSolve$WX",
    "NDSolve$WXX",
  ];
  if ALL[4..].iter().any(|n| expr_contains_ident(&rewritten, n)) {
    return Ok(None);
  }
  let coeff = differentiate_expr(&rewritten, "NDSolve$V")?;
  if ALL.iter().any(|n| expr_contains_ident(&coeff, n)) {
    return Ok(None);
  }
  let rest = crate::syntax::substitute_variables(
    &rewritten,
    &[("NDSolve$V", &Expr::Integer(0))],
  );
  // u_t = -rest / coeff
  let rate = call(
    "Times",
    vec![
      Expr::Integer(-1),
      rest,
      call("Power", vec![coeff, Expr::Integer(-1)]),
    ],
  );

  let complex = |re: &str, im: &str| {
    call(
      "Plus",
      vec![id_expr(re), call("Times", vec![id_expr("I"), id_expr(im)])],
    )
  };
  let (u, ux, uxx) = (
    complex(VARS[0], VARS[1]),
    complex(VARS[2], VARS[3]),
    complex(VARS[4], VARS[5]),
  );
  let rate = crate::syntax::substitute_variables(
    &rate,
    &[
      ("NDSolve$U", &u),
      ("NDSolve$UX", &ux),
      ("NDSolve$UXX", &uxx),
    ],
  );
  let var_names: Vec<String> = VARS
    .iter()
    .map(std::string::ToString::to_string)
    .chain([t_dom.name.clone(), x_dom.name.clone()])
    .collect();
  let rate_re_expr = real_part_of("Re", &rate)?;
  let rate_im_expr = real_part_of("Im", &rate)?;
  let ic_re_expr = real_part_of("Re", &ic)?;
  let ic_im_expr = real_part_of("Im", &ic)?;
  // A real profile under a real-valued rate stays real: the imaginary
  // component is then pinned at zero instead of drifting with FFT round-off.
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(v) if *v == 0.0)
  };
  let real_problem = is_zero(&ic_im_expr)
    && is_zero(&crate::evaluator::evaluate_expr_to_expr(
      &crate::syntax::substitute_variables(
        &rate_im_expr,
        &[
          (VARS[1], &Expr::Integer(0)),
          (VARS[3], &Expr::Integer(0)),
          (VARS[5], &Expr::Integer(0)),
        ],
      ),
    )?);
  let rate_re = NumFn::new(rate_re_expr, &var_names);
  let rate_im = NumFn::new(rate_im_expr, &var_names);
  let x_vars = [x_dom.name.clone()];
  let ic_re = NumFn::new(ic_re_expr, &x_vars);
  let ic_im = NumFn::new(ic_im_expr, &x_vars);

  // Periodic Fourier grid: the right end repeats the left one.
  let period = x_dom.max - x_dom.min;
  if period.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    || t_dom.max.partial_cmp(&t_dom.min) != Some(std::cmp::Ordering::Greater)
  {
    return Ok(None);
  }
  let n = ((period / TARGET_SPACING).ceil() as usize)
    .next_power_of_two()
    .clamp(MIN_POINTS, MAX_POINTS);
  let h = period / n as f64;
  let xs: Vec<f64> = (0..n).map(|j| x_dom.min + j as f64 * h).collect();

  let mut wr = Vec::with_capacity(n);
  let mut wi = Vec::with_capacity(n);
  for &x in &xs {
    wr.push(ic_re.eval(&[x])?);
    wi.push(ic_im.eval(&[x])?);
  }
  if wr.iter().chain(&wi).any(|v| !v.is_finite()) {
    return Ok(None);
  }

  let k_scale = 2.0 * std::f64::consts::PI / period;
  let wavenumber = |m: usize| -> f64 {
    let m = if m <= n / 2 {
      m as f64
    } else {
      m as f64 - n as f64
    };
    m * k_scale
  };
  let k_max = k_scale * (n / 2) as f64;

  let derivative = |t: f64,
                    re: &[f64],
                    im: &[f64]|
   -> Result<(Vec<f64>, Vec<f64>), InterpreterError> {
    let (mut fr, mut fi) = (re.to_vec(), im.to_vec());
    fft(&mut fr, &mut fi, false);
    let (mut x1r, mut x1i) = (vec![0.0; n], vec![0.0; n]);
    let (mut x2r, mut x2i) = (vec![0.0; n], vec![0.0; n]);
    for m in 0..n {
      let k = wavenumber(m);
      // i k (a + i b) = -k b + i k a; the Nyquist mode has no first derivative.
      if m != n / 2 {
        x1r[m] = -k * fi[m];
        x1i[m] = k * fr[m];
      }
      x2r[m] = -k * k * fr[m];
      x2i[m] = -k * k * fi[m];
    }
    fft(&mut x1r, &mut x1i, true);
    fft(&mut x2r, &mut x2i, true);
    let mut dr = vec![0.0; n];
    let mut di = vec![0.0; n];
    let mut vars = [0.0; 8];
    for j in 0..n {
      vars[..6]
        .copy_from_slice(&[re[j], im[j], x1r[j], x1i[j], x2r[j], x2i[j]]);
      vars[6] = t;
      vars[7] = xs[j];
      dr[j] = rate_re.eval(&vars)?;
      di[j] = if real_problem {
        0.0
      } else {
        rate_im.eval(&vars)?
      };
    }
    Ok((dr, di))
  };

  let span = t_dom.max - t_dom.min;
  let n_steps =
    ((span * k_max * k_max / 2.0).ceil() as usize).clamp(20, 400_000);
  let dt = span / n_steps as f64;
  let stride = (n_steps / OUTPUT_ROWS).max(1);

  let mut rows: Vec<(f64, Vec<f64>, Vec<f64>)> =
    vec![(t_dom.min, wr.clone(), wi.clone())];
  let mut t = t_dom.min;
  for step in 0..n_steps {
    let axpy = |a: &[f64], k: &[f64], c: f64| -> Vec<f64> {
      a.iter().zip(k).map(|(y, d)| y + c * d).collect()
    };
    let (k1r, k1i) = derivative(t, &wr, &wi)?;
    let (k2r, k2i) = derivative(
      t + 0.5 * dt,
      &axpy(&wr, &k1r, 0.5 * dt),
      &axpy(&wi, &k1i, 0.5 * dt),
    )?;
    let (k3r, k3i) = derivative(
      t + 0.5 * dt,
      &axpy(&wr, &k2r, 0.5 * dt),
      &axpy(&wi, &k2i, 0.5 * dt),
    )?;
    let (k4r, k4i) =
      derivative(t + dt, &axpy(&wr, &k3r, dt), &axpy(&wi, &k3i, dt))?;
    for j in 0..n {
      wr[j] += dt / 6.0 * (k1r[j] + 2.0 * k2r[j] + 2.0 * k3r[j] + k4r[j]);
      wi[j] += dt / 6.0 * (k1i[j] + 2.0 * k2i[j] + 2.0 * k3i[j] + k4i[j]);
    }
    t += dt;
    if wr.iter().chain(&wi).any(|v| !v.is_finite()) {
      return Ok(None);
    }
    if (step + 1) % stride == 0 || step + 1 == n_steps {
      let t_row = if step + 1 == n_steps { t_dom.max } else { t };
      rows.push((t_row, wr.clone(), wi.clone()));
    }
  }

  // `[time][space]` table of complex values, the right end repeating the
  // left one; transposed when the space variable was declared first.
  let ts: Vec<f64> = rows.iter().map(|r| r.0).collect();
  let mut xs_out = xs.clone();
  xs_out.push(x_dom.max);
  let table: Vec<Vec<(f64, f64)>> = rows
    .iter()
    .map(|(_, re, im)| {
      let mut row: Vec<(f64, f64)> =
        re.iter().copied().zip(im.iter().copied()).collect();
      row.push((re[0], im[0]));
      row
    })
    .collect();
  let t_is_dim0 = t_dom.name == dim0_dom.name;
  let (dim0, dim1) = if t_is_dim0 {
    (t_dom, x_dom)
  } else {
    (x_dom, t_dom)
  };
  let (c0, c1, grid): (&Vec<f64>, &Vec<f64>, Vec<Vec<(f64, f64)>>) =
    if t_is_dim0 {
      (&ts, &xs_out, table)
    } else {
      let cols = table[0].len();
      let transposed = (0..cols)
        .map(|c| table.iter().map(|row| row[c]).collect())
        .collect();
      (&xs_out, &ts, transposed)
    };
  let reals =
    |v: &[f64]| Expr::List(v.iter().map(|x| Expr::Real(*x)).collect());
  let domain = Expr::List(
    vec![
      Expr::List(vec![Expr::Real(dim0.min), Expr::Real(dim0.max)].into()),
      Expr::List(vec![Expr::Real(dim1.min), Expr::Real(dim1.max)].into()),
    ]
    .into(),
  );
  // One real-valued `InterpolatingFunction` per component: a complex table
  // would make every sample copy thousands of nested `a + b I` cells.
  let component = |pick: fn(&(f64, f64)) -> f64| {
    call(
      "InterpolatingFunction",
      vec![
        domain.clone(),
        Expr::List(
          grid
            .iter()
            .map(|row| {
              Expr::List(row.iter().map(|c| Expr::Real(pick(c))).collect())
            })
            .collect(),
        ),
        Expr::List(vec![Expr::Integer(3), Expr::Integer(3)].into()),
        Expr::List(vec![reals(c0), reals(c1)].into()),
      ],
    )
  };
  let solution = if grid.iter().flatten().all(|c| c.1 == 0.0) {
    component(|c| c.0)
  } else {
    let apply = |f: Expr| Expr::CurriedCall {
      func: Box::new(f),
      args: vec![id_expr(&dim0.name), id_expr(&dim1.name)],
    };
    Expr::NamedFunction {
      params: vec![dim0.name.clone(), dim1.name.clone()],
      body: Box::new(call(
        "Plus",
        vec![
          apply(component(|c| c.0)),
          call("Times", vec![id_expr("I"), apply(component(|c| c.1))]),
        ],
      )),
      bracketed: true,
    }
  };
  Ok(Some(Expr::List(
    vec![Expr::List(
      vec![rule_expr(Expr::Identifier(u_name.to_string()), solution)].into(),
    )]
    .into(),
  )))
}

/// In-place radix-2 FFT of a power-of-two length sequence; the inverse
/// transform includes the `1/n` normalisation.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
  let n = re.len();
  let mut j = 0;
  for i in 1..n {
    let mut bit = n >> 1;
    while j & bit != 0 {
      j ^= bit;
      bit >>= 1;
    }
    j |= bit;
    if i < j {
      re.swap(i, j);
      im.swap(i, j);
    }
  }
  let sign = if inverse { 1.0 } else { -1.0 };
  let mut len = 2;
  while len <= n {
    let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
    let (wr, wi) = (ang.cos(), ang.sin());
    for start in (0..n).step_by(len) {
      let (mut cr, mut ci) = (1.0, 0.0);
      for k in 0..len / 2 {
        let (a, b) = (start + k, start + k + len / 2);
        let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        let next = (cr * wr - ci * wi, cr * wi + ci * wr);
        cr = next.0;
        ci = next.1;
      }
    }
    len <<= 1;
  }
  if inverse {
    let scale = 1.0 / n as f64;
    for v in re.iter_mut().chain(im.iter_mut()) {
      *v *= scale;
    }
  }
}
