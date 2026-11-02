//! `JuliaSetPlot[c]` and `JuliaSetPlot[f, z]`: the filled Julia set of
//! `z^2 + c` (or of the polynomial `f`) by escape time, as a `Graphics` of
//! one `Raster`. Points whose orbit stays within the escape radius for the
//! whole iteration budget are black; the others are coloured by how many
//! steps they took to leave, through the same palette as wolframscript.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::field_plot::{
  apply_named_color_function, color_function_scheme_name,
};
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn rule(key: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(key.to_string())),
    replacement: Box::new(value),
  }
}

/// Escape iterations: a point still bounded after this many steps is in the
/// filled Julia set.
const MAX_ITERATIONS: usize = 50;

/// Escape radius of the quadratic family.
const ESCAPE_RADIUS: f64 = 3.0;

/// Raster columns (or rows, for a domain taller than wide).
const RESOLUTION: usize = 501;

/// wolframscript's colours for escape after `n` steps, `n = 1, 2, …`
/// (from blue through purple to orange).
const PALETTE: [(f64, f64, f64); 49] = [
  (0.4065135971340173, 0.5398078488845466, 0.7788308093144439),
  (0.4133528741247354, 0.5186060902133204, 0.7566031590946101),
  (0.41986647125875265, 0.49841393909786685, 0.735433968409054),
  (0.42638006839276993, 0.4782217879824133, 0.7142647777234978),
  (0.4328936655267872, 0.4580296368669598, 0.6930955870379417),
  (0.4397, 0.4368, 0.6709),
  (0.4462, 0.4166, 0.6497),
  (0.45276013678553984, 0.3964435759648266, 0.6285295554469956),
  (0.4593, 0.3763, 0.6074),
  (0.4661, 0.3550, 0.5851),
  (0.4726, 0.3349, 0.5640),
  (0.47914020517830974, 0.3146653639472399, 0.5427943331704934),
  (0.4857, 0.2945, 0.5216),
  (0.4925, 0.2733, 0.4994),
  (0.4990, 0.2531, 0.4782),
  (0.5165608207132388, 0.2556306790425012, 0.459267220322423),
  (0.5371, 0.2626, 0.4398),
  (0.5566, 0.2693, 0.4212),
  (0.5762, 0.2759, 0.4026),
  (0.5957010258915486, 0.2825383488031265, 0.38408402540302883),
  (0.6162, 0.2895, 0.3646),
  (0.6358, 0.2962, 0.3460),
  (0.6553, 0.3028, 0.3275),
  (0.6748412310698584, 0.3094460185637518, 0.3089008304836346),
  (0.6954, 0.3164, 0.2894),
  (0.7149, 0.3231, 0.2708),
  (0.7344, 0.3297, 0.2523),
  (0.753981436248168, 0.3363536883243771, 0.2337176355642403),
  (0.7745, 0.3433, 0.2142),
  (0.7940, 0.3500, 0.1957),
  (0.8136, 0.3566, 0.1771),
  (0.8340986809965802, 0.36359355153883727, 0.1576062530532487),
  (0.8536, 0.3702, 0.1390),
  (0.8732, 0.3769, 0.1205),
  (0.8927, 0.3835, 0.1019),
  (0.9132388861748901, 0.39050122129946263, 0.08242305813385442),
  (0.9328, 0.3971, 0.0639),
  (0.9523, 0.4038, 0.0453),
  (0.9719, 0.4104, 0.0267),
  (
    0.9923790913531998,
    0.41740889106008794,
    0.007239863214460143,
  ),
  (1.0, 0.4298, 0.0),
  (1.0, 0.4460, 0.0),
  (1.0, 0.4621, 0.0),
  (1.0, 0.4790034196384953, 0.0),
  (1.0, 0.5282, 0.0),
  (1.0, 0.5389, 0.0),
  (1.0, 0.5497, 0.0),
  (1.0, 0.5604, 0.0),
  (1.0, 0.6910, 0.0),
];

/// A complex number as `(re, im)`, when `e` evaluates to one.
fn complex_value(e: &Expr) -> Option<(f64, f64)> {
  let parts = evaluate_expr_to_expr(&call(
    "N",
    vec![Expr::List(
      vec![call("Re", vec![e.clone()]), call("Im", vec![e.clone()])].into(),
    )],
  ))
  .ok()?;
  match &parts {
    Expr::List(p) if p.len() == 2 => {
      Some((try_eval_to_f64(&p[0])?, try_eval_to_f64(&p[1])?))
    }
    _ => None,
  }
}

/// The polynomial's coefficients, constant term first.
fn polynomial_coefficients(f: &Expr, z: &Expr) -> Option<Vec<(f64, f64)>> {
  let list =
    evaluate_expr_to_expr(&call("CoefficientList", vec![f.clone(), z.clone()]))
      .ok()?;
  let Expr::List(items) = &list else {
    return None;
  };
  let coeffs: Option<Vec<(f64, f64)>> =
    items.iter().map(complex_value).collect();
  let mut coeffs = coeffs?;
  while coeffs.len() > 1
    && coeffs.last().is_some_and(|c| c.0 == 0.0 && c.1 == 0.0)
  {
    coeffs.pop();
  }
  (coeffs.len() >= 3).then_some(coeffs)
}

/// Horner evaluation of the polynomial at `(x, y)`.
fn horner(coeffs: &[(f64, f64)], (x, y): (f64, f64)) -> (f64, f64) {
  let mut acc = (0.0, 0.0);
  for &(cr, ci) in coeffs.iter().rev() {
    acc = (acc.0 * x - acc.1 * y + cr, acc.0 * y + acc.1 * x + ci);
  }
  acc
}

/// Steps until the orbit of `z0` leaves the disk of radius `radius`, or
/// `None` when it stays inside for the whole budget.
fn escape_time(
  coeffs: &[(f64, f64)],
  z0: (f64, f64),
  radius: f64,
) -> Option<usize> {
  let r2 = radius * radius;
  let mut z = z0;
  for n in 0..MAX_ITERATIONS {
    if z.0 * z.0 + z.1 * z.1 > r2 {
      return Some(n);
    }
    z = horner(coeffs, z);
  }
  None
}

/// A radius outside of which every orbit escapes: for `a_d z^d + …` with
/// `|z| > max(1, 2 (1 + Σ|a_k|) / |a_d|)` the leading term dominates.
fn escape_radius(coeffs: &[(f64, f64)]) -> f64 {
  let d = coeffs.len() - 1;
  let lead = coeffs[d].0.hypot(coeffs[d].1);
  if d == 2 && coeffs[1] == (0.0, 0.0) && coeffs[2] == (1.0, 0.0) {
    return ESCAPE_RADIUS.max(2.0 * coeffs[0].0.hypot(coeffs[0].1));
  }
  let rest: f64 = coeffs[..d].iter().map(|c| c.0.hypot(c.1)).sum();
  ESCAPE_RADIUS.max(2.0 * (1.0 + rest) / lead)
}

/// The bounding box of the filled Julia set, widened by 5% of its
/// diagonal, from a coarse escape-time scan.
fn automatic_range(
  coeffs: &[(f64, f64)],
  radius: f64,
) -> ((f64, f64), (f64, f64)) {
  const N: usize = 300;
  let r = radius.min(4.0);
  let (mut x0, mut x1, mut y0, mut y1) = (
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
  );
  for i in 0..N {
    let y = -r + 2.0 * r * (i as f64 + 0.5) / N as f64;
    for j in 0..N {
      let x = -r + 2.0 * r * (j as f64 + 0.5) / N as f64;
      if escape_time(coeffs, (x, y), radius).is_none() {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
      }
    }
  }
  if !(x0 <= x1 && y0 <= y1) {
    return ((-2.0, 2.0), (-2.0, 2.0));
  }
  let pad = 0.05 * (x1 - x0).hypot(y1 - y0) + 2.0 * r / N as f64;
  ((x0 - pad, x1 + pad), (y0 - pad, y1 + pad))
}

fn number_pair(e: &Expr) -> Option<(f64, f64)> {
  match e {
    Expr::List(p) if p.len() == 2 => {
      Some((try_eval_to_f64(&p[0])?, try_eval_to_f64(&p[1])?))
    }
    _ => None,
  }
}

pub fn julia_set_plot_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let is_option =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let positional = args.iter().take_while(|a| !is_option(a)).count();
  let coeffs = match positional {
    1 => complex_value(&args[0]).map(|c| vec![c, (0.0, 0.0), (1.0, 0.0)]),
    2 => polynomial_coefficients(&args[0], &args[1]),
    _ => None,
  };
  let Some(coeffs) = coeffs else {
    return Ok(None);
  };
  let radius = escape_radius(&coeffs);

  let mut range = None;
  let mut colors: Option<String> = None;
  let mut graphics_opts = Vec::new();
  for opt in &args[positional..] {
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
    let value = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| replacement.as_ref().clone());
    match pattern.as_ref() {
      Expr::Identifier(k) if k == "PlotRange" => {
        range = match &value {
          Expr::List(p) if p.len() == 2 => {
            match (number_pair(&p[0]), number_pair(&p[1])) {
              (Some(x), Some(y)) => Some((x, y)),
              _ => None,
            }
          }
          other => try_eval_to_f64(other)
            .filter(|r| *r > 0.0)
            .map(|r| ((-r, r), (-r, r))),
        };
        graphics_opts.push(opt.clone());
      }
      Expr::Identifier(k) if k == "ColorFunction" => {
        colors = color_function_scheme_name(&value);
      }
      Expr::Identifier(k)
        if matches!(
          k.as_str(),
          "MaxIterations" | "PerformanceGoal" | "Method" | "ImageResolution"
        ) => {}
      _ => graphics_opts.push(opt.clone()),
    }
  }
  let ((x0, x1), (y0, y1)) =
    range.unwrap_or_else(|| automatic_range(&coeffs, radius));
  if !(x1 > x0 && y1 > y0) {
    return Ok(None);
  }
  let (cols, rows) = if x1 - x0 >= y1 - y0 {
    let rows = ((RESOLUTION as f64) * (y1 - y0) / (x1 - x0)).round() as usize;
    (RESOLUTION, rows.max(2))
  } else {
    let cols = ((RESOLUTION as f64) * (x1 - x0) / (y1 - y0)).round() as usize;
    (cols.max(2), RESOLUTION)
  };
  let color_of = |n: Option<usize>| -> (f64, f64, f64) {
    match n {
      None => (0.0, 0.0, 0.0),
      Some(n) => match &colors {
        Some(name) => {
          let t = (n.max(1) - 1) as f64 / (MAX_ITERATIONS - 1) as f64;
          let (r, g, b) = apply_named_color_function(name, t);
          (r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
        }
        None => PALETTE[n.clamp(1, PALETTE.len()) - 1],
      },
    }
  };
  // Rows from the bottom up, each pixel sampled at its centre.
  let data: Vec<Expr> = (0..rows)
    .map(|i| {
      let y = y0 + (y1 - y0) * (i as f64 + 0.5) / rows as f64;
      Expr::List(
        (0..cols)
          .map(|j| {
            let x = x0 + (x1 - x0) * (j as f64 + 0.5) / cols as f64;
            let (r, g, b) = color_of(escape_time(&coeffs, (x, y), radius));
            Expr::List(vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into())
          })
          .collect(),
      )
    })
    .collect();
  let corner =
    |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  let raster = call(
    "Raster",
    vec![
      Expr::List(data.into()),
      Expr::List(vec![corner(x0, y0), corner(x1, y1)].into()),
    ],
  );
  let mut graphics_args = vec![raster];
  let sets_frame = graphics_opts.iter().any(|o| {
    matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(k) if k == "Frame"))
  });
  graphics_args.extend(graphics_opts);
  if !sets_frame {
    graphics_args.push(rule("Frame", Expr::Identifier("True".into())));
  }
  Ok(Some(evaluate_expr_to_expr(&call(
    "Graphics",
    graphics_args,
  ))?))
}
