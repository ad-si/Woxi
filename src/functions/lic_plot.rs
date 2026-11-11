//! `LineIntegralConvolutionPlot[{vx, vy}, {x, …}, {y, …}]` (or with a
//! `{{vx, vy}, {"noise", w, h}}` texture spec): white noise smeared along
//! the field's streamlines — each pixel the average of the noise over a
//! short streamline through it — shaded by the color function, as a
//! `Graphics` of one `Raster`.

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

fn id(name: &str) -> Expr {
  Expr::Identifier(name.to_string())
}

/// Default texture size, as in wolframscript.
const DEFAULT_SIZE: usize = 240;

/// One component of the field, compiled when it compiles.
struct Component {
  body: Expr,
  compiled: Option<crate::functions::polynomial_ast::solve::NumNode>,
  vars: [String; 2],
}

impl Component {
  fn new(body: &Expr, vars: &[String; 2]) -> Component {
    let compiled = crate::functions::polynomial_ast::solve::compile_numeric(
      body,
      vars.as_slice(),
    )
    .filter(|n| !n.has_fallback());
    Component {
      body: body.clone(),
      compiled,
      vars: vars.clone(),
    }
  }

  fn at(&self, x: f64, y: f64) -> f64 {
    if let Some(c) = &self.compiled {
      return c.eval(&[x, y]);
    }
    let e = crate::syntax::substitute_variable(
      &self.body,
      &self.vars[0],
      &Expr::Real(x),
    );
    let e =
      crate::syntax::substitute_variable(&e, &self.vars[1], &Expr::Real(y));
    evaluate_expr_to_expr(&e)
      .ok()
      .and_then(|v| try_eval_to_f64(&v))
      .unwrap_or(f64::NAN)
  }
}

/// A deterministic pseudo-random noise value in `[0, 1)` for pixel `i`.
fn noise(i: usize) -> f64 {
  let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^= z >> 31;
  (z >> 11) as f64 / (1u64 << 53) as f64
}

/// `{v, lo, hi}`: the variable, the bounds and the bounds as given
/// (exact bounds stay exact in the raster's corners).
fn iterator(spec: &Expr) -> Option<(String, f64, f64, [Expr; 2])> {
  let Expr::List(items) = spec else {
    return None;
  };
  let [Expr::Identifier(v), lo, hi] = items.as_slice() else {
    return None;
  };
  let lo_expr = evaluate_expr_to_expr(lo).ok()?;
  let hi_expr = evaluate_expr_to_expr(hi).ok()?;
  let lo = try_eval_to_f64(&lo_expr)?;
  let hi = try_eval_to_f64(&hi_expr)?;
  (hi > lo).then(|| (v.clone(), lo, hi, [lo_expr, hi_expr]))
}

pub fn line_integral_convolution_plot_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  if args.len() < 3 {
    return Ok(None);
  }
  let (Some((xv, x0, x1, xb)), Some((yv, y0, y1, yb))) =
    (iterator(&args[1]), iterator(&args[2]))
  else {
    return Ok(None);
  };
  let vars = [xv.clone(), yv.clone()];
  let names = [xv.as_str(), yv.as_str()];
  let spec = crate::functions::plot::eval_body_vars_symbolic(&args[0], &names);
  // `{{vx, vy}, texture}` or plain `{vx, vy}`.
  let (field, texture) = match &spec {
    Expr::List(items)
      if items.len() == 2
        && matches!(&items[0], Expr::List(v) if v.len() == 2) =>
    {
      (items[0].clone(), Some(items[1].clone()))
    }
    other => (other.clone(), None),
  };
  let Expr::List(components) = &field else {
    return Ok(None);
  };
  if components.len() != 2 {
    return Ok(None);
  }
  let (mut w, mut h) = match &texture {
    Some(Expr::List(t))
      if t.len() == 3 && matches!(&t[0], Expr::String(s) if s == "noise") =>
    {
      // A square texture of the first size (wolframscript's raster for
      // `{"noise", 50, 40}` is 50 × 50 too).
      match try_eval_to_f64(&t[1]) {
        Some(m) if m >= 2.0 => ((m as usize).min(1000), (m as usize).min(1000)),
        _ => (DEFAULT_SIZE, DEFAULT_SIZE),
      }
    }
    _ => (DEFAULT_SIZE, DEFAULT_SIZE),
  };
  let vx = Component::new(&components[0], &vars);
  let vy = Component::new(&components[1], &vars);

  let mut color_function = None;
  let mut scaling = true;
  let mut scale_px: Option<f64> = None;
  let mut options = Vec::new();
  for opt in &args[3..] {
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
    let value = || {
      evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.as_ref().clone())
    };
    match pattern.as_ref() {
      Expr::Identifier(k) if k == "ColorFunction" => {
        let v = value();
        if !matches!(&v, Expr::Identifier(a) if a == "Automatic") {
          color_function = Some(v);
        }
      }
      Expr::Identifier(k) if k == "ColorFunctionScaling" => {
        scaling = !matches!(value(), Expr::Identifier(ref v) if v == "False");
      }
      Expr::Identifier(k) if k == "LineIntegralConvolutionScale" => {
        scale_px = try_eval_to_f64(&value()).filter(|s| *s >= 0.0);
      }
      // `RasterSize -> n` (or `{nx, ny}`) is the size of the noise raster
      // the convolution smears, unless an explicit texture sets it.
      Expr::Identifier(k) if k == "RasterSize" => {
        let size = |e: &Expr| {
          try_eval_to_f64(e)
            .filter(|n| *n >= 2.0)
            .map(|n| (n as usize).min(1000))
        };
        if texture.is_none() {
          match &value() {
            Expr::List(ns) if ns.len() == 2 => {
              if let (Some(a), Some(b)) = (size(&ns[0]), size(&ns[1])) {
                (w, h) = (a, b);
              }
            }
            other => {
              if let Some(n) = size(other) {
                (w, h) = (n, n);
              }
            }
          }
        }
      }
      Expr::Identifier(k)
        if matches!(
          k.as_str(),
          "LightingAngle"
            | "PerformanceGoal"
            | "RegionFunction"
            | "VectorScale"
            | "VectorPoints"
            | "VectorStyle"
            | "StreamPoints"
        ) => {}
      _ => options.push(opt.clone()),
    }
  }

  // The field at every pixel center, rows from the bottom up.
  let px = (x1 - x0) / w as f64;
  let py = (y1 - y0) / h as f64;
  let mut field_at = vec![(f64::NAN, f64::NAN); w * h];
  for r in 0..h {
    let y = y0 + (r as f64 + 0.5) * py;
    for c in 0..w {
      let x = x0 + (c as f64 + 0.5) * px;
      field_at[r * w + c] = (vx.at(x, y), vy.at(x, y));
    }
  }
  let norms: Vec<f64> = field_at.iter().map(|(a, b)| a.hypot(*b)).collect();

  // Line integral convolution: average the noise along the streamline
  // through each pixel, half a pixel per step in either direction.
  // `LineIntegralConvolutionScale -> s` scales the default streamline
  // length (`0` leaves the noise unsmeared).
  let half_length = (w.max(h) as f64 / 16.0).max(4.0) * scale_px.unwrap_or(1.0);
  let steps = (2.0 * half_length).round() as usize;
  let lookup = |fx: f64, fy: f64| -> Option<usize> {
    if fx < 0.0 || fy < 0.0 || fx >= w as f64 || fy >= h as f64 {
      return None;
    }
    Some(fy as usize * w + fx as usize)
  };
  let mut texture_values = vec![0.0; w * h];
  for r in 0..h {
    for c in 0..w {
      let start = r * w + c;
      let mut sum = noise(start);
      let mut count = 1.0;
      for dir in [1.0, -1.0] {
        let (mut fx, mut fy) = (c as f64 + 0.5, r as f64 + 0.5);
        let mut at = start;
        for _ in 0..steps {
          let (u, v) = field_at[at];
          let len = u.hypot(v);
          if len.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            || !len.is_finite()
          {
            break;
          }
          // The direction in pixel units (the axes may be scaled apart).
          let (du, dv) = (u / px, v / py);
          let dl = du.hypot(dv);
          fx += dir * 0.5 * du / dl;
          fy += dir * 0.5 * dv / dl;
          match lookup(fx, fy) {
            Some(next) => {
              at = next;
              sum += noise(next);
              count += 1.0;
            }
            None => break,
          }
        }
      }
      texture_values[start] = sum / count;
    }
  }
  // Stretch the averaged noise (clustered about 1/2) to the full range.
  let mean = texture_values.iter().sum::<f64>() / texture_values.len() as f64;
  let sd = (texture_values
    .iter()
    .map(|t| (t - mean).powi(2))
    .sum::<f64>()
    / texture_values.len() as f64)
    .sqrt()
    .max(1e-9);
  let shade = |t: f64| (0.5 + (t - mean) / (4.0 * sd)).clamp(0.0, 1.0);

  // Colors from the color function, applied on a coarse grid (a general
  // function of x, y, vx, vy and the norm costs an evaluation each).
  let (nlo, nhi) = norms
    .iter()
    .filter(|n| n.is_finite())
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &n| {
      (a.min(n), b.max(n))
    });
  let scaled = |n: f64| {
    if nhi > nlo {
      ((n - nlo) / (nhi - nlo)).clamp(0.0, 1.0)
    } else {
      0.5
    }
  };
  const COARSE: usize = 128;
  let (gw, gh) = (w.min(COARSE), h.min(COARSE));
  let named = color_function.as_ref().and_then(color_function_scheme_name);
  let mut grid_colors = vec![(1.0, 1.0, 1.0); gw * gh];
  for gr in 0..gh {
    for gc in 0..gw {
      let (r, c) = ((gr * h + h / 2) / gh, (gc * w + w / 2) / gw);
      let i = r.min(h - 1) * w + c.min(w - 1);
      let (u, v) = field_at[i];
      let n = norms[i];
      let x = x0 + (c as f64 + 0.5) * px;
      let y = y0 + (r as f64 + 0.5) * py;
      let rgb = match (&color_function, &named) {
        (_, Some(name)) => {
          let (r, g, b) = apply_named_color_function(name, scaled(n));
          (r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
        }
        (Some(f), None) => {
          let s = |v: f64, lo: f64, hi: f64| {
            if scaling && hi > lo {
              (v - lo) / (hi - lo)
            } else {
              v
            }
          };
          let applied = Expr::CurriedCall {
            func: Box::new(f.clone()),
            args: vec![
              Expr::Real(s(x, x0, x1)),
              Expr::Real(s(y, y0, y1)),
              Expr::Real(u),
              Expr::Real(v),
              Expr::Real(if scaling { scaled(n) } else { n }),
            ],
          };
          evaluate_expr_to_expr(&applied)
            .ok()
            .and_then(|c| {
              crate::functions::graphics::parse_color_function_result(&c)
            })
            .map_or((1.0, 1.0, 1.0), |c| (c.r, c.g, c.b))
        }
        // Wolfram's default colors the field from brown (weak) to cream
        // (strong); `"SiennaTones"` over 0.4..0.8 of the rescaled norm
        // follows its gradient to within a few percent.
        (None, None) => {
          let (r, g, b) =
            apply_named_color_function("SiennaTones", 0.4 + 0.4 * scaled(n));
          (r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
        }
      };
      grid_colors[gr * gw + gc] = rgb;
    }
  }

  let data: Vec<Expr> = (0..h)
    .map(|r| {
      let gr = (r * gh / h).min(gh - 1);
      Expr::List(
        (0..w)
          .map(|c| {
            let gc = (c * gw / w).min(gw - 1);
            let (cr, cg, cb) = grid_colors[gr * gw + gc];
            let t = 0.35 + 0.65 * shade(texture_values[r * w + c]);
            Expr::List(
              vec![Expr::Real(cr * t), Expr::Real(cg * t), Expr::Real(cb * t)]
                .into(),
            )
          })
          .collect(),
      )
    })
    .collect();
  let corner =
    |x: &Expr, y: &Expr| Expr::List(vec![x.clone(), y.clone()].into());
  let raster = call(
    "Raster",
    vec![
      Expr::List(data.into()),
      Expr::List(vec![corner(&xb[0], &yb[0]), corner(&xb[1], &yb[1])].into()),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
      rule("ColorFunction", id("RGBColor")),
    ],
  );
  let given = |name: &str| {
    options.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(k) if k == name))
    })
  };
  let mut defaults = Vec::new();
  if !given("Frame") {
    defaults.push(rule("Frame", id("True")));
  }
  if !given("AspectRatio") {
    defaults.push(rule("AspectRatio", Expr::Integer(1)));
  }
  defaults.push(rule(
    "PlotRange",
    Expr::List(
      vec![
        Expr::List(xb.to_vec().into()),
        Expr::List(yb.to_vec().into()),
      ]
      .into(),
    ),
  ));
  let mut graphics_args = vec![Expr::List(vec![raster].into())];
  graphics_args.extend(options);
  graphics_args.extend(defaults);
  Ok(Some(evaluate_expr_to_expr(&call(
    "Graphics",
    graphics_args,
  ))?))
}
