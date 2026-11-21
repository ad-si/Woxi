//! The legacy `PlotLegends`` package's `ShowLegend`, once loaded.
//!
//! `ShowLegend[graphic, legend1, legend2, …]` places one or more boxed
//! legends over (or beside) a graphic. The package lays everything out in a
//! frame where the graphic is centered at the origin and its longer side
//! runs from -1 to 1; `LegendPosition` names a legend's lower-left corner
//! and `LegendSize` its width and height in that same frame, so a legend
//! may sit inside the graphic or stick out past its edge (the picture then
//! grows to hold it).
//!
//! Each legend is either a list of `{icon, label}` items (an icon is any
//! graphic, drawn in a key box to the left of its label) or a color
//! gradient `{colorFunction, n, minLabel, maxLabel}`, both followed by
//! options: `LegendPosition` (default `{-1, -1}`), `LegendSize` (`0.8`),
//! `LegendShadow` (`{0.05, -0.05}`, or `None`), `LegendBorder` (a thin
//! black frame, or `False`/`None`/a color) and `LegendBackground`
//! (white).

use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;

/// The name of an option rule's left-hand side without its context, so
/// `PlotLegends`LegendSize` and a bare `LegendSize` read the same.
fn option_short_name(pattern: &Expr) -> Option<&str> {
  match pattern {
    Expr::Identifier(name) => Some(name.rsplit('`').next().unwrap_or(name)),
    _ => None,
  }
}

fn real(x: f64) -> Expr {
  Expr::Real(x)
}

fn point(x: f64, y: f64) -> Expr {
  Expr::List(vec![real(x), real(y)].into())
}

fn pair(e: &Expr) -> Option<(f64, f64)> {
  match e {
    Expr::List(items) if items.len() == 2 => {
      Some((try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?))
    }
    _ => None,
  }
}

fn is_none_like(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(s) if s == "None" || s == "False")
}

/// One legend's settings, in the package's frame.
struct LegendSpec {
  /// `{icon, label}` rows, top to bottom.
  items: Vec<(Expr, Expr)>,
  position: (f64, f64),
  size: (f64, f64),
  shadow: Option<(f64, f64)>,
  /// The border's directive, `None` for no border.
  border: Option<Expr>,
  background: Expr,
}

impl LegendSpec {
  /// The legend's corners, its shadow included.
  fn bounds(&self) -> ((f64, f64), (f64, f64)) {
    let (x0, y0) = self.position;
    let (x1, y1) = (x0 + self.size.0, y0 + self.size.1);
    let (dx, dy) = self.shadow.unwrap_or((0.0, 0.0));
    (
      (x0.min(x0 + dx), y0.min(y0 + dy)),
      (x1.max(x1 + dx), y1.max(y1 + dy)),
    )
  }

  /// The legend's primitives: shadow, background box, border, and each
  /// row's icon (fitted into a key box) followed by its label.
  fn primitives(&self) -> Vec<Expr> {
    let (x0, y0) = self.position;
    let (w, h) = self.size;
    let rect = |ax: f64, ay: f64| {
      call("Rectangle", vec![point(ax, ay), point(ax + w, ay + h)])
    };
    let mut out = Vec::new();
    if let Some((dx, dy)) = self.shadow {
      out.push(Expr::List(
        vec![call("GrayLevel", vec![real(0.0)]), rect(x0 + dx, y0 + dy)].into(),
      ));
    }
    let edge = match &self.border {
      Some(directive) => call("EdgeForm", vec![directive.clone()]),
      None => call("EdgeForm", vec![Expr::Identifier("None".into())]),
    };
    out.push(Expr::List(
      vec![self.background.clone(), edge, rect(x0, y0)].into(),
    ));
    let n = self.items.len().max(1) as f64;
    let row_h = h / n;
    // A square key box filling most of its row, never wider than a third
    // of the legend so the labels keep their room.
    let key = (0.8 * row_h).min(w / 3.0);
    let margin = (row_h - key) / 2.0;
    for (i, (icon, label)) in self.items.iter().enumerate() {
      let cy = y0 + h - (i as f64 + 0.5) * row_h;
      let kx = x0 + margin + key / 2.0;
      out.push(call(
        "Inset",
        vec![
          icon.clone(),
          point(kx, cy),
          Expr::Identifier("Center".into()),
          point(key, key),
        ],
      ));
      let is_empty_label = matches!(label, Expr::String(s) if s.is_empty());
      if !is_empty_label {
        out.push(call(
          "Text",
          vec![
            label.clone(),
            point(x0 + 2.0 * margin + key + 0.25 * margin, cy),
            point(-1.0, 0.0),
          ],
        ));
      }
    }
    out
  }
}

/// Read one legend argument: `{{{icon, label}, …}, opts…}` or
/// `{colorFunction, n, minLabel, maxLabel, opts…}`.
fn parse_legend(arg: &Expr) -> Option<LegendSpec> {
  let Expr::List(parts) = arg else {
    return None;
  };
  let is_option =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let content: Vec<&Expr> =
    parts.iter().take_while(|e| !is_option(e)).collect();
  let options = &parts[content.len()..];

  let items: Vec<(Expr, Expr)> = match content.as_slice() {
    [Expr::List(rows)] => rows
      .iter()
      .map(|row| match row {
        Expr::List(cells) if cells.len() == 2 => {
          Some((cells[0].clone(), cells[1].clone()))
        }
        _ => None,
      })
      .collect::<Option<_>>()?,
    [function, n, min_label, max_label] => {
      let n = match evaluate_expr_to_expr(n).ok()? {
        Expr::Integer(n) if n >= 2 => n as usize,
        _ => return None,
      };
      // The top row shows the function's value at 1, the bottom row at 0.
      (0..n)
        .rev()
        .map(|k| {
          let t = k as f64 / (n - 1) as f64;
          let color =
            evaluate_expr_to_expr(&call_value(function, real(t))).ok()?;
          let swatch = evaluate_expr_to_expr(&call(
            "Graphics",
            vec![Expr::List(vec![color, call("Rectangle", vec![])].into())],
          ))
          .ok()?;
          let label = if k == n - 1 {
            (*max_label).clone()
          } else if k == 0 {
            (*min_label).clone()
          } else {
            Expr::String(String::new())
          };
          Some((swatch, label))
        })
        .collect::<Option<_>>()?
    }
    _ => return None,
  };
  if items.is_empty() {
    return None;
  }

  let mut spec = LegendSpec {
    items,
    position: (-1.0, -1.0),
    size: (0.8, 0.8),
    shadow: Some((0.05, -0.05)),
    border: Some(call("GrayLevel", vec![real(0.0)])),
    background: call("GrayLevel", vec![real(1.0)]),
  };
  for option in options {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = option
    else {
      continue;
    };
    let value = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| (**replacement).clone());
    match option_short_name(pattern) {
      Some("LegendPosition") => {
        if let Some(p) = pair(&value) {
          spec.position = p;
        }
      }
      Some("LegendSize") => {
        if let Some(s) = pair(&value) {
          spec.size = s;
        } else if let Some(s) = try_eval_to_f64(&value) {
          spec.size = (s, s);
        }
      }
      Some("LegendShadow") => {
        spec.shadow = if is_none_like(&value) {
          None
        } else {
          pair(&value).or(spec.shadow)
        };
      }
      Some("LegendBorder") => {
        spec.border = if is_none_like(&value) {
          None
        } else if matches!(&value, Expr::Identifier(s) if s == "True" || s == "Automatic")
        {
          spec.border
        } else {
          Some(value)
        };
      }
      Some("LegendBackground") => {
        spec.background = if is_none_like(&value) {
          call("Opacity", vec![real(0.0)])
        } else {
          value
        };
      }
      _ => {}
    }
  }
  Some(spec)
}

/// `f[t]` for a function given as a symbol or a pure function.
fn call_value(function: &Expr, t: Expr) -> Expr {
  match function {
    Expr::Identifier(name) => call(name, vec![t]),
    other => Expr::CurriedCall {
      func: Box::new(other.clone()),
      args: vec![t],
    },
  }
}

/// `PlotLegends`ShowLegend[graphic, legend…]`.
pub fn show_legend_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("PlotLegends`ShowLegend", args);
  let Some((graphic, legend_args)) = args.split_first() else {
    return Ok(original());
  };
  // A plot arrives already rendered; a plain `Graphics[…]` stays symbolic
  // until it is shown, so render it to learn its size.
  let rendered = match graphic {
    Expr::FunctionCall { name, args }
      if name == "Graphics" && !args.is_empty() =>
    {
      crate::functions::graphics::graphics_ast(args)?
    }
    other => other.clone(),
  };
  let Expr::Graphics {
    svg, is_3d: false, ..
  } = &rendered
  else {
    return Ok(original());
  };
  let Some(dims) = crate::functions::graphics::parse_svg_dimensions(svg) else {
    return Ok(original());
  };
  let (nat_w, nat_h) = (dims.nat_w, dims.nat_h);
  let Some(legends) = legend_args
    .iter()
    .map(parse_legend)
    .collect::<Option<Vec<_>>>()
  else {
    return Ok(original());
  };
  if legends.is_empty() || nat_w <= 0.0 || nat_h <= 0.0 {
    return Ok(original());
  }

  // The graphic's longer side spans -1..1, in the shape of its picture.
  let ratio = nat_h / nat_w;
  let (pw, ph) = if ratio >= 1.0 {
    (2.0 / ratio, 2.0)
  } else {
    (2.0, 2.0 * ratio)
  };
  let (mut lo, mut hi) = ((-pw / 2.0, -ph / 2.0), (pw / 2.0, ph / 2.0));
  let mut primitives = vec![call(
    "Inset",
    vec![
      graphic.clone(),
      point(0.0, 0.0),
      Expr::Identifier("Center".into()),
      point(pw, ph),
    ],
  )];
  for legend in &legends {
    let ((ax, ay), (bx, by)) = legend.bounds();
    lo = (lo.0.min(ax), lo.1.min(ay));
    hi = (hi.0.max(bx), hi.1.max(by));
    primitives.extend(legend.primitives());
  }
  // The graphic keeps its own width; the picture grows around it.
  let image_width = (nat_w * (hi.0 - lo.0) / pw).round();
  let shown = call(
    "Graphics",
    vec![
      Expr::List(primitives.into()),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("PlotRange".into())),
        replacement: Box::new(Expr::List(
          vec![point(lo.0, hi.0), point(lo.1, hi.1)].into(),
        )),
      },
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("ImageSize".into())),
        replacement: Box::new(real(image_width)),
      },
    ],
  );
  evaluate_expr_to_expr(&shown)
}
