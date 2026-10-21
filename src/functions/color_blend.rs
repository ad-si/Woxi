//! `Blend` over lists of color directives, and `ToColor`.
//!
//! Colors that all share one head are blended component by component in
//! their own space — `Hue` stays `Hue`, `CMYKColor` stays `CMYKColor` — and a
//! list that mixes heads is converted to `RGBColor` first. An alpha channel
//! is carried through when any color has one, the others counting as opaque.
//!
//! The arithmetic follows Wolfram's exactly:
//! * `Blend[{c1, …, cn}, t]` places the colors evenly on `[0, 1]`, clamps
//!   `t`, and interpolates the segment it lands in as `a + u (b - a)`, which
//!   for machine numbers is the single fused multiply-add `fma(u, b - a, a)`.
//! * `Blend[{c1, …, cn}, {w1, …, wn}]` and `Blend[{c1, …, cn}]` take the
//!   weighted mean `Σ (wi / Σw) ci`.
//! * `Blend[{{p1, c1}, …}, t]` interpolates between explicit positions.
//!
//! A hue outside `[-1, 1]` is reduced to its fractional part before
//! blending, so `Hue[1.3]` blends as `Hue[0.3]` while `Hue[1]` stays 1.
//!
//! Exact inputs give exact results (`Blend[{Red, Blue}, 1/3]` is
//! `RGBColor[2/3, 0, 1/3]`), evaluated symbolically.

use crate::syntax::Expr;

/// A color directive split into its head and components, plus its alpha.
#[derive(Clone)]
struct Color {
  head: String,
  comps: Vec<Expr>,
  alpha: Option<Expr>,
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn eval(e: Expr) -> Expr {
  crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
}

fn is_number(e: &Expr) -> bool {
  crate::functions::math_ast::try_eval_to_f64(e).is_some()
    && !matches!(e, Expr::String(_))
}

/// The number of color components each supported head carries, not counting
/// alpha.
fn component_count(head: &str) -> Option<usize> {
  match head {
    "GrayLevel" => Some(1),
    "RGBColor" | "Hue" | "XYZColor" | "LABColor" | "LCHColor" | "LUVColor" => {
      Some(3)
    }
    "CMYKColor" => Some(4),
    _ => None,
  }
}

fn parse_color(e: &Expr) -> Option<Color> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  let n = component_count(name)?;
  if !args.iter().all(is_number) {
    return None;
  }
  let mut comps: Vec<Expr> = args.iter().take(n).cloned().collect();
  match name.as_str() {
    // `Hue[h]` is full saturation and brightness.
    "Hue" if comps.len() == 1 => {
      comps.push(Expr::Integer(1));
      comps.push(Expr::Integer(1));
    }
    _ if comps.len() != n => return None,
    _ => {}
  }
  if name == "Hue" {
    comps[0] = reduce_hue(&comps[0]);
  }
  Some(Color {
    head: name.clone(),
    comps,
    alpha: args.get(n).cloned(),
  })
}

/// A hue beyond one full turn keeps only its fractional part.
fn reduce_hue(h: &Expr) -> Expr {
  match h {
    Expr::Real(v) if v.abs() > 1.0 => Expr::Real(v - v.trunc()),
    Expr::Real(_) => h.clone(),
    _ => match crate::functions::math_ast::try_eval_to_f64(h) {
      Some(v) if v.abs() > 1.0 => eval(call("FractionalPart", vec![h.clone()])),
      _ => h.clone(),
    },
  }
}

fn to_rgb(c: &Color) -> Option<Color> {
  match c.head.as_str() {
    "RGBColor" => Some(c.clone()),
    "GrayLevel" => Some(Color {
      head: "RGBColor".into(),
      comps: vec![c.comps[0].clone(); 3],
      alpha: c.alpha.clone(),
    }),
    _ => {
      let mut args = c.comps.clone();
      args.extend(c.alpha.clone());
      let converted = eval(call(
        "ColorConvert",
        vec![call(&c.head, args), Expr::String("RGB".into())],
      ));
      parse_color(&converted).filter(|rgb| rgb.head == "RGBColor")
    }
  }
}

/// Bring every color into one space: their shared head, or RGB.
fn common_space(colors: &[Color]) -> Option<Vec<Color>> {
  if colors.iter().all(|c| c.head == colors[0].head) {
    return Some(colors.to_vec());
  }
  colors.iter().map(to_rgb).collect()
}

fn build(head: &str, comps: Vec<Expr>, alpha: Option<Expr>) -> Expr {
  let mut args = comps;
  args.extend(alpha);
  call(head, args)
}

fn alpha_or_one(c: &Color) -> Expr {
  c.alpha.clone().unwrap_or(Expr::Integer(1))
}

/// `a + u (b - a)`: a fused multiply-add for machine numbers, symbolic
/// otherwise.
fn lerp(a: &Expr, b: &Expr, u: &Expr) -> Expr {
  let machine = [a, b, u].iter().any(|e| matches!(e, Expr::Real(_)));
  if machine {
    let f = |e: &Expr| crate::functions::math_ast::try_eval_to_f64(e);
    if let (Some(a), Some(b), Some(u)) = (f(a), f(b), f(u)) {
      return Expr::Real(u.mul_add(b - a, a));
    }
  }
  eval(call(
    "Plus",
    vec![
      a.clone(),
      call(
        "Times",
        vec![
          u.clone(),
          call(
            "Plus",
            vec![b.clone(), call("Times", vec![Expr::Integer(-1), a.clone()])],
          ),
        ],
      ),
    ],
  ))
}

fn lerp_colors(a: &Color, b: &Color, u: &Expr) -> Expr {
  let comps = a
    .comps
    .iter()
    .zip(&b.comps)
    .map(|(x, y)| lerp(x, y, u))
    .collect();
  let alpha = (a.alpha.is_some() || b.alpha.is_some())
    .then(|| lerp(&alpha_or_one(a), &alpha_or_one(b), u));
  build(&a.head, comps, alpha)
}

/// `Σ wi ci` for weights that already sum to one.
fn weighted_sum(colors: &[Color], weights: &[Expr]) -> Expr {
  let sum = |pick: &dyn Fn(&Color) -> Expr| {
    eval(call(
      "Plus",
      colors
        .iter()
        .zip(weights)
        .map(|(c, w)| call("Times", vec![w.clone(), pick(c)]))
        .collect(),
    ))
  };
  let comps = (0..colors[0].comps.len())
    .map(|i| sum(&|c: &Color| c.comps[i].clone()))
    .collect();
  let alpha = colors
    .iter()
    .any(|c| c.alpha.is_some())
    .then(|| sum(&alpha_or_one));
  build(&colors[0].head, comps, alpha)
}

fn emit_not_colors(list: &Expr) {
  crate::emit_message(&format!(
    "Blend::arg: {} is not a valid list of colors or images, or pairs of a \
     real number and a color or an image.",
    crate::syntax::expr_to_string(list)
  ));
}

fn emit_bad_weights(weights: &Expr, list: &Expr) {
  crate::emit_message(&format!(
    "Blend::argl: {} should be a real number or a list of non-negative \
     numbers that has the same length as {}.",
    crate::syntax::expr_to_string(weights),
    crate::syntax::expr_to_string(list)
  ));
}

/// `t` clamped to `[0, 1]`, keeping its exactness.
fn clamp_unit(t: &Expr, tf: f64) -> Expr {
  let bound = |v: i128| {
    if matches!(t, Expr::Real(_)) {
      Expr::Real(v as f64)
    } else {
      Expr::Integer(v)
    }
  };
  if tf < 0.0 {
    bound(0)
  } else if tf > 1.0 {
    bound(1)
  } else {
    t.clone()
  }
}

/// `Blend` of a list of color directives (`items`), with the optional
/// second argument. `None` leaves the call unevaluated.
pub fn blend_color_list(items: &[Expr], spec: Option<&Expr>) -> Option<Expr> {
  let list = Expr::List(items.to_vec().into());
  if items.len() < 2 {
    return None;
  }
  // Positioned form: {{p1, c1}, {p2, c2}, …}.
  if let Some(t) = spec
    && items
      .iter()
      .all(|it| matches!(it, Expr::List(p) if p.len() == 2))
  {
    return blend_positioned(items, t, &list);
  }
  let Some(colors) = items.iter().map(parse_color).collect::<Option<Vec<_>>>()
  else {
    // Symbolic entries (an unset variable, a pending Dynamic) are left
    // alone; only something that can never be a color is reported.
    if items.iter().any(|it| {
      matches!(it, Expr::Integer(_) | Expr::Real(_) | Expr::String(_))
        || matches!(it, Expr::Identifier(_))
    }) {
      emit_not_colors(&list);
    }
    return None;
  };
  let colors = common_space(&colors)?;
  let n = colors.len();
  match spec {
    None => {
      let w = eval(call(
        "Rational",
        vec![Expr::Integer(1), Expr::Integer(n as i128)],
      ));
      Some(weighted_sum(&colors, &vec![w; n]))
    }
    Some(Expr::List(weights)) => {
      if weights.len() != n || !weights.iter().all(is_number) {
        emit_bad_weights(spec.unwrap(), &list);
        return None;
      }
      let total = eval(call("Plus", weights.to_vec()));
      let normalized: Vec<Expr> = weights
        .iter()
        .map(|w| {
          eval(call(
            "Times",
            vec![
              w.clone(),
              call("Power", vec![total.clone(), Expr::Integer(-1)]),
            ],
          ))
        })
        .collect();
      Some(weighted_sum(&colors, &normalized))
    }
    Some(t) => {
      let Some(tf) = crate::functions::math_ast::try_eval_to_f64(t)
        .filter(|_| !matches!(t, Expr::String(_)))
      else {
        emit_bad_weights(t, &list);
        return None;
      };
      let t = clamp_unit(t, tf);
      let segments = (n - 1) as i128;
      // Position along the list, the segment it falls in, and the offset
      // within that segment.
      let pos = eval(call("Times", vec![t, Expr::Integer(segments)]));
      let posf = crate::functions::math_ast::try_eval_to_f64(&pos)?;
      let seg = (posf.floor().max(0.0) as usize).min(n - 2);
      let u = if seg == 0 {
        pos
      } else if let Expr::Real(p) = pos {
        Expr::Real(p - seg as f64)
      } else {
        eval(call("Plus", vec![pos, Expr::Integer(-(seg as i128))]))
      };
      Some(lerp_colors(&colors[seg], &colors[seg + 1], &u))
    }
  }
}

fn blend_positioned(items: &[Expr], t: &Expr, list: &Expr) -> Option<Expr> {
  let mut positions = Vec::with_capacity(items.len());
  let mut colors = Vec::with_capacity(items.len());
  for it in items {
    let Expr::List(pair) = it else { return None };
    let p = crate::functions::math_ast::try_eval_to_f64(&pair[0])?;
    let Some(c) = parse_color(&pair[1]) else {
      emit_not_colors(list);
      return None;
    };
    positions.push((pair[0].clone(), p));
    colors.push(c);
  }
  let colors = common_space(&colors)?;
  let Some(tf) = crate::functions::math_ast::try_eval_to_f64(t) else {
    emit_bad_weights(t, list);
    return None;
  };
  let n = colors.len();
  let whole = |c: &Color| build(&c.head, c.comps.clone(), c.alpha.clone());
  if tf <= positions[0].1 {
    return Some(whole(&colors[0]));
  }
  if tf >= positions[n - 1].1 {
    return Some(whole(&colors[n - 1]));
  }
  let seg = (0..n - 1)
    .find(|&i| tf >= positions[i].1 && tf <= positions[i + 1].1)
    .unwrap_or(0);
  let (p0, p1) = (&positions[seg].0, &positions[seg + 1].0);
  let u = eval(call(
    "Times",
    vec![
      call(
        "Plus",
        vec![
          t.clone(),
          call("Times", vec![Expr::Integer(-1), p0.clone()]),
        ],
      ),
      call(
        "Power",
        vec![
          call(
            "Plus",
            vec![
              p1.clone(),
              call("Times", vec![Expr::Integer(-1), p0.clone()]),
            ],
          ),
          Expr::Integer(-1),
        ],
      ),
    ],
  ));
  Some(lerp_colors(&colors[seg], &colors[seg + 1], &u))
}

/// `ToColor[color, head]` — `color` converted to the color space `head`
/// names.
pub fn to_color(args: &[Expr]) -> Option<Expr> {
  let [color, target] = args else { return None };
  let Expr::Identifier(head) = target else {
    return None;
  };
  let space = match head.as_str() {
    "RGBColor" => "RGB",
    "GrayLevel" => "Grayscale",
    "Hue" => "HSB",
    "CMYKColor" => "CMYK",
    "XYZColor" => "XYZ",
    "LABColor" => "LAB",
    "LCHColor" => "LCH",
    "LUVColor" => "LUV",
    _ => {
      crate::emit_message(&format!(
        "ToColor::colconv: {head} should be one of RGBColor, Hue, CMYKColor \
         or GrayLevel."
      ));
      return None;
    }
  };
  let Some(c) = parse_color(color) else {
    if is_number(color) || matches!(color, Expr::String(_)) {
      crate::emit_message(&format!(
        "ToColor::tocol: First argument {} in ToColor should be a valid \
         color directive.",
        crate::syntax::expr_to_string(color)
      ));
    }
    return None;
  };
  if c.head == *head {
    return Some(color.clone());
  }
  // GrayLevel widens to RGB exactly, as Wolfram does.
  if c.head == "GrayLevel" && head == "RGBColor" {
    return to_rgb(&c).map(|rgb| build("RGBColor", rgb.comps, rgb.alpha));
  }
  let converted = eval(call(
    "ColorConvert",
    vec![color.clone(), Expr::String(space.into())],
  ));
  match &converted {
    Expr::FunctionCall { name, .. } if name == head => Some(converted),
    _ => None,
  }
}
