//! AASTriangle / ASATriangle / SASTriangle — triangle constructors that
//! return an explicit `Triangle[{{0,0}, {c,0}, {cx,cy}}]`, sharing the
//! placement wolframscript uses: vertex A at the origin, side c along the
//! positive x axis.
//!
//! Angles must be numeric (exact or real); symbolic angles would need
//! wolframscript's trig canonicalization (`AASTriangle[a, Pi/3, 1]` gives
//! `Cos[a - Pi/6] Csc[a]` forms) and are left unevaluated. Symbolic side
//! lengths pass through fine.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, expr_to_output, unevaluated};

fn fc(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn eval(e: Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&e)
}

/// The numeric value of an angle argument, when it has one.
fn angle_value(e: &Expr) -> Option<f64> {
  let v = crate::functions::math_ast::try_eval_to_f64(e)?;
  v.is_finite().then_some(v)
}

/// A numeric side must be positive; a non-numeric side passes through
/// (wolframscript computes `SASTriangle[x, Pi/2, 4]` symbolically).
fn side_ok(e: &Expr) -> bool {
  match crate::functions::math_ast::try_eval_to_f64(e) {
    Some(v) => v.is_finite() && v > 0.0,
    None => true,
  }
}

/// Split an angle into (numerator, denominator) display strings when it is
/// fraction-shaped — a Rational or a product with a Rational coefficient —
/// so the ::asm message can typeset it as a 2D fraction like wolframscript.
fn fraction_parts(e: &Expr) -> Option<(String, String)> {
  match e {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      Some((expr_to_output(&args[0]), expr_to_output(&args[1])))
    }
    // A product with a Rational coefficient — Times may appear either as a
    // FunctionCall or as an infix BinaryOp node.
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      rational_times(&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => rational_times(left, right),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Some((expr_to_output(left), expr_to_output(right))),
    _ => None,
  }
}

/// The (numerator, denominator) strings of `coeff * rest` when `coeff` is a
/// Rational p/q: "p rest" over "q" (the coefficient 1 is not written).
fn rational_times(coeff: &Expr, rest: &Expr) -> Option<(String, String)> {
  if let Expr::FunctionCall { name, args } = coeff
    && name == "Rational"
    && args.len() == 2
  {
    let rest = expr_to_output(rest);
    let num = match &args[0] {
      Expr::Integer(1) => rest,
      p => format!("{} {}", expr_to_output(p), rest),
    };
    return Some((num, expr_to_output(&args[1])));
  }
  None
}

/// Emit `Head::asm` with wolframscript's layout: fraction angles are
/// typeset over three lines,
/// ```text
///                                     3 Pi     Pi
/// AASTriangle::asm: The sum of angles ---- and -- should be less than Pi.
///                                      4       2
/// ```
/// while inline angles (reals, integers) keep the message on one line.
fn emit_asm(head: &str, a1: &Expr, a2: &Expr) {
  let mut top = String::new();
  let mut mid = String::new();
  let mut bot = String::new();
  let mut any_fraction = false;
  let push_text =
    |top: &mut String, bot: &mut String, mid: &mut String, s: &str| {
      mid.push_str(s);
      top.push_str(&" ".repeat(s.chars().count()));
      bot.push_str(&" ".repeat(s.chars().count()));
    };
  let mut push_angle =
    |top: &mut String, bot: &mut String, mid: &mut String, e: &Expr| {
      match fraction_parts(e) {
        Some((num, den)) => {
          any_fraction = true;
          let w = num.chars().count().max(den.chars().count());
          let center = |s: &str| {
            let pad = (w - s.chars().count()) / 2;
            let mut out = " ".repeat(pad);
            out.push_str(s);
            out.push_str(&" ".repeat(w - pad - s.chars().count()));
            out
          };
          top.push_str(&center(&num));
          mid.push_str(&"-".repeat(w));
          bot.push_str(&center(&den));
        }
        None => {
          let s = expr_to_output(e);
          mid.push_str(&s);
          top.push_str(&" ".repeat(s.chars().count()));
          bot.push_str(&" ".repeat(s.chars().count()));
        }
      }
    };
  push_text(
    &mut top,
    &mut bot,
    &mut mid,
    &format!("{head}::asm: The sum of angles "),
  );
  push_angle(&mut top, &mut bot, &mut mid, a1);
  push_text(&mut top, &mut bot, &mut mid, " and ");
  push_angle(&mut top, &mut bot, &mut mid, a2);
  push_text(&mut top, &mut bot, &mut mid, " should be less than Pi.");
  if any_fraction {
    crate::emit_message(&format!(
      "{}\n{}\n{}",
      top.trim_end(),
      mid,
      bot.trim_end()
    ));
  } else {
    crate::emit_message(&mid);
  }
}

/// A symbolic (non-numeric) angle expression that can still plausibly
/// denote a scalar angle — strings and lists echo unevaluated instead.
fn symbolic_angle_ok(e: &Expr) -> bool {
  !matches!(e, Expr::String(_) | Expr::List(_))
}

/// Vertices {{0,0}, {c,0}, {cx,cy}} as an evaluated Triangle expression.
fn triangle(c: Expr, cx: Expr, cy: Expr) -> Result<Expr, InterpreterError> {
  let zero = || Expr::Integer(0);
  Ok(fc(
    "Triangle",
    vec![Expr::List(
      vec![
        Expr::List(vec![zero(), zero()].into()),
        Expr::List(vec![eval(c)?, zero()].into()),
        Expr::List(vec![eval(cx)?, eval(cy)?].into()),
      ]
      .into(),
    )],
  ))
}

/// Shared AAS/ASA core: angles α (at the origin) and β with either the side
/// a opposite α (AAS) or the included side c (ASA).
fn two_angle_triangle(
  head: &str,
  alpha: &Expr,
  beta: &Expr,
  side: &Expr,
  side_is_included: bool,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated(head, args));
  // Numeric angles must be positive with a sum below Pi; symbolic angles
  // evaluate symbolically (wolframscript computes `AASTriangle[a, Pi/3, 1]`
  // via its trig canonicalization).
  let (va, vb) = (angle_value(alpha), angle_value(beta));
  if va.is_some_and(|v| v <= 0.0)
    || vb.is_some_and(|v| v <= 0.0)
    || (va.is_none() && !symbolic_angle_ok(alpha))
    || (vb.is_none() && !symbolic_angle_ok(beta))
    || !side_ok(side)
  {
    return unevaluated();
  }
  if let (Some(va), Some(vb)) = (va, vb) {
    if va + vb >= std::f64::consts::PI {
      emit_asm(head, alpha, beta);
      return unevaluated();
    }
  } else if va.is_none() && vb.is_none() {
    // Fully symbolic angles are fine (wolframscript emits the general
    // Csc/Cot closed form), but a numeric angle of Pi or more with a
    // symbolic partner can never make a valid triangle.
  } else if va.or(vb).is_some_and(|v| v >= std::f64::consts::PI) {
    return unevaluated();
  }
  // γ = Pi - α - β; law of sines for the remaining sides, written with the
  // Csc/Cot forms wolframscript displays (`c*Csc[a]*Sin[a + b]`,
  // `c*Cot[a]*Sin[b]`). Numeric angles evaluate these to the same values
  // the plain Sin-quotient forms produce.
  let gamma = fc(
    "Plus",
    vec![
      Expr::Identifier("Pi".to_string()),
      fc("Times", vec![Expr::Integer(-1), alpha.clone()]),
      fc("Times", vec![Expr::Integer(-1), beta.clone()]),
    ],
  );
  let (c, cx, cy) = if side_is_included {
    // ASA: the given side is c; apex = c Sin[β]/Sin[γ] * (Cos[α], Sin[α]).
    (
      side.clone(),
      fc(
        "Times",
        vec![
          side.clone(),
          fc("Cos", vec![alpha.clone()]),
          fc("Csc", vec![gamma.clone()]),
          fc("Sin", vec![beta.clone()]),
        ],
      ),
      fc(
        "Times",
        vec![
          side.clone(),
          fc("Csc", vec![gamma]),
          fc("Sin", vec![alpha.clone()]),
          fc("Sin", vec![beta.clone()]),
        ],
      ),
    )
  } else {
    // AAS: the given side is a (opposite α); c = a Sin[γ] Csc[α],
    // apex = (a Sin[β] Cot[α], a Sin[β]).
    (
      fc(
        "Times",
        vec![
          side.clone(),
          fc("Csc", vec![alpha.clone()]),
          fc("Sin", vec![gamma]),
        ],
      ),
      fc(
        "Times",
        vec![
          side.clone(),
          fc("Cot", vec![alpha.clone()]),
          fc("Sin", vec![beta.clone()]),
        ],
      ),
      fc("Times", vec![side.clone(), fc("Sin", vec![beta.clone()])]),
    )
  };
  triangle(c, cx, cy)
}

/// AASTriangle[α, β, a] — angles α, β and the side a opposite α.
pub fn aas_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("AASTriangle", args));
  }
  two_angle_triangle("AASTriangle", &args[0], &args[1], &args[2], false, args)
}

/// ASATriangle[α, c, β] — angles α, β with the included side c.
pub fn asa_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("ASATriangle", args));
  }
  two_angle_triangle("ASATriangle", &args[0], &args[2], &args[1], true, args)
}

/// SASTriangle[a, γ, b] — sides a and b enclosing the angle γ. The third
/// side comes from the law of cosines; the apex is written with the
/// (b² - a b Cos[γ])/c and a b Sin[γ]/c forms wolframscript displays.
pub fn sas_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("SASTriangle", args));
  if args.len() != 3 {
    return unevaluated();
  }
  let (a, gamma, b) = (&args[0], &args[1], &args[2]);
  // A numeric included angle must lie in (0, Pi); symbolic angles compute
  // through the law-of-cosines closed form like wolframscript.
  let vg = angle_value(gamma);
  if vg.is_some_and(|v| v <= 0.0 || v >= std::f64::consts::PI)
    || (vg.is_none() && !symbolic_angle_ok(gamma))
    || !side_ok(a)
    || !side_ok(b)
  {
    return unevaluated();
  }
  let c = fc(
    "Sqrt",
    vec![fc(
      "Plus",
      vec![
        fc("Power", vec![a.clone(), Expr::Integer(2)]),
        fc("Power", vec![b.clone(), Expr::Integer(2)]),
        fc(
          "Times",
          vec![
            Expr::Integer(-2),
            a.clone(),
            b.clone(),
            fc("Cos", vec![gamma.clone()]),
          ],
        ),
      ],
    )],
  );
  let c_eval = eval(c)?;
  let inv_c = fc("Power", vec![c_eval.clone(), Expr::Integer(-1)]);
  let cx = fc(
    "Times",
    vec![
      fc(
        "Plus",
        vec![
          fc("Power", vec![b.clone(), Expr::Integer(2)]),
          fc(
            "Times",
            vec![
              Expr::Integer(-1),
              a.clone(),
              b.clone(),
              fc("Cos", vec![gamma.clone()]),
            ],
          ),
        ],
      ),
      inv_c.clone(),
    ],
  );
  let cy = fc(
    "Times",
    vec![a.clone(), b.clone(), fc("Sin", vec![gamma.clone()]), inv_c],
  );
  triangle(c_eval, cx, cy)
}

/// TriangleMeasurement[tri, "prop"] — scalar measurements of a triangle
/// given as Triangle[{p1, p2, p3}] or a bare three-point list. Supported
/// properties: "Area" (the one-argument default), "Perimeter",
/// "Semiperimeter", "Inradius", "Circumradius". Degenerate (collinear)
/// triangles emit ::invtri; unknown properties stay silently unevaluated.
///
/// Rational-coordinate triangles match wolframscript exactly. For
/// irrational side lengths the Inradius/Circumradius values are correct but
/// wolframscript's radical canonicalization (e.g. Sqrt[1105/2]/11 for
/// Sqrt[2210]/22) is not replicated.
pub fn triangle_measurement_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("TriangleMeasurement", args));
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }
  let prop = match args.get(1) {
    None => "Area".to_string(),
    Some(Expr::String(s)) => s.clone(),
    Some(_) => return unevaluated(),
  };
  if !matches!(
    prop.as_str(),
    "Area" | "Perimeter" | "Semiperimeter" | "Inradius" | "Circumradius"
  ) {
    return unevaluated();
  }

  // Extract the three 2D vertices from Triangle[{…}] or a bare list.
  let points = match &args[0] {
    Expr::FunctionCall { name, args: targs }
      if name == "Triangle" && targs.len() == 1 =>
    {
      &targs[0]
    }
    other => other,
  };
  let Expr::List(pts) = points else {
    return unevaluated();
  };
  if pts.len() != 3 {
    return unevaluated();
  }
  let mut coords: Vec<(Expr, Expr)> = Vec::with_capacity(3);
  for p in pts.iter() {
    match p {
      Expr::List(xy) if xy.len() == 2 => {
        coords.push((xy[0].clone(), xy[1].clone()))
      }
      _ => return unevaluated(),
    }
  }

  // Signed shoelace area ×2; zero means collinear (::invtri).
  let (ax, ay) = &coords[0];
  let (bx, by) = &coords[1];
  let (cx, cy) = &coords[2];
  let diff = |u: &Expr, v: &Expr| {
    fc(
      "Plus",
      vec![u.clone(), fc("Times", vec![Expr::Integer(-1), v.clone()])],
    )
  };
  let twice_signed_area = fc(
    "Plus",
    vec![
      fc("Times", vec![diff(bx, ax), diff(cy, ay)]),
      fc("Times", vec![Expr::Integer(-1), diff(cx, ax), diff(by, ay)]),
    ],
  );
  let signed = eval(twice_signed_area.clone())?;
  if crate::functions::math_ast::try_eval_to_f64(&signed) == Some(0.0) {
    crate::emit_message(&format!(
      "TriangleMeasurement::invtri: {} expected to specify a nondegenerate triangle in the plane.",
      expr_to_output(&args[0])
    ));
    return unevaluated();
  }
  let area = fc(
    "Times",
    vec![
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      },
      fc("Abs", vec![twice_signed_area]),
    ],
  );

  let dist = |p: &(Expr, Expr), q: &(Expr, Expr)| {
    fc(
      "Sqrt",
      vec![fc(
        "Plus",
        vec![
          fc("Power", vec![diff(&p.0, &q.0), Expr::Integer(2)]),
          fc("Power", vec![diff(&p.1, &q.1), Expr::Integer(2)]),
        ],
      )],
    )
  };
  let side_a = dist(&coords[1], &coords[2]);
  let side_b = dist(&coords[0], &coords[2]);
  let side_c = dist(&coords[0], &coords[1]);
  let half = |e: Expr| {
    fc(
      "Times",
      vec![
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
        },
        e,
      ],
    )
  };

  let result = match prop.as_str() {
    "Area" => area,
    "Perimeter" => fc("Plus", vec![side_a, side_b, side_c]),
    // Term-wise halves so rational parts fold (1/2 + 1/2 + Sqrt[2]/2 →
    // 1 + 1/Sqrt[2], matching wolframscript's display).
    "Semiperimeter" => {
      fc("Plus", vec![half(side_a), half(side_b), half(side_c)])
    }
    // r = Area / s
    "Inradius" => fc(
      "Times",
      vec![
        area,
        fc(
          "Power",
          vec![
            fc("Plus", vec![half(side_a), half(side_b), half(side_c)]),
            Expr::Integer(-1),
          ],
        ),
      ],
    ),
    // R = a b c / (4 Area)
    "Circumradius" => fc(
      "Times",
      vec![
        side_a,
        side_b,
        side_c,
        fc(
          "Power",
          vec![fc("Times", vec![Expr::Integer(4), area]), Expr::Integer(-1)],
        ),
      ],
    ),
    _ => unreachable!(),
  };
  eval(result)
}
