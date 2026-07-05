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
use crate::syntax::Expr;

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
      Some((
        crate::syntax::expr_to_output(&args[0]),
        crate::syntax::expr_to_output(&args[1]),
      ))
    }
    // A product with a Rational coefficient — Times may appear either as a
    // FunctionCall or as an infix BinaryOp node.
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      rational_times(&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => rational_times(left, right),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => Some((
      crate::syntax::expr_to_output(left),
      crate::syntax::expr_to_output(right),
    )),
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
    let rest = crate::syntax::expr_to_output(rest);
    let num = match &args[0] {
      Expr::Integer(1) => rest,
      p => format!("{} {}", crate::syntax::expr_to_output(p), rest),
    };
    return Some((num, crate::syntax::expr_to_output(&args[1])));
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
          let s = crate::syntax::expr_to_output(e);
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
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: head.to_string(),
      args: args.to_vec().into(),
    })
  };
  let (Some(va), Some(vb)) = (angle_value(alpha), angle_value(beta)) else {
    return unevaluated();
  };
  if va <= 0.0 || vb <= 0.0 || !side_ok(side) {
    return unevaluated();
  }
  if va + vb >= std::f64::consts::PI {
    emit_asm(head, alpha, beta);
    return unevaluated();
  }
  // γ = Pi - α - β; law of sines for the remaining sides.
  let gamma = fc(
    "Plus",
    vec![
      Expr::Identifier("Pi".to_string()),
      fc("Times", vec![Expr::Integer(-1), alpha.clone()]),
      fc("Times", vec![Expr::Integer(-1), beta.clone()]),
    ],
  );
  let ratio = |num_angle: Expr, den_angle: Expr| {
    fc(
      "Times",
      vec![
        side.clone(),
        fc("Sin", vec![num_angle]),
        fc("Power", vec![fc("Sin", vec![den_angle]), Expr::Integer(-1)]),
      ],
    )
  };
  let (c, b) = if side_is_included {
    // ASA: the given side is c; b = c Sin[β]/Sin[γ].
    (side.clone(), ratio(beta.clone(), gamma))
  } else {
    // AAS: the given side is a (opposite α); c = a Sin[γ]/Sin[α],
    // b = a Sin[β]/Sin[α].
    (
      ratio(gamma, alpha.clone()),
      ratio(beta.clone(), alpha.clone()),
    )
  };
  let cx = fc("Times", vec![b.clone(), fc("Cos", vec![alpha.clone()])]);
  let cy = fc("Times", vec![b, fc("Sin", vec![alpha.clone()])]);
  triangle(c, cx, cy)
}

/// AASTriangle[α, β, a] — angles α, β and the side a opposite α.
pub fn aas_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "AASTriangle".to_string(),
      args: args.to_vec().into(),
    });
  }
  two_angle_triangle("AASTriangle", &args[0], &args[1], &args[2], false, args)
}

/// ASATriangle[α, c, β] — angles α, β with the included side c.
pub fn asa_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "ASATriangle".to_string(),
      args: args.to_vec().into(),
    });
  }
  two_angle_triangle("ASATriangle", &args[0], &args[2], &args[1], true, args)
}

/// SASTriangle[a, γ, b] — sides a and b enclosing the angle γ. The third
/// side comes from the law of cosines; the apex is written with the
/// (b² - a b Cos[γ])/c and a b Sin[γ]/c forms wolframscript displays.
pub fn sas_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "SASTriangle".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 3 {
    return unevaluated();
  }
  let (a, gamma, b) = (&args[0], &args[1], &args[2]);
  let Some(vg) = angle_value(gamma) else {
    return unevaluated();
  };
  if vg <= 0.0 || vg >= std::f64::consts::PI || !side_ok(a) || !side_ok(b) {
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
