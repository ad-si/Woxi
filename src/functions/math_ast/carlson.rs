//! Carlson symmetric elliptic integrals R_C, R_F, R_D, R_J, R_G.
//!
//! The numeric kernels use Carlson's duplication algorithm (as given in
//! Numerical Recipes). Each public AST function stays symbolic for exact
//! (integer/rational) arguments and evaluates numerically once an argument is
//! inexact (a machine real, e.g. after `N`), matching wolframscript.

#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

/// True if `e` contains an inexact (machine) number anywhere.
fn is_inexact(e: &Expr) -> bool {
  match e {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => is_inexact(left) || is_inexact(right),
    Expr::UnaryOp { operand, .. } => is_inexact(operand),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(is_inexact)
    }
    _ => false,
  }
}

/// R_C(x, y) — degenerate Carlson integral.
fn carlson_rc(x: f64, y: f64) -> f64 {
  const ERRTOL: f64 = 0.0012;
  const C1: f64 = 0.3;
  const C2: f64 = 1.0 / 7.0;
  const C3: f64 = 0.375;
  const C4: f64 = 9.0 / 22.0;
  let (mut xt, mut yt, w) = if y > 0.0 {
    (x, y, 1.0)
  } else {
    (x - y, -y, x.sqrt() / (x - y).sqrt())
  };
  let mut s;
  let mut ave;
  loop {
    let alamb = 2.0 * xt.sqrt() * yt.sqrt() + yt;
    xt = 0.25 * (xt + alamb);
    yt = 0.25 * (yt + alamb);
    ave = (xt + yt + yt) / 3.0;
    s = (yt - ave) / ave;
    if s.abs() <= ERRTOL {
      break;
    }
  }
  w * (1.0 + s * s * (C1 + s * (C2 + s * (C3 + s * C4)))) / ave.sqrt()
}

/// R_F(x, y, z) — symmetric elliptic integral of the first kind.
fn carlson_rf(x: f64, y: f64, z: f64) -> f64 {
  const ERRTOL: f64 = 0.0025;
  const C1: f64 = 1.0 / 24.0;
  const C2: f64 = 0.1;
  const C3: f64 = 3.0 / 44.0;
  const C4: f64 = 1.0 / 14.0;
  let (mut xt, mut yt, mut zt) = (x, y, z);
  let (mut delx, mut dely, mut delz, mut ave);
  loop {
    let sqrtx = xt.sqrt();
    let sqrty = yt.sqrt();
    let sqrtz = zt.sqrt();
    let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
    xt = 0.25 * (xt + alamb);
    yt = 0.25 * (yt + alamb);
    zt = 0.25 * (zt + alamb);
    ave = (xt + yt + zt) / 3.0;
    delx = (ave - xt) / ave;
    dely = (ave - yt) / ave;
    delz = (ave - zt) / ave;
    if delx.abs().max(dely.abs()).max(delz.abs()) <= ERRTOL {
      break;
    }
  }
  let e2 = delx * dely - delz * delz;
  let e3 = delx * dely * delz;
  (1.0 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / ave.sqrt()
}

/// R_D(x, y, z) — symmetric elliptic integral of the second kind (R_J with the
/// last two arguments equal: R_D(x, y, z) = R_J(x, y, z, z)).
fn carlson_rd(x: f64, y: f64, z: f64) -> f64 {
  const ERRTOL: f64 = 0.0015;
  const C1: f64 = 3.0 / 14.0;
  const C2: f64 = 1.0 / 6.0;
  const C3: f64 = 9.0 / 22.0;
  const C4: f64 = 3.0 / 26.0;
  const C5: f64 = 0.25 * C3;
  const C6: f64 = 1.5 * C4;
  let (mut xt, mut yt, mut zt) = (x, y, z);
  let mut sum = 0.0;
  let mut fac = 1.0;
  let (mut delx, mut dely, mut delz, mut ave);
  loop {
    let sqrtx = xt.sqrt();
    let sqrty = yt.sqrt();
    let sqrtz = zt.sqrt();
    let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
    sum += fac / (sqrtz * (zt + alamb));
    fac *= 0.25;
    xt = 0.25 * (xt + alamb);
    yt = 0.25 * (yt + alamb);
    zt = 0.25 * (zt + alamb);
    ave = 0.2 * (xt + yt + 3.0 * zt);
    delx = (ave - xt) / ave;
    dely = (ave - yt) / ave;
    delz = (ave - zt) / ave;
    if delx.abs().max(dely.abs()).max(delz.abs()) <= ERRTOL {
      break;
    }
  }
  let ea = delx * dely;
  let eb = delz * delz;
  let ec = ea - eb;
  let ed = ea - 6.0 * eb;
  let ee = ed + ec + ec;
  3.0 * sum
    + fac
      * (1.0
        + ed * (-C1 + C5 * ed - C6 * delz * ee)
        + delz * (C2 * ee + delz * (-C3 * ec + delz * C4 * ea)))
      / (ave * ave.sqrt())
}

/// R_J(x, y, z, p) — symmetric elliptic integral of the third kind. Only the
/// principal branch p > 0 is implemented here; for p <= 0 the caller leaves the
/// expression unevaluated.
fn carlson_rj(x: f64, y: f64, z: f64, p: f64) -> f64 {
  const ERRTOL: f64 = 0.0015;
  const C1: f64 = 3.0 / 14.0;
  const C2: f64 = 1.0 / 3.0;
  const C3: f64 = 3.0 / 22.0;
  const C4: f64 = 3.0 / 26.0;
  const C5: f64 = 0.75 * C3;
  const C6: f64 = 1.5 * C4;
  const C7: f64 = 0.5 * C2;
  const C8: f64 = C3 + C3;
  let (mut xt, mut yt, mut zt, mut pt) = (x, y, z, p);
  let mut sum = 0.0;
  let mut fac = 1.0;
  let (mut delx, mut dely, mut delz, mut delp, mut ave);
  loop {
    let sqrtx = xt.sqrt();
    let sqrty = yt.sqrt();
    let sqrtz = zt.sqrt();
    let alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
    let alpha = (pt * (sqrtx + sqrty + sqrtz) + sqrtx * sqrty * sqrtz).powi(2);
    let beta = pt * (pt + alamb).powi(2);
    sum += fac * carlson_rc(alpha, beta);
    fac *= 0.25;
    xt = 0.25 * (xt + alamb);
    yt = 0.25 * (yt + alamb);
    zt = 0.25 * (zt + alamb);
    pt = 0.25 * (pt + alamb);
    ave = 0.2 * (xt + yt + zt + pt + pt);
    delx = (ave - xt) / ave;
    dely = (ave - yt) / ave;
    delz = (ave - zt) / ave;
    delp = (ave - pt) / ave;
    if delx.abs().max(dely.abs()).max(delz.abs()).max(delp.abs()) <= ERRTOL {
      break;
    }
  }
  let ea = delx * (dely + delz) + dely * delz;
  let eb = delx * dely * delz;
  let ec = delp * delp;
  let ed = ea - 3.0 * ec;
  let ee = eb + 2.0 * delp * (ea - ec);
  3.0 * sum
    + fac
      * (1.0
        + ed * (-C1 + C5 * ed - C6 * ee)
        + eb * (C7 + delp * (-C8 + delp * C4))
        + delp * ea * (C2 - delp * C3)
        - C2 * delp * ec)
      / (ave * ave.sqrt())
}

/// R_G(x, y, z) — symmetric elliptic integral of the second kind, evaluated via
/// 2 R_G = z R_F(x, y, z) - (1/3)(x - z)(y - z) R_D(x, y, z) + sqrt(x y / z),
/// choosing the largest argument as z to avoid division by a vanishing value.
fn carlson_rg(x: f64, y: f64, z: f64) -> f64 {
  // Order so that c is the largest (used as the formula's z); R_D is symmetric
  // in its first two arguments.
  let mut v = [x, y, z];
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let (a, b, c) = (v[0], v[1], v[2]);
  if c == 0.0 {
    return 0.0;
  }
  0.5
    * (c * carlson_rf(a, b, c)
      - (1.0 / 3.0) * (a - c) * (b - c) * carlson_rd(a, b, c)
      + (a * b / c).sqrt())
}

/// Generic AST wrapper: stay symbolic for exact input, evaluate `f` once any
/// argument is inexact and every argument reduces to a real.
fn carlson_ast(
  name: &str,
  args: &[Expr],
  arity: usize,
  f: impl Fn(&[f64]) -> Option<f64>,
) -> Result<Expr, InterpreterError> {
  let symbolic = || Ok(unevaluated(name, args));
  if args.len() != arity {
    return symbolic();
  }
  if !args.iter().any(is_inexact) {
    return symbolic();
  }
  let mut vals = Vec::with_capacity(arity);
  for a in args {
    match crate::functions::math_ast::expr_to_f64(a) {
      Some(v) => vals.push(v),
      None => return symbolic(),
    }
  }
  match f(&vals) {
    Some(r) => Ok(Expr::Real(r)),
    None => symbolic(),
  }
}

pub fn carlson_rc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  carlson_ast("CarlsonRC", args, 2, |v| Some(carlson_rc(v[0], v[1])))
}

pub fn carlson_rf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  carlson_ast("CarlsonRF", args, 3, |v| Some(carlson_rf(v[0], v[1], v[2])))
}

pub fn carlson_rd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  carlson_ast("CarlsonRD", args, 3, |v| Some(carlson_rd(v[0], v[1], v[2])))
}

pub fn carlson_rj_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Only the principal branch p > 0 is supported numerically.
  carlson_ast("CarlsonRJ", args, 4, |v| {
    if v[3] > 0.0 {
      Some(carlson_rj(v[0], v[1], v[2], v[3]))
    } else {
      None
    }
  })
}

pub fn carlson_rg_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  carlson_ast("CarlsonRG", args, 3, |v| Some(carlson_rg(v[0], v[1], v[2])))
}

pub fn carlson_re_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // CarlsonRE[x, y] is the complete integral (4/Pi) R_G(0, x, y).
  carlson_ast("CarlsonRE", args, 2, |v| {
    Some(4.0 / std::f64::consts::PI * carlson_rg(0.0, v[0], v[1]))
  })
}
