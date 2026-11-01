#[allow(unused_imports)]
use super::*;

/// Compute Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m) numerically
/// using the descending Landen (AGM) transformation.
/// Parameter m is the parameter (not the modulus k; m = k^2).
pub fn jacobi_elliptic(u: f64, m: f64) -> (f64, f64, f64) {
  // Edge cases
  if m.abs() < 1e-16 {
    // m = 0: sn = sin(u), cn = cos(u), dn = 1
    return (u.sin(), u.cos(), 1.0);
  }
  if (m - 1.0).abs() < 1e-16 {
    // m = 1: sn = tanh(u), cn = sech(u), dn = sech(u)
    let s = u.tanh();
    let c = 1.0 / u.cosh();
    return (s, c, c);
  }

  // AGM iteration to compute the sequence of a_n, b_n, c_n
  let mut a = vec![1.0];
  let mut b = vec![(1.0 - m).sqrt()];
  let mut c = vec![m.sqrt()];

  let n_max = 50;
  for _ in 0..n_max {
    let a_prev = *a.last().unwrap();
    let b_prev = *b.last().unwrap();
    a.push(f64::midpoint(a_prev, b_prev));
    b.push((a_prev * b_prev).sqrt());
    c.push((a_prev - b_prev) / 2.0);

    if c.last().unwrap().abs() < 1e-16 {
      break;
    }
  }

  let n = a.len() - 1;
  // phi_N = 2^N * a_N * u
  let mut phi = (1u64 << n as u64) as f64 * a[n] * u;

  // Descend: phi_{n-1} from phi_n
  for i in (1..=n).rev() {
    phi = f64::midpoint(phi, (c[i] / a[i] * phi.sin()).asin());
  }

  let sn = phi.sin();
  let cn = phi.cos();
  let dn = (1.0 - m * sn * sn).sqrt();

  (sn, cn, dn)
}

/// JacobiAmplitude[u, m] - amplitude for Jacobi elliptic functions
/// am(u, m) = arcsin(sn(u, m)), inverse of EllipticF
pub fn jacobi_amplitude_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiAmplitude expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiAmplitude[0, m] = 0
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // JacobiAmplitude[u, 0] = u (the modulus vanishes, so am collapses to u).
  if let Some(result) = elliptic_param_zero_reduces_to_first(u, m) {
    return Ok(result);
  }

  // JacobiAmplitude[u, 1] = Gudermannian[u]. (Real arguments are numericized by
  // the path below; this covers the exact-integer m == 1 case symbolically.)
  if matches!(m, Expr::Integer(1)) {
    return crate::evaluator::evaluate_expr_to_expr(&call1(
      "Gudermannian",
      u.clone(),
    ));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (try_eval_to_f64(u), try_eval_to_f64(m))
    && (matches!(u, Expr::Real(_)) || matches!(m, Expr::Real(_)))
  {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn.atan2(cn)));
  }

  Ok(unevaluated("JacobiAmplitude", args))
}

/// JacobiEpsilon[u, m] - the Jacobi epsilon function, the integral of dn^2,
/// equal to EllipticE[JacobiAmplitude[u, m], m]. Special cases match
/// wolframscript: JacobiEpsilon[0, m] = 0, JacobiEpsilon[u, 0] = u and
/// JacobiEpsilon[u, 1] = Tanh[u]; other exact/symbolic arguments stay symbolic.
pub fn jacobi_epsilon_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiEpsilon expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiEpsilon[0, m] = 0
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // JacobiEpsilon[u, 0] = u (the integrand dn^2 collapses to 1).
  if let Some(result) = elliptic_param_zero_reduces_to_first(u, m) {
    return Ok(result);
  }

  // JacobiEpsilon[u, 1] = Tanh[u].
  if matches!(m, Expr::Integer(1)) {
    return crate::evaluator::evaluate_expr_to_expr(&call1("Tanh", u.clone()));
  }

  // Numeric: JacobiEpsilon[u, m] = EllipticE[JacobiAmplitude[u, m], m].
  if let (Some(u_f), Some(m_f)) = (try_eval_to_f64(u), try_eval_to_f64(m))
    && (matches!(u, Expr::Real(_)) || matches!(m, Expr::Real(_)))
  {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    let am = sn.atan2(cn);
    return Ok(Expr::Real(elliptic_e_incomplete(am, m_f)));
  }

  Ok(unevaluated("JacobiEpsilon", args))
}

/// JacobiDN[u, m] - Jacobi elliptic function dn
pub fn jacobi_dn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiDN[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }

  // JacobiDN[u, 0] = 1
  if is_expr_zero(m) {
    return Ok(Expr::Integer(1));
  }

  // JacobiDN[u, 1] = Sech[u]
  if is_expr_one(m) {
    return Ok(call1("Sech", u.clone()));
  }

  // JacobiDN[-u, m] = JacobiDN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(call("JacobiDN", vec![inner, m.clone()]));
  }

  // Numeric evaluation only when an argument is inexact (a machine number);
  // exact arguments stay symbolic, matching wolframscript (N[...] numericizes
  // via the NumericFunction path, which makes the arguments inexact first).
  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn));
  }

  // Unevaluated
  Ok(unevaluated("JacobiDN", args))
}

/// JacobiSN[u, m] - Jacobi elliptic function sn
pub fn jacobi_sn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiSN[0, m] = 0
  if is_expr_zero(u) {
    return Ok(Expr::Integer(0));
  }

  // JacobiSN[u, 0] = Sin[u]
  if is_expr_zero(m) {
    return Ok(call1("Sin", u.clone()));
  }

  // JacobiSN[u, 1] = Tanh[u]
  if is_expr_one(m) {
    return Ok(call1("Tanh", u.clone()));
  }

  // JacobiSN[-u, m] = -JacobiSN[u, m] (odd function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(times2(
      Expr::Integer(-1),
      call("JacobiSN", vec![inner, m.clone()]),
    ));
  }

  // Numeric evaluation only when an argument is inexact (a machine number);
  // exact arguments stay symbolic, matching wolframscript (N[...] numericizes
  // via the NumericFunction path, which makes the arguments inexact first).
  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn));
  }

  // Unevaluated
  Ok(unevaluated("JacobiSN", args))
}

/// JacobiCN[u, m] - Jacobi elliptic function cn
pub fn jacobi_cn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiCN[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }

  // JacobiCN[u, 0] = Cos[u]
  if is_expr_zero(m) {
    return Ok(call1("Cos", u.clone()));
  }

  // JacobiCN[u, 1] = Sech[u]
  if is_expr_one(m) {
    return Ok(call1("Sech", u.clone()));
  }

  // JacobiCN[-u, m] = JacobiCN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(call("JacobiCN", vec![inner, m.clone()]));
  }

  // Numeric evaluation only when an argument is inexact (a machine number);
  // exact arguments stay symbolic, matching wolframscript (N[...] numericizes
  // via the NumericFunction path, which makes the arguments inexact first).
  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn));
  }

  // Unevaluated
  Ok(unevaluated("JacobiCN", args))
}

// ─── InverseJacobi functions ────────────────────────────────────────

/// Check if either argument is a Real (for numeric dispatch)
fn has_real_arg(a: &Expr, b: &Expr) -> bool {
  matches!(a, Expr::Real(_)) || matches!(b, Expr::Real(_))
}

/// `pq^-1(v, m)` at the parameter values where it reduces to an elementary
/// function or a constant: `v = 0` / `v = 1` (the zero and the quarter
/// period), `m = 0` and `m = 1` (circular and hyperbolic limits) and
/// `m = Infinity`, where every listed inverse collapses to 0.
fn inverse_jacobi_special(kind: &str, v: &Expr, m: &Expr) -> Option<Expr> {
  let inexact = has_real_arg(v, m);
  // pq(0) = 0 for the functions with sn as numerator.
  if is_expr_zero(v) && matches!(kind, "SN" | "SC" | "SD") {
    return Some(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  // pq(0) = 1 for the functions without sn.
  if is_expr_one(v) && matches!(kind, "CN" | "DN" | "CD" | "NC" | "ND" | "DC") {
    return Some(Expr::Integer(0));
  }
  if !inexact {
    // sn(K) = 1, cn(K) = cd(K) = 0.
    if (is_expr_one(v) && kind == "SN")
      || (is_expr_zero(v) && matches!(kind, "CN" | "CD"))
    {
      return Some(call1("EllipticK", m.clone()));
    }
  }
  if matches!(m, Expr::Identifier(s) if s == "Infinity") && kind != "DC" {
    return Some(Expr::Integer(0));
  }
  let elementary = if is_expr_zero(m) {
    match kind {
      "SN" | "SD" => "ArcSin",
      "CN" | "CD" => "ArcCos",
      "SC" => "ArcTan",
      "CS" => "ArcCot",
      "DS" | "NS" => "ArcCsc",
      "NC" | "DC" => "ArcSec",
      _ => return None,
    }
  } else if is_expr_one(m) {
    match kind {
      "SN" => "ArcTanh",
      "CN" | "DN" => "ArcSech",
      "SC" | "SD" => "ArcSinh",
      "DS" => "ArcCsch",
      "NS" => "ArcCoth",
      "NC" | "ND" => "ArcCosh",
      _ => return None,
    }
  } else {
    return None;
  };
  Some(call1(elementary, v.clone()))
}

/// Machine complex arithmetic for the complex inverse Jacobi path.
#[derive(Clone, Copy)]
struct Cx(f64, f64);

impl Cx {
  fn add(self, o: Cx) -> Cx {
    Cx(self.0 + o.0, self.1 + o.1)
  }
  fn sub(self, o: Cx) -> Cx {
    Cx(self.0 - o.0, self.1 - o.1)
  }
  fn mul(self, o: Cx) -> Cx {
    Cx(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
  }
  fn div(self, o: Cx) -> Cx {
    let d = o.0 * o.0 + o.1 * o.1;
    Cx(
      (self.0 * o.0 + self.1 * o.1) / d,
      (self.1 * o.0 - self.0 * o.1) / d,
    )
  }
  fn recip(self) -> Cx {
    Cx(1.0, 0.0).div(self)
  }
  /// Principal square root; the negative real axis maps to +i.
  fn sqrt(self) -> Cx {
    if self.1 == 0.0 {
      return if self.0 >= 0.0 {
        Cx(self.0.sqrt(), 0.0)
      } else {
        Cx(0.0, (-self.0).sqrt())
      };
    }
    let r = self.0.hypot(self.1);
    let re = f64::midpoint(r, self.0).sqrt();
    let im = ((r - self.0) / 2.0).sqrt();
    Cx(re, if self.1 < 0.0 { -im } else { im })
  }
  fn ln(self) -> Cx {
    Cx(self.0.hypot(self.1).ln(), self.1.atan2(self.0))
  }
  /// ArcSin[z] = -i Log[i z + Sqrt[1 - z^2]] (WL's branch cuts).
  fn asin(self) -> Cx {
    let iz = Cx(-self.1, self.0);
    let w = iz.add(Cx(1.0, 0.0).sub(self.mul(self)).sqrt()).ln();
    Cx(w.1, -w.0)
  }
  fn acos(self) -> Cx {
    Cx(std::f64::consts::FRAC_PI_2, 0.0).sub(self.asin())
  }
  /// ArcTan[z] = i/2 (Log[1 - i z] - Log[1 + i z]).
  fn atan(self) -> Cx {
    let iz = Cx(-self.1, self.0);
    let d = Cx(1.0, 0.0).sub(iz).ln().sub(Cx(1.0, 0.0).add(iz).ln());
    Cx(-d.1 / 2.0, d.0 / 2.0)
  }
}

/// `pq^-1(z, m)` for machine complex `z` and `m`: `F(phi, m)` of the
/// amplitude `phi` with `sn(phi) = Sin[phi]` the matching circular inverse
/// (`F(ArcSin[z])` for sn, `F(ArcCos[z])` for cn, ...), the reciprocal
/// functions through `pq^-1(z) = qp^-1(1/z)`.
fn inverse_jacobi_complex(kind: &str, z: Cx, m: Cx) -> Option<Cx> {
  let one = Cx(1.0, 0.0);
  let f = |phi: Cx| {
    let (re, im) = elliptic_f_c64((phi.0, phi.1), (m.0, m.1));
    Cx(re, im)
  };
  let recip = |z: Cx| (z.0 != 0.0 || z.1 != 0.0).then(|| z.recip());
  let r = match kind {
    "SN" => f(z.asin()),
    "CN" => f(z.acos()),
    "SC" => f(z.atan()),
    "CS" => {
      if z.0 == 0.0 && z.1 == 0.0 {
        f(Cx(std::f64::consts::FRAC_PI_2, 0.0))
      } else {
        f(z.recip().atan())
      }
    }
    "SD" => f(z.div(one.add(m.mul(z).mul(z)).sqrt()).asin()),
    "CD" => {
      let q = one.sub(m).div(one.sub(m.mul(z).mul(z))).sqrt();
      f(z.mul(q).acos())
    }
    "DN" => {
      if m.0 == 0.0 && m.1 == 0.0 {
        return None;
      }
      // dn(2 i K' - u) = -dn(u): the left half plane by reflection.
      let w = if z.0 < 0.0 { Cx(-z.0, -z.1) } else { z };
      let u = f(one.sub(w.mul(w)).div(m).sqrt().asin());
      if z.0 < 0.0 {
        let (kr, ki) = elliptic_k_c64((1.0 - m.0, -m.1));
        Cx(-2.0 * ki, 2.0 * kr).sub(u)
      } else {
        u
      }
    }
    "NS" => return inverse_jacobi_complex("SN", recip(z)?, m),
    "NC" => return inverse_jacobi_complex("CN", recip(z)?, m),
    "ND" => return inverse_jacobi_complex("DN", recip(z)?, m),
    "DS" => return inverse_jacobi_complex("SD", recip(z)?, m),
    "DC" => return inverse_jacobi_complex("CD", recip(z)?, m),
    _ => return None,
  };
  (r.0.is_finite() && r.1.is_finite()).then_some(r)
}

/// The numeric value of `InverseJacobi<kind>[v, m]` once an argument is
/// inexact: real arguments go through the arbitrary-precision core
/// (`elliptic_bigfloat`) — at 128 bits rounded to a machine number, or at
/// the lowest argument precision for arbitrary-precision input — and the
/// rest through the machine complex formulas.
fn inverse_jacobi_numeric(kind: &str, v: &Expr, m: &Expr) -> Option<Expr> {
  if !expr_is_inexact(v) && !expr_is_inexact(m) {
    return None;
  }
  let machine = contains_machine_real(v) || contains_machine_real(m);
  let real =
    |a: &Expr| is_real_numeric(a).then(|| try_eval_to_f64(a)).flatten();
  if let (Some(vf), Some(mf)) = (real(v), real(m))
    && (vf.is_finite() && mf.is_finite())
  {
    if !machine {
      let precision = [v, m]
        .iter()
        .filter_map(|a| match a {
          Expr::BigFloat(_, p) => Some(*p),
          _ => None,
        })
        .reduce(f64::min)?;
      let call =
        call(&format!("InverseJacobi{kind}"), vec![v.clone(), m.clone()]);
      if let Some(r) = bigfloat_function_value(&call, precision) {
        return Some(r);
      }
    } else if let Some((re, im)) = with_thread_consts(|cc| {
      let rm = astro_float::RoundingMode::ToEven;
      let bits = 128;
      let x = astro_float::BigFloat::from_f64(vf, bits);
      let mm = astro_float::BigFloat::from_f64(mf, bits);
      let (re, im) = inverse_jacobi_bigfloat(kind, &x, &mm, bits, rm, cc)?;
      Some((bigfloat_to_f64_approx(&re)?, bigfloat_to_f64_approx(&im)?))
    }) {
      return Some(if im == 0.0 {
        Expr::Real(re)
      } else {
        build_complex_float_expr(re, im)
      });
    }
  }
  let z = try_extract_complex_float(v)?;
  let mc = try_extract_complex_float(m)?;
  let r = inverse_jacobi_complex(kind, Cx(z.0, z.1), Cx(mc.0, mc.1))?;
  Some(build_complex_float_expr(r.0, r.1))
}

pub(crate) fn contains_machine_real(e: &Expr) -> bool {
  match e {
    Expr::Real(_) => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_machine_real(left) || contains_machine_real(right)
    }
    Expr::UnaryOp { operand, .. } => contains_machine_real(operand),
    Expr::FunctionCall { args, .. } => args.iter().any(contains_machine_real),
    _ => false,
  }
}

/// `InverseJacobi<kind>[v, m]`, the canonical implementation behind the
/// twelve inverse Jacobi functions.
fn inverse_jacobi_ast(
  kind: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let name = format!("InverseJacobi{kind}");
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{name} expects exactly 2 arguments"
    )));
  }
  let (v, m) = (&args[0], &args[1]);
  // pq(0) = 1: an exact 0 even with an inexact parameter.
  if is_expr_one(v) && matches!(kind, "CN" | "DN" | "CD" | "NC" | "ND" | "DC")
  {
    return Ok(Expr::Integer(0));
  }
  if let Some(r) = inverse_jacobi_numeric(kind, v, m) {
    return Ok(r);
  }
  if let Some(r) = inverse_jacobi_special(kind, v, m) {
    return Ok(r);
  }
  Ok(unevaluated(&name, args))
}

macro_rules! inverse_jacobi_entry {
  ($($fn_name:ident => $kind:literal),* $(,)?) => {
    $(
      #[doc = concat!("InverseJacobi", $kind, "[v, m]")]
      pub fn $fn_name(args: &[Expr]) -> Result<Expr, InterpreterError> {
        inverse_jacobi_ast($kind, args)
      }
    )*
  };
}

inverse_jacobi_entry! {
  inverse_jacobi_sn_ast => "SN",
  inverse_jacobi_cn_ast => "CN",
  inverse_jacobi_dn_ast => "DN",
  inverse_jacobi_cd_ast => "CD",
  inverse_jacobi_sc_ast => "SC",
  inverse_jacobi_cs_ast => "CS",
  inverse_jacobi_sd_ast => "SD",
  inverse_jacobi_ds_ast => "DS",
  inverse_jacobi_ns_ast => "NS",
  inverse_jacobi_nc_ast => "NC",
  inverse_jacobi_nd_ast => "ND",
  inverse_jacobi_dc_ast => "DC",
}

/// Extract the inner expression from a negated expression like Times[-1, x] or -x
fn extract_negated_expr(expr: &Expr) -> Option<Expr> {
  match expr {
    // Times[-1, x] form (FunctionCall)
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(args[0], Expr::Integer(-1)) =>
    {
      Some(args[1].clone())
    }
    // BinaryOp Times[-1, x] form
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(-1)) {
        Some(right.as_ref().clone())
      } else if matches!(right.as_ref(), Expr::Integer(-1)) {
        Some(left.as_ref().clone())
      } else {
        None
      }
    }
    // UnaryOp Minus form
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(operand.as_ref().clone()),
    _ => None,
  }
}

/// JacobiSC[u, m] - Jacobi SC elliptic function (SN/CN)
pub fn jacobi_sc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiSC[0, m] = 0
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }
  // JacobiSC[u, 0] = Tan[u]
  if is_expr_zero(m) {
    return Ok(call1("Tan", u.clone()));
  }
  // JacobiSC[u, 1] = Sinh[u]
  if is_expr_one(m) {
    return Ok(call1("Sinh", u.clone()));
  }

  // Numeric evaluation only when an argument is inexact (a machine number);
  // exact arguments stay symbolic, matching wolframscript (N[...] numericizes
  // via the NumericFunction path, which makes the arguments inexact first).
  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / cn));
  }

  Ok(unevaluated("JacobiSC", args))
}

/// JacobiDC[u, m] - Jacobi DC elliptic function (DN/CN)
pub fn jacobi_dc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiDC[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  // JacobiDC[u, 0] = Sec[u]
  if is_expr_zero(m) {
    return Ok(call1("Sec", u.clone()));
  }
  // JacobiDC[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation only when an argument is inexact (a machine number);
  // exact arguments stay symbolic, matching wolframscript (N[...] numericizes
  // via the NumericFunction path, which makes the arguments inexact first).
  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / cn));
  }

  Ok(unevaluated("JacobiDC", args))
}

/// JacobiCD[u, m] - Jacobi CD elliptic function (CN/DN)
pub fn jacobi_cd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCD expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiCD[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  // JacobiCD[u, 0] = Cos[u]
  if is_expr_zero(m) {
    return Ok(call1("Cos", u.clone()));
  }
  // JacobiCD[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation only when an argument is inexact (a machine number);
  // exact arguments stay symbolic, matching wolframscript (N[...] numericizes
  // via the NumericFunction path, which makes the arguments inexact first).
  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / dn));
  }

  Ok(unevaluated("JacobiCD", args))
}

/// JacobiSD[u, m] - Jacobi SD elliptic function (SN/DN)
pub fn jacobi_sd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSD expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiSD[0, m] = 0
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }
  if is_expr_zero(m) {
    return Ok(call1("Sin", u.clone()));
  }
  if is_expr_one(m) {
    return Ok(call1("Sinh", u.clone()));
  }

  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / dn));
  }

  Ok(unevaluated("JacobiSD", args))
}

/// JacobiCS[u, m] - Jacobi CS elliptic function (CN/SN)
pub fn jacobi_cs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(id_expr("ComplexInfinity"));
  }
  if is_expr_zero(m) {
    return Ok(call1("Cot", u.clone()));
  }
  if is_expr_one(m) {
    return Ok(call1("Csch", u.clone()));
  }

  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / sn));
  }

  Ok(unevaluated("JacobiCS", args))
}

/// JacobiDS[u, m] - Jacobi DS elliptic function (DN/SN)
pub fn jacobi_ds_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(id_expr("ComplexInfinity"));
  }
  if is_expr_zero(m) {
    return Ok(call1("Csc", u.clone()));
  }
  if is_expr_one(m) {
    return Ok(call1("Csch", u.clone()));
  }

  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / sn));
  }

  Ok(unevaluated("JacobiDS", args))
}

/// JacobiNS[u, m] - Jacobi NS elliptic function (1/SN)
pub fn jacobi_ns_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiNS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(id_expr("ComplexInfinity"));
  }
  if is_expr_zero(m) {
    return Ok(call1("Csc", u.clone()));
  }
  if is_expr_one(m) {
    return Ok(call1("Coth", u.clone()));
  }

  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / sn));
  }

  Ok(unevaluated("JacobiNS", args))
}

/// JacobiND[u, m] - Jacobi ND elliptic function (1/DN)
pub fn jacobi_nd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiND expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiND[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  if is_expr_zero(m) {
    return Ok(Expr::Integer(1));
  }
  if is_expr_one(m) {
    return Ok(call1("Cosh", u.clone()));
  }

  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / dn));
  }

  Ok(unevaluated("JacobiND", args))
}

/// JacobiNC[u, m] - Jacobi NC elliptic function (1/CN)
pub fn jacobi_nc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiNC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiNC[0, m] = 1
  if is_expr_zero(u) {
    if has_real_arg(u, m) {
      return Ok(Expr::Real(1.0));
    }
    return Ok(Expr::Integer(1));
  }
  if is_expr_zero(m) {
    return Ok(call1("Sec", u.clone()));
  }
  if is_expr_one(m) {
    return Ok(call1("Cosh", u.clone()));
  }

  if has_real_arg(u, m)
    && let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m))
  {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / cn));
  }

  Ok(unevaluated("JacobiNC", args))
}

/// Check if an expression is numerically zero
pub fn is_expr_zero(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(0) => true,
    Expr::Real(f) => *f == 0.0,
    _ => false,
  }
}

/// Check if an expression is numerically one
fn is_expr_one(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(1) => true,
    Expr::Real(f) => *f == 1.0,
    _ => false,
  }
}

/// Check if an expression represents -Infinity
pub fn is_neg_infinity(expr: &Expr) -> bool {
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) if s == "Infinity")
      || matches!(e, Expr::Constant(s) if s == "Infinity")
  };
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
    && matches!(&args[0], Expr::Integer(-1))
    && is_infinity(&args[1])
  {
    return true;
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
    && matches!(left.as_ref(), Expr::Integer(-1))
    && is_infinity(right.as_ref())
  {
    return true;
  }
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = expr
    && is_infinity(operand.as_ref())
  {
    return true;
  }
  false
}
