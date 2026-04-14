#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

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
    a.push((a_prev + b_prev) / 2.0);
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
    phi = (phi + (c[i] / a[i] * phi.sin()).asin()) / 2.0;
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

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (try_eval_to_f64(u), try_eval_to_f64(m))
    && (matches!(u, Expr::Real(_)) || matches!(m, Expr::Real(_)))
  {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn.atan2(cn)));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiAmplitude".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Sech".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiDN[-u, m] = JacobiDN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::FunctionCall {
      name: "JacobiDN".to_string(),
      args: vec![inner, m.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiDN".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiSN[u, 1] = Tanh[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Tanh".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiSN[-u, m] = -JacobiSN[u, m] (odd function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(Expr::FunctionCall {
        name: "JacobiSN".to_string(),
        args: vec![inner, m.clone()],
      }),
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiSN".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiCN[u, 1] = Sech[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sech".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiCN[-u, m] = JacobiCN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::FunctionCall {
      name: "JacobiCN".to_string(),
      args: vec![inner, m.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiCN".to_string(),
    args: args.to_vec(),
  })
}

// ─── InverseJacobi functions ────────────────────────────────────────

/// Check if either argument is a Real (for numeric dispatch)
fn has_real_arg(a: &Expr, b: &Expr) -> bool {
  matches!(a, Expr::Real(_)) || matches!(b, Expr::Real(_))
}

/// Helper: compute InverseJacobiSN numerically via EllipticF[ArcSin[v], m]
fn inverse_jacobi_sn_numeric(v: f64, m: f64) -> f64 {
  elliptic_f(v.asin(), m)
}

/// InverseJacobiSN[v, m]
pub fn inverse_jacobi_sn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiSN expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiSN[0, m] = 0
  if is_expr_zero(v) {
    if has_real_arg(v, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // InverseJacobiSN[1, m] = EllipticK[m]
  if is_expr_one(v) && !has_real_arg(v, m) {
    return Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: vec![m.clone()],
    });
  }

  // Numeric evaluation: EllipticF[ArcSin[v], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && (has_real_arg(v, m) || is_expr_one(v))
  {
    return Ok(Expr::Real(inverse_jacobi_sn_numeric(v_f, m_f)));
  }

  // InverseJacobiSN[x, 0] = ArcSin[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSin".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiSN[x, 1] = ArcTanh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcTanh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiSN".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiCN[v, m]
pub fn inverse_jacobi_cn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiCN expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiCN[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // InverseJacobiCN[0, m] = EllipticK[m] (symbolic only)
  if is_expr_zero(v) && !has_real_arg(v, m) {
    return Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: vec![m.clone()],
    });
  }

  // Numeric: EllipticF[ArcCos[v], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    return Ok(Expr::Real(elliptic_f(v_f.acos(), m_f)));
  }

  // InverseJacobiCN[x, 0] = ArcCos[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCos".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiCN[x, 1] = ArcSech[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSech".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiCN".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiDN[v, m]
pub fn inverse_jacobi_dn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiDN expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiDN[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(1-v^2)/m]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
    && m_f != 0.0
  {
    let arg = ((1.0 - v_f * v_f) / m_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiDN[x, 1] = ArcSech[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSech".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiDN".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiCD[v, m]
pub fn inverse_jacobi_cd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiCD expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiCD[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // InverseJacobiCD[0, m] = EllipticK[m] (symbolic only)
  if is_expr_zero(v) && !has_real_arg(v, m) {
    return Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: vec![m.clone()],
    });
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(1-v^2)/(1-m*v^2)]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = ((1.0 - v_f * v_f) / (1.0 - m_f * v_f * v_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiCD[x, 0] = ArcCos[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCos".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiCD".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiSC[v, m]
pub fn inverse_jacobi_sc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiSC expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiSC[0, m] = 0
  if is_expr_zero(v) {
    if has_real_arg(v, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[v/Sqrt[1+v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = v_f / (1.0 + v_f * v_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiSC[x, 0] = ArcTan[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcTan".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiSC[x, 1] = ArcSinh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSinh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiSC".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiCS[v, m]
pub fn inverse_jacobi_cs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiCS expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // Numeric: EllipticF[ArcSin[1/Sqrt[1+v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = 1.0 / (1.0 + v_f * v_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiCS[x, 0] = ArcCot[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCot".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiCS".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiSD[v, m]
pub fn inverse_jacobi_sd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiSD expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiSD[0, m] = 0
  if is_expr_zero(v) {
    if has_real_arg(v, m) {
      return Ok(Expr::Real(0.0));
    }
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[v/Sqrt[1+m*v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = v_f / (1.0 + m_f * v_f * v_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiSD[x, 0] = ArcSin[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSin".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiSD[x, 1] = ArcSinh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSinh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiSD".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiDS[v, m]
pub fn inverse_jacobi_ds_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiDS expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // Numeric: EllipticF[ArcSin[1/Sqrt[v^2+m]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = 1.0 / (v_f * v_f + m_f).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiDS[x, 0] = ArcCsc[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCsc".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiDS[x, 1] = ArcCsch[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCsch".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiDS".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiNS[v, m]
pub fn inverse_jacobi_ns_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiNS expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // Numeric: EllipticF[ArcSin[1/v], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = 1.0 / v_f;
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiNS[x, 0] = ArcCsc[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCsc".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiNS[x, 1] = ArcCoth[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCoth".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiNS".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiNC[v, m]
pub fn inverse_jacobi_nc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiNC expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiNC[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[1-1/v^2]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = (1.0 - 1.0 / (v_f * v_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiNC[x, 0] = ArcSec[x]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcSec".to_string(),
      args: vec![v.clone()],
    });
  }

  // InverseJacobiNC[x, 1] = ArcCosh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCosh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiNC".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiND[v, m]
pub fn inverse_jacobi_nd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiND expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiND[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(v^2-1)/(m*v^2)]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
    && m_f != 0.0
  {
    let arg = ((v_f * v_f - 1.0) / (m_f * v_f * v_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  // InverseJacobiND[x, 1] = ArcCosh[x]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "ArcCosh".to_string(),
      args: vec![v.clone()],
    });
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiND".to_string(),
    args: args.to_vec(),
  })
}

/// InverseJacobiDC[v, m]
pub fn inverse_jacobi_dc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseJacobiDC expects exactly 2 arguments".into(),
    ));
  }
  let v = &args[0];
  let m = &args[1];

  // InverseJacobiDC[1, m] = 0
  if is_expr_one(v) {
    return Ok(Expr::Integer(0));
  }

  // Numeric: EllipticF[ArcSin[Sqrt[(v^2-1)/(v^2-m)]], m]
  if let (Some(v_f), Some(m_f)) = (expr_to_f64(v), expr_to_f64(m))
    && has_real_arg(v, m)
  {
    let arg = ((v_f * v_f - 1.0) / (v_f * v_f - m_f)).sqrt();
    return Ok(Expr::Real(elliptic_f(arg.asin(), m_f)));
  }

  Ok(Expr::FunctionCall {
    name: "InverseJacobiDC".to_string(),
    args: args.to_vec(),
  })
}

/// Extract the inner expression from a negated expression like Times[-1, x] or -x
pub fn extract_negated_expr(expr: &Expr) -> Option<Expr> {
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
    return Ok(Expr::FunctionCall {
      name: "Tan".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiSC[u, 1] = Sinh[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sinh".to_string(),
      args: vec![u.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiSC".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Sec".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiDC[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiDC".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiCD[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiCD".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sinh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiSD".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Cot".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Csch".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiCS".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Csc".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Csch".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiDS".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Csc".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Coth".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiNS".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Cosh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiND".to_string(),
    args: args.to_vec(),
  })
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
    return Ok(Expr::FunctionCall {
      name: "Sec".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Cosh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiNC".to_string(),
    args: args.to_vec(),
  })
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
pub fn is_expr_one(expr: &Expr) -> bool {
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
