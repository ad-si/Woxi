//! WaveletPhi and WaveletPsi: scaling and wavelet functions.
//!
//! - HaarWavelet and ShannonWavelet have explicit formulas (returned
//!   symbolically for any x).
//! - The continuous families (MexicanHat, Morlet, Paul, DGaussian, Gabor)
//!   have explicit psi formulas; their phi does not exist.
//! - The compactly supported discrete families evaluate numerically via the
//!   cascade recursion phi(x) = 2 Sum_k a_k phi(2x - k) on a dyadic grid;
//!   symbolic arguments stay unevaluated (Wolfram returns an
//!   InterpolatingFunction there).
//! - MeyerWavelet and BattleLemarieWavelet evaluate numerically from their
//!   Fourier-domain definitions.

use super::filters::meyer_nu;
use super::{
  ContinuousWaveletSpec, WaveletSpec, parse_continuous_wavelet,
  parse_discrete_wavelet, unevaluated,
};
use crate::InterpreterError;
use crate::syntax::{Expr, expr_to_string};

fn num(e: &Expr) -> Option<f64> {
  crate::functions::math_ast::expr_to_num(e)
}

fn eval(e: Expr) -> Expr {
  crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
}

/// Parse and evaluate an internally generated Wolfram Language template with
/// `#x#` replaced by the rendered argument expression.
fn formula(template: &str, x: &Expr) -> Expr {
  let code = template.replace("#x#", &format!("({})", expr_to_string(x)));
  match crate::syntax::string_to_expr(&code) {
    Ok(e) => eval(e),
    Err(_) => Expr::Identifier("$Failed".to_string()),
  }
}

fn wrap_real_if_numeric(x: &Expr, v: f64) -> Expr {
  // Match precision behavior: numeric input gives machine reals.
  let _ = x;
  Expr::Real(v)
}

pub fn wavelet_psi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  wavelet_phi_psi_ast(args, false)
}

pub fn wavelet_phi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  wavelet_phi_psi_ast(args, true)
}

fn wavelet_phi_psi_ast(
  args: &[Expr],
  phi: bool,
) -> Result<Expr, InterpreterError> {
  let fname = if phi { "WaveletPhi" } else { "WaveletPsi" };
  let positional: Vec<&Expr> = args
    .iter()
    .filter(|a| !matches!(a, Expr::Rule { .. }))
    .collect();
  if positional.is_empty() || positional.len() > 3 {
    crate::emit_message(&format!(
      "{fname}::argt: {fname} called with an invalid number of arguments."
    ));
    return Ok(unevaluated(fname, args));
  }
  let wave = positional[0];
  // The pure-function form WaveletPhi[wave] would be an
  // InterpolatingFunction; only the explicit-formula families produce one.
  let Some(x) = positional.get(1) else {
    return Ok(unevaluated(fname, args));
  };
  let dual = matches!(positional.get(2), Some(Expr::String(s)) if s == "Dual");

  // Continuous families: explicit psi, no phi.
  if let Some(cspec) = parse_continuous_wavelet(wave) {
    if phi {
      crate::emit_message(&format!(
        "{}::nophi: The wavelet {} has no scaling function.",
        fname,
        expr_to_string(wave)
      ));
      return Ok(unevaluated(fname, args));
    }
    let result = continuous_psi(&cspec, x);
    // Inexact input gives a fully numeric result.
    if num(x).is_some() && matches!(x, Expr::Real(_) | Expr::BigFloat(_, _)) {
      return Ok(eval(Expr::FunctionCall {
        name: "N".to_string(),
        args: vec![result.clone()].into(),
      }));
    }
    return Ok(result);
  }

  let Some(spec) = parse_discrete_wavelet(wave) else {
    crate::emit_message(&format!(
      "{}::invw: {} is not a valid wavelet.",
      fname,
      expr_to_string(wave)
    ));
    return Ok(unevaluated(fname, args));
  };

  match &spec {
    WaveletSpec::Haar => Ok(haar_phi_psi(x, phi)),
    WaveletSpec::Shannon(_) => Ok(shannon_phi_psi(x, phi)),
    WaveletSpec::Meyer(n, _) => match num(x) {
      Some(t) => Ok(wrap_real_if_numeric(x, meyer_phi_psi_numeric(*n, t, phi))),
      None => Ok(unevaluated(fname, args)),
    },
    WaveletSpec::BattleLemarie(_, _) => match num(x) {
      Some(t) => {
        let filters = super::wavelet_filters(&spec).unwrap();
        Ok(wrap_real_if_numeric(
          x,
          cascade_eval(&filters, t, phi, dual),
        ))
      }
      None => Ok(unevaluated(fname, args)),
    },
    _ => match num(x) {
      Some(t) => {
        let Some(filters) = super::wavelet_filters(&spec) else {
          return Ok(unevaluated(fname, args));
        };
        Ok(wrap_real_if_numeric(
          x,
          cascade_eval(&filters, t, phi, dual),
        ))
      }
      None => Ok(unevaluated(fname, args)),
    },
  }
}

fn haar_phi_psi(x: &Expr, phi: bool) -> Expr {
  if phi {
    formula("Piecewise[{{1, 0 <= #x# < 1}}]", x)
  } else {
    formula("Piecewise[{{1, 0 <= #x# < 1/2}, {-1, 1/2 <= #x# < 1}}]", x)
  }
}

fn shannon_phi_psi(x: &Expr, phi: bool) -> Expr {
  if phi {
    // sinc(pi x)
    formula("Sinc[Pi #x#]", x)
  } else {
    // (Sin[2 Pi x] - Cos[Pi x]) / (Pi (1/2 - x))  — the doc formula
    // 2 (Sin[2 Pi x] - Cos[Pi x]) / (Pi - 2 Pi x)
    formula("(2 (Sin[2 Pi #x#] - Cos[Pi #x#]))/(Pi - 2 Pi #x#)", x)
  }
}

fn continuous_psi(spec: &ContinuousWaveletSpec, x: &Expr) -> Expr {
  match spec {
    ContinuousWaveletSpec::MexicanHat(sigma) => {
      let s = expr_to_string(sigma);
      let template = format!(
        "(2 (({s})^2 - #x#^2))/(Sqrt[3] ({s})^(5/2) Pi^(1/4)) * Exp[-#x#^2/(2 ({s})^2)]"
      );
      formula(&template, x)
    }
    ContinuousWaveletSpec::Morlet => formula(
      "(Cos[Pi Sqrt[2/Log[2]] #x#] - E^(-Pi^2/Log[2]))/(E^(#x#^2/2) Pi^(1/4))",
      x,
    ),
    ContinuousWaveletSpec::Gabor(omega) => {
      let w = expr_to_string(omega);
      let template = format!("Exp[I ({w}) #x# - #x#^2/2]/Pi^(1/4)");
      formula(&template, x)
    }
    ContinuousWaveletSpec::DGaussian(order) => {
      // (-1)^(n+1)/Sqrt[Gamma[n + 1/2]] D[Exp[-t^2/2], {t, n}]; the
      // derivative needs an explicit integer order.
      let Expr::Integer(n) = order else {
        return Expr::FunctionCall {
          name: "WaveletPsi".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "DGaussianWavelet".to_string(),
              args: vec![order.clone()].into(),
            },
            x.clone(),
          ]
          .into(),
        };
      };
      let template = format!(
        "(-1)^({n}+1)/Sqrt[Gamma[{n} + 1/2]] * (D[Exp[-wpsivar^2/2], {{wpsivar, {n}}}] /. wpsivar -> #x#)"
      );
      formula(&template, x)
    }
    ContinuousWaveletSpec::Paul(order) => {
      let m = expr_to_string(order);
      let template = format!(
        "(2^({m}) I^({m}) ({m})!)/Sqrt[Pi (2 ({m}))!] * (1 - I #x#)^(-(({m}) + 1))"
      );
      formula(&template, x)
    }
  }
}

/// Meyer scaling/wavelet function at a numeric point via the inverse
/// Fourier integral of the piecewise-defined transform.
fn meyer_phi_psi_numeric(order: u32, t: f64, phi: bool) -> f64 {
  let pi = std::f64::consts::PI;
  let steps = 6000;
  if phi {
    // PhiHat: 1 on |w| <= 2pi/3, cos(pi/2 nu(3|w|/(2pi) - 1)) in the band.
    // phi(t) = (1/pi) [ Integral_0^{2pi/3} cos(w t) dw
    //                 + Integral_{2pi/3}^{4pi/3} PhiHat cos(w t) dw ]
    let flat = if t.abs() < 1e-12 {
      2.0 * pi / 3.0
    } else {
      (2.0 * pi / 3.0 * t).sin() / t
    };
    let a = 2.0 * pi / 3.0;
    let b = 4.0 * pi / 3.0;
    let dw = (b - a) / steps as f64;
    let mut acc = 0.0;
    for i in 0..=steps {
      let w = a + i as f64 * dw;
      let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
      let m = (pi / 2.0 * meyer_nu(order, 3.0 * w / (2.0 * pi) - 1.0)).cos();
      acc += weight * m * (w * t).cos();
    }
    (flat + acc * dw) / pi
  } else {
    // PsiHat(w) = E^{i w/2} sin(pi/2 nu(3|w|/(2pi)-1)) on 2pi/3..4pi/3
    //           = E^{i w/2} cos(pi/2 nu(3|w|/(4pi)-1)) on 4pi/3..8pi/3.
    // psi(t) = (1/pi) Re Integral_0^inf PsiHat(w) e^{i w t} dw
    //        = (1/pi) Integral band m(w) cos(w (t + 1/2)) dw.
    let tt = t + 0.5;
    let mut acc = 0.0;
    let (a1, b1) = (2.0 * pi / 3.0, 4.0 * pi / 3.0);
    let dw1 = (b1 - a1) / steps as f64;
    for i in 0..=steps {
      let w = a1 + i as f64 * dw1;
      let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
      let m = (pi / 2.0 * meyer_nu(order, 3.0 * w / (2.0 * pi) - 1.0)).sin();
      acc += weight * m * (w * tt).cos() * dw1;
    }
    let (a2, b2) = (4.0 * pi / 3.0, 8.0 * pi / 3.0);
    let dw2 = (b2 - a2) / steps as f64;
    for i in 0..=steps {
      let w = a2 + i as f64 * dw2;
      let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
      let m = (pi / 2.0 * meyer_nu(order, 3.0 * w / (4.0 * pi) - 1.0)).cos();
      acc += weight * m * (w * tt).cos() * dw2;
    }
    acc / pi
  }
}

/// Evaluate phi (or psi) at a numeric point via the cascade algorithm on a
/// dyadic grid of spacing 2^-8 (the documented MaxRecursion default).
fn cascade_eval(
  filters: &super::filters::WaveletFilters,
  t: f64,
  phi: bool,
  dual: bool,
) -> f64 {
  let lo: Vec<(i64, f64)> = if dual {
    filters.dual_lo.clone()
  } else {
    filters.primal_lo.clone()
  };
  let hi: Vec<(i64, f64)> = if dual {
    super::filters::highpass_from(&filters.primal_lo)
  } else {
    super::filters::highpass_from(&filters.dual_lo)
  };
  let imin = lo.first().map_or(0, |p| p.0);
  let imax = lo.last().map_or(1, |p| p.0);
  let j = 8; // MaxRecursion
  let step = 1.0 / (1 << j) as f64;
  let len = ((imax - imin) as usize) << j;
  if len == 0 {
    return 0.0;
  }
  // phi values on the grid imin + k*step, k = 0..len
  let mut phi_vals = vec![0.0f64; len + 1];
  // Start from the box function on [imin, imin+1).
  for (k, v) in phi_vals.iter_mut().enumerate() {
    let x = imin as f64 + k as f64 * step;
    if (imin as f64..imin as f64 + 1.0).contains(&x) {
      *v = 1.0;
    }
  }
  fn grid_of(vals: &[f64], imin: i64, step: f64, x: f64) -> f64 {
    // Linear interpolation of the grid values at point x.
    let len = vals.len() - 1;
    let pos = (x - imin as f64) / step;
    if pos < 0.0 || pos > len as f64 {
      return 0.0;
    }
    let k = pos.floor() as usize;
    let frac = pos - k as f64;
    if k >= len {
      return vals[len];
    }
    vals[k] * (1.0 - frac) + vals[k + 1] * frac
  }
  for _ in 0..40 {
    let mut next = vec![0.0f64; len + 1];
    for (k, nv) in next.iter_mut().enumerate() {
      let x = imin as f64 + k as f64 * step;
      let mut acc = 0.0;
      for &(i, c) in &lo {
        acc += c * grid_of(&phi_vals, imin, step, 2.0 * x - i as f64);
      }
      *nv = 2.0 * acc;
    }
    phi_vals = next;
  }
  if phi {
    return grid_of(&phi_vals, imin, step, t);
  }
  // psi(x) = 2 Sum_k g_k phi(2x - k)
  let mut acc = 0.0;
  for &(i, c) in &hi {
    acc += c * grid_of(&phi_vals, imin, step, 2.0 * t - i as f64);
  }
  2.0 * acc
}
