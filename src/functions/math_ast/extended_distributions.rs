//! Extended parametrizations of continuous distributions whose basic form is
//! handled elsewhere: the generalized gamma `GammaDistribution[α, β, γ, μ]`,
//! the generalized inverse gamma `InverseGammaDistribution[α, β, γ, μ]`, the
//! scaled `InverseChiSquareDistribution[ν, ξ]`, the four-parameter
//! `GompertzMakehamDistribution[λ, ξ, θ, α]` and the generalized inverse
//! Gaussian `InverseGaussianDistribution[μ, λ, θ]`.
//!
//! Each property is a closed form taken from the Wolfram reference pages,
//! written in terms of the parameters `p1`, `p2`, … and the argument `x`
//! (the point for PDF / CDF, the probability for Quantile), and instantiated
//! with one simultaneous substitution.

use crate::InterpreterError;
use crate::syntax::Expr;

/// The closed form of `property` for an extended parametrization, or `None`
/// when the form or the property is not one of these.
fn template(property: &str, name: &str, n: usize) -> Option<&'static str> {
  Some(match (name, n, property) {
    ("GammaDistribution", 4, "PDF") => {
      "Piecewise[{{(p3*((x - p4)/p2)^(-1 + p1*p3))/(E^((x - p4)/p2)^p3*(p2*Gamma[p1])), x > p4}}, 0]"
    }
    ("GammaDistribution", 4, "CDF") => {
      "Piecewise[{{GammaRegularized[p1, 0, ((x - p4)/p2)^p3], x > p4}}, 0]"
    }
    ("GammaDistribution", 4, "Mean") => "p4 + (p2*Gamma[p1 + 1/p3])/Gamma[p1]",
    ("GammaDistribution", 4, "Variance") => {
      "(p2^2*(-Gamma[p1 + 1/p3]^2 + Gamma[p1]*Gamma[p1 + 2/p3]))/Gamma[p1]^2"
    }
    ("GammaDistribution", 4, "Median") => {
      "p4 + p2*InverseGammaRegularized[p1, 1/2]^(1/p3)"
    }
    ("GammaDistribution", 4, "Quantile") => {
      "ConditionalExpression[Piecewise[{{p4 + p2*InverseGammaRegularized[p1, 0, x]^(1/p3), 0 < x < 1}, {p4, x <= 0}}, Infinity], 0 <= x <= 1]"
    }
    ("InverseGammaDistribution", 4, "PDF") => {
      "Piecewise[{{(p3*(p2/(x - p4))^(1 + p1*p3))/(E^(p2/(x - p4))^p3*(p2*Gamma[p1])), x > p4}}, 0]"
    }
    ("InverseGammaDistribution", 4, "CDF") => {
      "Piecewise[{{GammaRegularized[p1, (p2/(x - p4))^p3], x > p4}}, 0]"
    }
    ("InverseGammaDistribution", 4, "Mean") => {
      "Piecewise[{{p4 + (p2*Gamma[p1 - 1/p3])/Gamma[p1], p1*p3 > 1}}, Indeterminate]"
    }
    ("InverseGammaDistribution", 4, "Variance") => {
      "Piecewise[{{(p2^2*(Gamma[p1]*Gamma[p1 - 2/p3] - Gamma[p1 - 1/p3]^2))/Gamma[p1]^2, p1*p3 > 2}}, Indeterminate]"
    }
    ("InverseGammaDistribution", 4, "Median") => {
      "p4 + p2*(1/InverseGammaRegularized[p1, 1/2])^(1/p3)"
    }
    ("InverseGammaDistribution", 4, "Quantile") => {
      "ConditionalExpression[Piecewise[{{p4 + p2/InverseGammaRegularized[p1, x]^(1/p3), 0 < x < 1}, {p4, x <= 0}}, Infinity], 0 <= x <= 1]"
    }
    // The one-parameter form has these moments too, but no other code
    // gives them.
    ("InverseChiSquareDistribution", 1, "Mean") => {
      "Piecewise[{{1/(-2 + p1), p1 > 2}}, Indeterminate]"
    }
    ("InverseChiSquareDistribution", 1, "Variance") => {
      "Piecewise[{{2/((-4 + p1)*(-2 + p1)^2), p1 > 4}}, Indeterminate]"
    }
    ("InverseChiSquareDistribution", 2, "PDF") => {
      "Piecewise[{{((p1*p2)/x)^(p1/2)/(2^(p1/2)*E^((p1*p2)/(2*x))*(x*Gamma[p1/2])), x > 0}}, 0]"
    }
    ("InverseChiSquareDistribution", 2, "CDF") => {
      "Piecewise[{{GammaRegularized[p1/2, (p1*p2)/(2*x)], x > 0}}, 0]"
    }
    ("InverseChiSquareDistribution", 2, "Mean") => {
      "Piecewise[{{(p1*p2)/(-2 + p1), p1 > 2}}, Indeterminate]"
    }
    ("InverseChiSquareDistribution", 2, "Variance") => {
      "Piecewise[{{(2*p1^2*p2^2)/((-4 + p1)*(-2 + p1)^2), p1 > 4}}, Indeterminate]"
    }
    ("InverseChiSquareDistribution", 2, "Median") => {
      "(p1*p2)/(2*InverseGammaRegularized[p1/2, 1/2])"
    }
    ("InverseChiSquareDistribution", 2, "Quantile") => {
      "ConditionalExpression[Piecewise[{{(p1*p2)/(2*InverseGammaRegularized[p1/2, x]), 0 < x < 1}, {0, x <= 0}}, Infinity], 0 <= x <= 1]"
    }
    ("GompertzMakehamDistribution", 4, "PDF") => {
      "Piecewise[{{(p1*(E^(x*p1) + p3 + 2*x*p4*p1)*p2)/E^((-1 + E^(x*p1) + x*p3*p1 + x^2*p4*p1^2)*p2), x >= 0}}, 0]"
    }
    ("GompertzMakehamDistribution", 4, "CDF") => {
      "Piecewise[{{1 - E^((1 - E^(x*p1))*p2 - x*p3*p1*p2 - x^2*p4*p1^2*p2), x >= 0}}, 0]"
    }
    ("InverseGaussianDistribution", 3, "PDF") => {
      "Piecewise[{{x^(-1 + p3)/(E^((p2*(1 + x^2/p1^2))/(2*x))*p1^p3*(2*BesselK[p3, p2/p1])), x > 0}}, 0]"
    }
    ("InverseGaussianDistribution", 3, "Mean") => {
      "(p1*BesselK[1 + p3, p2/p1])/BesselK[p3, p2/p1]"
    }
    ("InverseGaussianDistribution", 3, "Variance") => {
      "(p1^2*(-BesselK[1 + p3, p2/p1]^2 + BesselK[p3, p2/p1]*BesselK[2 + p3, p2/p1]))/BesselK[p3, p2/p1]^2"
    }
    _ => return None,
  })
}

/// Whether `name[dargs]` is one of the extended parametrizations.
pub(crate) fn is_extended_form(name: &str, n: usize) -> bool {
  matches!(
    (name, n),
    (
      "GammaDistribution"
        | "InverseGammaDistribution"
        | "GompertzMakehamDistribution",
      4
    ) | ("InverseChiSquareDistribution", 2)
      | ("InverseGaussianDistribution", 3)
  )
}

/// `property[name[dargs], x]` (or `property[name[dargs]]` when `x` is
/// `None`) from its closed form. `None` when there is none and the form is
/// not extended; an extended form without a closed form for `property`
/// stays unevaluated.
pub(crate) fn extended_property(
  property: &str,
  name: &str,
  dargs: &[Expr],
  x: Option<&Expr>,
) -> Option<Result<Expr, InterpreterError>> {
  let Some(src) = template(property, name, dargs.len()) else {
    if !is_extended_form(name, dargs.len()) {
      return None;
    }
    // No closed form: the property stays unevaluated rather than falling
    // through to code written for the basic parametrization.
    let mut args = vec![crate::helpers::call(name, dargs.to_vec())];
    args.extend(x.cloned());
    return Some(Ok(crate::helpers::call(property, args)));
  };
  let body = match crate::syntax::string_to_expr(src) {
    Ok(body) => body,
    Err(e) => return Some(Err(e)),
  };
  const PARAMS: [&str; 4] = ["p1", "p2", "p3", "p4"];
  let mut bindings: Vec<(&str, &Expr)> =
    PARAMS.iter().copied().zip(dargs.iter()).collect();
  if let Some(x) = x {
    bindings.push(("x", x));
  }
  let instantiated = crate::syntax::substitute_variables(&body, &bindings);
  Some(crate::evaluator::evaluate_expr_to_expr(&instantiated))
}

/// A standard gamma variate of shape `alpha` (unit scale).
fn standard_gamma(alpha: f64) -> Option<impl Fn() -> f64> {
  let gamma = rand_distr::Gamma::new(alpha, 1.0).ok()?;
  Some(move || {
    crate::with_rng(|rng| rand_distr::Distribution::sample(&gamma, rng))
  })
}

fn uniform01() -> f64 {
  use rand::RngExt;
  crate::with_rng(|rng| rng.random::<f64>())
}

/// A sampler for an extended parametrization with numeric parameters, or
/// `None` when the parameters are not numeric or out of range.
pub(crate) fn extended_sampler(
  name: &str,
  dargs: &[Expr],
) -> Option<Box<dyn Fn() -> f64>> {
  if !is_extended_form(name, dargs.len()) {
    return None;
  }
  let p: Vec<f64> = dargs
    .iter()
    .map(crate::functions::math_ast::try_eval_to_f64)
    .collect::<Option<_>>()?;
  if p.iter().any(|v| !v.is_finite()) {
    return None;
  }
  match name {
    // μ + β G^(1/γ), G ~ Gamma[α, 1].
    "GammaDistribution" => {
      let (a, b, g, m) = (p[0], p[1], p[2], p[3]);
      if !(a > 0.0 && b > 0.0 && g > 0.0) {
        return None;
      }
      let draw = standard_gamma(a)?;
      Some(Box::new(move || m + b * draw().powf(1.0 / g)))
    }
    // μ + β G^(-1/γ), G ~ Gamma[α, 1].
    "InverseGammaDistribution" => {
      let (a, b, g, m) = (p[0], p[1], p[2], p[3]);
      if !(a > 0.0 && b > 0.0 && g > 0.0) {
        return None;
      }
      let draw = standard_gamma(a)?;
      Some(Box::new(move || m + b * draw().powf(-1.0 / g)))
    }
    // ν ξ / X with X ~ ChiSquare[ν] = 2 Gamma[ν/2, 1].
    "InverseChiSquareDistribution" => {
      let (nu, xi) = (p[0], p[1]);
      if !(nu > 0.0 && xi > 0.0) {
        return None;
      }
      let draw = standard_gamma(nu / 2.0)?;
      Some(Box::new(move || nu * xi / (2.0 * draw())))
    }
    // Inversion of the cumulative hazard
    // H(t) = ξ (E^(λ t) - 1) + θ λ ξ t + α λ² ξ t², which increases from 0.
    "GompertzMakehamDistribution" => {
      let (l, xi, th, al) = (p[0], p[1], p[2], p[3]);
      if !(l > 0.0 && xi > 0.0 && th >= 0.0 && al >= 0.0) {
        return None;
      }
      let hazard = move |t: f64| {
        xi * (l * t).exp_m1() + th * l * xi * t + al * l * l * xi * t * t
      };
      Some(Box::new(move || {
        let target = -(1.0 - uniform01()).ln();
        let mut hi = 1.0 / l;
        while hazard(hi) < target {
          hi *= 2.0;
        }
        let mut lo = 0.0;
        for _ in 0..200 {
          let mid = f64::midpoint(lo, hi);
          if hazard(mid) < target {
            lo = mid;
          } else {
            hi = mid;
          }
          if hi - lo <= 1e-15 * hi {
            break;
          }
        }
        f64::midpoint(lo, hi)
      }))
    }
    // Generalized inverse Gaussian: X = μ Y where Y has density
    // ∝ y^(θ-1) E^(-ω (y + 1/y)/2), ω = λ/μ, drawn by ratio-of-uniforms.
    "InverseGaussianDistribution" => {
      let (mu, lam, th) = (p[0], p[1], p[2]);
      if !(mu > 0.0 && lam > 0.0) {
        return None;
      }
      let omega = lam / mu;
      let log_f =
        move |y: f64| (th - 1.0) * y.ln() - omega * (y + 1.0 / y) / 2.0;
      // Maximize g(s) over s = ln y by golden-section search.
      let maximize = |g: &dyn Fn(f64) -> f64| -> f64 {
        let (mut a, mut b) = (-50.0_f64, 50.0_f64);
        let r = (5.0_f64.sqrt() - 1.0) / 2.0;
        for _ in 0..200 {
          let c = b - r * (b - a);
          let d = a + r * (b - a);
          if g(c) > g(d) {
            b = d;
          } else {
            a = c;
          }
        }
        g(f64::midpoint(a, b))
      };
      // Bounds of the region {(u, v): 0 < u <= sqrt f(v/u)}.
      let log_u_max = maximize(&|s: f64| 0.5 * log_f(s.exp()));
      let log_v_max = maximize(&|s: f64| s + 0.5 * log_f(s.exp()));
      Some(Box::new(move || {
        loop {
          let u = uniform01() * log_u_max.exp();
          let v = uniform01() * log_v_max.exp();
          if u <= 0.0 {
            continue;
          }
          let y = v / u;
          if y > 0.0 && 2.0 * u.ln() <= log_f(y) {
            return mu * y;
          }
        }
      }))
    }
    _ => None,
  }
}
