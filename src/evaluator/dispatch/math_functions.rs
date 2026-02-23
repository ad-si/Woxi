#[allow(unused_imports)]
use super::*;

pub fn dispatch_math_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    // AST-native math functions
    "Plus" => {
      return Some(crate::functions::math_ast::plus_ast(args));
    }
    "Times" => {
      return Some(crate::functions::math_ast::times_ast(args));
    }
    "Minus" => {
      return Some(crate::functions::math_ast::minus_ast(args));
    }
    "Subtract" if args.len() == 2 => {
      return Some(crate::functions::math_ast::subtract_ast(args));
    }
    "Divide" => {
      if args.len() == 2 {
        return Some(crate::functions::math_ast::divide_ast(args));
      } else {
        println!(
          "\nDivide::argrx: Divide called with {} arguments; 2 arguments are expected.",
          args.len()
        );
        use std::io::{self, Write};
        io::stdout().flush().ok();
        return Some(Ok(Expr::FunctionCall {
          name: "Divide".to_string(),
          args: args.to_vec(),
        }));
      }
    }
    "Power" if args.len() == 2 => {
      return Some(crate::functions::math_ast::power_ast(args));
    }
    "Max" => {
      return Some(crate::functions::math_ast::max_ast(args));
    }
    "Min" => {
      return Some(crate::functions::math_ast::min_ast(args));
    }
    "RankedMax" if args.len() == 2 => {
      if let Expr::List(items) = &args[0] {
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = expr_to_i128(&args[1]) {
          let idx = (k - 1) as usize;
          if idx < sorted.len() {
            return Some(Ok(sorted[idx].clone()));
          }
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "RankedMax".to_string(),
        args: args.to_vec(),
      }));
    }
    "RankedMin" if args.len() == 2 => {
      if let Expr::List(items) = &args[0] {
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = expr_to_i128(&args[1]) {
          let idx = (k - 1) as usize;
          if idx < sorted.len() {
            return Some(Ok(sorted[idx].clone()));
          }
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "RankedMin".to_string(),
        args: args.to_vec(),
      }));
    }
    "Quantile" if args.len() == 2 => {
      return Some(crate::functions::math_ast::quantile_ast(args));
    }
    "Quartiles" if args.len() == 1 => {
      // Quartiles uses Quantile with parameters {{1/2, 0}, {0, 1}}
      // Formula: pos = 1/2 + n*q, then linear interpolation
      if let Expr::List(items) = &args[0] {
        if items.is_empty() {
          return Some(Ok(Expr::FunctionCall {
            name: "Quartiles".to_string(),
            args: args.to_vec(),
          }));
        }
        // Sort numerically
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = sorted.len() as i128;
        let mut results = Vec::new();
        for (qn, qd) in [(1i128, 4i128), (1, 2), (3, 4)] {
          // pos = 1/2 + n * q = 1/2 + n*qn/qd
          // pos = (qd + 2*n*qn) / (2*qd)
          let pos_num = qd + 2 * n * qn;
          let pos_den = 2 * qd;
          let j = pos_num / pos_den; // Floor
          let frac_num = pos_num - j * pos_den; // remainder
          // frac = frac_num / pos_den
          if frac_num == 0 {
            // Exact position
            results.push(sorted[(j - 1) as usize].clone());
          } else {
            // Interpolate: (1 - frac)*sorted[j-1] + frac*sorted[j]
            // = ((pos_den - frac_num)*sorted[j-1] + frac_num*sorted[j]) / pos_den
            let lo = &sorted[(j - 1) as usize];
            let hi = &sorted[j as usize];
            let w_lo = pos_den - frac_num;
            let w_hi = frac_num;
            // Try integer arithmetic
            if let (Some(lo_v), Some(hi_v)) = (
              crate::functions::math_ast::try_eval_to_f64(lo),
              crate::functions::math_ast::try_eval_to_f64(hi),
            ) {
              let lo_i = lo_v as i128;
              let hi_i = hi_v as i128;
              if lo_i as f64 == lo_v && hi_i as f64 == hi_v {
                // Exact rational: (w_lo*lo_i + w_hi*hi_i) / pos_den
                let num = w_lo * lo_i + w_hi * hi_i;
                results.push(crate::functions::math_ast::make_rational(
                  num, pos_den,
                ));
              } else {
                results.push(crate::functions::math_ast::num_to_expr(
                  (w_lo as f64 * lo_v + w_hi as f64 * hi_v) / pos_den as f64,
                ));
              }
            } else {
              // Symbolic fallback
              results.push(Expr::FunctionCall {
                name: "Quartiles".to_string(),
                args: args.to_vec(),
              });
              break;
            }
          }
        }
        if results.len() == 3 {
          return Some(Ok(Expr::List(results)));
        }
      }
    }
    "Abs" if args.len() == 1 => {
      return Some(crate::functions::math_ast::abs_ast(args));
    }
    "RealAbs" if args.len() == 1 => {
      // RealAbs is same as Abs for real-valued arguments
      match &args[0] {
        Expr::Real(f) => return Some(Ok(Expr::Real(f.abs()))),
        Expr::Integer(n) => return Some(Ok(Expr::Integer(n.abs()))),
        _ => {
          if let Some(n) = crate::functions::math_ast::try_eval_to_f64(&args[0])
          {
            return Some(Ok(crate::functions::math_ast::num_to_expr(n.abs())));
          }
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "RealAbs".to_string(),
        args: args.to_vec(),
      }));
    }
    "RealSign" if args.len() == 1 => {
      match &args[0] {
        Expr::Real(f) => {
          return Some(Ok(Expr::Integer(if *f > 0.0 {
            1
          } else if *f < 0.0 {
            -1
          } else {
            0
          })));
        }
        Expr::Integer(n) => {
          return Some(Ok(Expr::Integer(if *n > 0 {
            1
          } else if *n < 0 {
            -1
          } else {
            0
          })));
        }
        Expr::FunctionCall {
          name: rname,
          args: rargs,
        } if rname == "Rational" && rargs.len() == 2 => {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
            let sign = if (*n > 0 && *d > 0) || (*n < 0 && *d < 0) {
              1
            } else if *n == 0 {
              0
            } else {
              -1
            };
            return Some(Ok(Expr::Integer(sign)));
          }
        }
        _ => {
          // Stay symbolic for complex or symbolic args
          return Some(Ok(Expr::FunctionCall {
            name: "RealSign".to_string(),
            args: args.to_vec(),
          }));
        }
      }
    }
    "Sign" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sign_ast(args));
    }
    "Sqrt" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sqrt_ast(args));
    }
    "Surd" if args.len() == 2 => {
      return Some(crate::functions::math_ast::surd_ast(args));
    }
    "Floor" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::floor_ast(args));
    }
    "Ceiling" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::ceiling_ast(args));
    }
    "Round" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::round_ast(args));
    }
    "Mod" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::mod_ast(args));
    }
    "Quotient" if args.len() == 2 => {
      return Some(crate::functions::math_ast::quotient_ast(args));
    }
    "QuotientRemainder" if args.len() == 2 => {
      let q = match crate::functions::math_ast::quotient_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let r = match crate::functions::math_ast::mod_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      return Some(Ok(Expr::List(vec![q, r])));
    }
    "GCD" => {
      return Some(crate::functions::math_ast::gcd_ast(args));
    }
    "ExtendedGCD" if args.len() >= 2 => {
      return Some(crate::functions::math_ast::extended_gcd_ast(args));
    }
    "LCM" => {
      return Some(crate::functions::math_ast::lcm_ast(args));
    }
    "Total" => {
      return Some(crate::functions::math_ast::total_ast(args));
    }
    "Fourier" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::fourier_ast(args));
    }
    "InverseFourier" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_fourier_ast(args));
    }
    "Mean" if args.len() == 1 => {
      return Some(crate::functions::math_ast::mean_ast(args));
    }
    "Variance" if args.len() == 1 => {
      return Some(crate::functions::math_ast::variance_ast(args));
    }
    "StandardDeviation" if args.len() == 1 => {
      return Some(crate::functions::math_ast::standard_deviation_ast(args));
    }
    "GeometricMean" if args.len() == 1 => {
      return Some(crate::functions::math_ast::geometric_mean_ast(args));
    }
    "HarmonicMean" if args.len() == 1 => {
      return Some(crate::functions::math_ast::harmonic_mean_ast(args));
    }
    "RootMeanSquare" if args.len() == 1 => {
      return Some(crate::functions::math_ast::root_mean_square_ast(args));
    }
    "Covariance" if args.len() == 2 => {
      return Some(crate::functions::math_ast::covariance_ast(args));
    }
    "Correlation" if args.len() == 2 => {
      return Some(crate::functions::math_ast::correlation_ast(args));
    }
    "CentralMoment" if args.len() == 2 => {
      return Some(crate::functions::math_ast::central_moment_ast(args));
    }
    "Kurtosis" if args.len() == 1 => {
      return Some(crate::functions::math_ast::kurtosis_ast(args));
    }
    "Skewness" if args.len() == 1 => {
      return Some(crate::functions::math_ast::skewness_ast(args));
    }
    "IntegerLength" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::integer_length_ast(args));
    }
    "IntegerReverse" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::integer_reverse_ast(args));
    }
    "Rescale" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::rescale_ast(args));
    }
    "Normalize" if args.len() == 1 => {
      return Some(crate::functions::math_ast::normalize_ast(args));
    }
    "Norm" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::norm_ast(args));
    }
    "EuclideanDistance" if args.len() == 2 => {
      return Some(crate::functions::math_ast::euclidean_distance_ast(args));
    }
    "ManhattanDistance" if args.len() == 2 => {
      return Some(crate::functions::math_ast::manhattan_distance_ast(args));
    }
    "SquaredEuclideanDistance" if args.len() == 2 => {
      return Some(crate::functions::math_ast::squared_euclidean_distance_ast(
        args,
      ));
    }
    "Factorial" if args.len() == 1 => {
      return Some(crate::functions::math_ast::factorial_ast(args));
    }
    "Factorial2" if args.len() == 1 => {
      return Some(crate::functions::math_ast::factorial2_ast(args));
    }
    "Subfactorial" if args.len() == 1 => {
      return Some(crate::functions::math_ast::subfactorial_ast(args));
    }
    "Pochhammer" if args.len() == 2 => {
      return Some(crate::functions::math_ast::pochhammer_ast(args));
    }
    "Gamma" if args.len() == 1 => {
      return Some(crate::functions::math_ast::gamma_ast(args));
    }
    "BesselJ" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_j_ast(args));
    }
    "BesselY" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_y_ast(args));
    }
    "AiryAi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_ai_ast(args));
    }
    "Hypergeometric0F1" if args.len() == 2 => {
      return Some(crate::functions::math_ast::hypergeometric_0f1_ast(args));
    }
    "Hypergeometric1F1" if args.len() == 3 => {
      return Some(crate::functions::math_ast::hypergeometric1f1_ast(args));
    }
    "Hypergeometric2F1" if args.len() == 4 => {
      return Some(crate::functions::math_ast::hypergeometric2f1_ast(args));
    }
    "HypergeometricU" if args.len() == 3 => {
      return Some(crate::functions::math_ast::hypergeometric_u_ast(args));
    }
    "EllipticK" if args.len() == 1 => {
      return Some(crate::functions::math_ast::elliptic_k_ast(args));
    }
    "EllipticE" if args.len() == 1 => {
      return Some(crate::functions::math_ast::elliptic_e_ast(args));
    }
    "EllipticF" if args.len() == 2 => {
      return Some(crate::functions::math_ast::elliptic_f_ast(args));
    }
    "EllipticPi" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::elliptic_pi_ast(args));
    }
    "Zeta" if args.len() == 1 => {
      return Some(crate::functions::math_ast::zeta_ast(args));
    }
    "PolyGamma" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::polygamma_ast(args));
    }
    "LegendreP" if args.len() == 2 => {
      return Some(crate::functions::math_ast::legendre_p_ast(args));
    }
    "JacobiP" if args.len() == 4 => {
      return Some(crate::functions::math_ast::jacobi_p_ast(args));
    }
    "SphericalHarmonicY" if args.len() == 4 => {
      return Some(crate::functions::math_ast::spherical_harmonic_y_ast(args));
    }
    "LegendreQ" if args.len() == 2 => {
      return Some(crate::functions::math_ast::legendre_q_ast(args));
    }
    "PolyLog" if args.len() == 2 => {
      return Some(crate::functions::math_ast::polylog_ast(args));
    }
    "LerchPhi" if args.len() == 3 => {
      return Some(crate::functions::math_ast::lerch_phi_ast(args));
    }
    "ExpIntegralEi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::exp_integral_ei_ast(args));
    }
    "ExpIntegralE" if args.len() == 2 => {
      return Some(crate::functions::math_ast::exp_integral_e_ast(args));
    }
    "BesselI" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_i_ast(args));
    }
    "BesselK" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_k_ast(args));
    }
    "EllipticTheta" if args.len() == 3 => {
      return Some(crate::functions::math_ast::elliptic_theta_ast(args));
    }
    "WeierstrassP" if args.len() == 2 => {
      return Some(crate::functions::math_ast::weierstrass_p_ast(args));
    }
    "WeierstrassPPrime" if args.len() == 2 => {
      return Some(crate::functions::math_ast::weierstrass_p_prime_ast(args));
    }
    "JacobiAmplitude" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_amplitude_ast(args));
    }
    "JacobiDN" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_dn_ast(args));
    }
    "JacobiSN" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_sn_ast(args));
    }
    "JacobiCN" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_cn_ast(args));
    }
    "InverseJacobiSN" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_sn_ast(args));
    }
    "InverseJacobiCN" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_cn_ast(args));
    }
    "InverseJacobiDN" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_dn_ast(args));
    }
    "InverseJacobiCD" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_cd_ast(args));
    }
    "InverseJacobiSC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_sc_ast(args));
    }
    "InverseJacobiCS" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_cs_ast(args));
    }
    "InverseJacobiSD" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_sd_ast(args));
    }
    "InverseJacobiDS" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_ds_ast(args));
    }
    "InverseJacobiNS" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_ns_ast(args));
    }
    "InverseJacobiNC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_nc_ast(args));
    }
    "InverseJacobiND" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_nd_ast(args));
    }
    "InverseJacobiDC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_jacobi_dc_ast(args));
    }
    "JacobiSC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_sc_ast(args));
    }
    "JacobiDC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_dc_ast(args));
    }
    "JacobiCD" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_cd_ast(args));
    }
    "JacobiSD" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_sd_ast(args));
    }
    "JacobiCS" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_cs_ast(args));
    }
    "JacobiDS" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_ds_ast(args));
    }
    "JacobiNS" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_ns_ast(args));
    }
    "JacobiND" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_nd_ast(args));
    }
    "JacobiNC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_nc_ast(args));
    }
    "ChebyshevT" if args.len() == 2 => {
      return Some(crate::functions::math_ast::chebyshev_t_ast(args));
    }
    "ChebyshevU" if args.len() == 2 => {
      return Some(crate::functions::math_ast::chebyshev_u_ast(args));
    }
    "GegenbauerC" if args.len() == 3 => {
      return Some(crate::functions::math_ast::gegenbauer_c_ast(args));
    }
    "LaguerreL" if args.len() == 2 => {
      return Some(crate::functions::math_ast::laguerre_l_ast(args));
    }
    "Beta" if args.len() == 2 => {
      return Some(crate::functions::math_ast::beta_ast(args));
    }
    "LogIntegral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::log_integral_ast(args));
    }
    "HermiteH" if args.len() == 2 => {
      return Some(crate::functions::math_ast::hermite_h_ast(args));
    }
    "N" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::n_ast(args));
    }
    "RandomInteger" => {
      return Some(crate::functions::math_ast::random_integer_ast(args));
    }
    "RandomReal" => {
      return Some(crate::functions::math_ast::random_real_ast(args));
    }
    "RandomChoice" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_choice_ast(args));
    }
    "RandomSample" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_sample_ast(args));
    }
    "RandomVariate" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_variate_ast(args));
    }
    "SeedRandom" if args.len() <= 1 => {
      return Some(crate::functions::math_ast::seed_random_ast(args));
    }
    "Clip" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::clip_ast(args));
    }
    "Sin" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sin_ast(args));
    }
    "Cos" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cos_ast(args));
    }
    "Tan" if args.len() == 1 => {
      return Some(crate::functions::math_ast::tan_ast(args));
    }
    "Sec" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sec_ast(args));
    }
    "Csc" if args.len() == 1 => {
      return Some(crate::functions::math_ast::csc_ast(args));
    }
    "Cot" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cot_ast(args));
    }
    "Exp" if args.len() == 1 => {
      return Some(crate::functions::math_ast::exp_ast(args));
    }
    "Erf" if args.len() == 1 => {
      return Some(crate::functions::math_ast::erf_ast(args));
    }
    "Erfc" if args.len() == 1 => {
      return Some(crate::functions::math_ast::erfc_ast(args));
    }
    "Log" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::log_ast(args));
    }
    "Log10" if args.len() == 1 => {
      return Some(crate::functions::math_ast::log10_ast(args));
    }
    "Log2" if args.len() == 1 => {
      return Some(crate::functions::math_ast::log2_ast(args));
    }
    "ArcSin" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arcsin_ast(args));
    }
    "ArcCos" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccos_ast(args));
    }
    "ArcTan" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arctan_ast(args));
    }
    "Sinh" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sinh_ast(args));
    }
    "Cosh" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cosh_ast(args));
    }
    "Tanh" if args.len() == 1 => {
      return Some(crate::functions::math_ast::tanh_ast(args));
    }
    "Coth" if args.len() == 1 => {
      return Some(crate::functions::math_ast::coth_ast(args));
    }
    "Sech" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sech_ast(args));
    }
    "Csch" if args.len() == 1 => {
      return Some(crate::functions::math_ast::csch_ast(args));
    }
    "ArcSinh" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arcsinh_ast(args));
    }
    "ArcCosh" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccosh_ast(args));
    }
    "ArcTanh" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arctanh_ast(args));
    }
    "ArcCoth" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccoth_ast(args));
    }
    "ArcSech" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arcsech_ast(args));
    }
    "ArcCot" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccot_ast(args));
    }
    "ArcCsc" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccsc_ast(args));
    }
    "ArcSec" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arcsec_ast(args));
    }
    "ArcCsch" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccsch_ast(args));
    }
    "Gudermannian" if args.len() == 1 => {
      return Some(crate::functions::math_ast::gudermannian_ast(args));
    }
    "InverseGudermannian" if args.len() == 1 => {
      return Some(crate::functions::math_ast::inverse_gudermannian_ast(args));
    }
    "LogisticSigmoid" if args.len() == 1 => {
      return Some(crate::functions::math_ast::logistic_sigmoid_ast(args));
    }
    "ProductLog" if args.len() == 1 => {
      return Some(crate::functions::math_ast::product_log_ast(args));
    }
    "Prime" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_ast(args));
    }
    "Fibonacci" if args.len() == 1 => {
      return Some(crate::functions::math_ast::fibonacci_ast(args));
    }
    "LinearRecurrence" if args.len() == 3 => {
      return Some(crate::functions::math_ast::linear_recurrence_ast(args));
    }
    "IntegerDigits" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::integer_digits_ast(args));
    }
    "RealDigits" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::real_digits_ast(args));
    }
    "FromDigits" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::from_digits_ast(args));
    }
    "IntegerName" if args.len() == 1 => {
      return Some(crate::functions::math_ast::integer_name_ast(args));
    }
    "RomanNumeral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::roman_numeral_ast(args));
    }
    "FactorInteger" if args.len() == 1 => {
      return Some(crate::functions::math_ast::factor_integer_ast(args));
    }
    "IntegerPartitions" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::integer_partitions_ast(args));
    }
    "Divisors" if args.len() == 1 => {
      return Some(crate::functions::math_ast::divisors_ast(args));
    }
    "DivisorSigma" if args.len() == 2 => {
      return Some(crate::functions::math_ast::divisor_sigma_ast(args));
    }
    "MoebiusMu" if args.len() == 1 => {
      return Some(crate::functions::math_ast::moebius_mu_ast(args));
    }
    "EulerPhi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::euler_phi_ast(args));
    }
    "JacobiSymbol" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_symbol_ast(args));
    }
    "CoprimeQ" if args.len() >= 2 => {
      return Some(crate::functions::math_ast::coprime_q_ast(args));
    }
    "Re" if args.len() == 1 => {
      return Some(crate::functions::math_ast::re_ast(args));
    }
    "Im" if args.len() == 1 => {
      return Some(crate::functions::math_ast::im_ast(args));
    }
    "Conjugate" if args.len() == 1 => {
      return Some(crate::functions::math_ast::conjugate_ast(args));
    }
    "Arg" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arg_ast(args));
    }
    "Rationalize" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::rationalize_ast(args));
    }
    "Numerator" if args.len() == 1 => {
      return Some(crate::functions::math_ast::numerator_ast(args));
    }
    "Denominator" if args.len() == 1 => {
      return Some(crate::functions::math_ast::denominator_ast(args));
    }
    "Binomial" if args.len() == 2 => {
      return Some(crate::functions::math_ast::binomial_ast(args));
    }
    "Multinomial" => {
      return Some(crate::functions::math_ast::multinomial_ast(args));
    }
    "PowerMod" if args.len() == 3 => {
      return Some(crate::functions::math_ast::power_mod_ast(args));
    }
    "PrimePi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_pi_ast(args));
    }
    "PartitionsP" if args.len() == 1 => {
      return Some(crate::functions::math_ast::partitions_p_ast(args));
    }
    "PartitionsQ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::partitions_q_ast(args));
    }
    "ArithmeticGeometricMean" if args.len() == 2 => {
      return Some(crate::functions::math_ast::arithmetic_geometric_mean_ast(
        args,
      ));
    }
    "NextPrime" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::next_prime_ast(args));
    }
    "ModularInverse" if args.len() == 2 => {
      return Some(crate::functions::math_ast::modular_inverse_ast(args));
    }
    "BitLength" if args.len() == 1 => {
      return Some(crate::functions::math_ast::bit_length_ast(args));
    }
    "BitAnd" if !args.is_empty() => {
      return Some(crate::functions::math_ast::bit_and_ast(args));
    }
    "BitOr" if !args.is_empty() => {
      return Some(crate::functions::math_ast::bit_or_ast(args));
    }
    "BitXor" if !args.is_empty() => {
      return Some(crate::functions::math_ast::bit_xor_ast(args));
    }
    "BitNot" if args.len() == 1 => {
      return Some(crate::functions::math_ast::bit_not_ast(args));
    }
    "IntegerExponent" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::integer_exponent_ast(args));
    }
    "IntegerPart" if args.len() == 1 => {
      return Some(crate::functions::math_ast::integer_part_ast(args));
    }
    "FractionalPart" if args.len() == 1 => {
      return Some(crate::functions::math_ast::fractional_part_ast(args));
    }
    "MixedFractionParts" if args.len() == 1 => {
      return Some(crate::functions::math_ast::mixed_fraction_parts_ast(args));
    }
    "Chop" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::chop_ast(args));
    }
    "PowerExpand" if args.len() == 1 => {
      return Some(crate::functions::math_ast::power_expand_ast(args));
    }
    "Variables" if args.len() == 1 => {
      return Some(crate::functions::math_ast::variables_ast(args));
    }
    "CubeRoot" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cube_root_ast(args));
    }
    "Subdivide" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::subdivide_ast(args));
    }
    "DigitCount" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::digit_count_ast(args));
    }
    "DigitSum" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::digit_sum_ast(args));
    }
    "ContinuedFraction" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::continued_fraction_ast(args));
    }
    "FromContinuedFraction" if args.len() == 1 => {
      return Some(crate::functions::math_ast::from_continued_fraction_ast(
        args,
      ));
    }
    "LucasL" if args.len() == 1 => {
      return Some(crate::functions::math_ast::lucas_l_ast(args));
    }
    "ChineseRemainder" if args.len() == 2 => {
      return Some(crate::functions::math_ast::chinese_remainder_ast(args));
    }
    "DivisorSum" if args.len() == 2 => {
      return Some(crate::functions::math_ast::divisor_sum_ast(args));
    }
    "BernoulliB" if args.len() == 1 => {
      return Some(crate::functions::math_ast::bernoulli_b_ast(args));
    }
    "EulerE" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::euler_e_ast(args));
    }
    "BellB" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::bell_b_ast(args));
    }
    "PauliMatrix" if args.len() == 1 => {
      return Some(crate::functions::math_ast::pauli_matrix_ast(args));
    }
    "CatalanNumber" if args.len() == 1 => {
      return Some(crate::functions::math_ast::catalan_number_ast(args));
    }
    "StirlingS1" if args.len() == 2 => {
      return Some(crate::functions::math_ast::stirling_s1_ast(args));
    }
    "StirlingS2" if args.len() == 2 => {
      return Some(crate::functions::math_ast::stirling_s2_ast(args));
    }
    "FrobeniusNumber" if args.len() == 1 => {
      return Some(crate::functions::math_ast::frobenius_number_ast(args));
    }
    "HarmonicNumber" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::harmonic_number_ast(args));
    }
    "CoefficientList" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::coefficient_list_ast(
        args,
      ));
    }
    _ => {}
  }
  None
}
