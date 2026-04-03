#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::make_sqrt;

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
    "Divide" if args.len() == 2 => {
      return Some(crate::functions::math_ast::divide_ast(args));
    }
    "Power" if args.len() == 1 => {
      // OneIdentity: Power[x] -> x
      return Some(Ok(args[0].clone()));
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
    "InterquartileRange" if args.len() == 1 => {
      // InterquartileRange = Q3 - Q1
      // Call the Quartiles logic by dispatching
      if let Some(Ok(Expr::List(ref qs))) =
        dispatch_math_functions("Quartiles", args)
        && qs.len() == 3
      {
        let diff = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Minus,
          left: Box::new(qs[2].clone()),
          right: Box::new(qs[0].clone()),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&diff));
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
    "HammingWindow"
    | "HannWindow"
    | "BlackmanWindow"
    | "DirichletWindow"
    | "BartlettWindow"
    | "WelchWindow"
    | "CosineWindow"
    | "ConnesWindow"
    | "LanczosWindow"
    | "ExactBlackmanWindow"
      if args.len() == 1 =>
    {
      return Some(crate::functions::math_ast::window_function_ast(name, args));
    }
    "BandpassFilter" if args.len() >= 2 && args.len() <= 4 => {
      return Some(crate::functions::math_ast::bandpass_filter_ast(args));
    }
    "BandstopFilter" if args.len() >= 2 && args.len() <= 4 => {
      return Some(crate::functions::math_ast::bandstop_filter_ast(args));
    }
    "LowpassFilter" if args.len() >= 2 && args.len() <= 4 => {
      return Some(crate::functions::math_ast::lowpass_filter_ast(args));
    }
    "HighpassFilter" if args.len() >= 2 && args.len() <= 4 => {
      return Some(crate::functions::math_ast::highpass_filter_ast(args));
    }
    "Fourier" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::fourier_ast(args));
    }
    "InverseFourier" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_fourier_ast(args));
    }
    "ListFourierSequenceTransform" if args.len() == 2 => {
      return Some(
        crate::functions::math_ast::list_fourier_sequence_transform_ast(args),
      );
    }
    "Mean" if args.len() == 1 => {
      return Some(crate::functions::math_ast::mean_ast(args));
    }
    "LocationTest" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::location_test_ast(args));
    }
    "PearsonChiSquareTest" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::pearson_chi_square_test_ast(
        args,
      ));
    }
    "LatitudeLongitude" if args.len() == 1 => {
      return Some(crate::functions::math_ast::latitude_longitude_ast(args));
    }
    "Longitude" if args.len() == 1 => {
      return Some(crate::functions::math_ast::longitude_ast(args));
    }
    "Latitude" if args.len() == 1 => {
      return Some(crate::functions::math_ast::latitude_ast(args));
    }
    "GroupGenerators" if args.len() == 1 => {
      return Some(crate::functions::math_ast::group_generators_ast(args));
    }
    "Likelihood" if args.len() == 2 => {
      return Some(crate::functions::math_ast::likelihood_ast(args));
    }
    "DiscreteAsymptotic" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::math_ast::discrete_asymptotic_ast(args));
    }
    "Moment" if args.len() == 2 => {
      return Some(crate::functions::math_ast::moment_ast(args));
    }
    "Variance" if args.len() == 1 => {
      return Some(crate::functions::math_ast::variance_ast(args));
    }
    "StandardDeviation" if args.len() == 1 => {
      return Some(crate::functions::math_ast::standard_deviation_ast(args));
    }
    "Standardize" if !args.is_empty() && args.len() <= 3 => {
      return Some(standardize_ast(args));
    }
    "TrimmedMean" if args.len() == 2 => {
      // TrimmedMean[list, frac] — mean after removing frac fraction from each end
      if let Expr::List(elems) = &args[0]
        && let Some(frac) = expr_to_f64(&args[1])
      {
        let n = elems.len();
        let trim = (n as f64 * frac).round() as usize;
        if 2 * trim < n {
          // Sort elements
          let mut sorted: Vec<Expr> = elems.clone();
          sorted.sort_by(|a, b| {
            let fa = expr_to_f64(a).unwrap_or(0.0);
            let fb = expr_to_f64(b).unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
          });
          let trimmed = &sorted[trim..n - trim];
          let sum_expr = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: trimmed.to_vec(),
          };
          let result = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(sum_expr),
            right: Box::new(Expr::Integer(trimmed.len() as i128)),
          };
          return Some(evaluate_expr_to_expr(&result));
        }
      }
    }
    "WinsorizedMean" if args.len() == 2 => {
      // WinsorizedMean[list, frac] — replace extremes with boundary values then mean
      if let Expr::List(elems) = &args[0]
        && let Some(frac) = expr_to_f64(&args[1])
      {
        let n = elems.len();
        let trim = (n as f64 * frac).round() as usize;
        if 2 * trim < n {
          let mut sorted: Vec<Expr> = elems.clone();
          sorted.sort_by(|a, b| {
            let fa = expr_to_f64(a).unwrap_or(0.0);
            let fb = expr_to_f64(b).unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
          });
          // Replace bottom trim with sorted[trim], top trim with sorted[n-trim-1]
          let low = sorted[trim].clone();
          let high = sorted[n - trim - 1].clone();
          let mut winsorized = sorted.clone();
          for item in winsorized.iter_mut().take(trim) {
            *item = low.clone();
          }
          for item in winsorized.iter_mut().skip(n - trim) {
            *item = high.clone();
          }
          let sum_expr = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: winsorized,
          };
          let result = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(sum_expr),
            right: Box::new(Expr::Integer(n as i128)),
          };
          return Some(evaluate_expr_to_expr(&result));
        }
      }
    }
    "TrimmedVariance" if args.len() == 2 => {
      // TrimmedVariance[list, frac] — variance of trimmed data
      if let Expr::List(elems) = &args[0]
        && let Some(frac) = expr_to_f64(&args[1])
      {
        let n = elems.len();
        let trim = (n as f64 * frac).round() as usize;
        if 2 * trim < n {
          let mut sorted: Vec<Expr> = elems.clone();
          sorted.sort_by(|a, b| {
            let fa = expr_to_f64(a).unwrap_or(0.0);
            let fb = expr_to_f64(b).unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
          });
          let trimmed: Vec<Expr> = sorted[trim..n - trim].to_vec();
          return Some(crate::functions::math_ast::variance_ast(&[
            Expr::List(trimmed),
          ]));
        }
      }
    }
    "WinsorizedVariance" if args.len() == 2 => {
      // WinsorizedVariance[list, frac] — variance of winsorized data
      if let Expr::List(elems) = &args[0]
        && let Some(frac) = expr_to_f64(&args[1])
      {
        let n = elems.len();
        let trim = (n as f64 * frac).round() as usize;
        if 2 * trim < n {
          let mut sorted: Vec<Expr> = elems.clone();
          sorted.sort_by(|a, b| {
            let fa = expr_to_f64(a).unwrap_or(0.0);
            let fb = expr_to_f64(b).unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
          });
          let low = sorted[trim].clone();
          let high = sorted[n - trim - 1].clone();
          let mut winsorized = sorted;
          for item in winsorized.iter_mut().take(trim) {
            *item = low.clone();
          }
          for item in winsorized.iter_mut().skip(n - trim) {
            *item = high.clone();
          }
          return Some(crate::functions::math_ast::variance_ast(&[
            Expr::List(winsorized),
          ]));
        }
      }
    }
    "MeanDeviation" if args.len() == 1 => {
      return Some(crate::functions::math_ast::mean_deviation_ast(args));
    }
    "MedianDeviation" if args.len() == 1 => {
      return Some(crate::functions::math_ast::median_deviation_ast(args));
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
    "FactorialPower" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::factorial_power_ast(args));
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
    "BesselJZero" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_j_zero_ast(args));
    }
    "AiryAi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_ai_ast(args));
    }
    "AiryAiPrime" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_ai_prime_ast(args));
    }
    "AiryBi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_bi_ast(args));
    }
    "AiryBiPrime" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_bi_prime_ast(args));
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
    "EllipticE" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::elliptic_e_ast(args));
    }
    "EllipticF" if args.len() == 2 => {
      return Some(crate::functions::math_ast::elliptic_f_ast(args));
    }
    "EllipticPi" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::elliptic_pi_ast(args));
    }
    "JacobiZeta" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_zeta_ast(args));
    }
    "EllipticNomeQ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::elliptic_nome_q_ast(args));
    }
    "Zeta" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::zeta_ast(args));
    }
    "DirichletEta" if args.len() == 1 => {
      return Some(crate::functions::math_ast::dirichlet_eta_ast(args));
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
    "PolygonalNumber" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::polygonal_number_ast(args));
    }
    "PerfectNumber" if args.len() == 1 => {
      return Some(crate::functions::math_ast::perfect_number_ast(args));
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
    "CosIntegral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cos_integral_ast(args));
    }
    "SinIntegral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sin_integral_ast(args));
    }
    "SinhIntegral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sinh_integral_ast(args));
    }
    "CoshIntegral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cosh_integral_ast(args));
    }
    "BetaRegularized" if args.len() == 3 => {
      return Some(crate::functions::math_ast::beta_regularized_ast(args));
    }
    "GammaRegularized" if args.len() == 2 => {
      return Some(crate::functions::math_ast::gamma_regularized_ast(args));
    }
    "Hypergeometric1F1Regularized" if args.len() == 3 => {
      return Some(
        crate::functions::math_ast::hypergeometric_1f1_regularized_ast(args),
      );
    }
    "AppellF1" if args.len() == 6 => {
      return Some(crate::functions::math_ast::appell_f1_ast(args));
    }
    "AppellF2" if args.len() == 7 => {
      return Some(crate::functions::math_ast::appell_f2_ast(args));
    }
    "AppellF3" if args.len() == 7 => {
      return Some(crate::functions::math_ast::appell_f3_ast(args));
    }
    "AppellF4" if args.len() == 6 => {
      return Some(crate::functions::math_ast::appell_f4_ast(args));
    }
    "BesselI" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_i_ast(args));
    }
    "BesselK" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_k_ast(args));
    }
    "EllipticThetaPrime" if args.len() == 3 => {
      return Some(crate::functions::math_ast::elliptic_theta_prime_ast(args));
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
    "InverseWeierstrassP" if args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_weierstrass_p_ast(args));
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
    "RiemannR" if args.len() == 1 => {
      return Some(crate::functions::math_ast::riemann_r_ast(args));
    }
    "HypergeometricPFQ" if args.len() == 3 => {
      return Some(crate::functions::math_ast::hypergeometric_pfq_ast(args));
    }
    "MeijerG" if args.len() == 3 => {
      return Some(crate::functions::math_ast::meijer_g_ast(args));
    }
    "HypergeometricPFQRegularized" if args.len() == 3 => {
      return Some(
        crate::functions::math_ast::hypergeometric_pfq_regularized_ast(args),
      );
    }
    "Hypergeometric2F1Regularized" if args.len() == 4 => {
      return Some(
        crate::functions::math_ast::hypergeometric_2f1_regularized_ast(args),
      );
    }
    "QPochhammer" if args.len() == 3 => {
      return Some(crate::functions::math_ast::q_pochhammer_ast(args));
    }
    "SphericalBesselJ" if args.len() == 2 => {
      return Some(crate::functions::math_ast::spherical_bessel_j_ast(args));
    }
    "LogGamma" if args.len() == 1 => {
      return Some(crate::functions::math_ast::log_gamma_ast(args));
    }
    "HermiteH" if args.len() == 2 => {
      return Some(crate::functions::math_ast::hermite_h_ast(args));
    }
    "StruveH" if args.len() == 2 => {
      return Some(crate::functions::math_ast::struve_h_ast(args));
    }
    "AngerJ" if args.len() == 2 => {
      return Some(crate::functions::math_ast::anger_j_ast(args));
    }
    "WeberE" if args.len() == 2 => {
      return Some(crate::functions::math_ast::weber_e_ast(args));
    }
    "WignerD" if args.len() == 2 || args.len() == 4 => {
      return Some(crate::functions::math_ast::wigner_d_ast(args));
    }
    "StruveL" if args.len() == 2 => {
      return Some(crate::functions::math_ast::struve_l_ast(args));
    }
    "Hypergeometric0F1Regularized" if args.len() == 2 => {
      return Some(
        crate::functions::math_ast::hypergeometric_0f1_regularized_ast(args),
      );
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
    "PDF" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::pdf_ast(args));
    }
    "CDF" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::cdf_ast(args));
    }
    "Probability" if args.len() == 2 => {
      return Some(crate::functions::math_ast::probability_ast(args));
    }
    "Expectation" if args.len() == 2 => {
      return Some(crate::functions::math_ast::expectation_ast(args));
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
    "SinDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sin_degrees_ast(args));
    }
    "CosDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cos_degrees_ast(args));
    }
    "TanDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::tan_degrees_ast(args));
    }
    "CotDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::cot_degrees_ast(args));
    }
    "SecDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::sec_degrees_ast(args));
    }
    "CscDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::csc_degrees_ast(args));
    }
    "ArcSinDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arcsin_degrees_ast(args));
    }
    "ArcCosDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccos_degrees_ast(args));
    }
    "ArcTanDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arctan_degrees_ast(args));
    }
    "ArcCotDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccot_degrees_ast(args));
    }
    "ArcSecDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arcsec_degrees_ast(args));
    }
    "ArcCscDegrees" if args.len() == 1 => {
      return Some(crate::functions::math_ast::arccsc_degrees_ast(args));
    }
    "Sinc" if args.len() == 1 => {
      // Sinc[0] = 1
      let is_zero = match &args[0] {
        Expr::Integer(0) => true,
        Expr::Real(f) => *f == 0.0,
        _ => false,
      };
      if is_zero {
        return Some(Ok(Expr::Integer(1)));
      }
      // For numeric/exact args, compute Sin[x]/x
      let sin_result = crate::functions::math_ast::sin_ast(args);
      match sin_result {
        Ok(ref sin_val) => {
          // Only evaluate if Sin actually computed to a value (not symbolic Sin[x])
          let is_symbolic = matches!(sin_val,
            Expr::FunctionCall { name, .. } if name == "Sin");
          if !is_symbolic {
            let div_expr = Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(sin_val.clone()),
              right: Box::new(args[0].clone()),
            };
            return Some(crate::evaluator::evaluate_expr_to_expr(&div_expr));
          }
        }
        Err(e) => return Some(Err(e)),
      }
    }
    "Haversine" if args.len() == 1 => {
      // Haversine[x] = (1 - Cos[x]) / 2
      let cos_expr = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: args.to_vec(),
      };
      let expr = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Minus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(cos_expr),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      return Some(crate::evaluator::evaluate_expr_to_expr(&expr));
    }
    "InverseHaversine" if args.len() == 1 => {
      // InverseHaversine[x] = 2 * ArcSin[Sqrt[x]]
      let sqrt_expr = make_sqrt(args[0].clone());
      let asin_expr = Expr::FunctionCall {
        name: "ArcSin".to_string(),
        args: vec![sqrt_expr],
      };
      let expr = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(asin_expr),
      };
      return Some(crate::evaluator::evaluate_expr_to_expr(&expr));
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
    "Erfi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::erfi_ast(args));
    }
    "InverseErf" if args.len() == 1 => {
      return Some(crate::functions::math_ast::inverse_erf_ast(args));
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
    "ArcTan" if args.len() == 2 => {
      return Some(crate::functions::math_ast::arctan2_ast(args));
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
    "PrimeOmega" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_omega_ast(args));
    }
    "PrimeNu" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_nu_ast(args));
    }
    "MantissaExponent" if args.len() == 1 => match &args[0] {
      Expr::Real(f) => {
        if *f == 0.0 {
          return Some(Ok(Expr::List(vec![Expr::Real(0.0), Expr::Integer(0)])));
        }
        let abs_f = f.abs();
        let e = abs_f.log10().floor() as i128 + 1;
        let m = f / 10.0_f64.powi(e as i32);
        return Some(Ok(Expr::List(vec![Expr::Real(m), Expr::Integer(e)])));
      }
      Expr::Integer(n) => {
        if *n == 0 {
          return Some(Ok(Expr::List(vec![
            Expr::Integer(0),
            Expr::Integer(0),
          ])));
        }
        let abs_n = n.unsigned_abs();
        let e = (abs_n as f64).log10().floor() as i128 + 1;
        let denom = 10_i128.pow(e as u32);
        let mantissa = crate::functions::math_ast::make_rational_pub(*n, denom);
        return Some(Ok(Expr::List(vec![mantissa, Expr::Integer(e)])));
      }
      _ => {}
    },
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
    "MangoldtLambda" if args.len() == 1 => {
      // MangoldtLambda[n] = Log[p] if n = p^k for prime p, else 0
      if let Expr::Integer(n) = &args[0] {
        if *n <= 1 {
          return Some(Ok(Expr::Integer(0)));
        }
        let factors = crate::functions::math_ast::factor_integer_ast(args);
        if let Ok(Expr::List(ref pairs)) = factors {
          // Filter out {-1, 1} factor
          let real_factors: Vec<&Expr> = pairs
            .iter()
            .filter(|p| {
              if let Expr::List(pv) = p
                && let Expr::Integer(base) = &pv[0]
              {
                return *base != -1;
              }
              true
            })
            .collect();
          if real_factors.len() == 1 {
            // Only one prime factor -> n = p^k
            if let Expr::List(pv) = real_factors[0]
              && let Expr::Integer(p) = &pv[0]
            {
              return Some(Ok(Expr::FunctionCall {
                name: "Log".to_string(),
                args: vec![Expr::Integer(*p)],
              }));
            }
          }
          return Some(Ok(Expr::Integer(0)));
        }
      }
    }
    "LiouvilleLambda" if args.len() == 1 => {
      // LiouvilleLambda[n] = (-1)^Omega(n) where Omega counts prime factors with multiplicity
      if let Expr::Integer(n) = &args[0] {
        if *n == 0 {
          return Some(Ok(Expr::Integer(0)));
        }
        let omega = crate::functions::math_ast::prime_omega_ast(args);
        if let Ok(Expr::Integer(total)) = omega {
          return Some(Ok(Expr::Integer(if total % 2 == 0 { 1 } else { -1 })));
        }
      }
    }
    "EulerPhi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::euler_phi_ast(args));
    }
    "JacobiSymbol" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_symbol_ast(args));
    }
    "MultiplicativeOrder" if args.len() == 2 => {
      if let (Expr::Integer(a), Expr::Integer(n)) = (&args[0], &args[1])
        && *n > 0
      {
        let a_mod = ((*a % *n) + *n) % *n;
        if a_mod != 0 && crate::functions::math_ast::gcd_i128(a_mod, *n) == 1 {
          let mut power = a_mod;
          for k in 1..=*n {
            if power == 1 {
              return Some(Ok(Expr::Integer(k)));
            }
            power = (power * a_mod) % *n;
          }
        }
      }
    }
    // PrimitiveRoot[n] — smallest primitive root modulo n
    "PrimitiveRoot" if args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        if *n <= 1 {
          crate::emit_message(&format!(
            "PrimitiveRoot::intg: Integer greater than 1 expected at position 1 in PrimitiveRoot[{}].",
            n
          ));
          return Some(Ok(Expr::FunctionCall {
            name: "PrimitiveRoot".to_string(),
            args: args.to_vec(),
          }));
        }
        let n_val = *n;
        // Compute EulerPhi[n]
        let phi = crate::functions::math_ast::euler_phi_i128(n_val);
        // Find smallest g >= 2 with multiplicative order == phi
        // (or g=1 for n=2)
        let start = if n_val == 2 { 1 } else { 2 };
        for g in start..n_val {
          if crate::functions::math_ast::gcd_i128(g, n_val) != 1 {
            continue;
          }
          // Check multiplicative order of g mod n
          let mut power = g % n_val;
          let mut order = 1i128;
          while power != 1 && order <= phi {
            power = (power * g) % n_val;
            order += 1;
          }
          if power == 1 && order == phi {
            return Some(Ok(Expr::Integer(g)));
          }
        }
        // No primitive root exists (e.g., n=8)
        return Some(Ok(Expr::FunctionCall {
          name: "PrimitiveRoot".to_string(),
          args: args.to_vec(),
        }));
      }
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
    "ReIm" if args.len() == 1 => {
      let re = match crate::functions::math_ast::re_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let im = match crate::functions::math_ast::im_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      return Some(Ok(Expr::List(vec![re, im])));
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
    "BitShiftRight" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::bit_shift_right_ast(args));
    }
    "BitShiftLeft" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::bit_shift_left_ast(args));
    }
    "BitFlip" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bit_flip_ast(args));
    }
    "BitSet" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bit_set_ast(args));
    }
    "BitClear" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bit_clear_ast(args));
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
    "PowerExpand" if args.len() == 1 || args.len() == 2 => {
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
    "Convergents" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::convergents_ast(args));
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
    "NorlundB" if args.len() == 2 => {
      return Some(crate::functions::math_ast::norlund_b_ast(args));
    }
    "PrimeZetaP" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_zeta_p_ast(args));
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
    "ExpToTrig" if args.len() == 1 => {
      return Some(exp_to_trig_ast(&args[0]));
    }
    "TrigToExp" if args.len() == 1 => {
      return Some(trig_to_exp_ast(&args[0]));
    }
    "TrigExpand" if args.len() == 1 => {
      return Some(crate::functions::math_ast::trig_expand_ast(args));
    }
    "TrigReduce" if args.len() == 1 => {
      return Some(crate::functions::math_ast::trig_reduce_ast(args));
    }
    "SquareWave" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::square_wave_ast(args));
    }
    "TriangleWave" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::triangle_wave_ast(args));
    }
    "SawtoothWave" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::sawtooth_wave_ast(args));
    }
    "ParabolicCylinderD" if args.len() == 2 => {
      return Some(crate::functions::math_ast::parabolic_cylinder_d_ast(args));
    }
    "FromPolarCoordinates" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0]
        && elems.len() == 2
      {
        let r = &elems[0];
        let theta = &elems[1];
        let x = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(r.clone()),
          right: Box::new(Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![theta.clone()],
          }),
        };
        let y = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(r.clone()),
          right: Box::new(Expr::FunctionCall {
            name: "Sin".to_string(),
            args: vec![theta.clone()],
          }),
        };
        let result = Expr::List(vec![x, y]);
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
    }
    "ToPolarCoordinates" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0]
        && elems.len() == 2
      {
        let x = &elems[0];
        let y = &elems[1];
        let r = make_sqrt(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(x.clone()),
            right: Box::new(Expr::Integer(2)),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(y.clone()),
            right: Box::new(Expr::Integer(2)),
          }),
        });
        let theta = Expr::FunctionCall {
          name: "ArcTan".to_string(),
          args: vec![x.clone(), y.clone()],
        };
        let result = Expr::List(vec![r, theta]);
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
    }
    "FromSphericalCoordinates" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0]
        && elems.len() == 3
      {
        let r = &elems[0];
        let theta = &elems[1];
        let phi = &elems[2];
        let x = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(r.clone()),
            right: Box::new(Expr::FunctionCall {
              name: "Sin".to_string(),
              args: vec![theta.clone()],
            }),
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![phi.clone()],
          }),
        };
        let y = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(r.clone()),
            right: Box::new(Expr::FunctionCall {
              name: "Sin".to_string(),
              args: vec![theta.clone()],
            }),
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Sin".to_string(),
            args: vec![phi.clone()],
          }),
        };
        let z = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(r.clone()),
          right: Box::new(Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![theta.clone()],
          }),
        };
        let result = Expr::List(vec![x, y, z]);
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
    }
    "ToSphericalCoordinates" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0]
        && elems.len() == 3
      {
        let x = &elems[0];
        let y = &elems[1];
        let z = &elems[2];
        let sum_sq = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(x.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(y.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(z.clone()),
            right: Box::new(Expr::Integer(2)),
          }),
        };
        let r = make_sqrt(sum_sq);
        let xy_sq = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(x.clone()),
            right: Box::new(Expr::Integer(2)),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(y.clone()),
            right: Box::new(Expr::Integer(2)),
          }),
        };
        let theta = Expr::FunctionCall {
          name: "ArcTan".to_string(),
          args: vec![z.clone(), make_sqrt(xy_sq)],
        };
        let phi_expr = Expr::FunctionCall {
          name: "ArcTan".to_string(),
          args: vec![x.clone(), y.clone()],
        };
        let result = Expr::List(vec![r, theta, phi_expr]);
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
    }
    "ContinuedFractionK" if args.len() == 2 => {
      // ContinuedFractionK[f, {i, imin, imax}]
      if let Expr::List(ref spec) = args[1]
        && spec.len() == 3
        && let Expr::Identifier(var) = &spec[0]
      {
        let imin_expr = crate::evaluator::evaluate_expr_to_expr(&spec[1])
          .unwrap_or_else(|_| spec[1].clone());
        let imax_expr = crate::evaluator::evaluate_expr_to_expr(&spec[2])
          .unwrap_or_else(|_| spec[2].clone());
        if let (Expr::Integer(imin), Expr::Integer(imax)) =
          (&imin_expr, &imax_expr)
        {
          let body = &args[0];
          let mut acc = Expr::Integer(0);
          for k in (*imin..=*imax).rev() {
            let fk = crate::syntax::replace_identifier_in_expr(
              body,
              var,
              &Expr::Integer(k),
            );
            let fk_eval =
              crate::evaluator::evaluate_expr_to_expr(&fk).unwrap_or(fk);
            let denom = Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(fk_eval),
              right: Box::new(acc),
            };
            acc = Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(denom),
            };
            acc = crate::evaluator::evaluate_expr_to_expr(&acc).unwrap_or(acc);
          }
          return Some(Ok(acc));
        }
      }
    }
    "FindLinearRecurrence" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0] {
        return Some(find_linear_recurrence_impl(elems));
      }
    }
    // SSSTriangle[a, b, c] — triangle from three side lengths
    "SSSTriangle" if args.len() == 3 => {
      // Place A at origin, B at (c, 0), find C via law of cosines
      let a = &args[0];
      let b = &args[1];
      let c = &args[2];
      // cos(A) = (b^2 + c^2 - a^2) / (2*b*c)
      let cos_a = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(b.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(c.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(a.clone()),
            right: Box::new(Expr::Integer(2)),
          }),
        }),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(b.clone()),
            right: Box::new(c.clone()),
          }),
        }),
      };
      // cx = b * cos(A)
      let cx = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(b.clone()),
        right: Box::new(cos_a.clone()),
      };
      // cy = b * sin(A) = b * sqrt(1 - cos(A)^2)
      let cy = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(b.clone()),
        right: Box::new(make_sqrt(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(cos_a),
            right: Box::new(Expr::Integer(2)),
          }),
        })),
      };
      let cx_eval = crate::evaluator::evaluate_expr_to_expr(&cx).unwrap_or(cx);
      let cy_eval = crate::evaluator::evaluate_expr_to_expr(&cy).unwrap_or(cy);
      let c_eval = crate::evaluator::evaluate_expr_to_expr(c)
        .unwrap_or_else(|_| c.clone());
      let triangle = Expr::FunctionCall {
        name: "Triangle".to_string(),
        args: vec![Expr::List(vec![
          Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]),
          Expr::List(vec![c_eval, Expr::Integer(0)]),
          Expr::List(vec![cx_eval, cy_eval]),
        ])],
      };
      return Some(Ok(triangle));
    }
    // ExponentialMovingAverage[list, alpha]
    "ExponentialMovingAverage" if args.len() == 2 => {
      if let Expr::List(ref elems) = args[0]
        && !elems.is_empty()
      {
        let alpha = &args[1];
        let one_minus_alpha = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(alpha.clone()),
        };
        let one_minus_alpha_eval =
          crate::evaluator::evaluate_expr_to_expr(&one_minus_alpha)
            .unwrap_or(one_minus_alpha);
        let mut ema = elems[0].clone();
        let mut result = vec![ema.clone()];
        for elem in &elems[1..] {
          // ema = alpha * x + (1 - alpha) * ema
          let new_ema = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(alpha.clone()),
              right: Box::new(elem.clone()),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(one_minus_alpha_eval.clone()),
              right: Box::new(ema),
            }),
          };
          ema = crate::evaluator::evaluate_expr_to_expr(&new_ema)
            .unwrap_or(new_ema);
          result.push(ema.clone());
        }
        return Some(Ok(Expr::List(result)));
      }
    }
    // CircleThrough[{p1, p2, p3}] — circumscribed circle through 3 points
    "CircleThrough" if args.len() == 1 => {
      if let Expr::List(ref pts) = args[0]
        && pts.len() == 3
      {
        // Extract coordinates
        let coords: Vec<(&Expr, &Expr)> = pts
          .iter()
          .filter_map(|p| {
            if let Expr::List(c) = p
              && c.len() == 2
            {
              return Some((&c[0], &c[1]));
            }
            None
          })
          .collect();
        if coords.len() == 3 {
          let (x1, y1) = coords[0];
          let (x2, y2) = coords[1];
          let (x3, y3) = coords[2];
          // Build the circumcenter formula as AST and evaluate
          // D = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
          let d_expr = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(x1.clone()),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Minus,
                    left: Box::new(y2.clone()),
                    right: Box::new(y3.clone()),
                  }),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(x2.clone()),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Minus,
                    left: Box::new(y3.clone()),
                    right: Box::new(y1.clone()),
                  }),
                }),
              }),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(x3.clone()),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Minus,
                  left: Box::new(y1.clone()),
                  right: Box::new(y2.clone()),
                }),
              }),
            }),
          };
          // sq(p) = x^2 + y^2
          let sq = |x: &Expr, y: &Expr| Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(x.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(y.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          // h_num = sq1*(y2-y3) + sq2*(y3-y1) + sq3*(y1-y2)
          let h_num = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(sq(x1, y1)),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Minus,
                  left: Box::new(y2.clone()),
                  right: Box::new(y3.clone()),
                }),
              }),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(sq(x2, y2)),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Minus,
                  left: Box::new(y3.clone()),
                  right: Box::new(y1.clone()),
                }),
              }),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(sq(x3, y3)),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Minus,
                left: Box::new(y1.clone()),
                right: Box::new(y2.clone()),
              }),
            }),
          };
          // k_num = sq1*(x3-x2) + sq2*(x1-x3) + sq3*(x2-x1)
          let k_num = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(sq(x1, y1)),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Minus,
                  left: Box::new(x3.clone()),
                  right: Box::new(x2.clone()),
                }),
              }),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(sq(x2, y2)),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Minus,
                  left: Box::new(x1.clone()),
                  right: Box::new(x3.clone()),
                }),
              }),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(sq(x3, y3)),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Minus,
                left: Box::new(x2.clone()),
                right: Box::new(x1.clone()),
              }),
            }),
          };
          let h = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(h_num),
            right: Box::new(d_expr.clone()),
          };
          let k = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(k_num),
            right: Box::new(d_expr),
          };
          let h_eval = crate::evaluator::evaluate_expr_to_expr(&h).unwrap_or(h);
          let k_eval = crate::evaluator::evaluate_expr_to_expr(&k).unwrap_or(k);
          // r = sqrt((x1-h)^2 + (y1-k)^2)
          let r = make_sqrt(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Minus,
                left: Box::new(x1.clone()),
                right: Box::new(h_eval.clone()),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Minus,
                left: Box::new(y1.clone()),
                right: Box::new(k_eval.clone()),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
          });
          let r_eval = crate::evaluator::evaluate_expr_to_expr(&r).unwrap_or(r);
          return Some(Ok(Expr::FunctionCall {
            name: "Circle".to_string(),
            args: vec![Expr::List(vec![h_eval, k_eval]), r_eval],
          }));
        }
      }
    }
    "CoordinateBounds" if args.len() == 1 => {
      // CoordinateBounds[{{x1,y1,...}, {x2,y2,...}, ...}] returns {{xmin,xmax}, {ymin,ymax}, ...}
      if let Expr::List(points) = &args[0]
        && !points.is_empty()
        && let Expr::List(first) = &points[0]
      {
        let dim = first.len();
        let mut mins: Vec<Expr> = first.clone();
        let mut maxs: Vec<Expr> = first.clone();
        for pt in &points[1..] {
          if let Expr::List(coords) = pt
            && coords.len() == dim
          {
            for d in 0..dim {
              let less_than_min = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Less".to_string(),
                args: vec![coords[d].clone(), mins[d].clone()],
              });
              if let Ok(Expr::Identifier(ref s)) = less_than_min
                && s == "True"
              {
                mins[d] = coords[d].clone();
              }
              let greater_than_max =
                evaluate_expr_to_expr(&Expr::FunctionCall {
                  name: "Greater".to_string(),
                  args: vec![coords[d].clone(), maxs[d].clone()],
                });
              if let Ok(Expr::Identifier(ref s)) = greater_than_max
                && s == "True"
              {
                maxs[d] = coords[d].clone();
              }
            }
          }
        }
        let bounds: Vec<Expr> = (0..dim)
          .map(|d| Expr::List(vec![mins[d].clone(), maxs[d].clone()]))
          .collect();
        return Some(Ok(Expr::List(bounds)));
      }
    }
    "ChessboardDistance" | "ChebyshevDistance" if args.len() == 2 => {
      // ChessboardDistance[{a1,...,an}, {b1,...,bn}] = Max[Abs[a1-b1], ..., Abs[an-bn]]
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1])
        && a.len() == b.len()
        && !a.is_empty()
      {
        let diffs: Vec<Expr> = a
          .iter()
          .zip(b.iter())
          .map(|(ai, bi)| Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Integer(-1)),
                right: Box::new(bi.clone()),
              }),
            }],
          })
          .collect();
        let max_expr = Expr::FunctionCall {
          name: "Max".to_string(),
          args: diffs,
        };
        return Some(evaluate_expr_to_expr(&max_expr));
      }
    }
    "BrayCurtisDistance" if args.len() == 2 => {
      // BrayCurtisDistance[u, v] = Total[Abs[u - v]] / Total[Abs[u + v]]
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1])
        && a.len() == b.len()
        && !a.is_empty()
      {
        let mut num_terms = Vec::new();
        let mut den_terms = Vec::new();
        for (ai, bi) in a.iter().zip(b.iter()) {
          num_terms.push(Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Integer(-1)),
                right: Box::new(bi.clone()),
              }),
            }],
          });
          den_terms.push(Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(bi.clone()),
            }],
          });
        }
        let num = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: num_terms,
        };
        let den = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: den_terms,
        };
        let result = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(num),
          right: Box::new(den),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    "CanberraDistance" if args.len() == 2 => {
      // CanberraDistance[u, v] = Sum[Abs[ui - vi] / (Abs[ui] + Abs[vi])]
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1])
        && a.len() == b.len()
        && !a.is_empty()
      {
        let mut terms = Vec::new();
        for (ai, bi) in a.iter().zip(b.iter()) {
          let num = Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Integer(-1)),
                right: Box::new(bi.clone()),
              }),
            }],
          };
          let den = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![ai.clone()],
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![bi.clone()],
            }),
          };
          terms.push(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(num),
            right: Box::new(den),
          });
        }
        let sum = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: terms,
        };
        return Some(evaluate_expr_to_expr(&sum));
      }
    }
    "CosineDistance" if args.len() == 2 => {
      // CosineDistance[u, v] = 1 - (u.v) / (Norm[u] * Norm[v])
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1])
        && a.len() == b.len()
        && !a.is_empty()
      {
        let dot = Expr::FunctionCall {
          name: "Dot".to_string(),
          args: vec![args[0].clone(), args[1].clone()],
        };
        let norm_a = Expr::FunctionCall {
          name: "Norm".to_string(),
          args: vec![args[0].clone()],
        };
        let norm_b = Expr::FunctionCall {
          name: "Norm".to_string(),
          args: vec![args[1].clone()],
        };
        let result = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(dot),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(norm_a),
                right: Box::new(norm_b),
              }),
            }),
          }),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    "MaxFilter" if args.len() == 2 => {
      // MaxFilter[list, r] — replace each element with the max in a window of radius r
      if let (Expr::List(elems), Some(r)) = (&args[0], expr_to_i128(&args[1])) {
        let r = r as usize;
        let n = elems.len();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
          let lo = i.saturating_sub(r);
          let hi = if i + r < n { i + r } else { n - 1 };
          let window: Vec<Expr> = elems[lo..=hi].to_vec();
          let max_val = evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "Max".to_string(),
            args: window,
          })
          .unwrap_or(elems[i].clone());
          result.push(max_val);
        }
        return Some(Ok(Expr::List(result)));
      }
    }
    "MinFilter" if args.len() == 2 => {
      if let (Expr::List(elems), Some(r)) = (&args[0], expr_to_i128(&args[1])) {
        let r = r as usize;
        let n = elems.len();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
          let lo = i.saturating_sub(r);
          let hi = if i + r < n { i + r } else { n - 1 };
          let window: Vec<Expr> = elems[lo..=hi].to_vec();
          let min_val = evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "Min".to_string(),
            args: window,
          })
          .unwrap_or(elems[i].clone());
          result.push(min_val);
        }
        return Some(Ok(Expr::List(result)));
      }
    }
    "Upsample" if args.len() == 2 => {
      // Upsample[list, n] — insert n-1 zeros between each element
      if let (Expr::List(elems), Some(n)) = (&args[0], expr_to_i128(&args[1])) {
        let n = n as usize;
        if n > 0 {
          let mut result = Vec::new();
          for elem in elems {
            result.push(elem.clone());
            for _ in 1..n {
              result.push(Expr::Integer(0));
            }
          }
          return Some(Ok(Expr::List(result)));
        }
      }
    }
    "Downsample" if args.len() == 2 => {
      // Downsample[list, n] — take every n-th element
      if let (Expr::List(elems), Some(n)) = (&args[0], expr_to_i128(&args[1])) {
        let n = n as usize;
        if n > 0 {
          let result: Vec<Expr> = elems.iter().step_by(n).cloned().collect();
          return Some(Ok(Expr::List(result)));
        }
      }
    }
    "EulerAngles" if args.len() == 1 => {
      // EulerAngles[matrix] — extract ZYZ Euler angles from 3x3 rotation matrix
      // For ZYZ convention: R = Rz(alpha) Ry(beta) Rz(gamma)
      // beta = ArcCos[R33], alpha = ArcTan[-R23, R13], gamma = ArcTan[R32, R31]
      if let Expr::List(rows) = &args[0]
        && rows.len() == 3
      {
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == 3 => Some(cols.clone()),
            _ => None,
          })
          .collect();
        if matrix.len() == 3 {
          // Use numeric approach: evaluate elements
          let get = |i: usize, j: usize| -> f64 {
            match &matrix[i][j] {
              Expr::Integer(v) => *v as f64,
              Expr::Real(v) => *v,
              _ => {
                if let Ok(Expr::Real(v)) =
                  evaluate_expr_to_expr(&Expr::FunctionCall {
                    name: "N".to_string(),
                    args: vec![matrix[i][j].clone()],
                  })
                {
                  v
                } else {
                  0.0
                }
              }
            }
          };
          let r33 = get(2, 2);
          let beta = r33.acos();
          let (alpha, gamma) = if beta.abs() < 1e-10 {
            // beta ≈ 0: gimbal lock, alpha + gamma = atan2(R21, R11)
            let ag = get(1, 0).atan2(get(0, 0));
            (ag, 0.0)
          } else if (beta - std::f64::consts::PI).abs() < 1e-10 {
            // beta ≈ pi: alpha - gamma = atan2(R21, -R11)
            let ag = get(1, 0).atan2(-get(0, 0));
            (ag, 0.0)
          } else {
            let alpha = (-get(1, 2)).atan2(get(0, 2));
            let gamma = get(2, 1).atan2(get(2, 0));
            (alpha, gamma)
          };
          // Convert back to exact if close to simple values
          let to_expr = |v: f64| -> Expr {
            if v.abs() < 1e-14 {
              Expr::Integer(0)
            } else {
              Expr::Real(v)
            }
          };
          return Some(Ok(Expr::List(vec![
            to_expr(alpha),
            to_expr(beta),
            to_expr(gamma),
          ])));
        }
      }
    }
    // KroneckerSymbol[a, n] — generalized Jacobi symbol
    "KroneckerSymbol" if args.len() == 2 => {
      if let (Some(a), Some(n)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
      {
        let result = kronecker_symbol(a, n);
        return Some(Ok(Expr::Integer(result)));
      }
    }
    // NormalizedSquaredEuclideanDistance[u, v]
    // = (1/2) * Total[(u-v)^2] / (Total[(u-Mean[u])^2] + Total[(v-Mean[v])^2])
    "NormalizedSquaredEuclideanDistance" if args.len() == 2 => {
      if let (Expr::List(u), Expr::List(v)) = (&args[0], &args[1]) {
        if u.len() != v.len() || u.is_empty() {
          return None;
        }
        // Build expression: Divide[Total[Power[Subtract[u,v], 2]], 2 * Plus[Total[...], Total[...]]]
        let diff_sq: Vec<Expr> = u
          .iter()
          .zip(v.iter())
          .map(|(ui, vi)| Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  ui.clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), vi.clone()],
                  },
                ],
              },
              Expr::Integer(2),
            ],
          })
          .collect();
        let numerator = Expr::FunctionCall {
          name: "Total".to_string(),
          args: vec![Expr::List(diff_sq)],
        };
        // Variance-like terms
        let mean_u = Expr::FunctionCall {
          name: "Mean".to_string(),
          args: vec![Expr::List(u.clone())],
        };
        let mean_v = Expr::FunctionCall {
          name: "Mean".to_string(),
          args: vec![Expr::List(v.clone())],
        };
        let var_u: Vec<Expr> = u
          .iter()
          .map(|ui| Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  ui.clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), mean_u.clone()],
                  },
                ],
              },
              Expr::Integer(2),
            ],
          })
          .collect();
        let var_v: Vec<Expr> = v
          .iter()
          .map(|vi| Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  vi.clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), mean_v.clone()],
                  },
                ],
              },
              Expr::Integer(2),
            ],
          })
          .collect();
        let denominator = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Total".to_string(),
              args: vec![Expr::List(var_u)],
            },
            Expr::FunctionCall {
              name: "Total".to_string(),
              args: vec![Expr::List(var_v)],
            },
          ],
        };
        let result = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)],
            },
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                numerator,
                Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![denominator, Expr::Integer(-1)],
                },
              ],
            },
          ],
        };
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // CorrelationDistance[u, v] — 1 - Correlation[u, v]
    "CorrelationDistance" if args.len() == 2 => {
      // Build 1 - Correlation[u, v] and evaluate
      let corr_expr = Expr::FunctionCall {
        name: "Correlation".to_string(),
        args: vec![args[0].clone(), args[1].clone()],
      };
      let result_expr = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), corr_expr],
          },
        ],
      };
      return Some(evaluate_expr_to_expr(&result_expr));
    }
    // PowerModList[a, b, m] — modular power/root list
    // For integer b: returns {a^b mod m}
    // For Rational[1, k] b: finds all x in {0,...,m-1} such that x^k ≡ a (mod m)
    "PowerModList" if args.len() == 3 => {
      if let (Some(a), Some(m)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[2]))
      {
        if m <= 0 {
          return None;
        }
        // Check if exponent is a Rational 1/k (modular root)
        let root_exp = match &args[1] {
          Expr::FunctionCall { name, args: rargs }
            if name == "Rational"
              && rargs.len() == 2
              && matches!(&rargs[0], Expr::Integer(1))
              && matches!(&rargs[1], Expr::Integer(k) if *k > 0) =>
          {
            if let Expr::Integer(k) = &rargs[1] {
              Some(*k)
            } else {
              None
            }
          }
          _ => None,
        };
        if let Some(k) = root_exp {
          // Find all x where x^k ≡ a (mod m)
          let a_mod = ((a % m) + m) % m;
          let mut result = Vec::new();
          for x in 0..m {
            let mut power = 1i128;
            let mut base = x % m;
            let mut exp = k;
            while exp > 0 {
              if exp % 2 == 1 {
                power = (power * base) % m;
              }
              base = (base * base) % m;
              exp /= 2;
            }
            if power == a_mod {
              result.push(Expr::Integer(x));
            }
          }
          return Some(Ok(Expr::List(result)));
        } else if let Some(n) = expr_to_i128(&args[1]) {
          // Integer exponent: return {a^n mod m}
          if n < 0 {
            return None;
          }
          let mut power = 1i128;
          let mut base = ((a % m) + m) % m;
          let mut exp = n;
          while exp > 0 {
            if exp % 2 == 1 {
              power = (power * base) % m;
            }
            base = (base * base) % m;
            exp /= 2;
          }
          return Some(Ok(Expr::List(vec![Expr::Integer(power)])));
        }
      }
    }
    // ShearingMatrix[theta, v, n] — shearing transformation matrix
    // ShearingMatrix[theta, {v1,...}, {n1,...}] = I + Tan[theta] * outer(v, n)
    "ShearingMatrix" if args.len() == 3 => {
      if let (Expr::List(v), Expr::List(n_vec)) = (&args[1], &args[2]) {
        let dim = v.len();
        if dim != n_vec.len() {
          return None;
        }
        // Build identity + Tan[theta] * outer(v, n)
        let s = &Expr::FunctionCall {
          name: "Tan".to_string(),
          args: vec![args[0].clone()],
        };
        let mut rows = Vec::with_capacity(dim);
        for i in 0..dim {
          let mut row = Vec::with_capacity(dim);
          for j in 0..dim {
            let identity_val = if i == j {
              Expr::Integer(1)
            } else {
              Expr::Integer(0)
            };
            // s * v[i] * n[j]
            let shear = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![s.clone(), v[i].clone(), n_vec[j].clone()],
            };
            let entry = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![identity_val, shear],
            };
            row.push(evaluate_expr_to_expr(&entry).unwrap_or(entry));
          }
          rows.push(Expr::List(row));
        }
        return Some(Ok(Expr::List(rows)));
      }
    }
    // PrimitiveRootList[n] — list of primitive roots modulo n
    "PrimitiveRootList" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0]) {
        if n <= 1 {
          return Some(Ok(Expr::List(vec![])));
        }
        let n = n as u64;
        let phi = euler_totient(n);
        let mut roots = Vec::new();
        for g in 1..n {
          if is_primitive_root(g, n, phi) {
            roots.push(Expr::Integer(g as i128));
          }
        }
        return Some(Ok(Expr::List(roots)));
      }
    }
    // DMSList[degrees] — convert decimal degrees to {d, m, s}
    "DMSList" if args.len() == 1 => {
      let val =
        evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
      if let Some((num, den)) =
        crate::functions::math_ast::expr_to_rational(&val)
      {
        let d = num / den;
        let remainder = num - d * den;
        let min_num = remainder * 60;
        let m = min_num / den;
        let min_rem = min_num - m * den;
        let sec_num = min_rem * 60;
        let s = sec_num / den;
        let sec_rem = sec_num - s * den;
        if sec_rem == 0 {
          return Some(Ok(Expr::List(vec![
            Expr::Integer(d),
            Expr::Integer(m),
            Expr::Integer(s),
          ])));
        } else {
          let g = gcd_i128(sec_num.abs(), den.abs());
          return Some(Ok(Expr::List(vec![
            Expr::Integer(d),
            Expr::Integer(m),
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(sec_num / g), Expr::Integer(den / g)],
            },
          ])));
        }
      }
    }
    // AlternatingFactorial[n] = n! - (n-1)! + (n-2)! - ... + (-1)^n * 0!
    // Actually: af(0)=0, af(n) = n! - af(n-1) for n >= 1
    "AlternatingFactorial" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0]) {
        if n < 0 {
          return None;
        }
        let n = n as u64;
        // Compute iteratively: af(0) = 0, af(k) = k! - af(k-1)
        let mut factorials: Vec<num_bigint::BigInt> =
          vec![num_bigint::BigInt::from(1u64)];
        for k in 1..=n {
          factorials
            .push(factorials.last().unwrap() * num_bigint::BigInt::from(k));
        }
        let mut af = num_bigint::BigInt::from(0);
        for k in 1..=n {
          af = &factorials[k as usize] - &af;
        }
        let result_str = af.to_string();
        if let Ok(v) = result_str.parse::<i128>() {
          return Some(Ok(Expr::Integer(v)));
        }
        // For very large values, return as string
        return Some(Ok(Expr::Integer(result_str.parse().unwrap_or(0))));
      }
    }
    // AlphabeticOrder[s1, s2] — 1 if s1 < s2, -1 if s1 > s2, 0 if equal
    "AlphabeticOrder" if args.len() == 2 => {
      if let (Expr::String(s1), Expr::String(s2)) = (&args[0], &args[1]) {
        let s1_lower = s1.to_lowercase();
        let s2_lower = s2.to_lowercase();
        let result = match s1_lower.cmp(&s2_lower) {
          std::cmp::Ordering::Less => 1,
          std::cmp::Ordering::Greater => -1,
          std::cmp::Ordering::Equal => match s1.cmp(s2) {
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Greater => -1,
            std::cmp::Ordering::Equal => 0,
          },
        };
        return Some(Ok(Expr::Integer(result)));
      }
    }
    // BinaryDistance[u, v] — 0 if u == v, 1 otherwise
    "BinaryDistance" if args.len() == 2 => {
      let s1 = crate::syntax::expr_to_string(&args[0]);
      let s2 = crate::syntax::expr_to_string(&args[1]);
      return Some(Ok(Expr::Integer(if s1 == s2 { 0 } else { 1 })));
    }
    // SquaresR[k, n] — number of representations of n as sum of k squares
    "SquaresR" if args.len() == 2 => {
      if let (Some(k), Some(n)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
      {
        if n == 0 {
          return Some(Ok(Expr::Integer(1)));
        }
        if n < 0 {
          return Some(Ok(Expr::Integer(0)));
        }
        let n = n as i64;
        let k = k as usize;
        // Brute-force count: enumerate all integer tuples with sum of squares = n
        // Each component ranges from -isqrt(n) to isqrt(n)
        let max_val = (n as f64).sqrt() as i64;
        let count = count_squares_r(k, n, max_val, 0);
        return Some(Ok(Expr::Integer(count as i128)));
      }
    }
    // AddSides[rel, expr] — add expr to both sides of a relation
    "AddSides" if args.len() == 2 => {
      if let Some(result) = apply_to_sides(&args[0], &args[1], "Plus") {
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // SubtractSides[rel, expr] — subtract expr from both sides
    "SubtractSides" if args.len() == 2 => {
      let neg = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), args[1].clone()],
      };
      if let Some(result) = apply_to_sides(&args[0], &neg, "Plus") {
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // MultiplySides[rel, expr] — multiply both sides by expr
    "MultiplySides" if args.len() == 2 => {
      if let Some(result) = apply_to_sides(&args[0], &args[1], "Times") {
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // DivideSides[rel, expr] — divide both sides by expr
    "DivideSides" if args.len() == 2 => {
      let inv = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![args[1].clone(), Expr::Integer(-1)],
      };
      if let Some(result) = apply_to_sides(&args[0], &inv, "Times") {
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // ApplySides[f, rel] — apply function f to both sides of a relation
    "ApplySides" if args.len() == 2 => {
      if let Expr::Comparison {
        operands,
        operators,
      } = &args[1]
      {
        let new_operands: Vec<Expr> = operands
          .iter()
          .map(|op| Expr::FunctionCall {
            name: crate::syntax::expr_to_string(&args[0]),
            args: vec![op.clone()],
          })
          .collect();
        let result = Expr::Comparison {
          operands: new_operands,
          operators: operators.clone(),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // FromDMS[{d, m, s}] — convert degrees/minutes/seconds to decimal degrees
    "FromDMS" if args.len() == 1 => {
      match &args[0] {
        Expr::List(parts) => {
          // {d}, {d, m}, or {d, m, s}
          let d = parts.first().cloned().unwrap_or(Expr::Integer(0));
          let m = parts.get(1).cloned().unwrap_or(Expr::Integer(0));
          let s = parts.get(2).cloned().unwrap_or(Expr::Integer(0));
          // result = d + m/60 + s/3600
          let result = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              d,
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  m,
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(1), Expr::Integer(60)],
                  },
                ],
              },
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  s,
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(1), Expr::Integer(3600)],
                  },
                ],
              },
            ],
          };
          return Some(evaluate_expr_to_expr(&result));
        }
        // FromDMS[n] where n is just degrees
        Expr::Integer(_) | Expr::Real(_) => {
          return Some(Ok(args[0].clone()));
        }
        _ => {}
      }
    }
    // NArgMin[f, x] — numerical arg min
    "NArgMin" if args.len() == 2 => {
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(vec![Expr::Identifier(var.clone()), Expr::Integer(0)]),
        ];
        if let Ok(Expr::List(ref result)) =
          crate::functions::polynomial_ast::find_minimum_ast(&find_args, false)
          && result.len() == 2
          && let Expr::List(rules) = &result[1]
        {
          let args_list: Vec<Expr> = rules
            .iter()
            .filter_map(|r| {
              if let Expr::Rule { replacement, .. } = r {
                // Apply N[] to get numeric result
                let n_expr = Expr::FunctionCall {
                  name: "N".to_string(),
                  args: vec![replacement.as_ref().clone()],
                };
                Some(
                  evaluate_expr_to_expr(&n_expr)
                    .unwrap_or(replacement.as_ref().clone()),
                )
              } else {
                None
              }
            })
            .collect();
          if args_list.len() == 1 {
            return Some(Ok(args_list.into_iter().next().unwrap()));
          }
          return Some(Ok(Expr::List(args_list)));
        }
      }
    }
    // NArgMax[f, x] — numerical arg max
    "NArgMax" if args.len() == 2 => {
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(vec![Expr::Identifier(var.clone()), Expr::Integer(0)]),
        ];
        if let Ok(Expr::List(ref result)) =
          crate::functions::polynomial_ast::find_minimum_ast(&find_args, true)
          && result.len() == 2
          && let Expr::List(rules) = &result[1]
        {
          let args_list: Vec<Expr> = rules
            .iter()
            .filter_map(|r| {
              if let Expr::Rule { replacement, .. } = r {
                let n_expr = Expr::FunctionCall {
                  name: "N".to_string(),
                  args: vec![replacement.as_ref().clone()],
                };
                Some(
                  evaluate_expr_to_expr(&n_expr)
                    .unwrap_or(replacement.as_ref().clone()),
                )
              } else {
                None
              }
            })
            .collect();
          if args_list.len() == 1 {
            return Some(Ok(args_list.into_iter().next().unwrap()));
          }
          return Some(Ok(Expr::List(args_list)));
        }
      }
    }
    // ArrayResample[list, n] — resample a 1D array to n elements using linear interpolation
    "ArrayResample" if args.len() == 2 => {
      if let (Expr::List(elems), Some(n)) = (&args[0], expr_to_i128(&args[1])) {
        let n = n as usize;
        let m = elems.len();
        if n == 0 {
          return Some(Ok(Expr::List(vec![])));
        }
        if m == 0 {
          return Some(Ok(Expr::List(vec![])));
        }
        if n == 1 {
          return Some(Ok(Expr::List(vec![elems[0].clone()])));
        }
        if m == 1 {
          return Some(Ok(Expr::List(vec![elems[0].clone(); n])));
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
          // Map output index i to input coordinate
          // t = i * (m-1) / (n-1)
          let t_num = i as i128 * (m as i128 - 1);
          let t_den = n as i128 - 1;
          let idx = (t_num / t_den) as usize;
          let rem = t_num % t_den;
          if rem == 0 {
            // Exact index
            result.push(elems[idx].clone());
          } else {
            // Linear interpolation: elems[idx] + rem/t_den * (elems[idx+1] - elems[idx])
            let frac = Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(rem), Expr::Integer(t_den)],
            };
            let interp = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                elems[idx].clone(),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![
                    frac,
                    Expr::FunctionCall {
                      name: "Plus".to_string(),
                      args: vec![
                        elems[idx + 1].clone(),
                        Expr::FunctionCall {
                          name: "Times".to_string(),
                          args: vec![Expr::Integer(-1), elems[idx].clone()],
                        },
                      ],
                    },
                  ],
                },
              ],
            };
            result.push(evaluate_expr_to_expr(&interp).unwrap_or(interp));
          }
        }
        return Some(Ok(Expr::List(result)));
      }
    }
    // PolynomialLCM[p1, p2, ...] — least common multiple of polynomials
    "PolynomialLCM" if args.len() >= 2 => {
      // Build: Cancel[p1 * p2 / PolynomialGCD[p1, p2]]
      // For more than 2 args, fold pairwise
      let mut result = args[0].clone();
      for arg in &args[1..] {
        let gcd = Expr::FunctionCall {
          name: "PolynomialGCD".to_string(),
          args: vec![result.clone(), arg.clone()],
        };
        let product = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![result, arg.clone()],
        };
        let lcm = Expr::FunctionCall {
          name: "Cancel".to_string(),
          args: vec![Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              product,
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![gcd, Expr::Integer(-1)],
              },
            ],
          }],
        };
        result = evaluate_expr_to_expr(&lcm).unwrap_or(lcm);
      }
      return Some(Ok(result));
    }
    // CoordinateBoundsArray[{{xmin,xmax},{ymin,ymax},...}] — grid of coordinate tuples (step 1)
    // CoordinateBoundsArray[{{xmin,xmax},{ymin,ymax},...}, d] — grid with step d
    "CoordinateBoundsArray" if !args.is_empty() && args.len() <= 2 => {
      if let Expr::List(bounds) = &args[0] {
        // Parse bounds pairs as integer ranges
        let mut ranges: Vec<(i128, i128)> = Vec::new();
        let mut ok = true;
        for b in bounds {
          if let Expr::List(pair) = b
            && pair.len() == 2
            && let (Some(lo), Some(hi)) =
              (expr_to_i128(&pair[0]), expr_to_i128(&pair[1]))
          {
            ranges.push((lo, hi));
          } else {
            ok = false;
            break;
          }
        }
        if ok && !ranges.is_empty() {
          let step = if args.len() == 2 {
            expr_to_i128(&args[1]).unwrap_or(1)
          } else {
            1
          };
          if step > 0 {
            // Generate discrete values for each dimension
            let dim_values: Vec<Vec<i128>> = ranges
              .iter()
              .map(|&(lo, hi)| {
                let mut vals = Vec::new();
                let mut v = lo;
                while v <= hi {
                  vals.push(v);
                  v += step;
                }
                vals
              })
              .collect();

            // Build nested array: outer dimensions correspond to first dims
            fn build_grid(dim_values: &[Vec<i128>], prefix: &[i128]) -> Expr {
              if prefix.len() == dim_values.len() {
                // Create a coordinate tuple
                if prefix.len() == 1 {
                  Expr::List(vec![Expr::Integer(prefix[0])])
                } else {
                  Expr::List(prefix.iter().map(|&v| Expr::Integer(v)).collect())
                }
              } else {
                let dim_idx = prefix.len();
                let items: Vec<Expr> = dim_values[dim_idx]
                  .iter()
                  .map(|&v| {
                    let mut new_prefix = prefix.to_vec();
                    new_prefix.push(v);
                    build_grid(dim_values, &new_prefix)
                  })
                  .collect();
                Expr::List(items)
              }
            }

            let result = build_grid(&dim_values, &[]);
            return Some(Ok(result));
          }
        }
      }
    }
    "CantorStaircase" if args.len() == 1 => {
      return Some(cantor_staircase_ast(&args[0]));
    }
    "Midpoint" if args.len() == 1 => {
      return Some(midpoint_ast(&args[0]));
    }
    "QFactorial" if args.len() == 2 => {
      return Some(qfactorial_ast(&args[0], &args[1]));
    }
    _ => {}
  }
  None
}

/// Compute the Cantor staircase (devil's staircase) function.
/// For rational p/q in [0,1], computes the exact rational value.
/// Algorithm: Express x in base 3. If a digit 1 appears, replace it
/// and all subsequent digits with a single "1" in base 2.
/// Replace all 2s with 1s. Read the result in base 2.
fn cantor_staircase_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  use crate::functions::math_ast::{expr_to_i128, try_eval_to_f64};

  // Handle exact integers
  if let Some(n) = expr_to_i128(arg) {
    if n <= 0 {
      return Ok(Expr::Integer(0));
    } else {
      return Ok(Expr::Integer(1));
    }
  }

  // Handle rationals
  if let Some((num, den)) = extract_rational(arg) {
    if num <= 0 {
      return Ok(Expr::Integer(0));
    }
    if num >= den {
      return Ok(Expr::Integer(1));
    }
    // Compute cantor staircase for rational num/den in (0,1)
    return Ok(cantor_staircase_rational(num, den));
  }

  // Handle numeric values (Real)
  if let Some(x) = try_eval_to_f64(arg) {
    if x <= 0.0 {
      return Ok(Expr::Integer(0));
    }
    if x >= 1.0 {
      return Ok(Expr::Integer(1));
    }
    return Ok(Expr::Real(cantor_staircase_f64(x)));
  }

  // Return unevaluated for symbolic arguments
  Ok(Expr::FunctionCall {
    name: "CantorStaircase".to_string(),
    args: vec![arg.clone()],
  })
}

/// Extract numerator and denominator from a rational expression
fn extract_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1]) {
        Some((*a, *b))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Compute cantor staircase for exact rational p/q where 0 < p/q < 1
fn cantor_staircase_rational(p: i128, q: i128) -> Expr {
  use crate::functions::math_ast::make_rational;
  use std::collections::HashMap;

  // Use the ternary digit algorithm with cycle detection.
  // Generate ternary digits of p/q. For each digit:
  // - 0: binary digit 0
  // - 2: binary digit 1
  // - 1: output binary digit 1 and stop
  //
  // For repeating ternary expansions, we detect the cycle in remainders
  // and compute the repeating binary fraction as a geometric series.

  let mut result_num: i128 = 0;
  let mut result_den: i128 = 1;
  let mut current_p = p;
  let current_q = q;

  // Track remainders to detect cycles; map remainder -> (step_index, result_num, result_den)
  let mut seen: HashMap<i128, (usize, i128, i128)> = HashMap::new();

  for step in 0..256 {
    if let Some(&(cycle_start, cycle_num, cycle_den)) = seen.get(&current_p) {
      // We've entered a cycle. The repeating binary block is:
      // (result_num / result_den) - (cycle_num / cycle_den)
      // = (result_num * cycle_den - cycle_num * result_den) / (result_den * cycle_den)
      // This repeating block, as a geometric series with ratio 1/2^cycle_len:
      // total = prefix + repeating_block / (1 - 1/2^cycle_len)
      let cycle_len = step - cycle_start;
      // repeat block value = result_num/result_den - cycle_num/cycle_den
      let repeat_num = result_num * cycle_den - cycle_num * result_den;
      let repeat_den = result_den * cycle_den;
      // Total = prefix + block * pow2/(pow2-1)
      // = cycle_num/cycle_den + repeat_num/(repeat_den) * pow2/(pow2-1)
      let pow2 = 1i128 << cycle_len;
      let final_num =
        cycle_num * repeat_den * (pow2 - 1) + repeat_num * pow2 * cycle_den;
      let final_den = cycle_den * repeat_den * (pow2 - 1);
      return make_rational(final_num, final_den);
    }

    seen.insert(current_p, (step, result_num, result_den));

    // Multiply by 3 to get next ternary digit
    current_p *= 3;
    let digit = current_p / current_q;
    current_p %= current_q;

    result_den *= 2;

    match digit {
      0 => {
        result_num *= 2;
      }
      1 => {
        result_num = result_num * 2 + 1;
        return make_rational(result_num, result_den);
      }
      2 => {
        result_num = result_num * 2 + 1;
      }
      _ => {
        break;
      }
    }

    if current_p == 0 {
      return make_rational(result_num, result_den);
    }
  }

  // Fallback: use float
  Expr::Real(cantor_staircase_f64(p as f64 / q as f64))
}

/// Compute cantor staircase numerically for x in (0,1)
fn cantor_staircase_f64(x: f64) -> f64 {
  let mut result = 0.0;
  let mut power = 0.5;
  let mut current = x;

  for _ in 0..64 {
    current *= 3.0;
    if current >= 2.0 {
      result += power;
      current -= 2.0;
    } else if current >= 1.0 {
      result += power;
      return result;
    }
    power *= 0.5;
  }

  result
}

/// Midpoint[{p1, p2}] or Midpoint[Line[{p1, p2}]] - midpoint of two points
fn midpoint_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  // Extract the two points from {p1, p2} or Line[{p1, p2}]
  let points = match arg {
    Expr::List(items)
      if items.len() == 2
        && matches!(&items[0], Expr::List(_))
        && matches!(&items[1], Expr::List(_)) =>
    {
      items
    }
    Expr::FunctionCall { name, args } if name == "Line" && args.len() == 1 => {
      if let Expr::List(items) = &args[0] {
        if items.len() == 2 {
          items
        } else {
          return Ok(Expr::FunctionCall {
            name: "Midpoint".to_string(),
            args: vec![arg.clone()],
          });
        }
      } else {
        return Ok(Expr::FunctionCall {
          name: "Midpoint".to_string(),
          args: vec![arg.clone()],
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Midpoint".to_string(),
        args: vec![arg.clone()],
      });
    }
  };

  let p1 = &points[0];
  let p2 = &points[1];

  // (p1 + p2) / 2 - works for both scalar and vector points
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![p1.clone(), p2.clone()],
  };
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(sum),
    right: Box::new(Expr::Integer(2)),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// QFactorial[n, q] - q-analog of the factorial
/// [n]_q! = [1]_q * [2]_q * ... * [n]_q where [k]_q = (1 - q^k) / (1 - q)
fn qfactorial_ast(
  n_expr: &Expr,
  q_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let n = match crate::functions::math_ast::expr_to_i128(n_expr) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "QFactorial".to_string(),
        args: vec![n_expr.clone(), q_expr.clone()],
      });
    }
  };

  if n == 0 || n == 1 {
    return Ok(Expr::Integer(1));
  }

  // Compute product of [k]_q for k = 1 to n
  let mut factors = Vec::new();
  for k in 1..=n {
    // [k]_q = (1 - q^k) / (1 - q)
    let q_k = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(q_expr.clone()),
      right: Box::new(Expr::Integer(k as i128)),
    };
    let numerator = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(q_k),
    };
    let denominator = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(q_expr.clone()),
    };
    let factor = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(denominator),
    };
    factors.push(factor);
  }

  let product = Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors,
  };

  crate::evaluator::evaluate_expr_to_expr(&product)
}

/// Find minimum linear recurrence coefficients {c1, c2, ..., cd} such that
/// a[n] = c1*a[n-1] + c2*a[n-2] + ... + cd*a[n-d] for all valid n.
fn find_linear_recurrence_impl(seq: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::functions::math_ast::expr_to_rational;

  // Convert sequence to rationals
  let mut rats: Vec<(i128, i128)> = Vec::new();
  for e in seq {
    let ev =
      crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
    match expr_to_rational(&ev) {
      Some(r) => rats.push(r),
      None => {
        return Ok(Expr::FunctionCall {
          name: "FindLinearRecurrence".to_string(),
          args: vec![Expr::List(seq.to_vec())],
        });
      }
    }
  }

  let n = rats.len();
  // Try recurrence orders from 1 up to n/2
  for d in 1..=n / 2 {
    // Check if recurrence of order d works for all positions d..n-1
    // Build system: for each i in d..n: a[i] = c1*a[i-1] + c2*a[i-2] + ... + cd*a[i-d]
    // We need at least d equations to solve for d unknowns.
    if n < 2 * d {
      continue;
    }

    // Solve the system using the first d equations, then verify the rest
    // Use Cramer's rule / Gaussian elimination with rationals
    let mut matrix: Vec<Vec<(i128, i128)>> = Vec::new();
    let mut rhs: Vec<(i128, i128)> = Vec::new();

    for i in d..(2 * d).min(n) {
      let mut row = Vec::new();
      for j in 1..=d {
        row.push(rats[i - j]);
      }
      matrix.push(row);
      rhs.push(rats[i]);
    }

    // Gaussian elimination
    if let Some(coeffs) = solve_rational_system(&mut matrix, &mut rhs) {
      // Verify against remaining elements
      let mut valid = true;
      for i in (2 * d).min(n)..n {
        let mut sum = (0i128, 1i128);
        for (j, c) in coeffs.iter().enumerate() {
          let term = rat_mul(*c, rats[i - 1 - j]);
          sum = rat_add(sum, term);
        }
        if sum != rats[i] {
          valid = false;
          break;
        }
      }
      if valid {
        let result: Vec<Expr> = coeffs
          .iter()
          .map(|&(num, den)| rational_to_expr_local(num, den))
          .collect();
        return Ok(Expr::List(result));
      }
    }
  }

  // No recurrence found
  Ok(Expr::FunctionCall {
    name: "FindLinearRecurrence".to_string(),
    args: vec![Expr::List(seq.to_vec())],
  })
}

fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let n = a.0 * b.1 + b.0 * a.1;
  let d = a.1 * b.1;
  let g = gcd_i128(n.abs(), d.abs());
  (n / g, d / g)
}

fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let n = a.0 * b.0;
  let d = a.1 * b.1;
  let g = gcd_i128(n.abs(), d.abs());
  (n / g, d / g)
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  if a == 0 { 1 } else { a }
}

fn solve_rational_system(
  matrix: &mut [Vec<(i128, i128)>],
  rhs: &mut [(i128, i128)],
) -> Option<Vec<(i128, i128)>> {
  let n = matrix.len();
  if n == 0 {
    return Some(vec![]);
  }

  // Forward elimination
  for col in 0..n {
    // Find pivot
    let pivot_row = (col..n).find(|&r| matrix[r][col].0 != 0)?;
    if pivot_row != col {
      matrix.swap(col, pivot_row);
      rhs.swap(col, pivot_row);
    }

    let pivot = matrix[col][col];
    for row in (col + 1)..n {
      if matrix[row][col].0 == 0 {
        continue;
      }
      let factor = rat_div(matrix[row][col], pivot);
      for c in col..n {
        let sub = rat_mul(factor, matrix[col][c]);
        matrix[row][c] = rat_sub(matrix[row][c], sub);
      }
      let sub = rat_mul(factor, rhs[col]);
      rhs[row] = rat_sub(rhs[row], sub);
    }
  }

  // Back substitution
  let mut solution = vec![(0i128, 1i128); n];
  for i in (0..n).rev() {
    let mut val = rhs[i];
    for j in (i + 1)..n {
      let sub = rat_mul(matrix[i][j], solution[j]);
      val = rat_sub(val, sub);
    }
    if matrix[i][i].0 == 0 {
      return None;
    }
    solution[i] = rat_div(val, matrix[i][i]);
  }

  Some(solution)
}

fn rat_sub(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  rat_add(a, (-b.0, b.1))
}

fn rat_div(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  if b.0 < 0 {
    rat_mul(a, (-b.1, -b.0))
  } else {
    rat_mul(a, (b.1, b.0))
  }
}

fn rational_to_expr_local(n: i128, d: i128) -> Expr {
  let g = gcd_i128(n.abs(), d.abs());
  let (n, d) = (n / g, d / g);
  if d < 0 {
    rational_to_expr_local(-n, -d)
  } else if d == 1 {
    Expr::Integer(n)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(n), Expr::Integer(d)],
    }
  }
}

/// ExpToTrig[expr] — replace Exp[z] with trig/hyperbolic forms.
/// Exp[I*x] → Cos[x] + I*Sin[x], Exp[x] → Cosh[x] + Sinh[x].
fn exp_to_trig_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  let transformed = exp_to_trig_recursive(expr);
  crate::evaluator::evaluate_expr_to_expr(&transformed)
}

fn exp_to_trig_recursive(expr: &Expr) -> Expr {
  match expr {
    // E^z where E is the constant
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Constant(c) if c == "E")
      || matches!(left.as_ref(), Expr::Identifier(c) if c == "E") =>
    {
      let z = exp_to_trig_recursive(right);
      exp_to_trig_expand(&z)
    }
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && (matches!(&args[0], Expr::Constant(c) if c == "E")
          || matches!(&args[0], Expr::Identifier(c) if c == "E")) =>
    {
      let z = exp_to_trig_recursive(&args[1]);
      exp_to_trig_expand(&z)
    }
    // Recurse into function calls
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(exp_to_trig_recursive).collect(),
    },
    // Recurse into binary ops
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(exp_to_trig_recursive(left)),
      right: Box::new(exp_to_trig_recursive(right)),
    },
    // Recurse into unary ops
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(exp_to_trig_recursive(operand)),
    },
    // Recurse into lists
    Expr::List(items) => {
      Expr::List(items.iter().map(exp_to_trig_recursive).collect())
    }
    _ => expr.clone(),
  }
}

/// Given exponent z, return Cos[x] + I*Sin[x] if z = I*x,
/// otherwise Cosh[z] + Sinh[z].
fn exp_to_trig_expand(z: &Expr) -> Expr {
  // Check if z = I*x (purely imaginary)
  if let Some(x) = extract_imaginary_part(z) {
    // Cos[x] + I*Sin[x]
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Cos".to_string(),
          args: vec![x.clone()],
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Complex".to_string(),
              args: vec![Expr::Integer(0), Expr::Integer(1)],
            },
            Expr::FunctionCall {
              name: "Sin".to_string(),
              args: vec![x.clone()],
            },
          ],
        },
      ],
    }
  } else {
    // Cosh[z] + Sinh[z]
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Cosh".to_string(),
          args: vec![z.clone()],
        },
        Expr::FunctionCall {
          name: "Sinh".to_string(),
          args: vec![z.clone()],
        },
      ],
    }
  }
}

/// Check if an expression is the imaginary unit I
fn is_imaginary_unit(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(s) if s == "I")
    || matches!(expr, Expr::Constant(s) if s == "I")
    || matches!(expr, Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2
      && matches!(&args[0], Expr::Integer(0))
      && matches!(&args[1], Expr::Integer(1)))
}

/// Extract x from I*x, Times[I, x], Times[n, I], etc.
/// Returns Some(x) if z is purely imaginary, None otherwise.
fn extract_imaginary_part(z: &Expr) -> Option<Expr> {
  match z {
    // z = I itself (Complex[0, 1])
    _ if is_imaginary_unit(z) => Some(Expr::Integer(1)),
    // z = Times[I, x] or Times[x, I] or Times[n, I, x]
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Find the I factor
      let mut has_i = false;
      let mut other_factors = Vec::new();
      for arg in args {
        if !has_i && is_imaginary_unit(arg) {
          has_i = true;
        } else {
          other_factors.push(arg.clone());
        }
      }
      if has_i {
        if other_factors.len() == 1 {
          Some(other_factors.into_iter().next().unwrap())
        } else {
          Some(Expr::FunctionCall {
            name: "Times".to_string(),
            args: other_factors,
          })
        }
      } else {
        None
      }
    }
    // z = BinaryOp Times with I
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if is_imaginary_unit(left) {
        Some(*right.clone())
      } else if is_imaginary_unit(right) {
        Some(*left.clone())
      } else {
        None
      }
    }
    _ => None,
  }
}

/// TrigToExp[expr] — replace trig/hyperbolic functions with exponentials.
fn trig_to_exp_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  let transformed = trig_to_exp_recursive(expr);
  crate::evaluator::evaluate_expr_to_expr(&transformed)
}

fn trig_to_exp_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      let arg = trig_to_exp_recursive(&args[0]);
      let i = Expr::Identifier("I".to_string());
      let e = Expr::Constant("E".to_string());
      let half = Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)],
      };
      match name.as_str() {
        // Cos[x] = E^(I*x)/2 + E^(-I*x)/2
        "Cos" => {
          let ix = times(&[i.clone(), arg.clone()]);
          let e_ix = power(e.clone(), ix.clone());
          let e_nix = power(e.clone(), times(&[Expr::Integer(-1), ix]));
          plus(&[times(&[half.clone(), e_ix]), times(&[half, e_nix])])
        }
        // Sin[x] = -I*E^(I*x)/2 + I*E^(-I*x)/2
        "Sin" => {
          let ix = times(&[i.clone(), arg.clone()]);
          let e_ix = power(e.clone(), ix.clone());
          let e_nix = power(e.clone(), times(&[Expr::Integer(-1), ix]));
          plus(&[
            times(&[Expr::Integer(-1), i.clone(), half.clone(), e_ix]),
            times(&[i, half, e_nix]),
          ])
        }
        // Cosh[x] = E^x/2 + E^(-x)/2
        "Cosh" => {
          let e_x = power(e.clone(), arg.clone());
          let e_nx = power(e.clone(), times(&[Expr::Integer(-1), arg]));
          plus(&[times(&[half.clone(), e_x]), times(&[half, e_nx])])
        }
        // Sinh[x] = E^x/2 - E^(-x)/2
        "Sinh" => {
          let e_x = power(e.clone(), arg.clone());
          let e_nx = power(e.clone(), times(&[Expr::Integer(-1), arg]));
          plus(&[
            times(&[half.clone(), e_x]),
            times(&[Expr::Integer(-1), half, e_nx]),
          ])
        }
        // Tan[x] = -I*(E^(I*x) - E^(-I*x))/(E^(I*x) + E^(-I*x))
        "Tan" => {
          let ix = times(&[i.clone(), arg.clone()]);
          let e_ix = power(e.clone(), ix.clone());
          let e_nix = power(e.clone(), times(&[Expr::Integer(-1), ix]));
          times(&[
            Expr::Integer(-1),
            i,
            plus(&[e_ix.clone(), times(&[Expr::Integer(-1), e_nix.clone()])]),
            power(plus(&[e_ix, e_nix]), Expr::Integer(-1)),
          ])
        }
        // Other functions: recurse into args
        _ => Expr::FunctionCall {
          name: name.clone(),
          args: args.iter().map(trig_to_exp_recursive).collect(),
        },
      }
    }
    // Recurse into function calls with != 1 arg
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(trig_to_exp_recursive).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(trig_to_exp_recursive(left)),
      right: Box::new(trig_to_exp_recursive(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(trig_to_exp_recursive(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(trig_to_exp_recursive).collect())
    }
    _ => expr.clone(),
  }
}

fn times(factors: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.to_vec(),
  }
}

fn plus(terms: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.to_vec(),
  }
}

fn power(base: Expr, exp: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp],
  }
}

/// Euler's totient function
fn euler_totient(mut n: u64) -> u64 {
  let mut result = n;
  let mut p = 2u64;
  while p * p <= n {
    if n.is_multiple_of(p) {
      while n.is_multiple_of(p) {
        n /= p;
      }
      result -= result / p;
    }
    p += 1;
  }
  if n > 1 {
    result -= result / n;
  }
  result
}

/// Check if g is a primitive root modulo n
fn is_primitive_root(g: u64, n: u64, phi: u64) -> bool {
  if gcd_u64(g, n) != 1 {
    return false;
  }
  // g is a primitive root iff g^(phi/p) != 1 (mod n) for all prime factors p of phi
  let mut temp_phi = phi;
  let mut p = 2u64;
  while p * p <= temp_phi {
    if temp_phi.is_multiple_of(p) {
      if pow_mod(g, phi / p, n) == 1 {
        return false;
      }
      while temp_phi.is_multiple_of(p) {
        temp_phi /= p;
      }
    }
    p += 1;
  }
  if temp_phi > 1 && pow_mod(g, phi / temp_phi, n) == 1 {
    return false;
  }
  true
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
  let mut result = 1u64;
  base %= modulus;
  while exp > 0 {
    if exp % 2 == 1 {
      result = (result as u128 * base as u128 % modulus as u128) as u64;
    }
    exp >>= 1;
    base = (base as u128 * base as u128 % modulus as u128) as u64;
  }
  result
}

/// Kronecker symbol — generalization of Jacobi symbol to all integers
fn kronecker_symbol(a: i128, n: i128) -> i128 {
  if n == 0 {
    return if a == 1 || a == -1 { 1 } else { 0 };
  }
  if n == 1 {
    return 1;
  }
  if n == -1 {
    return if a < 0 { -1 } else { 1 };
  }

  // Handle n == 2
  if n == 2 {
    if a % 2 == 0 {
      return 0;
    }
    let a_mod_8 = a.rem_euclid(8);
    return if a_mod_8 == 1 || a_mod_8 == 7 { 1 } else { -1 };
  }
  if n == -2 {
    return kronecker_symbol(a, -1) * kronecker_symbol(a, 2);
  }

  // For negative n, factor out the sign
  if n < 0 {
    return kronecker_symbol(a, -1) * kronecker_symbol(a, -n);
  }

  // n > 2 and positive. Factor out powers of 2 from n.
  let mut n_rem = n;
  let mut result: i128 = 1;

  // Extract factor of 2
  let mut twos = 0;
  while n_rem % 2 == 0 {
    n_rem /= 2;
    twos += 1;
  }
  if twos > 0 {
    let k2 = kronecker_symbol(a, 2);
    for _ in 0..twos {
      result *= k2;
    }
  }

  // Now n_rem is odd and positive, use Jacobi symbol
  if n_rem > 1 {
    result *= crate::functions::jacobi_symbol(a, n_rem);
  }

  result
}

/// Count representations of n as sum of k squares (brute force recursion)
fn count_squares_r(
  k: usize,
  remaining: i64,
  max_val: i64,
  depth: usize,
) -> i64 {
  if depth == k {
    return if remaining == 0 { 1 } else { 0 };
  }
  let mut count = 0i64;
  let limit = (remaining as f64).sqrt() as i64;
  let limit = limit.min(max_val);
  for v in -limit..=limit {
    let sq = v * v;
    if sq > remaining {
      continue;
    }
    count += count_squares_r(k, remaining - sq, max_val, depth + 1);
  }
  count
}

/// Apply an operation to both sides of a comparison/equation.
/// e.g. apply_to_sides(x == 2, 3, "Plus") => x + 3 == 2 + 3
fn apply_to_sides(relation: &Expr, value: &Expr, op: &str) -> Option<Expr> {
  if let Expr::Comparison {
    operands,
    operators,
  } = relation
  {
    let new_operands: Vec<Expr> = operands
      .iter()
      .map(|operand| Expr::FunctionCall {
        name: op.to_string(),
        args: vec![operand.clone(), value.clone()],
      })
      .collect();
    Some(Expr::Comparison {
      operands: new_operands,
      operators: operators.clone(),
    })
  } else {
    None
  }
}

fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1]) {
        Some(*a as f64 / *b as f64)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Standardize[data] — subtract mean and divide by standard deviation
/// Standardize[data, f1, f2] — use f1 for location and f2 for scale
fn standardize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = &args[0];
  let items = match data {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Standardize".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(data.clone());
  }

  // Compute location (mean) and scale (standard deviation)
  let (location, scale): (Expr, Expr) = if args.len() >= 3 {
    let loc =
      crate::functions::list_helpers_ast::apply_func_ast(&args[1], data)?;
    let sc =
      crate::functions::list_helpers_ast::apply_func_ast(&args[2], data)?;
    (loc, sc)
  } else if args.len() == 2 {
    let loc =
      crate::functions::list_helpers_ast::apply_func_ast(&args[1], data)?;
    let sc =
      crate::functions::math_ast::standard_deviation_ast(&[data.clone()])?;
    (loc, sc)
  } else {
    let loc = crate::functions::math_ast::mean_ast(&[data.clone()])?;
    let sc =
      crate::functions::math_ast::standard_deviation_ast(&[data.clone()])?;
    (loc, sc)
  };

  // Compute (x - location) / scale for each element
  // Use evaluate_function_call_ast to call Subtract and Divide
  let mut result = Vec::new();
  for item in items {
    let diff = crate::evaluator::evaluate_function_call_ast(
      "Subtract",
      &[item.clone(), location.clone()],
    )?;
    let standardized = crate::evaluator::evaluate_function_call_ast(
      "Divide",
      &[diff, scale.clone()],
    )?;
    result.push(standardized);
  }
  Ok(Expr::List(result))
}
