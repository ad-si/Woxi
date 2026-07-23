#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::{
  expr_to_rational, gcd_i128, gcd_u64, make_sqrt,
};
use crate::syntax::{BinaryOperator, ComparisonOp, UnaryOperator, unevaluated};

/// Columnwise quartile-family statistic for a matrix argument.
///
/// `Quartiles`, `InterquartileRange`, `QuartileDeviation`, and
/// `QuartileSkewness` of a rectangular matrix (a list of equal-length lists)
/// are computed per column, matching wolframscript (e.g.
/// `InterquartileRange[{{1,10},{2,20},{3,30},{4,40}}]` → `{2, 20}`). Returns
/// `None` when the single argument is not such a matrix, leaving the
/// scalar/vector path to handle it.
/// True for an explicit numeric literal: integer, big integer, machine real, or
/// an exact `Rational[_, _]`. Symbolic constants (Pi, E, …) are not included.
fn is_numeric_literal(e: &Expr) -> bool {
  matches!(e, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
    || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
}

fn columnwise_quartile_stat(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let [Expr::List(items)] = args else {
    return None;
  };
  if items.is_empty() || !items.iter().all(|i| matches!(i, Expr::List(_))) {
    return None;
  }
  let rows: Vec<&[Expr]> = items
    .iter()
    .filter_map(|i| match i {
      Expr::List(r) => Some(r.as_ref()),
      _ => None,
    })
    .collect();
  let ncols = rows[0].len();
  if ncols == 0 || !rows.iter().all(|r| r.len() == ncols) {
    return None;
  }
  let mut out = Vec::with_capacity(ncols);
  for c in 0..ncols {
    let col: Vec<Expr> = rows.iter().map(|r| r[c].clone()).collect();
    let call = Expr::FunctionCall {
      name: name.to_string(),
      args: vec![Expr::List(col.into())].into(),
    };
    match crate::evaluator::evaluate_expr_to_expr(&call) {
      // A still-symbolic result means a column wasn't numeric; leave the whole
      // call unevaluated rather than emitting a half-symbolic list.
      Ok(v) if matches!(&v, Expr::FunctionCall { name: n, .. } if n == name) => {
        return Some(Ok(unevaluated(name, args)));
      }
      Ok(v) => out.push(v),
      Err(e) => return Some(Err(e)),
    }
  }
  Some(Ok(Expr::List(out.into())))
}

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
      return Some(crate::functions::math_ast::divide_head_ast(args));
    }
    "Power" => {
      // power_ast handles every arity: Power[] = 1, Power[x] = x, and
      // Power[a, b, c, ...] folds right-associatively (a^(b^(c^...))).
      return Some(crate::functions::math_ast::power_ast(args));
    }
    "Entropy" => {
      return Some(crate::functions::math_ast::entropy_ast(args));
    }
    "Max" => {
      return Some(crate::functions::math_ast::max_ast(args));
    }
    "Min" => {
      return Some(crate::functions::math_ast::min_ast(args));
    }
    "RankedMax" if args.len() == 2 => {
      // Lists rank their elements; associations rank their values.
      let items: Option<Vec<Expr>> = match &args[0] {
        Expr::List(items) => Some(items.to_vec()),
        Expr::Association(pairs) => {
          Some(pairs.iter().map(|(_, v)| v.clone()).collect())
        }
        _ => None,
      };
      if let Some(mut sorted) = items {
        // Ascending sort; positive k picks from the high end, negative k
        // from the low end. `RankedMax[list, -n]` = n-th smallest.
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = expr_to_i128(&args[1]) {
          let len = sorted.len() as i128;
          let idx = if k > 0 { len - k } else { -k - 1 };
          if (0..len).contains(&idx) {
            return Some(Ok(sorted[idx as usize].clone()));
          }
        }
      }
      return Some(Ok(unevaluated("RankedMax", args)));
    }
    "RankedMin" if args.len() == 2 => {
      // Lists rank their elements; associations rank their values.
      let items: Option<Vec<Expr>> = match &args[0] {
        Expr::List(items) => Some(items.to_vec()),
        Expr::Association(pairs) => {
          Some(pairs.iter().map(|(_, v)| v.clone()).collect())
        }
        _ => None,
      };
      if let Some(mut sorted) = items {
        // Ascending sort; positive k picks from the low end, negative k
        // from the high end. `RankedMin[list, -n]` = n-th largest.
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = expr_to_i128(&args[1]) {
          let len = sorted.len() as i128;
          let idx = if k > 0 { k - 1 } else { len + k };
          if (0..len).contains(&idx) {
            return Some(Ok(sorted[idx as usize].clone()));
          }
        }
      }
      return Some(Ok(unevaluated("RankedMin", args)));
    }
    "Quantile" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::quantile_ast(args));
    }
    "Quartiles" if args.len() == 1 => {
      // Quartiles[dist] for a distribution head — produce
      // {Quantile[dist, 1/4], Quantile[dist, 1/2], Quantile[dist, 3/4]}.
      if let Expr::FunctionCall { .. } = &args[0] {
        let qs = [(1i128, 4i128), (1, 2), (3, 4)];
        let mut results = Vec::with_capacity(3);
        let mut all_ok = true;
        for (qn, qd) in qs {
          let q_expr = crate::functions::math_ast::make_rational(qn, qd);
          let call = Expr::FunctionCall {
            name: "Quantile".to_string(),
            args: vec![args[0].clone(), q_expr].into(),
          };
          match crate::evaluator::evaluate_expr_to_expr(&call) {
            Ok(v) => {
              // Treat an unevaluated Quantile[...] result as failure so we
              // leave the whole call symbolic rather than emitting a
              // half-symbolic list.
              if matches!(&v, Expr::FunctionCall { name, .. } if name == "Quantile")
              {
                all_ok = false;
                break;
              }
              results.push(v);
            }
            Err(_) => {
              all_ok = false;
              break;
            }
          }
        }
        if all_ok && results.len() == 3 {
          return Some(Ok(Expr::List(results.into())));
        }
        return Some(Ok(unevaluated("Quartiles", args)));
      }
      // Quartiles uses Quantile with parameters {{1/2, 0}, {0, 1}}
      // Formula: pos = 1/2 + n*q, then linear interpolation
      // Matrix input: compute quartiles columnwise, i.e.
      // Map[Quartiles, Transpose[matrix]], matching wolframscript
      // (Quartiles[{{1,10},{2,20},{3,30},{4,40}}] →
      //  {{3/2, 5/2, 7/2}, {15, 25, 35}}).
      if let Some(r) = columnwise_quartile_stat("Quartiles", args) {
        return Some(r);
      }
      if let Expr::List(items) = &args[0] {
        if items.is_empty() {
          return Some(Ok(unevaluated("Quartiles", args)));
        }
        // Sort numerically
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = sorted.len() as i128;
        // Machine-real inputs must yield machine-real quartiles, not exact
        // rationals (e.g. Quartiles[{1., 2., 3., 4., 5.}] → {1.75, 3., 4.25}).
        let any_real = sorted
          .iter()
          .any(|e| matches!(e, Expr::Real(_) | Expr::BigFloat(_, _)));
        // Convert a 1-based position to a 0-based index, clamped to [1, n] so
        // positions outside the data fall back to the first/last element. This
        // also keeps single-element lists from underflowing (Quartiles[{5}] →
        // {5, 5, 5}).
        let clamp_idx = |i: i128| -> usize { (i.clamp(1, n) - 1) as usize };
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
            results.push(sorted[clamp_idx(j)].clone());
          } else {
            // Interpolate: (1 - frac)*sorted[j-1] + frac*sorted[j]
            // = ((pos_den - frac_num)*sorted[j-1] + frac_num*sorted[j]) / pos_den
            let lo = &sorted[clamp_idx(j)];
            let hi = &sorted[clamp_idx(j + 1)];
            let w_lo = pos_den - frac_num;
            let w_hi = frac_num;
            // Try integer arithmetic
            if let (Some(lo_v), Some(hi_v)) = (
              crate::functions::math_ast::try_eval_to_f64(lo),
              crate::functions::math_ast::try_eval_to_f64(hi),
            ) {
              let lo_i = lo_v as i128;
              let hi_i = hi_v as i128;
              if !any_real && lo_i as f64 == lo_v && hi_i as f64 == hi_v {
                // Exact rational: (w_lo*lo_i + w_hi*hi_i) / pos_den
                let num = w_lo * lo_i + w_hi * hi_i;
                results.push(crate::functions::math_ast::make_rational(
                  num, pos_den,
                ));
              } else {
                let v =
                  (w_lo as f64 * lo_v + w_hi as f64 * hi_v) / pos_den as f64;
                // Machine-real inputs stay real even when the interpolated
                // value is integral (e.g. 5.0 must print as `5.`, not `5`).
                if any_real {
                  results.push(Expr::Real(v));
                } else {
                  results.push(crate::functions::math_ast::num_to_expr(v));
                }
              }
            } else {
              // Symbolic fallback
              results.push(unevaluated("Quartiles", args));
              break;
            }
          }
        }
        if results.len() == 3 {
          return Some(Ok(Expr::List(results.into())));
        }
      }
    }
    "InterquartileRange" if args.len() == 1 => {
      // Distribution heads with a clean closed-form IQR. Without this
      // bypass the Q3 - Q1 path returns e.g.
      // Log[4]/λ - Log[4/3]/λ for ExponentialDistribution, because
      // Woxi doesn't fold Log[a] - Log[b] into Log[a/b] for symbolic
      // arguments.
      if let Expr::FunctionCall {
        name: dist_name,
        args: dargs,
      } = &args[0]
        && dist_name == "ExponentialDistribution"
        && dargs.len() == 1
      {
        let lambda = dargs[0].clone();
        // Only short-circuit when the rate is symbolic / exact; numeric
        // rates fall through to the Q3 - Q1 path which already returns
        // the right Real.
        if !matches!(&lambda, Expr::Real(_) | Expr::BigFloat(_, _)) {
          // Log[3]/λ
          let log3 = Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![Expr::Integer(3)].into(),
          };
          let iqr = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(log3),
            right: Box::new(lambda),
          };
          return Some(crate::evaluator::evaluate_expr_to_expr(&iqr));
        }
      }
      // Matrix input: one interquartile range per column.
      if let Some(r) = columnwise_quartile_stat("InterquartileRange", args) {
        return Some(r);
      }
      // InterquartileRange = Q3 - Q1
      // Call the Quartiles logic by dispatching
      if let Some(Ok(Expr::List(ref qs))) =
        dispatch_math_functions("Quartiles", args)
        && qs.len() == 3
      {
        let diff = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(qs[2].clone()),
          right: Box::new(qs[0].clone()),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&diff));
      }
    }
    // QuartileDeviation = (Q3 - Q1) / 2, i.e. half the interquartile range.
    "QuartileDeviation" if args.len() == 1 => {
      if let Some(r) = columnwise_quartile_stat("QuartileDeviation", args) {
        return Some(r);
      }
      if let Some(Ok(Expr::List(ref qs))) =
        dispatch_math_functions("Quartiles", args)
        && qs.len() == 3
      {
        let half_iqr = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(qs[2].clone()),
            right: Box::new(qs[0].clone()),
          }),
          right: Box::new(Expr::Integer(2)),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&half_iqr));
      }
    }
    // QuartileSkewness = (Q1 - 2 Q2 + Q3) / (Q3 - Q1), the Bowley skewness.
    // Reduces to Indeterminate (0/0) when all three quartiles coincide.
    "QuartileSkewness" if args.len() == 1 => {
      if let Some(r) = columnwise_quartile_stat("QuartileSkewness", args) {
        return Some(r);
      }
      if let Some(Ok(Expr::List(ref qs))) =
        dispatch_math_functions("Quartiles", args)
        && qs.len() == 3
      {
        use BinaryOperator::{Divide, Minus, Plus, Times};
        // numerator = Q1 - 2*Q2 + Q3
        let numerator = Expr::BinaryOp {
          op: Plus,
          left: Box::new(Expr::BinaryOp {
            op: Minus,
            left: Box::new(qs[0].clone()),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(qs[1].clone()),
            }),
          }),
          right: Box::new(qs[2].clone()),
        };
        let denominator = Expr::BinaryOp {
          op: Minus,
          left: Box::new(qs[2].clone()),
          right: Box::new(qs[0].clone()),
        };
        let skew = Expr::BinaryOp {
          op: Divide,
          left: Box::new(numerator),
          right: Box::new(denominator),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&skew));
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
        Expr::List(_) => {
          // Let Listable attribute handle threading
        }
        _ => {
          // Real-valued numeric x: |x| exactly. Negative values are negated
          // (RealAbs[Sqrt[2] - 3] -> 3 - Sqrt[2]); non-negative values are
          // returned unchanged. Floatifying here would lose exactness
          // (RealAbs[Pi] must stay Pi). Complex/symbolic x stays unevaluated.
          if let Some(v) = crate::functions::math_ast::try_eval_to_f64(&args[0])
          {
            if v < 0.0 {
              let neg = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), args[0].clone()].into(),
              };
              return Some(crate::evaluator::evaluate_expr_to_expr(&neg));
            }
            return Some(Ok(args[0].clone()));
          }
          return Some(Ok(unevaluated("RealAbs", args)));
        }
      }
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
        Expr::List(_) => {
          // Let Listable attribute handle threading
        }
        _ => {
          // Real-valued numeric x (Pi, Sqrt[2] - 3, …): decide the sign by its
          // numeric value. Complex or symbolic x stays unevaluated.
          if let Some(v) = crate::functions::math_ast::try_eval_to_f64(&args[0])
          {
            return Some(Ok(Expr::Integer(if v > 0.0 {
              1
            } else if v < 0.0 {
              -1
            } else {
              0
            })));
          }
          return Some(Ok(unevaluated("RealSign", args)));
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
    "Mod" if args.len() == 2 => {
      // Mod over two compatible quantities returns a quantity in the divisor's
      // unit; otherwise fall through to the ordinary numeric Mod.
      if let Some(result) =
        crate::functions::quantity_ast::try_quantity_mod(&args[0], &args[1])
      {
        return Some(result);
      }
      return Some(crate::functions::math_ast::mod_ast(args));
    }
    "Mod" if args.len() == 3 => {
      return Some(crate::functions::math_ast::mod_ast(args));
    }
    "Quotient" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::quotient_ast(args));
    }
    "QuotientRemainder" if args.len() == 2 => {
      // QuotientRemainder[n, 0] stays unevaluated (wolframscript), unlike the
      // 2-list {Quotient, Mod} we would otherwise assemble.
      if crate::functions::math_ast::is_literal_zero(&args[1]) {
        return Some(Ok(unevaluated("QuotientRemainder", args)));
      }
      let q = match crate::functions::math_ast::quotient_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let r = match crate::functions::math_ast::mod_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      return Some(Ok(Expr::List(vec![q, r].into())));
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
    "TukeyWindow" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::tukey_window_ast(args));
    }
    "BlackmanHarrisWindow"
    | "BlackmanNuttallWindow"
    | "NuttallWindow"
    | "FlatTopWindow"
    | "KaiserBesselWindow"
      if args.len() == 1 =>
    {
      return Some(crate::functions::math_ast::cosine_sum_window_ast(
        name, args,
      ));
    }
    "CauchyWindow" | "PoissonWindow" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::parametric_window_ast(
        name, args,
      ));
    }
    "BartlettHannWindow" if args.len() == 1 => {
      return Some(crate::functions::math_ast::parametric_window_ast(
        name, args,
      ));
    }
    "ParzenWindow" if args.len() == 1 => {
      return Some(crate::functions::math_ast::parzen_window_ast(args));
    }
    "BohmanWindow" if args.len() == 1 => {
      return Some(crate::functions::math_ast::bohman_window_ast(args));
    }
    "GaussianWindow" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::gaussian_window_ast(args));
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
    "FourierDST" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::fourier_dst_ast(args));
    }
    "FourierDCT" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::fourier_dct_ast(args));
    }
    "DiscreteHilbertTransform" if args.len() == 1 => {
      return Some(crate::functions::math_ast::discrete_hilbert_transform_ast(
        args,
      ));
    }
    "InverseFourier" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::inverse_fourier_ast(args));
    }
    "ListFourierSequenceTransform" if args.len() == 2 => {
      return Some(
        crate::functions::math_ast::list_fourier_sequence_transform_ast(args),
      );
    }
    "ListZTransform" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::list_z_transform_ast(args));
    }
    "DiscreteHadamardTransform" if args.len() == 1 => {
      return Some(
        crate::functions::math_ast::discrete_hadamard_transform_ast(args),
      );
    }
    // MultinormalDistribution statistics: the mean vector, covariance
    // matrix, and per-component variances read straight off the
    // constructor arguments
    "Mean" | "Covariance" | "Variance"
      if args.len() == 1
        && matches!(&args[0], Expr::FunctionCall { name: dn, args: da }
          if dn == "MultinormalDistribution" && da.len() == 2) =>
    {
      if let Expr::FunctionCall { args: da, .. } = &args[0]
        && let (Expr::List(mu), Expr::List(rows)) = (&da[0], &da[1])
      {
        match name {
          "Mean" => return Some(Ok(Expr::List(mu.clone()))),
          "Covariance" => return Some(Ok(da[1].clone())),
          _ => {
            let mut diag: Vec<Expr> = Vec::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
              match row {
                Expr::List(cols) if cols.len() == rows.len() => {
                  diag.push(cols[i].clone());
                }
                _ => return None,
              }
            }
            return Some(Ok(Expr::List(diag.into())));
          }
        }
      }
      return None;
    }
    // Empirical DataDistribution statistics (exact)
    "Mean" | "Variance"
      if args.len() == 1
        && matches!(&args[0], Expr::FunctionCall { name: dn, args: da }
          if (dn == "RiceDistribution" && da.len() == 2)
            || (dn == "MaxwellDistribution" && da.len() == 1)
            || (dn == "MoyalDistribution" && da.len() <= 2)) =>
    {
      if let Expr::FunctionCall { args: da, .. } = &args[0] {
        // Returned directly: the variance prints use Raw assemblies
        // that re-evaluation would re-canonicalize
        let dist_name = match &args[0] {
          Expr::FunctionCall { name: dn, .. } => dn.as_str(),
          _ => "",
        };
        let result = match dist_name {
          "RiceDistribution" => {
            crate::functions::math_ast::rice_mean_variance(&da[0], &da[1])
          }
          "MaxwellDistribution" => {
            crate::functions::math_ast::maxwell_mean_variance(&da[0])
          }
          _ => crate::functions::math_ast::moyal_mean_variance(da),
        };
        match result {
          Ok((mean, var)) => {
            return Some(Ok(if name == "Mean" { mean } else { var }));
          }
          Err(e) => return Some(Err(e)),
        }
      }
    }
    "Mean" | "Variance" | "StandardDeviation"
      if args.len() == 1
        && matches!(&args[0], Expr::FunctionCall { name: dn, args: da }
          if dn == "DataDistribution" && da.len() == 4) =>
    {
      if let Expr::FunctionCall { args: da, .. } = &args[0] {
        let mean = crate::functions::math_ast::data_distribution_moment(da, 1);
        match (name, mean) {
          ("Mean", Some(m)) => return Some(Ok(m)),
          ("Variance" | "StandardDeviation", Some(m)) => {
            if let Some(m2) =
              crate::functions::math_ast::data_distribution_moment(da, 2)
            {
              // Var = E[x^2] - mean^2
              let var =
                crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![
                    m2,
                    Expr::FunctionCall {
                      name: "Times".to_string(),
                      args: vec![
                        Expr::Integer(-1),
                        Expr::FunctionCall {
                          name: "Power".to_string(),
                          args: vec![m, Expr::Integer(2)].into(),
                        },
                      ]
                      .into(),
                    },
                  ]
                  .into(),
                });
              if let Ok(v) = var {
                if name == "Variance" {
                  return Some(Ok(v));
                }
                // StandardDeviation = Sqrt[Variance]
                let sd = crate::evaluator::evaluate_expr_to_expr(
                  &Expr::FunctionCall {
                    name: "Sqrt".to_string(),
                    args: vec![v].into(),
                  },
                );
                if let Ok(s) = sd {
                  return Some(Ok(s));
                }
              }
            }
          }
          _ => {}
        }
      }
      return None;
    }
    // ProductDistribution statistics: lists of the component values
    "Mean" | "Variance"
      if args.len() == 1
        && matches!(&args[0], Expr::FunctionCall { name: dn, args: da }
          if dn == "ProductDistribution" && !da.is_empty()) =>
    {
      if let Expr::FunctionCall { args: da, .. } = &args[0] {
        let mut out: Vec<Expr> = Vec::with_capacity(da.len());
        for d in da.iter() {
          let Expr::FunctionCall { name: cn, args: ca } = d else {
            return None;
          };
          let Ok((mean, var)) =
            crate::functions::math_ast::distribution_mean_variance(cn, ca)
          else {
            return None;
          };
          let component = if name == "Mean" { mean } else { var };
          match crate::evaluator::evaluate_expr_to_expr(&component) {
            Ok(v) => out.push(v),
            Err(_) => return None,
          }
        }
        return Some(Ok(Expr::List(out.into())));
      }
      return None;
    }
    "EmpiricalDistribution" if args.len() == 1 => {
      return Some(crate::functions::math_ast::empirical_distribution_ast(
        args,
      ));
    }
    "HistogramDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::histogram_distribution_ast(
        args,
      ));
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
    // Named sporadic groups are formal 0-argument heads that echo; the
    // Group* functions read their canonical data.
    "MathieuGroupM11" | "MathieuGroupM12" | "MathieuGroupM22"
    | "MathieuGroupM23" | "MathieuGroupM24"
      if args.is_empty() =>
    {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![].into(),
      }));
    }
    "GroupGenerators" if args.len() == 1 => {
      return Some(crate::functions::math_ast::group_generators_ast(args));
    }
    "GroupOrder" if args.len() == 1 => {
      return Some(crate::functions::math_ast::group_order_ast(args));
    }
    "GroupElements" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::group_elements_ast(args));
    }
    "GroupOrbits" if args.len() == 2 => {
      return Some(crate::functions::math_ast::group_orbits_ast(args));
    }
    "GroupElementQ" if args.len() == 2 => {
      return Some(crate::functions::math_ast::group_element_q_ast(args));
    }
    "GroupElementPosition" if args.len() == 2 => {
      return Some(crate::functions::math_ast::group_element_position_ast(
        args,
      ));
    }
    "GroupMultiplicationTable" if args.len() == 1 => {
      return Some(crate::functions::math_ast::group_multiplication_table_ast(
        args,
      ));
    }
    "GroupStabilizer" if args.len() == 2 => {
      return Some(crate::functions::math_ast::group_stabilizer_ast(args));
    }
    "CycleIndexPolynomial" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::cycle_index_polynomial_ast(
        args,
      ));
    }
    "EffectiveInterest" if args.len() == 2 => {
      return Some(crate::functions::math_ast::effective_interest_ast(args));
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
    "FactorialMoment" if args.len() == 2 => {
      return Some(crate::functions::math_ast::factorial_moment_ast(args));
    }
    "CharacteristicFunction" if args.len() == 2 => {
      return Some(crate::functions::math_ast::characteristic_function_ast(
        args,
      ));
    }
    "MomentGeneratingFunction" if args.len() == 2 => {
      return Some(crate::functions::math_ast::moment_generating_function_ast(
        args,
      ));
    }
    "FactorialMomentGeneratingFunction" if args.len() == 2 => {
      return Some(
        crate::functions::math_ast::factorial_moment_generating_function_ast(
          args,
        ),
      );
    }
    "CentralMomentGeneratingFunction" if args.len() == 2 => {
      return Some(
        crate::functions::math_ast::central_moment_generating_function_ast(
          args,
        ),
      );
    }
    "CumulantGeneratingFunction" if args.len() == 2 => {
      return Some(
        crate::functions::math_ast::cumulant_generating_function_ast(args),
      );
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
    "TrimmedMean" if (1..=2).contains(&args.len()) => {
      // TrimmedMean[list]              ≡ TrimmedMean[list, 0.05]
      // TrimmedMean[list, f]           — drop floor(f*n) smallest and largest
      // TrimmedMean[list, {f1, f2}]    — drop floor(f1*n) smallest, floor(f2*n) largest
      // A SparseArray data argument is handled via its dense form.
      if let Some(dense) =
        crate::functions::list_helpers_ast::densify_sparse_array(&args[0])
      {
        let mut new_args = args.to_vec();
        new_args[0] = dense;
        return Some(crate::evaluator::evaluate_function_call_ast(
          "TrimmedMean",
          &new_args,
        ));
      }
      if let Expr::List(elems) = &args[0] {
        let n = elems.len();
        // The trimming fraction must be a non-negative number < 0.5, or a list
        // of two non-negative numbers summing to < 1; otherwise emit arg2 and
        // stay unevaluated (rather than silently mis-trimming).
        let arg2_error = || {
          crate::emit_message(&format!(
            "TrimmedMean::arg2: The second argument {} is expected to be a non-negative number less than 0.5 or a list of two non-negative numbers that sum to less than 1.",
            crate::syntax::expr_to_string(&args[1])
          ));
          Some(Ok(unevaluated("TrimmedMean", args)))
        };
        let (trim_lo, trim_hi) = match args.get(1) {
          None => {
            let t = (n as f64 * 0.05).floor() as usize;
            (t, t)
          }
          Some(Expr::List(fs)) if fs.len() == 2 => {
            let Some(f1) = expr_to_f64(&fs[0]) else {
              return None;
            };
            let Some(f2) = expr_to_f64(&fs[1]) else {
              return None;
            };
            if f1 < 0.0 || f2 < 0.0 || f1 + f2 >= 1.0 {
              return arg2_error();
            }
            (
              (n as f64 * f1).floor() as usize,
              (n as f64 * f2).floor() as usize,
            )
          }
          Some(other) => {
            let Some(f) = expr_to_f64(other) else {
              return None;
            };
            if !(0.0..0.5).contains(&f) {
              return arg2_error();
            }
            let t = (n as f64 * f).floor() as usize;
            (t, t)
          }
        };
        if trim_lo + trim_hi < n {
          let mut sorted: Vec<Expr> = elems.to_vec();
          sorted.sort_by(|a, b| {
            let fa = expr_to_f64(a).unwrap_or(0.0);
            let fb = expr_to_f64(b).unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
          });
          let trimmed = &sorted[trim_lo..n - trim_hi];
          let sum_expr = unevaluated("Plus", trimmed);
          let result = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(sum_expr),
            right: Box::new(Expr::Integer(trimmed.len() as i128)),
          };
          return Some(evaluate_expr_to_expr(&result));
        }
      }
    }
    "WinsorizedMean" if args.len() == 2 => {
      // WinsorizedMean[list, f]         — replace the lowest/highest floor(f*n)
      //                                   values with the boundary value, mean.
      // WinsorizedMean[list, {f1, f2}]  — winsorize floor(f1*n) at the bottom,
      //                                   floor(f2*n) at the top.
      if let Expr::List(elems) = &args[0] {
        let n = elems.len();
        // The winsorizing fraction must be a non-negative number < 0.5, or a
        // list of two non-negative numbers summing to < 1; otherwise emit arg2
        // and stay unevaluated (rather than silently mis-winsorizing).
        let arg2_error = || {
          crate::emit_message(&format!(
            "WinsorizedMean::arg2: The second argument {} is expected to be a non-negative number less than 0.5 or a list of two non-negative numbers that sum to less than 1.",
            crate::syntax::expr_to_string(&args[1])
          ));
          Some(Ok(unevaluated("WinsorizedMean", args)))
        };
        let (trim_lo, trim_hi) = match &args[1] {
          Expr::List(fs) if fs.len() == 2 => {
            match (expr_to_f64(&fs[0]), expr_to_f64(&fs[1])) {
              (Some(f1), Some(f2)) => {
                if f1 < 0.0 || f2 < 0.0 || f1 + f2 >= 1.0 {
                  return arg2_error();
                }
                (
                  (n as f64 * f1).floor() as usize,
                  (n as f64 * f2).floor() as usize,
                )
              }
              _ => return None,
            }
          }
          other => match expr_to_f64(other) {
            Some(f) => {
              if !(0.0..0.5).contains(&f) {
                return arg2_error();
              }
              let t = (n as f64 * f).floor() as usize;
              (t, t)
            }
            None => return None,
          },
        };
        if trim_lo + trim_hi < n {
          let mut sorted: Vec<Expr> = elems.to_vec();
          sorted.sort_by(|a, b| {
            let fa = expr_to_f64(a).unwrap_or(0.0);
            let fb = expr_to_f64(b).unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
          });
          // Replace the bottom trim_lo with sorted[trim_lo] and the top
          // trim_hi with sorted[n-trim_hi-1].
          let low = sorted[trim_lo].clone();
          let high = sorted[n - trim_hi - 1].clone();
          let mut winsorized = sorted.clone();
          for item in winsorized.iter_mut().take(trim_lo) {
            *item = low.clone();
          }
          for item in winsorized.iter_mut().skip(n - trim_hi) {
            *item = high.clone();
          }
          let sum_expr = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: winsorized.into(),
          };
          let result = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(sum_expr),
            right: Box::new(Expr::Integer(n as i128)),
          };
          return Some(evaluate_expr_to_expr(&result));
        }
      }
    }
    "TrimmedVariance" if (1..=2).contains(&args.len()) => {
      // TrimmedVariance[list]            ≡ TrimmedVariance[list, 0.05]
      // TrimmedVariance[list, f]         — drop floor(f*n) from each end
      // TrimmedVariance[list, {f1, f2}]  — drop floor(f1*n) smallest,
      //                                    floor(f2*n) largest
      if let Expr::List(elems) = &args[0] {
        let n = elems.len();
        let arg2_error = || {
          crate::emit_message(&format!(
            "TrimmedVariance::arg2: The second argument {} is expected to be a non-negative number less than 0.5 or a list of two non-negative numbers that sum to less than 1.",
            crate::syntax::expr_to_string(&args[1])
          ));
          Some(Ok(unevaluated("TrimmedVariance", args)))
        };
        let (trim_lo, trim_hi) = match args.get(1) {
          None => {
            let t = (n as f64 * 0.05).floor() as usize;
            (t, t)
          }
          Some(Expr::List(fs)) if fs.len() == 2 => {
            let Some(f1) = expr_to_f64(&fs[0]) else {
              return None;
            };
            let Some(f2) = expr_to_f64(&fs[1]) else {
              return None;
            };
            if f1 < 0.0 || f2 < 0.0 || f1 + f2 >= 1.0 {
              return arg2_error();
            }
            (
              (n as f64 * f1).floor() as usize,
              (n as f64 * f2).floor() as usize,
            )
          }
          Some(other) => {
            let Some(f) = expr_to_f64(other) else {
              return None;
            };
            if !(0.0..0.5).contains(&f) {
              return arg2_error();
            }
            let t = (n as f64 * f).floor() as usize;
            (t, t)
          }
        };
        if trim_lo + trim_hi < n {
          let mut sorted: Vec<Expr> = elems.to_vec();
          sorted.sort_by(|a, b| {
            let fa = expr_to_f64(a).unwrap_or(0.0);
            let fb = expr_to_f64(b).unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
          });
          let trimmed: Vec<Expr> = sorted[trim_lo..n - trim_hi].to_vec();
          return Some(crate::functions::math_ast::variance_ast(&[
            Expr::List(trimmed.into()),
          ]));
        }
      }
    }
    "WinsorizedVariance" if args.len() == 2 => {
      // WinsorizedVariance[list, frac] — variance of winsorized data
      if let Expr::List(elems) = &args[0]
        && let Some(frac) = expr_to_f64(&args[1])
      {
        // A fraction outside [0, 0.5) is rejected with arg2.
        if !(0.0..0.5).contains(&frac) {
          crate::emit_message(&format!(
            "WinsorizedVariance::arg2: The second argument {} is expected to be a non-negative number less than 0.5 or a list of two non-negative numbers that sum to less than 1.",
            crate::syntax::expr_to_string(&args[1])
          ));
          return Some(Ok(unevaluated("WinsorizedVariance", args)));
        }
        let n = elems.len();
        let trim = (n as f64 * frac).floor() as usize;
        if 2 * trim < n {
          let mut sorted: Vec<Expr> = elems.to_vec();
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
            Expr::List(winsorized.into()),
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
    "ContraharmonicMean" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::contraharmonic_mean_ast(args));
    }
    "RootMeanSquare" if args.len() == 1 => {
      return Some(crate::functions::math_ast::root_mean_square_ast(args));
    }
    "Covariance" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::covariance_ast(args));
    }
    "CovarianceFunction" if args.len() == 2 => {
      // Sample autocovariance of a numeric time series at an integer lag.
      if let Some(result) =
        crate::functions::math_ast::covariance_function_data(&args[0], &args[1])
      {
        return Some(result);
      }
    }
    "CovarianceFunction" if args.len() == 3 => {
      return Some(crate::functions::math_ast::covariance_function_ast(args));
    }
    "AbsoluteCorrelation" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::absolute_correlation_ast(args));
    }
    "Correlation" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::correlation_ast(args));
    }
    "SpearmanRho" if args.len() == 2 => {
      return Some(crate::functions::math_ast::spearman_rho_ast(args));
    }
    // Any arg count: wrong counts stay silently unevaluated (matching
    // wolframscript, which emits no message for e.g. three arguments).
    "HoeffdingD" => {
      return Some(crate::functions::math_ast::hoeffding_d_ast(args));
    }
    "GoodmanKruskalGamma" => {
      return Some(crate::functions::math_ast::goodman_kruskal_gamma_ast(args));
    }
    "BlomqvistBeta" => {
      return Some(crate::functions::math_ast::blomqvist_beta_ast(args));
    }
    "CentralFeature" => {
      return Some(crate::functions::math_ast::central_feature_ast(args));
    }
    "ErlangB" if args.len() == 2 => {
      return Some(crate::functions::math_ast::erlang_b_ast(args));
    }
    "ErlangC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::erlang_c_ast(args));
    }
    "KendallTau" if args.len() == 2 => {
      return Some(crate::functions::math_ast::kendall_tau_ast(args));
    }
    "CentralMoment" if args.len() == 2 => {
      return Some(crate::functions::math_ast::central_moment_ast(args));
    }
    "Cumulant" if args.len() == 2 => {
      return Some(crate::functions::math_ast::cumulant_ast(args));
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
    "IntegerReverse" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::integer_reverse_ast(args));
    }
    "Rescale" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::rescale_ast(args));
    }
    "Normalize" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::normalize_ast(args));
    }
    "Norm" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::norm_ast(args));
    }
    "EuclideanDistance" if args.len() == 2 => {
      return Some(crate::functions::math_ast::euclidean_distance_ast(args));
    }
    "WarpingDistance" if args.len() == 2 => {
      return Some(crate::functions::math_ast::warping_distance_ast(args));
    }
    "WarpingCorrespondence" if args.len() == 2 => {
      return Some(crate::functions::math_ast::warping_correspondence_ast(
        args,
      ));
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
    "FareySequence" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::farey_sequence_ast(args));
    }
    "ZTest" if !args.is_empty() && args.len() <= 4 => {
      return Some(crate::functions::math_ast::ztest_ast(args));
    }
    "FisherRatioTest" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::fisher_ratio_test_ast(args));
    }
    "Fibonorial" if args.len() == 1 => {
      return Some(crate::functions::math_ast::fibonorial_ast(args));
    }
    "LogBarnesG" if args.len() == 1 => {
      return Some(crate::functions::math_ast::log_barnes_g_ast(args));
    }
    "MinkowskiQuestionMark" if args.len() == 1 => {
      return Some(crate::functions::math_ast::minkowski_question_mark_ast(
        args,
      ));
    }
    "DirichletCharacter" if args.len() == 3 => {
      return Some(crate::functions::dirichlet_ast::dirichlet_character_ast(
        args,
      ));
    }
    "DirichletL" if args.len() == 3 => {
      return Some(crate::functions::dirichlet_ast::dirichlet_l_ast(args));
    }
    "DirichletConvolve" if args.len() == 4 => {
      return Some(crate::functions::dirichlet_ast::dirichlet_convolve_ast(
        args,
      ));
    }
    "Pochhammer" if args.len() == 2 => {
      return Some(crate::functions::math_ast::pochhammer_ast(args));
    }
    "FactorialPower" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::factorial_power_ast(args));
    }
    "Gamma" if args.len() == 1 || args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::gamma_ast(args));
    }
    "BesselJ" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_j_ast(args));
    }
    "MathieuS" if args.len() == 3 => {
      return Some(crate::functions::math_ast::mathieu_s_ast(args));
    }
    "MathieuSPrime" if args.len() == 3 => {
      return Some(crate::functions::math_ast::mathieu_s_prime_ast(args));
    }
    "BesselY" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_y_ast(args));
    }
    "BesselJZero" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_j_zero_ast(args));
    }
    "BesselYZero" if args.len() == 2 => {
      return Some(crate::functions::math_ast::bessel_y_zero_ast(args));
    }
    "HankelH1" if args.len() == 2 => {
      return Some(crate::functions::math_ast::hankel_h1_ast(args));
    }
    "HankelH2" if args.len() == 2 => {
      return Some(crate::functions::math_ast::hankel_h2_ast(args));
    }
    "KelvinBer" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::kelvin_ber_ast(args));
    }
    "KelvinBei" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::kelvin_bei_ast(args));
    }
    "KelvinKer" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::kelvin_ker_ast(args));
    }
    "KelvinKei" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::kelvin_kei_ast(args));
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
    "ScorerHi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::scorer_hi_ast(args));
    }
    "ScorerGi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::scorer_gi_ast(args));
    }
    "AiryBiPrime" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_bi_prime_ast(args));
    }
    "AiryAiZero" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_ai_zero_ast(args));
    }
    "AiryBiZero" if args.len() == 1 => {
      return Some(crate::functions::math_ast::airy_bi_zero_ast(args));
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
    "MittagLefflerE" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::mittag_leffler_e_ast(args));
    }
    "NevilleThetaS" | "NevilleThetaC" | "NevilleThetaD" | "NevilleThetaN" => {
      return Some(crate::functions::math_ast::neville_theta_ast(name, args));
    }
    "EllipticK" if args.len() == 1 => {
      return Some(crate::functions::math_ast::elliptic_k_ast(args));
    }
    "EllipticExp" if args.len() == 2 => {
      return Some(crate::functions::math_ast::elliptic_exp_ast(args));
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
    "JacobiEpsilon" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_epsilon_ast(args));
    }
    "EllipticNomeQ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::elliptic_nome_q_ast(args));
    }
    "InverseEllipticNomeQ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::inverse_elliptic_nome_q_ast(
        args,
      ));
    }
    "CarlsonRC" if args.len() == 2 => {
      return Some(crate::functions::math_ast::carlson_rc_ast(args));
    }
    "CarlsonRF" if args.len() == 3 => {
      return Some(crate::functions::math_ast::carlson_rf_ast(args));
    }
    "CarlsonRD" if args.len() == 3 => {
      return Some(crate::functions::math_ast::carlson_rd_ast(args));
    }
    "CarlsonRJ" if args.len() == 4 => {
      return Some(crate::functions::math_ast::carlson_rj_ast(args));
    }
    "CarlsonRG" if args.len() == 3 => {
      return Some(crate::functions::math_ast::carlson_rg_ast(args));
    }
    "CarlsonRE" if args.len() == 2 => {
      return Some(crate::functions::math_ast::carlson_re_ast(args));
    }
    "DedekindEta" if args.len() == 1 => {
      return Some(crate::functions::math_ast::dedekind_eta_ast(args));
    }
    "ModularLambda" if args.len() == 1 => {
      return Some(crate::functions::math_ast::modular_lambda_ast(args));
    }
    "WeierstrassInvariants" if args.len() == 1 => {
      return Some(crate::functions::math_ast::weierstrass_invariants_ast(
        args,
      ));
    }
    "WeierstrassHalfPeriods" if args.len() == 1 => {
      return Some(crate::functions::math_ast::weierstrass_half_periods_ast(
        args,
      ));
    }
    "KleinInvariantJ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::klein_invariant_j_ast(args));
    }
    "Zeta" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::zeta_ast(args));
    }
    "HurwitzZeta" if args.len() == 2 => {
      return Some(crate::functions::math_ast::hurwitz_zeta_public_ast(args));
    }
    "DirichletEta" if args.len() == 1 => {
      return Some(crate::functions::math_ast::dirichlet_eta_ast(args));
    }
    "DirichletLambda" if args.len() == 1 => {
      return Some(crate::functions::math_ast::dirichlet_lambda_ast(args));
    }
    "DirichletBeta" if args.len() == 1 => {
      return Some(crate::functions::math_ast::dirichlet_beta_ast(args));
    }
    "RiemannSiegelZ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::riemann_siegel_z_ast(args));
    }
    "RiemannSiegelTheta" if args.len() == 1 => {
      return Some(crate::functions::math_ast::riemann_siegel_theta_ast(args));
    }
    "PolyGamma" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::polygamma_ast(args));
    }
    "LegendreP" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::legendre_p_ast(args));
    }
    "JacobiP" if args.len() == 4 => {
      return Some(crate::functions::math_ast::jacobi_p_ast(args));
    }
    "SphericalHarmonicY" if args.len() == 4 => {
      return Some(crate::functions::math_ast::spherical_harmonic_y_ast(args));
    }
    "LegendreQ" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::legendre_q_ast(args));
    }
    "PolyLog" if args.len() == 2 => {
      return Some(crate::functions::math_ast::polylog_ast(args));
    }
    "PolygonalNumber" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::polygonal_number_ast(args));
    }
    "Hyperfactorial" if args.len() == 1 => {
      return Some(crate::functions::math_ast::hyperfactorial_ast(args));
    }
    "DeBruijnSequence" if args.len() == 2 => {
      return Some(crate::functions::math_ast::debruijn_sequence_ast(args));
    }
    "BellY" if args.len() == 3 => {
      return Some(crate::functions::math_ast::bell_y_ast(args));
    }
    "FiniteGroupCount" if args.len() == 1 => {
      return Some(crate::functions::math_ast::finite_group_count_ast(args));
    }
    "FiniteAbelianGroupCount" if args.len() == 1 => {
      return Some(crate::functions::math_ast::finite_abelian_group_count_ast(
        args,
      ));
    }
    "PerfectNumber" if args.len() == 1 => {
      return Some(crate::functions::math_ast::perfect_number_ast(args));
    }
    "RamanujanTau" if args.len() == 1 => {
      return Some(crate::functions::math_ast::ramanujan_tau_ast(args));
    }
    "RamanujanTauTheta" if args.len() == 1 => {
      return Some(crate::functions::math_ast::ramanujan_tau_theta_ast(args));
    }
    "RamanujanTauZ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::ramanujan_tau_z_ast(args));
    }
    "RamanujanTauL" if args.len() == 1 => {
      return Some(crate::functions::math_ast::ramanujan_tau_l_ast(args));
    }
    "PowersRepresentations" if args.len() == 3 => {
      return Some(crate::functions::math_ast::powers_representations_ast(
        args,
      ));
    }
    "BarnesG" if args.len() == 1 => {
      return Some(crate::functions::math_ast::barnes_g_ast(args));
    }
    "LerchPhi" if args.len() == 3 => {
      return Some(crate::functions::math_ast::lerch_phi_ast(args));
    }
    "HurwitzLerchPhi" if args.len() == 3 => {
      return Some(crate::functions::math_ast::hurwitz_lerch_phi_ast(args));
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
    "FresnelS" if args.len() == 1 => {
      return Some(crate::functions::math_ast::fresnel_s_ast(args));
    }
    "FresnelC" if args.len() == 1 => {
      return Some(crate::functions::math_ast::fresnel_c_ast(args));
    }
    "FresnelF" | "FresnelG" if args.len() == 1 => {
      return Some(crate::functions::math_ast::fresnel_fg_ast(name, args));
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
    // Generalized form BetaRegularized[z0, z1, a, b] =
    // BetaRegularized[z1, a, b] - BetaRegularized[z0, a, b]. wolframscript
    // evaluates it (exactly or numerically) when every argument is a number,
    // and otherwise keeps the 4-argument form symbolic.
    "BetaRegularized" if args.len() == 4 => {
      if args.iter().all(is_numeric_literal) {
        let f = |z: &Expr| Expr::FunctionCall {
          name: "BetaRegularized".to_string(),
          args: vec![z.clone(), args[2].clone(), args[3].clone()].into(),
        };
        let diff = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(f(&args[1])),
          right: Box::new(f(&args[0])),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&diff));
      }
      return Some(Ok(unevaluated("BetaRegularized", args)));
    }
    "MarcumQ" if args.len() == 3 || args.len() == 4 => {
      return Some(crate::functions::math_ast::marcum_q_ast(args));
    }
    "OwenT" if args.len() == 2 => {
      return Some(crate::functions::math_ast::owen_t_ast(args));
    }
    "GammaRegularized" if args.len() == 2 => {
      return Some(crate::functions::math_ast::gamma_regularized_ast(args));
    }
    "InverseGammaRegularized" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::inverse_gamma_regularized_ast(
        args,
      ));
    }
    "InverseBetaRegularized" if args.len() == 3 || args.len() == 4 => {
      return Some(crate::functions::math_ast::inverse_beta_regularized_ast(
        args,
      ));
    }
    // Generalized form GammaRegularized[a, z0, z1] = GammaRegularized[a, z0] -
    // GammaRegularized[a, z1]. wolframscript keeps this symbolic except for the
    // elementary a == 1 case (GammaRegularized[1, z] = E^-z). Match that and
    // otherwise leave the 3-arg form unevaluated (rather than emitting ::argrx).
    "GammaRegularized" if args.len() == 3 => {
      // With an inexact (machine-real) argument, evaluate numerically as the
      // difference GammaRegularized[a, z0] - GammaRegularized[a, z1]; exact
      // arguments stay symbolic, matching wolframscript.
      if args.iter().any(|a| matches!(a, Expr::Real(_))) {
        let g = |z: &Expr| Expr::FunctionCall {
          name: "GammaRegularized".to_string(),
          args: vec![args[0].clone(), z.clone()].into(),
        };
        let diff = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(g(&args[1])),
          right: Box::new(g(&args[2])),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&diff));
      }
      if matches!(&args[0], Expr::Integer(1)) {
        let exp_neg = |z: &Expr| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::Constant("E".to_string()),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), z.clone()].into(),
            },
          ]
          .into(),
        };
        let diff = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(exp_neg(&args[1])),
          right: Box::new(exp_neg(&args[2])),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&diff));
      }
      return Some(Ok(unevaluated("GammaRegularized", args)));
    }
    "Hypergeometric1F1Regularized" if args.len() == 3 => {
      return Some(
        crate::functions::math_ast::hypergeometric_1f1_regularized_ast(args),
      );
    }
    "WhittakerM" if args.len() == 3 => {
      return Some(crate::functions::math_ast::whittaker_m_ast(args));
    }
    "WhittakerW" if args.len() == 3 => {
      return Some(crate::functions::math_ast::whittaker_w_ast(args));
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
    "GegenbauerC" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::gegenbauer_c_ast(args));
    }
    "ZernikeR" if args.len() == 3 => {
      return Some(crate::functions::math_ast::zernike_r_ast(args));
    }
    "LaguerreL" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::laguerre_l_ast(args));
    }
    "Beta" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::beta_ast(args));
    }
    // Generalized incomplete Beta: Beta[z0, z1, a, b] = Beta[z1, a, b] -
    // Beta[z0, a, b], evaluated when all arguments are numbers (symbolic
    // otherwise), matching wolframscript.
    "Beta" if args.len() == 4 => {
      if args.iter().all(is_numeric_literal) {
        let f = |z: &Expr| Expr::FunctionCall {
          name: "Beta".to_string(),
          args: vec![z.clone(), args[2].clone(), args[3].clone()].into(),
        };
        let diff = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(f(&args[1])),
          right: Box::new(f(&args[0])),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&diff));
      }
      return Some(Ok(unevaluated("Beta", args)));
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
    "QPochhammer" if (1..=3).contains(&args.len()) => {
      return Some(crate::functions::math_ast::q_pochhammer_ast(args));
    }
    "CoulombF" => {
      return Some(crate::functions::math_ast::coulomb_f_ast(args));
    }
    "CoulombG" => {
      return Some(crate::functions::math_ast::coulomb_g_ast(args));
    }
    "CoulombH1" => {
      return Some(crate::functions::math_ast::coulomb_h1_ast(args));
    }
    "CoulombH2" => {
      return Some(crate::functions::math_ast::coulomb_h2_ast(args));
    }
    "SphericalBesselJ" if args.len() == 2 => {
      return Some(crate::functions::math_ast::spherical_bessel_j_ast(args));
    }
    "SphericalBesselY" if args.len() == 2 => {
      return Some(crate::functions::math_ast::spherical_bessel_y_ast(args));
    }
    "SphericalHankelH1" if args.len() == 2 => {
      return Some(crate::functions::math_ast::spherical_hankel_h1_ast(args));
    }
    "SphericalHankelH2" if args.len() == 2 => {
      return Some(crate::functions::math_ast::spherical_hankel_h2_ast(args));
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
    "SetPrecision" if args.len() == 2 => {
      return Some(crate::functions::math_ast::set_precision_ast(args));
    }
    "SetAccuracy" if args.len() == 2 => {
      return Some(crate::functions::math_ast::set_accuracy_ast(args));
    }
    "RandomInteger" => {
      return Some(crate::functions::math_ast::random_integer_ast(args));
    }
    "RandomPrime" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_prime_ast(args));
    }
    "RandomReal" => {
      return Some(crate::functions::math_ast::random_real_ast(args));
    }
    "RandomComplex" => {
      return Some(crate::functions::math_ast::random_complex_ast(args));
    }
    "RandomColor" if args.len() <= 1 => {
      return Some(crate::functions::math_ast::random_color_ast(args));
    }
    "RandomDate" if args.len() <= 1 => {
      return Some(crate::functions::math_ast::random_date_ast(args));
    }
    "RandomTime" if args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_time_ast(args));
    }
    "Random" => {
      return Some(crate::functions::math_ast::random_ast(args));
    }
    "RandomChoice" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_choice_ast(args));
    }
    "RandomGraph" if !args.is_empty() => {
      return Some(crate::functions::math_ast::random_graph_ast(args));
    }
    "RandomSample" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_sample_ast(args));
    }
    "RandomPermutation" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::random_permutation_ast(args));
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
    "SurvivalFunction" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::survival_function_ast(args));
    }
    "HazardFunction" if args.len() == 2 => {
      return Some(crate::functions::math_ast::hazard_function_ast(args));
    }
    "LogLikelihood" if args.len() == 2 => {
      return Some(crate::functions::math_ast::log_likelihood_ast(args));
    }
    "TransformedDistribution" if args.len() == 2 => {
      return Some(crate::functions::math_ast::transformed_distribution_ast(
        args,
      ));
    }
    "CorrelationFunction" if args.len() == 2 => {
      return Some(crate::functions::math_ast::correlation_function_ast(args));
    }
    // SliceDistribution[proc, t] materializes the process time-slice
    // distribution.
    "SliceDistribution" if args.len() == 2 => {
      if let Expr::FunctionCall {
        name: pname,
        args: dargs,
      } = &args[0]
        && let Some(slice) = crate::functions::math_ast::distributions_slice(
          pname, dargs, &args[1],
        )
      {
        return Some(crate::evaluator::evaluate_expr_to_expr(&slice));
      }
      return Some(Ok(unevaluated("SliceDistribution", args)));
    }
    "CorrelationFunction" if args.len() == 3 => {
      if let Some(result) =
        crate::functions::math_ast::statistics_process_correlation(
          &args[0], &args[1], &args[2],
        )
      {
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
      return Some(Ok(unevaluated("CorrelationFunction", args)));
    }
    "AbsoluteCorrelationFunction" if args.len() == 3 => {
      if let Some(result) =
        crate::functions::math_ast::statistics_process_absolute_correlation(
          &args[0], &args[1], &args[2],
        )
      {
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
      return Some(Ok(unevaluated("AbsoluteCorrelationFunction", args)));
    }
    "AbsoluteCorrelationFunction" => {
      return Some(crate::functions::absolute_correlation_function_ast(args));
    }
    "BiweightMidvariance" => {
      return Some(
        crate::functions::math_ast::statistics_biweight_midvariance(args),
      );
    }
    // StieltjesGamma[0] is EulerGamma; positive integers and symbols stay
    // symbolic (N picks up the machine values via try_eval_to_f64); other
    // arguments emit StieltjesGamma::intnm like wolframscript
    "StieltjesGamma" if args.len() == 1 => {
      let unevaluated = unevaluated("StieltjesGamma", args);
      return Some(Ok(match &args[0] {
        Expr::Integer(0) => Expr::Identifier("EulerGamma".to_string()),
        Expr::Integer(n) if *n >= 1 => unevaluated,
        Expr::Identifier(_) => unevaluated,
        other => {
          crate::emit_message(&format!(
            "StieltjesGamma::intnm: Non-negative machine-sized integer \
             expected at position 1 in StieltjesGamma[{}].",
            crate::syntax::expr_to_string(other)
          ));
          unevaluated
        }
      }));
    }
    // Generalized StieltjesGamma[n, a]: only n = 0 has the closed form
    // -PolyGamma[0, a]; higher orders and symbolic n stay symbolic.
    "StieltjesGamma" if args.len() == 2 => {
      let unevaluated = unevaluated("StieltjesGamma", args);
      return Some(Ok(match &args[0] {
        Expr::Integer(0) => {
          // Expand distributes the minus sign so integer a matches
          // wolframscript (e.g. StieltjesGamma[0, 2] -> -1 + EulerGamma).
          let expr = Expr::FunctionCall {
            name: "Expand".to_string(),
            args: vec![Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(-1),
                Expr::FunctionCall {
                  name: "PolyGamma".to_string(),
                  args: vec![Expr::Integer(0), args[1].clone()].into(),
                },
              ]
              .into(),
            }]
            .into(),
          };
          crate::evaluator::evaluate_expr_to_expr(&expr).unwrap_or(unevaluated)
        }
        Expr::Integer(n) if *n >= 1 => unevaluated,
        Expr::Identifier(_) => unevaluated,
        other => {
          crate::emit_message(&format!(
            "StieltjesGamma::intnm: Non-negative machine-sized integer \
             expected at position 1 in StieltjesGamma[{}].",
            crate::syntax::expr_to_string(other)
          ));
          unevaluated
        }
      }));
    }
    // InverseCDF[dist, q] coincides with Quantile[dist, q] for the
    // closed-form distributions (numeric q only)
    "InverseCDF" if args.len() == 2 => {
      // InverseCDF[dist, {p1, ...}] threads over the probability list (the
      // second argument is always a scalar probability).
      if matches!(&args[0], Expr::FunctionCall { .. })
        && let Expr::List(ps) = &args[1]
      {
        let results: Result<Vec<Expr>, _> = ps
          .iter()
          .map(|p| {
            dispatch_math_functions("InverseCDF", &[args[0].clone(), p.clone()])
              .unwrap_or_else(|| {
                Ok(Expr::FunctionCall {
                  name: "InverseCDF".to_string(),
                  args: vec![args[0].clone(), p.clone()].into(),
                })
              })
          })
          .collect();
        return Some(results.map(|v| Expr::List(v.into())));
      }
      if let Expr::FunctionCall {
        name: dist_name,
        args: dargs,
      } = &args[0]
        && let Some(result) =
          crate::functions::math_ast::quantile_distribution_closed_form(
            dist_name, dargs, &args[1],
          )
      {
        return Some(Ok(result));
      }
      return Some(Ok(unevaluated("InverseCDF", args)));
    }
    "InverseSurvivalFunction" if args.len() == 2 => {
      if let Expr::FunctionCall {
        name: dist_name,
        args: dargs,
      } = &args[0]
        && let Some(result) =
          crate::functions::math_ast::inverse_survival_closed_form(
            dist_name, dargs, &args[1],
          )
      {
        return Some(Ok(result));
      }
      return Some(Ok(unevaluated("InverseSurvivalFunction", args)));
    }
    "Probability" if args.len() == 2 || args.len() == 3 => {
      // The optional third argument is `Assumptions -> …`, which we
      // currently ignore — pass through only the event and distribution.
      return Some(crate::functions::math_ast::probability_ast(&args[..2]));
    }
    "Expectation" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::expectation_ast(&args[..2]));
    }
    "NProbability" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::n_probability_ast(&args[..2]));
    }
    "NExpectation" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::n_expectation_ast(&args[..2]));
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
      // Sinc is even: Sinc[-x] = Sinc[x]. Fold the negation before the Sin[x]/x
      // rewrite below (Sin[-x] = -Sin[x] would otherwise expand it incorrectly).
      if let Some(pos) = crate::functions::math_ast::strip_negation(&args[0]) {
        return Some(crate::evaluator::evaluate_function_call_ast(
          "Sinc",
          &[pos],
        ));
      }
      // Sinc[0] = 1. An inexact zero gives the machine real 1., not the exact
      // integer 1.
      match &args[0] {
        Expr::Integer(0) => return Some(Ok(Expr::Integer(1))),
        Expr::Real(f) if *f == 0.0 => {
          return Some(Ok(Expr::Real(1.0)));
        }
        _ => {}
      }
      // Limits at infinity: Sin is bounded while the denominator diverges, so
      // Sinc[±Infinity] = 0. An undirected ComplexInfinity is Indeterminate.
      match &args[0] {
        Expr::Identifier(s) if s == "Infinity" => {
          return Some(Ok(Expr::Integer(0)));
        }
        Expr::Identifier(s) if s == "ComplexInfinity" => {
          return Some(Ok(Expr::Identifier("Indeterminate".to_string())));
        }
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") =>
        {
          return Some(Ok(Expr::Integer(0)));
        }
        Expr::FunctionCall { name, args: dargs }
          if name == "DirectedInfinity" && dargs.len() == 1 =>
        {
          if matches!(&dargs[0], Expr::Integer(1) | Expr::Integer(-1)) {
            return Some(Ok(Expr::Integer(0)));
          }
        }
        _ => {}
      }
      // Expand to Sin[x]/x only when Sin[x] evaluated to a variable-free closed
      // form — a genuine value (number/radical) as for Sinc[Pi/2] or Sinc[2 Pi].
      // When Sin merely rewrites to another symbolic form (Sin[Pi + x] = -Sin[x],
      // still containing a Sin; or Sin[Pi/2 + x] = Cos[x], still containing the
      // free variable x), wolframscript keeps Sinc[x] symbolic, so leave it
      // unevaluated. `not_a_value` flags either a residual Sin head or any free
      // symbol.
      fn not_a_value(e: &Expr) -> bool {
        match e {
          Expr::Identifier(_) => true,
          Expr::FunctionCall { name, args } => {
            name == "Sin" || args.iter().any(not_a_value)
          }
          Expr::BinaryOp { left, right, .. } => {
            not_a_value(left) || not_a_value(right)
          }
          Expr::UnaryOp { operand, .. } => not_a_value(operand),
          _ => false,
        }
      }
      let sin_result = crate::functions::math_ast::sin_ast(args);
      match sin_result {
        Ok(ref sin_val) => {
          if !not_a_value(sin_val) {
            let div_expr = Expr::BinaryOp {
              op: BinaryOperator::Divide,
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
      // Haversine is even: Haversine[-x] = Haversine[x].
      if let Some(pos) = crate::functions::math_ast::strip_negation(&args[0]) {
        return Some(crate::evaluator::evaluate_function_call_ast(
          "Haversine",
          &[pos],
        ));
      }
      // For numeric args (containing a Real literal), compute Sin[x/2]^2 —
      // numerically more stable than (1 - Cos[x])/2 and produces the same
      // f64 value as wolframscript.
      fn contains_real(e: &Expr) -> bool {
        match e {
          Expr::Real(_) | Expr::BigFloat(_, _) => true,
          Expr::BinaryOp { left, right, .. } => {
            contains_real(left) || contains_real(right)
          }
          Expr::UnaryOp { operand, .. } => contains_real(operand),
          Expr::FunctionCall { args, .. } => args.iter().any(contains_real),
          _ => false,
        }
      }
      if contains_real(&args[0]) {
        let half = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(args[0].clone()),
          right: Box::new(Expr::Integer(2)),
        };
        let sin_expr = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![half].into(),
        };
        let expr = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(sin_expr),
          right: Box::new(Expr::Integer(2)),
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&expr));
      }
      // Exact args: Haversine[x] = (1 - Cos[x])/2. wolframscript evaluates the
      // nice-angle cases (Pi/3 -> 1/4, 2 Pi -> 0, 2 Pi/3 -> 3/4, ...). Return
      // the computed value only when it reduces to a rational number — radical
      // results (e.g. Pi/4, Pi/5) are left unevaluated to avoid a canonical
      // radical-form divergence from wolframscript.
      let cos_x = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![args[0].clone()].into(),
      };
      let half = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(cos_x),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      if let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&half) {
        let is_rational = matches!(&result, Expr::Integer(_))
          || matches!(&result, Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2);
        if is_rational {
          return Some(Ok(result));
        }
      }
    }
    "InverseHaversine" if args.len() == 1 => {
      // Complex-numeric fast path: 2 * ArcSin[Sqrt[z]] computed with f64.
      // Only triggered when the argument contains a float component and has
      // a non-zero imaginary part; purely real or exact cases fall through
      // to the symbolic rewrite below.
      fn contains_float_expr(e: &Expr) -> bool {
        match e {
          Expr::Real(_) | Expr::BigFloat(_, _) => true,
          Expr::BinaryOp { left, right, .. } => {
            contains_float_expr(left) || contains_float_expr(right)
          }
          Expr::UnaryOp { operand, .. } => contains_float_expr(operand),
          Expr::FunctionCall { args, .. } => {
            args.iter().any(contains_float_expr)
          }
          _ => false,
        }
      }
      if contains_float_expr(&args[0])
        && let Some((a, b)) =
          crate::functions::math_ast::try_extract_complex_float(&args[0])
        && b != 0.0
      {
        // Compute 2·arcsin(sqrt(z)) via Hull–Fairgrieve–Tang decomposition at
        // ~130 bits of precision, then round to f64. Working in extended
        // precision produces a correctly-rounded result that matches
        // wolframscript bit-for-bit, avoiding the ULP drift that the pure-f64
        // path accumulates across hypot → sqrt → asin.
        use astro_float::{BigFloat, Consts, RoundingMode};
        let bits = 192usize;
        let rm = RoundingMode::ToEven;
        let mut cc = match Consts::new() {
          Ok(c) => c,
          Err(e) => {
            return Some(Err(InterpreterError::EvaluationError(format!(
              "BigFloat init error: {}",
              e
            ))));
          }
        };
        let big = |v: f64| BigFloat::from_f64(v, bits);
        let two = big(2.0);
        let one = big(1.0);
        // Complex sqrt of (a + b i): c = sqrt((|z|+a)/2), d = ±sqrt((|z|-a)/2).
        let a_bf = big(a);
        let b_bf = big(b);
        let r = a_bf
          .mul(&a_bf, bits, rm)
          .add(&b_bf.mul(&b_bf, bits, rm), bits, rm)
          .sqrt(bits, rm);
        let c_bf = r.add(&a_bf, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        let d_abs = r.sub(&a_bf, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        let d_bf = if b >= 0.0 { d_abs } else { d_abs.neg() };
        // α = (|w+1| + |w−1|)/2, β = (|w+1| − |w−1|)/2
        let cp1 = c_bf.add(&one, bits, rm);
        let cm1 = c_bf.sub(&one, bits, rm);
        let d_sq = d_bf.mul(&d_bf, bits, rm);
        let rp1 = cp1.mul(&cp1, bits, rm).add(&d_sq, bits, rm).sqrt(bits, rm);
        let rm1 = cm1.mul(&cm1, bits, rm).add(&d_sq, bits, rm).sqrt(bits, rm);
        let alpha = rp1.add(&rm1, bits, rm).div(&two, bits, rm);
        let beta = rp1.sub(&rm1, bits, rm).div(&two, bits, rm);
        // arcsin(β): β ∈ [-1, 1].
        let asin_re_bf = beta.asin(bits, rm, &mut cc);
        // log(α + sqrt(α² − 1))
        let alpha_sq_m1 = alpha.mul(&alpha, bits, rm).sub(&one, bits, rm);
        let log_arg = alpha.add(&alpha_sq_m1.sqrt(bits, rm), bits, rm);
        let asin_im_bf = log_arg.ln(bits, rm, &mut cc);
        let asin_im_bf = if b >= 0.0 {
          asin_im_bf
        } else {
          asin_im_bf.neg()
        };
        fn bf_to_f64(bf: &BigFloat, rm: RoundingMode, cc: &mut Consts) -> f64 {
          match bf.format(astro_float::Radix::Dec, rm, cc) {
            Ok(s) => {
              // Format: "-.123e3" style — f64 parser accepts scientific notation
              // but astro-float uses a leading dot (".123e3"); normalize to "0.123e3".
              let s2 = if let Some(rest) = s.strip_prefix('.') {
                format!("0.{}", rest)
              } else if let Some(rest) = s.strip_prefix("-.") {
                format!("-0.{}", rest)
              } else {
                s
              };
              s2.parse::<f64>().unwrap_or(0.0)
            }
            Err(_) => 0.0,
          }
        }
        let two_re = bf_to_f64(&asin_re_bf.mul(&two, bits, rm), rm, &mut cc);
        let two_im = bf_to_f64(&asin_im_bf.mul(&two, bits, rm), rm, &mut cc);
        return Some(crate::evaluator::evaluate_function_call_ast(
          "Complex",
          &[Expr::Real(two_re), Expr::Real(two_im)],
        ));
      }
      // Symbolic / real path: 2 * ArcSin[Sqrt[x]]. wolframscript only rewrites
      // to this form when the result is a clean closed form (e.g.
      // InverseHaversine[1/2] -> Pi/2, [1] -> Pi); otherwise it keeps the
      // InverseHaversine wrapper. So apply the rewrite but, if the result still
      // contains an unevaluated ArcSin (the rewrite did not simplify), return
      // the symbolic InverseHaversine instead. Real (inexact) arguments always
      // numericize the ArcSin, so they are unaffected.
      let sqrt_expr = make_sqrt(args[0].clone());
      let asin_expr = Expr::FunctionCall {
        name: "ArcSin".to_string(),
        args: vec![sqrt_expr].into(),
      };
      let expr = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(asin_expr),
      };
      let result = match crate::evaluator::evaluate_expr_to_expr(&expr) {
        Ok(r) => r,
        Err(e) => return Some(Err(e)),
      };
      fn contains_arcsin(e: &Expr) -> bool {
        match e {
          Expr::FunctionCall { name, args } => {
            name == "ArcSin" || args.iter().any(contains_arcsin)
          }
          Expr::BinaryOp { left, right, .. } => {
            contains_arcsin(left) || contains_arcsin(right)
          }
          Expr::UnaryOp { operand, .. } => contains_arcsin(operand),
          Expr::List(items) => items.iter().any(contains_arcsin),
          _ => false,
        }
      }
      if contains_arcsin(&result) {
        return Some(Ok(unevaluated("InverseHaversine", args)));
      }
      return Some(Ok(result));
    }
    "Exp" if args.len() == 1 => {
      return Some(crate::functions::math_ast::exp_ast(args));
    }
    "Erf" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::erf_ast(args));
    }
    "Erfc" if args.len() == 1 => {
      return Some(crate::functions::math_ast::erfc_ast(args));
    }
    "Erfi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::erfi_ast(args));
    }
    "DawsonF" if args.len() == 1 => {
      return Some(crate::functions::math_ast::dawson_f_ast(args));
    }
    "InverseErf" if args.len() == 1 => {
      return Some(crate::functions::math_ast::inverse_erf_ast(args));
    }
    "InverseErfc" if args.len() == 1 => {
      return Some(crate::functions::math_ast::inverse_erfc_ast(args));
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
    "RealExponent" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::real_exponent_ast(args));
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
    "LambertW" => {
      // LambertW is an alias for ProductLog — rewrite and evaluate
      return Some(crate::functions::math_ast::product_log_ast(args));
    }
    "ProductLog" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::product_log_ast(args));
    }
    "Prime" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_ast(args));
    }
    "Fibonacci" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::fibonacci_ast(args));
    }
    "LinearRecurrence" if args.len() == 3 => {
      return Some(crate::functions::math_ast::linear_recurrence_ast(args));
    }
    "IntegerDigits" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::integer_digits_ast(args));
    }
    "NumberExpand" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::number_expand_ast(args));
    }
    "NumberDecompose" if args.len() == 2 => {
      return Some(crate::functions::math_ast::number_decompose_ast(args));
    }
    "NumberCompose" if args.len() == 2 => {
      return Some(crate::functions::math_ast::number_compose_ast(args));
    }
    "NumberDigit" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::number_digit_ast(args));
    }
    "RealDigits" if !args.is_empty() && args.len() <= 4 => {
      return Some(crate::functions::math_ast::real_digits_ast(args));
    }
    "FromDigits" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::from_digits_ast(args));
    }
    "IntegerName" if args.len() == 1 || args.len() == 2 => {
      // Second argument (e.g., "Words") is the format hint; ignored for now.
      return Some(crate::functions::math_ast::integer_name_ast(args));
    }
    "RomanNumeral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::roman_numeral_ast(args));
    }
    "FromRomanNumeral" if args.len() == 1 => {
      return Some(crate::functions::math_ast::from_roman_numeral_ast(args));
    }
    "FactorInteger" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::factor_integer_ast(args));
    }
    "PrimeOmega" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_omega_ast(args));
    }
    "PrimeNu" if args.len() == 1 => {
      return Some(crate::functions::math_ast::prime_nu_ast(args));
    }
    "MantissaExponent" if args.len() == 1 || args.len() == 2 => {
      use crate::functions::math_ast::try_eval_to_f64;
      // Numerically evaluate base. If 2-arg form has a non-numeric base
      // (incl. symbolic constants like Pi/E), keep the call unevaluated.
      let base_num: f64 = if args.len() == 2 {
        match try_eval_to_f64(&args[1]) {
          Some(b) if b > 0.0 && b != 1.0 => b,
          _ => {
            return Some(Ok(unevaluated("MantissaExponent", args)));
          }
        }
      } else {
        10.0
      };
      // Numerically evaluate the value to compute the exponent.
      let val_num: f64 = match try_eval_to_f64(&args[0]) {
        Some(v) => v,
        None => {
          return Some(Ok(unevaluated("MantissaExponent", args)));
        }
      };
      if val_num == 0.0 {
        return Some(Ok(Expr::List(
          vec![Expr::Integer(0), Expr::Integer(0)].into(),
        )));
      }
      let e = (val_num.abs().ln() / base_num.ln()).floor() as i128 + 1;
      // Whether the result should be a numeric Real or stay symbolic:
      // any inexact (Real/BigFloat) component forces a Real result.
      fn is_inexact(expr: &Expr) -> bool {
        match expr {
          Expr::Real(_) | Expr::BigFloat(_, _) => true,
          Expr::FunctionCall { args, .. } => args.iter().any(is_inexact),
          Expr::List(items) => items.iter().any(is_inexact),
          _ => false,
        }
      }
      if is_inexact(&args[0]) || (args.len() == 2 && is_inexact(&args[1])) {
        // Scale by the EXACT positive power base^|e| (dividing for e >= 0,
        // multiplying for e < 0) rather than always dividing by base^e: for
        // e < 0, base^e is an inexact fraction that introduces a spurious last
        // digit (e.g. MantissaExponent[0.0012] -> 0.12, not 0.11999...).
        let m = if e >= 0 {
          val_num / base_num.powi(e as i32)
        } else {
          val_num * base_num.powi(-e as i32)
        };
        return Some(Ok(Expr::List(
          vec![Expr::Real(m), Expr::Integer(e)].into(),
        )));
      }
      // Special-case integer base 10 with integer value: keep mantissa
      // exact as a rational (existing behavior).
      if (args.len() == 1 || matches!(&args[1], Expr::Integer(_)))
        && let Expr::Integer(n) = &args[0]
      {
        let base_int: i128 = if args.len() == 2 {
          if let Expr::Integer(b) = &args[1] {
            *b
          } else {
            10
          }
        } else {
          10
        };
        if base_int >= 2 {
          let denom = (base_int as f64).powi(e as i32) as i128;
          if denom > 0 {
            let mantissa = crate::functions::math_ast::make_rational(*n, denom);
            return Some(Ok(Expr::List(
              vec![mantissa, Expr::Integer(e)].into(),
            )));
          }
        }
      }
      // Symbolic mantissa = value * base^(-e). Let Woxi simplify.
      let base_expr = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::Integer(10)
      };
      let mantissa = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          args[0].clone(),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![base_expr, Expr::Integer(-e)].into(),
          },
        ]
        .into(),
      };
      let mantissa_eval =
        crate::evaluator::evaluate_expr_to_expr(&mantissa).unwrap_or(mantissa);
      return Some(Ok(Expr::List(
        vec![mantissa_eval, Expr::Integer(e)].into(),
      )));
    }
    "IntegerPartitions" if !args.is_empty() && args.len() <= 4 => {
      return Some(crate::functions::math_ast::integer_partitions_ast(args));
    }
    "Divisors" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::divisors_ast(args));
    }
    "DivisorSigma" if args.len() == 2 || args.len() == 3 => {
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
                args: vec![Expr::Integer(*p)].into(),
              }));
            }
          }
          return Some(Ok(Expr::Integer(0)));
        }
      }
    }
    "LiouvilleLambda" if args.len() == 1 => {
      // LiouvilleLambda is Listable: thread over a list of arguments.
      if let Expr::List(items) = &args[0] {
        let results: Result<Vec<Expr>, _> = items
          .iter()
          .map(|x| {
            dispatch_math_functions("LiouvilleLambda", std::slice::from_ref(x))
              .unwrap_or_else(|| {
                Ok(Expr::FunctionCall {
                  name: "LiouvilleLambda".to_string(),
                  args: vec![x.clone()].into(),
                })
              })
          })
          .collect();
        return Some(results.map(|v| Expr::List(v.into())));
      }
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
    "CarmichaelLambda" if args.len() == 1 => {
      return Some(crate::functions::math_ast::carmichael_lambda_ast(args));
    }
    "JacobiSymbol" if args.len() == 2 => {
      return Some(crate::functions::math_ast::jacobi_symbol_ast(args));
    }
    // MultiplicativeOrder[k, n, {r1, ..., rs}] — generalized order: the
    // smallest m >= 1 for which k^m is congruent to one of the rᵢ modulo n.
    // Stays unevaluated when no power of k reaches any residue.
    "MultiplicativeOrder" if args.len() == 3 => {
      if let (Expr::Integer(k), Expr::Integer(n), Expr::List(residues)) =
        (&args[0], &args[1], &args[2])
        && *n > 0
      {
        // Normalise the target residues into [0, n).
        let mut targets: Vec<i128> = Vec::with_capacity(residues.len());
        let mut ok = true;
        for r in residues.iter() {
          match r {
            Expr::Integer(ri) => targets.push(((*ri % *n) + *n) % *n),
            _ => {
              ok = false;
              break;
            }
          }
        }
        if ok && !targets.is_empty() {
          let k_mod = ((*k % *n) + *n) % *n;
          let mut power = k_mod;
          let mut seen: Vec<i128> = Vec::new();
          let mut m = 1i128;
          loop {
            if targets.contains(&power) {
              return Some(Ok(Expr::Integer(m)));
            }
            if seen.contains(&power) {
              // Cycled without reaching a residue: no solution.
              break;
            }
            seen.push(power);
            power = (power * k_mod) % *n;
            m += 1;
          }
          // Leave unevaluated when unsolvable.
          return Some(Ok(unevaluated("MultiplicativeOrder", args)));
        }
      }
    }
    "MultiplicativeOrder" if args.len() == 2 => {
      if let (Expr::Integer(a), Expr::Integer(n)) = (&args[0], &args[1])
        && *n > 0
      {
        // Modulo 1 every integer is congruent (the ring is trivial), so the
        // order is 1 for any a. wolframscript: MultiplicativeOrder[a, 1] = 1.
        if *n == 1 {
          return Some(Ok(Expr::Integer(1)));
        }
        let a_mod = ((*a % *n) + *n) % *n;
        if a_mod != 0 && gcd_i128(a_mod, *n) == 1 {
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
    // PrimitiveRoot[n, k] — smallest primitive root modulo n that is >= k.
    // Uses the full list of primitive roots (PrimitiveRootList[n]); returns
    // unevaluated when no primitive root reaches k (matching wolframscript).
    "PrimitiveRoot" if args.len() == 2 => {
      if let (Some(n), Some(k)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
        && n > 1
      {
        let n_u = n as u64;
        let phi = euler_totient(n_u);
        for g in 1..n_u {
          if (g as i128) >= k && is_primitive_root(g, n_u, phi) {
            return Some(Ok(Expr::Integer(g as i128)));
          }
        }
        // No primitive root >= k: leave unevaluated.
        return Some(Ok(unevaluated("PrimitiveRoot", args)));
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
          return Some(Ok(unevaluated("PrimitiveRoot", args)));
        }
        let n_val = *n;
        // Smallest primitive root modulo `m` (g with multiplicative order
        // equal to EulerPhi[m]), or None when m has no primitive root.
        let smallest_pr = |m: i128| -> Option<i128> {
          let phi = crate::functions::math_ast::euler_phi_i128(m);
          let start = if m == 2 { 1 } else { 2 };
          for g in start..m {
            if gcd_i128(g, m) != 1 {
              continue;
            }
            let mut power = g % m;
            let mut order = 1i128;
            while power != 1 && order <= phi {
              power = (power * g) % m;
              order += 1;
            }
            if power == 1 && order == phi {
              return Some(g);
            }
          }
          None
        };
        // Is `m` an odd prime power p^k (k >= 1)?
        let is_odd_prime_power = |mut m: i128| -> bool {
          if m < 3 || m % 2 == 0 {
            return false;
          }
          let mut p = 3i128;
          while p * p <= m && m % p != 0 {
            p += 2;
          }
          let p = if m % p == 0 { p } else { m };
          while m % p == 0 {
            m /= p;
          }
          m == 1
        };
        // For n = 2 p^k (p an odd prime), wolframscript derives the
        // primitive root from p^k rather than searching mod n directly: it
        // takes g = PrimitiveRoot[p^k] and, when g is even, uses g + p^k so
        // that the result is odd (primitive roots mod 2m must be odd).
        // e.g. PrimitiveRoot[10] = 7 (from 2 mod 5), not the smaller 3.
        if n_val % 2 == 0 {
          let m = n_val / 2;
          if m > 1
            && is_odd_prime_power(m)
            && let Some(g) = smallest_pr(m)
          {
            let result = if g % 2 == 0 { g + m } else { g };
            return Some(Ok(Expr::Integer(result)));
          }
        }
        // General case: smallest primitive root modulo n directly.
        match smallest_pr(n_val) {
          Some(g) => return Some(Ok(Expr::Integer(g))),
          None => {
            // No primitive root exists (e.g., n=8, n=12).
            return Some(Ok(unevaluated("PrimitiveRoot", args)));
          }
        }
      }
    }
    "CoprimeQ" => {
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
      return Some(Ok(Expr::List(vec![re, im].into())));
    }
    "AbsArg" if args.len() == 1 => {
      let abs_val = match crate::functions::math_ast::abs_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let arg_val = match crate::functions::math_ast::arg_ast(args) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      return Some(Ok(Expr::List(vec![abs_val, arg_val].into())));
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
    // NumeratorDenominator[e] = {Numerator[e], Denominator[e]}
    "NumeratorDenominator" if args.len() == 1 => {
      let num = crate::evaluator::evaluate_function_call_ast(
        "Numerator",
        std::slice::from_ref(&args[0]),
      );
      let den = crate::evaluator::evaluate_function_call_ast(
        "Denominator",
        std::slice::from_ref(&args[0]),
      );
      return match (num, den) {
        (Ok(n), Ok(d)) => Some(Ok(Expr::List(vec![n, d].into()))),
        (Err(e), _) | (_, Err(e)) => Some(Err(e)),
      };
    }
    // BitGet[n, k] — bit k of n, using two's complement for negative n
    "BitGet" if args.len() == 2 => {
      use num_bigint::BigInt;
      use num_traits::{One, Signed, Zero};
      let n: Option<BigInt> = match &args[0] {
        Expr::Integer(v) => Some(BigInt::from(*v)),
        Expr::BigInteger(v) => Some(v.clone()),
        _ => None,
      };
      let k: Option<u64> = match &args[1] {
        Expr::Integer(v) if *v >= 0 => Some(*v as u64),
        Expr::BigInteger(v) if !v.is_negative() => {
          use num_traits::ToPrimitive;
          v.to_u64()
        }
        _ => None,
      };
      if let (Some(n), Some(k)) = (n, k) {
        // Two's complement: bit k of negative n is the complement of
        // bit k of (-n - 1)
        let bit = if n.is_negative() {
          let m = -n - BigInt::one();
          if (m >> k) & BigInt::one() == BigInt::one() {
            0
          } else {
            1
          }
        } else if (n >> k) & BigInt::one() == BigInt::one() {
          1
        } else {
          0
        };
        let _ = BigInt::zero();
        return Some(Ok(Expr::Integer(bit)));
      }
      return Some(Ok(unevaluated("BitGet", args)));
    }
    "Numerator" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::numerator_ast(args));
    }
    "Denominator" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::denominator_ast(args));
    }
    "Binomial" if args.len() == 2 => {
      return Some(crate::functions::math_ast::binomial_ast(args));
    }
    "PascalBinomial" if args.len() == 2 => {
      return Some(crate::functions::math_ast::pascal_binomial_ast(args));
    }
    "CardinalBSplineBasis" if args.len() == 2 => {
      // CardinalBSplineBasis[d, x] is the centered cardinal B-spline of degree
      // d, supported on [-(d+1)/2, (d+1)/2]:
      //   (1/d!) * sum_{k=0}^{d+1} (-1)^k Binomial[d+1, k] (x + (d+1)/2 - k)_+^d
      // where (y)_+^d = y^d for y > 0 and 0 otherwise. Evaluates when d is a
      // non-negative integer and x is numeric; exact x gives an exact result.
      let unevaluated = || unevaluated("CardinalBSplineBasis", args);
      let d = match expr_to_i128(&args[0]) {
        Some(d) if d >= 0 => d,
        Some(_) => {
          // A negative (or otherwise invalid) integer: message + unevaluated.
          crate::emit_message(&format!(
            "CardinalBSplineBasis::intnm: Non-negative machine-sized integer expected at position 1 in CardinalBSplineBasis[{}, {}].",
            crate::syntax::expr_to_string(&args[0]),
            crate::syntax::expr_to_string(&args[1]),
          ));
          return Some(Ok(unevaluated()));
        }
        None => return Some(Ok(unevaluated())),
      };
      // Symbolic x stays unevaluated (wolframscript keeps the piecewise form).
      let Some(xf) = crate::functions::math_ast::try_eval_to_f64(&args[1])
      else {
        return Some(Ok(unevaluated()));
      };
      // Strictly outside the support [-(d+1)/2, (d+1)/2] the value is exactly
      // 0 (an Integer, even for a machine-real x): wolframscript returns 0, not
      // 0.. Inside — including the boundary — the alternating sum is evaluated,
      // which for a machine-real x yields a machine-real result (e.g. 0. at the
      // support edge).
      let half = (d + 1) as f64 / 2.0;
      if xf > half || xf < -half {
        return Some(Ok(Expr::Integer(0)));
      }
      let mut terms: Vec<Expr> = Vec::new();
      for k in 0..=(d + 1) {
        // y = x + (d + 1 - 2k)/2
        let num = d + 1 - 2 * k;
        let shift = if num % 2 == 0 {
          Expr::Integer(num / 2)
        } else {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num), Expr::Integer(2)].into(),
          }
        };
        let y = crate::functions::math_ast::plus_ast(&[args[1].clone(), shift])
          .ok()?;
        // Include the term only where the truncated power is nonzero (y > 0).
        // For d >= 1 the term at y == 0 is 0 anyway; for d == 0 it must be
        // excluded, so the strict `> 0` test is correct for every degree.
        match crate::functions::math_ast::try_eval_to_f64(&y) {
          Some(yf) if yf > 0.0 => {}
          _ => continue,
        }
        let coef = crate::functions::binomial_coeff(d + 1, k);
        let signed = if k % 2 == 0 { coef } else { -coef };
        let power = if d == 0 {
          Expr::Integer(1)
        } else {
          crate::functions::math_ast::power_ast(&[y, Expr::Integer(d)]).ok()?
        };
        terms.push(
          crate::functions::math_ast::times_ast(&[
            Expr::Integer(signed),
            power,
          ])
          .ok()?,
        );
      }
      let sum = if terms.is_empty() {
        // Within/at the support but every truncated power vanishes (the left
        // boundary y = -(d+1)/2): the value is 0, but a machine-real x must
        // yield 0. rather than an exact 0 (strictly-outside already returned an
        // exact 0 above). Matches wolframscript.
        if crate::functions::math_ast::contains_inexact_real(&args[1]) {
          Expr::Real(0.0)
        } else {
          Expr::Integer(0)
        }
      } else {
        crate::functions::math_ast::plus_ast(&terms).ok()?
      };
      // Divide by d! (1 for d = 0).
      let factorial: i128 = (1..=d).product::<i128>().max(1);
      let result = crate::functions::math_ast::times_ast(&[
        sum,
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(factorial)].into(),
        },
      ])
      .ok()?;
      return Some(crate::evaluator::evaluate_expr_to_expr(&result));
    }
    "BernsteinBasis" if args.len() == 3 => {
      // BernsteinBasis[d, n, x] is the piecewise Bernstein basis polynomial:
      //   Binomial[d, n] x^n (1-x)^(d-n)   for 0 <= x <= 1,
      //   0                                otherwise.
      // It evaluates when d, n are integers with 0 <= n <= d and x is numeric;
      // otherwise it stays unevaluated (the lone x-independent exception is
      // d == n == 0, which is 1 for any x).
      let unevaluated = || unevaluated("BernsteinBasis", args);
      let (Some(d), Some(n)) = (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
      else {
        return Some(Ok(unevaluated()));
      };
      let Some(xf) = crate::functions::math_ast::try_eval_to_f64(&args[2])
      else {
        // Symbolic x: only d == n == 0 collapses to the constant 1.
        if d == 0 && n == 0 {
          return Some(Ok(Expr::Integer(1)));
        }
        return Some(Ok(unevaluated()));
      };
      // Out-of-range index keeps the expression unevaluated (with a message).
      if n < 0 {
        crate::emit_message(&format!(
          "BernsteinBasis::intnm: Non-negative machine-sized integer expected at position 2 in BernsteinBasis[{}, {}, {}].",
          crate::syntax::expr_to_string(&args[0]),
          crate::syntax::expr_to_string(&args[1]),
          crate::syntax::expr_to_string(&args[2]),
        ));
        return Some(Ok(unevaluated()));
      }
      if n > d {
        crate::emit_message(&format!(
          "BernsteinBasis::invidx2: Index {} should be a machine-sized integer between 0 and {}.",
          n, d,
        ));
        return Some(Ok(unevaluated()));
      }
      // Outside the unit interval the basis polynomial is exactly 0.
      if !(0.0..=1.0).contains(&xf) {
        return Some(Ok(Expr::Integer(0)));
      }
      // At the boundaries use exact 0/1 so the result is an exact integer
      // (wolframscript: BernsteinBasis[3, 0, 0.] is 1, not 1.).
      let x = if xf == 0.0 {
        Expr::Integer(0)
      } else if xf == 1.0 {
        Expr::Integer(1)
      } else {
        args[2].clone()
      };
      let coef = crate::functions::binomial_coeff(d, n);
      // A zero exponent yields 1 (the x^0 = 1 convention), avoiding a spurious
      // 0^0 = Indeterminate at the x = 0 / x = 1 boundaries.
      let xn = if n == 0 {
        Expr::Integer(1)
      } else {
        crate::functions::math_ast::power_ast(&[x.clone(), Expr::Integer(n)])
          .ok()?
      };
      let one_minus_x_dn = if d - n == 0 {
        Expr::Integer(1)
      } else {
        let one_minus_x = crate::functions::math_ast::plus_ast(&[
          Expr::Integer(1),
          crate::functions::math_ast::times_ast(&[
            Expr::Integer(-1),
            x.clone(),
          ])
          .ok()?,
        ])
        .ok()?;
        crate::functions::math_ast::power_ast(&[
          one_minus_x,
          Expr::Integer(d - n),
        ])
        .ok()?
      };
      let result = crate::functions::math_ast::times_ast(&[
        Expr::Integer(coef),
        xn,
        one_minus_x_dn,
      ])
      .ok()?;
      return Some(Ok(result));
    }
    "Multinomial" => {
      return Some(crate::functions::math_ast::multinomial_ast(args));
    }
    "PowerMod" if args.len() == 3 => {
      return Some(crate::functions::math_ast::power_mod_ast(args));
    }
    "MersennePrimeExponent" if args.len() == 1 => {
      return Some(crate::functions::math_ast::mersenne_prime_exponent_ast(
        args,
      ));
    }
    "MersennePrimeExponentQ" if args.len() == 1 => {
      return Some(crate::functions::math_ast::mersenne_prime_exponent_q_ast(
        args,
      ));
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
    // No arg guard: BitAnd[]/BitOr[]/BitXor[] return their identity element
    // (-1, 0, 0), handled inside the functions.
    "BitAnd" => {
      return Some(crate::functions::math_ast::bit_and_ast(args));
    }
    "BitOr" => {
      return Some(crate::functions::math_ast::bit_or_ast(args));
    }
    "BitXor" => {
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
    "FindDivisions" if args.len() == 2 => {
      return Some(crate::functions::math_ast::find_divisions_ast(args));
    }
    "DigitCount" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::digit_count_ast(args));
    }
    "DigitSum" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::digit_sum_ast(args));
    }
    "ThueMorse" if args.len() == 1 => {
      return Some(crate::functions::math_ast::thue_morse_ast(args));
    }
    "RudinShapiro" if args.len() == 1 => {
      return Some(crate::functions::math_ast::rudin_shapiro_ast(args));
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
    "LucasL" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::lucas_l_ast(args));
    }
    "ChineseRemainder" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::chinese_remainder_ast(args));
    }
    "DivisorSum" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::math_ast::divisor_sum_ast(args));
    }
    "BernoulliB" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::bernoulli_b_ast(args));
    }
    "NorlundB" if args.len() == 2 || args.len() == 3 => {
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
    "ThreeJSymbol" if args.len() == 3 => {
      return Some(crate::functions::math_ast::three_j_symbol_ast(args));
    }
    "SixJSymbol" if args.len() == 2 => {
      return Some(crate::functions::math_ast::six_j_symbol_ast(args));
    }
    "ClebschGordan" if args.len() == 3 => {
      return Some(crate::functions::math_ast::clebsch_gordan_ast(args));
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
    "FrobeniusSolve" if (2..=3).contains(&args.len()) => {
      return Some(crate::functions::math_ast::frobenius_solve_ast(args));
    }
    "HarmonicNumber" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::math_ast::harmonic_number_ast(args));
    }
    "AlternatingHarmonicNumber" if !args.is_empty() && args.len() <= 3 => {
      return Some(
        crate::functions::math_ast::alternating_harmonic_number_ast(args),
      );
    }
    "HyperHarmonicNumber" if (2..=4).contains(&args.len()) => {
      return Some(crate::functions::math_ast::hyper_harmonic_number_ast(args));
    }
    "MultipleHarmonicNumber" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::multiple_harmonic_number_ast(
        args,
      ));
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
    "ComplexExpand" if args.len() == 1 => {
      return Some(complex_expand_ast(&args[0]));
    }
    "ComplexExpand" if args.len() == 2 => {
      // ComplexExpand[expr, vars]: treat each name in `vars` as complex,
      // substituting it with Re[v] + I*Im[v], then expand as usual.
      let vars: Vec<String> = match &args[1] {
        Expr::Identifier(name) => vec![name.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|e| match e {
            Expr::Identifier(name) => Some(name.clone()),
            _ => None,
          })
          .collect(),
        _ => vec![],
      };
      let substituted = substitute_complex_vars(&args[0], &vars);
      return Some(complex_expand_with_expand(&substituted));
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
      // n-dim hyperspherical coordinates: input {r, t_1, ..., t_{n-1}}
      // gives n Cartesian components where
      //   x_k = r * Prod_{i<k} Sin[t_i] * Cos[t_k]   (k < n)
      //   x_n = r * Prod_{i<n} Sin[t_i]
      // The 2-D case reduces to {r*Cos[t], r*Sin[t]}.
      if let Expr::List(ref elems) = args[0]
        && elems.len() >= 2
      {
        let r = &elems[0];
        let thetas = &elems[1..];
        let n = thetas.len() + 1;
        let sin = |t: &Expr| -> Expr {
          Expr::FunctionCall {
            name: "Sin".to_string(),
            args: vec![t.clone()].into(),
          }
        };
        let cos = |t: &Expr| -> Expr {
          Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![t.clone()].into(),
          }
        };
        let mut coords = Vec::with_capacity(n);
        for k in 0..n {
          // Build factors r * Sin[t_0] * ... * Sin[t_{k-1}] * (Cos[t_k] if k<n-1, else 1)
          let mut factors = vec![r.clone()];
          for t in thetas.iter().take(k) {
            factors.push(sin(t));
          }
          if k < n - 1 {
            factors.push(cos(&thetas[k]));
          }
          let product = if factors.len() == 1 {
            factors.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: factors.into(),
            }
          };
          coords.push(product);
        }
        let result = Expr::List(coords.into());
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
    }
    "ToPolarCoordinates" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0]
        && elems.len() >= 2
      {
        // 2-D: {Sqrt[x^2 + y^2], ArcTan[x, y]} (special case — full r,
        // 2-arg ArcTan).
        if elems.len() == 2 {
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
            args: vec![x.clone(), y.clone()].into(),
          };
          let result = Expr::List(vec![r, theta].into());
          return Some(crate::evaluator::evaluate_expr_to_expr(&result));
        }
        // n-D (n ≥ 3): hyperspherical coordinates.
        //   r        = Sqrt[Σ_{i=1..n} x_i^2]
        //   θ_k      = ArcCos[x_k / Sqrt[Σ_{i=k..n} x_i^2]]   for k < n-1
        //   θ_{n-1}  = ArcTan[x_{n-1}, x_n]      (2-arg, ranges over (-π,π])
        let n = elems.len();
        // Helper: Sqrt[Σ_{i=start..n} x_i^2].
        let radical = |start: usize| -> Expr {
          let squares: Vec<Expr> = (start..n)
            .map(|i| Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(elems[i].clone()),
              right: Box::new(Expr::Integer(2)),
            })
            .collect();
          let sum = if squares.len() == 1 {
            squares.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: squares.into(),
            }
          };
          Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![sum].into(),
          }
        };

        let r = radical(0);
        let mut result = Vec::with_capacity(n);
        result.push(r);
        for k in 0..(n - 2) {
          let denom = radical(k);
          let frac = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(elems[k].clone()),
            right: Box::new(denom),
          };
          result.push(Expr::FunctionCall {
            name: "ArcCos".to_string(),
            args: vec![frac].into(),
          });
        }
        // Final angle: 2-arg ArcTan with the last two coordinates.
        result.push(Expr::FunctionCall {
          name: "ArcTan".to_string(),
          args: vec![elems[n - 2].clone(), elems[n - 1].clone()].into(),
        });
        let list = Expr::List(result.into());
        return Some(crate::evaluator::evaluate_expr_to_expr(&list));
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
              args: vec![theta.clone()].into(),
            }),
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![phi.clone()].into(),
          }),
        };
        let y = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(r.clone()),
            right: Box::new(Expr::FunctionCall {
              name: "Sin".to_string(),
              args: vec![theta.clone()].into(),
            }),
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Sin".to_string(),
            args: vec![phi.clone()].into(),
          }),
        };
        let z = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(r.clone()),
          right: Box::new(Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![theta.clone()].into(),
          }),
        };
        let result = Expr::List(vec![x, y, z].into());
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
          args: vec![z.clone(), make_sqrt(xy_sq)].into(),
        };
        let phi_expr = Expr::FunctionCall {
          name: "ArcTan".to_string(),
          args: vec![x.clone(), y.clone()].into(),
        };
        let result = Expr::List(vec![r, theta, phi_expr].into());
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
    }
    "ContinuedFractionK" if args.len() == 2 => {
      // ContinuedFractionK[f, {i, imax}] (imin defaults to 1) or
      // ContinuedFractionK[f, {i, imin, imax}].
      if let Expr::List(ref spec) = args[1]
        && (spec.len() == 2 || spec.len() == 3)
        && let Expr::Identifier(var) = &spec[0]
      {
        let (imin_src, imax_src) = if spec.len() == 2 {
          (Expr::Integer(1), spec[1].clone())
        } else {
          (spec[1].clone(), spec[2].clone())
        };
        let imin_expr = crate::evaluator::evaluate_expr_to_expr(&imin_src)
          .unwrap_or_else(|_| imin_src.clone());
        let imax_expr = crate::evaluator::evaluate_expr_to_expr(&imax_src)
          .unwrap_or_else(|_| imax_src.clone());
        // ContinuedFractionK[1, {n, 1, Infinity}] = -1 + GoldenRatio, the
        // fixed point of x = 1/(1 + x).
        if matches!(&args[0], Expr::Integer(1))
          && matches!(&imin_expr, Expr::Integer(1))
          && matches!(&imax_expr, Expr::Identifier(s) if s == "Infinity")
        {
          let result = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::Integer(-1),
              Expr::Identifier("GoldenRatio".to_string()),
            ]
            .into(),
          };
          return Some(crate::evaluator::evaluate_expr_to_expr(&result));
        }
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
    "ContinuedFractionK" if args.len() == 3 => {
      // ContinuedFractionK[f, g, {n, nmin, nmax}] =
      //   f1/(g1 + f2/(g2 + ... + fk/gk)), with fi, gi = f, g at n = i.
      // The iterator is either {n, nmax} (nmin defaults to 1) or
      // {n, nmin, nmax}, matching Table's iterator forms.
      if let Expr::List(ref spec) = args[2]
        && (spec.len() == 2 || spec.len() == 3)
        && let Expr::Identifier(var) = &spec[0]
      {
        let (nmin_src, nmax_src) = if spec.len() == 2 {
          (Expr::Integer(1), spec[1].clone())
        } else {
          (spec[1].clone(), spec[2].clone())
        };
        let nmin_expr = crate::evaluator::evaluate_expr_to_expr(&nmin_src)
          .unwrap_or_else(|_| nmin_src.clone());
        let nmax_expr = crate::evaluator::evaluate_expr_to_expr(&nmax_src)
          .unwrap_or_else(|_| nmax_src.clone());
        if let (Expr::Integer(nmin), Expr::Integer(nmax)) =
          (&nmin_expr, &nmax_expr)
        {
          let f = &args[0];
          let g = &args[1];
          let mut acc = Expr::Integer(0);
          for k in (*nmin..=*nmax).rev() {
            let fk = crate::syntax::replace_identifier_in_expr(
              f,
              var,
              &Expr::Integer(k),
            );
            let fk_eval =
              crate::evaluator::evaluate_expr_to_expr(&fk).unwrap_or(fk);
            let gk = crate::syntax::replace_identifier_in_expr(
              g,
              var,
              &Expr::Integer(k),
            );
            let gk_eval =
              crate::evaluator::evaluate_expr_to_expr(&gk).unwrap_or(gk);
            // acc = fk / (gk + acc)
            let denom = Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(gk_eval),
              right: Box::new(acc),
            };
            acc = Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(fk_eval),
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
    "AASTriangle" if args.len() == 3 => {
      return Some(crate::functions::math_ast::aas_triangle_ast(args));
    }
    "ASATriangle" if args.len() == 3 => {
      return Some(crate::functions::math_ast::asa_triangle_ast(args));
    }
    "SASTriangle" if args.len() == 3 => {
      return Some(crate::functions::math_ast::sas_triangle_ast(args));
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
        args: vec![Expr::List(
          vec![
            Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
            Expr::List(vec![c_eval, Expr::Integer(0)].into()),
            Expr::List(vec![cx_eval, cy_eval].into()),
          ]
          .into(),
        )]
        .into(),
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
        return Some(Ok(Expr::List(result.into())));
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
            args: vec![Expr::List(vec![h_eval, k_eval].into()), r_eval].into(),
          }));
        }
      }
    }
    "CoordinateBounds" if args.len() == 1 || args.len() == 2 => {
      // CoordinateBounds[coords] returns {{xmin,xmax}, {ymin,ymax}, ...}
      // CoordinateBounds[coords, pad] also pads each dimension. `pad` may be:
      //   - a scalar (number) or Scaled[s] applied uniformly in all dims
      //   - {p1, p2, ...}  (one entry per dim; entries may be scalars,
      //     Scaled[s], or {pmin, pmax} pairs whose elements may be scaled)
      //   - {{p1min, p1max}, ...} pair-per-dim form
      if let Expr::List(points) = &args[0]
        && !points.is_empty()
        && let Expr::List(first) = &points[0]
      {
        let dim = first.len();
        let mut mins: Vec<Expr> = first.to_vec();
        let mut maxs: Vec<Expr> = first.to_vec();
        for pt in &points[1..] {
          if let Expr::List(coords) = pt
            && coords.len() == dim
          {
            for d in 0..dim {
              let less_than_min = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Less".to_string(),
                args: vec![coords[d].clone(), mins[d].clone()].into(),
              });
              if let Ok(Expr::Identifier(ref s)) = less_than_min
                && s == "True"
              {
                mins[d] = coords[d].clone();
              }
              let greater_than_max =
                evaluate_expr_to_expr(&Expr::FunctionCall {
                  name: "Greater".to_string(),
                  args: vec![coords[d].clone(), maxs[d].clone()].into(),
                });
              if let Ok(Expr::Identifier(ref s)) = greater_than_max
                && s == "True"
              {
                maxs[d] = coords[d].clone();
              }
            }
          }
        }
        if args.len() == 1 {
          let bounds: Vec<Expr> = (0..dim)
            .map(|d| Expr::List(vec![mins[d].clone(), maxs[d].clone()].into()))
            .collect();
          return Some(Ok(Expr::List(bounds.into())));
        }

        // Resolve a single pad spec (scalar or Scaled[s]) given the range
        // width to use for the Scaled multiplier.
        fn resolve_pad(spec: &Expr, width: &Expr) -> Expr {
          if let Expr::FunctionCall { name, args } = spec
            && name == "Scaled"
            && args.len() == 1
          {
            return evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![args[0].clone(), width.clone()].into(),
            })
            .unwrap_or_else(|_| Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![args[0].clone(), width.clone()].into(),
            });
          }
          spec.clone()
        }

        // Per-dimension widths: maxs[d] - mins[d]
        let widths: Vec<Expr> = (0..dim)
          .map(|d| {
            evaluate_expr_to_expr(&Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(maxs[d].clone()),
              right: Box::new(mins[d].clone()),
            })
            .unwrap_or_else(|_| Expr::Integer(0))
          })
          .collect();

        // Build per-dimension (pad_min, pad_max).
        let pad_pairs: Option<Vec<(Expr, Expr)>> = match &args[1] {
          Expr::List(items) if items.len() == dim => Some(
            items
              .iter()
              .enumerate()
              .map(|(d, item)| match item {
                Expr::List(pair) if pair.len() == 2 => (
                  resolve_pad(&pair[0], &widths[d]),
                  resolve_pad(&pair[1], &widths[d]),
                ),
                _ => {
                  let p = resolve_pad(item, &widths[d]);
                  (p.clone(), p)
                }
              })
              .collect(),
          ),
          // Scalar Scaled[s] or any non-list spec: apply uniformly per dim.
          spec => Some(
            (0..dim)
              .map(|d| {
                let p = resolve_pad(spec, &widths[d]);
                (p.clone(), p)
              })
              .collect(),
          ),
        };

        if let Some(pads) = pad_pairs {
          let bounds: Vec<Expr> = (0..dim)
            .map(|d| {
              let new_min = evaluate_expr_to_expr(&Expr::BinaryOp {
                op: BinaryOperator::Minus,
                left: Box::new(mins[d].clone()),
                right: Box::new(pads[d].0.clone()),
              })
              .unwrap_or_else(|_| mins[d].clone());
              let new_max = evaluate_expr_to_expr(&Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(maxs[d].clone()),
                right: Box::new(pads[d].1.clone()),
              })
              .unwrap_or_else(|_| maxs[d].clone());
              Expr::List(vec![new_min, new_max].into())
            })
            .collect();
          return Some(Ok(Expr::List(bounds.into())));
        }
      }
    }
    "CoordinateBoundingBox" if args.len() == 1 || args.len() == 2 => {
      // CoordinateBoundingBox[pts, pad...] is the corner form of
      // CoordinateBounds: {{xmin, ymin, ...}, {xmax, ymax, ...}}, i.e.
      // Transpose[CoordinateBounds[...]]. Delegating reuses all of
      // CoordinateBounds' padding handling (scalar / per-dim / pair / Scaled).
      let bounds =
        evaluate_expr_to_expr(&unevaluated("CoordinateBounds", args));
      if let Ok(Expr::List(per_dim)) = &bounds
        && !per_dim.is_empty()
        && per_dim
          .iter()
          .all(|d| matches!(d, Expr::List(p) if p.len() == 2))
      {
        let mins: Vec<Expr> = per_dim
          .iter()
          .map(|d| match d {
            Expr::List(p) => p[0].clone(),
            _ => unreachable!(),
          })
          .collect();
        let maxs: Vec<Expr> = per_dim
          .iter()
          .map(|d| match d {
            Expr::List(p) => p[1].clone(),
            _ => unreachable!(),
          })
          .collect();
        return Some(Ok(Expr::List(
          vec![Expr::List(mins.into()), Expr::List(maxs.into())].into(),
        )));
      }
      return Some(Ok(unevaluated("CoordinateBoundingBox", args)));
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
              op: BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Integer(-1)),
                right: Box::new(bi.clone()),
              }),
            }]
            .into(),
          })
          .collect();
        let max_expr = Expr::FunctionCall {
          name: "Max".to_string(),
          args: diffs.into(),
        };
        return Some(evaluate_expr_to_expr(&max_expr));
      }
      // Scalar fallback: ChessboardDistance[a, b] = Abs[a - b]
      if !matches!(&args[0], Expr::List(_))
        && !matches!(&args[1], Expr::List(_))
      {
        let abs_expr = Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: Box::new(args[1].clone()),
            }),
          }]
          .into(),
        };
        return Some(evaluate_expr_to_expr(&abs_expr));
      }
    }
    "BrayCurtisDistance" if args.len() == 2 => {
      // BrayCurtisDistance[u, v] = Total[Abs[u - v]] / Total[Abs[u + v]]
      // Scalar fallback: Abs[a - b] / Abs[a + b]
      if !matches!(&args[0], Expr::List(_))
        && !matches!(&args[1], Expr::List(_))
      {
        let num = Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: Box::new(args[1].clone()),
            }),
          }]
          .into(),
        };
        let den = Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(args[0].clone()),
            right: Box::new(args[1].clone()),
          }]
          .into(),
        };
        let result = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(num),
          right: Box::new(den),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
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
              op: BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Integer(-1)),
                right: Box::new(bi.clone()),
              }),
            }]
            .into(),
          });
          den_terms.push(Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(bi.clone()),
            }]
            .into(),
          });
        }
        let num = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: num_terms.into(),
        };
        let den = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: den_terms.into(),
        };
        let result = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(num),
          right: Box::new(den),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    "CanberraDistance" if args.len() == 2 => {
      // CanberraDistance[u, v] = Sum[Abs[ui - vi] / (Abs[ui] + Abs[vi])]
      // Scalar fallback: Abs[a - b] / (Abs[a] + Abs[b])
      if !matches!(&args[0], Expr::List(_))
        && !matches!(&args[1], Expr::List(_))
      {
        let num = Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: Box::new(args[1].clone()),
            }),
          }]
          .into(),
        };
        let den = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![args[0].clone()].into(),
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![args[1].clone()].into(),
          }),
        };
        let result = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(num),
          right: Box::new(den),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1])
        && a.len() == b.len()
        && !a.is_empty()
      {
        let mut terms = Vec::new();
        for (ai, bi) in a.iter().zip(b.iter()) {
          let num = Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(ai.clone()),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Integer(-1)),
                right: Box::new(bi.clone()),
              }),
            }]
            .into(),
          };
          let den = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![ai.clone()].into(),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![bi.clone()].into(),
            }),
          };
          terms.push(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(num),
            right: Box::new(den),
          });
        }
        let sum = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: terms.into(),
        };
        return Some(evaluate_expr_to_expr(&sum));
      }
    }
    "CosineDistance" if args.len() == 2 => {
      // CosineDistance[u, v] = 1 - (u . Conjugate[v]) / (Norm[u] * Norm[v])
      // Scalar form: CosineDistance[x, y] = 1 - (x * Conjugate[y]) / (|x| * |y|).
      let is_scalar = |e: &Expr| !matches!(e, Expr::List(_));
      // Treat the scalar form only when at least one argument is a number
      // (Integer/Real/BigInt/BigFloat, possibly combined with I via Plus/Times).
      // Pure-symbolic pairs like CosineDistance[a, b] stay unevaluated
      // (matching wolframscript).
      fn is_zero_scalar(e: &Expr) -> bool {
        matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(f) if *f == 0.0)
      }
      fn has_numeric_atom(e: &Expr) -> bool {
        match e {
          Expr::Integer(_)
          | Expr::Real(_)
          | Expr::BigInteger(_)
          | Expr::BigFloat(_, _) => true,
          Expr::FunctionCall { name, args }
            if name == "Rational" || name == "Complex" =>
          {
            args.iter().any(has_numeric_atom)
          }
          Expr::BinaryOp { left, right, .. } => {
            has_numeric_atom(left) || has_numeric_atom(right)
          }
          Expr::UnaryOp { operand, .. } => has_numeric_atom(operand),
          _ => false,
        }
      }
      if is_scalar(&args[0])
        && is_scalar(&args[1])
        && (has_numeric_atom(&args[0]) || has_numeric_atom(&args[1]))
      {
        if is_zero_scalar(&args[0]) || is_zero_scalar(&args[1]) {
          let any_real = matches!(&args[0], Expr::Real(_))
            || matches!(&args[1], Expr::Real(_));
          return Some(Ok(if any_real {
            Expr::Real(0.0)
          } else {
            Expr::Integer(0)
          }));
        }
        // Factor the division as (u / |u|) * (Conj(v) / |v|) so that common
        // integer factors cancel before they can accumulate in a single
        // numerator (matches wolframscript's canonical output).
        let abs_u = Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![args[0].clone()].into(),
        };
        let abs_v = Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![args[1].clone()].into(),
        };
        let conj_v = Expr::FunctionCall {
          name: "Conjugate".to_string(),
          args: vec![args[1].clone()].into(),
        };
        let u_over_abs_u = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(args[0].clone()),
          right: Box::new(abs_u),
        };
        let conj_v_over_abs_v = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(conj_v),
          right: Box::new(abs_v),
        };
        let ratio = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(u_over_abs_u),
          right: Box::new(conj_v_over_abs_v),
        };
        let result = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(ratio),
          }),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1])
        && a.len() == b.len()
        && !a.is_empty()
      {
        // Special case: if either vector is all-zero, wolframscript returns
        // 0 (or 0. if any entry is Real), bypassing the 0/0 division.
        let is_zero_vec = |items: &[Expr]| -> bool {
          items.iter().all(|e| match e {
            Expr::Integer(0) => true,
            Expr::Real(f) => *f == 0.0,
            _ => false,
          })
        };
        if is_zero_vec(a) || is_zero_vec(b) {
          let any_real =
            a.iter().chain(b.iter()).any(|e| matches!(e, Expr::Real(_)));
          return Some(Ok(if any_real {
            Expr::Real(0.0)
          } else {
            Expr::Integer(0)
          }));
        }
        let conj_b = Expr::FunctionCall {
          name: "Conjugate".to_string(),
          args: vec![args[1].clone()].into(),
        };
        let dot = Expr::FunctionCall {
          name: "Dot".to_string(),
          args: vec![args[0].clone(), conj_b].into(),
        };
        let norm_a = Expr::FunctionCall {
          name: "Norm".to_string(),
          args: vec![args[0].clone()].into(),
        };
        let norm_b = Expr::FunctionCall {
          name: "Norm".to_string(),
          args: vec![args[1].clone()].into(),
        };
        let result = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(dot),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
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
      if let Some(out) = image_min_max_filter(&args[0], &args[1], false) {
        return Some(Ok(out));
      }
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
            args: window.into(),
          })
          .unwrap_or(elems[i].clone());
          result.push(max_val);
        }
        return Some(Ok(Expr::List(result.into())));
      }
      // First arg isn't a list/image — emit wolframscript's arg1 warning
      // and return unevaluated (matches MinFilter/MedianFilter/GaussianFilter).
      if !matches!(&args[0], Expr::Image { .. } | Expr::List(_)) {
        crate::emit_message(&format!(
          "MaxFilter::arg1: The first argument {} should be a rectangular array, image or video.",
          crate::syntax::expr_to_string(&args[0])
        ));
        return Some(Ok(unevaluated("MaxFilter", args)));
      }
    }
    "MinFilter" if args.len() == 2 => {
      if let Some(out) = image_min_max_filter(&args[0], &args[1], true) {
        return Some(Ok(out));
      }
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
            args: window.into(),
          })
          .unwrap_or(elems[i].clone());
          result.push(min_val);
        }
        return Some(Ok(Expr::List(result.into())));
      }
      // First arg isn't a list/image — emit wolframscript's arg1 warning.
      if !matches!(&args[0], Expr::Image { .. } | Expr::List(_)) {
        crate::emit_message(&format!(
          "MinFilter::arg1: The first argument {} should be a rectangular array, image or video.",
          crate::syntax::expr_to_string(&args[0])
        ));
        return Some(Ok(unevaluated("MinFilter", args)));
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
          return Some(Ok(Expr::List(result.into())));
        }
      }
    }
    "Downsample" if args.len() == 2 => {
      // Downsample[list, n] — take every n-th element
      if let (Expr::List(elems), Some(n)) = (&args[0], expr_to_i128(&args[1])) {
        let n = n as usize;
        if n > 0 {
          let result: Vec<Expr> = elems.iter().step_by(n).cloned().collect();
          return Some(Ok(Expr::List(result.into())));
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
            Expr::List(cols) if cols.len() == 3 => Some(cols.to_vec()),
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
                    args: vec![matrix[i][j].clone()].into(),
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
          return Some(Ok(Expr::List(
            vec![to_expr(alpha), to_expr(beta), to_expr(gamma)].into(),
          )));
        }
      }
    }
    // KroneckerSymbol[a, n] — generalized Jacobi symbol
    "KroneckerSymbol" if args.len() == 2 => {
      if let (Some(a), Some(n)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
      {
        let result = crate::functions::kronecker_symbol(a, n);
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
                    args: vec![Expr::Integer(-1), vi.clone()].into(),
                  },
                ]
                .into(),
              },
              Expr::Integer(2),
            ]
            .into(),
          })
          .collect();
        let numerator = Expr::FunctionCall {
          name: "Total".to_string(),
          args: vec![Expr::List(diff_sq.into())].into(),
        };
        // Variance-like terms
        let mean_u = Expr::FunctionCall {
          name: "Mean".to_string(),
          args: vec![Expr::List(u.clone())].into(),
        };
        let mean_v = Expr::FunctionCall {
          name: "Mean".to_string(),
          args: vec![Expr::List(v.clone())].into(),
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
                    args: vec![Expr::Integer(-1), mean_u.clone()].into(),
                  },
                ]
                .into(),
              },
              Expr::Integer(2),
            ]
            .into(),
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
                    args: vec![Expr::Integer(-1), mean_v.clone()].into(),
                  },
                ]
                .into(),
              },
              Expr::Integer(2),
            ]
            .into(),
          })
          .collect();
        let denominator = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Total".to_string(),
              args: vec![Expr::List(var_u.into())].into(),
            },
            Expr::FunctionCall {
              name: "Total".to_string(),
              args: vec![Expr::List(var_v.into())].into(),
            },
          ]
          .into(),
        };
        let result = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
            },
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                numerator,
                Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![denominator, Expr::Integer(-1)].into(),
                },
              ]
              .into(),
            },
          ]
          .into(),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // CorrelationDistance[u, v] — 1 - Correlation[u, v]
    "CorrelationDistance" if args.len() == 2 => {
      // Build 1 - Correlation[u, v] and evaluate
      let corr_expr = Expr::FunctionCall {
        name: "Correlation".to_string(),
        args: vec![args[0].clone(), args[1].clone()].into(),
      };
      let result_expr = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), corr_expr].into(),
          },
        ]
        .into(),
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
          return Some(Ok(Expr::List(result.into())));
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
          return Some(Ok(Expr::List(vec![Expr::Integer(power)].into())));
        }
      }
    }
    // ShearingMatrix[theta, v, n] — shearing transformation matrix
    // ShearingMatrix[theta, {v1,...}, {n1,...}] = I + Tan[theta] * outer(v, n)
    "ShearingMatrix" if args.len() == 3 => {
      if let (Expr::List(v), Expr::List(n_vec)) = (&args[1], &args[2]) {
        let dim = v.len();
        if dim == 0 || dim != n_vec.len() {
          return None;
        }
        // ShearingMatrix[theta, v, n] shears by `theta` along the component of
        // `v` lying in the hyperplane normal to `n`. With n_hat = n/|n| and
        // v_hat the unit vector of v projected into that hyperplane
        // (v_proj = v - (v.n_hat) n_hat), the matrix is
        //   I + Tan[theta] * (v_hat outer n_hat).
        let call = |name: &str, a: Vec<Expr>| Expr::FunctionCall {
          name: name.to_string(),
          args: a.into(),
        };
        let eval = |e: Expr| evaluate_expr_to_expr(&e).unwrap_or(e);

        // n.n and v.n (exact where possible).
        let nn = eval(call(
          "Plus",
          (0..dim)
            .map(|k| call("Times", vec![n_vec[k].clone(), n_vec[k].clone()]))
            .collect(),
        ));
        let vn = eval(call(
          "Plus",
          (0..dim)
            .map(|k| call("Times", vec![v[k].clone(), n_vec[k].clone()]))
            .collect(),
        ));
        // v_proj = v - (v.n / n.n) * n
        let coeff = eval(call("Divide", vec![vn, nn.clone()]));
        let v_proj: Vec<Expr> = (0..dim)
          .map(|i| {
            eval(call(
              "Subtract",
              vec![
                v[i].clone(),
                call("Times", vec![coeff.clone(), n_vec[i].clone()]),
              ],
            ))
          })
          .collect();
        // |v_proj|^2 — a zero projection has no shear direction.
        let vpp = eval(call(
          "Plus",
          (0..dim)
            .map(|k| call("Times", vec![v_proj[k].clone(), v_proj[k].clone()]))
            .collect(),
        ));
        let is_zero = matches!(&vpp, Expr::Integer(0))
          || matches!(&vpp, Expr::Real(r) if *r == 0.0);
        if is_zero {
          crate::emit_message(&format!(
            "ShearingMatrix::proj: The projection of {} onto the plane defined by {} has zero magnitude.",
            crate::syntax::expr_to_string(&args[1]),
            crate::syntax::expr_to_string(&args[2]),
          ));
          return Some(Ok(unevaluated("ShearingMatrix", args)));
        }

        let tan = call("Tan", vec![args[0].clone()]);
        let sqrt_vpp = call("Sqrt", vec![vpp]);
        let sqrt_nn = call("Sqrt", vec![nn]);
        let mut rows = Vec::with_capacity(dim);
        for i in 0..dim {
          let mut row = Vec::with_capacity(dim);
          for j in 0..dim {
            let delta = Expr::Integer(if i == j { 1 } else { 0 });
            // Tan[theta] * v_proj[i]*n[j] / (Sqrt[vpp]*Sqrt[nn])
            let shear = call(
              "Times",
              vec![
                tan.clone(),
                v_proj[i].clone(),
                n_vec[j].clone(),
                call("Power", vec![sqrt_vpp.clone(), Expr::Integer(-1)]),
                call("Power", vec![sqrt_nn.clone(), Expr::Integer(-1)]),
              ],
            );
            // wolframscript reports each entry over a common denominator
            // (e.g. `(2 + Tan[t])/2` rather than `1 + Tan[t]/2`); Together
            // reproduces that form while leaving single-term entries intact.
            row.push(eval(call(
              "Together",
              vec![call("Plus", vec![delta, shear])],
            )));
          }
          rows.push(Expr::List(row.into()));
        }
        return Some(Ok(Expr::List(rows.into())));
      }
    }
    // PrimitiveRootList[n] — list of primitive roots modulo n
    "PrimitiveRootList" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0]) {
        if n <= 1 {
          return Some(Ok(Expr::List(vec![].into())));
        }
        let n = n as u64;
        let phi = euler_totient(n);
        let mut roots = Vec::new();
        for g in 1..n {
          if is_primitive_root(g, n, phi) {
            roots.push(Expr::Integer(g as i128));
          }
        }
        return Some(Ok(Expr::List(roots.into())));
      }
    }
    // DMSList[degrees] — convert decimal degrees to {d, m, s}.
    // DMSList[{d, m, s}] — re-normalise an unnormalised DMS triple.
    "DMSList" if args.len() == 1 => {
      let val =
        evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());

      // List input: combine into decimal degrees first, then re-emit.
      if let Expr::List(items) = &val
        && items.len() == 3
      {
        // If any component is a Real, route through f64 — the result
        // carries the same float precision artifacts wolframscript shows
        // (e.g. `4.999999999997584` for `{11, -30, 5.}`).
        let any_real = items.iter().any(|e| matches!(e, Expr::Real(_)));
        if any_real {
          let d_num = crate::functions::math_ast::try_eval_to_f64(&items[0]);
          let m_num = crate::functions::math_ast::try_eval_to_f64(&items[1]);
          let s_num = crate::functions::math_ast::try_eval_to_f64(&items[2]);
          if let (Some(dv), Some(mv), Some(sv)) = (d_num, m_num, s_num) {
            let total_deg = dv + mv / 60.0 + sv / 3600.0;
            let sign = if total_deg < 0.0 { -1.0 } else { 1.0 };
            let abs_deg = total_deg.abs();
            let d_part = abs_deg.trunc();
            let min_total = (abs_deg - d_part) * 60.0;
            let m_part = min_total.trunc();
            let s_part = (min_total - m_part) * 60.0;
            let d_i = (sign * d_part) as i128;
            let m_i = (sign * m_part) as i128;
            let s_real = sign * s_part;
            return Some(Ok(Expr::List(
              vec![Expr::Integer(d_i), Expr::Integer(m_i), Expr::Real(s_real)]
                .into(),
            )));
          }
        }
        // All-rational input: combine into a rational degree value and
        // fall through to the existing rational-path conversion below.
        let parts: Option<Vec<(i128, i128)>> =
          items.iter().map(expr_to_rational).collect();
        if let Some(p) = parts
          && p.len() == 3
        {
          // total = d + m/60 + s/3600 expressed as a single (num, den).
          // Multiply each by 3600 to share a denominator, then sum.
          let (dn, dd) = p[0];
          let (mn, md) = p[1];
          let (sn, sd) = p[2];
          // num/den = (dn * 3600 * md * sd + mn * 60 * dd * sd + sn * dd * md)
          //         / (dd * md * sd * 3600)
          let num: i128 = dn
            .checked_mul(3600)?
            .checked_mul(md)?
            .checked_mul(sd)?
            .checked_add(mn.checked_mul(60)?.checked_mul(dd)?.checked_mul(sd)?)?
            .checked_add(sn.checked_mul(dd)?.checked_mul(md)?)?;
          let den: i128 =
            dd.checked_mul(md)?.checked_mul(sd)?.checked_mul(3600)?;
          let g = gcd_i128(num.abs(), den.abs()).max(1);
          let (num, den) = (num / g, den / g);
          // Now emit via the same logic as the scalar-rational path below.
          let d = num / den;
          let remainder = num - d * den;
          let min_num = remainder * 60;
          let m = min_num / den;
          let min_rem = min_num - m * den;
          let sec_num = min_rem * 60;
          let s = sec_num / den;
          let sec_rem = sec_num - s * den;
          if sec_rem == 0 {
            return Some(Ok(Expr::List(
              vec![Expr::Integer(d), Expr::Integer(m), Expr::Integer(s)].into(),
            )));
          } else {
            let g2 = gcd_i128(sec_num.abs(), den.abs());
            return Some(Ok(Expr::List(
              vec![
                Expr::Integer(d),
                Expr::Integer(m),
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![
                    Expr::Integer(sec_num / g2),
                    Expr::Integer(den / g2),
                  ]
                  .into(),
                },
              ]
              .into(),
            )));
          }
        }
      }

      if let Some((num, den)) = expr_to_rational(&val) {
        let d = num / den;
        let remainder = num - d * den;
        let min_num = remainder * 60;
        let m = min_num / den;
        let min_rem = min_num - m * den;
        let sec_num = min_rem * 60;
        let s = sec_num / den;
        let sec_rem = sec_num - s * den;
        if sec_rem == 0 {
          return Some(Ok(Expr::List(
            vec![Expr::Integer(d), Expr::Integer(m), Expr::Integer(s)].into(),
          )));
        } else {
          let g = gcd_i128(sec_num.abs(), den.abs());
          return Some(Ok(Expr::List(
            vec![
              Expr::Integer(d),
              Expr::Integer(m),
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(sec_num / g), Expr::Integer(den / g)]
                  .into(),
              },
            ]
            .into(),
          )));
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
      if let Some(result) = pair_sides(&args[0], &args[1], SideOp::Add) {
        return Some(evaluate_expr_to_expr(&result));
      }
      if let Some(result) = apply_to_sides(&args[0], &args[1], "Plus") {
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // SubtractSides[rel] — subtract the right-hand side from both sides,
    // giving `lhs - rhs OP 0` (valid for any relation).
    "SubtractSides" if args.len() == 1 => {
      if let Expr::Comparison { operands, .. } = &args[0]
        && operands.len() == 2
      {
        let neg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), operands[1].clone()].into(),
        };
        if let Some(result) = apply_to_sides(&args[0], &neg, "Plus") {
          return Some(evaluate_expr_to_expr(&result));
        }
      }
    }
    // DivideSides[rel] — divide both sides by the right-hand side, giving
    // `lhs/rhs == 1` (guarded by rhs != 0 when rhs may be zero). Restricted to
    // equations; dividing an inequality needs sign-dependent reasoning.
    "DivideSides" if args.len() == 1 => {
      if let Expr::Comparison {
        operands,
        operators,
      } = &args[0]
        && operands.len() == 2
        && operators.first() == Some(&ComparisonOp::Equal)
        && !is_zero_literal(&operands[1])
      {
        let rhs = operands[1].clone();
        let inv = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![rhs.clone(), Expr::Integer(-1)].into(),
        };
        if let Some(divided) = apply_to_sides(&args[0], &inv, "Times")
          && let Ok(divided) = evaluate_expr_to_expr(&divided)
        {
          if is_nonzero_number(&rhs) {
            return Some(Ok(divided));
          }
          let cond = Expr::Comparison {
            operands: vec![rhs, Expr::Integer(0)],
            operators: vec![ComparisonOp::NotEqual],
          };
          let branch = Expr::List(vec![divided, cond].into());
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![Expr::List(vec![branch].into()), args[0].clone()].into(),
          };
          return Some(evaluate_expr_to_expr(&pw));
        }
      }
    }
    // SubtractSides[rel, expr] — subtract expr from both sides
    "SubtractSides" if args.len() == 2 => {
      if let Some(result) = pair_sides(&args[0], &args[1], SideOp::Subtract) {
        return Some(evaluate_expr_to_expr(&result));
      }
      let neg = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), args[1].clone()].into(),
      };
      if let Some(result) = apply_to_sides(&args[0], &neg, "Plus") {
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // MultiplySides[rel, expr] — multiply both sides by expr
    "MultiplySides" if args.len() == 2 => {
      if let Some(result) = pair_sides(&args[0], &args[1], SideOp::Multiply) {
        return Some(evaluate_expr_to_expr(&result));
      }
      // Multiplying an equation by 0 collapses it; Wolfram errors, so keep it
      // unevaluated rather than emitting 0 == 0.
      if is_zero_literal(&args[1]) {
        return None;
      }
      if let Some(result) = apply_to_sides(&args[0], &args[1], "Times")
        && let Ok(scaled) = evaluate_expr_to_expr(&result)
      {
        return Some(guard_equation_scale(&args[0], &args[1], scaled));
      }
    }
    // DivideSides[rel, expr] — divide both sides by expr
    "DivideSides" if args.len() == 2 => {
      if let Some(result) = pair_sides(&args[0], &args[1], SideOp::Divide) {
        return Some(evaluate_expr_to_expr(&result));
      }
      if is_zero_literal(&args[1]) {
        return None;
      }
      let inv = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![args[1].clone(), Expr::Integer(-1)].into(),
      };
      if let Some(result) = apply_to_sides(&args[0], &inv, "Times")
        && let Ok(scaled) = evaluate_expr_to_expr(&result)
      {
        return Some(guard_equation_scale(&args[0], &args[1], scaled));
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
            args: vec![op.clone()].into(),
          })
          .collect();
        let result = Expr::Comparison {
          operands: new_operands,
          operators: operators.clone(),
        };
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    "DMSString" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::geo_math::dms_string_ast(args));
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
                    args: vec![Expr::Integer(1), Expr::Integer(60)].into(),
                  },
                ]
                .into(),
              },
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  s,
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(1), Expr::Integer(3600)].into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
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
    // Unconstrained form only. The constrained `{f, cons}` list form is
    // handled by dispatch_polynomial_functions (single-var sweep); skipping it
    // here avoids feeding the whole list to FindMinimum as the objective.
    "NArgMin"
      if args.len() == 2
        && !matches!(&args[0], Expr::List(items) if items.len() == 2) =>
    {
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(
            vec![Expr::Identifier(var.clone()), Expr::Integer(0)].into(),
          ),
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
                  args: vec![replacement.as_ref().clone()].into(),
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
          return Some(Ok(Expr::List(args_list.into())));
        }
      }
    }
    // NArgMax[f, x] — numerical arg max (unconstrained form only; the
    // constrained `{f, cons}` list form is handled in polynomial_functions).
    "NArgMax"
      if args.len() == 2
        && !matches!(&args[0], Expr::List(items) if items.len() == 2) =>
    {
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(
            vec![Expr::Identifier(var.clone()), Expr::Integer(0)].into(),
          ),
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
                  args: vec![replacement.as_ref().clone()].into(),
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
          return Some(Ok(Expr::List(args_list.into())));
        }
      }
    }
    // ArrayResample[list, n] — resample a 1D array to n elements using linear interpolation
    "ArrayResample" if args.len() == 2 => {
      if let (Expr::List(elems), Some(n)) = (&args[0], expr_to_i128(&args[1])) {
        let n = n as usize;
        let m = elems.len();
        if n == 0 {
          return Some(Ok(Expr::List(vec![].into())));
        }
        if m == 0 {
          return Some(Ok(Expr::List(vec![].into())));
        }
        if n == 1 {
          return Some(Ok(Expr::List(vec![elems[0].clone()].into())));
        }
        if m == 1 {
          return Some(Ok(Expr::List(vec![elems[0].clone(); n].into())));
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
              args: vec![Expr::Integer(rem), Expr::Integer(t_den)].into(),
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
                          args: vec![Expr::Integer(-1), elems[idx].clone()]
                            .into(),
                        },
                      ]
                      .into(),
                    },
                  ]
                  .into(),
                },
              ]
              .into(),
            };
            result.push(evaluate_expr_to_expr(&interp).unwrap_or(interp));
          }
        }
        return Some(Ok(Expr::List(result.into())));
      }
    }
    // PolynomialLCM[p1, p2, ...] — least common multiple of polynomials
    "PolynomialLCM" if args.len() >= 2 => {
      return Some(crate::functions::polynomial_ast::polynomial_lcm_ast(args));
    }
    // CoordinateBoundsArray[{{xmin,xmax},…}, spec, offsets] — grid of
    // coordinate tuples; spec is a step, per-dimension steps, or Into[n].
    "CoordinateBoundsArray" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::math_ast::coordinate_bounds_array_ast(
        args,
      ));
    }
    // CoordinateBoundingBoxArray[{mins, maxs}, spec, offsets] — the same
    // grid written with the two corner points.
    "CoordinateBoundingBoxArray" if !args.is_empty() && args.len() <= 3 => {
      return Some(
        crate::functions::math_ast::coordinate_bounding_box_array_ast(args),
      );
    }
    // DiracComb[x…] — 0 when any argument is a definite non-integer (a
    // Rational, or a Real with a fractional part); everything else —
    // integers, integer-valued reals, symbols, even Pi — stays unevaluated,
    // exactly like wolframscript. A single list argument threads.
    "DiracComb" if !args.is_empty() => {
      fn definite_non_integer(e: &Expr) -> bool {
        match e {
          Expr::Real(v) => v.is_finite() && v.fract() != 0.0,
          Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2 =>
          {
            // Canonical rationals are already non-integer.
            true
          }
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => definite_non_integer(operand),
          _ => false,
        }
      }
      if args.len() == 1
        && let Expr::List(items) = &args[0]
      {
        let threaded: Vec<Expr> = items
          .iter()
          .map(|item| {
            if definite_non_integer(item) {
              Expr::Integer(0)
            } else {
              Expr::FunctionCall {
                name: "DiracComb".to_string(),
                args: vec![item.clone()].into(),
              }
            }
          })
          .collect();
        return Some(Ok(Expr::List(threaded.into())));
      }
      if args.iter().any(definite_non_integer) {
        return Some(Ok(Expr::Integer(0)));
      }
      return Some(Ok(unevaluated("DiracComb", args)));
    }
    // ChampernowneNumber[] / ChampernowneNumber[b] — a numeric constant
    // that stays symbolic until numericized with N. Non-integer or < 2
    // numeric bases emit ::ibase; symbolic bases echo silently.
    "ChampernowneNumber" => {
      if args.len() <= 1
        && let Some(first) = args.first()
      {
        let valid = matches!(first, Expr::Integer(b) if *b >= 2);
        if !valid
          && crate::functions::math_ast::try_eval_to_f64(first).is_some()
        {
          crate::emit_message(&format!(
            "ChampernowneNumber::ibase: Base {} is not an integer greater than 1.",
            crate::syntax::expr_to_output(first)
          ));
        }
      }
      return Some(Ok(unevaluated("ChampernowneNumber", args)));
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
    "QGamma" if args.len() == 2 => {
      return Some(qgamma_ast(&args[0], &args[1]));
    }
    _ => {}
  }
  None
}

/// The value `1` for a q-series (QGamma/QFactorial) that collapses to unity,
/// carried at the precision of `q`: an inexact (machine-real) `q` yields the
/// machine real `1.`, matching wolframscript's numeric contagion, whereas an
/// exact `q` (integer/rational/symbolic) yields the exact integer `1`.
fn q_one(q_expr: &Expr) -> Expr {
  if matches!(q_expr, Expr::Real(_)) {
    Expr::Real(1.0)
  } else {
    Expr::Integer(1)
  }
}

/// QGamma[z, q] — the q-gamma function. For a positive integer first argument
/// `n`, `QGamma[n, q] = QFactorial[n-1, q] = ∏_{i=1}^{n-1} [i]_q`. Non-positive
/// integers are poles (ComplexInfinity). With a numeric `q` the exact value is
/// returned; with a symbolic `q`, wolframscript only expands the product for
/// `n <= 3` (e.g. `QGamma[3, q]` → `1 + q`) and otherwise stays unevaluated.
/// Non-integer first arguments are left unevaluated.
fn qgamma_ast(z_expr: &Expr, q_expr: &Expr) -> Result<Expr, InterpreterError> {
  use crate::functions::math_ast::{expr_to_f64, expr_to_i128};

  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "QGamma".to_string(),
      args: vec![z_expr.clone(), q_expr.clone()].into(),
    })
  };

  let n = match expr_to_i128(z_expr) {
    Some(n) => n,
    None => return unevaluated(),
  };
  if n <= 0 {
    // Poles at the non-positive integers.
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if n == 1 {
    // QGamma[1, q] = 1, but an inexact (machine-real) q yields an inexact 1.
    return Ok(q_one(q_expr));
  }

  // Numeric q: reuse the exact q-factorial of n-1.
  let q_is_numeric =
    expr_to_f64(q_expr).is_some() || expr_to_rational(q_expr).is_some();
  if q_is_numeric {
    return qfactorial_ast(&Expr::Integer(n - 1), q_expr);
  }

  // Symbolic q: match wolframscript, which only expands the product for n <= 3.
  if n > 3 {
    return unevaluated();
  }
  // Build ∏_{i=1}^{n-1} (1 + q + … + q^(i-1)) and evaluate it.
  let mut factors: Vec<Expr> = Vec::with_capacity((n - 1) as usize);
  for i in 1..n {
    let terms: Vec<Expr> = (0..i)
      .map(|j| match j {
        0 => Expr::Integer(1),
        1 => q_expr.clone(),
        _ => Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![q_expr.clone(), Expr::Integer(j)].into(),
        },
      })
      .collect();
    factors.push(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    });
  }
  let product = Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&product)
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
  if let Some((num, den)) = expr_to_rational(arg) {
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
    args: vec![arg.clone()].into(),
  })
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
            args: vec![arg.clone()].into(),
          });
        }
      } else {
        return Ok(Expr::FunctionCall {
          name: "Midpoint".to_string(),
          args: vec![arg.clone()].into(),
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Midpoint".to_string(),
        args: vec![arg.clone()].into(),
      });
    }
  };

  let p1 = &points[0];
  let p2 = &points[1];

  // (p1 + p2) / 2 - works for both scalar and vector points
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![p1.clone(), p2.clone()].into(),
  };
  let result = Expr::BinaryOp {
    op: BinaryOperator::Divide,
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
        args: vec![n_expr.clone(), q_expr.clone()].into(),
      });
    }
  };

  if n == 0 || n == 1 {
    // [0]_q! = [1]_q! = 1, but an inexact (machine-real) q yields an inexact 1.
    return Ok(q_one(q_expr));
  }

  // When q is symbolic (not a numeric value), match wolframscript and keep
  // `QFactorial[n, q]` unevaluated rather than expanding to a rational
  // form. The rational form blows up combinatorially for Series and Plot.
  if crate::functions::math_ast::expr_to_f64(q_expr).is_none()
    && expr_to_rational(q_expr).is_none()
  {
    return Ok(Expr::FunctionCall {
      name: "QFactorial".to_string(),
      args: vec![n_expr.clone(), q_expr.clone()].into(),
    });
  }

  // Compute product of [k]_q for k = 1 to n
  let mut factors = Vec::new();
  for k in 1..=n {
    // [k]_q = (1 - q^k) / (1 - q)
    let q_k = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(q_expr.clone()),
      right: Box::new(Expr::Integer(k as i128)),
    };
    let numerator = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(q_k),
    };
    let denominator = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(q_expr.clone()),
    };
    let factor = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(denominator),
    };
    factors.push(factor);
  }

  let product = Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  };

  crate::evaluator::evaluate_expr_to_expr(&product)
}

/// Find minimum linear recurrence coefficients {c1, c2, ..., cd} such that
/// a[n] = c1*a[n-1] + c2*a[n-2] + ... + cd*a[n-d] for all valid n.
fn find_linear_recurrence_impl(seq: &[Expr]) -> Result<Expr, InterpreterError> {
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
          args: vec![Expr::List(seq.to_vec().into())].into(),
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
        return Ok(Expr::List(result.into()));
      }
    }
  }

  // No recurrence found
  Ok(Expr::FunctionCall {
    name: "FindLinearRecurrence".to_string(),
    args: vec![Expr::List(seq.to_vec().into())].into(),
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
      args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
    }
  }
}

/// Substitute each variable named in `vars` with `Re[v] + I*Im[v]` so that
/// the subsequent ComplexExpand pass treats it as complex-valued.
/// True iff `expr` contains any identifier in `vars` anywhere.
fn contains_any_var(expr: &Expr, vars: &[String]) -> bool {
  match expr {
    Expr::Identifier(n) => vars.iter().any(|v| v == n),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| contains_any_var(a, vars))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_any_var(left, vars) || contains_any_var(right, vars)
    }
    Expr::UnaryOp { operand, .. } => contains_any_var(operand, vars),
    Expr::List(items) => items.iter().any(|a| contains_any_var(a, vars)),
    _ => false,
  }
}

fn substitute_complex_vars(expr: &Expr, vars: &[String]) -> Expr {
  match expr {
    Expr::Identifier(name) if vars.iter().any(|v| v == name) => {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::FunctionCall {
          name: "Re".to_string(),
          args: vec![Expr::Identifier(name.clone())].into(),
        }),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Identifier("I".to_string())),
          right: Box::new(Expr::FunctionCall {
            name: "Im".to_string(),
            args: vec![Expr::Identifier(name.clone())].into(),
          }),
        }),
      }
    }
    // `Re[z]`, `Im[z]`, `Arg[z]` for a complex-vars `z` are
    // primitives Wolfram emits as-is. `Arg[anything-with-z]` is also
    // left alone — Wolfram has no closed-form expansion for `Arg`.
    // For non-bare `Re`/`Im` arguments (e.g. `Re[2 z]`, `Re[z^2]`)
    // fall through to the usual substitution so the linear /
    // polynomial form gets expanded.
    Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "Re" | "Im")
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(n) if vars.iter().any(|v| v == n)) =>
    {
      expr.clone()
    }
    Expr::FunctionCall { name, args }
      if name == "Arg"
        && args.len() == 1
        && contains_any_var(&args[0], vars) =>
    {
      expr.clone()
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_complex_vars(a, vars))
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_complex_vars(left, vars)),
      right: Box::new(substitute_complex_vars(right, vars)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_complex_vars(operand, vars)),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| substitute_complex_vars(a, vars))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// ComplexExpand[expr] — expand complex-valued functions assuming all
/// variables are real. E.g. Sin[x + I*y] → Sin[x]*Cosh[y] + I*Cos[x]*Sinh[y].
fn complex_expand_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  // Re-evaluate so arithmetic left by the generic Plus/Times recursion folds
  // (e.g. the `-0` from Re[a + b I] = -Im[b] + Re[a] → -0 + a).
  let folded = ce_simplify(complex_expand_recursive(expr));
  // Distribute products and integer powers of sums, matching wolframscript:
  // ComplexExpand[(x+1)^2] = 1 + 2 x + x^2, and hence
  // ComplexExpand[Abs[x+1]^2] = 1 + 2 x + x^2 (via Abs[x+1] = Sqrt[(x+1)^2]).
  Ok(
    crate::evaluator::evaluate_function_call_ast("Expand", &[folded.clone()])
      .unwrap_or(folded),
  )
}

/// Like `complex_expand_ast` but additionally distributes products via
/// Expand, used by the 2-arg form `ComplexExpand[expr, vars]` so the
/// polynomial result is a single distributed Plus chain.
fn complex_expand_with_expand(expr: &Expr) -> Result<Expr, InterpreterError> {
  let expanded = complex_expand_recursive(expr);
  let distributed =
    crate::evaluator::evaluate_function_call_ast("Expand", &[expanded.clone()])
      .unwrap_or(expanded);
  // Distribute Log over positive multipliers and `Sqrt`:
  //   Log[positive_real * X] → Log[positive_real] + Log[X]
  //   Log[Sqrt[Y]] → Log[Y]/2
  // Matches wolframscript's `ComplexExpand[Abs[positive_const * z], …]`
  // shape (`Log[2] + Log[Re[z]^2 + Im[z]^2]/2`).
  let log_split = expand_log_in_complex_expand(&distributed);
  // Group terms by I-factor: re-emit as `<real> + I*<imag>` so the
  // imaginary contributions appear under a single `I*(…)` umbrella,
  // matching wolframscript's `ComplexExpand[…, vars]` shape.
  Ok(group_imag_terms(&log_split))
}

/// True when `e` is the rational `1/2` in any of the shapes our parser
/// or evaluator produces (`Rational[1, 2]`, `Divide(1, 2)`, `Real(0.5)`).
fn is_one_half(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!((&args[0], &args[1]), (Expr::Integer(1), Expr::Integer(2)))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      matches!(
        (left.as_ref(), right.as_ref()),
        (Expr::Integer(1), Expr::Integer(2))
      )
    }
    Expr::Real(v) => (*v - 0.5).abs() < 1e-15,
    _ => false,
  }
}

/// Walks the expanded ComplexExpand result and applies two Log
/// rewrites Wolfram emits in this context:
///   Log[Times[positive_const, …]] → Log[positive_const] + Log[Times[…]]
///   Log[Sqrt[X]] → Log[X]/2
fn expand_log_in_complex_expand(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      let inner = expand_log_in_complex_expand(&args[0]);
      // Log[Sqrt[X]] → Log[X] / 2 (also catches the canonical
      // `Power[X, Rational[1, 2]]` shape Wolfram emits internally).
      let sqrt_arg: Option<&Expr> = match &inner {
        Expr::FunctionCall {
          name: sn,
          args: sargs,
        } if sn == "Sqrt" && sargs.len() == 1 => Some(&sargs[0]),
        Expr::FunctionCall {
          name: pn,
          args: pargs,
        } if pn == "Power" && pargs.len() == 2 && is_one_half(&pargs[1]) => {
          Some(&pargs[0])
        }
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } if is_one_half(right) => Some(left.as_ref()),
        _ => None,
      };
      if let Some(x) = sqrt_arg {
        let log_x = Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![x.clone()].into(),
        };
        return ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(log_x),
          right: Box::new(Expr::Integer(2)),
        });
      }
      // Log[Times[positive_const, …]] → Log[positive_const] + Log[Times[…]]
      if let Expr::FunctionCall {
        name: tn,
        args: tfactors,
      } = &inner
        && tn == "Times"
      {
        let mut pos_const_factors: Vec<Expr> = Vec::new();
        let mut other_factors: Vec<Expr> = Vec::new();
        for f in tfactors {
          let is_positive_const = match f {
            Expr::Integer(n) => *n > 0,
            Expr::Real(v) => *v > 0.0,
            Expr::FunctionCall { name: rn, args: ra }
              if rn == "Rational" && ra.len() == 2 =>
            {
              matches!(&ra[0], Expr::Integer(n) if *n > 0)
                && matches!(&ra[1], Expr::Integer(d) if *d > 0)
            }
            _ => false,
          };
          if is_positive_const {
            pos_const_factors.push(f.clone());
          } else {
            other_factors.push(f.clone());
          }
        }
        if !pos_const_factors.is_empty() && !other_factors.is_empty() {
          let pos_const = if pos_const_factors.len() == 1 {
            pos_const_factors.remove(0)
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: pos_const_factors.into(),
            }
          };
          let rest = if other_factors.len() == 1 {
            other_factors.remove(0)
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: other_factors.into(),
            }
          };
          let log_pos = Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![pos_const].into(),
          };
          let log_rest = expand_log_in_complex_expand(&Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![rest].into(),
          });
          return ce_simplify(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(log_pos),
            right: Box::new(log_rest),
          });
        }
      }
      Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![inner].into(),
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(expand_log_in_complex_expand).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(expand_log_in_complex_expand(left)),
      right: Box::new(expand_log_in_complex_expand(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(expand_log_in_complex_expand(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(expand_log_in_complex_expand).collect())
    }
    _ => expr.clone(),
  }
}

/// If `expr` is a Plus chain whose terms each have at most one explicit
/// `I` (or `Complex[0,k]`) factor, regroup as `real_sum + I*imag_sum`.
/// Terms without an `I` factor are taken as fully real; terms with one
/// have that factor stripped and the remainder added to the imag bucket.
fn group_imag_terms(expr: &Expr) -> Expr {
  let terms: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![*left.clone(), *right.clone()],
    _ => return expr.clone(),
  };
  let mut real_parts: Vec<Expr> = Vec::new();
  let mut imag_parts: Vec<Expr> = Vec::new();
  let mut first_imag_pos: Option<usize> = None;
  for t in &terms {
    if !contains_explicit_i(t) {
      real_parts.push(t.clone());
      continue;
    }
    if let Some(stripped) = strip_one_i_factor(t) {
      if first_imag_pos.is_none() {
        first_imag_pos = Some(real_parts.len());
      }
      imag_parts.push(stripped);
    } else {
      // Term has multiple I factors (or I in a non-multiplicative
      // position); we can't safely group it. Bail.
      return expr.clone();
    }
  }
  let imag = match imag_parts.len() {
    0 => Expr::Integer(0),
    1 => imag_parts.remove(0),
    _ => crate::evaluator::evaluate_function_call_ast("Plus", &imag_parts)
      .unwrap_or(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: imag_parts.into(),
      }),
  };
  if is_expr_zero(&imag) {
    return match real_parts.len() {
      0 => Expr::Integer(0),
      1 => real_parts.remove(0),
      _ => crate::evaluator::evaluate_function_call_ast("Plus", &real_parts)
        .unwrap_or(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: real_parts.into(),
        }),
    };
  }
  let i_term = if matches!(imag, Expr::Integer(1)) {
    Expr::Identifier("I".to_string())
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Identifier("I".to_string()), imag].into(),
    }
  };
  if real_parts.is_empty() {
    return crate::evaluator::evaluate_expr_to_expr(&i_term).unwrap_or(i_term);
  }
  // Insert the `I*<imag>` umbrella at the position the first imag
  // term occupied in the original Plus, so the umbrella lands in the
  // same slot wolframscript prints. Skip the final evaluate pass so
  // the canonical Plus sort doesn't bury the umbrella back at the
  // start.
  let mut combined_args: Vec<Expr> = Vec::with_capacity(real_parts.len() + 1);
  let insert_at = first_imag_pos
    .unwrap_or(real_parts.len())
    .min(real_parts.len());
  combined_args.extend(real_parts.drain(..insert_at));
  combined_args.push(i_term);
  combined_args.extend(real_parts);
  if combined_args.len() == 1 {
    return combined_args.remove(0);
  }
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: combined_args.into(),
  }
}

fn is_expr_zero(e: &Expr) -> bool {
  matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(v) if *v == 0.0)
}

/// Strip a single multiplicative `I` factor (or fold a `Complex[0, k]`
/// into `k`) from a term, returning the remaining "real" coefficient.
/// Returns None when more than one `I` factor is present, or the `I`
/// is buried inside a head like `Sin[I*x]` where it can't be peeled
/// off as a top-level multiplier.
fn strip_one_i_factor(t: &Expr) -> Option<Expr> {
  fn collect_factors(e: &Expr) -> Vec<Expr> {
    match e {
      Expr::FunctionCall { name, args } if name == "Times" => {
        args.iter().flat_map(collect_factors).collect()
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        let mut v = collect_factors(left);
        v.extend(collect_factors(right));
        v
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        // `-X` is `(-1) * X` — push the sign into the factor list so
        // strip_one_i_factor can drop the I and keep the leading −1.
        let mut v = vec![Expr::Integer(-1)];
        v.extend(collect_factors(operand));
        v
      }
      _ => vec![e.clone()],
    }
  }
  let factors = collect_factors(t);
  let mut i_seen = false;
  let mut new_factors: Vec<Expr> = Vec::new();
  for f in factors {
    let is_i = matches!(&f, Expr::Identifier(s) if s == "I");
    let is_complex_imag = if let Expr::FunctionCall { name, args } = &f
      && name == "Complex"
      && args.len() == 2
      && is_expr_zero(&args[0])
      && !is_expr_zero(&args[1])
    {
      true
    } else {
      false
    };
    if is_i || is_complex_imag {
      if i_seen {
        return None;
      }
      i_seen = true;
      // Complex[0, k] contributes its imaginary scalar `k` to the rest.
      if is_complex_imag
        && let Expr::FunctionCall { args: cargs, .. } = &f
        && cargs.len() == 2
        && !matches!(&cargs[1], Expr::Integer(1))
      {
        new_factors.push(cargs[1].clone());
      }
    } else if contains_explicit_i(&f) {
      // Nested `I` inside a non-Times head means we can't cleanly peel
      // it off, e.g. `Sin[I*x]`.
      return None;
    } else {
      new_factors.push(f);
    }
  }
  if !i_seen {
    return None;
  }
  Some(match new_factors.len() {
    0 => Expr::Integer(1),
    1 => new_factors.remove(0),
    _ => crate::evaluator::evaluate_function_call_ast("Times", &new_factors)
      .unwrap_or(Expr::FunctionCall {
        name: "Times".to_string(),
        args: new_factors.into(),
      }),
  })
}

/// Recursively checks whether an expression contains an explicit `I` or
/// `Complex[0, k]` factor anywhere — used to decide whether
/// `group_imag_terms` should attempt a real/imag split on this term or
/// classify it as already fully real.
fn contains_explicit_i(e: &Expr) -> bool {
  match e {
    Expr::Identifier(s) => s == "I",
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      // Complex[a, 0] is real; Complex[a, b] with b≠0 carries an
      // imaginary component.
      !is_expr_zero(&args[1])
    }
    Expr::FunctionCall { args, .. } => args.iter().any(contains_explicit_i),
    Expr::BinaryOp { left, right, .. } => {
      contains_explicit_i(left) || contains_explicit_i(right)
    }
    Expr::UnaryOp { operand, .. } => contains_explicit_i(operand),
    Expr::List(items) => items.iter().any(contains_explicit_i),
    _ => false,
  }
}

fn ce_simplify(e: Expr) -> Expr {
  crate::evaluator::evaluate_expr_to_expr(&e)
    .unwrap_or_else(|_| crate::functions::simplify(e))
}

/// Split an expression into real and imaginary parts assuming all symbols are real.
/// Returns (real_part, imag_part) such that expr = real + I*imag.
fn split_real_imag(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    // a + I*b
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let (lr, li) = split_real_imag(left)?;
      let (rr, ri) = split_real_imag(right)?;
      Some((
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(lr),
          right: Box::new(rr),
        }),
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(li),
          right: Box::new(ri),
        }),
      ))
    }
    // a - b
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let (lr, li) = split_real_imag(left)?;
      let (rr, ri) = split_real_imag(right)?;
      Some((
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(lr),
          right: Box::new(rr),
        }),
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(li),
          right: Box::new(ri),
        }),
      ))
    }
    // c * expr — split each side and combine via complex multiplication.
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (lr, li) = split_real_imag(left)?;
      let (rr, ri) = split_real_imag(right)?;
      // (lr + li i) * (rr + ri i) = (lr*rr - li*ri) + (lr*ri + li*rr) i
      Some((
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(lr.clone()),
            right: Box::new(rr.clone()),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(li.clone()),
            right: Box::new(ri.clone()),
          }),
        }),
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(lr),
            right: Box::new(ri),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(li),
            right: Box::new(rr),
          }),
        }),
      ))
    }
    // I itself
    Expr::Identifier(s) if s == "I" => {
      Some((Expr::Integer(0), Expr::Integer(1)))
    }
    // FunctionCall Times[...] — fold each arg as a complex split and combine.
    Expr::FunctionCall { name, args } if name == "Times" => {
      if args.is_empty() {
        return None;
      }
      let (mut acc_re, mut acc_im) = split_real_imag(&args[0])?;
      for arg in &args[1..] {
        let (rr, ri) = split_real_imag(arg)?;
        // (acc_re + acc_im*I) * (rr + ri*I)
        let new_re = ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(acc_re.clone()),
            right: Box::new(rr.clone()),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(acc_im.clone()),
            right: Box::new(ri.clone()),
          }),
        });
        let new_im = ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(acc_re.clone()),
            right: Box::new(ri),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(acc_im),
            right: Box::new(rr),
          }),
        });
        acc_re = new_re;
        acc_im = new_im;
      }
      Some((acc_re, acc_im))
    }
    // FunctionCall Plus[...] with I terms
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut real_parts = Vec::new();
      let mut imag_parts = Vec::new();
      for arg in args {
        if let Some((r, i)) = split_real_imag(arg) {
          real_parts.push(r);
          imag_parts.push(i);
        } else {
          real_parts.push(arg.clone());
          imag_parts.push(Expr::Integer(0));
        }
      }
      let real = if real_parts.len() == 1 {
        real_parts.remove(0)
      } else {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: real_parts.into(),
        }
      };
      let imag = if imag_parts.len() == 1 {
        imag_parts.remove(0)
      } else {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: imag_parts.into(),
        }
      };
      Some((ce_simplify(real), ce_simplify(imag)))
    }
    // Power[base, n] where n is a non-negative integer and base splits
    // into a + b*I. Expand via the binomial theorem so the polynomial
    // case (a + b*I)^n contributes to both real and imag parts.
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => power_split_real_imag(left, right),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      power_split_real_imag(&args[0], &args[1])
    }
    // Abs[v] is real-valued, but under ComplexExpand it rewrites to
    // Sqrt[Re[v]^2 + Im[v]^2] (e.g. Sqrt[x^2] for real x). Carry that form
    // so nested uses like Re[Abs[x]^2] reduce to x^2 rather than Abs[x]^2.
    Expr::FunctionCall { name, .. } if name == "Abs" => {
      Some((complex_expand_recursive(expr), Expr::Integer(0)))
    }
    // Re[v] / Im[v] / Arg[v]: treat as real.
    Expr::FunctionCall { name, .. }
      if matches!(name.as_str(), "Re" | "Im" | "Arg") =>
    {
      Some((expr.clone(), Expr::Integer(0)))
    }
    // -(a + b*I) = -a - b*I
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (r, i) = split_real_imag(operand)?;
      let neg = |e: Expr| {
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(e),
        })
      };
      Some((neg(r), neg(i)))
    }
    // (a + b*I) / (c + d*I). For a real denominator this is just
    // (a/c) + (b/c)*I; a complex denominator uses (n * Conjugate[d])/|d|^2.
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let (nr, ni) = split_real_imag(left)?;
      let (dr, di) = split_real_imag(right)?;
      let div = |a: Expr, b: Expr| {
        ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(a),
          right: Box::new(b),
        })
      };
      if is_expr_zero(&di) {
        Some((div(nr, dr.clone()), div(ni, dr)))
      } else {
        // denom = dr^2 + di^2; result = (n * conj(d)) / denom
        let sq = |e: Expr| Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(e),
          right: Box::new(Expr::Integer(2)),
        };
        let denom = ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(sq(dr.clone())),
          right: Box::new(sq(di.clone())),
        });
        let mul = |a: Expr, b: Expr| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(a),
          right: Box::new(b),
        };
        // (nr + ni i)(dr - di i) = (nr*dr + ni*di) + (ni*dr - nr*di) i
        let re_num = ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(mul(nr.clone(), dr.clone())),
          right: Box::new(mul(ni.clone(), di.clone())),
        });
        let im_num = ce_simplify(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(mul(ni, dr)),
          right: Box::new(mul(nr, di)),
        });
        Some((div(re_num, denom.clone()), div(im_num, denom)))
      }
    }
    // A Rational is a real-valued atom (e.g. the 3/2 in `(3*I)/2`).
    Expr::FunctionCall { name, .. } if name == "Rational" => {
      Some((expr.clone(), Expr::Integer(0)))
    }
    // Numeric atoms and plain symbols: treat as real-valued.
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(_, _)
    | Expr::Identifier(_)
    | Expr::Constant(_) => Some((expr.clone(), Expr::Integer(0))),
    _ => None,
  }
}

/// Decompose `base^exp` into real and imaginary parts when base = a + b*I and
/// exp is a non-negative integer. Returns None otherwise.
fn power_split_real_imag(base: &Expr, exp: &Expr) -> Option<(Expr, Expr)> {
  // First, if `base` is real and `exp` is real (split returns
  // imag-zero for both), the entire `base^exp` is real and we can
  // return it as-is. This catches `Power[real, -1]` (denominator
  // factors) so split_real_imag flows through Plus/Times rather than
  // bailing.
  if let Some((b_re, b_im)) = split_real_imag(base)
    && matches!(b_im, Expr::Integer(0))
    && let Some((_, e_im)) = split_real_imag(exp)
    && matches!(e_im, Expr::Integer(0))
  {
    // Use the rewritten real part of the base, not the original: under
    // ComplexExpand Abs[x] becomes Sqrt[x^2], so Re[Abs[x]^2] = x^2.
    return Some((
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b_re, exp.clone()].into(),
      },
      Expr::Integer(0),
    ));
  }
  // Positive real base with complex exponent:
  //   b^(a + I*c) = b^a · (Cos[c·Log[b]] + I·Sin[c·Log[b]]).
  // For E (Log[E] = 1) the inner argument collapses to c.
  if is_complex_expand_real_base(base)
    && let Some((re_e, im_e)) = split_real_imag(exp)
    && !is_expr_zero(&im_e)
  {
    let exp_a = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base.clone()),
      right: Box::new(re_e),
    };
    let inner_arg = if matches!(base, Expr::Identifier(s) | Expr::Constant(s) if s == "E")
    {
      im_e.clone()
    } else {
      ce_simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(im_e),
        right: Box::new(Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![base.clone()].into(),
        }),
      })
    };
    let cos_part = Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![inner_arg.clone()].into(),
    };
    let sin_part = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![inner_arg].into(),
    };
    let real = ce_simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(exp_a.clone()),
      right: Box::new(cos_part),
    });
    let imag = ce_simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(exp_a),
      right: Box::new(sin_part),
    });
    return Some((real, imag));
  }
  // Exponent must be a non-negative integer literal.
  let n = match exp {
    Expr::Integer(n) if *n >= 0 => *n,
    _ => return None,
  };
  let (a, b) = split_real_imag(base)?;
  // Skip if imag part is exactly 0 — let normal recursion handle it.
  if matches!(b, Expr::Integer(0)) {
    return None;
  }
  let mut real_terms: Vec<Expr> = Vec::new();
  let mut imag_terms: Vec<Expr> = Vec::new();
  for k in 0..=n {
    let coef = crate::functions::binomial_coeff(n, k);
    // i^k cycles: 0→+real, 1→+imag, 2→-real, 3→-imag.
    let phase = k.rem_euclid(4);
    let sign: i128 = match phase {
      0 | 1 => 1,
      _ => -1,
    };
    let signed_coef = sign * coef;
    let term = make_term(signed_coef, &a, n - k, &b, k);
    if phase % 2 == 0 {
      real_terms.push(term);
    } else {
      imag_terms.push(term);
    }
  }
  Some((sum_terms(real_terms), sum_terms(imag_terms)))
}

fn make_term(coef: i128, a: &Expr, ai: i128, b: &Expr, bi: i128) -> Expr {
  let mut factors: Vec<Expr> = Vec::new();
  if coef != 1 {
    factors.push(Expr::Integer(coef));
  }
  if ai > 0 {
    if ai == 1 {
      factors.push(a.clone());
    } else {
      factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(a.clone()),
        right: Box::new(Expr::Integer(ai)),
      });
    }
  }
  if bi > 0 {
    if bi == 1 {
      factors.push(b.clone());
    } else {
      factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(b.clone()),
        right: Box::new(Expr::Integer(bi)),
      });
    }
  }
  if factors.is_empty() {
    return Expr::Integer(coef);
  }
  if factors.len() == 1 {
    return factors.into_iter().next().unwrap();
  }
  ce_simplify(Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  })
}

fn sum_terms(terms: Vec<Expr>) -> Expr {
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  if terms.len() == 1 {
    return terms.into_iter().next().unwrap();
  }
  ce_simplify(Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  })
}

/// Returns true when the base of a Power can be treated as a positive
/// real value for the purpose of complex-expansion (so
/// `base^(a + I*b) = base^a * (Cos[b*Log[base]] + I*Sin[b*Log[base]])`).
/// Recognizes E, positive integer literals, and positive Real literals.
fn is_complex_expand_real_base(base: &Expr) -> bool {
  match base {
    Expr::Identifier(s) | Expr::Constant(s) if s == "E" => true,
    Expr::Integer(n) if *n > 0 => true,
    Expr::Real(f) if *f > 0.0 => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!((&args[0], &args[1]), (Expr::Integer(n), Expr::Integer(d)) if *n > 0 && *d > 0)
    }
    _ => false,
  }
}

/// Closed-form expansions for `Abs[arg]` under `ComplexExpand`. Returns
/// `None` when the argument doesn't fit one of the recognised shapes so
/// the caller falls through to the split-real-imag path.
fn abs_complex_expand_rewrite(arg: &Expr) -> Option<Expr> {
  // Abs[a * b * …] → Abs[a] * Abs[b] * …
  let times_args: Option<Vec<Expr>> = match arg {
    Expr::FunctionCall { name, args } if name == "Times" => Some(args.to_vec()),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Some(vec![*left.clone(), *right.clone()]),
    _ => None,
  };
  if let Some(factors) = times_args
    && factors.len() >= 2
  {
    // Wrap each factor in Abs *without* pre-expanding. The outer
    // `complex_expand_recursive` pass that produced this rewrite will
    // recurse into the resulting Abs[…] sub-calls, where this same
    // helper can apply the Power/Log closed forms to the un-expanded
    // factor (e.g. recognising `Abs[Power[2, z]]` as `2^Re[z]`).
    let abs_factors: Vec<Expr> = factors
      .iter()
      .map(|f| {
        complex_expand_recursive(&Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![f.clone()].into(),
        })
      })
      .collect();
    return Some(Expr::FunctionCall {
      name: "Times".to_string(),
      args: abs_factors.into(),
    });
  }
  // Abs[Power[positive_real, c]] → Power[positive_real, Re[c]]
  let power_parts: Option<(Expr, Expr)> = match arg {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => Some((*left.clone(), *right.clone())),
    _ => None,
  };
  if let Some((base, exp)) = power_parts
    && is_complex_expand_real_base(&base)
  {
    let re_exp = Expr::FunctionCall {
      name: "Re".to_string(),
      args: vec![exp].into(),
    };
    let re_exp = complex_expand_recursive(&re_exp);
    return Some(Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base, re_exp].into(),
    });
  }
  // Abs[Log[w]] → Sqrt[Log[Abs[w]]^2 + Arg[w]^2]
  if let Expr::FunctionCall { name, args } = arg
    && name == "Log"
    && args.len() == 1
  {
    let w = &args[0];
    let abs_w = Expr::FunctionCall {
      name: "Abs".to_string(),
      args: vec![w.clone()].into(),
    };
    let log_abs_w = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![complex_expand_recursive(&abs_w)].into(),
    };
    let arg_w = Expr::FunctionCall {
      name: "Arg".to_string(),
      args: vec![w.clone()].into(),
    };
    let arg_w = complex_expand_recursive(&arg_w);
    let log_sq = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(log_abs_w),
      right: Box::new(Expr::Integer(2)),
    };
    let arg_sq = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(arg_w),
      right: Box::new(Expr::Integer(2)),
    };
    return Some(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(log_sq),
        right: Box::new(arg_sq),
      }]
      .into(),
    });
  }
  None
}

fn complex_expand_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      let arg = &args[0];
      // ComplexExpand distributes Abs across multiplicative factors:
      //   Abs[a*b] = Abs[a] * Abs[b], Abs[Power[positive, c]] = Power[positive, Re[c]],
      //   Abs[Log[w]] = Sqrt[Log[Abs[w]]^2 + Arg[w]^2].
      // Apply these rewrites before falling back to the
      // split-real-imag path so symbolic factors that wouldn't split
      // (e.g. `Log[2*(Re[z] + I*Im[z])]`) still get an explicit
      // closed form.
      if name == "Abs"
        && let Some(rewritten) = abs_complex_expand_rewrite(arg)
      {
        return ce_simplify(rewritten);
      }
      // Try to split the argument into real + I*imag parts
      if let Some((re, im)) = split_real_imag(arg) {
        // Check if imaginary part is zero
        let im_is_zero = matches!(&im, Expr::Integer(0));
        if !im_is_zero {
          match name.as_str() {
            // Sin[a + I*b] = Sin[a]*Cosh[b] + I*Cos[a]*Sinh[b]
            "Sin" => {
              let sin_a = Expr::FunctionCall {
                name: "Sin".to_string(),
                args: vec![re.clone()].into(),
              };
              let cos_a = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![re].into(),
              };
              let cosh_b = Expr::FunctionCall {
                name: "Cosh".to_string(),
                args: vec![im.clone()].into(),
              };
              let sinh_b = Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![im].into(),
              };
              return ce_simplify(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(sin_a),
                  right: Box::new(cosh_b),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Identifier("I".to_string())),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(cos_a),
                    right: Box::new(sinh_b),
                  }),
                }),
              });
            }
            // Cos[a + I*b] = Cos[a]*Cosh[b] - I*Sin[a]*Sinh[b]
            "Cos" => {
              let cos_a = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![re.clone()].into(),
              };
              let sin_a = Expr::FunctionCall {
                name: "Sin".to_string(),
                args: vec![re].into(),
              };
              let cosh_b = Expr::FunctionCall {
                name: "Cosh".to_string(),
                args: vec![im.clone()].into(),
              };
              let sinh_b = Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![im].into(),
              };
              return ce_simplify(Expr::BinaryOp {
                op: BinaryOperator::Minus,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(cos_a),
                  right: Box::new(cosh_b),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Identifier("I".to_string())),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(sin_a),
                    right: Box::new(sinh_b),
                  }),
                }),
              });
            }
            // Sinh[a + I*b] = Sinh[a]*Cos[b] + I*Cosh[a]*Sin[b]
            "Sinh" => {
              let sinh_a = Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![re.clone()].into(),
              };
              let cosh_a = Expr::FunctionCall {
                name: "Cosh".to_string(),
                args: vec![re].into(),
              };
              let cos_b = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![im.clone()].into(),
              };
              let sin_b = Expr::FunctionCall {
                name: "Sin".to_string(),
                args: vec![im].into(),
              };
              return ce_simplify(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(sinh_a),
                  right: Box::new(cos_b),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Identifier("I".to_string())),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(cosh_a),
                    right: Box::new(sin_b),
                  }),
                }),
              });
            }
            // Tanh[a + I*b] = (Sinh[2a] + I*Sin[2b]) / (Cos[2b] + Cosh[2a]).
            // The split-into-real-and-imag form Wolfram emits keeps the
            // shared denominator Cos[2b] + Cosh[2a] under both numerators
            // — emit it the same way so display matches `ComplexExpand`'s.
            "Tanh" => {
              let two_a = Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Integer(2)),
                right: Box::new(re.clone()),
              };
              let two_b = Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Integer(2)),
                right: Box::new(im.clone()),
              };
              let sinh_2a = Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![two_a.clone()].into(),
              };
              let sin_2b = Expr::FunctionCall {
                name: "Sin".to_string(),
                args: vec![two_b.clone()].into(),
              };
              let cos_2b = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![two_b].into(),
              };
              let cosh_2a = Expr::FunctionCall {
                name: "Cosh".to_string(),
                args: vec![two_a].into(),
              };
              let denom = Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(cos_2b),
                right: Box::new(cosh_2a),
              };
              let real_part = Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(sinh_2a),
                right: Box::new(denom.clone()),
              };
              let imag_part = Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(sin_2b),
                right: Box::new(denom),
              };
              return ce_simplify(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(real_part),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Identifier("I".to_string())),
                  right: Box::new(imag_part),
                }),
              });
            }
            // Cosh[a + I*b] = Cosh[a]*Cos[b] + I*Sinh[a]*Sin[b]
            "Cosh" => {
              let cosh_a = Expr::FunctionCall {
                name: "Cosh".to_string(),
                args: vec![re.clone()].into(),
              };
              let sinh_a = Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![re].into(),
              };
              let cos_b = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![im.clone()].into(),
              };
              let sin_b = Expr::FunctionCall {
                name: "Sin".to_string(),
                args: vec![im].into(),
              };
              return ce_simplify(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(cosh_a),
                  right: Box::new(cos_b),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Identifier("I".to_string())),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(sinh_a),
                    right: Box::new(sin_b),
                  }),
                }),
              });
            }
            // Exp[a + I*b] = E^a*Cos[b] + I*E^a*Sin[b]
            "Exp" => {
              let exp_a = Expr::FunctionCall {
                name: "Exp".to_string(),
                args: vec![re].into(),
              };
              let cos_b = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![im.clone()].into(),
              };
              let sin_b = Expr::FunctionCall {
                name: "Sin".to_string(),
                args: vec![im].into(),
              };
              return ce_simplify(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(exp_a.clone()),
                  right: Box::new(cos_b),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Identifier("I".to_string())),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(exp_a),
                    right: Box::new(sin_b),
                  }),
                }),
              });
            }
            // Abs[a + I*b] = Sqrt[a^2 + b^2]
            "Abs" => {
              return ce_simplify(Expr::FunctionCall {
                name: "Sqrt".to_string(),
                args: vec![Expr::BinaryOp {
                  op: BinaryOperator::Plus,
                  left: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left: Box::new(re),
                    right: Box::new(Expr::Integer(2)),
                  }),
                  right: Box::new(Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left: Box::new(im),
                    right: Box::new(Expr::Integer(2)),
                  }),
                }]
                .into(),
              });
            }
            _ => {}
          }
        }
        // Re/Im/Conjugate hold whether or not the imaginary part is zero —
        // ComplexExpand assumes every symbol is real, so Re[a] = a, Im[a] = 0,
        // Conjugate[a] = a. (When im != 0 this gives Re[a+I b] = a, etc.)
        match name.as_str() {
          "Re" => return ce_simplify(re),
          "Im" => return ce_simplify(im),
          // Real argument (im == 0): Abs[u] = Sqrt[u^2]. The im != 0 case is
          // handled in the block above. wolframscript treats every symbol as
          // real, so ComplexExpand[Abs[x]] = Sqrt[x^2] and hence
          // ComplexExpand[Abs[x]^2] = x^2, ComplexExpand[Abs[x]^3] =
          // (x^2)^(3/2).
          "Abs" => {
            return ce_simplify(Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(re),
                  right: Box::new(Expr::Integer(2)),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(im),
                  right: Box::new(Expr::Integer(2)),
                }),
              }]
              .into(),
            });
          }
          "Conjugate" => {
            return ce_simplify(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(re),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Identifier("I".to_string())),
                right: Box::new(im),
              }),
            });
          }
          _ => {}
        }
      }

      // No complex expansion possible, recurse into children
      Expr::FunctionCall {
        name: name.clone(),
        args: args.iter().map(complex_expand_recursive).collect(),
      }
    }
    // Handle base^(a + I*b) for a positive real-valued base:
    //   base^(a + I*b) = base^a * (Cos[b*Log[base]] + I*Sin[b*Log[base]])
    // E (constant) and positive integers are treated as positive reals.
    // For E specifically, Log[E] = 1, so b*Log[E] simplifies to b.
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } if is_complex_expand_real_base(base) => {
      if let Some((re, im)) = split_real_imag(exp) {
        let im_is_zero = matches!(&im, Expr::Integer(0));
        if !im_is_zero {
          let exp_a = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: base.clone(),
            right: Box::new(re),
          };
          // Inner Cos/Sin argument: b * Log[base]; collapse to b for E.
          let inner_arg = if matches!(base.as_ref(), Expr::Identifier(s) if s == "E")
            || matches!(base.as_ref(), Expr::Constant(s) if s == "E")
          {
            im.clone()
          } else {
            ce_simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(im.clone()),
              right: Box::new(Expr::FunctionCall {
                name: "Log".to_string(),
                args: vec![(**base).clone()].into(),
              }),
            })
          };
          let cos_b = Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![inner_arg.clone()].into(),
          };
          let sin_b = Expr::FunctionCall {
            name: "Sin".to_string(),
            args: vec![inner_arg].into(),
          };
          return ce_simplify(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(exp_a.clone()),
              right: Box::new(cos_b),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Identifier("I".to_string())),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(exp_a),
                right: Box::new(sin_b),
              }),
            }),
          });
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(complex_expand_recursive(base)),
        right: Box::new(complex_expand_recursive(exp)),
      }
    }
    // Recurse into binary ops
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(complex_expand_recursive(left)),
      right: Box::new(complex_expand_recursive(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(complex_expand_recursive(operand)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(complex_expand_recursive).collect(),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(complex_expand_recursive).collect())
    }
    _ => expr.clone(),
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
      op: BinaryOperator::Power,
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
          args: vec![x.clone()].into(),
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Complex".to_string(),
              args: vec![Expr::Integer(0), Expr::Integer(1)].into(),
            },
            Expr::FunctionCall {
              name: "Sin".to_string(),
              args: vec![x.clone()].into(),
            },
          ]
          .into(),
        },
      ]
      .into(),
    }
  } else {
    // Cosh[z] + Sinh[z]
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Cosh".to_string(),
          args: vec![z.clone()].into(),
        },
        Expr::FunctionCall {
          name: "Sinh".to_string(),
          args: vec![z.clone()].into(),
        },
      ]
      .into(),
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
            args: other_factors.into(),
          })
        }
      } else {
        None
      }
    }
    // z = BinaryOp Times with I
    Expr::BinaryOp {
      op: BinaryOperator::Times,
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
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
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
        // Tan[x] = I*(E^(-I*x) - E^(I*x))/(E^(-I*x) + E^(I*x))
        // Mirror the Cot construction (E^(-I*x) term first) so the negated
        // term is the positive-exponent E^(I*x); this matches wolframscript's
        // form and avoids rendering E^(-I*x) as -(1/E^(I*x)).
        "Tan" => {
          let ix = times(&[i.clone(), arg.clone()]);
          let e_ix = power(e.clone(), ix.clone());
          let e_nix = power(e.clone(), times(&[Expr::Integer(-1), ix]));
          times(&[
            i,
            plus(&[e_nix.clone(), times(&[Expr::Integer(-1), e_ix.clone()])]),
            power(plus(&[e_nix, e_ix]), Expr::Integer(-1)),
          ])
        }
        // Sec[x] = 2/(E^(I*x) + E^(-I*x))
        "Sec" => {
          let ix = times(&[i.clone(), arg.clone()]);
          let e_ix = power(e.clone(), ix.clone());
          let e_nix = power(e.clone(), times(&[Expr::Integer(-1), ix]));
          times(&[
            Expr::Integer(2),
            power(plus(&[e_ix, e_nix]), Expr::Integer(-1)),
          ])
        }
        // Csc[x] = -2*I/(E^(-I*x) - E^(I*x))
        "Csc" => {
          let ix = times(&[i.clone(), arg.clone()]);
          let e_ix = power(e.clone(), ix.clone());
          let e_nix = power(e.clone(), times(&[Expr::Integer(-1), ix]));
          times(&[
            Expr::Integer(-2),
            i,
            power(
              plus(&[e_nix, times(&[Expr::Integer(-1), e_ix])]),
              Expr::Integer(-1),
            ),
          ])
        }
        // Cot[x] = -I*(E^(-I*x) + E^(I*x))/(E^(-I*x) - E^(I*x))
        "Cot" => {
          let ix = times(&[i.clone(), arg.clone()]);
          let e_ix = power(e.clone(), ix.clone());
          let e_nix = power(e.clone(), times(&[Expr::Integer(-1), ix]));
          times(&[
            Expr::Integer(-1),
            i,
            plus(&[e_nix.clone(), e_ix.clone()]),
            power(
              plus(&[e_nix, times(&[Expr::Integer(-1), e_ix])]),
              Expr::Integer(-1),
            ),
          ])
        }
        // Sech[x] = 2/(E^x + E^(-x))
        "Sech" => {
          let e_x = power(e.clone(), arg.clone());
          let e_nx = power(e.clone(), times(&[Expr::Integer(-1), arg]));
          times(&[
            Expr::Integer(2),
            power(plus(&[e_x, e_nx]), Expr::Integer(-1)),
          ])
        }
        // ArcTan[x] = I/2 Log[1 - I x] - I/2 Log[1 + I x]
        "ArcTan" => {
          let ix = times(&[i.clone(), arg.clone()]);
          plus(&[
            times(&[
              i.clone(),
              half.clone(),
              log_of(plus(&[
                Expr::Integer(1),
                times(&[Expr::Integer(-1), ix.clone()]),
              ])),
            ]),
            times(&[
              Expr::Integer(-1),
              i.clone(),
              half.clone(),
              log_of(plus(&[Expr::Integer(1), ix])),
            ]),
          ])
        }
        // ArcCot[x] = I/2 Log[1 - I/x] - I/2 Log[1 + I/x]
        "ArcCot" => {
          let i_over_x =
            times(&[i.clone(), power(arg.clone(), Expr::Integer(-1))]);
          plus(&[
            times(&[
              i.clone(),
              half.clone(),
              log_of(plus(&[
                Expr::Integer(1),
                times(&[Expr::Integer(-1), i_over_x.clone()]),
              ])),
            ]),
            times(&[
              Expr::Integer(-1),
              i.clone(),
              half.clone(),
              log_of(plus(&[Expr::Integer(1), i_over_x])),
            ]),
          ])
        }
        // ArcSec[x] = Pi/2 + I Log[Sqrt[1 - x^-2] + I/x]
        "ArcSec" => {
          let inner = plus(&[
            sqrt_of(plus(&[
              Expr::Integer(1),
              times(&[
                Expr::Integer(-1),
                power(arg.clone(), Expr::Integer(-2)),
              ]),
            ])),
            times(&[i.clone(), power(arg.clone(), Expr::Integer(-1))]),
          ]);
          plus(&[
            times(&[Expr::Constant("Pi".to_string()), half.clone()]),
            times(&[i.clone(), log_of(inner)]),
          ])
        }
        // ArcCsc[x] = -I Log[Sqrt[1 - x^-2] + I/x]
        "ArcCsc" => {
          let inner = plus(&[
            sqrt_of(plus(&[
              Expr::Integer(1),
              times(&[
                Expr::Integer(-1),
                power(arg.clone(), Expr::Integer(-2)),
              ]),
            ])),
            times(&[i.clone(), power(arg.clone(), Expr::Integer(-1))]),
          ]);
          times(&[Expr::Integer(-1), i.clone(), log_of(inner)])
        }
        // ArcSinh[x] = Log[x + Sqrt[1 + x^2]]
        "ArcSinh" => log_of(plus(&[
          arg.clone(),
          sqrt_of(plus(&[
            Expr::Integer(1),
            power(arg.clone(), Expr::Integer(2)),
          ])),
        ])),
        // ArcCosh[x] = Log[x + Sqrt[-1 + x] Sqrt[1 + x]]
        "ArcCosh" => log_of(plus(&[
          arg.clone(),
          times(&[
            sqrt_of(plus(&[Expr::Integer(-1), arg.clone()])),
            sqrt_of(plus(&[Expr::Integer(1), arg.clone()])),
          ]),
        ])),
        // ArcTanh[x] = -1/2 Log[1 - x] + 1/2 Log[1 + x]
        "ArcTanh" => plus(&[
          times(&[
            Expr::Integer(-1),
            half.clone(),
            log_of(plus(&[
              Expr::Integer(1),
              times(&[Expr::Integer(-1), arg.clone()]),
            ])),
          ]),
          times(&[
            half.clone(),
            log_of(plus(&[Expr::Integer(1), arg.clone()])),
          ]),
        ]),
        // ArcCoth[x] = -1/2 Log[1 - x^-1] + 1/2 Log[1 + x^-1]
        "ArcCoth" => {
          let x_inv = power(arg.clone(), Expr::Integer(-1));
          plus(&[
            times(&[
              Expr::Integer(-1),
              half.clone(),
              log_of(plus(&[
                Expr::Integer(1),
                times(&[Expr::Integer(-1), x_inv.clone()]),
              ])),
            ]),
            times(&[half.clone(), log_of(plus(&[Expr::Integer(1), x_inv]))]),
          ])
        }
        // NOTE: Coth and Csch are intentionally left to the default recursive
        // branch. Their exponential forms are value-correct but the denominator
        // contains a negative *real* E-power (`-E^(-x)`), which Woxi's core
        // renderer canonicalizes as `-(1/E^x)` rather than wolframscript's
        // `-E^(-x)`. See the documented negative-E-exponent display divergence.
        // (Csc is handled above: its denominator uses imaginary exponents
        // `E^(-I*x) - E^(I*x)`, which render identically to wolframscript.)
        // ArcSin, ArcCos, ArcCsch and ArcSech are likewise omitted: their Log
        // arguments are a Plus whose term order Woxi canonicalizes differently
        // (e.g. `Sqrt[1-x^2] + I x` vs wolframscript's `I x + Sqrt[1-x^2]`).
        // Tanh[x] = E^x/(E^(-x) + E^x) - E^(-x)/(E^(-x) + E^x)
        "Tanh" => {
          let e_x = power(e.clone(), arg.clone());
          let e_nx = power(e.clone(), times(&[Expr::Integer(-1), arg]));
          let denom =
            power(plus(&[e_nx.clone(), e_x.clone()]), Expr::Integer(-1));
          plus(&[
            times(&[e_x, denom.clone()]),
            times(&[Expr::Integer(-1), e_nx, denom]),
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
  unevaluated("Times", factors)
}

fn plus(terms: &[Expr]) -> Expr {
  unevaluated("Plus", terms)
}

fn power(base: Expr, exp: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp].into(),
  }
}

fn log_of(arg: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![arg].into(),
  }
}

fn sqrt_of(arg: Expr) -> Expr {
  power(
    arg,
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
    },
  )
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
        args: vec![operand.clone(), value.clone()].into(),
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

/// True when `e` is a literal zero (integer or real).
fn is_zero_literal(e: &Expr) -> bool {
  matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(f) if *f == 0.0)
}

/// True when `e` is a numeric literal that is provably non-zero (so dividing by
/// it needs no `!= 0` guard). Rationals are stored as Rational[p, q].
fn is_nonzero_number(e: &Expr) -> bool {
  match e {
    Expr::Integer(n) => *n != 0,
    Expr::Real(f) => *f != 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!(&args[0], Expr::Integer(p) if *p != 0)
    }
    _ => false,
  }
}

/// Wrap the result of multiplying/dividing both sides of an equation by a
/// scalar in a `c != 0` guard when the scalar may be zero, matching
/// wolframscript: MultiplySides[a == b, c] -> Piecewise[{{a c == b c, c != 0}},
/// a == b]. A provably non-zero numeric scalar needs no guard, and only
/// equations are guarded (inequalities need sign-dependent reasoning that is
/// left to the plain scaled result).
fn guard_equation_scale(
  relation: &Expr,
  scalar: &Expr,
  scaled: Expr,
) -> Result<Expr, InterpreterError> {
  let is_equation = matches!(
    relation,
    Expr::Comparison { operands, operators }
      if operands.len() == 2
        && operators.first() == Some(&ComparisonOp::Equal)
  );
  if !is_equation || is_nonzero_number(scalar) {
    return Ok(scaled);
  }
  let cond = Expr::Comparison {
    operands: vec![scalar.clone(), Expr::Integer(0)],
    operators: vec![ComparisonOp::NotEqual],
  };
  let branch = Expr::List(vec![scaled, cond].into());
  let pw = Expr::FunctionCall {
    name: "Piecewise".to_string(),
    args: vec![Expr::List(vec![branch].into()), relation.clone()].into(),
  };
  evaluate_expr_to_expr(&pw)
}

/// Per-operand combiner for the `*Sides` family.
#[derive(Clone, Copy)]
enum SideOp {
  Add,
  Subtract,
  Multiply,
  Divide,
}

/// Handle the case where the second argument of a `*Sides` function is itself
/// a two-sided equation `c == d`: the corresponding sides are combined, so
/// e.g. SubtractSides[a == b, c == d] -> a - c == b - d. For DivideSides the
/// result is guarded by `c != 0`, matching wolframscript. Returns None unless
/// both relations have exactly two operands and the second is an equation.
fn pair_sides(relation: &Expr, second: &Expr, op: SideOp) -> Option<Expr> {
  let (
    Expr::Comparison {
      operands,
      operators,
    },
    Expr::Comparison {
      operands: v_ops,
      operators: v_operators,
    },
  ) = (relation, second)
  else {
    return None;
  };
  if operands.len() != 2 || v_ops.len() != 2 {
    return None;
  }
  // The second argument must be an equation for the pairing to be meaningful.
  if v_operators.first() != Some(&ComparisonOp::Equal) {
    return None;
  }
  // Multiplying/dividing an inequality requires sign-dependent reasoning
  // (the direction can flip), which Wolfram expresses with a larger
  // Piecewise. Only handle the clean case where the first relation is an
  // equation; Add/Subtract preserve any relation's direction.
  if matches!(op, SideOp::Multiply | SideOp::Divide)
    && operators.first() != Some(&ComparisonOp::Equal)
  {
    return None;
  }

  let combine = |a: &Expr, b: &Expr| -> Expr {
    match op {
      SideOp::Add => Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![a.clone(), b.clone()].into(),
      },
      SideOp::Subtract => Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          a.clone(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), b.clone()].into(),
          },
        ]
        .into(),
      },
      SideOp::Multiply => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![a.clone(), b.clone()].into(),
      },
      SideOp::Divide => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          a.clone(),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![b.clone(), Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      },
    }
  };

  let paired = Expr::Comparison {
    operands: vec![
      combine(&operands[0], &v_ops[0]),
      combine(&operands[1], &v_ops[1]),
    ],
    operators: operators.clone(),
  };

  match op {
    // Dividing by `c` is only reversible when c != 0, so guard it like
    // wolframscript: Piecewise[{{paired, c != 0}}, relation].
    SideOp::Divide => {
      let cond = Expr::Comparison {
        operands: vec![v_ops[0].clone(), Expr::Integer(0)],
        operators: vec![ComparisonOp::NotEqual],
      };
      let branch = Expr::List(vec![paired, cond].into());
      Some(Expr::FunctionCall {
        name: "Piecewise".to_string(),
        args: vec![Expr::List(vec![branch].into()), relation.clone()].into(),
      })
    }
    _ => Some(paired),
  }
}

/// Apply a per-channel min (`use_min = true`) or max filter to an
/// Image using a (2r+1)×(2r+1) window clipped at image boundaries.
/// Returns None when `data` isn't an Image or `radius` isn't a
/// non-negative integer-like value.
fn image_min_max_filter(
  data: &Expr,
  radius: &Expr,
  use_min: bool,
) -> Option<Expr> {
  let Expr::Image {
    color_space: _,
    width,
    height,
    channels,
    data: pixels,
    image_type,
  } = data
  else {
    return None;
  };
  let r = match radius {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    Expr::Real(f) if *f >= 0.0 => f.round() as usize,
    _ => return None,
  };
  let w = *width as usize;
  let h = *height as usize;
  let ch = *channels as usize;
  let mut new_data = vec![0.0_f64; pixels.len()];
  for c_idx in 0..ch {
    for y in 0..h {
      for x in 0..w {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        let x0 = x.saturating_sub(r);
        let x1 = (x + r).min(w - 1);
        let mut best = pixels[(y0 * w + x0) * ch + c_idx];
        for yy in y0..=y1 {
          for xx in x0..=x1 {
            let v = pixels[(yy * w + xx) * ch + c_idx];
            if use_min {
              if v < best {
                best = v;
              }
            } else if v > best {
              best = v;
            }
          }
        }
        new_data[(y * w + x) * ch + c_idx] = best;
      }
    }
  }
  Some(Expr::Image {
    color_space: None,
    width: *width,
    height: *height,
    channels: *channels,
    data: std::sync::Arc::new(new_data),
    image_type: *image_type,
  })
}

use crate::functions::math_ast::expr_to_f64;

/// Standardize[data] — subtract mean and divide by standard deviation
/// Standardize[data, f1, f2] — use f1 for location and f2 for scale
fn standardize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = &args[0];
  let items = match data {
    Expr::List(items) => items,
    _ => {
      crate::emit_message(
        "Standardize::vectmat: The first argument is expected to be a vector or matrix.",
      );
      return Ok(unevaluated("Standardize", args));
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
  Ok(Expr::List(result.into()))
}
