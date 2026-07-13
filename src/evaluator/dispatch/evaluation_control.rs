#[allow(unused_imports)]
use super::*;

pub fn dispatch_evaluation_control(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "HoldForm" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "HoldForm".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "Hold" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: "Hold".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "HoldComplete" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: "HoldComplete".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "Unevaluated" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: "Unevaluated".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ReleaseHold" if args.len() == 1 => {
      // ReleaseHold removes Hold/HoldForm/HoldComplete/HoldPattern wrappers
      // wherever they appear (one top-down pass, like ReplaceAll — it does not
      // descend into the content it just released, so ReleaseHold[Hold[Hold[…]]]
      // keeps the inner Hold). The stripped expression is then evaluated.
      let stripped = release_hold_rec(&args[0]);
      return Some(evaluate_expr_to_expr(&stripped));
    }
    "TimeRemaining" if args.is_empty() => {
      return Some(Ok(Expr::Identifier("Infinity".to_string())));
    }
    "Out" => {
      // `Out[]` and `Out[k]` for k <= 0 resolve to the cached previous
      // output if one is available (set after each successful evaluation
      // by `interpret_with_stdout`). Without a cached value we collapse to
      // `Out[0]` for parity with wolframscript on `$Line == 1`. Positive
      // integers stay symbolic — we don't keep numbered history.
      let target: Option<i128> = match args {
        [] => Some(0),
        [Expr::Integer(n)] => Some(*n),
        _ => None,
      };
      if let Some(k) = target {
        if k <= 0
          && let Some(prev) = crate::get_last_output()
        {
          return Some(Ok(prev));
        }
        let normalized = if k <= 0 { 0 } else { k };
        return Some(Ok(Expr::FunctionCall {
          name: "Out".to_string(),
          args: vec![Expr::Integer(normalized)].into(),
        }));
      }
    }
    "Evaluate" if args.len() == 1 => {
      return Some(Ok(args[0].clone()));
    }
    // `Evaluate[a, b, c]` returns `Sequence[a, b, c]`, which then splices
    // into the surrounding context. Matches wolframscript's
    // `Hold[Evaluate[1, 2]]` → `Hold[1, 2]`.
    "Evaluate" => {
      return Some(Ok(Expr::FunctionCall {
        name: "Sequence".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "RegularExpression" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "RegularExpression".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "UniformDistribution" if args.len() <= 1 => {
      let uni_args = if args.is_empty() {
        vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "UniformDistribution".to_string(),
        args: uni_args.into(),
      }));
    }
    "NormalDistribution" => {
      let norm_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "NormalDistribution".to_string(),
        args: norm_args.into(),
      }));
    }
    "ExponentialDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ExponentialDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "PoissonDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "PoissonDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BernoulliDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BernoulliDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "InverseGammaDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "InverseGammaDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "GammaDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "GammaDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MultinormalDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MultinormalDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ProductDistribution" if args.len() >= 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ProductDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "UniformSumDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "UniformSumDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BetaBinomialDistribution" if args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BetaBinomialDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BetaPrimeDistribution" if (2..=4).contains(&args.len()) => {
      return Some(Ok(Expr::FunctionCall {
        name: "BetaPrimeDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "NoncentralChiSquareDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "NoncentralChiSquareDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ExponentialPowerDistribution" if args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ExponentialPowerDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "RiceDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "RiceDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MinStableDistribution" if args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MinStableDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MaxStableDistribution" if args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MaxStableDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "TriangularDistribution" if args.len() <= 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "TriangularDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MaxwellDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MaxwellDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "WignerSemicircleDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "WignerSemicircleDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "SechDistribution" if args.len() <= 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "SechDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MoyalDistribution" if args.len() <= 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MoyalDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BorelTannerDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BorelTannerDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "PoissonConsulDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "PoissonConsulDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "SuzukiDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "SuzukiDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MeixnerDistribution" if args.len() == 4 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MeixnerDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BenktanderGibratDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BenktanderGibratDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "GumbelDistribution" if args.len() <= 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "GumbelDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ZipfDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ZipfDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BenfordDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BenfordDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BenktanderWeibullDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BenktanderWeibullDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "SinghMaddalaDistribution" if args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: "SinghMaddalaDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "WaringYuleDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "WaringYuleDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "Query" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: "Query".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BetaDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BetaDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "StudentTDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "StudentTDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "LogNormalDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "LogNormalDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "LogisticDistribution" => {
      let logistic_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "LogisticDistribution".to_string(),
        args: logistic_args.into(),
      }));
    }
    "GompertzMakehamDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "GompertzMakehamDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "InverseGaussianDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "InverseGaussianDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "FrechetDistribution" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: "FrechetDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ExtremeValueDistribution" => {
      let evd_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else if args.len() == 2 {
        args.to_vec()
      } else {
        return None;
      };
      return Some(Ok(Expr::FunctionCall {
        name: "ExtremeValueDistribution".to_string(),
        args: evd_args.into(),
      }));
    }
    "InverseChiSquareDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "InverseChiSquareDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ChiSquareDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ChiSquareDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ParetoDistribution" if (2..=4).contains(&args.len()) => {
      return Some(Ok(Expr::FunctionCall {
        name: "ParetoDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "WeibullDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "WeibullDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "GeometricDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "GeometricDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "LogSeriesDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "LogSeriesDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "NakagamiDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "NakagamiDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "LogLogisticDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "LogLogisticDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "HypergeometricDistribution" if args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: "HypergeometricDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "BinormalDistribution" if (1..=3).contains(&args.len()) => {
      return Some(Ok(Expr::FunctionCall {
        name: "BinormalDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "CauchyDistribution" => {
      let cauchy_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "CauchyDistribution".to_string(),
        args: cauchy_args.into(),
      }));
    }
    "DiscreteUniformDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "DiscreteUniformDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "LaplaceDistribution" => {
      let laplace_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else if args.len() == 2 {
        args.to_vec()
      } else {
        return None;
      };
      return Some(Ok(Expr::FunctionCall {
        name: "LaplaceDistribution".to_string(),
        args: laplace_args.into(),
      }));
    }
    "RayleighDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "RayleighDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "NegativeBinomialDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "NegativeBinomialDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MultinomialDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MultinomialDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "NegativeMultinomialDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "NegativeMultinomialDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // DiscreteMarkovProcess and its distribution wrappers are symbolic
    // objects consumed by PDF/CDF/Mean/Variance.
    "DiscreteMarkovProcess"
    | "StationaryDistribution"
    | "FirstPassageTimeDistribution" => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec().into(),
      }));
    }
    // StateSpaceModel[{a, b, c, d}] is a symbolic control-system object:
    // it echoes unevaluated and is consumed by ObservabilityMatrix /
    // ControllabilityMatrix.
    "StateSpaceModel" => {
      return Some(Ok(Expr::FunctionCall {
        name: "StateSpaceModel".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // FailureDistribution[bexpr, {{x1, d1}, …}] normalizes the event
    // variables to their positional indices (x || y becomes 1 || 2),
    // exactly as wolframscript displays it. Validation (positive
    // unateness) happens at CDF/PDF time, not here.
    "FailureDistribution" if args.len() == 2 => {
      fn substitute(e: &Expr, map: &[(String, i128)]) -> Expr {
        match e {
          Expr::Identifier(v) => {
            for (name, idx) in map {
              if name == v {
                return Expr::Integer(*idx);
              }
            }
            e.clone()
          }
          Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
            op: *op,
            left: Box::new(substitute(left, map)),
            right: Box::new(substitute(right, map)),
          },
          Expr::UnaryOp { op, operand } => Expr::UnaryOp {
            op: *op,
            operand: Box::new(substitute(operand, map)),
          },
          Expr::FunctionCall { name, args }
            if name == "And" || name == "Or" || name == "Not" =>
          {
            Expr::FunctionCall {
              name: name.clone(),
              args: args.iter().map(|a| substitute(a, map)).collect(),
            }
          }
          _ => e.clone(),
        }
      }
      if let Expr::List(pairs) = &args[1]
        && !pairs.is_empty()
        && pairs.iter().all(|p| {
          matches!(p, Expr::List(kv)
            if kv.len() == 2 && matches!(&kv[0], Expr::Identifier(_)))
        })
      {
        let mut map: Vec<(String, i128)> = Vec::new();
        let mut new_pairs: Vec<Expr> = Vec::new();
        for (i, p) in pairs.iter().enumerate() {
          let Expr::List(kv) = p else { unreachable!() };
          let Expr::Identifier(v) = &kv[0] else {
            unreachable!()
          };
          map.push((v.clone(), i as i128 + 1));
          new_pairs.push(Expr::List(
            vec![Expr::Integer(i as i128 + 1), kv[1].clone()].into(),
          ));
        }
        return Some(Ok(Expr::FunctionCall {
          name: "FailureDistribution".to_string(),
          args: vec![substitute(&args[0], &map), Expr::List(new_pairs.into())]
            .into(),
        }));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "FailureDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // StandbyDistribution[Exp[λ1], {Exp[λ2], …}] with perfect switching
    // normalizes to HypoexponentialDistribution[{λ1, λ2, …}]
    // (wolframscript-verified, also for symbolic rates). Other component
    // kinds and the switching-probability/switch-distribution forms stay
    // unevaluated.
    "StandbyDistribution" => {
      let rate = |e: &Expr| -> Option<Expr> {
        match e {
          Expr::FunctionCall { name, args }
            if name == "ExponentialDistribution" && args.len() == 1 =>
          {
            Some(args[0].clone())
          }
          _ => None,
        }
      };
      if args.len() == 2
        && let Some(r1) = rate(&args[0])
        && let Expr::List(rest) = &args[1]
        && !rest.is_empty()
        && let Some(mut rates) =
          rest.iter().map(&rate).collect::<Option<Vec<Expr>>>()
      {
        rates.insert(0, r1);
        return Some(Ok(Expr::FunctionCall {
          name: "HypoexponentialDistribution".to_string(),
          args: vec![Expr::List(rates.into())].into(),
        }));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "StandbyDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // The constructor never validates (wolframscript echoes even
    // non-symmetric matrices silently); Mean/Variance validate.
    "WishartMatrixDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "WishartMatrixDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "MultivariatePoissonDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MultivariatePoissonDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "DirichletDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "DirichletDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // HalfSpace[n] normalizes to HalfSpace[n, 0] (wolframscript-verified).
    "HalfSpace" if args.len() == 1 && matches!(&args[0], Expr::List(_)) => {
      return Some(Ok(Expr::FunctionCall {
        name: "HalfSpace".to_string(),
        args: vec![args[0].clone(), Expr::Integer(0)].into(),
      }));
    }
    // SphericalShell normalizes to its full form
    // SphericalShell[center, {rinner, router}]: the default shell is
    // {1/2, 1}, a single radius r means {r/2, r}, and a bare radius pair
    // gets the origin center. (wolframscript-verified.)
    "SphericalShell" if args.len() <= 2 => {
      let origin = || {
        Expr::List(
          vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)].into(),
        )
      };
      let normalized = match args {
        [] => Some((
          origin(),
          Expr::List(
            vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
              Expr::Integer(1),
            ]
            .into(),
          ),
        )),
        [Expr::List(radii)] if radii.len() == 2 => {
          Some((origin(), args[0].clone()))
        }
        [r] if !matches!(r, Expr::List(_)) => {
          let half = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(r.clone()),
            right: Box::new(Expr::Integer(2)),
          });
          match half {
            Ok(half) => {
              Some((origin(), Expr::List(vec![half, r.clone()].into())))
            }
            Err(_) => None,
          }
        }
        _ => None,
      };
      return Some(Ok(match normalized {
        Some((center, radii)) => Expr::FunctionCall {
          name: "SphericalShell".to_string(),
          args: vec![center, radii].into(),
        },
        None => Expr::FunctionCall {
          name: "SphericalShell".to_string(),
          args: args.to_vec().into(),
        },
      }));
    }
    // CapsuleShape[] / CapsuleShape[r] normalize to the full form with the
    // default x-axis endpoints {{-1, 0, 0}, {1, 0, 0}}.
    "CapsuleShape" if args.len() <= 2 => {
      let default_points = || {
        let pt = |x: i128| {
          Expr::List(
            vec![Expr::Integer(x), Expr::Integer(0), Expr::Integer(0)].into(),
          )
        };
        Expr::List(vec![pt(-1), pt(1)].into())
      };
      let normalized = match args {
        [] => Some((default_points(), Expr::Integer(1))),
        [r] if !matches!(r, Expr::List(_)) => {
          Some((default_points(), r.clone()))
        }
        _ => None,
      };
      return Some(Ok(match normalized {
        Some((points, r)) => Expr::FunctionCall {
          name: "CapsuleShape".to_string(),
          args: vec![points, r].into(),
        },
        None => Expr::FunctionCall {
          name: "CapsuleShape".to_string(),
          args: args.to_vec().into(),
        },
      }));
    }
    "ArcSinDistribution" if args.is_empty() => {
      // Default: ArcSinDistribution[{0, 1}]
      return Some(Ok(Expr::FunctionCall {
        name: "ArcSinDistribution".to_string(),
        args: vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())]
          .into(),
      }));
    }
    "ArcSinDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ArcSinDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "HalfNormalDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "HalfNormalDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "ChiDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ChiDistribution".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "StableDistribution"
      if args.len() == 2 || args.len() == 4 || args.len() == 5 =>
    {
      // Normalize to canonical 5-parameter form: StableDistribution[1, alpha, beta, mu, sigma]
      // 2-param: StableDistribution[alpha, beta] -> StableDistribution[1, alpha, beta, 0, 1]
      // 4-param: StableDistribution[alpha, beta, mu, sigma] -> StableDistribution[1, alpha, beta, mu, sigma]
      // 5-param: already canonical
      let canonical_args = match args.len() {
        2 => vec![
          Expr::Integer(1),
          args[0].clone(),
          args[1].clone(),
          Expr::Integer(0),
          Expr::Integer(1),
        ],
        4 => vec![
          Expr::Integer(1),
          args[0].clone(),
          args[1].clone(),
          args[2].clone(),
          args[3].clone(),
        ],
        _ => args.to_vec(), // 5-param, already canonical
      };
      return Some(Ok(Expr::FunctionCall {
        name: "StableDistribution".to_string(),
        args: canonical_args.into(),
      }));
    }
    // DistributionParameterQ[dist] — test if a distribution's parameters are valid
    "DistributionParameterQ" if args.len() == 1 => {
      if let Expr::FunctionCall {
        name: dist_name,
        args: dist_args,
      } = &args[0]
      {
        let result = validate_distribution_params(dist_name, dist_args);
        return Some(Ok(Expr::Identifier(
          if result { "True" } else { "False" }.to_string(),
        )));
      }
      // Not a recognized distribution — return unevaluated
      return Some(Ok(Expr::FunctionCall {
        name: "DistributionParameterQ".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // ByteArray[{b1, b2, ...}] — create a byte array from a list of unsigned bytes
    // ByteArray["base64string"] — create a byte array from base64
    "ByteArray" if args.len() == 1 => {
      match &args[0] {
        Expr::List(items) => {
          // Validate all items are integers in 0..255, encode as base64
          use base64::Engine;
          let engine = base64::engine::general_purpose::STANDARD;
          let mut raw_bytes = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if (0..=255).contains(n) => {
                raw_bytes.push(*n as u8);
              }
              _ => {
                crate::emit_message(
                  "ByteArray::lend: The argument at position 1 in ByteArray[...] should be a vector of unsigned byte values or a Base64-encoded string.",
                );
                return Some(Ok(Expr::FunctionCall {
                  name: "ByteArray".to_string(),
                  args: args.to_vec().into(),
                }));
              }
            }
          }
          let b64 = engine.encode(&raw_bytes);
          return Some(Ok(Expr::FunctionCall {
            name: "ByteArray".to_string(),
            args: vec![Expr::String(b64)].into(),
          }));
        }
        Expr::String(s) => {
          // Validate base64 string, then store as-is
          use base64::Engine;
          let engine = base64::engine::general_purpose::STANDARD;
          match engine.decode(s) {
            Ok(_) => {
              return Some(Ok(Expr::FunctionCall {
                name: "ByteArray".to_string(),
                args: vec![Expr::String(s.clone())].into(),
              }));
            }
            Err(_) => {
              crate::emit_message(
                "ByteArray::lend: The argument at position 1 in ByteArray[...] should be a vector of unsigned byte values or a Base64-encoded string.",
              );
              return Some(Ok(Expr::FunctionCall {
                name: "ByteArray".to_string(),
                args: args.to_vec().into(),
              }));
            }
          }
        }
        _ => {
          crate::emit_message(&format!(
            "ByteArray::lend: The argument at position 1 in ByteArray[{}] should be a vector of unsigned byte values or a Base64-encoded string.",
            crate::syntax::expr_to_string(&args[0])
          ));
          return Some(Ok(Expr::FunctionCall {
            name: "ByteArray".to_string(),
            args: args.to_vec().into(),
          }));
        }
      }
    }
    // NumericArray[list] / NumericArray[list, type] — typed numeric array.
    // Without an explicit type, auto-detect the smallest type that fits all
    // elements (currently only `UnsignedInteger8` for non-negative integers
    // ≤ 255). The result keeps its underlying list payload for First/Last
    // and AtomQ already returns True for the head; OutputForm then renders
    // it as `NumericArray[<dim>, type]` to match wolframscript.
    "NumericArray" if args.len() == 1 || args.len() == 2 => {
      let payload = &args[0];
      let dtype: Option<String> = if args.len() == 2 {
        if let Expr::String(s) = &args[1] {
          Some(s.clone())
        } else {
          None
        }
      } else {
        detect_numeric_array_dtype(payload)
      };
      match dtype {
        Some(t) => {
          return Some(Ok(Expr::FunctionCall {
            name: "NumericArray".to_string(),
            args: vec![payload.clone(), Expr::String(t)].into(),
          }));
        }
        None => {
          return Some(Ok(Expr::FunctionCall {
            name: "NumericArray".to_string(),
            args: args.to_vec().into(),
          }));
        }
      }
    }
    // CensoredDistribution[{min, max}, dist] — censored distribution
    "CensoredDistribution" if args.len() == 2 => {
      // Evaluate the underlying distribution to normalize it
      let dist = crate::evaluator::evaluate_expr_to_expr(&args[1])
        .unwrap_or_else(|_| args[1].clone());
      let bounds = crate::evaluator::evaluate_expr_to_expr(&args[0])
        .unwrap_or_else(|_| args[0].clone());
      return Some(Ok(Expr::FunctionCall {
        name: "CensoredDistribution".to_string(),
        args: vec![bounds, dist].into(),
      }));
    }
    "Names" if args.len() <= 1 => {
      // Include both user-defined names and built-in function names
      // (from functions.csv) so patterns like "List*" match builtins.
      let mut all_names: Vec<String> = crate::get_defined_names();
      for b in crate::evaluator::get_builtin_function_names() {
        if !all_names.iter().any(|n| n == b) {
          all_names.push(b.to_string());
        }
      }
      // Also include names that exist in the CSV but lack a description
      // (e.g. `ListAnimate`, `ListDeconvolve`) — they're valid built-in
      // symbols that Wolfram lists, even if we haven't implemented them.
      for b in crate::evaluator::known_wolfram_function_names() {
        if !all_names.iter().any(|n| n == b) {
          all_names.push(b.to_string());
        }
      }
      // Match wolframscript's case-insensitive alphabetical sort so
      // `Listable` sorts between `List` and `ListAnimate`, not last.
      all_names.sort_by_key(|n| n.to_lowercase());
      if args.is_empty() {
        let items: Vec<Expr> =
          all_names.into_iter().map(Expr::String).collect();
        return Some(Ok(Expr::List(items.into())));
      }
      if let Expr::String(pattern) = &args[0] {
        // Strip a leading `Global`` context — Woxi stores user symbols
        // without a context prefix, so `Global`foo` refers to the same
        // symbol as `foo`. `System`` context narrows to built-ins.
        let (effective_pattern, scope) =
          if let Some(rest) = pattern.strip_prefix("Global`") {
            (rest.to_string(), Some("Global"))
          } else if let Some(rest) = pattern.strip_prefix("System`") {
            (rest.to_string(), Some("System"))
          } else {
            (pattern.clone(), None)
          };
        let user_names: std::collections::HashSet<String> =
          crate::get_defined_names().into_iter().collect();
        // Wolfram name patterns: `*` matches any run of characters (0+);
        // `@` matches one or more lowercase letters (so `List@` matches
        // `Listable`, `Listen`, but not `List` itself).
        let regex_pattern = format!(
          "^{}$",
          effective_pattern
            .replace('.', "\\.")
            .replace('*', ".*")
            .replace('@', "[a-z]+")
        );
        let re = regex::Regex::new(&regex_pattern);
        if let Ok(re) = re {
          let items: Vec<Expr> = all_names
            .into_iter()
            .filter(|n| re.is_match(n))
            .filter(|n| match scope {
              None => true,
              Some("Global") => user_names.contains(n),
              Some("System") => !user_names.contains(n),
              _ => true,
            })
            .map(Expr::String)
            .collect();
          return Some(Ok(Expr::List(items.into())));
        }
      }
      return Some(Ok(Expr::List(vec![].into())));
    }
    "ValueQ" if args.len() == 1 => {
      // ValueQ[head[args...]] is True iff `head` has any OwnValues or
      // DownValues (matching wolframscript). For a bare symbol the same
      // rule applies. Note: this does NOT require the specific
      // call-site to match an existing rule — any definition on the head
      // is enough.
      let head_name: Option<&str> = match &args[0] {
        Expr::Identifier(sym) => Some(sym.as_str()),
        Expr::FunctionCall { name, .. } => Some(name.as_str()),
        _ => None,
      };
      if let Some(sym) = head_name {
        let has_value = ENV.with(|e| e.borrow().contains_key(sym));
        let has_func = crate::FUNC_DEFS.with(|m| m.borrow().contains_key(sym));
        // Memoized literal definitions (e.g. `Foo[5] = "five"`) live in
        // MEMO_VALUES, not FUNC_DEFS, but still count as a DownValue.
        let has_memo =
          crate::MEMO_VALUES.with(|m| m.borrow().contains_key(sym));
        return Some(Ok(Expr::Identifier(
          if has_value || has_func || has_memo {
            "True"
          } else {
            "False"
          }
          .to_string(),
        )));
      }
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    "Piecewise" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::control_flow_ast::piecewise_ast(args));
    }
    "If" if args.len() >= 2 && args.len() <= 4 => {
      let cond = match evaluate_expr_to_expr(&args[0]) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
      };
      if matches!(&cond, Expr::Identifier(s) if s == "True") {
        return Some(evaluate_expr_to_expr(&args[1]));
      } else if matches!(&cond, Expr::Identifier(s) if s == "False") {
        if args.len() >= 3 {
          return Some(evaluate_expr_to_expr(&args[2]));
        } else {
          return Some(Ok(Expr::Identifier("Null".to_string())));
        }
      } else if args.len() == 4 {
        return Some(evaluate_expr_to_expr(&args[3]));
      }
    }
    // Trace[expr] — minimal implementation: return
    // {HoldForm[original], HoldForm[evaluated]} when the two differ,
    // or {} when evaluation is idempotent, matching wolframscript.
    "Trace" if args.len() == 1 => {
      let original = args[0].clone();
      let evaluated = match crate::evaluator::evaluate_expr_to_expr(&original) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      // Compare via printed form since Expr doesn't implement PartialEq.
      let orig_str = crate::syntax::expr_to_string(&original);
      let eval_str = crate::syntax::expr_to_string(&evaluated);
      if orig_str == eval_str {
        return Some(Ok(Expr::List(vec![].into())));
      }
      let wrap = |e: Expr| Expr::FunctionCall {
        name: "HoldForm".into(),
        args: vec![e].into(),
      };
      return Some(Ok(Expr::List(
        vec![wrap(original), wrap(evaluated)].into(),
      )));
    }
    // Stack[] - return the current evaluation stack as a list of strings.
    // Exclude the outermost entry (which is the 'Stack' call itself) so that
    // a top-level 'Stack[]' returns the empty list, matching wolframscript.
    "Stack" if args.is_empty() => {
      let mut stack = crate::get_eval_stack();
      // Remove the trailing "Stack" entry pushed for this call itself.
      if stack.last().is_some_and(|s| s == "Stack") {
        stack.pop();
      }
      let items: Vec<Expr> = stack.into_iter().map(Expr::String).collect();
      return Some(Ok(Expr::List(items.into())));
    }
    _ => {}
  }
  None
}

/// Recursively strip Hold-family wrappers for `ReleaseHold`, mirroring a single
/// top-down `ReplaceAll` pass: `Hold[e…]`/`HoldComplete[e…]` and
/// `HoldForm[e]`/`HoldPattern[e]` are replaced by their content, and the
/// content is NOT re-scanned (so a nested wrapper inside released content is
/// kept). Multi-argument Hold/HoldComplete release to a `Sequence`. Non-wrapper
/// heads recurse into their parts. `Defer` is intentionally not released.
fn release_hold_rec(e: &Expr) -> Expr {
  match e {
    Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "Hold" | "HoldComplete")
        && !args.is_empty() =>
    {
      if args.len() == 1 {
        args[0].clone()
      } else {
        Expr::FunctionCall {
          name: "Sequence".to_string(),
          args: args.clone(),
        }
      }
    }
    Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "HoldForm" | "HoldPattern")
        && args.len() == 1 =>
    {
      args[0].clone()
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(release_hold_rec).collect::<Vec<_>>().into(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(release_hold_rec)
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(release_hold_rec(left)),
      right: Box::new(release_hold_rec(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(release_hold_rec(operand)),
    },
    other => other.clone(),
  }
}

/// Pick the smallest NumericArray dtype that fits every element in the
/// (possibly nested) list `e`. Currently recognises only
/// `UnsignedInteger8` — the type wolframscript uses by default for
/// integer matrices in the 0..=255 range. Returns None when the payload
/// isn't a list of suitable values.
fn detect_numeric_array_dtype(e: &Expr) -> Option<String> {
  fn walk(e: &Expr, all_uint8: &mut bool) -> bool {
    match e {
      Expr::List(items) => items.iter().all(|i| walk(i, all_uint8)),
      Expr::Integer(n) => {
        if !(0..=255).contains(n) {
          *all_uint8 = false;
        }
        true
      }
      _ => false,
    }
  }
  let mut all_uint8 = true;
  if !walk(e, &mut all_uint8) {
    return None;
  }
  if all_uint8 {
    Some("UnsignedInteger8".to_string())
  } else {
    None
  }
}

/// Helper to check if a numeric expression is positive.
fn is_positive(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n > 0,
    Expr::Real(f) => *f > 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        (*n > 0 && *d > 0) || (*n < 0 && *d < 0)
      } else {
        false
      }
    }
    _ => false,
  }
}

/// Helper to check if a numeric expression is a probability (in [0, 1]).
fn is_probability(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n == 0 || *n == 1,
    Expr::Real(f) => (0.0..=1.0).contains(f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        let val = *n as f64 / *d as f64;
        (0.0..=1.0).contains(&val)
      } else {
        false
      }
    }
    _ => false,
  }
}

/// Helper to check if a numeric expression is a positive integer.
fn is_positive_integer(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(n) if *n > 0)
}

/// Validate distribution parameters. Returns true if valid.
fn validate_distribution_params(name: &str, args: &[Expr]) -> bool {
  match name {
    "NormalDistribution" => {
      // NormalDistribution[mu, sigma] — sigma must be positive
      if args.len() == 2 {
        is_positive(&args[1])
      } else {
        args.is_empty() // NormalDistribution[] uses defaults
      }
    }
    "ExponentialDistribution" => {
      // ExponentialDistribution[lambda] — lambda must be positive
      args.len() == 1 && is_positive(&args[0])
    }
    "PoissonDistribution" => {
      // PoissonDistribution[mu] — mu must be positive
      args.len() == 1 && is_positive(&args[0])
    }
    "BernoulliDistribution" => {
      // BernoulliDistribution[p] — p must be in [0, 1]
      args.len() == 1 && is_probability(&args[0])
    }
    "BinomialDistribution" => {
      // BinomialDistribution[n, p] — n positive integer, p in [0, 1]
      args.len() == 2
        && is_positive_integer(&args[0])
        && is_probability(&args[1])
    }
    "UniformDistribution" => {
      // UniformDistribution[{a, b}] — a < b
      if args.is_empty() {
        return true;
      }
      if args.len() == 1 {
        if let Expr::List(bounds) = &args[0]
          && bounds.len() == 2
        {
          // Check a < b
          match (&bounds[0], &bounds[1]) {
            (Expr::Integer(a), Expr::Integer(b)) => a < b,
            (Expr::Real(a), Expr::Real(b)) => a < b,
            (Expr::Integer(a), Expr::Real(b)) => (*a as f64) < *b,
            (Expr::Real(a), Expr::Integer(b)) => *a < (*b as f64),
            _ => false,
          }
        } else {
          false
        }
      } else {
        false
      }
    }
    "GeometricDistribution" => {
      // GeometricDistribution[p] — p in (0, 1]
      args.len() == 1 && is_probability(&args[0]) && is_positive(&args[0])
    }
    "BenfordDistribution" => {
      // BenfordDistribution[b] — integer base b >= 2
      args.len() == 1 && matches!(&args[0], Expr::Integer(b) if *b >= 2)
    }
    "BenktanderWeibullDistribution" => {
      // BenktanderWeibullDistribution[a, b] — a > 0 and 0 < b <= 1
      args.len() == 2
        && is_positive(&args[0])
        && is_positive(&args[1])
        && is_probability(&args[1])
    }
    "SinghMaddalaDistribution" => {
      // SinghMaddalaDistribution[q, a, b] — all parameters positive
      args.len() == 3 && args.iter().all(is_positive)
    }
    "LogSeriesDistribution" => {
      // LogSeriesDistribution[theta] — theta in (0, 1)
      args.len() == 1 && is_probability(&args[0]) && is_positive(&args[0])
    }
    "NakagamiDistribution" => {
      // NakagamiDistribution[m, w] — both positive
      args.len() == 2 && is_positive(&args[0]) && is_positive(&args[1])
    }
    "LogLogisticDistribution" => {
      // LogLogisticDistribution[g, s] — both positive
      args.len() == 2 && is_positive(&args[0]) && is_positive(&args[1])
    }
    "GammaDistribution" => {
      // GammaDistribution[alpha, beta] — both positive
      args.len() == 2 && is_positive(&args[0]) && is_positive(&args[1])
    }
    "BetaDistribution" => {
      // BetaDistribution[alpha, beta] — both positive
      args.len() == 2 && is_positive(&args[0]) && is_positive(&args[1])
    }
    "ChiSquareDistribution" => {
      // ChiSquareDistribution[k] — k positive
      args.len() == 1 && is_positive(&args[0])
    }
    "StudentTDistribution" => {
      // StudentTDistribution[nu] — nu positive
      args.len() == 1 && is_positive(&args[0])
    }
    "WeibullDistribution" => {
      // WeibullDistribution[alpha, beta] — both positive
      args.len() == 2 && is_positive(&args[0]) && is_positive(&args[1])
    }
    "CauchyDistribution" => {
      // CauchyDistribution[a, b] — b positive
      args.len() == 2 && is_positive(&args[1])
    }
    "LogNormalDistribution" => {
      // LogNormalDistribution[mu, sigma] — sigma positive
      args.len() == 2 && is_positive(&args[1])
    }
    "NegativeBinomialDistribution" => {
      // NegativeBinomialDistribution[n, p] — n positive, p in (0, 1]
      args.len() == 2
        && is_positive(&args[0])
        && is_probability(&args[1])
        && is_positive(&args[1])
    }
    "HalfNormalDistribution" => {
      // HalfNormalDistribution[theta] — theta positive
      args.len() == 1 && is_positive(&args[0])
    }
    "ChiDistribution" => {
      // ChiDistribution[k] — k positive
      args.len() == 1 && is_positive(&args[0])
    }
    "FRatioDistribution" => {
      // FRatioDistribution[n, m] — both positive
      args.len() == 2 && is_positive(&args[0]) && is_positive(&args[1])
    }
    "LaplaceDistribution" => {
      // LaplaceDistribution[mu, beta] — beta positive
      if args.is_empty() {
        true
      } else {
        args.len() == 2 && is_positive(&args[1])
      }
    }
    // Recognize as a distribution but with symbolic params — assume valid
    _ => {
      // Check if it's a known distribution name
      let known = [
        "DiscreteUniformDistribution",
        "MultinormalDistribution",
        "DirichletDistribution",
        "InverseGaussianDistribution",
        "RayleighDistribution",
        "MaxwellDistribution",
        "GumbelDistribution",
        "StableDistribution",
        "TruncatedDistribution",
        "CensoredDistribution",
        "MixtureDistribution",
        "ParameterMixtureDistribution",
        "TransformedDistribution",
      ];
      known.contains(&name) || name.ends_with("Distribution")
    }
  }
}
