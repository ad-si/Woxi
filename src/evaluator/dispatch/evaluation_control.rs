use super::*;

/// Resolve an `In[…]` / `Out[…]` argument list to the input line it names.
///
/// Returns the argument as written (`None` for the bare `In[]` / `Out[]`
/// form, which means the previous line) together with the absolute line
/// number it points at: a negative argument counts back from `$Line`, a
/// non-negative one already is the line. `None` overall means the
/// reference is malformed — wolframscript's `::intm` / `::argt` message
/// has been emitted and the caller should leave the call unevaluated.
///
/// Shared by both heads so their index arithmetic and messages cannot
/// drift apart.
pub(crate) fn history_line_index(
  name: &str,
  args: &[Expr],
) -> Option<(Option<i128>, i128)> {
  let offset = match args {
    [] => None,
    // Only a machine-sized integer names a line; wolframscript rejects a
    // real, a symbol, a string or a bignum with the same message.
    [Expr::Integer(n)] if i64::try_from(*n).is_ok() => Some(*n),
    [_] => {
      crate::emit_message(&format!(
        "{name}::intm: Machine-sized integer expected at position 1 in {}.",
        expr_to_output(&unevaluated(name, args))
      ));
      return None;
    }
    _ => {
      crate::emit_message(&format!(
        "{name}::argt: {name} called with {} arguments; 0 or 1 arguments \
         are expected.",
        args.len()
      ));
      return None;
    }
  };
  let k = offset.unwrap_or(-1);
  let index = if k < 0 {
    crate::current_line().saturating_add(k)
  } else {
    k
  };
  Some((offset, index))
}

pub fn dispatch_evaluation_control(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "HoldForm" if args.len() == 1 => {
      return Some(Ok(unevaluated("HoldForm", args)));
    }
    "Hold" if !args.is_empty() => {
      return Some(Ok(unevaluated("Hold", args)));
    }
    "HoldComplete" if !args.is_empty() => {
      return Some(Ok(unevaluated("HoldComplete", args)));
    }
    // `HoldCompleteForm` is `HoldForm`'s HoldAllComplete sibling: it keeps its
    // argument unevaluated for display (`HoldCompleteForm[2 + 3]` prints as
    // `HoldCompleteForm[2 + 3]`) and, unlike `HoldForm`, does not honour a
    // wrapping `Evaluate` — see the HoldAllComplete attribute in
    // `attributes.rs`, which is what suppresses argument evaluation.
    "HoldCompleteForm" if !args.is_empty() => {
      return Some(Ok(unevaluated("HoldCompleteForm", args)));
    }
    "Unevaluated" if !args.is_empty() => {
      return Some(Ok(unevaluated("Unevaluated", args)));
    }
    // `Release` is the older spelling of ReleaseHold, still in System`.
    "ReleaseHold" | "Release" if args.len() == 1 => {
      // ReleaseHold removes Hold/HoldForm/HoldComplete/HoldPattern wrappers
      // wherever they appear (one top-down pass, like ReplaceAll — it does not
      // descend into the content it just released, so ReleaseHold[Hold[Hold[…]]]
      // keeps the inner Hold). The stripped expression is then evaluated.
      let stripped = release_hold_rec(&args[0]);
      return Some(evaluate_expr_to_expr(&stripped));
    }
    "TimeRemaining" if args.is_empty() => {
      return Some(Ok(id_expr("Infinity")));
    }
    "Out" => {
      // `Out[]` is `Out[-1]`, and a negative index counts back from the
      // current input line: `Out[-k]` is `Out[$Line - k]` (which is what
      // the `%%…` shortcuts parse to). The resolved line is looked up in
      // the session's numbered output history, so `Out[9]` / `%9` return
      // what line 9 produced.
      //
      // A line that was never evaluated — including every line in
      // script mode, where `$Line` is always 1 and no history is kept —
      // leaves the reference symbolic as `Out[k]`, with the index clamped
      // at 0, matching wolframscript.
      let Some((offset, index)) = history_line_index("Out", args) else {
        return Some(Ok(unevaluated("Out", args)));
      };
      if crate::output_history_enabled() {
        if index > 0
          && let Some(prev) = crate::get_output_at_line(index)
        {
          return Some(Ok(prev));
        }
        // Hosts that never advance `$Line` (woxi-studio, the playground)
        // accumulate no numbered history, but `%` must still reach the
        // previous cell's result.
        if offset == Some(-1)
          && let Some(prev) = crate::get_last_output()
        {
          return Some(Ok(prev));
        }
      }
      return Some(Ok(call1("Out", Expr::Integer(index.max(0)))));
    }
    "Evaluate" if args.len() == 1 => {
      return Some(Ok(args[0].clone()));
    }
    // `Evaluate[a, b, c]` returns `Sequence[a, b, c]`, which then splices
    // into the surrounding context. Matches wolframscript's
    // `Hold[Evaluate[1, 2]]` → `Hold[1, 2]`.
    "Evaluate" => {
      return Some(Ok(unevaluated("Sequence", args)));
    }
    "CategoricalDistribution" => {
      return match crate::functions::math_ast::categorical::categorical_distribution(args) {
        Ok(Some(canonical)) => Some(Ok(canonical)),
        Ok(None) => Some(Ok(unevaluated("CategoricalDistribution", args))),
        Err(e) => Some(Err(e)),
      };
    }
    "MarginalDistribution" if args.len() == 2 => {
      if let Some(r) =
        crate::functions::math_ast::categorical::categorical_marginal(
          &args[0], &args[1],
        )
      {
        return Some(r);
      }
    }
    "RegularExpression" if args.len() == 1 => {
      return Some(Ok(unevaluated("RegularExpression", args)));
    }
    // Location and scale parameters that are quantities in one unit make a
    // distribution of quantities:
    // NormalDistribution[Quantity[180, "Centimeters"], Quantity[5, …]] is
    // QuantityDistribution[NormalDistribution[180, 5], "Centimeters"].
    "NormalDistribution"
    | "UniformDistribution"
    | "WeibullDistribution"
    | "ParetoDistribution"
    | "RayleighDistribution"
      if let Some(result) = quantity_parameter_distribution(name, args) =>
    {
      return Some(Ok(result));
    }
    "UniformDistribution" if args.len() <= 1 => {
      let uni_args = if args.is_empty() {
        vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())]
      } else {
        args.to_vec()
      };
      return Some(Ok(call("UniformDistribution", uni_args)));
    }
    "NormalDistribution" => {
      let norm_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(call("NormalDistribution", norm_args)));
    }
    "ExponentialDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("ExponentialDistribution", args)));
    }
    "PoissonDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("PoissonDistribution", args)));
    }
    "BernoulliDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("BernoulliDistribution", args)));
    }
    "InverseGammaDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("InverseGammaDistribution", args)));
    }
    "GammaDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("GammaDistribution", args)));
    }
    "MultinormalDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("MultinormalDistribution", args)));
    }
    "ProductDistribution" if args.len() >= 2 => {
      return Some(Ok(unevaluated("ProductDistribution", args)));
    }
    "UniformSumDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(unevaluated("UniformSumDistribution", args)));
    }
    "BetaBinomialDistribution" if args.len() == 3 => {
      return Some(Ok(unevaluated("BetaBinomialDistribution", args)));
    }
    "BetaPrimeDistribution" if (2..=4).contains(&args.len()) => {
      return Some(Ok(unevaluated("BetaPrimeDistribution", args)));
    }
    "NoncentralStudentTDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("NoncentralStudentTDistribution", args)));
    }
    "NoncentralChiSquareDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("NoncentralChiSquareDistribution", args)));
    }
    "ExponentialPowerDistribution" if args.len() == 3 => {
      return Some(Ok(unevaluated("ExponentialPowerDistribution", args)));
    }
    "RiceDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("RiceDistribution", args)));
    }
    "MinStableDistribution" if args.len() == 3 => {
      return Some(Ok(unevaluated("MinStableDistribution", args)));
    }
    "MaxStableDistribution" if args.len() == 3 => {
      return Some(Ok(unevaluated("MaxStableDistribution", args)));
    }
    "TriangularDistribution" if args.len() <= 2 => {
      return Some(Ok(unevaluated("TriangularDistribution", args)));
    }
    "MaxwellDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("MaxwellDistribution", args)));
    }
    // The scale defaults to 1 and is written out: MarchenkoPasturDistribution[
    // l] is MarchenkoPasturDistribution[l, 1].
    "MarchenkoPasturDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated(
        "MarchenkoPasturDistribution",
        &[args[0].clone(), Expr::Integer(1)],
      )));
    }
    "MarchenkoPasturDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("MarchenkoPasturDistribution", args)));
    }
    "WignerSemicircleDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(unevaluated("WignerSemicircleDistribution", args)));
    }
    "SechDistribution" if args.len() <= 2 => {
      return Some(Ok(unevaluated("SechDistribution", args)));
    }
    "MoyalDistribution" if args.len() <= 2 => {
      return Some(Ok(unevaluated("MoyalDistribution", args)));
    }
    "BorelTannerDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("BorelTannerDistribution", args)));
    }
    "PoissonConsulDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("PoissonConsulDistribution", args)));
    }
    "SuzukiDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("SuzukiDistribution", args)));
    }
    "MeixnerDistribution" if args.len() == 4 => {
      return Some(Ok(unevaluated("MeixnerDistribution", args)));
    }
    "BenktanderGibratDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("BenktanderGibratDistribution", args)));
    }
    "GumbelDistribution" if args.len() <= 2 => {
      return Some(Ok(unevaluated("GumbelDistribution", args)));
    }
    "SkewNormalDistribution" if args.len() == 3 => {
      return Some(Ok(unevaluated("SkewNormalDistribution", args)));
    }
    "ZipfDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(unevaluated("ZipfDistribution", args)));
    }
    "BenfordDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("BenfordDistribution", args)));
    }
    "BenktanderWeibullDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("BenktanderWeibullDistribution", args)));
    }
    "SinghMaddalaDistribution" if args.len() == 3 => {
      return Some(Ok(unevaluated("SinghMaddalaDistribution", args)));
    }
    "WaringYuleDistribution" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(unevaluated("WaringYuleDistribution", args)));
    }
    "Query" if !args.is_empty() => {
      return Some(Ok(unevaluated("Query", args)));
    }
    "BetaDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("BetaDistribution", args)));
    }
    "StudentTDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("StudentTDistribution", args)));
    }
    "LogNormalDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("LogNormalDistribution", args)));
    }
    "LogisticDistribution" => {
      let logistic_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(call("LogisticDistribution", logistic_args)));
    }
    "GompertzMakehamDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("GompertzMakehamDistribution", args)));
    }
    "InverseGaussianDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("InverseGaussianDistribution", args)));
    }
    "FrechetDistribution" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(unevaluated("FrechetDistribution", args)));
    }
    "ExtremeValueDistribution" => {
      let evd_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else if args.len() == 2 {
        args.to_vec()
      } else {
        return None;
      };
      return Some(Ok(call("ExtremeValueDistribution", evd_args)));
    }
    "InverseChiSquareDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("InverseChiSquareDistribution", args)));
    }
    "ChiSquareDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("ChiSquareDistribution", args)));
    }
    "ParetoDistribution" if (2..=4).contains(&args.len()) => {
      return Some(Ok(unevaluated("ParetoDistribution", args)));
    }
    "WeibullDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("WeibullDistribution", args)));
    }
    "GeometricDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("GeometricDistribution", args)));
    }
    "LogSeriesDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("LogSeriesDistribution", args)));
    }
    "NakagamiDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("NakagamiDistribution", args)));
    }
    "LogLogisticDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("LogLogisticDistribution", args)));
    }
    "HypergeometricDistribution" if args.len() == 3 => {
      return Some(Ok(unevaluated("HypergeometricDistribution", args)));
    }
    "BinormalDistribution" if (1..=3).contains(&args.len()) => {
      return Some(Ok(unevaluated("BinormalDistribution", args)));
    }
    "CauchyDistribution" => {
      let cauchy_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(call("CauchyDistribution", cauchy_args)));
    }
    "DiscreteUniformDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("DiscreteUniformDistribution", args)));
    }
    "LaplaceDistribution" => {
      let laplace_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else if args.len() == 2 {
        args.to_vec()
      } else {
        return None;
      };
      return Some(Ok(call("LaplaceDistribution", laplace_args)));
    }
    "RayleighDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("RayleighDistribution", args)));
    }
    "NegativeBinomialDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("NegativeBinomialDistribution", args)));
    }
    "MultinomialDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("MultinomialDistribution", args)));
    }
    "NegativeMultinomialDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("NegativeMultinomialDistribution", args)));
    }
    // WienerProcess[] normalizes to WienerProcess[0, 1]
    // (wolframscript-verified).
    "WienerProcess" if args.is_empty() => {
      return Some(Ok(call(
        "WienerProcess",
        vec![Expr::Integer(0), Expr::Integer(1)],
      )));
    }
    // BrownianBridgeProcess[] and the two-point form normalize with the
    // default variance scale 1; a single argument is an arity error
    // (wolframscript-verified argtu message).
    "BrownianBridgeProcess" => {
      let is_pt = |e: &Expr| matches!(e, Expr::List(p) if p.len() == 2);
      let normalized: Vec<Expr> = match args {
        [] => vec![
          Expr::Integer(1),
          Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
          Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()),
        ],
        [p1, p2] if is_pt(p1) && is_pt(p2) => {
          vec![Expr::Integer(1), p1.clone(), p2.clone()]
        }
        [_] => {
          crate::emit_message(
            "BrownianBridgeProcess::argtu: BrownianBridgeProcess called with 1 argument; 2 or 3 arguments are expected.",
          );
          args.to_vec()
        }
        _ => args.to_vec(),
      };
      return Some(Ok(call("BrownianBridgeProcess", normalized)));
    }
    // Random-process objects are symbolic; their time slices proc[t]
    // are consumed by PDF/CDF/Mean/Variance.
    "WienerProcess"
    | "GeometricBrownianMotionProcess"
    | "OrnsteinUhlenbeckProcess"
    | "PoissonProcess"
    | "BinomialProcess"
    | "BernoulliProcess"
    | "WhiteNoiseProcess"
    | "ARProcess"
    | "MAProcess"
    | "ARIMAProcess"
    | "SARMAProcess"
    | "SARIMAProcess" => {
      return Some(Ok(unevaluated(name, args)));
    }
    // RandomFunction[proc, {t0, t1, dt}] simulates one realization of the
    // process, returned as a TemporalData object. Only WienerProcess and
    // OrnsteinUhlenbeckProcess are simulated (see
    // `timeseries_ast::random_function_ast`); any other process, or an
    // unrecognized argument shape, is left unevaluated rather than erroring.
    "RandomFunction" => {
      return Some(crate::functions::timeseries_ast::random_function_ast(args));
    }
    // DiscreteMarkovProcess and its distribution wrappers are symbolic
    // objects consumed by PDF/CDF/Mean/Variance.
    "DiscreteMarkovProcess"
    | "StationaryDistribution"
    | "FirstPassageTimeDistribution" => {
      return Some(Ok(unevaluated(name, args)));
    }
    // StateSpaceModel[{a, b, c, d}] is a symbolic control-system object:
    // it echoes unevaluated and is consumed by ObservabilityMatrix /
    // ControllabilityMatrix.
    "StateSpaceModel" => {
      return Some(Ok(unevaluated("StateSpaceModel", args)));
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
          Expr::BinaryOp { op, left, right } => {
            binop(*op, substitute(left, map), substitute(right, map))
          }
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
        return Some(Ok(call(
          "FailureDistribution",
          vec![substitute(&args[0], &map), Expr::List(new_pairs.into())],
        )));
      }
      return Some(Ok(unevaluated("FailureDistribution", args)));
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
        return Some(Ok(call1(
          "HypoexponentialDistribution",
          Expr::List(rates.into()),
        )));
      }
      return Some(Ok(unevaluated("StandbyDistribution", args)));
    }
    // The constructor never validates (wolframscript echoes even
    // non-symmetric matrices silently); Mean/Variance validate.
    "GaussianOrthogonalMatrixDistribution"
    | "GaussianUnitaryMatrixDistribution"
      if args.len() == 1 || args.len() == 2 =>
    {
      return Some(Ok(unevaluated(name, args)));
    }
    // The property (args[0]) stays held, as it is a function of the matrix
    // variable; the ensemble inside `Distributed[x, ensemble]` is evaluated
    // so that its size and scale parameters resolve.
    "MatrixPropertyDistribution" if args.len() == 2 => {
      let spec = match &args[1] {
        Expr::FunctionCall {
          name: dname,
          args: dspec,
        } if dname == "Distributed" && dspec.len() == 2 => {
          match crate::evaluator::evaluate_expr_to_expr(&dspec[1]) {
            Ok(dist) => unevaluated("Distributed", &[dspec[0].clone(), dist]),
            Err(e) => return Some(Err(e)),
          }
        }
        other => other.clone(),
      };
      return Some(Ok(unevaluated(
        "MatrixPropertyDistribution",
        &[args[0].clone(), spec],
      )));
    }
    "WishartMatrixDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("WishartMatrixDistribution", args)));
    }
    "MultivariateHypergeometricDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated(
        "MultivariateHypergeometricDistribution",
        args,
      )));
    }
    "MultivariateTDistribution" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(unevaluated("MultivariateTDistribution", args)));
    }
    "MultivariatePoissonDistribution" if args.len() == 2 => {
      return Some(Ok(unevaluated("MultivariatePoissonDistribution", args)));
    }
    "DirichletDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("DirichletDistribution", args)));
    }
    // HalfSpace[n] normalizes to HalfSpace[n, 0] (wolframscript-verified).
    "HalfSpace" if args.len() == 1 && matches!(&args[0], Expr::List(_)) => {
      return Some(Ok(call(
        "HalfSpace",
        vec![args[0].clone(), Expr::Integer(0)],
      )));
    }
    // The three centre-plus-radius-pair regions normalize to their full form
    // `head[center, {rinner, router}]`: the default shell is {1/2, 1}, a bare
    // radius pair gets the origin as its centre, and a single radius r means
    // {r/2, r} — except for `Torus`, where wolframscript leaves `Torus[r]`
    // alone. `Annulus` is the 2-D member, so its origin has two coordinates.
    // (wolframscript-verified, symbolic radii included.)
    "SphericalShell" | "Annulus" | "Torus" if args.len() <= 2 => {
      let dimension = if name == "Annulus" { 2 } else { 3 };
      let origin = || Expr::List(vec![Expr::Integer(0); dimension].into());
      let normalized = match args {
        [] => Some((
          origin(),
          Expr::List(
            vec![
              call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
              Expr::Integer(1),
            ]
            .into(),
          ),
        )),
        [Expr::List(radii)] if radii.len() == 2 => {
          Some((origin(), args[0].clone()))
        }
        [r] if name != "Torus" && !matches!(r, Expr::List(_)) => {
          let half = crate::evaluator::evaluate_expr_to_expr(&div2(
            r.clone(),
            Expr::Integer(2),
          ));
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
        Some((center, radii)) => call(name, vec![center, radii]),
        None => unevaluated(name, args),
      }));
    }
    // A region primitive called with no arguments is not left as written:
    // wolframscript fills in the default its documentation names, so
    // `Sphere[]` *evaluates to* `Sphere[{0, 0, 0}]` and is one level deeper
    // than the bare call. (wolframscript-verified for every head listed.)
    "Sphere" | "Ball" | "Cuboid" | "Cylinder" | "Cone" | "Tube"
    | "Parallelepiped" | "Pyramid" | "Prism" | "Hexahedron"
      if args.is_empty() =>
    {
      let point = |coords: [i128; 3]| {
        Expr::List(coords.map(Expr::Integer).to_vec().into())
      };
      let points = |coords: &[[i128; 3]]| {
        Expr::List(coords.iter().map(|c| point(*c)).collect::<Vec<_>>().into())
      };
      let defaults = match name {
        // A single corner/centre at the origin; the size stays implicit.
        "Sphere" | "Ball" | "Cuboid" => vec![point([0, 0, 0])],
        // The unit-radius z-axis segment from -1 to 1.
        "Cylinder" | "Cone" | "Tube" => {
          vec![points(&[[0, 0, -1], [0, 0, 1]])]
        }
        // A corner plus the three edge vectors spanning it.
        "Parallelepiped" => {
          vec![point([0, 0, 0]), points(&[[1, 0, 0], [1, 1, 0], [1, 1, 1]])]
        }
        // A square base of side 2 centred on the origin, apex at z = 1.
        "Pyramid" => vec![points(&[
          [-1, -1, 0],
          [1, -1, 0],
          [1, 1, 0],
          [-1, 1, 0],
          [0, 0, 1],
        ])],
        // The unit triangular prism: two right triangles, base then top.
        "Prism" => vec![points(&[
          [0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 1],
          [0, 1, 1],
        ])],
        // The unit cube's eight vertices, base face then top face.
        _ => vec![points(&[
          [0, 0, 0],
          [1, 0, 0],
          [1, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 1],
          [1, 1, 1],
          [0, 1, 1],
        ])],
      };
      return Some(Ok(call(name, defaults)));
    }
    // StadiumShape[] / StadiumShape[r] normalize to the full form with the
    // default endpoints {{-1, 0}, {1, 0}} (the 2-D capsule analog).
    "StadiumShape" if args.len() <= 2 => {
      let default_points = || {
        let pt =
          |x: i128| Expr::List(vec![Expr::Integer(x), Expr::Integer(0)].into());
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
        Some((points, r)) => call("StadiumShape", vec![points, r]),
        None => unevaluated("StadiumShape", args),
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
        Some((points, r)) => call("CapsuleShape", vec![points, r]),
        None => unevaluated("CapsuleShape", args),
      }));
    }
    "ArcSinDistribution" if args.is_empty() => {
      // Default: ArcSinDistribution[{0, 1}]
      return Some(Ok(call(
        "ArcSinDistribution",
        vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())],
      )));
    }
    "ArcSinDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("ArcSinDistribution", args)));
    }
    "HalfNormalDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("HalfNormalDistribution", args)));
    }
    "ChiDistribution" if args.len() == 1 => {
      return Some(Ok(unevaluated("ChiDistribution", args)));
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
      return Some(Ok(call("StableDistribution", canonical_args)));
    }
    // DistributionParameterQ[dist] — test if a distribution's parameters are valid
    "DistributionParameterQ" if args.len() == 1 => {
      if let Expr::FunctionCall {
        name: dist_name,
        args: dist_args,
      } = &args[0]
      {
        let result = validate_distribution_params(dist_name, dist_args);
        return Some(Ok(bool_expr(result)));
      }
      // Not a recognized distribution — return unevaluated
      return Some(Ok(unevaluated("DistributionParameterQ", args)));
    }
    // DistributionParameterAssumptions[dist] — conditions on the parameters
    // under which `dist` is a valid distribution
    "DistributionParameterAssumptions" if args.len() == 1 => {
      if let Expr::FunctionCall {
        name: dist_name,
        args: dist_args,
      } = &args[0]
        && let Some(conds) =
          distribution_parameter_assumptions(dist_name, dist_args)
      {
        let result = if conds.len() == 1 {
          conds.into_iter().next().unwrap()
        } else {
          call("And", conds)
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
      return Some(Ok(unevaluated("DistributionParameterAssumptions", args)));
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
                return Some(Ok(unevaluated("ByteArray", args)));
              }
            }
          }
          let b64 = engine.encode(&raw_bytes);
          return Some(Ok(call1("ByteArray", Expr::String(b64))));
        }
        Expr::String(s) => {
          // Validate base64 string, then store as-is
          use base64::Engine;
          let engine = base64::engine::general_purpose::STANDARD;
          if engine.decode(s).is_ok() {
            return Some(Ok(call1("ByteArray", Expr::String(s.clone()))));
          }
          crate::emit_message(
            "ByteArray::lend: The argument at position 1 in ByteArray[...] should be a vector of unsigned byte values or a Base64-encoded string.",
          );
          return Some(Ok(unevaluated("ByteArray", args)));
        }
        _ => {
          crate::emit_message(&format!(
            "ByteArray::lend: The argument at position 1 in ByteArray[{}] should be a vector of unsigned byte values or a Base64-encoded string.",
            expr_to_string(&args[0])
          ));
          return Some(Ok(unevaluated("ByteArray", args)));
        }
      }
    }
    // RawArray[type, list] — the legacy spelling of NumericArray, with the
    // arguments the other way round. `RawArray["UnsignedInteger8", data] ===
    // NumericArray[data, "UnsignedInteger8"]`, and a Demonstrations texture
    // stored as `Image[CompressedData[…]]` decompresses to this form.
    "RawArray" if args.len() == 2 => {
      return dispatch_evaluation_control(
        "NumericArray",
        &[args[1].clone(), args[0].clone()],
      );
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
          return Some(Ok(call(
            "NumericArray",
            vec![payload.clone(), Expr::String(t)],
          )));
        }
        None => {
          return Some(Ok(unevaluated("NumericArray", args)));
        }
      }
    }
    // NumericArray[list, type, "ClipAndCoerce" | "ClipAndRound"]: values
    // outside the type's range are clipped to it; "ClipAndRound" also
    // rounds reals to the nearest integer for an integer type.
    "NumericArray"
      if args.len() == 3
        && matches!(&args[1], Expr::String(_))
        && matches!(&args[2], Expr::String(m) if m == "ClipAndCoerce" || m == "ClipAndRound") =>
    {
      let (Expr::String(ty), Expr::String(method)) = (&args[1], &args[2])
      else {
        return None;
      };
      let Some(data) =
        clip_numeric_array_data(&args[0], ty, method == "ClipAndRound")
      else {
        return Some(Ok(unevaluated("NumericArray", args)));
      };
      return Some(Ok(call(
        "NumericArray",
        vec![data, Expr::String(ty.clone())],
      )));
    }
    // CensoredDistribution[{min, max}, dist] — censored distribution
    "CensoredDistribution" if args.len() == 2 => {
      // Evaluate the underlying distribution to normalize it
      let dist = crate::evaluator::evaluate_expr_to_expr(&args[1])
        .unwrap_or_else(|_| args[1].clone());
      let bounds = crate::evaluator::evaluate_expr_to_expr(&args[0])
        .unwrap_or_else(|_| args[0].clone());
      return Some(Ok(call("CensoredDistribution", vec![bounds, dist])));
    }
    // Names["pattern", opts] with IgnoreCase -> True | False and
    // ResolveContextAliases -> True | False.
    "Names" if args.len() >= 2 && matches!(&args[0], Expr::String(_)) => {
      let Expr::String(pattern) = &args[0] else {
        return None;
      };
      let mut ignore_case = false;
      let mut resolve_aliases = true;
      let mut spelling_correction = false;
      for opt in &args[1..] {
        match crate::functions::expr_form::rule_parts(opt) {
          Some((Expr::Identifier(n), Expr::Identifier(flag), _))
            if (flag == "True" || flag == "False")
              && (n == "IgnoreCase"
                || n == "ResolveContextAliases"
                || n == "SpellingCorrection") =>
          {
            match n.as_str() {
              "IgnoreCase" => ignore_case = flag == "True",
              "SpellingCorrection" => spelling_correction = flag == "True",
              _ => resolve_aliases = flag == "True",
            }
          }
          _ => return Some(Ok(unevaluated("Names", args))),
        }
      }
      // SpellingCorrection -> True: a literal name also matches the names
      // one edit (insertion, deletion, substitution or transposition of
      // adjacent characters) away from it.
      let names = if spelling_correction && !pattern.contains(['*', '@']) {
        let fold = |s: &str| {
          if ignore_case {
            s.to_lowercase()
          } else {
            s.to_string()
          }
        };
        let target: Vec<char> = fold(pattern).chars().collect();
        crate::evaluator::contexts::names_matching_with(
          "*",
          ignore_case,
          resolve_aliases,
        )
        .into_iter()
        .filter(|n| {
          let candidate: Vec<char> = fold(n).chars().collect();
          within_one_edit(&target, &candidate)
        })
        .collect()
      } else {
        crate::evaluator::contexts::names_matching_with(
          pattern,
          ignore_case,
          resolve_aliases,
        )
      };
      return Some(Ok(Expr::List(
        names.into_iter().map(Expr::String).collect(),
      )));
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
      all_names.sort_by_key(|n| crate::evaluator::contexts::name_sort_key(n));
      if args.is_empty() {
        let items: Vec<Expr> =
          all_names.into_iter().map(Expr::String).collect();
        return Some(Ok(Expr::List(items.into())));
      }
      if let Expr::String(pattern) = &args[0] {
        let names = crate::evaluator::contexts::names_matching(pattern);
        return Some(Ok(Expr::List(
          names.into_iter().map(Expr::String).collect(),
        )));
      }
      return Some(Ok(Expr::List(vec![].into())));
    }
    // ValueQ[expr, Method -> m]: "TrialEvaluation" asks whether evaluating
    // expr changes it; "SymbolDefinitionsPresent" is the default test.
    "ValueQ"
      if args.len() == 2
        && let Expr::Rule {
          pattern,
          replacement,
        } = &args[1]
        && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Method")
        && let Expr::String(method) = replacement.as_ref()
        && matches!(
          method.as_str(),
          "TrialEvaluation" | "SymbolDefinitionsPresent"
        ) =>
    {
      if method == "SymbolDefinitionsPresent" {
        return dispatch_evaluation_control("ValueQ", &args[..1]);
      }
      let evaluated = match evaluate_expr_to_expr(&args[0]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      return Some(Ok(bool_expr(
        !crate::evaluator::pattern_matching::expr_equal(&args[0], &evaluated),
      )));
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
        return Some(Ok(bool_expr(has_value || has_func || has_memo)));
      }
      return Some(Ok(bool_expr(false)));
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
        }
        return Some(Ok(null_expr()));
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
      let orig_str = expr_to_string(&original);
      let eval_str = expr_to_string(&evaluated);
      if orig_str == eval_str {
        return Some(Ok(Expr::List(vec![].into())));
      }
      let wrap = |e: Expr| call1("HoldForm", e);
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
    Expr::BinaryOp { op, left, right } => {
      binop(*op, release_hold_rec(left), release_hold_rec(right))
    }
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
/// `QuantityDistribution[dist[magnitudes…], unit]` for a distribution whose
/// dimensional parameters (location and scale; `UniformDistribution`'s
/// bounds) are quantities in one unit and whose shape parameters are plain.
fn quantity_parameter_distribution(name: &str, args: &[Expr]) -> Option<Expr> {
  let is_quantity = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "Quantity");
  // The positions of the dimensional parameters.
  let dimensional: &[usize] = match (name, args.len()) {
    ("NormalDistribution", 2) => &[0, 1],
    ("UniformDistribution", 1) => {
      let Expr::List(bounds) = &args[0] else {
        return None;
      };
      let (magnitudes, unit) = common_unit_magnitudes(bounds)?;
      return Some(call(
        "QuantityDistribution",
        vec![call(name, vec![Expr::List(magnitudes.into())]), unit],
      ));
    }
    ("WeibullDistribution", 2) => &[1],
    ("ParetoDistribution", 2) => &[0],
    ("RayleighDistribution", 1) => &[0],
    _ => return None,
  };
  let picked: Vec<Expr> =
    dimensional.iter().map(|&i| args[i].clone()).collect();
  let (magnitudes, unit) = common_unit_magnitudes(&picked)?;
  let mut new_args = args.to_vec();
  for (k, &i) in dimensional.iter().enumerate() {
    new_args[i] = magnitudes[k].clone();
  }
  if new_args.iter().any(is_quantity) {
    return None;
  }
  let inner = crate::evaluator::evaluate_function_call_ast(name, &new_args)
    .unwrap_or_else(|_| call(name, new_args));
  Some(call("QuantityDistribution", vec![inner, unit]))
}

/// The magnitudes of `args` when every one is a `Quantity` in the same unit,
/// together with that unit.
fn common_unit_magnitudes(args: &[Expr]) -> Option<(Vec<Expr>, Expr)> {
  let mut unit: Option<&Expr> = None;
  let mut magnitudes = Vec::with_capacity(args.len());
  for a in args {
    let Expr::FunctionCall { name, args: q } = a else {
      return None;
    };
    if name != "Quantity" || q.len() != 2 {
      return None;
    }
    match unit {
      None => unit = Some(&q[1]),
      Some(u) if expr_to_string(u) == expr_to_string(&q[1]) => {}
      Some(_) => return None,
    }
    magnitudes.push(q[0].clone());
  }
  Some((magnitudes, unit?.clone()))
}

fn is_positive_integer(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(n) if *n > 0)
}

/// Validate distribution parameters. Returns true if valid.
/// Parameter constraints of the common continuous distributions: `true`
/// means the parameter must be a positive real, `false` any real.
fn distribution_parameter_kinds(
  name: &str,
  n: usize,
) -> Option<&'static [bool]> {
  Some(match (name, n) {
    ("NormalDistribution" | "LogNormalDistribution", 2) => &[false, true],
    (
      "CauchyDistribution" | "LaplaceDistribution" | "LogisticDistribution",
      2,
    ) => &[false, true],
    ("GumbelDistribution" | "LevyDistribution", 2) => &[false, true],
    ("StudentTDistribution", 3) => &[false, true, true],
    (
      "GammaDistribution"
      | "InverseGammaDistribution"
      | "WeibullDistribution"
      | "BetaDistribution"
      | "ParetoDistribution"
      | "FrechetDistribution"
      | "NakagamiDistribution",
      2,
    ) => &[true, true],
    (
      "ExponentialDistribution"
      | "PoissonDistribution"
      | "ChiSquareDistribution"
      | "RayleighDistribution"
      | "HalfNormalDistribution"
      | "ChiDistribution"
      | "StudentTDistribution",
      1,
    ) => &[true],
    _ => return None,
  })
}

/// `DistributionParameterAssumptions` conditions, one per parameter.
fn distribution_parameter_assumptions(
  name: &str,
  args: &[Expr],
) -> Option<Vec<Expr>> {
  let kinds = distribution_parameter_kinds(name, args.len())?;
  Some(
    kinds
      .iter()
      .zip(args)
      .map(|(positive, p)| {
        if *positive {
          call("Greater", vec![p.clone(), Expr::Integer(0)])
        } else {
          call("Element", vec![p.clone(), id_expr("Reals")])
        }
      })
      .collect(),
  )
}

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

/// The payload of `NumericArray[data, ty, "Clip…"]`: every number clipped
/// to the range of `ty` and stored as that type stores it (integers for the
/// integer types, single- or double-precision reals for the real ones).
/// Reals become integers only when `round` is set. `None` for an unknown
/// type or a non-numeric element.
fn clip_numeric_array_data(data: &Expr, ty: &str, round: bool) -> Option<Expr> {
  let int_range: Option<(i128, i128)> = match ty {
    "UnsignedInteger8" => Some((0, u8::MAX as i128)),
    "UnsignedInteger16" => Some((0, u16::MAX as i128)),
    "UnsignedInteger32" => Some((0, u32::MAX as i128)),
    "UnsignedInteger64" => Some((0, u64::MAX as i128)),
    "Integer8" => Some((i8::MIN as i128, i8::MAX as i128)),
    "Integer16" => Some((i16::MIN as i128, i16::MAX as i128)),
    "Integer32" => Some((i32::MIN as i128, i32::MAX as i128)),
    "Integer64" => Some((i64::MIN as i128, i64::MAX as i128)),
    "Real32" | "Real64" => None,
    _ => return None,
  };
  fn walk(
    e: &Expr,
    int_range: Option<(i128, i128)>,
    single: bool,
    round: bool,
  ) -> Option<Expr> {
    if let Expr::List(items) = e {
      let out: Option<Vec<Expr>> = items
        .iter()
        .map(|x| walk(x, int_range, single, round))
        .collect();
      return Some(Expr::List(out?.into()));
    }
    if let Some((lo, hi)) = int_range {
      let n: i128 = match e {
        Expr::Integer(n) => *n,
        Expr::BigInteger(b) => {
          use num_traits::Signed;
          if b.is_negative() { lo } else { hi }
        }
        Expr::Real(f) if round && f.is_finite() => {
          let r = f.round_ties_even();
          if r < lo as f64 {
            lo
          } else if r > hi as f64 {
            hi
          } else {
            r as i128
          }
        }
        _ => return None,
      };
      Some(Expr::Integer(n.clamp(lo, hi)))
    } else {
      let v = crate::functions::math_ast::try_eval_to_f64(e)?;
      if v.is_nan() {
        return None;
      }
      let max = if single { f32::MAX as f64 } else { f64::MAX };
      let clipped = v.clamp(-max, max);
      Some(Expr::Real(if single {
        clipped as f32 as f64
      } else {
        clipped
      }))
    }
  }
  walk(data, int_range, ty == "Real32", round)
}

/// Whether the optimal-string-alignment distance between a and b is at most
/// one: equal, or one insertion, deletion, substitution or transposition of
/// adjacent characters apart.
fn within_one_edit(a: &[char], b: &[char]) -> bool {
  if a == b {
    return true;
  }
  let (la, lb) = (a.len(), b.len());
  if la.abs_diff(lb) > 1 {
    return false;
  }
  let prefix = a.iter().zip(b).take_while(|(x, y)| x == y).count();
  match la.cmp(&lb) {
    std::cmp::Ordering::Equal => {
      a[prefix + 1..] == b[prefix + 1..]
        || (prefix + 1 < la
          && a[prefix] == b[prefix + 1]
          && a[prefix + 1] == b[prefix]
          && a[prefix + 2..] == b[prefix + 2..])
    }
    std::cmp::Ordering::Less => a[prefix..] == b[prefix + 1..],
    std::cmp::Ordering::Greater => a[prefix + 1..] == b[prefix..],
  }
}
