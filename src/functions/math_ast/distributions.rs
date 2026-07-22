#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, bool_expr, expr_to_string,
  unevaluated,
};

/// Helper to build a binary operation expression
fn binop(op: BinaryOperator, left: Expr, right: Expr) -> Expr {
  Expr::BinaryOp {
    op,
    left: Box::new(left),
    right: Box::new(right),
  }
}

fn plus(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Plus, a, b)
}

fn times(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Times, a, b)
}

fn divide(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Divide, a, b)
}

fn power(base: Expr, exp: Expr) -> Expr {
  binop(BinaryOperator::Power, base, exp)
}

fn minus(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Minus, a, b)
}

fn sqrt(a: Expr) -> Expr {
  make_sqrt(a)
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn factorial(a: Expr) -> Expr {
  call("Factorial", vec![a])
}

fn gamma(z: Expr) -> Expr {
  call("Gamma", vec![z])
}

fn e() -> Expr {
  Expr::Constant("E".to_string())
}

fn pi() -> Expr {
  Expr::Constant("Pi".to_string())
}

fn neg(a: Expr) -> Expr {
  Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand: Box::new(a),
  }
}

fn infinity() -> Expr {
  Expr::Identifier("Infinity".to_string())
}

fn neg_infinity() -> Expr {
  neg(infinity())
}

fn indeterminate() -> Expr {
  Expr::Identifier("Indeterminate".to_string())
}

fn int(n: i128) -> Expr {
  Expr::Integer(n)
}

fn piecewise(pairs: Vec<(Expr, Expr)>, default: Expr) -> Expr {
  let cases = Expr::List(
    pairs
      .into_iter()
      .map(|(val, cond)| Expr::List(vec![val, cond].into()))
      .collect(),
  );
  call("Piecewise", vec![cases, default])
}

fn comparison(left: Expr, op: ComparisonOp, right: Expr) -> Expr {
  Expr::Comparison {
    operands: vec![left, right],
    operators: vec![op],
  }
}

fn comparison3(
  a: Expr,
  op1: ComparisonOp,
  b: Expr,
  op2: ComparisonOp,
  c: Expr,
) -> Expr {
  Expr::Comparison {
    operands: vec![a, b, c],
    operators: vec![op1, op2],
  }
}

fn eval(expr: Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// `ErlangDistribution[k, λ]` is identical to `GammaDistribution[k, 1/λ]`
/// (Erlang parameterises by rate, Gamma by scale). Return the equivalent
/// Gamma parameters `{k, 1/λ}` so every distribution property reuses the Gamma
/// machinery and matches wolframscript.
pub fn erlang_gamma_dargs(
  dargs: &[Expr],
) -> Result<Vec<Expr>, InterpreterError> {
  let inv_lambda = eval(divide(int(1), dargs[1].clone()))?;
  Ok(vec![dargs[0].clone(), inv_lambda])
}

/// PDF[dist, x] - Probability density function
/// PDF[dist] - Pure function form (returns unevaluated for now)
pub fn pdf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PDF expects 1 or 2 arguments".into(),
    ));
  }

  let dist = &args[0];

  // PDF[DiscreteMarkovProcess[...][t], x] — the state distribution after
  // t steps, as a Boole sum.
  if args.len() == 2
    && let Expr::CurriedCall { func, args: targs } = dist
    && let Expr::FunctionCall { name, args: dargs } = func.as_ref()
    && targs.len() == 1
  {
    if name == "DiscreteMarkovProcess" {
      return dmp_step_pdf(dargs, &targs[0], &args[1])
        .map(|r| r.unwrap_or_else(|| unevaluated("PDF", args)));
    }
    if let Some(slice) = process_slice_distribution(name, dargs, &targs[0]) {
      return pdf_ast(&[slice, args[1].clone()]);
    }
  }

  // Extract distribution name and parameters
  let (dist_name, dargs) = match dist {
    Expr::FunctionCall { name, args: dargs } => {
      (name.as_str(), dargs.as_slice())
    }
    _ => {
      return Ok(unevaluated("PDF", args));
    }
  };

  // PDF[StationaryDistribution[DiscreteMarkovProcess[...]], x].
  if dist_name == "StationaryDistribution"
    && args.len() == 2
    && dargs.len() == 1
    && let Expr::FunctionCall { name, args: mp } = &dargs[0]
    && name == "DiscreteMarkovProcess"
  {
    return dmp_stationary_pdf(mp, &args[1])
      .map(|r| r.unwrap_or_else(|| unevaluated("PDF", args)));
  }

  if args.len() == 1 {
    // PDF[dist] - return unevaluated for now
    return Ok(unevaluated("PDF", args));
  }

  // PDF[MixtureDistribution[{w1, …}, {d1, …}], x] is the weight-normalized sum
  // of the component PDFs: Σ w_i PDF[d_i, x] / Σ w_i.
  if dist_name == "MixtureDistribution" && dargs.len() == 2 {
    let x = args[1].clone();
    return Ok(
      super::statistics::mixture_weighted_component_quantity(dargs, |d| {
        pdf_ast(&[d.clone(), x.clone()])
      })?
      .unwrap_or_else(|| unevaluated("PDF", args)),
    );
  }

  // ErlangDistribution[k, λ] == GammaDistribution[k, 1/λ]
  if dist_name == "ErlangDistribution" && dargs.len() == 2 {
    let gamma = call("GammaDistribution", erlang_gamma_dargs(dargs)?);
    return pdf_ast(&[gamma, args[1].clone()]);
  }

  // Thread a univariate PDF over a list of points (discrete PDFs otherwise leak
  // the list into a Piecewise condition). Multivariate distributions take a
  // list as a single point, so they are excluded.
  if let Expr::List(xs) = &args[1]
    && !is_multivariate_distribution(dist_name)
  {
    let results: Result<Vec<Expr>, InterpreterError> = xs
      .iter()
      .map(|xi| pdf_ast(&[args[0].clone(), xi.clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let x = args[1].clone();

  match dist_name {
    "ProbabilityDistribution" => pdf_probability_distribution(dargs, x),
    "NormalDistribution" => pdf_normal(dargs, x),
    "MultinormalDistribution" => pdf_multinormal(dargs, x),
    "ProductDistribution" => pdf_product_distribution(dargs, x),
    "DataDistribution" => match histogram_pdf_cdf(dargs, &x, false)
      .or_else(|| data_distribution_pdf_cdf(dargs, &x, false).map(Ok))
    {
      Some(v) => v,
      None => Ok(call("PDF", vec![unevaluated("DataDistribution", dargs), x])),
    },
    "UniformDistribution" => pdf_uniform(dargs, x),
    "UniformSumDistribution" => pdf_uniform_sum(dargs, x),
    "BetaBinomialDistribution" => pdf_beta_binomial(dargs, x),
    "BetaPrimeDistribution" => pdf_beta_prime(dargs, x),
    "NoncentralChiSquareDistribution" => pdf_noncentral_chi_square(dargs, x),
    "ExponentialPowerDistribution" => pdf_exponential_power(dargs, x),
    "RiceDistribution" => pdf_rice(dargs, x),
    "MinStableDistribution" => pdf_min_stable(dargs, x),
    "MaxStableDistribution" => pdf_max_stable(dargs, x),
    "TriangularDistribution" => pdf_triangular(dargs, x),
    "MaxwellDistribution" => pdf_maxwell(dargs, x),
    "BirnbaumSaundersDistribution" => pdf_birnbaum_saunders(dargs, x),
    "LevyDistribution" => pdf_levy(dargs, x),
    "LindleyDistribution" => pdf_lindley(dargs, x),
    "WignerSemicircleDistribution" => pdf_wigner_semicircle(dargs, x),
    "SechDistribution" => pdf_sech(dargs, x),
    "MoyalDistribution" => pdf_moyal(dargs, x),
    "BorelTannerDistribution" => pdf_borel_tanner(dargs, x),
    "BenktanderGibratDistribution" => pdf_benktander_gibrat(dargs, x),
    "GumbelDistribution" => pdf_gumbel(dargs, x),
    "SkewNormalDistribution" => pdf_skew_normal(dargs, x),
    "ZipfDistribution" => pdf_zipf(dargs, x),
    "BenfordDistribution" => pdf_benford(dargs, x),
    "BenktanderWeibullDistribution" => pdf_benktander_weibull(dargs, x),
    "ExponentialDistribution" => pdf_exponential(dargs, x),
    "PoissonDistribution" => pdf_poisson(dargs, x),
    "PoissonConsulDistribution" => pdf_poisson_consul(dargs, x),
    "MeixnerDistribution" => pdf_meixner(dargs, x),
    "LogGammaDistribution" => pdf_loggamma(dargs, x),
    "SkellamDistribution" => pdf_skellam(dargs, x),
    "HypoexponentialDistribution" => pdf_hypoexponential(dargs, x),
    "CoxianDistribution" => pdf_coxian(dargs, x),
    "HyperexponentialDistribution" => pdf_hyperexponential(dargs, x),
    "VonMisesDistribution" => pdf_vonmises(dargs, x),
    "BeniniDistribution" => pdf_benini(dargs, x),
    "HotellingTSquareDistribution" => pdf_hotelling(dargs, x),
    "TukeyLambdaDistribution" => tukey_lambda_pdf_cdf(dargs, x, true),
    "TsallisQGaussianDistribution" => tsallis_qgaussian_pdf(dargs, x),
    "VarianceGammaDistribution" => variance_gamma_pdf(dargs, x),
    "HoytDistribution" => hoyt_pdf(dargs, x),
    "FailureDistribution" => pdf_failure_distribution(dargs, x),
    "FirstPassageTimeDistribution" => pdf_first_passage(dargs, x),
    "BernoulliDistribution" => pdf_bernoulli(dargs, x),
    "BinomialDistribution" => pdf_binomial(dargs, x),
    "HypergeometricDistribution" => pdf_hypergeometric(dargs, x),
    "BinormalDistribution" => pdf_binormal(dargs, x),
    "InverseGammaDistribution" => pdf_inverse_gamma(dargs, x),
    "GammaDistribution" => pdf_gamma(dargs, x),
    "BetaDistribution" => pdf_beta(dargs, x),
    "KumaraswamyDistribution" => pdf_kumaraswamy(dargs, x),
    "PowerDistribution" => pdf_power(dargs, x),
    "PERTDistribution" => pdf_pert(dargs, x),
    "StudentTDistribution" => pdf_student_t(dargs, x),
    "LogNormalDistribution" => pdf_lognormal(dargs, x),
    "ChiSquareDistribution" => pdf_chi_square(dargs, x),
    "ParetoDistribution" => pdf_pareto(dargs, x),
    "WeibullDistribution" => pdf_weibull(dargs, x),
    "GeometricDistribution" => pdf_geometric(dargs, x),
    "LogSeriesDistribution" => pdf_log_series(dargs, x),
    "NakagamiDistribution" => pdf_nakagami(dargs, x),
    "LogLogisticDistribution" => pdf_log_logistic(dargs, x),
    "CauchyDistribution" => pdf_cauchy(dargs, x),
    "DiscreteUniformDistribution" => pdf_discrete_uniform(dargs, x),
    "LaplaceDistribution" => pdf_laplace(dargs, x),
    "RayleighDistribution" => pdf_rayleigh(dargs, x),
    "MultinomialDistribution" => pdf_multinomial(dargs, x),
    "NegativeMultinomialDistribution" => pdf_negative_multinomial(dargs, x),
    "DirichletDistribution" => pdf_dirichlet(dargs, x),
    "NegativeBinomialDistribution" => pdf_negative_binomial(dargs, x),
    "MultivariatePoissonDistribution" => pdf_multivariate_poisson(dargs, x),
    "HalfNormalDistribution" => pdf_half_normal(dargs, x),
    "ChiDistribution" => pdf_chi(dargs, x),
    "LogisticDistribution" => pdf_logistic(dargs, x),
    "InverseChiSquareDistribution" => pdf_inverse_chi_square(dargs, x),
    "FrechetDistribution" => pdf_frechet(dargs, x),
    "ExtremeValueDistribution" => pdf_extreme_value(dargs, x),
    "GompertzMakehamDistribution" => pdf_gompertz_makeham(dargs, x),
    "InverseGaussianDistribution" => pdf_inverse_gaussian(dargs, x),
    "StableDistribution" => pdf_stable(dargs, x),
    "ArcSinDistribution" => pdf_arcsin(dargs, x),
    "PascalDistribution" => pdf_pascal(dargs, x),
    "DagumDistribution" => pdf_dagum(dargs, x),
    "SinghMaddalaDistribution" => pdf_singh_maddala(dargs, x),
    "HyperbolicDistribution" => pdf_hyperbolic(dargs, x),
    "NoncentralFRatioDistribution" => pdf_noncentral_f(dargs, x),
    "FRatioDistribution" => pdf_f_ratio(dargs, x),
    "WaringYuleDistribution" => pdf_waring_yule(dargs, x),
    "JohnsonDistribution" => pdf_johnson(dargs, x),
    _ => Ok(unevaluated("PDF", args)),
  }
}

/// `SurvivalFunction[dist, x] = 1 - CDF[dist, x]`.
pub fn survival_function_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "SurvivalFunction expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 1 {
    // SurvivalFunction[dist] — symbolic pure form, leave unevaluated.
    return Ok(unevaluated("SurvivalFunction", args));
  }
  // FailureDistribution has its own survival shape:
  // Piecewise[{{1, t < 0}}, 1 - cdfvalue] (wolframscript-verified).
  if let Expr::FunctionCall { name, args: dargs } = &args[0]
    && name == "FailureDistribution"
  {
    if let Some((value, _)) = failure_distribution_cdf_value(dargs, &args[1])? {
      let complement =
        call("Plus", vec![int(1), call("Times", vec![int(-1), value])]);
      let below = comparison(args[1].clone(), ComparisonOp::Less, int(0));
      return eval(piecewise(vec![(int(1), below)], complement));
    }
    return Ok(unevaluated("SurvivalFunction", args));
  }
  let cdf = cdf_ast(args)?;

  // A Piecewise CDF complements piece by piece (with the pieces folded
  // individually — evaluating the assembled Piecewise would mangle the
  // exponentials), matching wolframscript's
  // Piecewise[{{E^(-(a*x)), x >= 0}}, 1] instead of 1 - Piecewise[...].
  if let Expr::FunctionCall { name, args: pargs } = &cdf
    && name == "Piecewise"
    && !pargs.is_empty()
    && let Expr::List(pieces) = &pargs[0]
  {
    let mut new_pieces: Vec<Expr> = Vec::with_capacity(pieces.len());
    let mut all_pairs = true;
    for piece in pieces {
      if let Expr::List(pair) = piece
        && pair.len() == 2
      {
        let val = eval(minus(int(1), pair[0].clone()))?;
        new_pieces.push(Expr::List(vec![val, pair[1].clone()].into()));
      } else {
        all_pairs = false;
        break;
      }
    }
    if all_pairs {
      let default = pargs.get(1).cloned().unwrap_or(int(0));
      let new_default = eval(minus(int(1), default))?;
      return Ok(call(
        "Piecewise",
        vec![Expr::List(new_pieces.into()), new_default],
      ));
    }
  }

  // Erfc-based CDF (normal family): 1 - Erfc[z]/2 = Erfc[-z]/2.
  // Simplify normalizes the negated argument the way wolframscript
  // prints it: (-m + x)/(Sqrt[2]*s) rather than -((m - x)/(Sqrt[2]*s)).
  if let Some(z) = match_erfc_half(&cdf) {
    let neg_z = eval(call("Simplify", vec![times(int(-1), z)]))?;
    return Ok(divide(call("Erfc", vec![neg_z]), int(2)));
  }

  eval(minus(int(1), cdf))
}

/// `HazardFunction[dist, x] = PDF[dist, x] / SurvivalFunction[dist, x]`.
///
/// Supported for distributions whose per-piece ratio simplifies to
/// wolframscript's closed form (Exponential, Pareto) plus the standard
/// normal. Other distributions stay unevaluated rather than risking a
/// differently-shaped (though equivalent) answer.
pub fn hazard_function_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("HazardFunction", args);
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let (dist_name, dargs) = match &args[0] {
    Expr::FunctionCall { name, args } => (name.as_str(), args.as_slice()),
    _ => return Ok(unevaluated(args)),
  };
  let x = args[1].clone();

  // Standard normal: Sqrt[2/Pi]/(E^(x^2/2)*Erfc[x/Sqrt[2]])
  let is_std_normal = dist_name == "NormalDistribution"
    && (dargs.is_empty()
      || (dargs.len() == 2
        && matches!(&dargs[0], Expr::Integer(0))
        && matches!(&dargs[1], Expr::Integer(1))));
  if is_std_normal && matches!(&x, Expr::Identifier(_)) {
    let sqrt = |e: Expr| call("Sqrt", vec![e]);
    return Ok(divide(
      sqrt(divide(int(2), pi())),
      call(
        "Times",
        vec![
          power(
            Expr::Identifier("E".to_string()),
            divide(power(x.clone(), int(2)), int(2)),
          ),
          call("Erfc", vec![divide(x.clone(), sqrt(int(2)))]),
        ],
      ),
    ));
  }

  if !matches!(dist_name, "ExponentialDistribution" | "ParetoDistribution") {
    return Ok(unevaluated(args));
  }

  let simplify_ratio = |p: Expr, s: Expr| -> Result<Expr, InterpreterError> {
    eval(call("Simplify", vec![divide(p, s)]))
  };

  let pdf = pdf_ast(args)?;
  let sf = survival_function_ast(args)?;

  // Piecewise PDF and SF with identical conditions: divide piece-wise
  if let (
    Expr::FunctionCall {
      name: pn,
      args: pargs,
    },
    Expr::FunctionCall {
      name: sn,
      args: sargs,
    },
  ) = (&pdf, &sf)
    && pn == "Piecewise"
    && sn == "Piecewise"
    && !pargs.is_empty()
    && !sargs.is_empty()
    && let (Expr::List(p_pieces), Expr::List(s_pieces)) = (&pargs[0], &sargs[0])
    && p_pieces.len() == s_pieces.len()
  {
    let mut new_pieces: Vec<Expr> = Vec::with_capacity(p_pieces.len());
    for (pp, sp) in p_pieces.iter().zip(s_pieces.iter()) {
      let (Expr::List(p_pair), Expr::List(s_pair)) = (pp, sp) else {
        return Ok(unevaluated(args));
      };
      if p_pair.len() != 2
        || s_pair.len() != 2
        || expr_to_string(&p_pair[1]) != expr_to_string(&s_pair[1])
      {
        return Ok(unevaluated(args));
      }
      let ratio = simplify_ratio(p_pair[0].clone(), s_pair[0].clone())?;
      new_pieces.push(Expr::List(vec![ratio, p_pair[1].clone()].into()));
    }
    return Ok(call(
      "Piecewise",
      vec![Expr::List(new_pieces.into()), int(0)],
    ));
  }

  // Numeric argument: the piecewise branch has already been selected
  simplify_ratio(pdf, sf)
}

/// Match `Erfc[z]/2` (as Divide or Times[1/2, ...]) and return z.
fn match_erfc_half(expr: &Expr) -> Option<Expr> {
  let erfc_arg = |e: &Expr| -> Option<Expr> {
    if let Expr::FunctionCall { name, args } = e
      && name == "Erfc"
      && args.len() == 1
    {
      Some(args[0].clone())
    } else {
      None
    }
  };
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Integer(2)) => erfc_arg(left),
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      match (&args[0], &args[1]) {
        (Expr::FunctionCall { name: rn, args: ra }, other)
        | (other, Expr::FunctionCall { name: rn, args: ra })
          if rn == "Rational"
            && ra.len() == 2
            && matches!(&ra[0], Expr::Integer(1))
            && matches!(&ra[1], Expr::Integer(2)) =>
        {
          erfc_arg(other)
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Closed-form Quantile[dist, q] for distributions whose inverse CDF is
/// elementary. Returns `Some(expr)` when the head is recognised. Callers
/// should try this before the numerical fallback.
pub fn quantile_distribution_closed_form(
  dist_name: &str,
  dargs: &[Expr],
  q: &Expr,
) -> Option<Expr> {
  // Numeric q in [0, 1] only; the symbolic-q forms wolframscript returns
  // are ConditionalExpression/Piecewise wrappers that are out of scope.
  let q_num = crate::functions::math_ast::expr_to_num(q)?;
  if !(0.0..=1.0).contains(&q_num) {
    return None;
  }
  let is_exact_q = !matches!(q, Expr::Real(_));

  // Builders for the elementary inverse-CDF formulas below.
  let log = |x: Expr| call("Log", vec![x]);
  let sqrt = |x: Expr| call("Sqrt", vec![x]);
  let power = |b: Expr, e: Expr| call("Power", vec![b, e]);
  let neg = |x: Expr| times(int(-1), x);
  let one_minus_q = || minus(int(1), q.clone());

  match dist_name {
    // Empirical DataDistribution: the smallest support value whose
    // cumulative weight reaches q (inclusive)
    "DataDistribution" => {
      let (weights, values) = data_distribution_parts(dargs)?;
      let target = q_num;
      let mut cum = 0.0;
      for (w, v) in weights.iter().zip(values.iter()) {
        cum += w.0 as f64 / w.1 as f64;
        if cum >= target - 1e-12 {
          return Some(if v.1 == 1 {
            Expr::Integer(v.0)
          } else {
            crate::functions::math_ast::make_rational(v.0, v.1)
          });
        }
      }
      values.last().map(|v| {
        if v.1 == 1 {
          Expr::Integer(v.0)
        } else {
          crate::functions::math_ast::make_rational(v.0, v.1)
        }
      })
    }
    // Quantile[ExponentialDistribution[lambda], q] = -Log[1 - q] / lambda
    "ExponentialDistribution" if dargs.len() == 1 => {
      // Edges: the generic formula would leave Infinity/lambda unreduced
      if q_num == 1.0 && is_exact_q {
        return Some(infinity());
      }
      if q_num == 0.0 && is_exact_q {
        return Some(int(0));
      }
      let lambda = dargs[0].clone();
      let one_minus_q = minus(int(1), q.clone());
      let log_term = call("Log", vec![one_minus_q]);
      let neg_log = times(int(-1), log_term);
      let expr = divide(neg_log, lambda);
      eval(expr).ok()
    }
    // Quantile[CauchyDistribution[a, b], q] = a + b*Tan[Pi*(q - 1/2)]
    "CauchyDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(neg_infinity());
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let (a, b) = (dargs[0].clone(), dargs[1].clone());
      let pi = pi();
      let q_minus_half =
        minus(q.clone(), crate::functions::math_ast::make_rational(1, 2));
      let tan = call("Tan", vec![times(pi, q_minus_half)]);
      eval(plus(a, times(b, tan))).ok()
    }
    // Quantile[WeibullDistribution[k, λ], q] = λ (-Log[1 - q])^(1/k)
    "WeibullDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(int(0));
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let (k, lam) = (dargs[0].clone(), dargs[1].clone());
      eval(times(
        lam,
        power(neg(log(one_minus_q())), divide(int(1), k)),
      ))
      .ok()
    }
    // Quantile[ParetoDistribution[k, α], q] = k (1 - q)^(-1/α)
    "ParetoDistribution" if dargs.len() == 2 => {
      if is_exact_q && q_num == 1.0 {
        return Some(infinity());
      }
      let (kmin, alpha) = (dargs[0].clone(), dargs[1].clone());
      eval(times(kmin, power(one_minus_q(), divide(int(-1), alpha)))).ok()
    }
    // Quantile[RayleighDistribution[σ], q] = σ Sqrt[-Log[(1 - q)^2]]
    "RayleighDistribution" if dargs.len() == 1 => {
      if is_exact_q && q_num == 1.0 {
        return Some(infinity());
      }
      let sigma = dargs[0].clone();
      eval(times(sigma, sqrt(neg(log(power(one_minus_q(), int(2))))))).ok()
    }
    // Quantile[LaplaceDistribution[μ, β], q]: μ + β Log[2 q] for q ≤ 1/2,
    // else μ − β Log[2 (1 − q)].
    "LaplaceDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(neg_infinity());
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let (mu, beta) = (dargs[0].clone(), dargs[1].clone());
      let body = if q_num <= 0.5 {
        plus(mu, times(beta, log(times(int(2), q.clone()))))
      } else {
        minus(mu, times(beta, log(times(int(2), one_minus_q()))))
      };
      eval(body).ok()
    }
    // Quantile[LogisticDistribution[μ, β], q] = μ − β Log[-1 + 1/q]
    "LogisticDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(neg_infinity());
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let (mu, beta) = (dargs[0].clone(), dargs[1].clone());
      eval(minus(
        mu,
        times(beta, log(plus(int(-1), divide(int(1), q.clone())))),
      ))
      .ok()
    }
    // Quantile[GumbelDistribution[μ, β], q] = μ + β Log[-Log[1 - q]]
    "GumbelDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(neg_infinity());
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let (mu, beta) = (dargs[0].clone(), dargs[1].clone());
      eval(plus(mu, times(beta, log(neg(log(one_minus_q())))))).ok()
    }
    // Quantile[UniformDistribution[{a, b}], q] = (1 - q)*a + q*b
    "UniformDistribution" if dargs.len() == 1 => {
      let (a, b) = match &dargs[0] {
        Expr::List(bounds) if bounds.len() == 2 => {
          (bounds[0].clone(), bounds[1].clone())
        }
        _ => return None,
      };
      eval(plus(
        times(minus(int(1), q.clone()), a),
        times(q.clone(), b),
      ))
      .ok()
    }
    // Quantile[NormalDistribution[m, s], q] = m - Sqrt[2]*s*InverseErfc[2q]
    "NormalDistribution" if dargs.is_empty() || dargs.len() == 2 => {
      let (m, s) = if dargs.is_empty() {
        (int(0), int(1))
      } else {
        (dargs[0].clone(), dargs[1].clone())
      };
      if is_exact_q {
        if q_num == 0.0 {
          return Some(neg_infinity());
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
        // InverseErfc[1] = 0: the median is exactly m
        if let Some((1, 2)) = crate::functions::math_ast::expr_to_rational(q) {
          return eval(m).ok();
        }
        // m - Sqrt[2]*s*InverseErfc[2q]. Evaluating the result reflects
        // InverseErfc[2q] for q > 1/2 (2q > 1 → -InverseErfc[2-2q]) and folds
        // the sign, matching wolframscript while preserving the factor order.
        let two_q = eval(times(int(2), q.clone())).ok()?;
        let inverse_erfc = call("InverseErfc", vec![two_q]);
        let sqrt2 = call("Sqrt", vec![int(2)]);
        let factors: Vec<Expr> = match &s {
          Expr::Integer(1) => vec![sqrt2, inverse_erfc],
          _ => vec![sqrt2, s.clone(), inverse_erfc],
        };
        let term = neg(call("Times", factors));
        let result = match &m {
          Expr::Integer(0) => term,
          _ => call("Plus", vec![m.clone(), term]),
        };
        return eval(result).ok();
      }
      // Machine-precision q: numeric inverse CDF (requires numeric m, s)
      let m_num = crate::functions::math_ast::expr_to_num(&m)?;
      let s_num = crate::functions::math_ast::expr_to_num(&s)?;
      let z = crate::functions::math_ast::inverse_erf_f64(2.0 * q_num - 1.0);
      Some(Expr::Real(m_num + s_num * std::f64::consts::SQRT_2 * z))
    }
    // Quantile[GammaDistribution[a, b], q] = b InverseGammaRegularized[a, 0, q]
    "GammaDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(int(0));
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let (a, b) = (dargs[0].clone(), dargs[1].clone());
      let igr = call("InverseGammaRegularized", vec![a, int(0), q.clone()]);
      eval(times(b, igr)).ok()
    }
    // ChiSquareDistribution[v] = GammaDistribution[v/2, 2], so the quantile is
    // 2 InverseGammaRegularized[v/2, 0, q].
    "ChiSquareDistribution" if dargs.len() == 1 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(int(0));
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let half_v = eval(divide(dargs[0].clone(), int(2))).ok()?;
      let igr =
        call("InverseGammaRegularized", vec![half_v, int(0), q.clone()]);
      eval(times(int(2), igr)).ok()
    }
    // Quantile[BetaDistribution[a, b], q] = InverseBetaRegularized[q, a, b].
    // For an exact interior q wolframscript returns an algebraic Root object,
    // which Woxi does not reproduce, so that case is left unevaluated; the
    // numeric (machine-real q) path evaluates by bisection.
    "BetaDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(int(0));
        }
        if q_num == 1.0 {
          return Some(int(1));
        }
        return None;
      }
      let (a, b) = (dargs[0].clone(), dargs[1].clone());
      eval(call("InverseBetaRegularized", vec![q.clone(), a, b])).ok()
    }
    // Quantile[StudentTDistribution[nu], q]. For q > 1/2 this is
    // Sqrt[nu (1/InverseBetaRegularized[2(1-q), nu/2, 1/2] - 1)], with the sign
    // flipped below the median and 0 at it. Only numeric nu is handled:
    // wolframscript keeps the radical joined (Sqrt[nu (...)]) there, whereas a
    // symbolic nu splits it into Sqrt[nu] Sqrt[...]. (The location/scale form
    // StudentTDistribution[mu, sigma, nu] is not yet a recognized distribution
    // in Woxi, so it is intentionally not handled here.)
    "StudentTDistribution" if dargs.len() == 1 => {
      let nu = dargs[0].clone();
      crate::functions::math_ast::expr_to_num(&nu)?;

      if is_exact_q && q_num == 0.0 {
        return Some(neg_infinity());
      }
      if is_exact_q && q_num == 1.0 {
        return Some(infinity());
      }
      if q_num == 0.5 {
        return Some(int(0));
      }
      let s = if q_num > 0.5 {
        eval(times(int(2), one_minus_q())).ok()?
      } else {
        eval(times(int(2), q.clone())).ok()?
      };
      let ibr = call(
        "InverseBetaRegularized",
        vec![
          s,
          divide(nu.clone(), int(2)),
          crate::functions::math_ast::make_rational(1, 2),
        ],
      );
      let radical = sqrt(times(nu, plus(int(-1), power(ibr, int(-1)))));
      eval(if q_num < 0.5 { neg(radical) } else { radical }).ok()
    }
    // Quantile[FRatioDistribution[n, m], q] =
    //   (m/n) (1/InverseBetaRegularized[1, -q, m/2, n/2] - 1).
    // The generalized 4-arg InverseBetaRegularized[1, -q, m/2, n/2] equals the
    // 3-arg InverseBetaRegularized[1 - q, m/2, n/2]; wolframscript prints the
    // 4-arg form. Only numeric n, m are handled: a symbolic m would reorder the
    // (Plus)*m product relative to wolframscript's m*(Plus) ordering.
    "FRatioDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(int(0));
        }
        if q_num == 1.0 {
          return Some(infinity());
        }
      }
      let (n, m) = (dargs[0].clone(), dargs[1].clone());
      crate::functions::math_ast::expr_to_num(&n)?;
      crate::functions::math_ast::expr_to_num(&m)?;
      let ibr = call(
        "InverseBetaRegularized",
        vec![
          int(1),
          eval(neg(q.clone())).ok()?,
          divide(m.clone(), int(2)),
          divide(n.clone(), int(2)),
        ],
      );
      eval(divide(times(m, plus(int(-1), power(ibr, int(-1)))), n)).ok()
    }
    "BinomialDistribution"
    | "PoissonDistribution"
    | "GeometricDistribution"
    | "NegativeBinomialDistribution"
    | "PascalDistribution"
    | "BernoulliDistribution"
    | "DiscreteUniformDistribution" => {
      quantile_discrete(dist_name, dargs, q, q_num)
    }
    _ => None,
  }
}

/// InverseSurvivalFunction[dist, q] — the value x with SurvivalFunction == q,
/// i.e. InverseCDF[dist, 1 - q]. For most distributions wolframscript's form
/// agrees with InverseCDF at 1 - q (so we delegate); the gamma/normal family is
/// expressed natively in the survival parametrization (upper InverseErfc /
/// InverseGammaRegularized). Returns `None` (unevaluated) for everything else.
pub fn inverse_survival_closed_form(
  dist_name: &str,
  dargs: &[Expr],
  q: &Expr,
) -> Option<Expr> {
  let q_num = crate::functions::math_ast::expr_to_num(q)?;
  if !(0.0..=1.0).contains(&q_num) {
    return None;
  }
  let is_exact_q = !matches!(q, Expr::Real(_));

  match dist_name {
    // InverseSurvivalFunction[NormalDistribution[m, s], q]
    //   = m + Sqrt[2] s InverseErfc[2 q]
    "NormalDistribution" if dargs.is_empty() || dargs.len() == 2 => {
      let (m, s) = if dargs.is_empty() {
        (int(0), int(1))
      } else {
        (dargs[0].clone(), dargs[1].clone())
      };
      if is_exact_q {
        if q_num == 0.0 {
          return Some(infinity());
        }
        if q_num == 1.0 {
          return Some(neg_infinity());
        }
        // InverseErfc[1] = 0, so the median survival point is exactly m.
        if let Some((1, 2)) = crate::functions::math_ast::expr_to_rational(q) {
          return eval(m).ok();
        }
      }
      let two_q = eval(times(int(2), q.clone())).ok()?;
      let inverse_erfc = call("InverseErfc", vec![two_q]);
      let sqrt2 = call("Sqrt", vec![int(2)]);
      let factors: Vec<Expr> = match &s {
        Expr::Integer(1) => vec![sqrt2, inverse_erfc],
        _ => vec![sqrt2, s.clone(), inverse_erfc],
      };
      let term = call("Times", factors);
      // Evaluate so InverseErfc[2q] reflects for q > 1/2 (matching the Quantile
      // path), e.g. Sqrt[2] InverseErfc[3/2] -> -(Sqrt[2] InverseErfc[1/2]).
      let result = match &m {
        Expr::Integer(0) => term,
        _ => call("Plus", vec![m.clone(), term]),
      };
      eval(result).ok()
    }
    // InverseSurvivalFunction[GammaDistribution[a, b], q]
    //   = b InverseGammaRegularized[a, q]
    "GammaDistribution" if dargs.len() == 2 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(infinity());
        }
        if q_num == 1.0 {
          return Some(int(0));
        }
      }
      let igr =
        call("InverseGammaRegularized", vec![dargs[0].clone(), q.clone()]);
      eval(times(dargs[1].clone(), igr)).ok()
    }
    // ChiSquareDistribution[v] = GammaDistribution[v/2, 2]
    "ChiSquareDistribution" if dargs.len() == 1 => {
      if is_exact_q {
        if q_num == 0.0 {
          return Some(infinity());
        }
        if q_num == 1.0 {
          return Some(int(0));
        }
      }
      let igr = call(
        "InverseGammaRegularized",
        vec![divide(dargs[0].clone(), int(2)), q.clone()],
      );
      eval(times(int(2), igr)).ok()
    }
    // Distributions whose survival inverse shares the InverseCDF[dist, 1 - q]
    // form (verified against wolframscript): delegate to the quantile.
    "ExponentialDistribution"
    | "UniformDistribution"
    | "CauchyDistribution"
    | "LogisticDistribution"
    | "WeibullDistribution"
    | "ParetoDistribution"
    | "GumbelDistribution"
    | "LaplaceDistribution"
    | "RayleighDistribution" => {
      let one_minus_q = eval(minus(int(1), q.clone())).ok()?;
      quantile_distribution_closed_form(dist_name, dargs, &one_minus_q)
    }
    _ => None,
  }
}

/// The integer support `(min, max)` of a discrete distribution; `max` is `None`
/// for unbounded support.
fn discrete_support(
  dist_name: &str,
  dargs: &[Expr],
) -> Option<(i128, Option<i128>)> {
  let as_int = |e: &Expr| {
    crate::functions::math_ast::expr_to_num(e)
      .filter(|v| v.fract() == 0.0)
      .map(|v| v as i128)
  };
  match dist_name {
    "BinomialDistribution" if dargs.len() == 2 => {
      Some((0, Some(as_int(&dargs[0])?)))
    }
    "PoissonDistribution" if dargs.len() == 1 => Some((0, None)),
    "GeometricDistribution" if dargs.len() == 1 => Some((0, None)),
    "NegativeBinomialDistribution" if dargs.len() == 2 => Some((0, None)),
    // Pascal support starts at the number of successes n.
    "PascalDistribution" if dargs.len() == 2 => {
      Some((as_int(&dargs[0])?, None))
    }
    "BernoulliDistribution" if dargs.len() == 1 => Some((0, Some(1))),
    "DiscreteUniformDistribution" if dargs.len() == 1 => match &dargs[0] {
      Expr::List(b) if b.len() == 2 => {
        Some((as_int(&b[0])?, Some(as_int(&b[1])?)))
      }
      _ => None,
    },
    _ => None,
  }
}

/// Quantile/InverseCDF for a discrete distribution: the smallest integer k in
/// the support with CDF[dist, k] >= q. q <= 0 gives the support minimum and
/// q >= 1 the support maximum (Infinity for unbounded support).
fn quantile_discrete(
  dist_name: &str,
  dargs: &[Expr],
  q: &Expr,
  q_num: f64,
) -> Option<Expr> {
  let (kmin, kmax) = discrete_support(dist_name, dargs)?;
  if q_num <= 0.0 {
    return Some(int(kmin));
  }
  if q_num >= 1.0 {
    return Some(kmax.map(int).unwrap_or_else(infinity));
  }
  let dist = unevaluated(dist_name, dargs);
  let mut k = kmin;
  loop {
    let cdf_k = eval(call("CDF", vec![dist.clone(), int(k)])).ok()?;
    let cmp =
      eval(comparison(cdf_k, ComparisonOp::GreaterEqual, q.clone())).ok()?;
    if matches!(&cmp, Expr::Identifier(s) if s == "True") {
      return Some(int(k));
    }
    k += 1;
    if let Some(m) = kmax
      && k >= m
    {
      return Some(int(m));
    }
    if k - kmin > 1_000_000 {
      return None; // safety valve for pathological inputs
    }
  }
}

/// Numerical inverse-CDF for distributions whose CDF Woxi can compute
/// symbolically or numerically. Returns `Some(quantile)` if successful,
/// `None` if the head isn't recognised (caller falls back), or `Err` for
/// hard failures.
pub fn quantile_distribution_numeric(
  dist_name: &str,
  dargs: &[Expr],
  q: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  // Only handle ProbabilityDistribution numerically for now — the standard
  // distributions have their own closed-form quantile elsewhere (or stay
  // symbolic). Extending to LogNormal/Pareto/etc. is a future task.
  if dist_name != "ProbabilityDistribution" {
    return Ok(None);
  }
  if dargs.len() != 2 {
    return Ok(None);
  }
  let q_val = match q {
    Expr::Integer(n) => *n as f64,
    Expr::Real(f) => *f,
    _ => return Ok(None),
  };
  if !(0.0..=1.0).contains(&q_val) {
    return Err(InterpreterError::EvaluationError(format!(
      "Quantile: probability {q_val} not in [0, 1]"
    )));
  }

  let Expr::List(items) = &dargs[1] else {
    return Ok(None);
  };
  if items.len() != 3 {
    return Ok(None);
  }
  let Expr::Identifier(_var) = &items[0] else {
    return Ok(None);
  };
  let lo = try_eval_to_f64(&items[1]).unwrap_or(0.0);
  let hi_expr = &items[2];
  // Bracket the upper bound; for Infinity, start with lo + 1 and expand.
  let mut hi_known = try_eval_to_f64(hi_expr);
  let hi_is_infinite = matches!(hi_expr, Expr::Identifier(n) if n == "Infinity")
    || hi_known.map(|h| h.is_infinite()).unwrap_or(false);
  if hi_is_infinite {
    hi_known = None;
  }
  let dist_expr = unevaluated("ProbabilityDistribution", dargs);
  // Helper: numeric CDF evaluation at `x`.
  let cdf_at = |x: f64| -> Option<f64> {
    let val = cdf_ast(&[dist_expr.clone(), Expr::Real(x)]).ok()?;
    try_eval_to_f64(&val)
  };

  // Establish an upper bracket. For infinite support, double until CDF ≥ q.
  let mut hi: f64 = match hi_known {
    Some(h) => h,
    None => {
      let mut h = (lo + 1.0).max(1.0);
      for _ in 0..60 {
        match cdf_at(h) {
          Some(c) if c >= q_val => break,
          _ => h *= 2.0,
        }
      }
      h
    }
  };
  let mut lo_b = lo;
  // Bisection: ~50 iterations gives ≈1e-15 precision for any sane support.
  for _ in 0..80 {
    let mid = 0.5 * (lo_b + hi);
    let Some(c) = cdf_at(mid) else {
      return Ok(None);
    };
    if c < q_val {
      lo_b = mid;
    } else {
      hi = mid;
    }
    if (hi - lo_b).abs() < 1e-14 {
      break;
    }
  }
  Ok(Some(Expr::Real(0.5 * (lo_b + hi))))
}

/// PDF[ProbabilityDistribution[pdf_expr, {var, lo, hi}], t] substitutes `t`
/// for the dummy variable in the pdf expression, returning a piecewise that
/// is zero outside `[lo, hi]`.
fn pdf_probability_distribution(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  use crate::functions::plot::substitute_var;
  if dargs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ProbabilityDistribution: missing pdf".into(),
    ));
  }
  let pdf = &dargs[0];
  // Univariate: single iterator. Multivariate PDF[…, {a, b}] isn't supported here.
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("ProbabilityDistribution", dargs), x],
    ))
  };
  if dargs.len() != 2 {
    return unevaluated(x);
  }
  let Expr::List(items) = &dargs[1] else {
    return unevaluated(x);
  };
  if items.len() != 3 {
    return unevaluated(x);
  }
  let Expr::Identifier(var) = &items[0] else {
    return unevaluated(x);
  };
  let lo = items[1].clone();
  let hi = items[2].clone();
  let density = substitute_var(pdf, var, &x);
  let cond =
    comparison3(lo, ComparisonOp::LessEqual, x, ComparisonOp::LessEqual, hi);
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// CDF[ProbabilityDistribution[pdf, {var, lo, hi}], t] = Integrate[pdf, {var, lo, t}].
/// Returns 0 for t < lo and 1 for t > hi (piecewise).
fn cdf_probability_distribution(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("ProbabilityDistribution", dargs), x],
    ))
  };
  if dargs.len() != 2 {
    return unevaluated(x);
  }
  let pdf = &dargs[0];
  let Expr::List(items) = &dargs[1] else {
    return unevaluated(x);
  };
  if items.len() != 3 {
    return unevaluated(x);
  }
  let Expr::Identifier(var) = &items[0] else {
    return unevaluated(x);
  };
  let lo = items[1].clone();
  let integral = call(
    "Integrate",
    vec![
      pdf.clone(),
      Expr::List(vec![Expr::Identifier(var.clone()), lo, x].into()),
    ],
  );
  eval(integral)
}

/// PDF[NormalDistribution[mu, sigma], x] = 1/(E^((-mu + x)^2/(2*sigma^2))*Sqrt[2*Pi]*sigma)
fn pdf_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (mu, sigma) = match dargs.len() {
    0 => (int(0), int(1)),
    2 => (dargs[0].clone(), dargs[1].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "NormalDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };

  // 1/(E^((x - mu)^2/(2*sigma^2)) * Sqrt[2*Pi] * sigma)
  let diff = minus(x, mu);
  let diff_sq = power(diff, int(2));
  let two_sigma_sq = times(int(2), power(sigma.clone(), int(2)));
  let exponent = divide(diff_sq, two_sigma_sq);
  let exp_part = power(e(), exponent);
  let sqrt_part = sqrt(times(int(2), pi()));
  let denominator = times(times(exp_part, sqrt_part), sigma);
  let result = divide(int(1), denominator);
  eval(result)
}

/// PDF[UniformDistribution[{a, b}], x] = Piecewise[{{1/(b-a), a <= x <= b}}, 0]
fn pdf_uniform(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    1 => {
      if let Expr::List(bounds) = &dargs[0] {
        if bounds.len() == 2 {
          (bounds[0].clone(), bounds[1].clone())
        } else {
          return Err(InterpreterError::EvaluationError(
            "UniformDistribution: expected {min, max}".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "UniformDistribution: expected a list {min, max}".into(),
        ));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "UniformDistribution expects 0 or 1 argument".into(),
      ));
    }
  };

  let density = eval(divide(int(1), minus(b.clone(), a.clone())))?;
  let cond =
    comparison3(a, ComparisonOp::LessEqual, x, ComparisonOp::LessEqual, b);

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[ExponentialDistribution[lambda], x] = Piecewise[{{lambda*E^(-lambda*x), x >= 0}}, 0]
fn pdf_exponential(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExponentialDistribution expects 1 argument".into(),
    ));
  }
  let lambda = dargs[0].clone();

  let density =
    eval(divide(lambda.clone(), power(e(), times(lambda, x.clone()))))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[PoissonDistribution[mu], k] = Piecewise[{{mu^k/(E^mu * k!), k >= 0}}, 0]
fn pdf_poisson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PoissonDistribution expects 1 argument".into(),
    ));
  }
  let mu = dargs[0].clone();

  let numerator = power(mu.clone(), x.clone());
  let denominator = times(power(e(), mu), factorial(x.clone()));
  let density = eval(divide(numerator, denominator))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[MeixnerDistribution[a, b, m, d], x] =
///   2^(2d-1) E^(b(x-m)/a) Cos[b/2]^(2d)
///     Gamma[d - I(x-m)/a] Gamma[d + I(x-m)/a] / (a Pi Gamma[2d])
/// Support is all reals (no Piecewise). Numeric points evaluate through the
/// complex Gamma factors (Chop drops the residual zero imaginary part); the
/// fully-symbolic form matches wolframscript except for the Times order of the
/// two conjugate Gamma factors.
fn pdf_meixner(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 4 {
    return Err(InterpreterError::EvaluationError(
      "MeixnerDistribution expects 4 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();
  let m = dargs[2].clone();
  let d = dargs[3].clone();

  let xm = minus(x, m); // x - m
  let i = Expr::Identifier("I".to_string());
  let iy = divide(times(i, xm.clone()), a.clone()); // I (x-m)/a
  let two_pow = power(int(2), plus(int(-1), times(int(2), d.clone())));
  let e_part = power(e(), divide(times(b.clone(), xm), a.clone()));
  let cos_part =
    power(unary_fn("Cos", divide(b, int(2))), times(int(2), d.clone()));
  let g1 = unary_fn("Gamma", minus(d.clone(), iy.clone()));
  let g2 = unary_fn("Gamma", plus(d.clone(), iy));
  let numerator = times(times(times(times(two_pow, e_part), cos_part), g1), g2);
  let denominator = times(times(a, pi()), unary_fn("Gamma", times(int(2), d)));
  eval(divide(numerator, denominator))
}

/// PDF[PoissonConsulDistribution[m, lam], k] =
///   Piecewise[{{(E^(-(k lam) - m) m (k lam + m)^(k-1))/k!, k >= 0}}, 0]
/// Values are exact; the symbolic form differs from wolframscript only in the
/// Times factor order of `m` vs the power (a Times-canonicalization quirk),
/// and StandardDeviation keeps the (1-lam) factor inside the Sqrt.
fn pdf_poisson_consul(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PoissonConsulDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let lam = dargs[1].clone();
  let k = x.clone();

  let klam = times(k.clone(), lam);
  let klam_m = plus(klam.clone(), m.clone()); // k lam + m
  // E^(-(k lam) - m)
  let exp_part = power(e(), minus(times(int(-1), klam), m.clone()));
  // (k lam + m)^(-1 + k)
  let pow_part = power(klam_m, plus(int(-1), k.clone()));
  let numerator = times(times(exp_part, m), pow_part);
  let density = eval(divide(numerator, factorial(k.clone())))?;
  let cond = comparison(k, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(density, cond)], int(0)))
}

fn unary_fn(name: &str, arg: Expr) -> Expr {
  call(name, vec![arg])
}

/// PDF[SkellamDistribution[a, b], k] =
///   (a/b)^(k/2) * E^(-a-b) * BesselI[k, 2 Sqrt[a b]]   (for all integers k).
fn pdf_skellam(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SkellamDistribution expects 2 arguments".into(),
    ));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let ratio_pow =
    power(divide(a.clone(), b.clone()), divide(x.clone(), int(2)));
  let exp_part = power(
    e(),
    plus(times(int(-1), a.clone()), times(int(-1), b.clone())),
  );
  let bessel = call(
    "BesselI",
    vec![x, times(int(2), unary_fn("Sqrt", times(a, b)))],
  );
  eval(times(times(ratio_pow, exp_part), bessel))
}

/// CDF[SkellamDistribution[a, b], k] =
///   1 - MarcumQ[-Floor[k], Sqrt[2] Sqrt[a], Sqrt[2] Sqrt[b]].
fn cdf_skellam(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SkellamDistribution expects 2 arguments".into(),
    ));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let sqrt2 = unary_fn("Sqrt", int(2));
  let marcum = call(
    "MarcumQ",
    vec![
      times(int(-1), unary_fn("Floor", x)),
      times(sqrt2.clone(), unary_fn("Sqrt", a)),
      times(sqrt2, unary_fn("Sqrt", b)),
    ],
  );
  eval(minus(int(1), marcum))
}

/// PDF[BernoulliDistribution[p], k] = Piecewise[{{1-p, k==0}, {p, k==1}}, 0]
fn pdf_bernoulli(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BernoulliDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();

  let one_minus_p = eval(minus(int(1), p.clone()))?;
  let cond0 = comparison(x.clone(), ComparisonOp::Equal, int(0));
  let cond1 = comparison(x, ComparisonOp::Equal, int(1));

  eval(piecewise(vec![(one_minus_p, cond0), (p, cond1)], int(0)))
}

/// PDF[BinomialDistribution[n, p], k] =
///   Piecewise[{{Binomial[n, k] p^k (1-p)^(n-k), 0 <= k <= n}}, 0]
fn pdf_binomial(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();

  let binom = call("Binomial", vec![n.clone(), x.clone()]);
  let p_k = power(p.clone(), x.clone());
  // (1 - p)^(n - k); pre-evaluate the base and exponent so they collapse.
  let q_nk = power(eval(minus(int(1), p))?, eval(minus(n.clone(), x.clone()))?);
  let density = eval(times(binom, times(p_k, q_nk)))?;
  // 0 <= k <= n.
  let cond = Expr::Comparison {
    operands: vec![int(0), x, n],
    operators: vec![ComparisonOp::LessEqual, ComparisonOp::LessEqual],
  };
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[HypergeometricDistribution[n, ns, nt], k] =
///   Piecewise[
///     {{Binomial[ns, k]*Binomial[nt-ns, n-k]/Binomial[nt, n], 0 <= k <= n}},
///     0]
/// (For non-trivial parameter constraints, the support is
///  max(0, n + ns - nt) <= k <= min(n, ns); when the implicit bounds are
///  automatically satisfied (e.g. n <= ns and n + ns <= nt), the
///  Piecewise condition collapses to 0 <= k <= n.)
fn pdf_hypergeometric(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "HypergeometricDistribution expects 3 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let ns = dargs[1].clone();
  let nt = dargs[2].clone();

  let binom = |a: Expr, b: Expr| call("Binomial", vec![a, b]);
  // Pre-evaluate the density expression so e.g. Binomial[100 - 50, ...]
  // collapses to Binomial[50, ...]. Piecewise holds its arguments, so we
  // can't rely on a subsequent global pass to simplify these.
  let numerator = times(
    binom(ns.clone(), x.clone()),
    binom(
      eval(minus(nt.clone(), ns))?,
      eval(minus(n.clone(), x.clone()))?,
    ),
  );
  let denominator = eval(binom(nt, n.clone()))?;
  let density = eval(divide(numerator, denominator))?;
  // Build 0 <= k <= n as a single chained comparison.
  let cond = Expr::Comparison {
    operands: vec![int(0), x, n],
    operators: vec![ComparisonOp::LessEqual, ComparisonOp::LessEqual],
  };
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[BinormalDistribution[rho], {x, y}],
/// PDF[BinormalDistribution[{sigma1, sigma2}, rho], {x, y}],
/// PDF[BinormalDistribution[{mu1, mu2}, {sigma1, sigma2}, rho], {x, y}]:
///   PDF = exp(-Q / (2 (1 - rho^2))) / (2 Pi sigma1 sigma2 Sqrt[1 - rho^2])
/// where Q is the standardised quadratic form
///   ((x - mu1)/sigma1)^2
///   - 2 rho ((x - mu1)/sigma1) ((y - mu2)/sigma2)
///   + ((y - mu2)/sigma2)^2.
fn pdf_binormal(dargs: &[Expr], xy: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |xy: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("BinormalDistribution", dargs), xy],
    ))
  };
  let Expr::List(coords) = &xy else {
    return unevaluated(xy);
  };
  if coords.len() != 2 {
    return unevaluated(xy);
  }
  let x = coords[0].clone();
  let y = coords[1].clone();
  let (mu1, mu2, sigma1, sigma2, rho) = match dargs.len() {
    1 => (int(0), int(0), int(1), int(1), dargs[0].clone()),
    2 => {
      let Expr::List(sigmas) = &dargs[0] else {
        return Err(InterpreterError::EvaluationError(
          "BinormalDistribution: expected {sigma1, sigma2}".into(),
        ));
      };
      if sigmas.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BinormalDistribution: expected {sigma1, sigma2}".into(),
        ));
      }
      (
        int(0),
        int(0),
        sigmas[0].clone(),
        sigmas[1].clone(),
        dargs[1].clone(),
      )
    }
    3 => {
      let (Expr::List(mus), Expr::List(sigmas)) = (&dargs[0], &dargs[1]) else {
        return Err(InterpreterError::EvaluationError(
          "BinormalDistribution: expected {mu1, mu2}, {sigma1, sigma2}".into(),
        ));
      };
      if mus.len() != 2 || sigmas.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BinormalDistribution: expected {mu1, mu2}, {sigma1, sigma2}".into(),
        ));
      }
      (
        mus[0].clone(),
        mus[1].clone(),
        sigmas[0].clone(),
        sigmas[1].clone(),
        dargs[2].clone(),
      )
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "BinormalDistribution expects 1, 2, or 3 arguments".into(),
      ));
    }
  };
  // Standardised coordinates u = (x - mu1)/sigma1, v = (y - mu2)/sigma2.
  let u = divide(minus(x, mu1), sigma1.clone());
  let v = divide(minus(y, mu2), sigma2.clone());
  // Q = u^2 - 2 rho u v + v^2
  let q = plus(
    minus(
      power(u.clone(), int(2)),
      times(int(2), times(rho.clone(), times(u, v.clone()))),
    ),
    power(v, int(2)),
  );
  // exponent = -Q / (2 (1 - rho^2))
  let one_minus_rho_sq = minus(int(1), power(rho.clone(), int(2)));
  let _ = &rho;
  let exponent =
    divide(times(int(-1), q), times(int(2), one_minus_rho_sq.clone()));
  // PDF = E^exponent / (2 Pi Sqrt[1 - rho^2] sigma1 sigma2)
  let denom = times(
    times(int(2), pi()),
    times(sqrt(one_minus_rho_sq), times(sigma1, sigma2)),
  );
  let numer = power(e(), exponent);
  eval(divide(numer, denom))
}

/// PDF[CauchyDistribution[a, b], x] = 1/(Pi*b*(1+((x-a)/b)^2))
fn pdf_cauchy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    2 => (dargs[0].clone(), dargs[1].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "CauchyDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };
  // 1 / (Pi * b * (1 + ((x - a) / b)^2))
  let diff = minus(x, a);
  let ratio = divide(diff, b.clone());
  let denom = times(times(pi(), b), plus(int(1), power(ratio, int(2))));
  eval(divide(int(1), denom))
}

/// PDF[GeometricDistribution[p], k] = Piecewise[{{(1-p)^k * p, k >= 0}}, 0]
fn pdf_geometric(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "GeometricDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();
  let one_minus_p = minus(int(1), p.clone());
  let density = eval(times(power(one_minus_p, x.clone()), p))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[LogSeriesDistribution[t], k] =
///   Piecewise[{{-(t^k/(k*Log[1 - t])), k >= 1}}, 0]
fn pdf_log_series(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LogSeriesDistribution expects 1 argument".into(),
    ));
  }
  let t = dargs[0].clone();
  let log_1mt = unary_fn("Log", minus(int(1), t.clone()));
  let density = times(
    int(-1),
    divide(power(t, x.clone()), times(x.clone(), log_1mt)),
  );
  // For a concrete integer k, pick the branch directly. Feeding k = 0 through
  // the piecewise would evaluate the density (k in the denominator) and emit a
  // spurious Power::infy message even though the 0 branch is selected.
  if let Expr::Integer(k) = &x {
    return if *k >= 1 { eval(density) } else { Ok(int(0)) };
  }
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(1));
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[NakagamiDistribution[m, w], x] = Piecewise[{{
///   (2 (m/w)^m x^(-1 + 2 m))/(E^((m x^2)/w) Gamma[m]), x > 0}}, 0]
fn pdf_nakagami(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NakagamiDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let w = dargs[1].clone();
  // numerator = 2 (m/w)^m x^(-1 + 2 m)
  let numer = times(
    times(int(2), power(divide(m.clone(), w.clone()), m.clone())),
    power(x.clone(), plus(int(-1), times(int(2), m.clone()))),
  );
  // denominator = E^((m x^2)/w) Gamma[m]
  let denom = times(
    power(e(), divide(times(m.clone(), power(x.clone(), int(2))), w)),
    unary_fn("Gamma", m),
  );
  let density = divide(numer, denom);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// CDF[NakagamiDistribution[m, w], x] =
///   Piecewise[{{GammaRegularized[m, 0, (m x^2)/w], x > 0}}, 0]
fn cdf_nakagami(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NakagamiDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let w = dargs[1].clone();
  let value = call(
    "GammaRegularized",
    vec![
      m.clone(),
      int(0),
      divide(times(m, power(x.clone(), int(2))), w),
    ],
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// PDF[LogLogisticDistribution[g, s], x] =
///   Piecewise[{{(g x^(-1 + g))/(s^g (1 + (x/s)^g)^2), x > 0}}, 0]
fn pdf_log_logistic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogLogisticDistribution expects 2 arguments".into(),
    ));
  }
  let g = dargs[0].clone();
  let s = dargs[1].clone();
  // numerator = g x^(-1 + g)
  let numer = times(g.clone(), power(x.clone(), plus(int(-1), g.clone())));
  // denominator = s^g (1 + (x/s)^g)^2
  let denom = times(
    power(s.clone(), g.clone()),
    power(plus(int(1), power(divide(x.clone(), s), g)), int(2)),
  );
  let density = divide(numer, denom);
  // Concrete x <= 0 is outside the support. Evaluating the density there would
  // raise x^(-1 + g) (or (x/s)^g) at x = 0 for g < 1 and emit a spurious
  // Power::infy message even though the piecewise selects the 0 branch.
  if let Some(xv) = crate::functions::math_ast::expr_to_num(&x)
    && xv <= 0.0
  {
    return Ok(int(0));
  }
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// CDF[LogLogisticDistribution[g, s], x] =
///   Piecewise[{{(1 + (x/s)^(-g))^(-1), x > 0}}, 0]
fn cdf_log_logistic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogLogisticDistribution expects 2 arguments".into(),
    ));
  }
  let g = dargs[0].clone();
  let s = dargs[1].clone();
  let value = power(
    plus(int(1), power(divide(x.clone(), s), times(int(-1), g))),
    int(-1),
  );
  // Concrete x <= 0 is outside the support; return 0 directly. At x = 0 the
  // value has (x/s)^(-g) = 0^(-g), which would emit a spurious Power::infy
  // message even though the piecewise selects the 0 branch.
  if let Some(xv) = crate::functions::math_ast::expr_to_num(&x)
    && xv <= 0.0
  {
    return Ok(int(0));
  }
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// ─── CDF ──────────────────────────────────────────────────────────────

/// CDF[dist, x] - Cumulative distribution function
/// Distributions whose sample value is itself a list (a single multivariate
/// point), so a list argument to PDF/CDF must NOT be threaded over.
fn is_multivariate_distribution(name: &str) -> bool {
  matches!(
    name,
    "MultinormalDistribution"
      | "BinormalDistribution"
      | "MultivariatePoissonDistribution"
      | "MultinomialDistribution"
      | "NegativeMultinomialDistribution"
      | "DirichletDistribution"
      | "ProductDistribution"
      | "CopulaDistribution"
  )
}

pub fn cdf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "CDF expects 1 or 2 arguments".into(),
    ));
  }

  let dist = &args[0];

  // CDF over process time slices delegates to the slice distribution.
  if args.len() == 2
    && let Expr::CurriedCall { func, args: targs } = dist
    && let Expr::FunctionCall { name, args: dargs } = func.as_ref()
    && targs.len() == 1
    && let Some(slice) = process_slice_distribution(name, dargs, &targs[0])
  {
    return cdf_ast(&[slice, args[1].clone()]);
  }

  let (dist_name, dargs) = match dist {
    Expr::FunctionCall { name, args: dargs } => {
      (name.as_str(), dargs.as_slice())
    }
    _ => {
      return Ok(unevaluated("CDF", args));
    }
  };

  if args.len() == 1 {
    return Ok(unevaluated("CDF", args));
  }

  // CDF[MixtureDistribution[{w1, …}, {d1, …}], x] is the weight-normalized sum
  // of the component CDFs: Σ w_i CDF[d_i, x] / Σ w_i.
  if dist_name == "MixtureDistribution" && dargs.len() == 2 {
    let x = args[1].clone();
    return Ok(
      super::statistics::mixture_weighted_component_quantity(dargs, |d| {
        cdf_ast(&[d.clone(), x.clone()])
      })?
      .unwrap_or_else(|| unevaluated("CDF", args)),
    );
  }

  // ErlangDistribution[k, λ] == GammaDistribution[k, 1/λ]
  if dist_name == "ErlangDistribution" && dargs.len() == 2 {
    let gamma = call("GammaDistribution", erlang_gamma_dargs(dargs)?);
    return cdf_ast(&[gamma, args[1].clone()]);
  }

  // For a univariate distribution, a list of values means "evaluate the CDF at
  // each point" — thread over it. (Multivariate distributions take a list as a
  // single point, so they are excluded.) Without this, discrete CDFs built as
  // a Piecewise leak a list into the Piecewise condition.
  if let Expr::List(xs) = &args[1]
    && !is_multivariate_distribution(dist_name)
  {
    let results: Result<Vec<Expr>, InterpreterError> = xs
      .iter()
      .map(|xi| cdf_ast(&[args[0].clone(), xi.clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let x = args[1].clone();

  match dist_name {
    "ProbabilityDistribution" => cdf_probability_distribution(dargs, x),
    "BenfordDistribution" => cdf_benford(dargs, x),
    "BenktanderWeibullDistribution" => cdf_benktander_weibull(dargs, x),
    "SinghMaddalaDistribution" => cdf_singh_maddala(dargs, x),
    "UniformSumDistribution" => cdf_uniform_sum(dargs, x),
    "BetaBinomialDistribution" => cdf_beta_binomial(dargs, x),
    "BetaPrimeDistribution" => cdf_beta_prime(dargs, x),
    "NoncentralChiSquareDistribution" => cdf_noncentral_chi_square(dargs, x),
    "ExponentialPowerDistribution" => cdf_exponential_power(dargs, x),
    "RiceDistribution" => cdf_rice(dargs, x),
    "MinStableDistribution" => cdf_min_stable(dargs, x),
    "MaxStableDistribution" => cdf_max_stable(dargs, x),
    "TriangularDistribution" => cdf_triangular(dargs, x),
    "MaxwellDistribution" => cdf_maxwell(dargs, x),
    "BirnbaumSaundersDistribution" => cdf_birnbaum_saunders(dargs, x),
    "LevyDistribution" => cdf_levy(dargs, x),
    "LindleyDistribution" => cdf_lindley(dargs, x),
    "WignerSemicircleDistribution" => cdf_wigner_semicircle(dargs, x),
    "SechDistribution" => cdf_sech(dargs, x),
    "MoyalDistribution" => cdf_moyal(dargs, x),
    "BenktanderGibratDistribution" => cdf_benktander_gibrat(dargs, x),
    "GumbelDistribution" => cdf_gumbel(dargs, x),
    "SkewNormalDistribution" => cdf_skew_normal(dargs, x),
    "NormalDistribution" => cdf_normal(dargs, x),
    "DataDistribution" => match histogram_pdf_cdf(dargs, &x, true)
      .or_else(|| data_distribution_pdf_cdf(dargs, &x, true).map(Ok))
    {
      Some(v) => v,
      None => Ok(call("CDF", vec![unevaluated("DataDistribution", dargs), x])),
    },
    "UniformDistribution" => cdf_uniform(dargs, x),
    "ExponentialDistribution" => cdf_exponential(dargs, x),
    "PoissonDistribution" => cdf_poisson(dargs, x),
    "SkellamDistribution" => cdf_skellam(dargs, x),
    "HypoexponentialDistribution" => cdf_hypoexponential(dargs, x),
    "CoxianDistribution" => cdf_coxian(dargs, x),
    "HyperexponentialDistribution" => cdf_hyperexponential(dargs, x),
    "BeniniDistribution" => cdf_benini(dargs, x),
    "HotellingTSquareDistribution" => cdf_hotelling(dargs, x),
    "TukeyLambdaDistribution" => tukey_lambda_pdf_cdf(dargs, x, false),
    "TsallisQGaussianDistribution" => tsallis_qgaussian_cdf(dargs, x),
    "FailureDistribution" => cdf_failure_distribution(dargs, x),
    "FirstPassageTimeDistribution" => cdf_first_passage(dargs, x),
    "BernoulliDistribution" => cdf_bernoulli(dargs, x),
    "BinomialDistribution" => cdf_binomial(dargs, x),
    "NegativeBinomialDistribution" => cdf_negative_binomial(dargs, x),
    "PascalDistribution" => cdf_pascal(dargs, x),
    "InverseGammaDistribution" => cdf_inverse_gamma(dargs, x),
    "GammaDistribution" => cdf_gamma(dargs, x),
    "BetaDistribution" => cdf_beta(dargs, x),
    "KumaraswamyDistribution" => cdf_kumaraswamy(dargs, x),
    "PowerDistribution" => cdf_power(dargs, x),
    "PERTDistribution" => cdf_pert(dargs, x),
    "ExpGammaDistribution" => cdf_expgamma(dargs, x),
    "LogGammaDistribution" => cdf_loggamma(dargs, x),
    "LogNormalDistribution" => cdf_lognormal(dargs, x),
    "ChiSquareDistribution" => cdf_chi_square(dargs, x),
    "ParetoDistribution" => cdf_pareto(dargs, x),
    "WeibullDistribution" => cdf_weibull(dargs, x),
    "GeometricDistribution" => cdf_geometric(dargs, x),
    "LogSeriesDistribution" => cdf_log_series(dargs, x),
    "NakagamiDistribution" => cdf_nakagami(dargs, x),
    "LogLogisticDistribution" => cdf_log_logistic(dargs, x),
    "CauchyDistribution" => cdf_cauchy(dargs, x),
    "DiscreteUniformDistribution" => cdf_discrete_uniform(dargs, x),
    "LaplaceDistribution" => cdf_laplace(dargs, x),
    "RayleighDistribution" => cdf_rayleigh(dargs, x),
    "NegativeMultinomialDistribution" => cdf_negative_multinomial(dargs, x),
    "HalfNormalDistribution" => cdf_half_normal(dargs, x),
    "ChiDistribution" => cdf_chi(dargs, x),
    "LogisticDistribution" => cdf_logistic(dargs, x),
    "InverseChiSquareDistribution" => cdf_inverse_chi_square(dargs, x),
    "FrechetDistribution" => cdf_frechet(dargs, x),
    "ExtremeValueDistribution" => cdf_extreme_value(dargs, x),
    "GompertzMakehamDistribution" => cdf_gompertz_makeham(dargs, x),
    "InverseGaussianDistribution" => cdf_inverse_gaussian(dargs, x),
    "StableDistribution" => cdf_stable(dargs, x),
    "StudentTDistribution" => cdf_student_t(dargs, x),
    "FRatioDistribution" => cdf_f_ratio(dargs, x),
    "WaringYuleDistribution" => cdf_waring_yule(dargs, x),
    "JohnsonDistribution" => cdf_johnson(dargs, x),
    _ => Ok(unevaluated("CDF", args)),
  }
}

/// CDF[NormalDistribution[mu, sigma], x] = Erfc[(mu - x)/(Sqrt[2]*sigma)]/2
fn cdf_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (mu, sigma) = match dargs.len() {
    0 => (int(0), int(1)),
    2 => (dargs[0].clone(), dargs[1].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "NormalDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };

  // Erfc[(mu - x) / (Sqrt[2] * sigma)] / 2
  let erfc_arg = divide(minus(mu, x), times(sqrt(int(2)), sigma));
  let erfc_call = call("Erfc", vec![erfc_arg]);
  let result = divide(erfc_call, int(2));
  eval(result)
}

/// CDF[UniformDistribution[{a, b}], x] = Piecewise[{{(x-a)/(b-a), a<=x<=b}, {1, x>b}}, 0]
fn cdf_uniform(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    1 => {
      if let Expr::List(bounds) = &dargs[0] {
        if bounds.len() == 2 {
          (bounds[0].clone(), bounds[1].clone())
        } else {
          return Err(InterpreterError::EvaluationError(
            "UniformDistribution: expected {min, max}".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "UniformDistribution: expected a list {min, max}".into(),
        ));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "UniformDistribution expects 0 or 1 argument".into(),
      ));
    }
  };

  let value = eval(divide(
    minus(x.clone(), a.clone()),
    minus(b.clone(), a.clone()),
  ))?;
  let cond_middle = comparison3(
    a,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::LessEqual,
    b.clone(),
  );
  let cond_above = comparison(x, ComparisonOp::Greater, b);

  eval(piecewise(
    vec![(value, cond_middle), (int(1), cond_above)],
    int(0),
  ))
}

/// CDF[ExponentialDistribution[lambda], x] = Piecewise[{{1 - E^(-lambda*x), x >= 0}}, 0]
fn cdf_exponential(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExponentialDistribution expects 1 argument".into(),
    ));
  }
  let lambda = dargs[0].clone();

  let value = eval(minus(int(1), power(e(), times(neg(lambda), x.clone()))))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[PoissonDistribution[mu], k] = Piecewise[{{GammaRegularized[Floor[k]+1, mu], k >= 0}}, 0]
/// For integer k, this equals sum_{i=0}^{k} mu^i * E^(-mu) / i!
fn cdf_poisson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PoissonDistribution expects 1 argument".into(),
    ));
  }
  let mu = dargs[0].clone();

  // GammaRegularized[Floor[k] + 1, mu]
  let floor_k_plus_1 = plus(call("Floor", vec![x.clone()]), int(1));
  let value = call("GammaRegularized", vec![floor_k_plus_1, mu]);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));

  let result = piecewise(vec![(value, cond)], int(0));
  eval(result)
}

/// CDF[BernoulliDistribution[p], k] = Piecewise[{{0, k<0}, {1-p, 0<=k<1}}, 1]
fn cdf_bernoulli(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BernoulliDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();

  let one_minus_p = eval(minus(int(1), p))?;
  let cond_neg = comparison(x.clone(), ComparisonOp::Less, int(0));
  let cond_middle = comparison3(
    int(0),
    ComparisonOp::LessEqual,
    x,
    ComparisonOp::Less,
    int(1),
  );

  eval(piecewise(
    vec![(int(0), cond_neg), (one_minus_p, cond_middle)],
    int(1),
  ))
}

/// CDF[BinomialDistribution[n, p], k] =
///   Piecewise[{{BetaRegularized[1 - p, n - Floor[k], 1 + Floor[k]],
///               0 <= k < n}, {1, k >= n}}, 0]
fn cdf_binomial(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();

  let floor_k = call("Floor", vec![x.clone()]);
  // BetaRegularized[1 - p, n - Floor[k], 1 + Floor[k]] is the regularized
  // incomplete beta form of the binomial CDF; it collapses to the exact
  // rational at numeric points and stays symbolic otherwise.
  let value = call(
    "BetaRegularized",
    vec![
      minus(int(1), p),
      minus(n.clone(), floor_k.clone()),
      plus(int(1), floor_k),
    ],
  );
  let cond_mid = comparison3(
    int(0),
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::Less,
    n.clone(),
  );
  let cond_high = comparison(x, ComparisonOp::GreaterEqual, n);

  eval(piecewise(
    vec![(value, cond_mid), (int(1), cond_high)],
    int(0),
  ))
}

/// CDF[LogSeriesDistribution[t], k] =
///   Piecewise[{{1 + Beta[t, 1 + Floor[k], 0]/Log[1 - t], k >= 1}}, 0]
fn cdf_log_series(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LogSeriesDistribution expects 1 argument".into(),
    ));
  }
  let t = dargs[0].clone();
  let log_1mt = unary_fn("Log", minus(int(1), t.clone()));
  // Beta[t, 1 + Floor[k], 0] — the incomplete beta B_t(1 + Floor[k], 0).
  let beta = call(
    "Beta",
    vec![t, plus(int(1), unary_fn("Floor", x.clone())), int(0)],
  );
  let value = plus(int(1), divide(beta, log_1mt));
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(1));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[GeometricDistribution[p], k] = Piecewise[{{1 - (1-p)^(Floor[k]+1), k >= 0}}, 0]
fn cdf_geometric(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "GeometricDistribution expects 1 argument".into(),
    ));
  }
  let p = dargs[0].clone();
  let one_minus_p = minus(int(1), p);
  let floor_k_plus_1 = plus(call("Floor", vec![x.clone()]), int(1));
  let value = minus(int(1), power(one_minus_p, floor_k_plus_1));
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[NegativeBinomialDistribution[n, p], k] =
///   Piecewise[{{BetaRegularized[p, n, 1 + Floor[k]], k >= 0}}, 0]
/// (the regularized incomplete beta form; collapses to the exact rational at
/// integer points and stays symbolic otherwise).
fn cdf_negative_binomial(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NegativeBinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();
  let floor_k = call("Floor", vec![x.clone()]);
  let value = call("BetaRegularized", vec![p, n, plus(int(1), floor_k)]);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[PascalDistribution[n, p], k] =
///   Piecewise[{{BetaRegularized[p, n, 1 - n + Floor[k]], k >= n}}, 0]
/// The Pascal distribution counts the number of trials until the n-th success,
/// so its support starts at k = n (vs. k = 0 for NegativeBinomialDistribution).
fn cdf_pascal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PascalDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();
  let floor_k = call("Floor", vec![x.clone()]);
  // 1 - n + Floor[k]
  let third = plus(minus(int(1), n.clone()), floor_k);
  let value = call("BetaRegularized", vec![p, n.clone(), third]);
  let cond = comparison(x, ComparisonOp::GreaterEqual, n);
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[CauchyDistribution[a, b], x] = 1/2 + ArcTan[(x-a)/b]/Pi
fn cdf_cauchy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let (a, b) = match dargs.len() {
    0 => (int(0), int(1)),
    2 => (dargs[0].clone(), dargs[1].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "CauchyDistribution expects 0 or 2 arguments".into(),
      ));
    }
  };
  // 1/2 + ArcTan[(x - a) / b] / Pi
  let arctan = call("ArcTan", vec![divide(minus(x, a), b)]);
  eval(plus(
    call("Rational", vec![int(1), int(2)]),
    divide(arctan, pi()),
  ))
}

/// PDF[GammaDistribution[alpha, beta], x] = Piecewise[{{x^(alpha-1) E^(-x/beta) / (beta^alpha Gamma[alpha]), x > 0}}, 0]
fn pdf_gamma(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GammaDistribution expects 2 arguments".into(),
    ));
  }
  let alpha = dargs[0].clone();
  let beta = dargs[1].clone();

  // x^(alpha-1)
  let x_part = power(x.clone(), minus(alpha.clone(), int(1)));
  // E^(-x/beta)
  let exp_part = power(e(), neg(divide(x.clone(), beta.clone())));
  // beta^alpha * Gamma[alpha]
  let denom = times(power(beta, alpha.clone()), call("Gamma", vec![alpha]));
  let value = eval(divide(times(x_part, exp_part), denom))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[GammaDistribution[alpha, beta], x] = Piecewise[{{GammaRegularized[alpha, 0, x/beta], x > 0}}, 0]
fn cdf_gamma(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GammaDistribution expects 2 arguments".into(),
    ));
  }
  let alpha = dargs[0].clone();
  let beta = dargs[1].clone();

  let value = call(
    "GammaRegularized",
    vec![alpha, int(0), divide(x.clone(), beta)],
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[InverseGammaDistribution[a, b], x] = Piecewise[{{(b/x)^a/(E^(b/x)*x*Gamma[a]), x > 0}}, 0]
fn pdf_inverse_gamma(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGammaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  // (b/x)^a
  let bx_a = power(divide(b.clone(), x.clone()), a.clone());
  // E^(b/x)
  let exp_part = power(e(), divide(b, x.clone()));
  // x * Gamma[a]
  let denom = times(x.clone(), call("Gamma", vec![a]));
  let value = eval(divide(bx_a, times(exp_part, denom)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[InverseGammaDistribution[a, b], x] = Piecewise[{{GammaRegularized[a, b/x], x > 0}}, 0]
fn cdf_inverse_gamma(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGammaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let value = call("GammaRegularized", vec![a, divide(b, x.clone())]);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[LogisticDistribution[m, s], x] = E^((m - x)/s)/((1 + E^((m - x)/s))^2*s)
fn pdf_logistic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogisticDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let s = dargs[1].clone();

  // E^((m - x)/s)
  let exp_val = power(e(), divide(minus(m, x), s.clone()));
  // (1 + E^(...))^2
  let denom_sq = power(plus(int(1), exp_val.clone()), int(2));
  let value = divide(exp_val, times(denom_sq, s));
  eval(value)
}

// CDF[LogisticDistribution[m, s], x] = 1/(1 + E^((m - x)/s))
fn cdf_logistic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogisticDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let s = dargs[1].clone();

  let exp_val = power(e(), divide(minus(m, x), s));
  let denom = plus(int(1), exp_val);
  eval(power(denom, int(-1)))
}

// PDF[InverseChiSquareDistribution[n], x] = Piecewise[{{(x^(-1))^(1+n/2)/(2^(n/2)*E^(1/(2*x))*Gamma[n/2]), x > 0}}, 0]
fn pdf_inverse_chi_square(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();

  // (x^(-1))^(1 + n/2) / (2^(n/2) * E^(1/(2*x)) * Gamma[n/2])
  // Use BinaryOp Power for (x^(-1))^(1+n/2) which the evaluator flattens.
  // Then wrap in divide which puts it in the numerator.
  let x_part = power(
    power(x.clone(), int(-1)),
    plus(int(1), divide(n.clone(), int(2))),
  );
  // 2^(n/2)
  let two_part = power(int(2), divide(n.clone(), int(2)));
  // E^(1/(2*x))
  let exp_part = power(e(), divide(int(1), times(int(2), x.clone())));
  // Gamma[n/2]
  let gamma_part = call("Gamma", vec![divide(n, int(2))]);
  let value =
    eval(divide(x_part, times(two_part, times(exp_part, gamma_part))))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[InverseChiSquareDistribution[n], x] = Piecewise[{{GammaRegularized[n/2, 1/(2*x)], x > 0}}, 0]
fn cdf_inverse_chi_square(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();

  let value = call(
    "GammaRegularized",
    vec![divide(n, int(2)), divide(int(1), times(int(2), x.clone()))],
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[FrechetDistribution[a, b], x] = Piecewise[{{(a*(x/b)^(-1 - a))/(b*E^(x/b)^(-a)), x > 0}}, 0]
// PDF[FrechetDistribution[a, b, c], x] = Piecewise[{{(a*((x-c)/b)^(-1 - a))/(b*E^((x-c)/b)^(-a)), x > c}}, 0]
fn pdf_frechet(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 && dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "FrechetDistribution expects 2 or 3 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();
  let (shifted, threshold) = if dargs.len() == 3 {
    let c = dargs[2].clone();
    (minus(x.clone(), c.clone()), c)
  } else {
    (x.clone(), int(0))
  };

  let xb = divide(shifted, b.clone());
  // ((x-c)/b)^(-1 - a)
  let xb_part = power(xb.clone(), neg(plus(int(1), a.clone())));
  // E^((x-c)/b)^(-a)
  let exp_part = power(e(), power(xb, neg(a.clone())));
  let value = eval(divide(times(a, xb_part), times(b, exp_part)))?;
  let cond = comparison(x, ComparisonOp::Greater, threshold);
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[FrechetDistribution[a, b], x] = Piecewise[{{E^(-(x/b)^(-a)), x > 0}}, 0]
// CDF[FrechetDistribution[a, b, c], x] = Piecewise[{{E^(-((x-c)/b)^(-a)), x > c}}, 0]
fn cdf_frechet(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 && dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "FrechetDistribution expects 2 or 3 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();
  let (shifted, threshold) = if dargs.len() == 3 {
    let c = dargs[2].clone();
    (minus(x.clone(), c.clone()), c)
  } else {
    (x.clone(), int(0))
  };

  let value = power(e(), neg(power(divide(shifted, b), neg(a))));
  let value = eval(value)?;
  let cond = comparison(x, ComparisonOp::Greater, threshold);
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[ExtremeValueDistribution[a, b], x] = E^(-E^((a - x)/b) + (a - x)/b)/b
fn pdf_extreme_value(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ExtremeValueDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let ab = divide(minus(a.clone(), x), b.clone());
  // E^(-E^((a-x)/b) + (a-x)/b)
  let exp_arg = plus(neg(power(e(), ab.clone())), ab);
  let exp_arg_eval = eval(exp_arg)?;
  eval(divide(power(e(), exp_arg_eval), b))
}

// CDF[ExtremeValueDistribution[a, b], x] = E^(-E^((a - x)/b))
fn cdf_extreme_value(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ExtremeValueDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let ab = divide(minus(a, x), b);
  eval(power(e(), neg(power(e(), ab))))
}

// PDF[GompertzMakehamDistribution[l, x0], x] = Piecewise[{{E^(l*x + (1 - E^(l*x))*x0)*l*x0, x >= 0}}, 0]
fn pdf_gompertz_makeham(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GompertzMakehamDistribution expects 2 arguments".into(),
    ));
  }
  let l = dargs[0].clone();
  let x0 = dargs[1].clone();

  // l*x
  let lx = times(l.clone(), x.clone());
  // E^(l*x)
  let e_lx = power(e(), lx.clone());
  // (1 - E^(l*x))*x0
  let inner = times(minus(int(1), e_lx), x0.clone());
  // l*x + (1 - E^(l*x))*x0
  let exp_arg = plus(lx, inner);
  let value = eval(times(times(power(e(), exp_arg), l), x0))?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[GompertzMakehamDistribution[l, x0], x] = Piecewise[{{1 - E^((1 - E^(l*x))*x0), x >= 0}}, 0]
fn cdf_gompertz_makeham(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GompertzMakehamDistribution expects 2 arguments".into(),
    ));
  }
  let l = dargs[0].clone();
  let x0 = dargs[1].clone();

  let e_lx = power(e(), times(l, x.clone()));
  let inner = times(minus(int(1), e_lx), x0);
  let value = minus(int(1), power(e(), inner));
  let value = eval(value)?;
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// PDF[InverseGaussianDistribution[m, l], x] = Piecewise[{{Sqrt[l/x^3]/(E^((l*(-m+x)^2)/(2*m^2*x))*Sqrt[2*Pi]), x > 0}}, 0]
fn pdf_inverse_gaussian(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGaussianDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let l = dargs[1].clone();

  // Sqrt[l/x^3]
  let numer = sqrt(divide(l.clone(), power(x.clone(), int(3))));
  // l*(-m+x)^2 / (2*m^2*x)
  let exp_arg = divide(
    times(l, power(minus(x.clone(), m.clone()), int(2))),
    times(int(2), times(power(m, int(2)), x.clone())),
  );
  // E^(exp_arg)
  let exp_part = power(e(), exp_arg);
  // Sqrt[2*Pi]
  let sqrt_2pi = sqrt(times(int(2), pi()));
  let value = eval(divide(numer, times(exp_part, sqrt_2pi)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// CDF[InverseGaussianDistribution[m, l], x]
fn cdf_inverse_gaussian(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseGaussianDistribution expects 2 arguments".into(),
    ));
  }
  let m = dargs[0].clone();
  let l = dargs[1].clone();

  // Sqrt[l/x]
  let sqrt_lx = sqrt(divide(l.clone(), x.clone()));
  // Erfc[((m - x)*Sqrt[l/x])/(Sqrt[2]*m)]/2
  let erfc1_arg = divide(
    times(minus(m.clone(), x.clone()), sqrt_lx.clone()),
    times(sqrt(int(2)), m.clone()),
  );
  let erfc1 = divide(call("Erfc", vec![erfc1_arg]), int(2));
  // E^((2*l)/m) * Erfc[(Sqrt[l/x]*(m + x))/(Sqrt[2]*m)]/2
  let exp_part = power(e(), divide(times(int(2), l), m.clone()));
  let erfc2_arg = divide(
    times(sqrt_lx, plus(m.clone(), x.clone())),
    times(sqrt(int(2)), m),
  );
  let erfc2 = divide(call("Erfc", vec![erfc2_arg]), int(2));
  let value = eval(plus(erfc1, times(exp_part, erfc2)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

// ─── Probability ─────────────────────────────────────────────────────

/// Probability[event, x \[Distributed] dist]
/// Probability[event, x \[Distributed] d1 && y \[Distributed] d2 && ...]
/// event can be: x > a, x < a, x >= a, x <= a, x == k, a < x < b, etc.
pub fn probability_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Probability expects 2 arguments".into(),
    ));
  }

  let event = &args[0];
  let dist_spec = &args[1];

  // A certain / impossible event has probability 1 / 0 under any distribution.
  if let Expr::Identifier(b) = event {
    if b == "True" {
      return Ok(Expr::Integer(1));
    }
    if b == "False" {
      return Ok(Expr::Integer(0));
    }
  }

  // Conditional probability: P[A \[Conditioned] B, x ~ d] = P[A && B, x ~ d] / P[B, x ~ d].
  if let Expr::FunctionCall { name, args: cargs } = event
    && name == "Conditioned"
    && cargs.len() == 2
  {
    let joint = call("And", vec![cargs[0].clone(), cargs[1].clone()]);
    let p_joint = probability_ast(&[joint, dist_spec.clone()])?;
    let p_cond = probability_ast(&[cargs[1].clone(), dist_spec.clone()])?;
    return eval(divide(p_joint, p_cond));
  }
  let unevaluated = || Ok(unevaluated("Probability", args));
  // Joint distribution form:
  //   x \[Distributed] d1 && y \[Distributed] d2 && ...
  // Handle it by enumerating the finite discrete support of every variable.
  if let Some(pairs) = collect_distributed_pairs(dist_spec)
    && pairs.len() >= 2
  {
    if let Some(result) = try_joint_probability_discrete(event, &pairs)? {
      return eval(call("Together", vec![result]));
    }
    return unevaluated();
  }

  // Parse Distributed[var, dist] from the second argument
  let (var_name, dist) = match dist_spec {
    Expr::FunctionCall { name, args: dargs }
      if name == "Distributed" && dargs.len() == 2 =>
    {
      if let Expr::Identifier(v) = &dargs[0] {
        (v.as_str(), &dargs[1])
      } else {
        return unevaluated();
      }
    }
    _ => {
      return unevaluated();
    }
  };

  // Empirical (data-list) distribution:
  //   Probability[event, x \[Distributed] {d1, …, dn}]
  // is the fraction of data points that satisfy the event. Only comparison
  // events (inequalities/equalities and their logical combinations) are
  // treated this way; other predicates (e.g. EvenQ[x]) fall through to the
  // event-algebra path, matching wolframscript.
  if let Expr::List(data) = dist
    && !data.is_empty()
    && is_comparison_event(event)
  {
    let mut count: i128 = 0;
    for d in data.iter() {
      let substituted = crate::evaluator::evaluate_expr_to_expr(&call(
        "ReplaceAll",
        vec![
          event.clone(),
          Expr::Rule {
            pattern: Box::new(Expr::Identifier(var_name.to_string())),
            replacement: Box::new(d.clone()),
          },
        ],
      ))?;
      if matches!(&substituted, Expr::Identifier(s) if s == "True") {
        count += 1;
      }
    }
    return eval(divide(int(count), int(data.len() as i128)));
  }

  // P[event, x \[Distributed] ProbabilityDistribution[pdf, {x, lo, hi}]]
  // = Integrate[pdf * Boole[event], {x, lo, hi}], reduced to a sub-range
  // integral for simple `x > k` / `x < k` / `a < x < b` events.
  if let Expr::FunctionCall { name, args: dargs } = dist
    && name == "ProbabilityDistribution"
    && let Some(result) =
      try_probability_probability_distribution(event, var_name, dargs)?
  {
    return Ok(result);
  }

  // Determine if distribution is discrete
  let is_discrete = matches!(
    dist,
    Expr::FunctionCall { name, .. }
    if matches!(name.as_str(), "PoissonDistribution" | "BernoulliDistribution" | "BinomialDistribution" | "GeometricDistribution" | "NegativeBinomialDistribution" | "DiscreteUniformDistribution" | "PascalDistribution")
  );

  // Parse the event condition and compute probability
  let result = probability_from_event(event, var_name, dist, is_discrete)?;
  // Apply Together to normalize fractions (e.g. 1 - E^(-2) → (-1 + E^2)/E^2)
  eval(call("Together", vec![result]))
}

/// True when `event` is a comparison chain (inequality/equality) or a logical
/// combination of such — i.e. an event wolframscript interprets as a subset of
/// the sample space. Arbitrary predicate calls (e.g. `EvenQ[x]`) are not
/// comparison events, so the empirical-probability path leaves them alone.
fn is_comparison_event(event: &Expr) -> bool {
  match event {
    Expr::Comparison { .. } => true,
    Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "And"
          | "Or"
          | "Not"
          | "Nor"
          | "Nand"
          | "Xor"
          | "Implies"
          | "Equivalent"
      ) =>
    {
      !args.is_empty() && args.iter().all(is_comparison_event)
    }
    _ => false,
  }
}

fn probability_from_event(
  event: &Expr,
  var: &str,
  dist: &Expr,
  is_discrete: bool,
) -> Result<Expr, InterpreterError> {
  // Handle And conditions: a < x && x < b  →  a < x < b
  if let Expr::FunctionCall { name, args } = event
    && name == "And"
    && args.len() == 2
  {
    // Try to extract compound inequality: lower < x && x < upper
    if let (Some((lo, lo_op)), Some((hi, hi_op))) = (
      extract_bound(&args[0], var, true),
      extract_bound(&args[1], var, false),
    ) {
      // P[lo < x < hi] = CDF[dist, hi] - CDF[dist, lo]
      let cdf_hi = cdf_ast(&[dist.clone(), hi])?;
      let cdf_lo = cdf_ast(&[dist.clone(), lo])?;
      let result = minus(cdf_hi, cdf_lo);
      // Adjust for inclusive/exclusive bounds if needed (for continuous, same)
      let _ = (lo_op, hi_op); // For continuous distributions, < vs <= doesn't matter
      return eval(result);
    }
    // Try reversed: x < upper && lower < x
    if let (Some((hi, hi_op)), Some((lo, lo_op))) = (
      extract_bound(&args[0], var, false),
      extract_bound(&args[1], var, true),
    ) {
      let cdf_hi = cdf_ast(&[dist.clone(), hi])?;
      let cdf_lo = cdf_ast(&[dist.clone(), lo])?;
      let result = minus(cdf_hi, cdf_lo);
      let _ = (lo_op, hi_op);
      return eval(result);
    }
  }

  // Handle Comparison: a < x, x > a, a <= x <= b, etc.
  if let Expr::Comparison {
    operands,
    operators,
  } = event
  {
    // Two-operand comparison: x > a, x < a, x >= a, x <= a, x == a
    if operands.len() == 2 && operators.len() == 1 {
      let op = &operators[0];
      let (left, right) = (&operands[0], &operands[1]);

      let is_left_var = matches!(left, Expr::Identifier(n) if n == var);
      let is_right_var = matches!(right, Expr::Identifier(n) if n == var);

      // x == k → PDF[dist, k] for discrete distributions
      if *op == ComparisonOp::Equal {
        if is_left_var {
          return pdf_ast(&[dist.clone(), right.clone()]);
        }
        if is_right_var {
          return pdf_ast(&[dist.clone(), left.clone()]);
        }
      }

      // x > a → 1 - CDF[dist, a]
      if is_left_var
        && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
      {
        let cdf_val = cdf_ast(&[dist.clone(), right.clone()])?;
        return eval(minus(int(1), cdf_val));
      }
      // x < a → CDF[dist, a]
      if is_left_var
        && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
      {
        return cdf_ast(&[dist.clone(), right.clone()]);
      }
      // a > x → CDF[dist, a]  (same as x < a)
      if is_right_var
        && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
      {
        return cdf_ast(&[dist.clone(), left.clone()]);
      }
      // a < x → 1 - CDF[dist, a]
      if is_right_var
        && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
      {
        let cdf_val = cdf_ast(&[dist.clone(), left.clone()])?;
        return eval(minus(int(1), cdf_val));
      }
    }

    // Three-operand comparison: a < x < b, a <= x <= b
    if operands.len() == 3 && operators.len() == 2 {
      let (lo, mid, hi) = (&operands[0], &operands[1], &operands[2]);
      if matches!(mid, Expr::Identifier(n) if n == var) {
        // lo < x < hi → CDF[dist, hi] - CDF[dist, lo]
        let both_less = matches!(
          (&operators[0], &operators[1]),
          (
            ComparisonOp::Less | ComparisonOp::LessEqual,
            ComparisonOp::Less | ComparisonOp::LessEqual
          )
        );
        if both_less {
          let cdf_hi = cdf_ast(&[dist.clone(), hi.clone()])?;
          let cdf_lo = cdf_ast(&[dist.clone(), lo.clone()])?;
          return eval(minus(cdf_hi, cdf_lo));
        }
      }
    }
  }

  let _ = is_discrete;

  // Unevaluated fallback
  Ok(call(
    "Probability",
    vec![
      event.clone(),
      call(
        "Distributed",
        vec![Expr::Identifier(var.to_string()), dist.clone()],
      ),
    ],
  ))
}

/// Walk an `And[Distributed[x1, d1], Distributed[x2, d2], ...]` expression
/// (or a single `Distributed[x, d]`) and collect every `(var, dist)` pair.
/// Returns `None` if any branch isn't a well-formed `Distributed[...]`.
fn collect_distributed_pairs(expr: &Expr) -> Option<Vec<(String, Expr)>> {
  fn walk(expr: &Expr, out: &mut Vec<(String, Expr)>) -> bool {
    match expr {
      Expr::FunctionCall { name, args } if name == "And" => {
        args.iter().all(|a| walk(a, out))
      }
      Expr::FunctionCall { name, args }
        if name == "Distributed" && args.len() == 2 =>
      {
        if let Expr::Identifier(v) = &args[0] {
          out.push((v.clone(), args[1].clone()));
          true
        } else {
          false
        }
      }
      _ => false,
    }
  }
  let mut out = Vec::new();
  if walk(expr, &mut out) && !out.is_empty() {
    Some(out)
  } else {
    None
  }
}

/// Return the finite integer support of a discrete distribution as
/// `(Vec<Expr>, point_probability)` if and only if we can enumerate it
/// cheaply. Currently supports `DiscreteUniformDistribution[{a, b}]`.
fn discrete_finite_support(dist: &Expr) -> Option<(Vec<Expr>, Expr)> {
  let Expr::FunctionCall { name, args } = dist else {
    return None;
  };
  match name.as_str() {
    "DiscreteUniformDistribution" => {
      if args.len() != 1 {
        return None;
      }
      let Expr::List(range) = &args[0] else {
        return None;
      };
      if range.len() != 2 {
        return None;
      }
      let (Expr::Integer(lo), Expr::Integer(hi)) = (&range[0], &range[1])
      else {
        return None;
      };
      if hi < lo {
        return None;
      }
      let count = hi - lo + 1;
      let support: Vec<Expr> = (*lo..=*hi).map(Expr::Integer).collect();
      let prob = call("Rational", vec![int(1), int(count)]);
      Some((support, prob))
    }
    _ => None,
  }
}

/// Attempt to compute a joint discrete probability by enumerating the
/// Cartesian product of each variable's finite support. Returns `Ok(None)`
/// if any distribution isn't a supported discrete finite distribution.
fn try_joint_probability_discrete(
  event: &Expr,
  pairs: &[(String, Expr)],
) -> Result<Option<Expr>, InterpreterError> {
  // Collect (support, point_prob) for every variable.
  let mut per_var: Vec<(String, Vec<Expr>, Expr)> =
    Vec::with_capacity(pairs.len());
  for (var, dist) in pairs {
    let Some((support, prob)) = discrete_finite_support(dist) else {
      return Ok(None);
    };
    per_var.push((var.clone(), support, prob));
  }

  // Accumulate probability mass for all points where the event evaluates
  // to True after substituting the sampled values for each variable.
  let mut total: Expr = Expr::Integer(0);
  // Iterate the Cartesian product via an index vector.
  let sizes: Vec<usize> = per_var.iter().map(|(_, s, _)| s.len()).collect();
  let mut idx = vec![0usize; sizes.len()];
  loop {
    // Build the substituted event.
    let mut substituted = event.clone();
    let mut point_prob: Expr = Expr::Integer(1);
    for (i, (var, support, pprob)) in per_var.iter().enumerate() {
      substituted = crate::functions::plot::substitute_var(
        &substituted,
        var,
        &support[idx[i]],
      );
      point_prob = eval(call("Times", vec![point_prob, pprob.clone()]))?;
    }
    let evaluated = evaluate_expr_to_expr(&substituted)?;
    let is_true = matches!(&evaluated, Expr::Identifier(n) if n == "True");
    if is_true {
      total = eval(call("Plus", vec![total, point_prob]))?;
    }

    // Advance the index vector (little-endian odometer).
    let mut i = 0;
    loop {
      if i == sizes.len() {
        return Ok(Some(total));
      }
      idx[i] += 1;
      if idx[i] < sizes[i] {
        break;
      }
      idx[i] = 0;
      i += 1;
    }
  }
}

/// Extract a bound from a comparison involving the variable.
/// If `lower` is true, look for patterns like "a < x" or "a <= x" (lower bound = a).
/// If `lower` is false, look for patterns like "x < b" or "x <= b" (upper bound = b).
/// Returns (bound_value, operator).
fn extract_bound(
  expr: &Expr,
  var: &str,
  lower: bool,
) -> Option<(Expr, ComparisonOp)> {
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && operands.len() == 2
    && operators.len() == 1
  {
    let op = &operators[0];
    let (left, right) = (&operands[0], &operands[1]);
    let is_left_var = matches!(left, Expr::Identifier(n) if n == var);
    let is_right_var = matches!(right, Expr::Identifier(n) if n == var);

    if lower {
      // Looking for: a < x or a <= x (lower bound is a)
      if is_right_var
        && matches!(op, ComparisonOp::Less | ComparisonOp::LessEqual)
      {
        return Some((left.clone(), *op));
      }
      // x > a or x >= a (lower bound is a)
      if is_left_var
        && matches!(op, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
      {
        return Some((right.clone(), *op));
      }
    } else {
      // Looking for: x < b or x <= b (upper bound is b)
      if is_left_var
        && matches!(op, ComparisonOp::Less | ComparisonOp::LessEqual)
      {
        return Some((right.clone(), *op));
      }
      // b > x or b >= x (upper bound is b)
      if is_right_var
        && matches!(op, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
      {
        return Some((left.clone(), *op));
      }
    }
  }
  None
}

// ─── Expectation ─────────────────────────────────────────────────────

/// Expectation[f(x), x \[Distributed] dist]
/// Computes E[f(x)] for known distributions.
pub fn expectation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Expectation expects 2 arguments".into(),
    ));
  }

  let expr = &args[0];
  let dist_spec = &args[1];
  let unevaluated = || Ok(unevaluated("Expectation", args));

  // Parse Distributed[vars, dist] — vars may be a single Identifier or a List
  // (for multivariate distributions such as ProbabilityDistribution[..., {x,…}, {y,…}]).
  let (vars, dist) = match dist_spec {
    Expr::FunctionCall { name, args: dargs }
      if name == "Distributed" && dargs.len() == 2 =>
    {
      let vars = match &dargs[0] {
        Expr::Identifier(v) => vec![v.clone()],
        Expr::List(items) => {
          let mut names = Vec::with_capacity(items.len());
          for item in items.iter() {
            if let Expr::Identifier(n) = item {
              names.push(n.clone());
            } else {
              return unevaluated();
            }
          }
          names
        }
        _ => {
          return unevaluated();
        }
      };
      (vars, &dargs[1])
    }
    _ => {
      return unevaluated();
    }
  };

  // Empirical (data-list) distribution:
  //   Expectation[expr, x \[Distributed] {d1, …, dn}]
  // is the sample mean of `expr` with `x` replaced by each data point.
  if let Expr::List(data) = dist
    && vars.len() == 1
    && !data.is_empty()
  {
    let var = &vars[0];
    let substituted: Vec<Expr> = data
      .iter()
      .map(|d| {
        call(
          "ReplaceAll",
          vec![
            expr.clone(),
            Expr::Rule {
              pattern: Box::new(Expr::Identifier(var.clone())),
              replacement: Box::new(d.clone()),
            },
          ],
        )
      })
      .collect();
    return crate::evaluator::evaluate_expr_to_expr(&call(
      "Mean",
      vec![Expr::List(substituted.into())],
    ));
  }

  // Get distribution name and parameters
  let (dist_name, dargs) = match dist {
    Expr::FunctionCall { name, args: da } => (name.as_str(), da.as_slice()),
    _ => {
      return unevaluated();
    }
  };

  // ProbabilityDistribution[pdf, {x, lo, hi}, …] — evaluate as
  // Integrate[expr*pdf, {x, lo, hi}, …] after renaming the distribution's
  // dummy variables to the names supplied in `Distributed[vars, …]`.
  if dist_name == "ProbabilityDistribution"
    && let Some(result) =
      try_expectation_probability_distribution(expr, &vars, dargs)?
  {
    return Ok(result);
  }

  // CensoredDistribution[{a, b}, base] truncates `base` to the interval
  // [a, b], piling mass that falls outside onto the boundary. So
  //   E[f(x), x ~ CensoredDistribution[{a, b}, base]]
  //     = f(a)*P[X < a] + ∫_a^b f(x)*pdf_base(x) dx + f(b)*P[X > b]
  // where the boundary probabilities come from base's CDF.
  if dist_name == "CensoredDistribution"
    && vars.len() == 1
    && let Some(result) = try_expectation_censored(expr, &vars[0], dargs)?
  {
    return Ok(result);
  }

  // Additively-separable expectations against a multivariate
  // distribution with known marginals reduce to a sum of univariate
  // expectations. Each term in the Plus must touch at most one of the
  // distribution variables (constants pass through, mixed terms like
  // `x*y` keep the call symbolic).
  if vars.len() >= 2
    && let Some(marginals) = multivariate_marginals(dist_name, dargs, &vars)
    && let Some(result) =
      try_separable_multivariate_expectation(expr, &vars, &marginals)?
  {
    return Ok(result);
  }

  // For unsupported parameterizations (e.g. list-of-vars over a standard
  // distribution) just return unevaluated.
  if vars.len() != 1 {
    return unevaluated();
  }
  let var_name = vars.into_iter().next().unwrap();

  // Compute E[f(x)] using known moment formulas
  // First try to get mean and variance for common distributions
  let (mean, variance) = distribution_mean_variance(dist_name, dargs)?;

  // Check if expr is just the variable (E[x] = mean)
  if matches!(expr, Expr::Identifier(n) if *n == var_name) {
    return eval(mean);
  }

  // Check if expr is x^2 (E[x^2] = Var + Mean^2)
  if is_power_of_var(expr, &var_name, 2) {
    let result = plus(variance.clone(), power(mean.clone(), int(2)));
    return eval(result);
  }

  // Check for linear expressions: a*x + b
  if let Some((a, b)) = extract_linear(expr, &var_name) {
    // E[a*x + b] = a*E[x] + b
    let result = plus(times(a, mean), b);
    return eval(result);
  }

  // MGF identity: E[c · Exp[t·x]] = c · MGF_X(t). Recognised for normal
  // distributions where MGF(t) = exp(μ t + σ² t²/2), giving exact
  // symbolic results like `3 Sqrt[E]` instead of numerical surrogates.
  if dist_name == "NormalDistribution"
    && let Some((c, t)) = extract_mgf_pattern(expr, &var_name)
  {
    let mu = mean.clone();
    let sigma_sq = variance.clone();
    // exponent = μ t + σ² t² / 2
    let mu_t = times(mu, t.clone());
    let half = call("Rational", vec![int(1), int(2)]);
    let sigma_sq_t_sq_half =
      times(times(sigma_sq, power(t.clone(), int(2))), half);
    let exponent = plus(mu_t, sigma_sq_t_sq_half);
    let mgf = power(Expr::Identifier("E".to_string()), exponent);
    return eval(times(c, mgf));
  }

  // Polynomial integrands: exact raw moments (the numerical fallback
  // below truncates integration domains and used to return e.g. 0.7422
  // for E[x^3] under ExponentialDistribution[2] instead of 3/4)
  if let Some(result) =
    polynomial_expectation(expr, &var_name, dist_name, dargs)
  {
    return Ok(result);
  }

  // E[Boole[cond]] = Probability[cond, dist] — exact, rather than the numerical
  // integration below (which both loses precision and misses the clean closed
  // form, e.g. E[Boole[x > 0]] over a standard normal is exactly 1/2).
  if let Expr::FunctionCall { name, args: bargs } = expr
    && name == "Boole"
    && bargs.len() == 1
  {
    let prob = call("Probability", vec![bargs[0].clone(), dist_spec.clone()]);
    return crate::evaluator::evaluate_expr_to_expr(&prob);
  }

  // Try exact symbolic integration E[g(x)] = ∫ g(x) f(x) dx over the support
  // before falling back to quadrature, so e.g.
  // Expectation[Sin[x], x ∈ UniformDistribution[{0, Pi}]] = 2/Pi rather than a
  // numerical surrogate.
  if let Some(result) =
    try_symbolic_expectation(expr, &var_name, dist_name, dargs)
  {
    return Ok(result);
  }

  // For more complex expressions, use numerical integration
  expectation_numerical(expr, &var_name, dist_name, dargs)
}

/// Attempt an exact expectation via symbolic definite integration of
/// `g(x) f(x)` over the distribution's support. Returns None when the support
/// is not known symbolically or the definite integral does not evaluate.
fn try_symbolic_expectation(
  expr: &Expr,
  var: &str,
  dist_name: &str,
  dargs: &[Expr],
) -> Option<Expr> {
  // E[g(x)] over UniformDistribution[{a, b}] is Integrate[g, {x, a, b}]/(b-a).
  let (lo, hi) = match dist_name {
    "UniformDistribution" => match dargs {
      [] => (int(0), int(1)),
      [Expr::List(bounds)] if bounds.len() == 2 => {
        (bounds[0].clone(), bounds[1].clone())
      }
      _ => return None,
    },
    _ => return None,
  };
  let iter = Expr::List(
    vec![Expr::Identifier(var.to_string()), lo.clone(), hi.clone()].into(),
  );
  let integral = call("Integrate", vec![expr.clone(), iter]);
  let result = eval(divide(integral, minus(hi, lo))).ok()?;
  // A definite integral that could not be done leaves an Integrate head; reject
  // it so the numerical fallback runs.
  if expr_contains_head(&result, "Integrate") {
    return None;
  }
  Some(result)
}

/// Whether `expr` contains a `FunctionCall` with the given head anywhere.
fn expr_contains_head(expr: &Expr, head: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == head || args.iter().any(|a| expr_contains_head(a, head))
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_head(left, head) || expr_contains_head(right, head)
    }
    Expr::UnaryOp { operand, .. } => expr_contains_head(operand, head),
    Expr::List(items) => items.iter().any(|a| expr_contains_head(a, head)),
    _ => false,
  }
}

/// Match expressions of the form `c · Exp[t·x]` (with `c`, `t`
/// constant w.r.t. `x`) and return `(c, t)`. Handles bare `Exp[x]`
/// (c = 1, t = 1) and `Exp[x]^k` shapes too via the parser's standard
/// `E^…` rewrites.
fn extract_mgf_pattern(expr: &Expr, var: &str) -> Option<(Expr, Expr)> {
  // Strip a leading constant multiplier: extract `c` and remainder.
  let (c, rest) = strip_constant_multiplier(expr, var);
  // `rest` should now be `Exp[t · x]` (i.e. `Power[E, t·x]`).
  let exponent = match &rest {
    Expr::FunctionCall { name, args } if name == "Exp" && args.len() == 1 => {
      args[0].clone()
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      let base_is_e =
        matches!(&args[0], Expr::Identifier(n) | Expr::Constant(n) if n == "E");
      if !base_is_e {
        return None;
      }
      args[1].clone()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base_is_e = matches!(left.as_ref(), Expr::Identifier(n) | Expr::Constant(n) if n == "E");
      if !base_is_e {
        return None;
      }
      *right.clone()
    }
    _ => return None,
  };
  // exponent must be linear in `var`: a·var + b with b independent of var.
  let (a, b) = extract_linear(&exponent, var)?;
  // We only handle the homogeneous case b = 0; otherwise the factor
  // exp(b) would multiply through and that's better folded into `c`.
  let b_is_zero = matches!(&b, Expr::Integer(0))
    || matches!(&b, Expr::Real(v) if v.abs() < 1e-300);
  if !b_is_zero {
    return None;
  }
  Some((c, a))
}

/// Split `expr` into `(c, rest)` where `c` is constant w.r.t. `var`.
/// Pulls out a single Times-factor or returns `(1, expr)`.
fn strip_constant_multiplier(expr: &Expr, var: &str) -> (Expr, Expr) {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut consts: Vec<Expr> = Vec::new();
      let mut rest: Vec<Expr> = Vec::new();
      for arg in args.iter() {
        if contains_variable(arg, var) {
          rest.push(arg.clone());
        } else {
          consts.push(arg.clone());
        }
      }
      let c = match consts.len() {
        0 => Expr::Integer(1),
        1 => consts.remove(0),
        _ => call("Times", consts),
      };
      let r = match rest.len() {
        0 => Expr::Integer(1),
        1 => rest.remove(0),
        _ => call("Times", rest),
      };
      (c, r)
    }
    other => (Expr::Integer(1), other.clone()),
  }
}

/// Compute `Probability[event, x \[Distributed] ProbabilityDistribution[pdf, {x, lo, hi}]]`
/// by integrating the pdf over the sub-range carved out by `event`.
///
/// Supported event shapes (single variable, single iterator only):
///   - `x > k`, `x >= k`        → Integrate[pdf, {x, max(k,lo), hi}]
///   - `x < k`, `x <= k`        → Integrate[pdf, {x, lo, min(k,hi)}]
///   - `a < x < b`, `a ≤ x ≤ b` → Integrate[pdf, {x, max(a,lo), min(b,hi)}]
///   - `x \[Conditioned] y`     → P[x ∧ y] / P[y]
/// Returns `None` if the shape isn't recognised so the caller can fall back.
fn try_probability_probability_distribution(
  event: &Expr,
  var_name: &str,
  dargs: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::plot::substitute_var;

  if dargs.len() != 2 {
    return Ok(None);
  }
  let pdf_raw = &dargs[0];
  let iter = &dargs[1];

  let Expr::List(iter_items) = iter else {
    return Ok(None);
  };
  if iter_items.len() != 3 {
    return Ok(None);
  }
  let Expr::Identifier(dummy) = &iter_items[0] else {
    return Ok(None);
  };
  let lo = &iter_items[1];
  let hi = &iter_items[2];

  // Rename the distribution's dummy var to the user-supplied var_name in the pdf.
  let pdf = if dummy == var_name {
    pdf_raw.clone()
  } else {
    substitute_var(pdf_raw, dummy, &Expr::Identifier(var_name.to_string()))
  };

  // Helper: build Integrate[pdf, {var, a, b}].
  let build_integral = |a: Expr, b: Expr| -> Expr {
    call(
      "Integrate",
      vec![
        pdf.clone(),
        Expr::List(vec![Expr::Identifier(var_name.to_string()), a, b].into()),
      ],
    )
  };

  // Conditional events come in as Conditioned[lhs, rhs] (now parsed natively).
  // P[A \[Conditioned] B, x ~ d] = P[A ∧ B, x ~ d] / P[B, x ~ d] —
  // handled by the caller; nothing to do here.

  // `And[a, b]`: intersection of two sub-ranges, i.e. the tighter range. We
  // detect simple `var > k1 && var > k2` (etc.) shapes that reduce to a
  // single bounded integral.
  if let Expr::FunctionCall {
    name: head,
    args: aargs,
  } = event
    && head == "And"
    && aargs.len() == 2
  {
    let extract = |e: &Expr| -> Option<(ComparisonOp, Expr)> {
      // Returns (op, threshold) for `var <op> k` or `k <op> var` rephrased
      // so the variable is on the left.
      if let Expr::Comparison {
        operands,
        operators,
      } = e
        && operands.len() == 2
        && operators.len() == 1
      {
        let op = operators[0];
        let left = &operands[0];
        let right = &operands[1];
        let is_left_var = matches!(left, Expr::Identifier(n) if n == var_name);
        let is_right_var =
          matches!(right, Expr::Identifier(n) if n == var_name);
        if is_left_var {
          return Some((op, right.clone()));
        }
        if is_right_var {
          let flipped = match op {
            ComparisonOp::Less => ComparisonOp::Greater,
            ComparisonOp::LessEqual => ComparisonOp::GreaterEqual,
            ComparisonOp::Greater => ComparisonOp::Less,
            ComparisonOp::GreaterEqual => ComparisonOp::LessEqual,
            other => other,
          };
          return Some((flipped, left.clone()));
        }
      }
      None
    };
    if let (Some((op1, k1)), Some((op2, k2))) =
      (extract(&aargs[0]), extract(&aargs[1]))
    {
      let is_lower = |op: &ComparisonOp| {
        matches!(op, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
      };
      let is_upper = |op: &ComparisonOp| {
        matches!(op, ComparisonOp::Less | ComparisonOp::LessEqual)
      };
      // Both lower bounds → use the larger one. Both upper → use the smaller.
      // One of each → bounded interval.
      let new_lo;
      let new_hi;
      if is_lower(&op1) && is_lower(&op2) {
        new_lo = call("Max", vec![k1, k2]);
        new_hi = hi.clone();
      } else if is_upper(&op1) && is_upper(&op2) {
        new_lo = lo.clone();
        new_hi = call("Min", vec![k1, k2]);
      } else if is_lower(&op1) && is_upper(&op2) {
        new_lo = k1;
        new_hi = k2;
      } else if is_upper(&op1) && is_lower(&op2) {
        new_lo = k2;
        new_hi = k1;
      } else {
        return Ok(None);
      }
      let integral = build_integral(new_lo, new_hi);
      return Ok(Some(eval(integral)?));
    }
  }

  // Two-operand comparison
  if let Expr::Comparison {
    operands,
    operators,
  } = event
    && operands.len() == 2
    && operators.len() == 1
  {
    let op = &operators[0];
    let left = &operands[0];
    let right = &operands[1];
    let is_left_var = matches!(left, Expr::Identifier(n) if n == var_name);
    let is_right_var = matches!(right, Expr::Identifier(n) if n == var_name);

    // x > k or x >= k  →  Integrate[pdf, {x, k, hi}]
    if is_left_var
      && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
    {
      let integral = build_integral(right.clone(), hi.clone());
      return Ok(Some(eval(integral)?));
    }
    // x < k or x <= k  →  Integrate[pdf, {x, lo, k}]
    if is_left_var
      && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
    {
      let integral = build_integral(lo.clone(), right.clone());
      return Ok(Some(eval(integral)?));
    }
    // k < x → x > k
    if is_right_var
      && (*op == ComparisonOp::Less || *op == ComparisonOp::LessEqual)
    {
      let integral = build_integral(left.clone(), hi.clone());
      return Ok(Some(eval(integral)?));
    }
    // k > x → x < k
    if is_right_var
      && (*op == ComparisonOp::Greater || *op == ComparisonOp::GreaterEqual)
    {
      let integral = build_integral(lo.clone(), left.clone());
      return Ok(Some(eval(integral)?));
    }
  }

  // Three-operand chained comparison: a < x < b
  if let Expr::Comparison {
    operands,
    operators,
  } = event
    && operands.len() == 3
    && operators.len() == 2
    && matches!(&operands[1], Expr::Identifier(n) if n == var_name)
  {
    let both_less = matches!(
      (&operators[0], &operators[1]),
      (
        ComparisonOp::Less | ComparisonOp::LessEqual,
        ComparisonOp::Less | ComparisonOp::LessEqual
      )
    );
    if both_less {
      let integral = build_integral(operands[0].clone(), operands[2].clone());
      return Ok(Some(eval(integral)?));
    }
  }

  Ok(None)
}

/// Numerical wrapper for `Probability` — returns `N[Probability[…]]`.
pub fn n_probability_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let prob = probability_ast(args)?;
  // If it stayed unevaluated, re-wrap as NProbability so the user sees the
  // numerical-evaluation request rather than the symbolic fallback.
  if matches!(&prob, Expr::FunctionCall { name, .. } if name == "Probability") {
    return Ok(unevaluated("NProbability", args));
  }
  eval(call("N", vec![prob]))
}

/// Numerical wrapper for `Expectation` — returns `N[Expectation[…]]`.
pub fn n_expectation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let exp = expectation_ast(args)?;
  if matches!(&exp, Expr::FunctionCall { name, .. } if name == "Expectation") {
    return Ok(unevaluated("NExpectation", args));
  }
  eval(call("N", vec![exp]))
}

/// Implements `Expectation[f(x), x ~ CensoredDistribution[{a, b}, base]]`.
/// `dargs` is `[{a, b}, base_dist]`. Returns `None` if the shape doesn't
/// match so the caller can fall through.
fn try_expectation_censored(
  expr: &Expr,
  var: &str,
  dargs: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::plot::substitute_var;
  if dargs.len() != 2 {
    return Ok(None);
  }
  let Expr::List(bounds) = &dargs[0] else {
    return Ok(None);
  };
  if bounds.len() != 2 {
    return Ok(None);
  }
  let a = bounds[0].clone();
  let b = bounds[1].clone();
  let base = &dargs[1];

  // For a ProbabilityDistribution base, we have direct access to the pdf and
  // its domain. The integral over [a, b] uses the renamed pdf.
  if let Expr::FunctionCall {
    name: base_name,
    args: base_args,
  } = base
    && base_name == "ProbabilityDistribution"
    && base_args.len() == 2
    && let Expr::List(iter) = &base_args[1]
    && iter.len() == 3
    && let Expr::Identifier(dummy) = &iter[0]
  {
    let pdf = if dummy == var {
      base_args[0].clone()
    } else {
      substitute_var(&base_args[0], dummy, &Expr::Identifier(var.to_string()))
    };
    // Integrate f(x) * pdf(x) from a to b.
    let mid_integral = call(
      "Integrate",
      vec![
        times(expr.clone(), pdf.clone()),
        Expr::List(
          vec![Expr::Identifier(var.to_string()), a.clone(), b.clone()].into(),
        ),
      ],
    );
    // P[X < a] and P[X > b] via integrals of pdf over (lo, a) / (b, hi).
    let p_below = {
      let lo = iter[1].clone();
      call(
        "Integrate",
        vec![
          pdf.clone(),
          Expr::List(
            vec![Expr::Identifier(var.to_string()), lo, a.clone()].into(),
          ),
        ],
      )
    };
    let p_above = {
      let hi = iter[2].clone();
      call(
        "Integrate",
        vec![
          pdf,
          Expr::List(
            vec![Expr::Identifier(var.to_string()), b.clone(), hi].into(),
          ),
        ],
      )
    };
    let f_at_a = substitute_var(expr, var, &a);
    let f_at_b = substitute_var(expr, var, &b);
    let total = plus(
      times(f_at_a, p_below),
      plus(eval(mid_integral)?, times(f_at_b, p_above)),
    );
    return Ok(Some(eval(total)?));
  }

  // For a base distribution Woxi already understands, use its CDF / PDF.
  // We compute the components via integrals against PDF if available.
  let base_name = match base {
    Expr::FunctionCall { name, .. } => name.as_str(),
    _ => return Ok(None),
  };
  // Generic numerical fallback would go here. For now, only the
  // ProbabilityDistribution path is wired up since that covers the
  // example notebook; bail out otherwise.
  let _ = base_name;
  Ok(None)
}

/// Compute `Expectation[f, vars \[Distributed] ProbabilityDistribution[pdf, {a,…}, {b,…}, …]]`
/// by reducing to `Integrate[f * pdf, {a,…}, …]` over the declared domain.
///
/// `dargs` is the argument list of the `ProbabilityDistribution` head:
/// `[pdf_expr, {var1, lo, hi}, {var2, lo, hi}, …]`.
/// `vars` are the names supplied in the `Distributed[…]` LHS; they replace
/// the distribution's dummy variable names so the user can write
/// `Expectation[x, x ~ ProbabilityDistribution[2/y^3, {y, 1, ∞}]]`
/// and have `y` rebound to `x` for the integral.
fn try_expectation_probability_distribution(
  expr: &Expr,
  vars: &[String],
  dargs: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::plot::substitute_var;

  if dargs.len() < 2 {
    return Ok(None);
  }
  let pdf = &dargs[0];
  let iters = &dargs[1..];

  // Number of supplied variables must match the number of iterators.
  if iters.len() != vars.len() {
    return Ok(None);
  }

  // Substitute distribution's dummy variable names with user-supplied names
  // in both the pdf and integration iterators.
  let mut pdf_sub = pdf.clone();
  let mut new_iters: Vec<Expr> = Vec::with_capacity(iters.len());
  for (i, iter) in iters.iter().enumerate() {
    let Expr::List(items) = iter else {
      return Ok(None);
    };
    if items.len() != 3 {
      return Ok(None);
    }
    let Expr::Identifier(dummy) = &items[0] else {
      return Ok(None);
    };
    let target = &vars[i];
    if dummy != target {
      let new_var = Expr::Identifier(target.clone());
      pdf_sub = substitute_var(&pdf_sub, dummy, &new_var);
    }
    new_iters.push(Expr::List(
      vec![
        Expr::Identifier(target.clone()),
        items[1].clone(),
        items[2].clone(),
      ]
      .into(),
    ));
  }

  // Build Integrate[expr * pdf, iter1, iter2, …]
  let integrand = times(expr.clone(), pdf_sub);
  let mut integrate_args: Vec<Expr> = Vec::with_capacity(1 + new_iters.len());
  integrate_args.push(integrand);
  integrate_args.extend(new_iters);
  let integral = call("Integrate", integrate_args);
  Ok(Some(eval(integral)?))
}

/// Parameters of a BinormalDistribution as `(m1, m2, s1, s2, rho)`.
///   BinormalDistribution[{m1, m2}, {s1, s2}, rho]  — full form
///   BinormalDistribution[rho]                       — standard (means 0,
///                                                     unit variances)
/// Returns `None` for any other shape.
pub fn binormal_params(
  dargs: &[Expr],
) -> Option<(Expr, Expr, Expr, Expr, Expr)> {
  match dargs {
    [Expr::List(m), Expr::List(s), rho] if m.len() == 2 && s.len() == 2 => {
      Some((
        m[0].clone(),
        m[1].clone(),
        s[0].clone(),
        s[1].clone(),
        rho.clone(),
      ))
    }
    [rho] => Some((
      Expr::Integer(0),
      Expr::Integer(0),
      Expr::Integer(1),
      Expr::Integer(1),
      rho.clone(),
    )),
    _ => None,
  }
}

/// Returns (Mean, Variance) as symbolic expressions for known distributions.
pub fn distribution_mean_variance(
  dist_name: &str,
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  match dist_name {
    "ErlangDistribution" if dargs.len() == 2 => distribution_mean_variance(
      "GammaDistribution",
      &erlang_gamma_dargs(dargs)?,
    ),
    "CoxianDistribution" => coxian_mean_variance(dargs),
    "VonMisesDistribution" => {
      let Some(()) = vonmises_checked(dargs) else {
        return Err(InterpreterError::EvaluationError(
          "VonMisesDistribution: invalid parameters".into(),
        ));
      };
      // Only the mean has a closed form; Variance/StandardDeviation
      // stay unevaluated in wolframscript (VonMises is not listed for
      // the variance dispatch, so the second component is unused).
      Ok((dargs[0].clone(), indeterminate()))
    }
    "HyperexponentialDistribution" => hyperexponential_mean_variance(dargs),
    "BeniniDistribution" => benini_mean_variance(dargs),
    "HotellingTSquareDistribution" => hotelling_mean_variance(dargs),
    "TukeyLambdaDistribution" => tukey_lambda_mean_variance(dargs),
    "TsallisQGaussianDistribution" => tsallis_qgaussian_mean_variance(dargs),
    "VarianceGammaDistribution" => variance_gamma_mean_variance(dargs),
    "HoytDistribution" => hoyt_mean_variance(dargs),
    "CompoundPoissonDistribution" => compound_poisson_mean_variance(dargs),
    "WakebyDistribution" => wakeby_mean_variance(dargs),
    "MaxwellDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "MaxwellDistribution expects 1 argument".into(),
        ));
      }
      maxwell_mean_variance(&dargs[0])
    }
    "BorelTannerDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BorelTannerDistribution expects 2 arguments".into(),
        ));
      }
      borel_tanner_mean_variance(&dargs[0], &dargs[1])
    }
    "BenktanderGibratDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BenktanderGibratDistribution expects 2 arguments".into(),
        ));
      }
      benktander_gibrat_mean_variance(&dargs[0], &dargs[1])
    }
    "ZipfDistribution" => zipf_mean_variance(dargs),
    "BenfordDistribution" => benford_mean_variance(dargs),
    "BenktanderWeibullDistribution" => benktander_weibull_mean_variance(dargs),
    "GumbelDistribution" => gumbel_mean_variance(dargs),
    "SkewNormalDistribution" => skew_normal_mean_variance(dargs),
    "SechDistribution" => sech_mean_variance(dargs),
    "WignerSemicircleDistribution" => wigner_mean_variance(dargs),
    "TriangularDistribution" => triangular_mean_variance(dargs),
    "MaxStableDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "MaxStableDistribution expects 3 arguments".into(),
        ));
      }
      max_stable_mean_variance(&dargs[0], &dargs[1], &dargs[2])
    }
    "MinStableDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "MinStableDistribution expects 3 arguments".into(),
        ));
      }
      min_stable_mean_variance(&dargs[0], &dargs[1], &dargs[2])
    }
    "RiceDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "RiceDistribution expects 2 arguments".into(),
        ));
      }
      rice_mean_variance(&dargs[0], &dargs[1])
    }
    "ExponentialPowerDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "ExponentialPowerDistribution expects 3 arguments".into(),
        ));
      }
      let (k, m, s) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
      // Mean = m; Var = k^(2/k) s^2 Gamma[3/k] / Gamma[1/k]
      let var = divide(
        times(
          power(k.clone(), divide(int(2), k.clone())),
          times(
            power(s, int(2)),
            call("Gamma", vec![divide(int(3), k.clone())]),
          ),
        ),
        call("Gamma", vec![power(k, int(-1))]),
      );
      Ok((m, var))
    }
    "NoncentralChiSquareDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "NoncentralChiSquareDistribution expects 2 arguments".into(),
        ));
      }
      let (nu, lam) = (dargs[0].clone(), dargs[1].clone());
      // Mean = v + l, Var = 2 v + 4 l
      let mean = plus(nu.clone(), lam.clone());
      let var = plus(times(int(2), nu), times(int(4), lam));
      Ok((mean, var))
    }
    "BetaPrimeDistribution" if dargs.len() == 4 => {
      beta_prime4_mean_variance(dargs)
    }
    "BetaPrimeDistribution" => {
      if dargs.len() != 2 && dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "BetaPrimeDistribution expects 2 to 4 arguments".into(),
        ));
      }
      // The 3-argument form BetaPrimeDistribution[p, q, s] scales the standard
      // mean by s and the variance by s^2.
      let scale = if dargs.len() == 3 {
        dargs[2].clone()
      } else {
        int(1)
      };
      let (p, q) = (dargs[0].clone(), dargs[1].clone());
      // Resolve the q-conditions up front for numeric q so the dead
      // branch never evaluates (a literal p/0 would message infy)
      let q_num: Option<f64> = match &q {
        Expr::Integer(v) => Some(*v as f64),
        Expr::Real(v) => Some(*v),
        Expr::FunctionCall { name, args } if name == "Rational" => {
          match (&args[0], &args[1]) {
            (Expr::Integer(a), Expr::Integer(b)) if *b != 0 => {
              Some(*a as f64 / *b as f64)
            }
            _ => None,
          }
        }
        _ => None,
      };
      let mean_base = divide(p.clone(), plus(int(-1), q.clone()));
      // Mean = Piecewise[{{p/(q-1), q > 1}}, Infinity]; the 3-arg form scales
      // this by the scale parameter s.
      let mean_value = if dargs.len() == 3 {
        eval(times(scale.clone(), mean_base))?
      } else {
        mean_base
      };
      let mean = match q_num {
        Some(v) if v > 1.0 => mean_value,
        Some(_) => infinity(),
        None => piecewise(
          vec![(
            mean_value,
            comparison(q.clone(), ComparisonOp::Greater, int(1)),
          )],
          infinity(),
        ),
      };
      // Var = Piecewise[{{p(p+q-1)/((q-2)(q-1)^2), q > 2}}, Indeterminate];
      // the 3-arg form scales this by s^2.
      let var_base = divide(
        times(p.clone(), plus(plus(int(-1), p.clone()), q.clone())),
        times(
          plus(int(-2), q.clone()),
          power(plus(int(-1), q.clone()), int(2)),
        ),
      );
      let var_value = if dargs.len() == 3 {
        eval(times(power(scale.clone(), int(2)), var_base))?
      } else {
        var_base
      };
      let var = match q_num {
        Some(v) if v > 2.0 => var_value,
        Some(_) => indeterminate(),
        None => piecewise(
          vec![(var_value, comparison(q, ComparisonOp::Greater, int(2)))],
          indeterminate(),
        ),
      };
      Ok((mean, var))
    }
    "BetaBinomialDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "BetaBinomialDistribution expects 3 arguments".into(),
        ));
      }
      let (a, b, n) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
      // Mean = a n / (a + b)
      let mean =
        divide(times(a.clone(), n.clone()), plus(a.clone(), b.clone()));
      // Var = a b n (a + b + n) / ((a + b)^2 (1 + a + b))
      let var = divide(
        times(
          times(a.clone(), b.clone()),
          times(n.clone(), plus(plus(a.clone(), b.clone()), n)),
        ),
        times(
          power(plus(a.clone(), b.clone()), int(2)),
          plus(int(1), plus(a, b)),
        ),
      );
      Ok((mean, var))
    }
    "ExpGammaDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "ExpGammaDistribution expects 3 arguments".into(),
        ));
      }
      let (k, t, m) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
      let polygamma = |n: i128| call("PolyGamma", vec![int(n), k.clone()]);
      // Mean = m + t PolyGamma[0, k]; Variance = t^2 PolyGamma[1, k].
      let mean = plus(m, times(t.clone(), polygamma(0)));
      let var = times(power(t, int(2)), polygamma(1));
      Ok((mean, var))
    }
    "LogGammaDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "LogGammaDistribution expects 3 arguments".into(),
        ));
      }
      let (a, b, m) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
      // Mean = Piecewise[{{-1 + (1-b)^(-a) + m, b < 1}}, Infinity]
      let mean_val = plus(
        plus(
          int(-1),
          power(minus(int(1), b.clone()), times(int(-1), a.clone())),
        ),
        m,
      );
      let mean = piecewise(
        vec![(mean_val, comparison(b.clone(), ComparisonOp::Less, int(1)))],
        infinity(),
      );
      // Var = Piecewise[{{(1-2b)^(-a) - (1-b)^(-2a), b < 1/2}}, Infinity]
      let var_val = minus(
        power(
          minus(int(1), times(int(2), b.clone())),
          times(int(-1), a.clone()),
        ),
        power(minus(int(1), b.clone()), times(int(-2), a)),
      );
      let var = piecewise(
        vec![(
          var_val,
          comparison(b, ComparisonOp::Less, divide(int(1), int(2))),
        )],
        infinity(),
      );
      Ok((mean, var))
    }
    "PERTDistribution" => {
      let Some((a, b, m, g)) = pert_params(dargs) else {
        return Err(InterpreterError::EvaluationError(
          "PERTDistribution expects {min, max} and a mode".into(),
        ));
      };
      if dargs.len() == 2 {
        // Mean = (a + b + 4 m)/6; Variance = ((-a+5b-4m)(-5a+b+4m))/252.
        let mean = eval(divide(
          plus(plus(a.clone(), b.clone()), times(int(4), m.clone())),
          int(6),
        ))?;
        let f1 = plus(
          plus(times(int(-1), a.clone()), times(int(5), b.clone())),
          times(int(-4), m.clone()),
        );
        let f2 = plus(plus(times(int(-5), a), b), times(int(4), m));
        let var = eval(divide(times(f1, f2), int(252)))?;
        Ok((mean, var))
      } else {
        // Mean = (a + b + g m)/(2 + g);
        // Variance = ((-a + b + b g - g m)(b - a(1 + g) + g m)) /
        //   ((2 + g)^2 (3 + g)). The symbolic form is value-correct but
        //   the caller's canonicalization orders the two Plus factors
        //   differently from wolframscript (the known sum-vs-sum Times
        //   ordering divergence).
        let mean = eval(divide(
          plus(plus(a.clone(), b.clone()), times(g.clone(), m.clone())),
          plus(int(2), g.clone()),
        ))?;
        let numeric = [&a, &b, &m, &g]
          .iter()
          .all(|e| crate::functions::math_ast::try_eval_to_f64(e).is_some());
        let f1 = call(
          "Plus",
          vec![
            times(int(-1), a.clone()),
            b.clone(),
            times(b.clone(), g.clone()),
            times(int(-1), times(g.clone(), m.clone())),
          ],
        );
        let f2 = call(
          "Plus",
          vec![
            b,
            times(int(-1), times(a, plus(int(1), g.clone()))),
            times(g.clone(), m),
          ],
        );
        let var_expr = divide(
          times(f1, f2),
          times(power(plus(int(2), g.clone()), int(2)), plus(int(3), g)),
        );
        let var = if numeric { eval(var_expr)? } else { var_expr };
        Ok((mean, var))
      }
    }
    "PowerDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "PowerDistribution expects 2 arguments".into(),
        ));
      }
      let (k, a) = (dargs[0].clone(), dargs[1].clone());
      // Mean = a/(k + a k); Variance = a/((1+a)^2 (2+a) k^2).
      let mean = eval(divide(
        a.clone(),
        plus(k.clone(), times(a.clone(), k.clone())),
      ))?;
      let var = eval(divide(
        a.clone(),
        times(
          times(power(plus(int(1), a.clone()), int(2)), plus(int(2), a)),
          power(k, int(2)),
        ),
      ))?;
      Ok((mean, var))
    }
    "KumaraswamyDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "KumaraswamyDistribution expects 2 arguments".into(),
        ));
      }
      let (a, b) = (dargs[0].clone(), dargs[1].clone());
      let beta = |y: Expr| call("Beta", vec![b.clone(), y]);
      // Mean = b Beta[b, 1 + 1/a]; raw 2nd moment = b Beta[b, 1 + 2/a].
      let mean =
        times(b.clone(), beta(plus(int(1), divide(int(1), a.clone()))));
      let raw2 = times(b.clone(), beta(plus(int(1), divide(int(2), a))));
      // Variance = E[x^2] - Mean^2.
      let var = eval(minus(raw2, power(mean.clone(), int(2))))?;
      Ok((mean, var))
    }
    "BetaNegativeBinomialDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "BetaNegativeBinomialDistribution expects 3 arguments".into(),
        ));
      }
      let (a, b, n) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
      let a_minus_1 = plus(int(-1), a.clone());
      // Mean = Piecewise[{{b n/(a-1), a > 1}}, Infinity]
      let mean_val = divide(times(b.clone(), n.clone()), a_minus_1.clone());
      let mean = piecewise(
        vec![(
          mean_val,
          comparison(a.clone(), ComparisonOp::Greater, int(1)),
        )],
        infinity(),
      );
      // Var = Piecewise[{{b(a+b-1)n(a+n-1)/((a-2)(a-1)^2), a > 2}}, Infinity]
      let num = times(
        times(b.clone(), plus(plus(int(-1), a.clone()), b)),
        times(n.clone(), plus(plus(int(-1), a.clone()), n)),
      );
      let den = times(plus(int(-2), a.clone()), power(a_minus_1, int(2)));
      let var_val = divide(num, den);
      let var = piecewise(
        vec![(var_val, comparison(a, ComparisonOp::Greater, int(2)))],
        infinity(),
      );
      Ok((mean, var))
    }
    "UniformSumDistribution" => {
      // Mean = n(a+b)/2, Var = n(b-a)^2/12 (default range {0, 1})
      let (n, a, b) = match dargs {
        [n] => (n.clone(), int(0), int(1)),
        [n, Expr::List(bounds)] if bounds.len() == 2 => {
          (n.clone(), bounds[0].clone(), bounds[1].clone())
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "UniformSumDistribution expects n or n, {min, max}".into(),
          ));
        }
      };
      let mean = divide(times(plus(a.clone(), b.clone()), n.clone()), int(2));
      let var = divide(times(power(minus(b, a), int(2)), n), int(12));
      Ok((mean, var))
    }
    "NormalDistribution" => {
      let (mu, sigma) = match dargs.len() {
        0 => (int(0), int(1)),
        2 => (dargs[0].clone(), dargs[1].clone()),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "NormalDistribution expects 0 or 2 arguments".into(),
          ));
        }
      };
      // Mean = mu, Var = sigma^2
      Ok((mu, power(sigma, int(2))))
    }
    "UniformDistribution" => {
      let (a, b) = match dargs.len() {
        0 => (int(0), int(1)),
        1 => {
          if let Expr::List(bounds) = &dargs[0] {
            if bounds.len() == 2 {
              (bounds[0].clone(), bounds[1].clone())
            } else {
              return Err(InterpreterError::EvaluationError(
                "UniformDistribution: expected {min, max}".into(),
              ));
            }
          } else {
            return Err(InterpreterError::EvaluationError(
              "UniformDistribution: expected a list".into(),
            ));
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "UniformDistribution expects 0 or 1 argument".into(),
          ));
        }
      };
      // Mean = (a+b)/2, Var = (b-a)^2/12
      let mean = divide(plus(a.clone(), b.clone()), int(2));
      let var = divide(power(minus(b, a), int(2)), int(12));
      Ok((mean, var))
    }
    "ExponentialDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "ExponentialDistribution expects 1 argument".into(),
        ));
      }
      let lambda = dargs[0].clone();
      // Mean = 1/lambda, Var = 1/lambda^2
      let mean = divide(int(1), lambda.clone());
      let var = divide(int(1), power(lambda, int(2)));
      Ok((mean, var))
    }
    "PoissonDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "PoissonDistribution expects 1 argument".into(),
        ));
      }
      let mu = dargs[0].clone();
      // Mean = mu, Var = mu
      Ok((mu.clone(), mu))
    }
    "BernoulliDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "BernoulliDistribution expects 1 argument".into(),
        ));
      }
      let p = dargs[0].clone();
      // Mean = p, Var = p(1-p)
      let var = times(p.clone(), minus(int(1), p.clone()));
      Ok((p, var))
    }
    "SkellamDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "SkellamDistribution expects 2 arguments".into(),
        ));
      }
      // X = Poisson(a) - Poisson(b): Mean = a - b, Var = a + b.
      let (a, b) = (dargs[0].clone(), dargs[1].clone());
      Ok((minus(a.clone(), b.clone()), plus(a, b)))
    }
    "PolyaAeppliDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "PolyaAeppliDistribution expects 2 arguments".into(),
        ));
      }
      // Geometric-Poisson: Mean = t/(1-p), Var = (1+p) t / (1-p)^2.
      let (t, p) = (dargs[0].clone(), dargs[1].clone());
      let one_minus_p = minus(int(1), p.clone());
      let mean = divide(t.clone(), one_minus_p.clone());
      let var = divide(times(plus(int(1), p), t), power(one_minus_p, int(2)));
      Ok((mean, var))
    }
    "ChiDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "ChiDistribution expects 1 argument".into(),
        ));
      }
      // Mean = Sqrt[2] Gamma[(1+k)/2] / Gamma[k/2];
      // Var  = k - 2 Gamma[(1+k)/2]^2 / Gamma[k/2]^2.
      // Simplify re-combines the Sqrt ratios that arise from evaluating the
      // half-integer Gammas (e.g. Sqrt[Pi]/Sqrt[2] -> Sqrt[Pi/2]).
      let k = dargs[0].clone();
      let g_kp1 = unary_fn("Gamma", divide(plus(int(1), k.clone()), int(2)));
      let g_k = unary_fn("Gamma", divide(k.clone(), int(2)));
      let mean_raw =
        divide(times(unary_fn("Sqrt", int(2)), g_kp1.clone()), g_k.clone());
      let mean = eval(unary_fn("Simplify", mean_raw))?;
      let var_raw = minus(
        k,
        divide(times(int(2), power(g_kp1, int(2))), power(g_k, int(2))),
      );
      let var = eval(unary_fn("Simplify", var_raw))?;
      Ok((mean, var))
    }
    "BinomialDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BinomialDistribution expects 2 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let p = dargs[1].clone();
      // Mean = n*p, Var = n*(1-p)*p
      let mean = times(n.clone(), p.clone());
      let var = times(times(n, minus(int(1), p.clone())), p);
      Ok((mean, var))
    }
    "GammaDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "GammaDistribution expects 2 arguments".into(),
        ));
      }
      let alpha = dargs[0].clone();
      let beta = dargs[1].clone();
      // Mean = alpha*beta, Var = alpha*beta^2
      let mean = times(alpha.clone(), beta.clone());
      let var = times(alpha, power(beta, int(2)));
      Ok((mean, var))
    }
    "HypergeometricDistribution" => {
      // HypergeometricDistribution[n, n_succ, n_tot]: draw n without
      // replacement from a population of n_tot with n_succ successes.
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "HypergeometricDistribution expects 3 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let ns = dargs[1].clone();
      let nt = dargs[2].clone();
      // Mean = (n*ns)/nt.
      let mean = divide(times(n.clone(), ns.clone()), nt.clone());
      // Var = (n*ns*(1 - ns/nt)*(nt - n)) / ((nt - 1)*nt).
      let var = divide(
        times(
          times(
            times(n.clone(), ns.clone()),
            minus(int(1), divide(ns, nt.clone())),
          ),
          minus(nt.clone(), n),
        ),
        times(minus(nt.clone(), int(1)), nt),
      );
      Ok((mean, var))
    }
    "BetaDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BetaDistribution expects 2 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      // Mean = a/(a+b), Var = a*b / ((a+b)^2 * (a+b+1))
      let ab = plus(a.clone(), b.clone());
      let mean = divide(a.clone(), ab.clone());
      let var = divide(
        times(a, b),
        times(power(ab.clone(), int(2)), plus(ab, int(1))),
      );
      Ok((mean, var))
    }
    "StudentTDistribution" => {
      // Standard StudentTDistribution[nu] (mean 0, scale 1) or the
      // location-scale StudentTDistribution[m, s, nu].
      let (m, s, nu) = match dargs.len() {
        1 => (int(0), int(1), dargs[0].clone()),
        3 => (dargs[0].clone(), dargs[1].clone(), dargs[2].clone()),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "StudentTDistribution expects 1 or 3 arguments".into(),
          ));
        }
      };
      // Mean = Piecewise[{{m, nu > 1}}, Indeterminate]
      let mean = piecewise(
        vec![(m, comparison(nu.clone(), ComparisonOp::Greater, int(1)))],
        indeterminate(),
      );
      // Var = Piecewise[{{s^2 nu/(nu-2), nu > 2}}, Indeterminate];
      // for the 1-arg form s == 1, giving nu/(nu-2).
      let var_value = if matches!(s, Expr::Integer(1)) {
        divide(nu.clone(), plus(int(-2), nu.clone()))
      } else {
        divide(
          times(power(s, int(2)), nu.clone()),
          plus(int(-2), nu.clone()),
        )
      };
      let var = piecewise(
        vec![(var_value, comparison(nu, ComparisonOp::Greater, int(2)))],
        indeterminate(),
      );
      Ok((mean, var))
    }
    "FRatioDistribution" => {
      // FRatioDistribution[n, m]: F distribution with n and m degrees of
      // freedom. Mean and variance exist only for m > 2 and m > 4.
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "FRatioDistribution expects 2 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let m = dargs[1].clone();
      // Mean = Piecewise[{{m/(-2 + m), m > 2}}, Indeterminate]
      let mean = piecewise(
        vec![(
          divide(m.clone(), plus(int(-2), m.clone())),
          comparison(m.clone(), ComparisonOp::Greater, int(2)),
        )],
        indeterminate(),
      );
      // Var = Piecewise[{{(2 m^2 (-2 + m + n)) /
      //                   ((-4 + m) (-2 + m)^2 n), m > 4}}, Indeterminate]
      let var_num = times(
        times(int(2), power(m.clone(), int(2))),
        plus(plus(int(-2), m.clone()), n.clone()),
      );
      let var_den = times(
        times(
          plus(int(-4), m.clone()),
          power(plus(int(-2), m.clone()), int(2)),
        ),
        n,
      );
      let var = piecewise(
        vec![(
          divide(var_num, var_den),
          comparison(m, ComparisonOp::Greater, int(4)),
        )],
        indeterminate(),
      );
      Ok((mean, var))
    }
    "LogNormalDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "LogNormalDistribution expects 2 arguments".into(),
        ));
      }
      let mu = dargs[0].clone();
      let sigma = dargs[1].clone();
      // Mean = E^(mu + sigma^2/2)
      let mean = power(
        Expr::Identifier("E".to_string()),
        plus(mu.clone(), divide(power(sigma.clone(), int(2)), int(2))),
      );
      // Var = E^(2*mu + sigma^2) * (E^(sigma^2) - 1)
      let var = times(
        power(
          Expr::Identifier("E".to_string()),
          plus(times(int(2), mu), power(sigma.clone(), int(2))),
        ),
        minus(
          power(Expr::Identifier("E".to_string()), power(sigma, int(2))),
          int(1),
        ),
      );
      Ok((mean, var))
    }
    "ChiSquareDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "ChiSquareDistribution expects 1 argument".into(),
        ));
      }
      let k = dargs[0].clone();
      // Mean = k, Var = 2*k
      let mean = k.clone();
      let var = times(int(2), k);
      Ok((mean, var))
    }
    "ParetoDistribution" if dargs.len() == 3 => pareto3_mean_variance(dargs),
    "ParetoDistribution" if dargs.len() == 4 => pareto4_mean_variance(dargs),
    "ParetoDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "ParetoDistribution expects 2 arguments".into(),
        ));
      }
      let k = dargs[0].clone();
      let a = dargs[1].clone();
      // Mean = Piecewise[{{a*k/(-1+a), a > 1}}, Indeterminate]
      let mean = piecewise(
        vec![(
          divide(times(a.clone(), k.clone()), plus(int(-1), a.clone())),
          comparison(a.clone(), ComparisonOp::Greater, int(1)),
        )],
        indeterminate(),
      );
      // Var = Piecewise[{{a*k^2 / ((-2+a)*(-1+a)^2), a > 2}}, Indeterminate]
      let var = piecewise(
        vec![(
          divide(
            times(a.clone(), power(k, int(2))),
            times(
              plus(int(-2), a.clone()),
              power(plus(int(-1), a.clone()), int(2)),
            ),
          ),
          comparison(a, ComparisonOp::Greater, int(2)),
        )],
        indeterminate(),
      );
      Ok((mean, var))
    }
    "WeibullDistribution" => {
      if dargs.len() != 2 && dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "WeibullDistribution expects 2 or 3 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      // Mean = b * Gamma[1 + 1/a]; the 3-argument form adds the location m.
      let base_mean = times(
        b.clone(),
        call("Gamma", vec![plus(int(1), divide(int(1), a.clone()))]),
      );
      let mean = if dargs.len() == 3 {
        plus(dargs[2].clone(), base_mean)
      } else {
        base_mean
      };
      // Var = b^2 * (Gamma[1 + 2/a] - Gamma[1 + 1/a]^2)
      let var = times(
        power(b, int(2)),
        minus(
          call("Gamma", vec![plus(int(1), divide(int(2), a.clone()))]),
          power(call("Gamma", vec![plus(int(1), divide(int(1), a))]), int(2)),
        ),
      );
      Ok((mean, var))
    }
    "HalfNormalDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "HalfNormalDistribution expects 1 argument".into(),
        ));
      }
      let theta = dargs[0].clone();
      // Mean = 1/theta
      let mean = divide(int(1), theta.clone());
      // Variance = (Pi - 2) / (2 * theta^2). Built as (-2 + Pi)/(...) so
      // the rendered output matches wolframscript's canonical ordering.
      let var =
        divide(plus(int(-2), pi()), times(int(2), power(theta, int(2))));
      Ok((mean, var))
    }
    "InverseGaussianDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "InverseGaussianDistribution expects 2 arguments".into(),
        ));
      }
      let mu = dargs[0].clone();
      let lambda = dargs[1].clone();
      // Mean = mu; Variance = mu^3 / lambda
      let mean = mu.clone();
      let var = divide(power(mu, int(3)), lambda);
      Ok((mean, var))
    }
    "LogisticDistribution" => {
      let (mu, beta) = match dargs.len() {
        0 => (int(0), int(1)),
        2 => (dargs[0].clone(), dargs[1].clone()),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "LogisticDistribution expects 0 or 2 arguments".into(),
          ));
        }
      };
      // Mean = mu; Variance = beta^2 * Pi^2 / 3.
      let mean = mu;
      let var = divide(times(power(beta, int(2)), power(pi(), int(2))), int(3));
      Ok((mean, var))
    }
    "HypoexponentialDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "HypoexponentialDistribution expects 1 argument".into(),
        ));
      }
      let Expr::List(rates) = &dargs[0] else {
        return Err(InterpreterError::EvaluationError(
          "HypoexponentialDistribution: expected a list of rates".into(),
        ));
      };
      if rates.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "HypoexponentialDistribution: rate list cannot be empty".into(),
        ));
      }
      // Mean = sum_i 1/lambda_i; Variance = sum_i 1/lambda_i^2. The
      // hypoexponential is a sum of independent exponentials, so the
      // mean and variance add term by term.
      let mean_terms: Vec<Expr> =
        rates.iter().map(|r| divide(int(1), r.clone())).collect();
      let var_terms: Vec<Expr> = rates
        .iter()
        .map(|r| divide(int(1), power(r.clone(), int(2))))
        .collect();
      let mean = call("Plus", mean_terms);
      let var = call("Plus", var_terms);
      Ok((mean, var))
    }
    "ExtremeValueDistribution" => {
      let (a, b) = match dargs.len() {
        0 => (int(0), int(1)),
        2 => (dargs[0].clone(), dargs[1].clone()),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "ExtremeValueDistribution expects 0 or 2 arguments".into(),
          ));
        }
      };
      // Mean = a + b * EulerGamma
      let mean = plus(
        a,
        times(b.clone(), Expr::Constant("EulerGamma".to_string())),
      );
      // Variance = b^2 * Pi^2 / 6
      let var = divide(times(power(b, int(2)), power(pi(), int(2))), int(6));
      Ok((mean, var))
    }
    "StableDistribution" => {
      // Canonical 5-arg form: StableDistribution[type, alpha, beta, mu, sigma].
      // type ∈ {0, 1} selects between the two standard parametrisations.
      if dargs.len() != 5 {
        return Err(InterpreterError::EvaluationError(
          "StableDistribution expects 5 arguments (canonical form)".into(),
        ));
      }
      let type_ = dargs[0].clone();
      let alpha = dargs[1].clone();
      let beta = dargs[2].clone();
      let mu = dargs[3].clone();
      let sigma = dargs[4].clone();
      let indet = indeterminate();
      // Mean exists when 1 < alpha <= 2.
      // Type 0: Mean = mu - beta * sigma * Tan[Pi * alpha / 2]
      // Type 1: Mean = mu
      let mean_branch = match &type_ {
        Expr::Integer(0) => {
          let tan_arg = divide(times(alpha.clone(), pi()), int(2));
          let tan_term = call("Tan", vec![tan_arg]);
          minus(mu.clone(), times(times(beta, sigma.clone()), tan_term))
        }
        _ => mu.clone(),
      };
      let mean = piecewise(
        vec![(
          mean_branch,
          comparison3(
            int(2),
            ComparisonOp::GreaterEqual,
            alpha.clone(),
            ComparisonOp::Greater,
            int(1),
          ),
        )],
        indet.clone(),
      );
      // Variance is finite only for alpha == 2 (Gaussian limit), where it
      // is 2 * sigma^2. Holds for both type parametrisations.
      let var = piecewise(
        vec![(
          times(int(2), power(sigma, int(2))),
          comparison(alpha, ComparisonOp::Equal, int(2)),
        )],
        indet,
      );
      Ok((mean, var))
    }
    "InverseGammaDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "InverseGammaDistribution expects 2 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      let indet = indeterminate();
      // Mean = Piecewise[{{b/(-1 + a), a > 1}}, Indeterminate]
      let mean_branch = divide(b.clone(), plus(int(-1), a.clone()));
      let mean = piecewise(
        vec![(
          mean_branch,
          comparison(a.clone(), ComparisonOp::Greater, int(1)),
        )],
        indet.clone(),
      );
      // Variance = Piecewise[{{b^2/((-2 + a)*(-1 + a)^2), a > 2}},
      //                      Indeterminate]
      let denom = times(
        plus(int(-2), a.clone()),
        power(plus(int(-1), a.clone()), int(2)),
      );
      let var_branch = divide(power(b, int(2)), denom);
      let var = piecewise(
        vec![(var_branch, comparison(a, ComparisonOp::Greater, int(2)))],
        indet,
      );
      Ok((mean, var))
    }
    "GompertzMakehamDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "GompertzMakehamDistribution expects 2 arguments".into(),
        ));
      }
      let lambda = dargs[0].clone();
      let xi = dargs[1].clone();
      // Mean = (E^xi * Gamma[0, xi]) / lambda
      let gamma_0_xi = call("Gamma", vec![int(0), xi.clone()]);
      let mean = divide(times(power(e(), xi), gamma_0_xi), lambda.clone());
      // Variance has no simple closed form in elementary functions;
      // GompertzMakehamDistribution is intentionally absent from the
      // Variance dispatch list, so this placeholder is never returned
      // to the user. Provide an unevaluated stub so the tuple typechecks.
      let var = call(
        "Variance",
        vec![unevaluated("GompertzMakehamDistribution", dargs)],
      );
      Ok((mean, var))
    }
    "FrechetDistribution" => {
      if dargs.len() != 2 && dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "FrechetDistribution expects 2 or 3 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      let mu = if dargs.len() == 3 {
        Some(dargs[2].clone())
      } else {
        None
      };
      // Mean = Piecewise[{{μ + b * Gamma[1 - 1/a], 1 < a}}, Infinity]
      let gamma_1_minus_inv_a =
        call("Gamma", vec![minus(int(1), divide(int(1), a.clone()))]);
      let b_gamma = times(b.clone(), gamma_1_minus_inv_a.clone());
      let mean_value = match &mu {
        Some(m) => plus(m.clone(), b_gamma),
        None => b_gamma,
      };
      let mean = piecewise(
        vec![(
          mean_value,
          comparison(int(1), ComparisonOp::Less, a.clone()),
        )],
        infinity(),
      );
      // Var = Piecewise[{{b^2 * (Gamma[1 - 2/a] - Gamma[1 - 1/a]^2), a > 2}}, Infinity]
      // (Variance is translation-invariant — μ drops out.)
      let gamma_1_minus_2_a =
        call("Gamma", vec![minus(int(1), divide(int(2), a.clone()))]);
      let var = piecewise(
        vec![(
          times(
            power(b, int(2)),
            minus(gamma_1_minus_2_a, power(gamma_1_minus_inv_a, int(2))),
          ),
          comparison(a, ComparisonOp::Greater, int(2)),
        )],
        infinity(),
      );
      Ok((mean, var))
    }
    "GeometricDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "GeometricDistribution expects 1 argument".into(),
        ));
      }
      let p = dargs[0].clone();
      // Mean = -1 + 1/p, Var = (1-p)/p^2. wolframscript returns the mean in
      // the expanded `Apart` form (-1 + p^(-1)), but keeps the variance as the
      // combined fraction.
      let mean = plus(int(-1), power(p.clone(), int(-1)));
      let var = divide(minus(int(1), p.clone()), power(p, int(2)));
      Ok((mean, var))
    }
    // SuzukiDistribution has no closed-form PDF/CDF (wolframscript leaves the
    // symbolic forms unevaluated and only computes the PDF numerically via
    // integration, which is not implemented here). Its Mean and Variance are
    // clean closed forms. The fully-symbolic Variance differs from
    // wolframscript only in Plus term order (value-correct); every numeric
    // parameterization agrees exactly.
    "MeixnerDistribution" => {
      if dargs.len() != 4 {
        return Err(InterpreterError::EvaluationError(
          "MeixnerDistribution expects 4 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      let m = dargs[2].clone();
      let d = dargs[3].clone();
      // Mean = m + a d Tan[b/2]
      let mean = plus(
        m,
        times(
          times(a.clone(), d.clone()),
          unary_fn("Tan", divide(b.clone(), int(2))),
        ),
      );
      // Variance = a^2 d Sec[b/2]^2 / 2
      let var = divide(
        times(
          times(power(a, int(2)), d),
          power(unary_fn("Sec", divide(b, int(2))), int(2)),
        ),
        int(2),
      );
      Ok((mean, var))
    }
    "SuzukiDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "SuzukiDistribution expects 2 arguments".into(),
        ));
      }
      let m = dargs[0].clone();
      let n = dargs[1].clone();
      let n2 = power(n.clone(), int(2));
      // Mean = E^(m + n^2/2) Sqrt[Pi/2]
      let mean = times(
        power(e(), plus(m.clone(), divide(n2.clone(), int(2)))),
        sqrt(divide(pi(), int(2))),
      );
      // Variance = E^(2 m + n^2) (2 E^(n^2) - Pi/2)
      let var = times(
        power(e(), plus(times(int(2), m), n2.clone())),
        minus(times(int(2), power(e(), n2)), divide(pi(), int(2))),
      );
      Ok((mean, var))
    }
    "PoissonConsulDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "PoissonConsulDistribution expects 2 arguments".into(),
        ));
      }
      let m = dargs[0].clone();
      let lam = dargs[1].clone();
      let one_minus_lam = minus(int(1), lam);
      // Mean = m/(1 - lam), Variance = m/(1 - lam)^3.
      let mean = divide(m.clone(), one_minus_lam.clone());
      let var = divide(m, power(one_minus_lam, int(3)));
      Ok((mean, var))
    }
    "LogSeriesDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "LogSeriesDistribution expects 1 argument".into(),
        ));
      }
      let t = dargs[0].clone();
      let log_1mt = unary_fn("Log", minus(int(1), t.clone()));
      // Mean = -(t/((1 - t) Log[1 - t]))
      let mean = times(
        int(-1),
        divide(t.clone(), times(minus(int(1), t.clone()), log_1mt.clone())),
      );
      // Var = -((t (t + Log[1 - t])) / ((-1 + t)^2 Log[1 - t]^2))
      let var = times(
        int(-1),
        divide(
          times(t.clone(), plus(t.clone(), log_1mt.clone())),
          times(power(plus(int(-1), t), int(2)), power(log_1mt, int(2))),
        ),
      );
      Ok((mean, var))
    }
    "NakagamiDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "NakagamiDistribution expects 2 arguments".into(),
        ));
      }
      let m = dargs[0].clone();
      let w = dargs[1].clone();
      let poch = call(
        "Pochhammer",
        vec![m.clone(), call("Rational", vec![int(1), int(2)])],
      );
      // Mean = (Sqrt[w] Pochhammer[m, 1/2])/Sqrt[m]
      let mean = divide(times(sqrt(w.clone()), poch.clone()), sqrt(m.clone()));
      // Var = w - (w Pochhammer[m, 1/2]^2)/m
      let var = minus(w.clone(), divide(times(w, power(poch, int(2))), m));
      Ok((mean, var))
    }
    "LogLogisticDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "LogLogisticDistribution expects 2 arguments".into(),
        ));
      }
      let g = dargs[0].clone();
      let s = dargs[1].clone();
      let csc = |arg: Expr| call("Csc", vec![arg]);
      // Mean = Piecewise[{{(Pi s Csc[Pi/g])/g, g > 1}}, Indeterminate]
      let mean_val = divide(
        times(times(pi(), s.clone()), csc(divide(pi(), g.clone()))),
        g.clone(),
      );
      let mean = piecewise(
        vec![(
          mean_val,
          comparison(g.clone(), ComparisonOp::Greater, int(1)),
        )],
        indeterminate(),
      );
      // Var = Piecewise[{{(Pi s^2 (-(Pi Csc[Pi/g]^2) + 2 g Csc[(2 Pi)/g]))/g^2,
      //   g > 2}}, Indeterminate]
      let csc1 = csc(divide(pi(), g.clone()));
      let csc2 = csc(divide(times(int(2), pi()), g.clone()));
      let inner = plus(
        times(int(-1), times(pi(), power(csc1, int(2)))),
        times(times(int(2), g.clone()), csc2),
      );
      let var_val = divide(
        times(times(pi(), power(s, int(2))), inner),
        power(g.clone(), int(2)),
      );
      let var = piecewise(
        vec![(var_val, comparison(g, ComparisonOp::Greater, int(2)))],
        indeterminate(),
      );
      Ok((mean, var))
    }
    "CauchyDistribution" => {
      // Mean and Variance are both Indeterminate for Cauchy
      let indet = indeterminate();
      Ok((indet.clone(), indet))
    }
    "LaplaceDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "LaplaceDistribution expects 2 arguments".into(),
        ));
      }
      let mu = dargs[0].clone();
      let b = dargs[1].clone();
      // Mean = mu, Var = 2*b^2
      let var = times(int(2), power(b, int(2)));
      Ok((mu, var))
    }
    "RayleighDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "RayleighDistribution expects 1 argument".into(),
        ));
      }
      let s = dargs[0].clone();
      // Mean = Sqrt[Pi/2] * s
      let mean = times(sqrt(divide(pi(), int(2))), s.clone());
      // Var = (2 - Pi/2) * s^2
      let var = times(minus(int(2), divide(pi(), int(2))), power(s, int(2)));
      Ok((mean, var))
    }
    "LevyDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "LevyDistribution expects 2 arguments".into(),
        ));
      }
      // Heavy-tailed: both Mean and Variance diverge.
      let infinity = infinity();
      Ok((infinity.clone(), infinity))
    }
    "LindleyDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "LindleyDistribution expects 1 argument".into(),
        ));
      }
      let d = dargs[0].clone();
      // Mean = (2 + d) / (d (1 + d))
      let mean = divide(
        plus(int(2), d.clone()),
        times(d.clone(), plus(int(1), d.clone())),
      );
      // Variance = 2/d^2 - (1 + d)^(-2)
      let var = minus(
        divide(int(2), power(d.clone(), int(2))),
        power(plus(int(1), d), int(-2)),
      );
      Ok((mean, var))
    }
    "BirnbaumSaundersDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "BirnbaumSaundersDistribution expects 2 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let l = dargs[1].clone();
      // Mean = (2 + a^2) / (2 l)
      let mean = divide(
        plus(int(2), power(a.clone(), int(2))),
        times(int(2), l.clone()),
      );
      // Variance = (a^2 (4 + 5 a^2)) / (4 l^2)
      let var = divide(
        times(
          power(a.clone(), int(2)),
          plus(int(4), times(int(5), power(a.clone(), int(2)))),
        ),
        times(int(4), power(l, int(2))),
      );
      Ok((mean, var))
    }
    "DiscreteUniformDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "DiscreteUniformDistribution expects 1 argument".into(),
        ));
      }
      let (imin, imax) = match &dargs[0] {
        Expr::List(bounds) if bounds.len() == 2 => {
          (bounds[0].clone(), bounds[1].clone())
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "DiscreteUniformDistribution expects a list {imin, imax}".into(),
          ));
        }
      };
      // Mean = (imin + imax) / 2
      let mean = divide(plus(imin.clone(), imax.clone()), int(2));
      // Variance = ((imax - imin + 1)^2 - 1) / 12, i.e. (n^2 - 1)/12 for the
      // n = imax - imin + 1 equally-likely values (wolframscript's form).
      let n = plus(int(1), minus(imax, imin));
      let var = divide(minus(power(n, int(2)), int(1)), int(12));
      Ok((mean, var))
    }
    "NegativeBinomialDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "NegativeBinomialDistribution expects 2 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let p = dargs[1].clone();
      // Mean = n*(1-p)/p, Var = n*(1-p)/p^2
      // Express variance as Times[n, (1-p), p^(-2)] so Sqrt can extract p^(-1)
      let one_minus_p = minus(int(1), p.clone());
      let mean = divide(times(n.clone(), one_minus_p.clone()), p.clone());
      let var = call("Times", vec![n, one_minus_p, power(p, int(-2))]);
      Ok((mean, var))
    }
    "PascalDistribution" => {
      if dargs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "PascalDistribution expects 2 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let p = dargs[1].clone();
      // Mean = n/p, Var = n*(1-p)/p^2
      let mean = divide(n.clone(), p.clone());
      let one_minus_p = minus(int(1), p.clone());
      let var = call("Times", vec![n, one_minus_p, power(p, int(-2))]);
      Ok((mean, var))
    }
    "DagumDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "DagumDistribution expects 3 arguments".into(),
        ));
      }
      let p = dargs[0].clone();
      let a = dargs[1].clone();
      let b = dargs[2].clone();
      // Mean = b * Gamma[(-1 + a)/a] * Gamma[1/a + p] / Gamma[p]
      let mean = times(
        times(
          b.clone(),
          gamma(divide(minus(a.clone(), int(1)), a.clone())),
        ),
        divide(
          gamma(plus(divide(int(1), a.clone()), p.clone())),
          gamma(p.clone()),
        ),
      );
      // Var = -Mean^2 + b^2 * Gamma[(-2+a)/a] * Gamma[2/a + p] / Gamma[p]
      let second_moment = times(
        times(
          power(b, int(2)),
          gamma(divide(minus(a.clone(), int(2)), a.clone())),
        ),
        divide(gamma(plus(divide(int(2), a), p.clone())), gamma(p)),
      );
      let var = minus(second_moment, power(mean.clone(), int(2)));
      Ok((mean, var))
    }
    "SinghMaddalaDistribution" => singh_maddala_mean_variance(dargs),
    "NoncentralFRatioDistribution" => {
      if dargs.len() != 3 {
        return Err(InterpreterError::EvaluationError(
          "NoncentralFRatioDistribution expects 3 arguments".into(),
        ));
      }
      let n = dargs[0].clone();
      let m = dargs[1].clone();
      let l = dargs[2].clone();

      // Mean = Piecewise[{{m*(l+n)/((m-2)*n), m > 2}}, Indeterminate]
      let mean_expr = divide(
        times(m.clone(), plus(l.clone(), n.clone())),
        times(plus(int(-2), m.clone()), n.clone()),
      );
      let mean = piecewise(
        vec![(
          mean_expr,
          comparison(m.clone(), ComparisonOp::Greater, int(2)),
        )],
        indeterminate(),
      );

      // Variance = Piecewise[{{2*m^2*((l+n)^2 + (m-2)*(2*l+n)) / ((m-4)*(m-2)^2*n^2), m > 4}}, Indeterminate]
      let l_plus_n = plus(l.clone(), n.clone());
      let var_num = times(
        times(int(2), power(m.clone(), int(2))),
        plus(
          power(l_plus_n, int(2)),
          times(plus(int(-2), m.clone()), plus(times(int(2), l), n.clone())),
        ),
      );
      let var_den = times(
        times(
          plus(int(-4), m.clone()),
          power(plus(int(-2), m.clone()), int(2)),
        ),
        power(n, int(2)),
      );
      let var_expr = divide(var_num, var_den);
      let var = piecewise(
        vec![(var_expr, comparison(m, ComparisonOp::Greater, int(4)))],
        indeterminate(),
      );

      Ok((mean, var))
    }
    "HyperbolicDistribution" => {
      if dargs.len() != 4 {
        return Err(InterpreterError::EvaluationError(
          "HyperbolicDistribution expects 4 arguments".into(),
        ));
      }
      let a = dargs[0].clone();
      let b = dargs[1].clone();
      let d = dargs[2].clone();
      let m = dargs[3].clone();

      let besselk = |n: Expr, z: Expr| call("BesselK", vec![n, z]);

      let a2_minus_b2 =
        minus(power(a.clone(), int(2)), power(b.clone(), int(2)));
      let sqrt_a2_minus_b2 = sqrt(a2_minus_b2.clone());
      let k_arg = times(sqrt_a2_minus_b2.clone(), d.clone());
      let bk1 = besselk(int(1), k_arg.clone());
      let bk2 = besselk(int(2), k_arg.clone());
      let bk3 = besselk(int(3), k_arg);

      // Mean = m + b*d*BesselK[2, Sqrt[a^2-b^2]*d] / (Sqrt[a^2-b^2]*BesselK[1, Sqrt[a^2-b^2]*d])
      let mean = plus(
        m,
        divide(
          times(times(b.clone(), d.clone()), bk2.clone()),
          times(sqrt_a2_minus_b2, bk1.clone()),
        ),
      );

      // Variance = d*BesselK[2,z]/(sqrt_ab*BesselK[1,z])
      //          - b^2*d^2*BesselK[2,z]^2/((a^2-b^2)*BesselK[1,z]^2)
      //          + b^2*d^2*BesselK[3,z]/((a^2-b^2)*BesselK[1,z])
      let sqrt_ab = sqrt(a2_minus_b2.clone());
      let b2d2 = times(power(b, int(2)), power(d.clone(), int(2)));
      let term1 = divide(times(d, bk2.clone()), times(sqrt_ab, bk1.clone()));
      let term2 = divide(
        times(b2d2.clone(), power(bk2, int(2))),
        times(a2_minus_b2.clone(), power(bk1.clone(), int(2))),
      );
      let term3 = divide(times(b2d2, bk3), times(a2_minus_b2, bk1));
      let var = plus(minus(term1, term2), term3);
      Ok((mean, var))
    }
    "JohnsonDistribution" => {
      if dargs.len() != 5 {
        return Err(InterpreterError::EvaluationError(
          "JohnsonDistribution expects 5 arguments".into(),
        ));
      }
      let type_str = match &dargs[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "JohnsonDistribution: first argument must be a string type".into(),
          ));
        }
      };
      let gamma = dargs[1].clone();
      let delta = dargs[2].clone();
      let mu = dargs[3].clone();
      let sigma = dargs[4].clone();
      match type_str.as_str() {
        "SN" => {
          // Mean = (delta*mu - gamma*sigma)/delta
          let mean = divide(
            minus(
              times(delta.clone(), mu),
              times(gamma.clone(), sigma.clone()),
            ),
            delta.clone(),
          );
          // Var = sigma^2/delta^2
          let var = divide(power(sigma, int(2)), power(delta, int(2)));
          Ok((mean, var))
        }
        "SL" => {
          // Mean = mu + E^((1 - 2*delta*gamma)/(2*delta^2)) * sigma
          let exp_arg = divide(
            minus(int(1), times(int(2), times(delta.clone(), gamma.clone()))),
            times(int(2), power(delta.clone(), int(2))),
          );
          let mean = plus(mu, times(power(e(), exp_arg), sigma.clone()));
          // Var = E^((1 - 2*delta*gamma)/delta^2) * (-1 + E^(1/delta^2)) * sigma^2
          let exp_arg2 = divide(
            minus(int(1), times(int(2), times(delta.clone(), gamma))),
            power(delta.clone(), int(2)),
          );
          let inv_delta_sq = divide(int(1), power(delta.clone(), int(2)));
          let var = times(
            times(
              power(e(), exp_arg2),
              minus(power(e(), inv_delta_sq), int(1)),
            ),
            power(sigma, int(2)),
          );
          Ok((mean, var))
        }
        "SU" => {
          // Mean = mu - sigma * E^(1/(2*delta^2)) * Sinh[gamma/delta]
          // (Wolfram expands Sinh differently, so this is added to skip list)
          let delta_sq = power(delta.clone(), int(2));
          let exp_half =
            power(e(), divide(int(1), times(int(2), delta_sq.clone())));
          let sinh_gd =
            call("Sinh", vec![divide(gamma.clone(), delta.clone())]);
          let mean = minus(mu, times(sigma.clone(), times(exp_half, sinh_gd)));
          // Var = (sigma^2/2) * (Exp[1/delta^2] - 1) * (Exp[1/delta^2]*Cosh[2*gamma/delta] + 1)
          let exp_full = power(e(), divide(int(1), delta_sq));
          let cosh_2gd =
            call("Cosh", vec![divide(times(int(2), gamma), delta)]);
          let var = times(
            divide(power(sigma, int(2)), int(2)),
            times(
              minus(exp_full.clone(), int(1)),
              plus(times(exp_full, cosh_2gd), int(1)),
            ),
          );
          Ok((mean, var))
        }
        "SB" => {
          // Mean for SB doesn't have a simple closed form
          Err(InterpreterError::EvaluationError(
            "JohnsonDistribution SB: no closed-form mean/variance".into(),
          ))
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "JohnsonDistribution: unknown type {type_str}"
        ))),
      }
    }
    "ArcSinDistribution" => {
      if dargs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "ArcSinDistribution expects 1 argument (a list {a, b})".into(),
        ));
      }
      let (a, b) = match &dargs[0] {
        Expr::List(bounds) if bounds.len() == 2 => {
          (bounds[0].clone(), bounds[1].clone())
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "ArcSinDistribution expects a list {a, b}".into(),
          ));
        }
      };
      // Mean = (a + b) / 2
      let mean = divide(plus(a.clone(), b.clone()), int(2));
      // Variance = (b - a)^2 / 8
      let var = divide(power(minus(b, a), int(2)), int(8));
      Ok((mean, var))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Expectation: unsupported distribution {dist_name}"
    ))),
  }
}

/// Check if expression is var^n
fn is_power_of_var(expr: &Expr, var: &str, n: i128) -> bool {
  match expr {
    Expr::BinaryOp { op, left, right } if *op == BinaryOperator::Power => {
      matches!(left.as_ref(), Expr::Identifier(v) if v == var)
        && matches!(right.as_ref(), Expr::Integer(k) if *k == n)
    }
    _ => false,
  }
}

/// Try to extract a linear expression a*x + b from expr.
/// Returns Some((a, b)) if expr is linear in var.
fn extract_linear(expr: &Expr, var: &str) -> Option<(Expr, Expr)> {
  match expr {
    // Pure variable: x → (1, 0)
    Expr::Identifier(n) if n == var => Some((int(1), int(0))),
    // a * x
    Expr::BinaryOp { op, left, right } if *op == BinaryOperator::Times => {
      if matches!(right.as_ref(), Expr::Identifier(n) if n == var)
        && !contains_var(left, var)
      {
        Some((left.as_ref().clone(), int(0)))
      } else if matches!(left.as_ref(), Expr::Identifier(n) if n == var)
        && !contains_var(right, var)
      {
        Some((right.as_ref().clone(), int(0)))
      } else {
        None
      }
    }
    // a + b (try linear decomposition)
    Expr::BinaryOp { op, left, right } if *op == BinaryOperator::Plus => {
      // If left is linear in var and right is constant (or vice versa)
      if !contains_var(right, var)
        && let Some((a, b)) = extract_linear(left, var)
      {
        let new_b = plus(b, right.as_ref().clone());
        return Some((a, new_b));
      }
      if !contains_var(left, var)
        && let Some((a, b)) = extract_linear(right, var)
      {
        let new_b = plus(left.as_ref().clone(), b);
        return Some((a, new_b));
      }
      None
    }
    Expr::BinaryOp { op, left, right } if *op == BinaryOperator::Minus => {
      if !contains_var(right, var)
        && let Some((a, b)) = extract_linear(left, var)
      {
        let new_b = minus(b, right.as_ref().clone());
        return Some((a, new_b));
      }
      None
    }
    // FunctionCall Times[...] form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Find which arg is the variable and which are constants
      let mut var_idx = None;
      for (i, arg) in args.iter().enumerate() {
        if contains_var(arg, var) {
          if var_idx.is_some() {
            return None; // multiple args contain var
          }
          var_idx = Some(i);
        }
      }
      if let Some(vi) = var_idx {
        if !matches!(&args[vi], Expr::Identifier(n) if n == var) {
          return None;
        }
        let mut coeff_parts: Vec<Expr> = Vec::new();
        for (i, arg) in args.iter().enumerate() {
          if i != vi {
            coeff_parts.push(arg.clone());
          }
        }
        let coeff = if coeff_parts.len() == 1 {
          coeff_parts.pop().unwrap()
        } else {
          call("Times", coeff_parts)
        };
        Some((coeff, int(0)))
      } else {
        None
      }
    }
    // FunctionCall Plus[...] form
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut total_a = int(0);
      let mut total_b = int(0);
      for arg in args {
        if contains_var(arg, var) {
          if let Some((a, b)) = extract_linear(arg, var) {
            total_a = plus(total_a, a);
            total_b = plus(total_b, b);
          } else {
            return None;
          }
        } else {
          total_b = plus(total_b, arg.clone());
        }
      }
      Some((total_a, total_b))
    }
    // Constant (no var)
    _ if !contains_var(expr, var) => None, // Not linear, it's constant - caller handles
    _ => None,
  }
}

fn contains_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(n) => n == var,
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) => false,
    Expr::BinaryOp { left, right, .. } => {
      contains_var(left, var) || contains_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_var(operand, var),
    Expr::FunctionCall { name, args } => {
      if name == "Rational" && args.len() == 2 {
        return false; // Rational[n, d] is a constant
      }
      args.iter().any(|a| contains_var(a, var))
    }
    Expr::List(items) => items.iter().any(|a| contains_var(a, var)),
    _ => true, // conservative
  }
}

/// Numerical expectation via Monte Carlo or numerical integration.
fn expectation_numerical(
  expr: &Expr,
  var: &str,
  dist_name: &str,
  dargs: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::functions::plot::substitute_var;

  // Get integration range and PDF for quadrature
  let n_points = 1000;

  // Determine integration bounds based on distribution
  let (lo, hi): (f64, f64) = match dist_name {
    "UniformDistribution" => {
      let (a, b) = match dargs.len() {
        0 => (0.0, 1.0),
        1 => {
          if let Expr::List(bounds) = &dargs[0] {
            if bounds.len() == 2 {
              let a = try_eval_to_f64(&bounds[0]).unwrap_or(0.0);
              let b = try_eval_to_f64(&bounds[1]).unwrap_or(1.0);
              (a, b)
            } else {
              (0.0, 1.0)
            }
          } else {
            (0.0, 1.0)
          }
        }
        _ => (0.0, 1.0),
      };
      (a, b)
    }
    "ExponentialDistribution" => {
      // Integrate from 0 to ~10/lambda
      let lambda = try_eval_to_f64(&dargs[0]).unwrap_or(1.0);
      (0.0, 10.0 / lambda)
    }
    "NormalDistribution" => {
      let (mu, sigma) = match dargs.len() {
        0 => (0.0, 1.0),
        2 => {
          let m = try_eval_to_f64(&dargs[0]).unwrap_or(0.0);
          let s = try_eval_to_f64(&dargs[1]).unwrap_or(1.0);
          (m, s)
        }
        _ => (0.0, 1.0),
      };
      (mu - 6.0 * sigma, mu + 6.0 * sigma)
    }
    _ => {
      // Return unevaluated for unsupported distributions
      return Ok(call(
        "Expectation",
        vec![
          expr.clone(),
          call(
            "Distributed",
            vec![
              Expr::Identifier(var.to_string()),
              unevaluated(dist_name, dargs),
            ],
          ),
        ],
      ));
    }
  };

  // Numerical integration: E[f(x)] = integral f(x) * pdf(x) dx
  let dx = (hi - lo) / n_points as f64;
  let mut sum = 0.0;
  let dist_expr = unevaluated(dist_name, dargs);

  for i in 0..=n_points {
    let x = lo + i as f64 * dx;
    let x_expr = Expr::Real(x);

    // Evaluate f(x)
    let fx_sub = substitute_var(expr, var, &x_expr);
    let fx_val = evaluate_expr_to_expr(&fx_sub)
      .ok()
      .and_then(|e| try_eval_to_f64(&e))
      .unwrap_or(0.0);

    // Evaluate pdf(x)
    let pdf_val = pdf_ast(&[dist_expr.clone(), x_expr])
      .ok()
      .and_then(|e| try_eval_to_f64(&e))
      .unwrap_or(0.0);

    let weight = if i == 0 || i == n_points { 0.5 } else { 1.0 };
    sum += weight * fx_val * pdf_val;
  }
  sum *= dx;

  // Round to reasonable precision
  let result = (sum * 1e10).round() / 1e10;
  if (result - result.round()).abs() < 1e-8 {
    Ok(Expr::Integer(result.round() as i128))
  } else {
    Ok(Expr::Real(result))
  }
}

/// PDF[BetaDistribution[a, b], x] = Piecewise[{{x^(a-1) * (1-x)^(b-1) / Beta[a,b], 0 < x < 1}}, 0]
fn pdf_beta(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BetaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  // x^(a-1)
  let x_part = power(x.clone(), minus(a.clone(), int(1)));
  // (1-x)^(b-1)
  let one_minus_x_part =
    power(minus(int(1), x.clone()), minus(b.clone(), int(1)));
  // Beta[a, b]
  let beta_fn = call("Beta", vec![a, b]);
  let value = eval(divide(times(x_part, one_minus_x_part), beta_fn))?;
  let cond =
    comparison3(int(0), ComparisonOp::Less, x, ComparisonOp::Less, int(1));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// Extract ({a, b}, m, g) from PERTDistribution[{a, b}, m] (g defaults to
/// 4) or PERTDistribution[{a, b}, m, g]. Returns None for malformed specs.
fn pert_params(dargs: &[Expr]) -> Option<(Expr, Expr, Expr, Expr)> {
  if dargs.len() != 2 && dargs.len() != 3 {
    return None;
  }
  let Expr::List(minmax) = &dargs[0] else {
    return None;
  };
  if minmax.len() != 2 {
    return None;
  }
  let g = if dargs.len() == 3 {
    dargs[2].clone()
  } else {
    int(4)
  };
  Some((minmax[0].clone(), minmax[1].clone(), dargs[1].clone(), g))
}

/// PDF[PERTDistribution[{a, b}, m], x] — a Beta distribution rescaled to
/// (a, b) with shape exponents g (m - a)/(b - a) and g (b - m)/(b - a);
/// the default (PERT) shape parameter is g = 4. Machine-real arguments
/// numericize the result like wolframscript.
fn pdf_pert(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let Some((a, b, m, g)) = pert_params(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "PERTDistribution expects {min, max} and a mode".into(),
    ));
  };
  let real_arg = matches!(x, Expr::Real(_) | Expr::BigFloat(..));
  let width = minus(b.clone(), a.clone());
  let e_low =
    divide(times(g.clone(), minus(m.clone(), a.clone())), width.clone());
  let e_high =
    divide(times(g.clone(), minus(b.clone(), m.clone())), width.clone());
  let beta_call = call(
    "Beta",
    vec![plus(int(1), e_low.clone()), plus(int(1), e_high.clone())],
  );
  let powers = times(
    power(minus(b.clone(), x.clone()), e_high),
    power(minus(x.clone(), a.clone()), e_low),
  );
  // The 2-argument (g = 4) form divides by (b-a)^5; the modified form
  // keeps (b-a)^(-1-g) as a leading factor — matching wolframscript's
  // displayed shapes.
  let value = if dargs.len() == 2 {
    divide(powers, times(power(width, int(5)), beta_call))
  } else {
    divide(
      times(power(width, minus(int(-1), g.clone()).clone()), powers),
      beta_call,
    )
  };
  let cond = comparison3(a, ComparisonOp::Less, x, ComparisonOp::Less, b);
  let result = eval(piecewise(vec![(value, cond)], int(0)))?;
  if real_arg {
    return eval(unary_fn("N", result));
  }
  Ok(result)
}

/// CDF[PERTDistribution[{a, b}, m], x] =
///   Piecewise[{{BetaRegularized[(x-a)/(b-a), 1 + g(m-a)/(b-a),
///   1 + g(b-m)/(b-a)], a < x < b}, {1, x >= b}}, 0].
fn cdf_pert(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let Some((a, b, m, g)) = pert_params(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "PERTDistribution expects {min, max} and a mode".into(),
    ));
  };
  let real_arg = matches!(x, Expr::Real(_) | Expr::BigFloat(..));
  let width = minus(b.clone(), a.clone());
  let e_low =
    divide(times(g.clone(), minus(m.clone(), a.clone())), width.clone());
  let e_high = divide(times(g, minus(b.clone(), m)), width.clone());
  let reg = call(
    "BetaRegularized",
    vec![
      divide(minus(x.clone(), a.clone()), width),
      plus(int(1), e_low),
      plus(int(1), e_high),
    ],
  );
  let cond1 = comparison3(
    a,
    ComparisonOp::Less,
    x.clone(),
    ComparisonOp::Less,
    b.clone(),
  );
  let cond2 = comparison(x, ComparisonOp::GreaterEqual, b);
  let result = eval(piecewise(vec![(reg, cond1), (int(1), cond2)], int(0)))?;
  if real_arg {
    return eval(unary_fn("N", result));
  }
  Ok(result)
}

/// PDF[PowerDistribution[k, a], x] =
///   Piecewise[{{a k^a x^(a-1), 0 < x <= 1/k}}, 0].
fn pdf_power(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PowerDistribution expects 2 arguments".into(),
    ));
  }
  let (k, a) = (dargs[0].clone(), dargs[1].clone());
  let value = times(
    times(a.clone(), power(k.clone(), a.clone())),
    power(x.clone(), minus(a, int(1))),
  );
  let real_arg = matches!(x, Expr::Real(_) | Expr::BigFloat(..));
  let cond = comparison3(
    int(0),
    ComparisonOp::Less,
    x,
    ComparisonOp::LessEqual,
    power(k, int(-1)),
  );
  let result = eval(piecewise(vec![(value, cond)], int(0)))?;
  // wolframscript numericizes the whole result for machine-real arguments
  // (0. and 1. outside the support), unlike most other distributions.
  if real_arg {
    return eval(unary_fn("N", result));
  }
  Ok(result)
}

/// CDF[PowerDistribution[k, a], x] =
///   Piecewise[{{(k x)^a, 0 < x <= 1/k}, {1, x > 1/k}}, 0].
fn cdf_power(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PowerDistribution expects 2 arguments".into(),
    ));
  }
  let (k, a) = (dargs[0].clone(), dargs[1].clone());
  let value = power(times(k.clone(), x.clone()), a.clone());
  let cond1 = comparison3(
    int(0),
    ComparisonOp::Less,
    x.clone(),
    ComparisonOp::LessEqual,
    power(k.clone(), int(-1)),
  );
  let real_arg = matches!(x, Expr::Real(_) | Expr::BigFloat(..));
  let cond2 = comparison(x, ComparisonOp::Greater, power(k, int(-1)));
  let result = eval(piecewise(vec![(value, cond1), (int(1), cond2)], int(0)))?;
  if real_arg {
    return eval(unary_fn("N", result));
  }
  Ok(result)
}

/// PDF[KumaraswamyDistribution[a, b], x] =
///   Piecewise[{{a b x^(a-1) (1 - x^a)^(b-1), 0 < x < 1}}, 0].
fn pdf_kumaraswamy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KumaraswamyDistribution expects 2 arguments".into(),
    ));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let x_pow = power(x.clone(), minus(a.clone(), int(1)));
  let one_minus = power(
    minus(int(1), power(x.clone(), a.clone())),
    minus(b.clone(), int(1)),
  );
  let value = times(times(a, b), times(x_pow, one_minus));
  let cond =
    comparison3(int(0), ComparisonOp::Less, x, ComparisonOp::Less, int(1));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[KumaraswamyDistribution[a, b], x] =
///   Piecewise[{{1 - (1 - x^a)^b, 0 < x < 1}, {1, x >= 1}}, 0].
fn cdf_kumaraswamy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KumaraswamyDistribution expects 2 arguments".into(),
    ));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let value = minus(int(1), power(minus(int(1), power(x.clone(), a)), b));
  let cond1 = comparison3(
    int(0),
    ComparisonOp::Less,
    x.clone(),
    ComparisonOp::Less,
    int(1),
  );
  let cond2 = comparison(x, ComparisonOp::GreaterEqual, int(1));
  eval(piecewise(vec![(value, cond1), (int(1), cond2)], int(0)))
}

/// PDF[LogGammaDistribution[a, b, m], x] =
///   Piecewise[{{Log[1-m+x]^(a-1) / (b^a (1-m+x)^((1+b)/b) Gamma[a]), x >= m}}, 0].
fn pdf_loggamma(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "LogGammaDistribution expects 3 arguments".into(),
    ));
  }
  let (a, b, m) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  // 1 - m + x
  let shifted = plus(plus(int(1), times(int(-1), m.clone())), x.clone());
  let num = power(unary_fn("Log", shifted.clone()), minus(a.clone(), int(1)));
  let den = times(
    power(b.clone(), a.clone()),
    times(
      power(shifted, divide(plus(int(1), b.clone()), b)),
      unary_fn("Gamma", a),
    ),
  );
  let value = divide(num, den);
  let cond = comparison(x, ComparisonOp::GreaterEqual, m);
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[LogGammaDistribution[a, b, m], x] =
///   Piecewise[{{GammaRegularized[a, 0, Log[1-m+x]/b], x >= m}}, 0].
fn cdf_loggamma(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "LogGammaDistribution expects 3 arguments".into(),
    ));
  }
  let (a, b, m) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let shifted = plus(plus(int(1), times(int(-1), m.clone())), x.clone());
  let reg = call(
    "GammaRegularized",
    vec![a, int(0), divide(unary_fn("Log", shifted), b)],
  );
  let cond = comparison(x, ComparisonOp::GreaterEqual, m);
  eval(piecewise(vec![(reg, cond)], int(0)))
}

/// CDF[ExpGammaDistribution[k, t, m], x] = GammaRegularized[k, 0, E^((x-m)/t)].
fn cdf_expgamma(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "ExpGammaDistribution expects 3 arguments".into(),
    ));
  }
  let (k, t, m) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let arg = power(e(), divide(minus(x, m), t));
  eval(call("GammaRegularized", vec![k, int(0), arg]))
}

/// CDF[BetaDistribution[a, b], x] = Piecewise[{{BetaRegularized[x, a, b], 0 < x < 1}, {1, x >= 1}}, 0]
fn cdf_beta(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BetaDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let value = call("BetaRegularized", vec![x.clone(), a, b]);
  let cond1 = comparison3(
    int(0),
    ComparisonOp::Less,
    x.clone(),
    ComparisonOp::Less,
    int(1),
  );
  let cond2 = comparison(x, ComparisonOp::GreaterEqual, int(1));
  eval(piecewise(vec![(value, cond1), (int(1), cond2)], int(0)))
}

/// CDF[StudentTDistribution[nu], x] = Piecewise[
///   {{BetaRegularized[nu/(nu + x^2), nu/2, 1/2]/2, x <= 0}},
///   (1 + BetaRegularized[x^2/(nu + x^2), 1/2, nu/2])/2]
fn cdf_student_t(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  // Standard: numerator nu, x^2, threshold 0. Location-scale (3-arg, [m,s,nu]):
  // numerator s^2 nu, (x-m)^2, threshold m.
  let (nu, num_lead, x_sq, threshold) = match dargs.len() {
    1 => {
      let nu = dargs[0].clone();
      (nu.clone(), nu, power(x.clone(), int(2)), int(0))
    }
    3 => {
      let (m, s, nu) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
      let s2nu = times(power(s, int(2)), nu.clone());
      (nu, s2nu, power(minus(x.clone(), m.clone()), int(2)), m)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "StudentTDistribution expects 1 or 3 arguments".into(),
      ));
    }
  };
  let half = divide(int(1), int(2));
  let nu_over_2 = divide(nu, int(2));
  let denom = plus(num_lead.clone(), x_sq.clone());
  // Left branch: BetaRegularized[num_lead/denom, nu/2, 1/2] / 2
  let left_arg = divide(num_lead, denom.clone());
  let left_beta = call(
    "BetaRegularized",
    vec![left_arg, nu_over_2.clone(), half.clone()],
  );
  let left_value = divide(left_beta, int(2));
  // Right branch: (1 + BetaRegularized[x_sq/denom, 1/2, nu/2]) / 2
  let right_arg = divide(x_sq, denom);
  let right_beta = call("BetaRegularized", vec![right_arg, half, nu_over_2]);
  let right_value = divide(plus(int(1), right_beta), int(2));
  let cond = comparison(x, ComparisonOp::LessEqual, threshold);
  eval(piecewise(vec![(left_value, cond)], right_value))
}

/// PDF[StudentTDistribution[nu], x] = (1 + x^2/nu)^(-(1+nu)/2) / (Sqrt[nu] * Beta[nu/2, 1/2])
fn pdf_student_t(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  pdf_student_t_impl(dargs, x)
}

/// PDF[StudentTDistribution[nu], x] = (nu/(nu+x^2))^((1+nu)/2) / (Sqrt[nu] Beta[nu/2, 1/2]);
/// PDF[StudentTDistribution[m, s, nu], x] uses (x-m)^2/s^2 and an extra factor s.
fn pdf_student_t_impl(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let (loc_scale, nu) = match dargs.len() {
    1 => (None, dargs[0].clone()),
    3 => (Some((dargs[0].clone(), dargs[1].clone())), dargs[2].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "StudentTDistribution expects 1 or 3 arguments".into(),
      ));
    }
  };
  // x_term = x^2 (standard) or (x-m)^2/s^2 (location-scale).
  let x_term = match &loc_scale {
    Some((m, s)) => {
      divide(power(minus(x, m.clone()), int(2)), power(s.clone(), int(2)))
    }
    None => power(x, int(2)),
  };
  // (nu/(nu + x_term))^((1+nu)/2)
  let inner = divide(nu.clone(), plus(nu.clone(), x_term));
  let exponent = divide(plus(int(1), nu.clone()), int(2));
  let numerator = power(inner, exponent);
  let beta = call(
    "Beta",
    vec![divide(nu.clone(), int(2)), divide(int(1), int(2))],
  );
  // Denominator: [s *] Sqrt[nu] * Beta[nu/2, 1/2].
  let denominator = match &loc_scale {
    Some((_, s)) => times(s.clone(), times(sqrt(nu.clone()), beta)),
    None => times(sqrt(nu), beta),
  };
  eval(divide(numerator, denominator))
}

/// PDF[LogNormalDistribution[mu, sigma], x] = Piecewise[{{1/(E^((Log[x]-mu)^2/(2*sigma^2))*Sqrt[2*Pi]*sigma*x), x > 0}}, 0]
fn pdf_lognormal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogNormalDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let sigma = dargs[1].clone();

  // 1 / (E^((Log[x] - mu)^2 / (2*sigma^2)) * Sqrt[2*Pi] * sigma * x)
  let log_x = call("Log", vec![x.clone()]);
  let exponent = divide(
    power(minus(log_x, mu), int(2)),
    times(int(2), power(sigma.clone(), int(2))),
  );
  let denom = times(
    times(
      power(Expr::Identifier("E".to_string()), exponent),
      sqrt(times(int(2), Expr::Identifier("Pi".to_string()))),
    ),
    times(sigma, x.clone()),
  );
  let pdf_val = divide(int(1), denom);

  // Piecewise[{{pdf_val, x > 0}}, 0]
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[LogNormalDistribution[mu, sigma], x] = Piecewise[{{Erfc[-(Log[x]-mu)/(Sqrt[2]*sigma)]/2, x > 0}}, 0]
fn cdf_lognormal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LogNormalDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let sigma = dargs[1].clone();

  // Erfc[-(Log[x] - mu) / (Sqrt[2] * sigma)] / 2
  let log_x = call("Log", vec![x.clone()]);
  let arg = neg(divide(minus(log_x, mu), times(sqrt(int(2)), sigma)));
  let cdf_val = divide(call("Erfc", vec![arg]), int(2));

  // Piecewise[{{cdf_val, x > 0}}, 0]
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[ChiSquareDistribution[k], x] = Piecewise[{{x^(k/2-1) / (2^(k/2) * E^(x/2) * Gamma[k/2]), x > 0}}, 0]
fn pdf_chi_square(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let k = dargs[0].clone();

  // x^(k/2 - 1)
  let x_power = power(x.clone(), minus(divide(k.clone(), int(2)), int(1)));
  // 2^(k/2)
  let two_power = power(int(2), divide(k.clone(), int(2)));
  // E^(x/2)
  let exp_part = power(e(), divide(x.clone(), int(2)));
  // Gamma[k/2]
  let gamma_part = call("Gamma", vec![divide(k, int(2))]);
  let denom = times(times(two_power, exp_part), gamma_part);
  let pdf_val = divide(x_power, denom);

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// PDF[FRatioDistribution[n, m], x] =
///   Piecewise[{{n^(n/2) m^(m/2) x^(n/2-1) / ((m + n x)^((n+m)/2) Beta[n/2, m/2]),
///               x > 0}}, 0].
/// The evaluator reduces the raw formula to wolframscript's printed form.
fn pdf_f_ratio(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FRatioDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let m = dargs[1].clone();

  // n^(n/2) * m^(m/2) * x^(n/2 - 1)
  let numer = times(
    times(
      power(n.clone(), divide(n.clone(), int(2))),
      power(m.clone(), divide(m.clone(), int(2))),
    ),
    power(x.clone(), minus(divide(n.clone(), int(2)), int(1))),
  );
  // (m + n*x)^((n+m)/2)
  let denom_power = power(
    plus(m.clone(), times(n.clone(), x.clone())),
    divide(plus(n.clone(), m.clone()), int(2)),
  );
  // Beta[n/2, m/2]
  let beta = call("Beta", vec![divide(n, int(2)), divide(m, int(2))]);
  let pdf_val = divide(numer, times(denom_power, beta));

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[FRatioDistribution[n, m], x] =
///   Piecewise[{{BetaRegularized[n x / (n x + m), n/2, m/2], x > 0}}, 0].
fn cdf_f_ratio(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FRatioDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let m = dargs[1].clone();

  // n x / (n x + m)
  let nx = times(n.clone(), x.clone());
  let arg = divide(nx.clone(), plus(nx, m.clone()));
  let reg = call(
    "BetaRegularized",
    vec![arg, divide(n, int(2)), divide(m, int(2))],
  );

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(reg, cond)], int(0)))
}

/// PDF[WaringYuleDistribution[a, b], k] =
///   Piecewise[{{a Pochhammer[b, k] / Pochhammer[a + b, 1 + k], k >= 0}}, 0].
fn pdf_waring_yule(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WaringYuleDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let poch =
    |first: Expr, second: Expr| call("Pochhammer", vec![first, second]);
  // a * Pochhammer[b, k] / Pochhammer[a + b, 1 + k]
  let numer = times(a.clone(), poch(b.clone(), x.clone()));
  let denom = poch(plus(a, b), plus(int(1), x.clone()));
  let pmf = divide(numer, denom);

  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(pmf, cond)], int(0)))
}

/// CDF[WaringYuleDistribution[a, b], k] =
///   Piecewise[{{1 - Pochhammer[b, 1 + Floor[k]] / Pochhammer[a + b, 1 + Floor[k]],
///               k >= 0}}, 0].
fn cdf_waring_yule(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WaringYuleDistribution expects 2 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();

  let floor_k = call("Floor", vec![x.clone()]);
  let idx = plus(int(1), floor_k);
  let poch =
    |first: Expr, second: Expr| call("Pochhammer", vec![first, second]);
  // 1 - Pochhammer[b, 1 + Floor[k]] / Pochhammer[a + b, 1 + Floor[k]]
  let ratio = divide(poch(b.clone(), idx.clone()), poch(plus(a, b), idx));
  let cdf = minus(int(1), ratio);

  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(cdf, cond)], int(0)))
}

/// CDF[ChiSquareDistribution[k], x] = Piecewise[{{GammaRegularized[k/2, 0, x/2], x > 0}}, 0]
fn cdf_chi_square(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiSquareDistribution expects 1 argument".into(),
    ));
  }
  let k = dargs[0].clone();

  let cdf_val = call(
    "GammaRegularized",
    vec![divide(k, int(2)), int(0), divide(x.clone(), int(2))],
  );

  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[ParetoDistribution[k, a], x] = Piecewise[{{a*k^a*x^(-1-a), x >= k}}, 0]
fn pdf_pareto(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  // 3-argument (Type II / Lomax, scale k, location m) and 4-argument (extra
  // shape g) generalized forms.
  if dargs.len() == 3 || dargs.len() == 4 {
    let (k, a, m) = (dargs[0].clone(), dargs[1].clone(), {
      if dargs.len() == 3 {
        dargs[2].clone()
      } else {
        dargs[3].clone()
      }
    });
    let cond = comparison(x.clone(), ComparisonOp::GreaterEqual, m.clone());
    let body = if dargs.len() == 3 {
      // (a ((k - m + x)/k)^(-1 - a)) / k
      let inner =
        divide(plus(minus(k.clone(), m.clone()), x.clone()), k.clone());
      divide(times(a.clone(), power(inner, minus(int(-1), a))), k)
    } else {
      // (a (-m + x)^(-1 + 1/g) (1 + (k/(-m + x))^(-1/g))^(-1 - a)) / (g k^(1/g))
      let g = dargs[2].clone();
      let inv_g = divide(int(1), g.clone());
      let xm = minus(x.clone(), m.clone());
      let factor1 = power(xm.clone(), plus(int(-1), inv_g.clone()));
      let inner2 = power(divide(k.clone(), xm), times(int(-1), inv_g.clone()));
      let factor2 = power(plus(int(1), inner2), minus(int(-1), a.clone()));
      let num = times(a, times(factor1, factor2));
      let den = times(g, power(k, inv_g));
      divide(num, den)
    };
    return eval(piecewise(vec![(body, cond)], int(0)));
  }
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ParetoDistribution expects 2 arguments".into(),
    ));
  }
  let k = dargs[0].clone();
  let a = dargs[1].clone();

  // a * k^a * x^(-1-a)
  let pdf_val = times(
    times(a.clone(), power(k.clone(), a.clone())),
    power(x.clone(), neg(plus(int(1), a))),
  );
  let cond = comparison(x, ComparisonOp::GreaterEqual, k);
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[ParetoDistribution[k, a], x] = Piecewise[{{1 - (k/x)^a, x >= k}}, 0]
fn cdf_pareto(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() == 3 || dargs.len() == 4 {
    let (k, a, m) = (dargs[0].clone(), dargs[1].clone(), {
      if dargs.len() == 3 {
        dargs[2].clone()
      } else {
        dargs[3].clone()
      }
    });
    let cond = comparison(x.clone(), ComparisonOp::GreaterEqual, m.clone());
    let neg_a = times(int(-1), a);
    let body = if dargs.len() == 3 {
      // 1 - (1 + (-m + x)/k)^(-a)
      minus(
        int(1),
        power(plus(int(1), divide(minus(x.clone(), m), k)), neg_a),
      )
    } else {
      // 1 - (1 + ((-m + x)/k)^(1/g))^(-a)
      let g = dargs[2].clone();
      let ratio = power(divide(minus(x.clone(), m), k), divide(int(1), g));
      minus(int(1), power(plus(int(1), ratio), neg_a))
    };
    return eval(piecewise(vec![(body, cond)], int(0)));
  }
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ParetoDistribution expects 2 arguments".into(),
    ));
  }
  let k = dargs[0].clone();
  let a = dargs[1].clone();

  let cdf_val = minus(int(1), power(divide(k.clone(), x.clone()), a));
  let cond = comparison(x, ComparisonOp::GreaterEqual, k);
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[WeibullDistribution[a, b], x] = Piecewise[{{a*(x/b)^(a-1) / (b * E^((x/b)^a)), x > 0}}, 0]
fn pdf_weibull(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 && dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "WeibullDistribution expects 2 or 3 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();
  // The 3-argument form shifts the support by the location m: the body uses
  // (x - m) and the condition becomes x > m (rather than x > 0).
  let (xv, cond) = if dargs.len() == 3 {
    let m = dargs[2].clone();
    (
      minus(x.clone(), m.clone()),
      comparison(x, ComparisonOp::Greater, m),
    )
  } else {
    (x.clone(), comparison(x, ComparisonOp::Greater, int(0)))
  };

  // a * (xv/b)^(a-1) / (b * E^((xv/b)^a))
  let xb = divide(xv, b.clone());
  let numerator = times(a.clone(), power(xb.clone(), minus(a.clone(), int(1))));
  let denom = times(b, power(e(), power(xb, a)));
  let pdf_val = divide(numerator, denom);

  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[WeibullDistribution[a, b], x] = Piecewise[{{1 - E^(-(x/b)^a), x > 0}}, 0]
fn cdf_weibull(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 && dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "WeibullDistribution expects 2 or 3 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();
  // The 3-argument form shifts the support by the location m.
  let (xv, cond) = if dargs.len() == 3 {
    let m = dargs[2].clone();
    (
      minus(x.clone(), m.clone()),
      comparison(x, ComparisonOp::Greater, m),
    )
  } else {
    (x.clone(), comparison(x, ComparisonOp::Greater, int(0)))
  };

  let xb = divide(xv, b);
  let cdf_val = minus(int(1), power(e(), neg(power(xb, a))));

  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[DiscreteUniformDistribution[{imin, imax}], x] = Piecewise[{{1/(imax-imin+1), imin <= x <= imax}}, 0]
fn pdf_discrete_uniform(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DiscreteUniformDistribution expects 1 argument".into(),
    ));
  }
  let (imin, imax) = match &dargs[0] {
    Expr::List(bounds) if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DiscreteUniformDistribution expects a list {imin, imax}".into(),
      ));
    }
  };
  let n = eval(plus(minus(imax.clone(), imin.clone()), int(1)))?;
  let pdf_val = divide(int(1), n);
  let cond = comparison3(
    imin,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::LessEqual,
    imax,
  );
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[DiscreteUniformDistribution[{imin, imax}], x]
fn cdf_discrete_uniform(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DiscreteUniformDistribution expects 1 argument".into(),
    ));
  }
  let (imin, imax) = match &dargs[0] {
    Expr::List(bounds) if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DiscreteUniformDistribution expects a list {imin, imax}".into(),
      ));
    }
  };
  let n = eval(plus(minus(imax.clone(), imin.clone()), int(1)))?;
  let floor_x = call("Floor", vec![x.clone()]);
  let cdf_val = divide(plus(minus(floor_x, imin.clone()), int(1)), n);
  let cond_low = comparison(x.clone(), ComparisonOp::Less, imin.clone());
  let cond_mid = comparison3(
    imin,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::Less,
    imax.clone(),
  );
  let cond_high = comparison(x, ComparisonOp::GreaterEqual, imax);
  eval(piecewise(
    vec![(int(0), cond_low), (cdf_val, cond_mid), (int(1), cond_high)],
    int(0),
  ))
}

/// PDF[LaplaceDistribution[mu, b], x] = E^(-Abs[x-mu]/b) / (2*b)
fn pdf_laplace(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LaplaceDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let b = dargs[1].clone();
  let abs_diff = call("Abs", vec![minus(x, mu)]);
  let pdf_val = divide(
    power(e(), neg(divide(abs_diff, b.clone()))),
    times(int(2), b),
  );
  eval(pdf_val)
}

/// CDF[LaplaceDistribution[mu, b], x] = Piecewise[{{E^((x-mu)/b)/2, x < mu}}, 1 - E^(-(x-mu)/b)/2]
fn cdf_laplace(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LaplaceDistribution expects 2 arguments".into(),
    ));
  }
  let mu = dargs[0].clone();
  let b = dargs[1].clone();
  let diff = minus(x.clone(), mu.clone());
  let low_val = divide(power(e(), divide(diff.clone(), b.clone())), int(2));
  let high_val =
    minus(int(1), divide(power(e(), neg(divide(diff, b))), int(2)));
  let cond = comparison(x, ComparisonOp::Less, mu);
  eval(piecewise(vec![(low_val, cond)], high_val))
}

/// PDF[RayleighDistribution[sigma], x] = (x/sigma^2) * E^(-x^2/(2*sigma^2)), x > 0
fn pdf_rayleigh(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RayleighDistribution expects 1 argument".into(),
    ));
  }
  let sigma = dargs[0].clone();
  let s2 = power(sigma, int(2));
  let pdf_val = times(
    divide(x.clone(), s2.clone()),
    power(
      e(),
      neg(divide(power(x.clone(), int(2)), times(int(2), s2))),
    ),
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[RayleighDistribution[sigma], x] = 1 - E^(-x^2/(2*sigma^2)), x > 0
fn cdf_rayleigh(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RayleighDistribution expects 1 argument".into(),
    ));
  }
  let sigma = dargs[0].clone();
  let s2 = power(sigma, int(2));
  let cdf_val = minus(
    int(1),
    power(
      e(),
      neg(divide(power(x.clone(), int(2)), times(int(2), s2))),
    ),
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(cdf_val, cond)], int(0)))
}

/// PDF[MultinomialDistribution[n, {p1, ..., pm}], {x1, ..., xm}]
/// = n! / (x1! * ... * xm!) * p1^x1 * ... * pm^xm when sum(xi) == n and all xi >= 0
/// = 0 otherwise
/// Expressed via products of Binomial coefficients:
/// Binomial[x1+x2, x2] * Binomial[x1+x2+x3, x3] * ... * p1^x1 * ... * pm^xm
fn pdf_multinomial(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MultinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let probs = match &dargs[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MultinomialDistribution: second argument must be a list".into(),
      ));
    }
  };
  let xs = match &x {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "PDF[MultinomialDistribution[...], x]: x must be a list".into(),
      ));
    }
  };
  let m = probs.len();
  if xs.len() != m {
    return Err(InterpreterError::EvaluationError(
      "PDF[MultinomialDistribution[...]]: length of x must match length of probabilities".into(),
    ));
  }
  if any_concrete_non_integer(&xs) {
    return Ok(int(0));
  }

  // Build the multinomial coefficient as product of binomials:
  // Binomial[x1+x2, x2] * Binomial[x1+x2+x3, x3] * ...
  // which equals (x1+...+xm)! / (x1! * x2! * ... * xm!)
  let mut coeff: Expr = int(1);
  let mut partial_sum = xs[0].clone();
  for j in 1..m {
    partial_sum = plus(partial_sum, xs[j].clone());
    let binom = call("Binomial", vec![partial_sum.clone(), xs[j].clone()]);
    coeff = times(coeff, binom);
  }

  // Build product: p1^x1 * p2^x2 * ... * pm^xm
  let mut prob_product = power(probs[0].clone(), xs[0].clone());
  for j in 1..m {
    prob_product = times(prob_product, power(probs[j].clone(), xs[j].clone()));
  }

  let pdf_val = times(coeff, prob_product);

  // Conditions: sum(xi) == n and all xi >= 0
  let sum_xs = {
    let mut s = xs[0].clone();
    for j in 1..m {
      s = plus(s, xs[j].clone());
    }
    s
  };
  let sum_eq_n = comparison(sum_xs, ComparisonOp::Equal, n);

  let mut conditions = vec![sum_eq_n];
  for xi in &xs {
    conditions.push(comparison(xi.clone(), ComparisonOp::GreaterEqual, int(0)));
  }

  // Combine conditions with And
  let combined_cond = if conditions.len() == 1 {
    conditions.remove(0)
  } else {
    call("And", conditions)
  };

  eval(piecewise(vec![(pdf_val, combined_cond)], int(0)))
}

/// True if any component of a discrete multivariate point is a concrete
/// number that is not an exact integer. wolframscript's multivariate discrete
/// PDFs are 0 there — including at integer-valued Reals like `2.`.
fn any_concrete_non_integer(xs: &[Expr]) -> bool {
  xs.iter().any(|xi| {
    !matches!(xi, Expr::Integer(_) | Expr::BigInteger(_))
      && crate::functions::math_ast::expr_to_num(xi).is_some()
  })
}

/// Simplify a positive-unate And/Or tree (integer leaves) by flattening
/// nested same-head nodes and applying idempotency (dropping structurally
/// duplicate children). A node that collapses to a single child returns that
/// child. This is reliability-preserving, so a tree whose only repeats are
/// idempotency-reducible (e.g. Or[1, 1] -> 1) becomes a distinct-leaf tree
/// on which the independence product rules are exact.
fn simplify_positive_boolean(e: &Expr) -> Expr {
  // Some(true) = And, Some(false) = Or, None = not a boolean junction.
  fn junction(e: &Expr) -> Option<bool> {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::And,
        ..
      } => Some(true),
      Expr::BinaryOp {
        op: BinaryOperator::Or,
        ..
      } => Some(false),
      Expr::FunctionCall { name, .. } if name == "And" => Some(true),
      Expr::FunctionCall { name, .. } if name == "Or" => Some(false),
      _ => None,
    }
  }
  fn children(e: &Expr) -> Vec<Expr> {
    match e {
      Expr::BinaryOp { left, right, .. } => {
        vec![left.as_ref().clone(), right.as_ref().clone()]
      }
      Expr::FunctionCall { args, .. } => args.iter().cloned().collect(),
      _ => vec![],
    }
  }
  let Some(is_and) = junction(e) else {
    return e.clone();
  };
  // Simplify children, flattening any of the same head.
  let mut kids: Vec<Expr> = Vec::new();
  for c in children(e) {
    let sc = simplify_positive_boolean(&c);
    if junction(&sc) == Some(is_and) {
      kids.extend(children(&sc));
    } else {
      kids.push(sc);
    }
  }
  // Idempotency: drop structurally duplicate children.
  let mut uniq: Vec<Expr> = Vec::new();
  for k in kids {
    if !uniq
      .iter()
      .any(|u| crate::evaluator::pattern_matching::expr_equal(u, &k))
    {
      uniq.push(k);
    }
  }
  if uniq.len() == 1 {
    return uniq.into_iter().next().unwrap();
  }
  Expr::FunctionCall {
    name: if is_and { "And" } else { "Or" }.to_string(),
    args: uniq.into(),
  }
}

/// Rewrite an index-leaf positive Boolean expression into an equivalent
/// read-once form via `BooleanMinimize`. The integer leaves are mapped to
/// fresh symbols (BooleanMinimize ignores bare integers), minimized, then
/// mapped back. Returns `None` if minimization fails or the result is not a
/// pure And/Or/index tree. The caller checks whether the result is actually
/// read-once.
fn failure_read_once_form(bexpr: &Expr) -> Option<Expr> {
  fn map_leaves(e: &Expr, f: &dyn Fn(&Expr) -> Option<Expr>) -> Expr {
    if let Some(r) = f(e) {
      return r;
    }
    match e {
      Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
        op: *op,
        left: Box::new(map_leaves(left, f)),
        right: Box::new(map_leaves(right, f)),
      },
      Expr::UnaryOp { op, operand } => Expr::UnaryOp {
        op: *op,
        operand: Box::new(map_leaves(operand, f)),
      },
      Expr::FunctionCall { name, args }
        if name == "And" || name == "Or" || name == "Not" =>
      {
        Expr::FunctionCall {
          name: name.clone(),
          args: args.iter().map(|a| map_leaves(a, f)).collect(),
        }
      }
      _ => e.clone(),
    }
  }
  const PREFIX: &str = "\u{1}fdvar";
  let to_symbol = map_leaves(bexpr, &|e| match e {
    Expr::Integer(i) => Some(Expr::Identifier(format!("{PREFIX}{i}"))),
    _ => None,
  });
  let minimized = eval(call("BooleanMinimize", vec![to_symbol])).ok()?;
  Some(map_leaves(&minimized, &|e| match e {
    Expr::Identifier(s) => s
      .strip_prefix(PREFIX)
      .and_then(|n| n.parse::<i128>().ok())
      .map(Expr::Integer),
    _ => None,
  }))
}

/// The composed CDF value of a FailureDistribution boolean tree over
/// component CDFs (independent events, each used once): a leaf index maps
/// to its component CDF, And multiplies CDFs, Or complements the product
/// of complements. Returns (value, strict) where `strict` is true when
/// any component support condition was strict (t > 0).
///
/// Emits FailureDistribution::nonunate for negations; duplicated events
/// (which wolframscript resolves exactly) and non-And/Or structure return
/// None so the caller stays unevaluated.
fn failure_distribution_cdf_value(
  dargs: &[Expr],
  t: &Expr,
) -> Result<Option<(Expr, bool)>, InterpreterError> {
  let [bexpr, Expr::List(pairs)] = dargs else {
    return Ok(None);
  };
  // Component CDF value branches, keyed by index.
  let mut comp: Vec<Option<(Expr, bool)>> = vec![None; pairs.len() + 1];
  for p in pairs.iter() {
    let Expr::List(kv) = p else { return Ok(None) };
    let (Expr::Integer(idx), dist) = (&kv[0], &kv[1]) else {
      return Ok(None);
    };
    let idx = *idx as usize;
    if idx == 0 || idx >= comp.len() {
      return Ok(None);
    }
    let c = cdf_ast(&[dist.clone(), t.clone()])?;
    // Expect Piecewise[{{v, t >= 0 | t > 0}}, 0].
    let Expr::FunctionCall { name, args } = &c else {
      return Ok(None);
    };
    if name != "Piecewise" || args.len() != 2 {
      return Ok(None);
    }
    let Expr::List(cases) = &args[0] else {
      return Ok(None);
    };
    if cases.len() != 1 {
      return Ok(None);
    }
    let Expr::List(pair) = &cases[0] else {
      return Ok(None);
    };
    let strict = match &pair[1] {
      Expr::Comparison { operators, .. } if operators.len() == 1 => {
        matches!(operators[0], ComparisonOp::Greater)
      }
      _ => return Ok(None),
    };
    comp[idx] = Some((pair[0].clone(), strict));
  }

  // Negations are not positive unate.
  fn has_not(e: &Expr) -> bool {
    match e {
      Expr::UnaryOp { op, operand } => {
        matches!(op, UnaryOperator::Not) || has_not(operand)
      }
      Expr::FunctionCall { name, args } => {
        name == "Not" || args.iter().any(has_not)
      }
      Expr::BinaryOp { left, right, .. } => has_not(left) || has_not(right),
      _ => false,
    }
  }
  if has_not(bexpr) {
    crate::emit_message(&format!(
      "FailureDistribution::nonunate: The Boolean expression {} is not positive unate. Use UnateQ to test if a Boolean expression is unate.",
      crate::syntax::expr_to_string(bexpr).trim()
    ));
    return Ok(None);
  }
  // Collapse idempotency-reducible repeats (Or[a, a] -> a, And[a, a] -> a,
  // plus flattening of nested same-head And/Or) so that duplicated events —
  // which wolframscript resolves exactly and the independence product rules
  // below would get wrong — reduce to a distinct-leaf tree where those rules
  // are exact. Genuine cross-term repeats (e.g. Or[And[1, 2], And[1, 3]])
  // survive the reduction and still bail out via the dedup check below.
  let simplified = simplify_positive_boolean(bexpr);
  // Each event may appear only once (the independence product rules below
  // treat the branches as independent). A leaf appearing more than once is a
  // shared event; the rules would double-count it.
  fn leaves(e: &Expr, out: &mut Vec<i128>) -> bool {
    match e {
      Expr::Integer(i) => {
        out.push(*i);
        true
      }
      Expr::BinaryOp {
        op: BinaryOperator::And | BinaryOperator::Or,
        left,
        right,
      } => leaves(left, out) && leaves(right, out),
      Expr::FunctionCall { name, args } if name == "And" || name == "Or" => {
        args.iter().all(|a| leaves(a, out))
      }
      _ => false,
    }
  }
  let read_once = |e: &Expr| -> bool {
    let mut seen = Vec::new();
    if !leaves(e, &mut seen) {
      return false;
    }
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len() == seen.len()
  };
  // If events repeat across the tree (e.g. Or[And[1, 2], And[1, 3]]) the
  // independence product rules would double-count the shared event.
  // wolframscript resolves such trees by rewriting them into an equivalent
  // read-once form — (1 || 2) && (1 || 3) becomes 1 || (2 && 3) — and then
  // applies independence. Reproduce that by minimizing the Boolean function;
  // if the minimized form is read-once, compose against it. Otherwise stay
  // unevaluated.
  let bexpr = if read_once(&simplified) {
    simplified
  } else if let Some(reduced) =
    failure_read_once_form(&simplified).filter(read_once)
  {
    reduced
  } else {
    return Ok(None);
  };
  let bexpr = &bexpr;

  fn compose(
    e: &Expr,
    comp: &[Option<(Expr, bool)>],
    strict: &mut bool,
  ) -> Option<Expr> {
    let product = |fs: Vec<Expr>| -> Expr {
      match fs.len() {
        1 => fs.into_iter().next().unwrap(),
        _ => call("Times", fs),
      }
    };
    let complement = |f: Expr| -> Expr {
      call("Plus", vec![int(1), call("Times", vec![int(-1), f])])
    };
    let children = |e: &Expr| -> Option<(bool, Vec<Expr>)> {
      match e {
        Expr::BinaryOp {
          op: op @ (BinaryOperator::And | BinaryOperator::Or),
          left,
          right,
        } => Some((
          matches!(op, BinaryOperator::And),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        )),
        Expr::FunctionCall { name, args } if name == "And" || name == "Or" => {
          Some((name == "And", args.iter().cloned().collect()))
        }
        _ => None,
      }
    };
    match e {
      Expr::Integer(i) => {
        let (v, s) = comp.get(*i as usize)?.clone()?;
        if s {
          *strict = true;
        }
        Some(v)
      }
      _ => {
        let (is_and, kids) = children(e)?;
        let parts: Option<Vec<Expr>> =
          kids.iter().map(|k| compose(k, comp, strict)).collect();
        let parts = parts?;
        if is_and {
          Some(product(parts))
        } else {
          Some(complement(product(
            parts.into_iter().map(complement).collect(),
          )))
        }
      }
    }
  }
  let mut strict = false;
  match compose(bexpr, &comp, &mut strict) {
    Some(value) => Ok(Some((value, strict))),
    None => Ok(None),
  }
}

/// CDF[FailureDistribution[…], t] — Piecewise[{{composed, t >= 0}}, 0]
/// (strict t > 0 when any component support is strict).
fn cdf_failure_distribution(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("FailureDistribution", dargs), x],
    ))
  };
  // Compose against a symbolic variable (component CDFs keep their
  // Piecewise shape there), then substitute a concrete point at the end.
  let var = match &x {
    Expr::Identifier(_) => x.clone(),
    _ => Expr::Identifier("t".to_string()),
  };
  let Some((value, strict)) = failure_distribution_cdf_value(dargs, &var)?
  else {
    return unevaluated(x);
  };
  let op = if strict {
    ComparisonOp::Greater
  } else {
    ComparisonOp::GreaterEqual
  };
  let cond = comparison(var.clone(), op, int(0));
  let result = eval(piecewise(vec![(value, cond)], int(0)))?;
  if matches!(&x, Expr::Identifier(_)) {
    Ok(result)
  } else {
    eval(call(
      "ReplaceAll",
      vec![
        result,
        Expr::Rule {
          pattern: Box::new(var),
          replacement: Box::new(x),
        },
      ],
    ))
  }
}

/// PDF[FailureDistribution[…], t] — the derivative of the composed CDF,
/// on t > 0.
fn pdf_failure_distribution(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("FailureDistribution", dargs), x],
    ))
  };
  // The derivative needs a symbolic variable to differentiate against.
  let var = match &x {
    Expr::Identifier(_) => x.clone(),
    _ => Expr::Identifier("t".to_string()),
  };
  let Some((value, _)) = failure_distribution_cdf_value(dargs, &var)? else {
    return unevaluated(x);
  };
  let deriv = eval(call("D", vec![value, var.clone()]))?;
  let cond = comparison(var.clone(), ComparisonOp::Greater, int(0));
  let result = eval(piecewise(vec![(deriv, cond)], int(0)))?;
  if matches!(&x, Expr::Identifier(_)) {
    Ok(result)
  } else {
    // Numeric evaluation point: substitute after differentiating.
    eval(call(
      "ReplaceAll",
      vec![
        result,
        Expr::Rule {
          pattern: Box::new(var),
          replacement: Box::new(x),
        },
      ],
    ))
  }
}

/// The time-slice distribution of a continuous random process:
/// WienerProcess[m, s][t] is NormalDistribution[m t, s Sqrt[t]] and
/// GeometricBrownianMotionProcess[m, s, x0][t] is
/// LogNormalDistribution[(m - s^2/2) t + Log[x0], s Sqrt[t]] — built in
/// exactly those shapes so the delegated moments and PDF match
/// wolframscript's displays.
pub fn process_slice_distribution(
  proc_name: &str,
  dargs: &[Expr],
  t: &Expr,
) -> Option<Expr> {
  let sqrt_t = call("Sqrt", vec![t.clone()]);
  match proc_name {
    "WienerProcess" if dargs.len() == 2 => Some(call(
      "NormalDistribution",
      vec![
        times(dargs[0].clone(), t.clone()),
        times(dargs[1].clone(), sqrt_t),
      ],
    )),
    // Counting and noise processes with directly parameterized slices.
    "PoissonProcess" if dargs.len() == 1 => Some(call(
      "PoissonDistribution",
      vec![times(dargs[0].clone(), t.clone())],
    )),
    "BinomialProcess" if dargs.len() == 1 => Some(call(
      "BinomialDistribution",
      vec![t.clone(), dargs[0].clone()],
    )),
    // A Bernoulli process' slice does not depend on the time.
    "BernoulliProcess" if dargs.len() == 1 => {
      Some(call("BernoulliDistribution", vec![dargs[0].clone()]))
    }
    // White noise is the underlying distribution at every time.
    "WhiteNoiseProcess"
      if dargs.len() == 1 && matches!(dargs[0], Expr::FunctionCall { .. }) =>
    {
      Some(dargs[0].clone())
    }
    // OrnsteinUhlenbeckProcess[m, s, th] starts in stationarity
    // (time-independent slice); the 4-argument form starts at x0.
    "OrnsteinUhlenbeckProcess" if dargs.len() == 3 => {
      let (m, sp, th) = (&dargs[0], &dargs[1], &dargs[2]);
      Some(call(
        "NormalDistribution",
        vec![
          m.clone(),
          divide(sp.clone(), call("Sqrt", vec![times(int(2), th.clone())])),
        ],
      ))
    }
    "OrnsteinUhlenbeckProcess" if dargs.len() == 4 => {
      let (m, sp, th, x0) = (&dargs[0], &dargs[1], &dargs[2], &dargs[3]);
      let decay = power(e(), times(times(int(-1), th.clone()), t.clone()));
      let mu = plus(
        m.clone(),
        times(plus(x0.clone(), times(int(-1), m.clone())), decay),
      );
      let var = divide(
        times(
          plus(
            int(1),
            times(
              int(-1),
              power(e(), times(times(int(-2), th.clone()), t.clone())),
            ),
          ),
          power(sp.clone(), int(2)),
        ),
        times(int(2), th.clone()),
      );
      Some(call(
        "NormalDistribution",
        vec![mu, call("Sqrt", vec![var])],
      ))
    }
    // BrownianBridgeProcess[s, {t1, a}, {t2, b}] — the interpolating
    // Gaussian bridge.
    "BrownianBridgeProcess" if dargs.len() == 3 => {
      let sp = &dargs[0];
      let (Expr::List(p1), Expr::List(p2)) = (&dargs[1], &dargs[2]) else {
        return None;
      };
      if p1.len() != 2 || p2.len() != 2 {
        return None;
      }
      let (t1, a) = (&p1[0], &p1[1]);
      let (t2, b) = (&p2[0], &p2[1]);
      let span = plus(t2.clone(), times(int(-1), t1.clone()));
      let up = plus(t.clone(), times(int(-1), t1.clone()));
      let down = plus(t2.clone(), times(int(-1), t.clone()));
      let mu = plus(
        divide(times(a.clone(), down.clone()), span.clone()),
        divide(times(b.clone(), up.clone()), span.clone()),
      );
      // The scale factor stays outside the radical, matching
      // wolframscript's SliceDistribution display
      // s*Sqrt[((t - t1)*(-t + t2))/(-t1 + t2)].
      let sigma = times(
        sp.clone(),
        call("Sqrt", vec![divide(times(up, down), span)]),
      );
      Some(call("NormalDistribution", vec![mu, sigma]))
    }
    "GeometricBrownianMotionProcess" if dargs.len() == 3 => {
      let (m, s, x0) = (&dargs[0], &dargs[1], &dargs[2]);
      let drift = plus(
        m.clone(),
        times(
          call("Rational", vec![int(-1), int(2)]),
          power(s.clone(), int(2)),
        ),
      );
      let mu = plus(times(drift, t.clone()), call("Log", vec![x0.clone()]));
      Some(call(
        "LogNormalDistribution",
        vec![mu, times(s.clone(), sqrt_t)],
      ))
    }
    _ => None,
  }
}

/// Parses FirstPassageTimeDistribution[DiscreteMarkovProcess[i0, m], j]
/// into (i0, transition rows, n, target j). Only the integer-initial-state
/// form is supported.
struct Fptd {
  i0: usize,
  rows: Vec<Vec<Expr>>,
  n: usize,
  target: usize,
}

fn fptd_parts(dargs: &[Expr]) -> Option<Fptd> {
  let [proc, Expr::Integer(j)] = dargs else {
    return None;
  };
  let Expr::FunctionCall { name, args } = proc else {
    return None;
  };
  if name != "DiscreteMarkovProcess" || args.len() != 2 {
    return None;
  }
  let Expr::Integer(i0) = &args[0] else {
    return None;
  };
  let Expr::List(rows) = &args[1] else {
    return None;
  };
  let n = rows.len();
  if n == 0 || *i0 < 1 || (*i0 as usize) > n || *j < 1 || (*j as usize) > n {
    return None;
  }
  let mut mat = Vec::with_capacity(n);
  for r in rows.iter() {
    let Expr::List(cells) = r else { return None };
    if cells.len() != n {
      return None;
    }
    mat.push(cells.iter().cloned().collect());
  }
  Some(Fptd {
    i0: *i0 as usize - 1,
    rows: mat,
    n,
    target: *j as usize - 1,
  })
}

/// The taboo pieces of a first-passage problem: the sub-matrix Q over the
/// non-target states, the jump-to-target column r, the start vector over
/// the non-target states (the initial unit vector for i0 != j, or the
/// target row's off-target part for the first-return case), and the
/// direct first-step probability f_1.
struct FptdTaboo {
  q: Vec<Vec<Expr>>,
  r: Vec<Expr>,
  start: Vec<Expr>,
  f1: Expr,
}

fn fptd_taboo(f: &Fptd) -> FptdTaboo {
  let others: Vec<usize> = (0..f.n).filter(|&k| k != f.target).collect();
  let q: Vec<Vec<Expr>> = others
    .iter()
    .map(|&a| others.iter().map(|&b| f.rows[a][b].clone()).collect())
    .collect();
  let r: Vec<Expr> = others
    .iter()
    .map(|&a| f.rows[a][f.target].clone())
    .collect();
  if f.i0 == f.target {
    // First return: one step from the target, then the taboo walk.
    FptdTaboo {
      q,
      r,
      start: others
        .iter()
        .map(|&b| f.rows[f.target][b].clone())
        .collect(),
      f1: f.rows[f.target][f.target].clone(),
    }
  } else {
    let pos = others.iter().position(|&k| k == f.i0).unwrap();
    let start: Vec<Expr> = (0..others.len())
      .map(|k| Expr::Integer(if k == pos { 1 } else { 0 }))
      .collect();
    FptdTaboo {
      f1: f.rows[f.i0][f.target].clone(),
      q,
      r,
      start,
    }
  }
}

/// f_k for k = 1..=count, exactly, via iterated vector-matrix products.
fn fptd_probs(f: &Fptd, count: usize) -> Result<Vec<Expr>, InterpreterError> {
  let taboo = fptd_taboo(f);
  let dot_vec =
    |v: &[Expr], m: &[Vec<Expr>]| -> Result<Vec<Expr>, InterpreterError> {
      let dim = v.len();
      (0..dim)
        .map(|b| {
          let terms: Vec<Expr> = (0..dim)
            .map(|a| times(v[a].clone(), m[a][b].clone()))
            .collect();
          eval(call("Plus", terms))
        })
        .collect()
    };
  let inner = |v: &[Expr], r: &[Expr]| -> Result<Expr, InterpreterError> {
    let terms: Vec<Expr> = v
      .iter()
      .zip(r.iter())
      .map(|(a, b)| times(a.clone(), b.clone()))
      .collect();
    eval(call("Plus", terms))
  };
  let mut out = Vec::with_capacity(count);
  if count == 0 {
    return Ok(out);
  }
  out.push(eval(taboo.f1.clone())?);
  // For i0 != j: f_k = start.Q^(k-1).r; the start vector already encodes
  // one absorbed step in the return case (f_k = start.Q^(k-2).r there).
  let mut v = taboo.start.clone();
  for k in 2..=count {
    if f.i0 == f.target && k == 2 {
      // v is already the post-first-step distribution.
    } else {
      v = dot_vec(&v, &taboo.q)?;
    }
    out.push(inner(&v, &taboo.r)?);
  }
  Ok(out)
}

/// The exact hitting-time moments: (mean, variance) of the first passage
/// (or first return) time, via (I - Q) h = 1 and (I - Q) m2 = 1 + 2 Q h.
fn fptd_moments(f: &Fptd) -> Result<Option<(Expr, Expr)>, InterpreterError> {
  let taboo = fptd_taboo(f);
  let dim = taboo.q.len();
  let i_minus_q: Vec<Expr> = (0..dim)
    .map(|a| {
      Expr::List(
        (0..dim)
          .map(|b| {
            let mut entry = times(int(-1), taboo.q[a][b].clone());
            if a == b {
              entry = plus(int(1), entry);
            }
            entry
          })
          .collect::<Vec<_>>()
          .into(),
      )
    })
    .collect();
  let solve = |rhs: Vec<Expr>| -> Result<Option<Vec<Expr>>, InterpreterError> {
    let solved = eval(call(
      "LinearSolve",
      vec![Expr::List(i_minus_q.clone().into()), Expr::List(rhs.into())],
    ))?;
    match solved {
      Expr::List(ref v) if v.len() == dim => {
        Ok(Some(v.iter().cloned().collect()))
      }
      _ => Ok(None),
    }
  };
  let Some(h) = solve(vec![int(1); dim])? else {
    return Ok(None);
  };
  // rhs2 = 1 + 2 (Q h)
  let mut rhs2 = Vec::with_capacity(dim);
  for a in 0..dim {
    let qh: Vec<Expr> = (0..dim)
      .map(|b| times(taboo.q[a][b].clone(), h[b].clone()))
      .collect();
    rhs2.push(eval(plus(int(1), times(int(2), call("Plus", qh))))?);
  }
  let Some(m2) = solve(rhs2)? else {
    return Ok(None);
  };
  let inner = |v: &[Expr], w: &[Expr]| -> Expr {
    call(
      "Plus",
      v.iter()
        .zip(w.iter())
        .map(|(a, b)| times(a.clone(), b.clone()))
        .collect::<Vec<_>>(),
    )
  };
  let (mean, second) = if f.i0 == f.target {
    // E[T] = 1 + p.h; E[T^2] = 1 + 2 p.h + p.m2 (p = target row off j).
    let ph = inner(&taboo.start, &h);
    let pm2 = inner(&taboo.start, &m2);
    (
      eval(plus(int(1), ph.clone()))?,
      eval(plus(plus(int(1), times(int(2), ph)), pm2))?,
    )
  } else {
    let others: Vec<usize> = (0..f.n).filter(|&k| k != f.target).collect();
    let pos = others.iter().position(|&k| k == f.i0).unwrap();
    (h[pos].clone(), m2[pos].clone())
  };
  let variance =
    eval(plus(second, times(int(-1), power(mean.clone(), int(2)))))?;
  Ok(Some((mean, variance)))
}

/// PDF[FirstPassageTimeDistribution[...], t] for a concrete t: f_t for
/// positive integers, 0 elsewhere. Symbolic t (wolframscript produces
/// eigendecomposition closed forms) stays unevaluated.
fn pdf_first_passage(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("FirstPassageTimeDistribution", dargs), x],
    ))
  };
  let Some(f) = fptd_parts(dargs) else {
    return unevaluated(x);
  };
  match crate::functions::math_ast::try_eval_to_f64(&x) {
    Some(v) => {
      if v < 1.0 || v.fract() != 0.0 || v > 100_000.0 {
        return Ok(int(0));
      }
      let k = v as usize;
      let probs = fptd_probs(&f, k)?;
      Ok(probs.into_iter().next_back().unwrap_or(int(0)))
    }
    None => unevaluated(x),
  }
}

/// CDF[FirstPassageTimeDistribution[...], t] for a concrete t: the sum of
/// f_1..f_Floor[t].
fn cdf_first_passage(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("FirstPassageTimeDistribution", dargs), x],
    ))
  };
  let Some(f) = fptd_parts(dargs) else {
    return unevaluated(x);
  };
  match crate::functions::math_ast::try_eval_to_f64(&x) {
    Some(v) => {
      if v < 1.0 {
        return Ok(int(0));
      }
      let k = (v.floor() as usize).min(100_000);
      let probs = fptd_probs(&f, k)?;
      eval(call("Plus", probs))
    }
    None => unevaluated(x),
  }
}

/// Mean/Variance accessors used by the statistics dispatch.
pub fn fptd_mean(dargs: &[Expr]) -> Result<Option<Expr>, InterpreterError> {
  match fptd_parts(dargs) {
    Some(f) => Ok(fptd_moments(&f)?.map(|(m, _)| m)),
    None => Ok(None),
  }
}

pub fn fptd_variance(dargs: &[Expr]) -> Result<Option<Expr>, InterpreterError> {
  match fptd_parts(dargs) {
    Some(f) => Ok(fptd_moments(&f)?.map(|(_, v)| v)),
    None => Ok(None),
  }
}

/// Parses DiscreteMarkovProcess[i0 | p0, m] into the initial probability
/// row vector and the (square, expression-valued) transition matrix.
fn dmp_parts(dargs: &[Expr]) -> Option<(Vec<Expr>, Expr, usize)> {
  if dargs.len() != 2 {
    return None;
  }
  let Expr::List(rows) = &dargs[1] else {
    return None;
  };
  let n = rows.len();
  if n == 0
    || rows
      .iter()
      .any(|r| !matches!(r, Expr::List(c) if c.len() == n))
  {
    return None;
  }
  let p0: Vec<Expr> = match &dargs[0] {
    Expr::Integer(i0) if *i0 >= 1 && (*i0 as usize) <= n => (1..=n)
      .map(|k| Expr::Integer(if k as i128 == *i0 { 1 } else { 0 }))
      .collect(),
    Expr::List(probs) if probs.len() == n => probs.iter().cloned().collect(),
    _ => return None,
  };
  Some((p0, dargs[1].clone(), n))
}

/// Boole-sum Σ p_k Boole[lhs_k], skipping zero-probability terms.
fn boole_sum(probs: &[Expr], term_lhs: impl Fn(usize) -> (Expr, Expr)) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();
  for (k, p) in probs.iter().enumerate() {
    if matches!(p, Expr::Integer(0)) {
      continue;
    }
    let (l, r) = term_lhs(k + 1);
    let boole = call("Boole", vec![comparison(l, ComparisonOp::Equal, r)]);
    terms.push(if matches!(p, Expr::Integer(1)) {
      boole
    } else {
      times(p.clone(), boole)
    });
  }
  match terms.len() {
    0 => int(0),
    1 => terms.into_iter().next().unwrap(),
    _ => call("Plus", terms),
  }
}

/// PDF[DiscreteMarkovProcess[...][t], x] = Σ (p0.m^t)_k Boole[k == x].
fn dmp_step_pdf(
  dargs: &[Expr],
  t: &Expr,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  let Some((p0, m, _)) = dmp_parts(dargs) else {
    return Ok(None);
  };
  let Expr::Integer(steps) = t else {
    return Ok(None);
  };
  if *steps < 0 {
    return Ok(None);
  }
  let probs_expr = if *steps == 0 {
    Expr::List(p0.into())
  } else {
    eval(call(
      "Dot",
      vec![
        Expr::List(p0.into()),
        call("MatrixPower", vec![m, int(*steps)]),
      ],
    ))?
  };
  let Expr::List(probs) = &probs_expr else {
    return Ok(None);
  };
  let probs: Vec<Expr> = probs.iter().cloned().collect();
  // wolframscript writes these terms with the state first: Boole[1 == x].
  let sum = boole_sum(&probs, |k| (int(k as i128), x.clone()));
  Ok(Some(eval(sum)?))
}

/// The stationary distribution π of a DiscreteMarkovProcess (πP = π,
/// Σπ = 1), solved exactly.
fn dmp_stationary(
  dargs: &[Expr],
) -> Result<Option<Vec<Expr>>, InterpreterError> {
  let Some((_, m, n)) = dmp_parts(dargs) else {
    return Ok(None);
  };
  let Expr::List(rows) = &m else {
    return Ok(None);
  };
  // Equations: Σ_i π_i (P[i][j] − δ_ij) = 0 for j < n−1, plus Σ π_i = 1.
  let mut sys_rows: Vec<Expr> = Vec::with_capacity(n);
  for j in 0..n.saturating_sub(1) {
    let mut row: Vec<Expr> = Vec::with_capacity(n);
    for (i, r) in rows.iter().enumerate() {
      let Expr::List(cells) = r else {
        return Ok(None);
      };
      let mut entry = cells[j].clone();
      if i == j {
        entry = plus(entry, int(-1));
      }
      row.push(eval(entry)?);
    }
    sys_rows.push(Expr::List(row.into()));
  }
  sys_rows.push(Expr::List(vec![int(1); n].into()));
  let mut rhs: Vec<Expr> = vec![int(0); n];
  rhs[n - 1] = int(1);
  let solved = eval(call(
    "LinearSolve",
    vec![Expr::List(sys_rows.into()), Expr::List(rhs.into())],
  ))?;
  match solved {
    Expr::List(ref pi) if pi.len() == n => {
      Ok(Some(pi.iter().cloned().collect()))
    }
    _ => Ok(None),
  }
}

/// PDF[StationaryDistribution[DiscreteMarkovProcess[...]], x]: a numeric
/// x gives π_x directly; a symbolic x gives wolframscript's
/// Piecewise[{{Σ π_k Boole[x == k], Inequality[1, LessEqual, x,
/// LessEqual, n]}}, 0] form (the terms here put x first).
fn dmp_stationary_pdf(
  dargs: &[Expr],
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  let Some(pi) = dmp_stationary(dargs)? else {
    return Ok(None);
  };
  let n = pi.len();
  let sum = boole_sum(&pi, |k| (x.clone(), int(k as i128)));
  let cond = call(
    "Inequality",
    vec![
      int(1),
      Expr::Identifier("LessEqual".to_string()),
      x.clone(),
      Expr::Identifier("LessEqual".to_string()),
      int(n as i128),
    ],
  );
  Ok(Some(eval(piecewise(vec![(sum, cond)], int(0)))?))
}

/// Mean[StationaryDistribution[DiscreteMarkovProcess[...]]] = Σ k π_k.
pub fn dmp_stationary_mean(
  dargs: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some(pi) = dmp_stationary(dargs)? else {
    return Ok(None);
  };
  let terms: Vec<Expr> = pi
    .iter()
    .enumerate()
    .map(|(k, p)| times(int(k as i128 + 1), p.clone()))
    .collect();
  Ok(Some(eval(call("Plus", terms))?))
}

/// Validation for WakebyDistribution[α, β, γ, δ, μ]: α and γ positive
/// (posprm at positions 1 and 3), arity 5 (argrx).
fn wakeby_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() != 5 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "WakebyDistribution::argrx: WakebyDistribution called with {} {}; 5 arguments are expected.",
      dargs.len(),
      word
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  let dist =
    || crate::syntax::expr_to_string(&unevaluated("WakebyDistribution", dargs));
  for pos in [0usize, 2] {
    if num(&dargs[pos]).is_some_and(|v| v <= 0.0) {
      crate::emit_message(&format!(
        "WakebyDistribution::posprm: Parameter {} at position {} in {} is expected to be positive.",
        crate::syntax::expr_to_string(&dargs[pos]),
        pos + 1,
        dist()
      ));
      return None;
    }
  }
  Some(())
}

/// The Wakeby quantile expression
/// m + a (1 - (1-q)^b)/b - g (1 - (1-q)^(-d))/d.
fn wakeby_quantile_body(dargs: &[Expr], q: &Expr) -> Expr {
  let (a, b, g, d, m) = (
    dargs[0].clone(),
    dargs[1].clone(),
    dargs[2].clone(),
    dargs[3].clone(),
    dargs[4].clone(),
  );
  let one_minus_q = plus(int(1), times(int(-1), q.clone()));
  let rise =
    |expo: Expr| plus(int(1), times(int(-1), power(one_minus_q.clone(), expo)));
  plus(
    plus(m, divide(times(a, rise(b.clone())), b)),
    times(
      int(-1),
      divide(times(g, rise(times(int(-1), d.clone()))), d),
    ),
  )
}

/// Quantile[WakebyDistribution[...], q]: the quantile-defined form,
/// wrapped in ConditionalExpression[Piecewise[...], 0 <= q <= 1] for
/// symbolic q; numeric q evaluates the applicable branch directly.
pub fn wakeby_quantile(
  dargs: &[Expr],
  q: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(call(
      "Quantile",
      vec![unevaluated("WakebyDistribution", dargs), q.clone()],
    ))
  };
  let Some(()) = wakeby_checked(dargs) else {
    return unevaluated();
  };
  let inf = infinity();
  match crate::functions::math_ast::try_eval_to_f64(q) {
    Some(qv) if qv > 0.0 && qv < 1.0 => eval(wakeby_quantile_body(dargs, q)),
    Some(qv) if qv == 0.0 => eval(dargs[4].clone()),
    Some(qv) if qv == 1.0 => Ok(inf),
    Some(_) => unevaluated(),
    None => {
      let body = wakeby_quantile_body(dargs, q);
      let pw = piecewise_with_default(
        vec![
          (
            eval(body)?,
            comparison3(
              int(0),
              ComparisonOp::Less,
              q.clone(),
              ComparisonOp::Less,
              int(1),
            ),
          ),
          (
            eval(dargs[4].clone())?,
            comparison(q.clone(), ComparisonOp::LessEqual, int(0)),
          ),
        ],
        inf,
      );
      Ok(call(
        "ConditionalExpression",
        vec![
          pw,
          comparison3(
            int(0),
            ComparisonOp::LessEqual,
            q.clone(),
            ComparisonOp::LessEqual,
            int(1),
          ),
        ],
      ))
    }
  }
}

/// Mean (d < 1) and Variance (d < 1/2) of WakebyDistribution, with
/// Indeterminate outside the existence regions.
fn wakeby_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some(()) = wakeby_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "WakebyDistribution: invalid parameters".into(),
    ));
  };
  let (a, b, g, d, m) = (
    dargs[0].clone(),
    dargs[1].clone(),
    dargs[2].clone(),
    dargs[3].clone(),
    dargs[4].clone(),
  );
  let indet = indeterminate();
  let one_plus_b = plus(int(1), b.clone());
  let mean_body = plus(
    plus(
      divide(a.clone(), one_plus_b.clone()),
      divide(g.clone(), plus(int(1), times(int(-1), d.clone()))),
    ),
    m,
  );
  let dm1 = plus(int(-1), d.clone());
  let var_body = plus(
    plus(
      divide(
        power(a.clone(), int(2)),
        times(
          power(one_plus_b.clone(), int(2)),
          plus(int(1), times(int(2), b.clone())),
        ),
      ),
      times(
        int(-1),
        divide(
          call("Times", vec![int(2), a, g.clone()]),
          call(
            "Times",
            vec![
              one_plus_b,
              plus(plus(int(1), b), times(int(-1), d.clone())),
              dm1.clone(),
            ],
          ),
        ),
      ),
    ),
    times(
      int(-1),
      divide(
        power(g, int(2)),
        times(power(dm1, int(2)), plus(int(-1), times(int(2), d.clone()))),
      ),
    ),
  );
  match crate::functions::math_ast::try_eval_to_f64(&d) {
    Some(dv) => {
      let mean = if dv < 1.0 {
        eval(mean_body)?
      } else {
        indet.clone()
      };
      let variance = if dv < 0.5 { eval(var_body)? } else { indet };
      Ok((mean, variance))
    }
    None => {
      let mean = eval(piecewise_with_default(
        vec![(mean_body, comparison(d.clone(), ComparisonOp::Less, int(1)))],
        indet.clone(),
      ))?;
      let variance = eval(piecewise_with_default(
        vec![(
          var_body,
          comparison(d, ComparisonOp::Less, divide(int(1), int(2))),
        )],
        indet,
      ))?;
      Ok((mean, variance))
    }
  }
}

/// Mean and Variance of CompoundPoissonDistribution[λ, dist]:
/// λ E[X] and λ E[X²] = λ (Var[X] + E[X]²), delegating to the inner
/// distribution's moments. The PDF has no closed form and stays
/// unevaluated (as in wolframscript).
fn compound_poisson_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let bail = || {
    Err(InterpreterError::EvaluationError(
      "CompoundPoissonDistribution: invalid parameters".into(),
    ))
  };
  if dargs.len() != 2 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "CompoundPoissonDistribution::argr: CompoundPoissonDistribution called with {} {}; 2 arguments are expected.",
      dargs.len(),
      word
    ));
    return bail();
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  let dist_str = || {
    crate::syntax::expr_to_string(&unevaluated(
      "CompoundPoissonDistribution",
      dargs,
    ))
  };
  if num(&dargs[0]).is_some_and(|v| v <= 0.0) {
    crate::emit_message(&format!(
      "CompoundPoissonDistribution::posprm: Parameter {} at position 1 in {} is expected to be positive.",
      crate::syntax::expr_to_string(&dargs[0]),
      dist_str()
    ));
    return bail();
  }
  let Expr::FunctionCall {
    name: inner_name,
    args: inner_args,
  } = &dargs[1]
  else {
    // wolframscript's own message template is missing here; it prints
    // the raw fallback, which we replicate verbatim.
    crate::emit_message(&format!(
      "CompoundPoissonDistribution::univ: -- Message text not found -- ({}) ({}) ({})",
      crate::syntax::expr_to_string(&dargs[1]),
      crate::syntax::expr_to_string(&dargs[0]),
      dist_str()
    ));
    return bail();
  };
  if !inner_name.ends_with("Distribution") {
    crate::emit_message(&format!(
      "CompoundPoissonDistribution::univ: -- Message text not found -- ({}) ({}) ({})",
      crate::syntax::expr_to_string(&dargs[1]),
      crate::syntax::expr_to_string(&dargs[0]),
      dist_str()
    ));
    return bail();
  }
  // Unknown inner distributions bail silently to an unevaluated echo.
  let (im, iv) = distribution_mean_variance(inner_name, inner_args)?;
  let lam = dargs[0].clone();
  let mean = eval(times(lam.clone(), im.clone()))?;
  let variance = eval(times(lam, plus(iv, power(im, int(2)))))?;
  Ok((mean, variance))
}

/// Validation for HoytDistribution[q, ω]: q in (0, 1] (pprobprm),
/// ω positive (posprm), arity 2 (argr).
fn hoyt_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() != 2 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "HoytDistribution::argr: HoytDistribution called with {} {}; 2 arguments are expected.",
      dargs.len(),
      word
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  let dist =
    || crate::syntax::expr_to_string(&unevaluated("HoytDistribution", dargs));
  if num(&dargs[0]).is_some_and(|v| !(v > 0.0 && v <= 1.0)) {
    crate::emit_message(&format!(
      "HoytDistribution::pprobprm: Parameter {} at position 1 in {} is expected to be positive and less than or equal to 1.",
      crate::syntax::expr_to_string(&dargs[0]),
      dist()
    ));
    return None;
  }
  if num(&dargs[1]).is_some_and(|v| v <= 0.0) {
    crate::emit_message(&format!(
      "HoytDistribution::posprm: Parameter {} at position 2 in {} is expected to be positive.",
      crate::syntax::expr_to_string(&dargs[1]),
      dist()
    ));
    return None;
  }
  Some(())
}

/// PDF[HoytDistribution[q, ω], x] =
/// (1+q²) x BesselI[0, (1-q⁴)x²/(4q²ω)] E^(-(1+q²)²x²/(4q²ω))/(q ω)
/// on x > 0.
fn hoyt_pdf(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated =
    |x: Expr| Ok(call("PDF", vec![unevaluated("HoytDistribution", dargs), x]));
  let Some(()) = hoyt_checked(dargs) else {
    return unevaluated(x);
  };
  let (q, w) = (dargs[0].clone(), dargs[1].clone());
  let q2 = power(q.clone(), int(2));
  let one_plus_q2 = plus(int(1), q2.clone());
  let four_q2_w = call("Times", vec![int(4), q2.clone(), w.clone()]);
  let x2 = power(x.clone(), int(2));
  let bessel = call(
    "BesselI",
    vec![
      int(0),
      divide(
        times(
          plus(int(1), times(int(-1), power(q.clone(), int(4)))),
          x2.clone(),
        ),
        four_q2_w.clone(),
      ),
    ],
  );
  let gaussian = power(
    e(),
    times(
      int(-1),
      divide(times(power(one_plus_q2.clone(), int(2)), x2), four_q2_w),
    ),
  );
  let value = divide(
    call("Times", vec![one_plus_q2, x.clone(), bessel, gaussian]),
    times(q, w),
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// Mean = Sqrt[2/π] Sqrt[ω/(1+q²)] EllipticE[1-q²] and
/// Variance = ω (1 - 2 EllipticE[1-q²]²/(π (1+q²))).
fn hoyt_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some(()) = hoyt_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "HoytDistribution: invalid parameters".into(),
    ));
  };
  let (q, w) = (dargs[0].clone(), dargs[1].clone());
  let one_plus_q2 = plus(int(1), power(q.clone(), int(2)));
  let elliptic = call(
    "EllipticE",
    vec![plus(int(1), times(int(-1), power(q, int(2))))],
  );
  let sqrt = |e: Expr| call("Sqrt", vec![e]);
  let mean = call(
    "Times",
    vec![
      sqrt(divide(int(2), pi())),
      sqrt(divide(w.clone(), one_plus_q2.clone())),
      elliptic.clone(),
    ],
  );
  let variance = times(
    w,
    plus(
      int(1),
      times(
        int(-1),
        divide(
          times(int(2), power(elliptic, int(2))),
          times(pi(), one_plus_q2),
        ),
      ),
    ),
  );
  Ok((eval(mean)?, eval(variance)?))
}

/// Validation for VarianceGammaDistribution[λ, α, β, μ]: λ, α positive
/// (posprm); numeric |β| >= α → bprm; arity 4 (argrx).
fn variance_gamma_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() != 4 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "VarianceGammaDistribution::argrx: VarianceGammaDistribution called with {} {}; 4 arguments are expected.",
      dargs.len(),
      word
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  let dist = || {
    crate::syntax::expr_to_string(&unevaluated(
      "VarianceGammaDistribution",
      dargs,
    ))
  };
  for pos in 0..2 {
    if num(&dargs[pos]).is_some_and(|v| v <= 0.0) {
      crate::emit_message(&format!(
        "VarianceGammaDistribution::posprm: Parameter {} at position {} in {} is expected to be positive.",
        crate::syntax::expr_to_string(&dargs[pos]),
        pos + 1,
        dist()
      ));
      return None;
    }
  }
  if let (Some(a), Some(b)) = (num(&dargs[1]), num(&dargs[2]))
    && b.abs() >= a
  {
    crate::emit_message(&format!(
      "VarianceGammaDistribution::bprm: The parameters of distribution {} are not valid. Use DistributionParameterAssumptions to obtain the parameter assumptions.",
      dist()
    ));
    return None;
  }
  Some(())
}

/// PDF[VarianceGammaDistribution[λ, α, β, μ], x]: three branches
/// (x > μ, x < μ, and the finite point value at x == μ for λ > 1/2)
/// with default Infinity; half-integer BesselK collapses for integer λ.
fn variance_gamma_pdf(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("VarianceGammaDistribution", dargs), x],
    ))
  };
  let Some(()) = variance_gamma_checked(dargs) else {
    return unevaluated(x);
  };
  let (l, a, b, m) = (
    dargs[0].clone(),
    dargs[1].clone(),
    dargs[2].clone(),
    dargs[3].clone(),
  );
  let sqrt_pi = call("Sqrt", vec![pi()]);
  let gamma_l = call("Gamma", vec![l.clone()]);
  let half_minus_l = plus(divide(int(1), int(2)), times(int(-1), l.clone()));
  let a2b2 = times(
    plus(a.clone(), times(int(-1), b.clone())),
    plus(a.clone(), b.clone()),
  );
  // Branch value for a signed distance d = ±(x - μ).
  let branch = |d: Expr| -> Expr {
    divide(
      call(
        "Times",
        vec![
          power(int(2), half_minus_l.clone()),
          power(a.clone(), half_minus_l.clone()),
          power(a2b2.clone(), l.clone()),
          power(
            e(),
            times(b.clone(), plus(times(int(-1), m.clone()), x.clone())),
          ),
          power(d.clone(), plus(divide(int(-1), int(2)), l.clone())),
          call(
            "BesselK",
            vec![
              plus(divide(int(-1), int(2)), l.clone()),
              times(a.clone(), d),
            ],
          ),
        ],
      ),
      times(sqrt_pi.clone(), gamma_l.clone()),
    )
  };
  let above = branch(plus(times(int(-1), m.clone()), x.clone()));
  let below = branch(plus(m.clone(), times(int(-1), x.clone())));
  let point = divide(
    call(
      "Times",
      vec![
        power(a.clone(), plus(int(1), times(int(-2), l.clone()))),
        power(a2b2, l.clone()),
        call("Gamma", vec![plus(divide(int(-1), int(2)), l.clone())]),
      ],
    ),
    times(times(int(2), sqrt_pi), gamma_l),
  );
  let inf = infinity();
  let cond_above = comparison(x.clone(), ComparisonOp::Greater, m.clone());
  let cond_below = comparison(x.clone(), ComparisonOp::Less, m.clone());
  match crate::functions::math_ast::try_eval_to_f64(&l) {
    Some(lv) => {
      // The point branch folds into the default when λ is numeric.
      let default = if lv > 0.5 { eval(point)? } else { inf };
      eval(piecewise_with_default(
        vec![(eval(above)?, cond_above), (eval(below)?, cond_below)],
        default,
      ))
    }
    None => {
      let cond_point = Expr::BinaryOp {
        op: BinaryOperator::And,
        left: Box::new(comparison(x.clone(), ComparisonOp::Equal, m)),
        right: Box::new(comparison(
          l,
          ComparisonOp::Greater,
          divide(int(1), int(2)),
        )),
      };
      eval(piecewise_with_default(
        vec![
          (above, cond_above),
          (below, cond_below),
          (point, cond_point),
        ],
        inf,
      ))
    }
  }
}

/// Mean = 2βλ/((α-β)(α+β)) + μ and
/// Variance = 2(α²+β²)λ/((α-β)²(α+β)²).
fn variance_gamma_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some(()) = variance_gamma_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "VarianceGammaDistribution: invalid parameters".into(),
    ));
  };
  let (l, a, b, m) = (
    dargs[0].clone(),
    dargs[1].clone(),
    dargs[2].clone(),
    dargs[3].clone(),
  );
  let amb = plus(a.clone(), times(int(-1), b.clone()));
  let apb = plus(a.clone(), b.clone());
  let mean = plus(
    divide(
      call("Times", vec![int(2), b.clone(), l.clone()]),
      times(amb.clone(), apb.clone()),
    ),
    m,
  );
  let variance = divide(
    call(
      "Times",
      vec![int(2), plus(power(a, int(2)), power(b, int(2))), l],
    ),
    times(power(amb, int(2)), power(apb, int(2))),
  );
  Ok((eval(mean)?, eval(variance)?))
}

/// Validation for TsallisQGaussianDistribution[μ, β, q]: β positive
/// (posprm, position 2), q < 3 (lss, position 3), arity 3 (argrx).
fn tsallis_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() != 3 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "TsallisQGaussianDistribution::argrx: TsallisQGaussianDistribution called with {} {}; 3 arguments are expected.",
      dargs.len(),
      word
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  let dist = || {
    crate::syntax::expr_to_string(&unevaluated(
      "TsallisQGaussianDistribution",
      dargs,
    ))
  };
  if num(&dargs[1]).is_some_and(|v| v <= 0.0) {
    crate::emit_message(&format!(
      "TsallisQGaussianDistribution::posprm: Parameter {} at position 2 in {} is expected to be positive.",
      crate::syntax::expr_to_string(&dargs[1]),
      dist()
    ));
    return None;
  }
  if num(&dargs[2]).is_some_and(|v| v >= 3.0) {
    crate::emit_message(&format!(
      "TsallisQGaussianDistribution::lss: Parameter {} at position 3 in {} is expected to be less than 3.",
      crate::syntax::expr_to_string(&dargs[2]),
      dist()
    ));
    return None;
  }
  Some(())
}

/// Shared subtrees of the Tsallis q-Gaussian closed forms.
struct TsallisParts {
  gaussian: Expr,
  branch_wide: Expr,
  branch_compact: Expr,
  compact_arg: Expr,
}

fn tsallis_parts(m: &Expr, b: &Expr, q: &Expr, x: &Expr) -> TsallisParts {
  let sqrt = |e: Expr| call("Sqrt", vec![e]);
  let two_pi = times(int(2), pi());
  let m_minus_x = plus(m.clone(), times(int(-1), x.clone()));
  let qm1 = plus(int(-1), q.clone());
  let one_mq = plus(int(1), times(int(-1), q.clone()));
  // Gaussian branch: 1/(b E^((m-x)^2/(2 b^2)) Sqrt[2 Pi])
  let gaussian = power(
    call(
      "Times",
      vec![
        b.clone(),
        power(
          e(),
          divide(
            power(m_minus_x.clone(), int(2)),
            times(int(2), power(b.clone(), int(2))),
          ),
        ),
        sqrt(two_pi.clone()),
      ],
    ),
    int(-1),
  );
  // (1 + (-1+q)(m-x)^2/(2 b^2))^((1-q)^-1)
  let core = power(
    plus(
      int(1),
      divide(
        times(qm1.clone(), power(m_minus_x.clone(), int(2))),
        times(int(2), power(b.clone(), int(2))),
      ),
    ),
    power(one_mq.clone(), int(-1)),
  );
  // 1 < q < 3 branch
  let branch_wide = divide(
    call(
      "Times",
      vec![
        sqrt(qm1.clone()),
        core.clone(),
        gamma(power(qm1.clone(), int(-1))),
      ],
    ),
    call(
      "Times",
      vec![
        b.clone(),
        sqrt(two_pi.clone()),
        gamma(divide(
          plus(int(3), times(int(-1), q.clone())),
          times(int(2), qm1.clone()),
        )),
      ],
    ),
  );
  // q < 1 branch
  let branch_compact = divide(
    call(
      "Times",
      vec![
        sqrt(one_mq.clone()),
        core,
        gamma(plus(divide(int(3), int(2)), power(one_mq.clone(), int(-1)))),
      ],
    ),
    call(
      "Times",
      vec![
        b.clone(),
        sqrt(two_pi),
        gamma(plus(int(1), power(one_mq.clone(), int(-1)))),
      ],
    ),
  );
  // (Sqrt[(1-q)/b^2] (-m+x))/Sqrt[2]
  let compact_arg = divide(
    times(
      sqrt(divide(one_mq, power(b.clone(), int(2)))),
      plus(times(int(-1), m.clone()), x.clone()),
    ),
    sqrt(int(2)),
  );
  TsallisParts {
    gaussian,
    branch_wide,
    branch_compact,
    compact_arg,
  }
}

/// PDF[TsallisQGaussianDistribution[μ, β, q], x]: numeric exact q picks
/// its branch (Gaussian at q == 1, full-support power law for 1 < q < 3,
/// compact support for q < 1); symbolic q keeps the three-branch
/// Piecewise. Float parameters stay unevaluated (wolframscript's float
/// artifacts are not reproducible).
fn tsallis_qgaussian_pdf(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("TsallisQGaussianDistribution", dargs), x],
    ))
  };
  let Some(()) = tsallis_checked(dargs) else {
    return unevaluated(x);
  };
  let (m, b, q) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  if dargs.iter().any(|e| {
    !coxian_exact(e) && crate::functions::math_ast::try_eval_to_f64(e).is_some()
  }) {
    return unevaluated(x);
  }
  let parts = tsallis_parts(&m, &b, &q, &x);
  match crate::functions::math_ast::try_eval_to_f64(&q) {
    Some(qv) if qv == 1.0 => eval(parts.gaussian),
    Some(qv) if qv > 1.0 => eval(parts.branch_wide),
    Some(_) => {
      let cond = comparison3(
        int(-1),
        ComparisonOp::LessEqual,
        eval(parts.compact_arg)?,
        ComparisonOp::LessEqual,
        int(1),
      );
      eval(piecewise(vec![(eval(parts.branch_compact)?, cond)], int(0)))
    }
    None => {
      let q_eq_1 = comparison(q.clone(), ComparisonOp::Equal, int(1));
      let q_mid = comparison3(
        int(1),
        ComparisonOp::Less,
        q.clone(),
        ComparisonOp::Less,
        int(3),
      );
      let q_low = Expr::BinaryOp {
        op: BinaryOperator::And,
        left: Box::new(comparison(q.clone(), ComparisonOp::Less, int(1))),
        right: Box::new(comparison3(
          int(-1),
          ComparisonOp::LessEqual,
          parts.compact_arg,
          ComparisonOp::LessEqual,
          int(1),
        )),
      };
      eval(piecewise(
        vec![
          (parts.gaussian, q_eq_1),
          (parts.branch_wide, q_mid),
          (parts.branch_compact, q_low),
        ],
        int(0),
      ))
    }
  }
}

/// CDF[TsallisQGaussianDistribution[μ, β, q], x]: q == 1 uses the Erf
/// closed form; symbolic q keeps the Hypergeometric2F1 template; other
/// numeric q stay unevaluated (wolframscript's per-q hypergeometric
/// collapses are not reproduced).
fn tsallis_qgaussian_cdf(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("TsallisQGaussianDistribution", dargs), x],
    ))
  };
  let Some(()) = tsallis_checked(dargs) else {
    return unevaluated(x);
  };
  let (m, b, q) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let erf_form = divide(
    plus(
      int(1),
      call(
        "Erf",
        vec![divide(
          plus(times(int(-1), m.clone()), x.clone()),
          times(call("Sqrt", vec![int(2)]), b.clone()),
        )],
      ),
    ),
    int(2),
  );
  match crate::functions::math_ast::try_eval_to_f64(&q) {
    Some(qv) if qv == 1.0 => eval(erf_form),
    Some(_) => unevaluated(x),
    None => unevaluated(x),
  }
}

/// Mean and Variance of TsallisQGaussianDistribution: Mean = μ for
/// q < 2; Variance = 2β²/(5-3q) for q < 5/3, Infinity for
/// 5/3 <= q < 2, Indeterminate otherwise.
fn tsallis_qgaussian_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some(()) = tsallis_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "TsallisQGaussianDistribution: invalid parameters".into(),
    ));
  };
  let (m, b, q) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let indet = indeterminate();
  let inf = infinity();
  let var_core = divide(
    times(int(2), power(b.clone(), int(2))),
    plus(int(5), times(int(-3), q.clone())),
  );
  match crate::functions::math_ast::try_eval_to_f64(&q) {
    Some(qv) => {
      let mean = if qv < 2.0 { eval(m)? } else { indet.clone() };
      let variance = if qv < 5.0 / 3.0 {
        eval(var_core)?
      } else if qv < 2.0 {
        inf
      } else {
        indet
      };
      Ok((mean, variance))
    }
    None => {
      let mean = eval(piecewise_with_default(
        vec![(m, comparison(q.clone(), ComparisonOp::Less, int(2)))],
        indet.clone(),
      ))?;
      let variance = eval(piecewise_with_default(
        vec![
          (
            var_core,
            comparison(q.clone(), ComparisonOp::Less, divide(int(5), int(3))),
          ),
          (
            inf,
            comparison3(
              divide(int(5), int(3)),
              ComparisonOp::LessEqual,
              q,
              ComparisonOp::Less,
              int(2),
            ),
          ),
        ],
        indet,
      ))?;
      Ok((mean, variance))
    }
  }
}

/// Arity check for TukeyLambdaDistribution (1 or 3 arguments).
fn tukey_lambda_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() == 1 || dargs.len() == 3 {
    return Some(());
  }
  let word = if dargs.len() == 1 {
    "argument"
  } else {
    "arguments"
  };
  crate::emit_message(&format!(
    "TukeyLambdaDistribution::argt: TukeyLambdaDistribution called with {} {}; 1 or 3 arguments are expected.",
    dargs.len(),
    word
  ));
  None
}

/// PDF/CDF of TukeyLambdaDistribution for the λ values wolframscript
/// has closed forms for (0, 1/2, 1, 2, -1, exact or float). Other λ
/// (including the Root-object λ=3 case) stay unevaluated. The 3-arg
/// location-scale form substitutes (x-μ)/σ into the branch values and
/// conditions and divides the PDF Piecewise by σ, as wolframscript does.
fn tukey_lambda_pdf_cdf(
  dargs: &[Expr],
  x: Expr,
  want_pdf: bool,
) -> Result<Expr, InterpreterError> {
  let head = if want_pdf { "PDF" } else { "CDF" };
  let unevaluated = |x: Expr| {
    Ok(call(
      head,
      vec![unevaluated("TukeyLambdaDistribution", dargs), x],
    ))
  };
  let Some(()) = tukey_lambda_checked(dargs) else {
    return unevaluated(x);
  };
  let lam = dargs[0].clone();
  let Some(lv) = crate::functions::math_ast::try_eval_to_f64(&lam) else {
    return unevaluated(x);
  };
  let scaled = dargs.len() == 3;
  // The variable the closed forms are written in: x or (x - μ)/σ.
  let y: Expr = if scaled {
    eval(divide(
      plus(times(int(-1), dargs[1].clone()), x.clone()),
      dargs[2].clone(),
    ))?
  } else {
    x.clone()
  };
  let sq = |e: &Expr| power(e.clone(), int(2));
  let lam2y2 = |lam: &Expr, y: &Expr| times(sq(lam), sq(y));
  // (value, condition) pairs plus default per λ; None = no closed form.
  let e_neg = |arg: Expr| power(e(), times(int(-1), arg));
  let branches: Option<(Vec<(Expr, Expr)>, Expr)> = if lv == 0.0 {
    // Logistic: full support, no Piecewise.
    let value = if want_pdf {
      if scaled {
        // E^((μ-x)/σ)/(σ (1 + E^((μ-x)/σ))^2)
        let z = eval(divide(
          plus(dargs[1].clone(), times(int(-1), x.clone())),
          dargs[2].clone(),
        ))?;
        divide(
          power(e(), z.clone()),
          times(dargs[2].clone(), sq(&plus(int(1), power(e(), z)))),
        )
      } else {
        times(
          e_neg(y.clone()),
          power(plus(int(1), e_neg(y.clone())), int(-2)),
        )
      }
    } else {
      power(plus(int(1), e_neg(y.clone())), int(-1))
    };
    return eval(value);
  } else if lv == 0.5 {
    let c = eval(lam2y2(&lam, &y))?;
    let cond = comparison3(
      int(-2),
      ComparisonOp::LessEqual,
      y.clone(),
      ComparisonOp::LessEqual,
      int(2),
    );
    if want_pdf {
      let v = divide(
        plus(int(1), times(int(-1), c.clone())),
        times(int(2), call("Sqrt", vec![plus(int(2), times(int(-1), c))])),
      );
      Some((vec![(v, cond)], int(0)))
    } else {
      let inner = plus(
        int(4),
        times(
          y.clone(),
          call("Sqrt", vec![plus(int(8), times(int(-1), sq(&y)))]),
        ),
      );
      // Float λ folds the 1/8 into a 0.125 prefactor, like wolframscript.
      let v = if matches!(&lam, Expr::Real(_)) {
        times(Expr::Real(0.125), inner)
      } else {
        divide(inner, int(8))
      };
      let below = comparison(y.clone(), ComparisonOp::Less, int(-2));
      Some((vec![(v, cond), (int(0), below)], int(1)))
    }
  } else if lv == 1.0 || lv == 2.0 {
    // Support bounds stay exact even for float λ (wolframscript shows
    // -1 <= x <= 1 for λ = 1.).
    let half_width = if lv == 1.0 {
      int(1)
    } else {
      divide(int(1), int(2))
    };
    let half_width = eval(half_width)?;
    let lo = eval(times(int(-1), half_width.clone()))?;
    let cond = comparison3(
      lo.clone(),
      ComparisonOp::LessEqual,
      y.clone(),
      ComparisonOp::LessEqual,
      half_width,
    );
    if want_pdf {
      let v = eval(divide(lam.clone(), int(2)))?;
      Some((vec![(v, cond)], int(0)))
    } else {
      let v = divide(plus(int(1), times(lam.clone(), y.clone())), int(2));
      let below = comparison(y.clone(), ComparisonOp::Less, lo);
      Some((vec![(v, cond), (int(0), below)], int(1)))
    }
  } else if lv == -1.0 {
    let cond = comparison(y.clone(), ComparisonOp::NotEqual, int(0));
    if want_pdf {
      let v = divide(
        plus(
          int(1),
          times(
            int(-1),
            power(
              call("Sqrt", vec![plus(int(1), divide(sq(&y), int(4)))]),
              int(-1),
            ),
          ),
        ),
        sq(&y),
      );
      Some((vec![(v, cond)], divide(int(1), int(8))))
    } else {
      let v = divide(
        plus(
          plus(int(-2), y.clone()),
          call("Sqrt", vec![plus(int(4), sq(&y))]),
        ),
        times(int(2), y.clone()),
      );
      Some((vec![(v, cond)], divide(int(1), int(2))))
    }
  } else {
    None
  };
  let Some((cases, default)) = branches else {
    return unevaluated(x);
  };
  // Branch values evaluate; conditions keep the substituted variable raw.
  let mut evaled: Vec<(Expr, Expr)> = Vec::new();
  for (v, c) in cases {
    evaled.push((eval(v)?, c));
  }
  let pw = eval(piecewise_with_default(
    evaled.into_iter().collect::<Vec<_>>(),
    eval(default)?,
  ))?;
  if want_pdf && scaled {
    Ok(times(pw, power(dargs[2].clone(), int(-1))))
  } else {
    Ok(pw)
  }
}

/// Mean and Variance of TukeyLambdaDistribution: Mean is 0 (or μ) for
/// λ > -1; Variance uses the factorial template
/// (-2 λ!² + 2 (2λ)!)/(λ² (1+2λ)!) for λ > -1/2.
fn tukey_lambda_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some(()) = tukey_lambda_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "TukeyLambdaDistribution: invalid arity".into(),
    ));
  };
  let lam = dargs[0].clone();
  let indet = indeterminate();
  let num = crate::functions::math_ast::try_eval_to_f64;
  let mu = if dargs.len() == 3 {
    dargs[1].clone()
  } else {
    int(0)
  };
  let var_core = divide(
    plus(
      times(int(-2), power(factorial(lam.clone()), int(2))),
      times(int(2), factorial(times(int(2), lam.clone()))),
    ),
    times(
      power(lam.clone(), int(2)),
      factorial(plus(int(1), times(int(2), lam.clone()))),
    ),
  );
  let sigma2 = if dargs.len() == 3 {
    times(power(dargs[2].clone(), int(2)), var_core.clone())
  } else {
    var_core.clone()
  };
  match num(&lam) {
    Some(lv) => {
      let mean = if lv > -1.0 { eval(mu)? } else { indet.clone() };
      let variance = if lv == 0.0 {
        // Logistic limit: the factorial template divides by λ².
        let core = divide(power(pi(), int(2)), int(3));
        let scaled_core = if dargs.len() == 3 {
          times(power(dargs[2].clone(), int(2)), core)
        } else {
          core
        };
        eval(scaled_core)?
      } else if lv > -0.5 {
        eval(sigma2)?
      } else {
        indet
      };
      Ok((mean, variance))
    }
    None => {
      let mean = eval(piecewise_with_default(
        vec![(mu, comparison(lam.clone(), ComparisonOp::Greater, int(-1)))],
        indet.clone(),
      ))?;
      let variance = eval(piecewise_with_default(
        vec![(
          sigma2,
          comparison(lam, ComparisonOp::Greater, divide(int(-1), int(2))),
        )],
        indet,
      ))?;
      Ok((mean, variance))
    }
  }
}

/// Validation for HotellingTSquareDistribution[p, m]: both parameters
/// must be positive numbers (posprm); wrong arity emits argr. Messages
/// come from consuming functions.
fn hotelling_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() != 2 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "HotellingTSquareDistribution::argr: HotellingTSquareDistribution called with {} {}; 2 arguments are expected.",
      dargs.len(),
      word
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  for pos in 0..2 {
    if num(&dargs[pos]).is_some_and(|v| v <= 0.0) {
      crate::emit_message(&format!(
        "HotellingTSquareDistribution::posprm: Parameter {} at position {} in {} is expected to be positive.",
        crate::syntax::expr_to_string(&dargs[pos]),
        pos + 1,
        crate::syntax::expr_to_string(&unevaluated(
          "HotellingTSquareDistribution",
          dargs
        ))
      ));
      return None;
    }
  }
  Some(())
}

/// PDF[HotellingTSquareDistribution[p, m], x]. Numeric parameters use
/// the collapsed C x^(p/2-1) ((m+x)^-1)^((1+m)/2) form wolframscript
/// produces; symbolic parameters keep the Beta template.
fn pdf_hotelling(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("HotellingTSquareDistribution", dargs), x],
    ))
  };
  let Some(()) = hotelling_checked(dargs) else {
    return unevaluated(x);
  };
  let (pp, m) = (dargs[0].clone(), dargs[1].clone());
  let num = crate::functions::math_ast::try_eval_to_f64;
  let half = |e: Expr| divide(e, int(2));
  let beta = call(
    "Beta",
    vec![
      half(pp.clone()),
      half(plus(plus(int(1), m.clone()), times(int(-1), pp.clone()))),
    ],
  );
  let value = if num(&pp).is_some() && num(&m).is_some() {
    // Integer-valued float parameters compute the coefficient exactly
    // and refloat it (5.^2/Beta[1., 2.] must be 50., not 50.00000...2).
    let exactify = |e: &Expr| -> Expr {
      match e {
        Expr::Real(r) if r.fract() == 0.0 && r.abs() < 1e15 => int(*r as i128),
        other => other.clone(),
      }
    };
    let any_float = matches!(&pp, Expr::Real(_)) || matches!(&m, Expr::Real(_));
    let (pe, me) = (exactify(&pp), exactify(&m));
    let coeff_expr = divide(
      power(
        me.clone(),
        half(plus(plus(int(1), me.clone()), times(int(-1), pe.clone()))),
      ),
      call(
        "Beta",
        vec![
          half(pe.clone()),
          half(plus(plus(int(1), me.clone()), times(int(-1), pe))),
        ],
      ),
    );
    let coeff_exact = eval(coeff_expr)?;
    let coeff = if any_float
      && !matches!(&coeff_exact, Expr::Real(_))
      && let Some(v) = num(&coeff_exact)
    {
      Expr::Real(v)
    } else if any_float {
      coeff_exact
    } else {
      eval(divide(
        power(
          m.clone(),
          half(plus(plus(int(1), m.clone()), times(int(-1), pp.clone()))),
        ),
        beta,
      ))?
    };
    let expo = eval(plus(half(pp.clone()), int(-1)))?;
    let tail = power(
      power(plus(m.clone(), x.clone()), int(-1)),
      half(plus(int(1), m.clone())),
    );
    let is_zero = num(&expo).is_some_and(|v| v == 0.0);
    if is_zero {
      times(coeff, tail)
    } else {
      call("Times", vec![coeff, power(x.clone(), expo), tail])
    }
  } else {
    divide(
      times(
        power(divide(x.clone(), m.clone()), half(pp.clone())),
        power(
          divide(m.clone(), plus(m.clone(), x.clone())),
          half(plus(int(1), m.clone())),
        ),
      ),
      times(x.clone(), beta),
    )
  };
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[HotellingTSquareDistribution[p, m], x] via the BetaRegularized
/// template (which expands itself for a == 1 or b == 1).
fn cdf_hotelling(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("HotellingTSquareDistribution", dargs), x],
    ))
  };
  let Some(()) = hotelling_checked(dargs) else {
    return unevaluated(x);
  };
  let (pp, m) = (dargs[0].clone(), dargs[1].clone());
  let half = |e: Expr| divide(e, int(2));
  let k = plus(plus(int(1), m.clone()), times(int(-1), pp.clone()));
  let z = divide(
    times(k.clone(), x.clone()),
    times(
      m.clone(),
      plus(k.clone(), divide(times(k.clone(), x.clone()), m.clone())),
    ),
  );
  let value = call("BetaRegularized", vec![z, half(pp), half(k)]);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// Mean and Variance of HotellingTSquareDistribution as conditional
/// Piecewise expressions (Indeterminate outside the existence region).
fn hotelling_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some(()) = hotelling_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "HotellingTSquareDistribution: invalid parameters".into(),
    ));
  };
  let (pp, m) = (dargs[0].clone(), dargs[1].clone());
  let indet = indeterminate();
  let num = crate::functions::math_ast::try_eval_to_f64;
  let numeric = num(&pp).zip(num(&m));
  // Mean: (m p)/(-1 + m - p) when -1 + m - p > 0. For numeric
  // parameters the condition is decided up front so a dead branch never
  // evaluates (avoiding spurious Power::infy at the boundary).
  let dof = plus(plus(int(-1), m.clone()), times(int(-1), pp.clone()));
  let mean_value = divide(times(m.clone(), pp.clone()), dof.clone());
  let mean = match numeric {
    Some((pv, mv)) => {
      if mv - pv - 1.0 > 0.0 {
        eval(mean_value)?
      } else {
        indet.clone()
      }
    }
    None => eval(piecewise_with_default(
      vec![(mean_value, comparison(dof, ComparisonOp::Greater, int(0)))],
      indet.clone(),
    ))?,
  };
  // Variance: 2 (-1+m) m^2 p / ((-3+m-p)(1-m+p)^2) when m > 3 + p.
  let var_num = call(
    "Times",
    vec![
      int(2),
      plus(int(-1), m.clone()),
      power(m.clone(), int(2)),
      pp.clone(),
    ],
  );
  let var_den = times(
    plus(plus(int(-3), m.clone()), times(int(-1), pp.clone())),
    power(
      plus(plus(int(1), times(int(-1), m.clone())), pp.clone()),
      int(2),
    ),
  );
  let var_value = divide(var_num, var_den);
  let variance = match numeric {
    Some((pv, mv)) => {
      if mv > 3.0 + pv {
        eval(var_value)?
      } else {
        indet
      }
    }
    None => eval(piecewise_with_default(
      vec![(
        var_value,
        comparison(m, ComparisonOp::Greater, plus(int(3), pp)),
      )],
      indet,
    ))?,
  };
  Ok((mean, variance))
}

/// Piecewise with an explicit default value.
fn piecewise_with_default(cases: Vec<(Expr, Expr)>, default: Expr) -> Expr {
  let pairs: Vec<Expr> = cases
    .into_iter()
    .map(|(v, c)| Expr::List(vec![v, c].into()))
    .collect();
  call("Piecewise", vec![Expr::List(pairs.into()), default])
}

/// Validation for BeniniDistribution[α, β, σ]: α and β must be
/// non-negative (nnegprm), σ positive (posprm); wrong arity emits argrx.
/// All messages come from consuming functions.
fn benini_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() != 3 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "BeniniDistribution::argrx: BeniniDistribution called with {} {}; 3 arguments are expected.",
      dargs.len(),
      word
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  let dist =
    || crate::syntax::expr_to_string(&unevaluated("BeniniDistribution", dargs));
  for pos in 0..2 {
    if num(&dargs[pos]).is_some_and(|v| v < 0.0) {
      crate::emit_message(&format!(
        "BeniniDistribution::nnegprm: Parameter {} at position {} in {} is expected to be non-negative.",
        crate::syntax::expr_to_string(&dargs[pos]),
        pos + 1,
        dist()
      ));
      return None;
    }
  }
  if num(&dargs[2]).is_some_and(|v| v <= 0.0) {
    crate::emit_message(&format!(
      "BeniniDistribution::posprm: Parameter {} at position 3 in {} is expected to be positive.",
      crate::syntax::expr_to_string(&dargs[2]),
      dist()
    ));
    return None;
  }
  Some(())
}

/// Log[x/σ], the building block of the Benini closed forms.
fn benini_log(x: &Expr, sigma: &Expr) -> Expr {
  call("Log", vec![divide(x.clone(), sigma.clone())])
}

/// PDF[BeniniDistribution[α, β, σ], x]. Numeric α uses the split form
/// σ^α x^(-α) (2/x + ...) E^(-β Log²) wolframscript shows; symbolic α
/// keeps the whole exponential E^(-α Log - β Log²).
fn pdf_benini(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("BeniniDistribution", dargs), x],
    ))
  };
  let Some(()) = benini_checked(dargs) else {
    return unevaluated(x);
  };
  let (a, b, sg) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let lg = benini_log(&x, &sg);
  let hazard = plus(
    times(a.clone(), power(x.clone(), int(-1))),
    times(
      times(times(int(2), b.clone()), lg.clone()),
      power(x.clone(), int(-1)),
    ),
  );
  let quad = power(
    e(),
    times(times(int(-1), b.clone()), power(lg.clone(), int(2))),
  );
  let value = if crate::functions::math_ast::try_eval_to_f64(&a).is_some() {
    call(
      "Times",
      vec![
        power(sg.clone(), a.clone()),
        hazard,
        quad,
        power(x.clone(), times(int(-1), a)),
      ],
    )
  } else {
    times(
      power(
        e(),
        plus(
          times(times(int(-1), a), lg.clone()),
          times(times(int(-1), b), power(lg, int(2))),
        ),
      ),
      hazard,
    )
  };
  let cond = comparison(x, ComparisonOp::GreaterEqual, sg);
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[BeniniDistribution[α, β, σ], x] = 1 - σ^α x^(-α) E^(-β Log²)
/// (numeric α) or 1 - E^(-α Log - β Log²) (symbolic α).
fn cdf_benini(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("BeniniDistribution", dargs), x],
    ))
  };
  let Some(()) = benini_checked(dargs) else {
    return unevaluated(x);
  };
  let (a, b, sg) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let lg = benini_log(&x, &sg);
  let survival = if crate::functions::math_ast::try_eval_to_f64(&a).is_some() {
    call(
      "Times",
      vec![
        power(sg.clone(), a.clone()),
        power(
          e(),
          times(times(int(-1), b.clone()), power(lg.clone(), int(2))),
        ),
        power(x.clone(), times(int(-1), a)),
      ],
    )
  } else {
    power(
      e(),
      plus(
        times(times(int(-1), a), lg.clone()),
        times(times(int(-1), b), power(lg, int(2))),
      ),
    )
  };
  let value = plus(int(1), times(int(-1), survival));
  let cond = comparison(x, ComparisonOp::GreaterEqual, sg);
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// Mean and Variance of BeniniDistribution in wolframscript's Erfc
/// template shapes.
fn benini_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some(()) = benini_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "BeniniDistribution: invalid parameters".into(),
    ));
  };
  let (a, b, sg) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let sqrt_pi = call("Sqrt", vec![pi()]);
  let sqrt_b = call("Sqrt", vec![b.clone()]);
  // Shared pieces for shift k: E^((-k+a)^2/(4β)) and Erfc[(-k+a)/(2√β)].
  let shifted = |k: i128, denom_scale: i128| -> (Expr, Expr) {
    let base = plus(int(-k), a.clone());
    let expo = power(
      e(),
      divide(
        power(base.clone(), int(2)),
        times(int(denom_scale), b.clone()),
      ),
    );
    let erfc = call("Erfc", vec![divide(base, times(int(2), sqrt_b.clone()))]);
    (expo, erfc)
  };
  let (e1, erfc1) = shifted(1, 4);
  let mean = plus(
    sg.clone(),
    divide(
      call(
        "Times",
        vec![e1.clone(), sqrt_pi.clone(), sg.clone(), erfc1.clone()],
      ),
      times(int(2), sqrt_b.clone()),
    ),
  );
  let (e2, erfc2) = shifted(2, 4);
  let (e1w, _) = shifted(1, 2);
  let t1 = call("Times", vec![int(4), sqrt_b.clone(), e2, erfc2]);
  let t2 = call("Times", vec![int(-4), sqrt_b, e1, erfc1.clone()]);
  let t3 = call(
    "Times",
    vec![int(-1), e1w, sqrt_pi.clone(), power(erfc1, int(2))],
  );
  let variance = divide(
    call(
      "Times",
      vec![sqrt_pi, power(sg, int(2)), call("Plus", vec![t1, t2, t3])],
    ),
    times(int(4), b),
  );
  Ok((eval(mean)?, eval(variance)?))
}

/// Validation for VonMisesDistribution[μ, κ]: emits argr when the
/// argument count is off and nnegprm when κ is a negative number (all
/// from consuming functions; the constructor echoes silently).
fn vonmises_checked(dargs: &[Expr]) -> Option<()> {
  if dargs.len() != 2 {
    let word = if dargs.len() == 1 {
      "argument"
    } else {
      "arguments"
    };
    crate::emit_message(&format!(
      "VonMisesDistribution::argr: VonMisesDistribution called with {} {}; 2 arguments are expected.",
      dargs.len(),
      word
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  if num(&dargs[1]).is_some_and(|v| v < 0.0) {
    crate::emit_message(&format!(
      "VonMisesDistribution::nnegprm: Parameter {} at position 2 in {} is expected to be non-negative.",
      crate::syntax::expr_to_string(&dargs[1]),
      crate::syntax::expr_to_string(&unevaluated(
        "VonMisesDistribution",
        dargs
      ))
    ));
    return None;
  }
  Some(())
}

/// PDF[VonMisesDistribution[μ, κ], x] =
/// Piecewise[{{E^(κ Cos[μ - x])/(2 π BesselI[0, κ]), μ-π <= x <= μ+π}}, 0].
/// CDF has no closed form and stays unevaluated, as in wolframscript.
fn pdf_vonmises(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("VonMisesDistribution", dargs), x],
    ))
  };
  let Some(()) = vonmises_checked(dargs) else {
    return unevaluated(x);
  };
  let (m, k) = (dargs[0].clone(), dargs[1].clone());
  let pi = pi();
  let cos = call("Cos", vec![plus(m.clone(), times(int(-1), x.clone()))]);
  let bessel = call("BesselI", vec![int(0), k.clone()]);
  let value = times(
    power(e(), times(k, cos)),
    power(call("Times", vec![int(2), pi.clone(), bessel]), int(-1)),
  );
  let cond = comparison3(
    minus(m.clone(), pi.clone()),
    ComparisonOp::LessEqual,
    x,
    ComparisonOp::LessEqual,
    plus(m, pi),
  );
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// Validated HyperexponentialDistribution arguments (probs, rates).
/// Emits eqln / vprobprm / vrpos (in that order) and returns None on
/// failure; non-list shapes fail silently. The probability sum check
/// only applies when every weight is numeric, with a 1-ULP style
/// tolerance for floats (wolframscript accepts 0.3 + 0.7).
fn hyperexponential_checked(dargs: &[Expr]) -> Option<(Vec<Expr>, Vec<Expr>)> {
  let [Expr::List(probs), Expr::List(rates)] = dargs else {
    return None;
  };
  let dist = || {
    crate::syntax::expr_to_string(&unevaluated(
      "HyperexponentialDistribution",
      dargs,
    ))
  };
  if probs.len() != rates.len() {
    crate::emit_message(&format!(
      "HyperexponentialDistribution::eqln: The values {} and {} at positions 1 and 2 in {} are expected to have the same length.",
      crate::syntax::expr_to_string(&dargs[0]),
      crate::syntax::expr_to_string(&dargs[1]),
      dist()
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  let vals: Vec<Option<f64>> = probs.iter().map(&num).collect();
  let negative = vals.iter().any(|v| v.is_some_and(|v| v < 0.0));
  let bad_sum = vals.iter().all(|v| v.is_some())
    && (vals.iter().map(|v| v.unwrap()).sum::<f64>() - 1.0).abs() > 1e-13;
  if negative || bad_sum {
    crate::emit_message(&format!(
      "HyperexponentialDistribution::vprobprm: The value {} at position 1 in {} is expected to be a list of non-negative numbers summing to 1.",
      crate::syntax::expr_to_string(&dargs[0]),
      dist()
    ));
    return None;
  }
  if rates.iter().any(|r| num(r).is_some_and(|v| v <= 0.0)) {
    crate::emit_message(&format!(
      "HyperexponentialDistribution::vrpos: The value {} at position 2 in {} is expected to be a list of positive numbers.",
      crate::syntax::expr_to_string(&dargs[1]),
      dist()
    ));
    return None;
  }
  Some((probs.to_vec(), rates.to_vec()))
}

/// Per-rate coefficient sums of the hyperexponential mixture when all
/// rates are numeric: groups repeated rates (coefficients merge) and
/// orders by descending rate. `scale_by_rate` selects PDF (p λ) vs
/// CDF (p) coefficients.
fn hyperexponential_numeric_terms(
  probs: &[Expr],
  rates: &[Expr],
  scale_by_rate: bool,
) -> Option<Vec<(Expr, Expr)>> {
  let num = crate::functions::math_ast::try_eval_to_f64;
  let vals: Vec<f64> = rates.iter().map(num).collect::<Option<_>>()?;
  let mut groups: Vec<(f64, Expr, Vec<Expr>)> = Vec::new();
  for (i, v) in vals.iter().enumerate() {
    let coeff = if scale_by_rate {
      times(probs[i].clone(), rates[i].clone())
    } else {
      probs[i].clone()
    };
    match groups.iter_mut().find(|(gv, ..)| gv == v) {
      Some((.., cs)) => cs.push(coeff),
      None => groups.push((*v, rates[i].clone(), vec![coeff])),
    }
  }
  groups.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
  groups
    .into_iter()
    .map(|(_, rate, cs)| {
      let sum = call("Plus", cs);
      eval(sum).ok().map(|c| (c, rate))
    })
    .collect()
}

/// PDF[HyperexponentialDistribution[{p...}, {λ...}], x]:
/// Piecewise[{{Σ p_i λ_i E^(-λ_i x), x >= 0}}, 0]. Numeric rates merge
/// and sort descending; symbolic rates keep the given order with the
/// (λ p) coefficient shape wolframscript uses.
fn pdf_hyperexponential(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("HyperexponentialDistribution", dargs), x],
    ))
  };
  let Some((probs, rates)) = hyperexponential_checked(dargs) else {
    return unevaluated(x);
  };
  let terms: Vec<Expr> =
    match hyperexponential_numeric_terms(&probs, &rates, true) {
      Some(groups) => groups
        .into_iter()
        .map(|(c, rate)| coxian_exp_term(c, None, &rate, &x))
        .collect(),
      None => probs
        .iter()
        .zip(rates.iter())
        .map(|(p, l)| coxian_exp_term(times(l.clone(), p.clone()), None, l, &x))
        .collect(),
    };
  let value = call("Plus", terms);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[HyperexponentialDistribution[{p...}, {λ...}], x]:
/// Piecewise[{{1 - Σ p_i E^(-λ_i x), x >= 0}}, 0].
fn cdf_hyperexponential(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("HyperexponentialDistribution", dargs), x],
    ))
  };
  let Some((probs, rates)) = hyperexponential_checked(dargs) else {
    return unevaluated(x);
  };
  let mut terms: Vec<Expr> = vec![int(1)];
  match hyperexponential_numeric_terms(&probs, &rates, false) {
    Some(groups) => {
      for (c, rate) in groups {
        let neg = eval(times(int(-1), c))?;
        terms.push(coxian_exp_term(neg, None, &rate, &x));
      }
    }
    None => {
      for (p, l) in probs.iter().zip(rates.iter()) {
        terms.push(coxian_exp_term(times(int(-1), p.clone()), None, l, &x));
      }
    }
  }
  let value = call("Plus", terms);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// Mean = Σ p_i/λ_i and Variance = 2 Σ p_i/λ_i² - Mean², in
/// wolframscript's symbolic shapes.
fn hyperexponential_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some((probs, rates)) = hyperexponential_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "HyperexponentialDistribution: invalid parameters".into(),
    ));
  };
  let term =
    |p: &Expr, l: &Expr, k: i128| times(p.clone(), power(l.clone(), int(k)));
  let mean = call(
    "Plus",
    probs
      .iter()
      .zip(rates.iter())
      .map(|(p, l)| term(p, l, -1))
      .collect::<Vec<_>>(),
  );
  let second = times(
    int(2),
    call(
      "Plus",
      probs
        .iter()
        .zip(rates.iter())
        .map(|(p, l)| term(p, l, -2))
        .collect::<Vec<_>>(),
    ),
  );
  let variance = plus(second, times(int(-1), power(mean.clone(), int(2))));
  Ok((eval(mean)?, eval(variance)?))
}

/// Validated CoxianDistribution arguments: (alphas, rates) with
/// len(rates) == len(alphas) + 1. Emits wolframscript's eqln2 /
/// vprobprm2 / vrpos messages (in that order) and returns None when a
/// check fails; non-list shapes fail silently.
fn coxian_checked(dargs: &[Expr]) -> Option<(Vec<Expr>, Vec<Expr>)> {
  let [Expr::List(alphas), Expr::List(rates)] = dargs else {
    return None;
  };
  let dist =
    || crate::syntax::expr_to_string(&unevaluated("CoxianDistribution", dargs));
  if rates.len() != alphas.len() + 1 {
    crate::emit_message(&format!(
      "CoxianDistribution::eqln2: The length of {} at position 2 should be 1 more than the length of {} at position 1 in {}.",
      crate::syntax::expr_to_string(&dargs[1]),
      crate::syntax::expr_to_string(&dargs[0]),
      dist()
    ));
    return None;
  }
  let num = crate::functions::math_ast::try_eval_to_f64;
  if alphas
    .iter()
    .any(|a| num(a).is_some_and(|v| !(0.0..=1.0).contains(&v)))
  {
    crate::emit_message(&format!(
      "CoxianDistribution::vprobprm2: The value {} at position 1 in {} is expected to be a list of numbers between 0 and 1, inclusive.",
      crate::syntax::expr_to_string(&dargs[0]),
      dist()
    ));
    return None;
  }
  if rates.iter().any(|r| num(r).is_some_and(|v| v <= 0.0)) {
    crate::emit_message(&format!(
      "CoxianDistribution::vrpos: The value {} at position 2 in {} is expected to be a list of positive numbers.",
      crate::syntax::expr_to_string(&dargs[1]),
      dist()
    ));
    return None;
  }
  Some((alphas.to_vec(), rates.to_vec()))
}

/// Phase-exit weights of a Coxian distribution as unevaluated exprs:
/// w_k = α1···α_{k-1}(1-α_k) for k < m and w_m = α1···α_{m-1}, shaped
/// to reproduce wolframscript's symbolic moment forms.
fn coxian_weights(alphas: &[Expr]) -> Vec<Expr> {
  let m = alphas.len() + 1;
  let one_minus = |a: &Expr| plus(int(1), times(int(-1), a.clone()));
  let product = |fs: Vec<Expr>| -> Expr {
    match fs.len() {
      0 => int(1),
      1 => fs.into_iter().next().unwrap(),
      _ => call("Times", fs),
    }
  };
  (1..=m)
    .map(|k| {
      let mut fs: Vec<Expr> = alphas[..k - 1].to_vec();
      if k < m {
        fs.push(one_minus(&alphas[k - 1]));
      }
      product(fs)
    })
    .collect()
}

/// Mean and variance of CoxianDistribution[{α...}, {λ...}] in
/// wolframscript's symbolic shapes: Mean = Σ w_k (1/λ1 + … + 1/λk) and
/// E[X²] uses 2 w_1/λ1² for the first phase.
fn coxian_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some((alphas, rates)) = coxian_checked(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "CoxianDistribution: invalid parameters".into(),
    ));
  };
  let weights = coxian_weights(&alphas);
  let inv_sum = |k: usize| -> Expr {
    if k == 1 {
      power(rates[0].clone(), int(-1))
    } else {
      call(
        "Plus",
        rates[..k]
          .iter()
          .map(|r| power(r.clone(), int(-1)))
          .collect::<Vec<_>>(),
      )
    }
  };
  let m = rates.len();
  let mean_terms: Vec<Expr> = (1..=m)
    .map(|k| times(weights[k - 1].clone(), inv_sum(k)))
    .collect();
  let mean = call("Plus", mean_terms.clone());
  let e2_terms: Vec<Expr> = (1..=m)
    .map(|k| {
      if k == 1 {
        times(
          times(int(2), weights[0].clone()),
          power(rates[0].clone(), int(-2)),
        )
      } else {
        let mut parts: Vec<Expr> = rates[..k]
          .iter()
          .map(|r| power(r.clone(), int(-2)))
          .collect();
        parts.push(power(inv_sum(k), int(2)));
        times(weights[k - 1].clone(), call("Plus", parts))
      }
    })
    .collect();
  let mut var_terms = e2_terms;
  var_terms.push(times(int(-1), power(mean.clone(), int(2))));
  let variance = call("Plus", var_terms);
  Ok((eval(mean)?, eval(variance)?))
}

/// True for expressions that evaluate to an exact number (no machine
/// reals anywhere) — the precondition for the Coxian closed forms.
fn coxian_exact(e: &Expr) -> bool {
  fn no_real(e: &Expr) -> bool {
    match e {
      Expr::Real(_) | Expr::BigFloat(..) => false,
      Expr::List(items) => items.iter().all(no_real),
      Expr::FunctionCall { args, .. } => args.iter().all(no_real),
      Expr::BinaryOp { left, right, .. } => no_real(left) && no_real(right),
      Expr::UnaryOp { operand, .. } => no_real(operand),
      _ => true,
    }
  }
  no_real(e) && crate::functions::math_ast::try_eval_to_f64(e).is_some()
}

/// Coxian rate layout for the closed forms: Some(true) if all rates are
/// exact and pairwise distinct, Some(false) if exact and all equal;
/// None otherwise (mixed repetition and symbolic/float rates stay
/// unevaluated like the hypoexponential Erlang gap).
fn coxian_rate_layout(alphas: &[Expr], rates: &[Expr]) -> Option<bool> {
  if !alphas.iter().all(coxian_exact) || !rates.iter().all(coxian_exact) {
    return None;
  }
  let vals: Vec<f64> = rates
    .iter()
    .map(|r| crate::functions::math_ast::try_eval_to_f64(r).unwrap())
    .collect();
  let distinct = vals
    .iter()
    .enumerate()
    .all(|(i, v)| vals[..i].iter().all(|w| w != v));
  let equal = vals.iter().all(|v| *v == vals[0]);
  if distinct {
    Some(true)
  } else if equal {
    Some(false)
  } else {
    None
  }
}

/// Per-exponential coefficients of the Coxian density/survival for
/// pairwise-distinct rates: pdf = Σ_i λi C_i e^(-λi x) and
/// 1 - CDF = Σ_i D_i e^(-λi x) with D_i = Σ_{k>i} w_k Π_{j≤k, j≠i}
/// λj/(λj - λi); C_i shares the sum but scaled by λi.
fn coxian_distinct_coefficients(
  alphas: &[Expr],
  rates: &[Expr],
) -> Result<Vec<Expr>, InterpreterError> {
  let weights = coxian_weights(alphas);
  let m = rates.len();
  let mut coeffs = Vec::with_capacity(m);
  for i in 0..m {
    let mut sum_terms: Vec<Expr> = Vec::new();
    for (k, w) in weights.iter().enumerate().skip(i) {
      // component k+1 uses rates[0..=k]
      let mut fs: Vec<Expr> = vec![w.clone()];
      for (j, r) in rates[..=k].iter().enumerate() {
        if j == i {
          continue;
        }
        fs.push(divide(r.clone(), minus(r.clone(), rates[i].clone())));
      }
      sum_terms.push(call("Times", fs));
    }
    coeffs.push(eval(call("Plus", sum_terms))?);
  }
  Ok(coeffs)
}

/// Exponential factor E^(-λ x) (optionally with an x power in front).
fn coxian_exp_term(
  coeff: Expr,
  xpow: Option<Expr>,
  rate: &Expr,
  x: &Expr,
) -> Expr {
  let e_part = power(e(), times(times(int(-1), rate.clone()), x.clone()));
  match xpow {
    Some(p) => times(times(coeff, p), e_part),
    None => times(coeff, e_part),
  }
}

/// PDF[CoxianDistribution[...], x] for exact all-distinct or all-equal
/// rates; Piecewise[{{Σ terms, x >= 0}}, 0] with terms in descending
/// rate order (distinct) or ascending x power (equal rates).
fn pdf_coxian(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("CoxianDistribution", dargs), x],
    ))
  };
  let Some((alphas, rates)) = coxian_checked(dargs) else {
    return unevaluated(x);
  };
  let Some(distinct) = coxian_rate_layout(&alphas, &rates) else {
    return unevaluated(x);
  };
  let terms: Vec<Expr> = if distinct {
    let coeffs = coxian_distinct_coefficients(&alphas, &rates)?;
    let mut order: Vec<usize> = (0..rates.len()).collect();
    let val = |i: usize| {
      crate::functions::math_ast::try_eval_to_f64(&rates[i]).unwrap()
    };
    order.sort_by(|&a, &b| val(b).partial_cmp(&val(a)).unwrap());
    order
      .iter()
      .map(|&i| {
        let c = eval(times(coeffs[i].clone(), rates[i].clone()))?;
        Ok(coxian_exp_term(c, None, &rates[i], &x))
      })
      .collect::<Result<_, InterpreterError>>()?
  } else {
    // All-equal rates: Σ_k w_k λ^k x^(k-1) e^(-λ x)/(k-1)!.
    let weights = coxian_weights(&alphas);
    let lam = &rates[0];
    (1..=rates.len())
      .map(|k| {
        let mut fact = int(1);
        for j in 2..k {
          fact = times(fact, int(j as i128));
        }
        let c = eval(divide(
          times(weights[k - 1].clone(), power(lam.clone(), int(k as i128))),
          fact,
        ))?;
        let xpow = if k == 1 {
          None
        } else if k == 2 {
          Some(x.clone())
        } else {
          Some(power(x.clone(), int((k - 1) as i128)))
        };
        Ok(coxian_exp_term(c, xpow, lam, &x))
      })
      .collect::<Result<_, InterpreterError>>()?
  };
  let value = call("Plus", terms);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[CoxianDistribution[...], x] for exact all-distinct or all-equal
/// rates. The all-equal form reproduces wolframscript's Piecewise
/// default of 1 (a WS quirk: CDF[..., -1] is 1 there).
fn cdf_coxian(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let uneval = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("CoxianDistribution", dargs), x],
    ))
  };
  let Some((alphas, rates)) = coxian_checked(dargs) else {
    return uneval(x);
  };
  let Some(distinct) = coxian_rate_layout(&alphas, &rates) else {
    return uneval(x);
  };
  let mut terms: Vec<Expr> = vec![int(1)];
  let default;
  if distinct {
    let coeffs = coxian_distinct_coefficients(&alphas, &rates)?;
    let mut order: Vec<usize> = (0..rates.len()).collect();
    let val = |i: usize| {
      crate::functions::math_ast::try_eval_to_f64(&rates[i]).unwrap()
    };
    order.sort_by(|&a, &b| val(b).partial_cmp(&val(a)).unwrap());
    for &i in &order {
      let c = eval(times(int(-1), coeffs[i].clone()))?;
      terms.push(coxian_exp_term(c, None, &rates[i], &x));
    }
    default = int(0);
  } else {
    // Survival of the equal-rate mixture: Σ_j (Σ_{k>j} w_k) λ^j x^j
    // e^(-λ x)/j! subtracted from 1.
    let weights = coxian_weights(&alphas);
    let lam = &rates[0];
    let m = rates.len();
    for j in 0..m {
      let mut fact = int(1);
      for f in 2..=j {
        fact = times(fact, int(f as i128));
      }
      let wsum = unevaluated("Plus", &weights[j..]);
      let c = eval(times(
        int(-1),
        divide(times(wsum, power(lam.clone(), int(j as i128))), fact),
      ))?;
      let xpow = if j == 0 {
        None
      } else if j == 1 {
        Some(x.clone())
      } else {
        Some(power(x.clone(), int(j as i128)))
      };
      terms.push(coxian_exp_term(c, xpow, lam, &x));
    }
    default = int(1);
  }
  let value = call("Plus", terms);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(value, cond)], default))
}

/// The rates of a HypoexponentialDistribution[{λ1, …, λn}] when all are
/// concrete and pairwise distinct — the precondition for the
/// distinct-rate closed forms below. (Repeated rates need Erlang-style
/// terms wolframscript produces, e.g. 4 x E^(-2 x) for {2, 2}; those and
/// symbolic rates stay unevaluated.)
fn hypoexponential_distinct_rates(dargs: &[Expr]) -> Option<Vec<(Expr, f64)>> {
  match dargs {
    [Expr::List(rates)] if !rates.is_empty() => {
      let keyed: Option<Vec<(Expr, f64)>> = rates
        .iter()
        .map(|r| {
          crate::functions::math_ast::try_eval_to_f64(r).map(|f| (r.clone(), f))
        })
        .collect();
      let keyed = keyed?;
      for i in 0..keyed.len() {
        for j in 0..i {
          if keyed[i].1 == keyed[j].1 {
            return None;
          }
        }
      }
      Some(keyed)
    }
    _ => None,
  }
}

/// The distinct-rate mixture coefficients c_i = Π_{j≠i} λj/(λj − λi),
/// as exact expressions.
fn hypoexponential_coefficients(
  rates: &[(Expr, f64)],
) -> Result<Vec<Expr>, InterpreterError> {
  let mut coeffs = Vec::with_capacity(rates.len());
  for i in 0..rates.len() {
    let mut num: Vec<Expr> = Vec::new();
    let mut den: Vec<Expr> = Vec::new();
    for j in 0..rates.len() {
      if j == i {
        continue;
      }
      num.push(rates[j].0.clone());
      den.push(minus(rates[j].0.clone(), rates[i].0.clone()));
    }
    let product = |fs: Vec<Expr>| -> Expr {
      match fs.len() {
        0 => int(1),
        1 => fs.into_iter().next().unwrap(),
        _ => call("Times", fs),
      }
    };
    coeffs.push(eval(divide(product(num), product(den)))?);
  }
  Ok(coeffs)
}

/// PDF[HypoexponentialDistribution[{λ1, …}], x] for distinct concrete
/// rates: Piecewise[{{Σ c_i λ_i E^(-λ_i x), x > 0}}, 0].
fn pdf_hypoexponential(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("HypoexponentialDistribution", dargs), x],
    ))
  };
  let Some(rates) = hypoexponential_distinct_rates(dargs) else {
    return unevaluated(x);
  };
  let coeffs = hypoexponential_coefficients(&rates)?;
  let terms: Vec<Expr> = rates
    .iter()
    .zip(coeffs.iter())
    .map(|((rate, _), c)| {
      times(
        times(c.clone(), rate.clone()),
        power(e(), times(times(int(-1), rate.clone()), x.clone())),
      )
    })
    .collect();
  let value = call("Plus", terms);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// CDF[HypoexponentialDistribution[{λ1, …}], x] for distinct concrete
/// rates: Piecewise[{{1 - Σ c_i E^(-λ_i x), x > 0}}, 0].
fn cdf_hypoexponential(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("HypoexponentialDistribution", dargs), x],
    ))
  };
  let Some(rates) = hypoexponential_distinct_rates(dargs) else {
    return unevaluated(x);
  };
  let coeffs = hypoexponential_coefficients(&rates)?;
  let mut terms: Vec<Expr> = vec![int(1)];
  for ((rate, _), c) in rates.iter().zip(coeffs.iter()) {
    terms.push(times(
      times(int(-1), c.clone()),
      power(e(), times(times(int(-1), rate.clone()), x.clone())),
    ));
  }
  let value = call("Plus", terms);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(value, cond)], int(0)))
}

/// Parses `NegativeMultinomialDistribution[n, {p1, …, pk}]` arguments into
/// `(n, probs)`.
fn negative_multinomial_params(
  dargs: &[Expr],
) -> Result<(Expr, crate::ExprList), InterpreterError> {
  match dargs {
    [n, Expr::List(probs)] if !probs.is_empty() => {
      Ok((n.clone(), probs.clone()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "NegativeMultinomialDistribution expects a parameter n and a list of failure probabilities".into(),
    )),
  }
}

/// The complement `1 - p1 - … - pk` of the failure probabilities, grouped as
/// `1 - (p1 + … + pk)` so machine-precision parameters fold in
/// wolframscript's order (1 - (0.2 + 0.4) ≠ 1 - 0.2 - 0.4 in f64).
fn negative_multinomial_success(probs: &[Expr]) -> Expr {
  let mut sum = probs[0].clone();
  for p in probs.iter().skip(1) {
    sum = plus(sum, p.clone());
  }
  minus(int(1), sum)
}

/// PDF[NegativeMultinomialDistribution[n, {p1, ..., pk}], {x1, ..., xk}]
/// = (1 - p1 - ... - pk)^n * p1^x1 * ... * pk^xk
///   * Pochhammer[n, x1 + ... + xk] / (x1! * ... * xk!)
///   when all xi >= 0
/// = 0 otherwise
fn pdf_negative_multinomial(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("NegativeMultinomialDistribution", dargs), x],
    ))
  };
  let (n, probs) = negative_multinomial_params(dargs)?;
  let xs = match &x {
    Expr::List(items) if items.len() == probs.len() => items.clone(),
    _ => {
      // A point of the wrong shape stays unevaluated, as in wolframscript.
      return unevaluated(x);
    }
  };
  if any_concrete_non_integer(&xs) {
    return Ok(int(0));
  }

  // Numerator: (1 - Σp)^n * p1^x1 * ... * pk^xk * Pochhammer[n, Σx]
  let mut numer = power(negative_multinomial_success(&probs), n.clone());
  for (p, xi) in probs.iter().zip(xs.iter()) {
    numer = times(numer, power(p.clone(), xi.clone()));
  }
  let mut sum_xs = xs[0].clone();
  for xi in xs.iter().skip(1) {
    sum_xs = plus(sum_xs, xi.clone());
  }
  numer = times(numer, call("Pochhammer", vec![n, sum_xs]));

  // Denominator: x1! * ... * xk!
  let mut denom = factorial(xs[0].clone());
  for xi in xs.iter().skip(1) {
    denom = times(denom, factorial(xi.clone()));
  }
  let pdf_val = divide(numer, denom);

  // Support: all xi >= 0.
  let conditions: Vec<Expr> = xs
    .iter()
    .map(|xi| comparison(xi.clone(), ComparisonOp::GreaterEqual, int(0)))
    .collect();
  let combined_cond = if conditions.len() == 1 {
    conditions.into_iter().next().unwrap()
  } else {
    call("And", conditions)
  };

  eval(piecewise(vec![(pdf_val, combined_cond)], int(0)))
}

/// CDF[NegativeMultinomialDistribution[n, {p1, ..., pk}], {x1, ..., xk}]
/// for a fully numeric distribution and point: the finite sum of the PDF
/// over the integer grid 0 <= i_j <= Floor[x_j]. Symbolic arguments stay
/// unevaluated (wolframscript's combined closed form for symbolic n is a
/// canonicalization rabbit hole).
fn cdf_negative_multinomial(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("NegativeMultinomialDistribution", dargs), x],
    ))
  };
  let Ok((n, probs)) = negative_multinomial_params(dargs) else {
    return unevaluated(x);
  };
  if crate::functions::math_ast::expr_to_num(&n).is_none()
    || probs
      .iter()
      .any(|p| crate::functions::math_ast::expr_to_num(p).is_none())
  {
    return unevaluated(x);
  }
  let Expr::List(xs) = &x else {
    return unevaluated(x);
  };
  if xs.len() != probs.len() {
    return unevaluated(x);
  }
  let mut bounds: Vec<i64> = Vec::with_capacity(xs.len());
  for xi in xs.iter() {
    let Some(v) = crate::functions::math_ast::expr_to_num(xi) else {
      return unevaluated(x.clone());
    };
    let f = v.floor();
    if f < 0.0 {
      return Ok(int(0));
    }
    bounds.push(f as i64);
  }

  // Row-major walk over the grid, accumulating PDF terms.
  let mut point = vec![0i64; bounds.len()];
  let mut total: Option<Expr> = None;
  loop {
    let point_expr =
      Expr::List(point.iter().map(|&i| Expr::Integer(i as i128)).collect());
    let term = pdf_negative_multinomial(dargs, point_expr)?;
    total = Some(match total {
      Some(acc) => plus(acc, term),
      None => term,
    });
    // Advance the mixed-radix counter (last component fastest).
    let mut dim = point.len();
    let mut advanced = false;
    while dim > 0 {
      dim -= 1;
      if point[dim] < bounds[dim] {
        point[dim] += 1;
        advanced = true;
        break;
      }
      point[dim] = 0;
    }
    if !advanced {
      return eval(total.expect("grid is never empty"));
    }
  }
}

/// PDF[MultivariatePoissonDistribution[μ_0, {μ_1, μ_2}], {x, y}] for the
/// bivariate case. The closed form (Wolfram's choice):
///
///   (-μ_0)^x · μ_2^(y-x) · HypergeometricU[-x, 1-x+y, -μ_1·μ_2/μ_0]
///   ───────────────────────────────────────────────────────────────
///              E^(μ_0+μ_1+μ_2) · x! · y!
///
/// with support x ≥ 0 ∧ y ≥ 0; outside the support the PDF is 0.
fn pdf_multivariate_poisson(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MultivariatePoissonDistribution expects 2 arguments".into(),
    ));
  }
  let mu0 = dargs[0].clone();
  let (mu1, mu2) = match &dargs[1] {
    Expr::List(items) if items.len() == 2 => {
      (items[0].clone(), items[1].clone())
    }
    // The closed form below is only for the bivariate case; defer
    // higher-dimensional inputs to the unevaluated form by erroring out
    // and letting `pdf_ast` swallow the error… simpler to short-circuit
    // here with an unchanged head.
    _ => {
      return Ok(call(
        "PDF",
        vec![unevaluated("MultivariatePoissonDistribution", dargs), x],
      ));
    }
  };
  let (xv, yv) = match &x {
    Expr::List(items) if items.len() == 2 => {
      (items[0].clone(), items[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "PDF[MultivariatePoissonDistribution[...], x]: x must be a 2-element list"
          .into(),
      ));
    }
  };

  // Pre-evaluate the numeric-parameter subexpressions so the final
  // Piecewise doesn't leak unsimplified literals (`E^(1 + 2 + 3)` etc).
  // The symbolic `x`, `y` are kept intact.
  let neg_mu0 = eval(times(int(-1), mu0.clone()))?;
  let neg_mu1_mu2_over_mu0 = eval(times(
    int(-1),
    divide(times(mu1.clone(), mu2.clone()), mu0.clone()),
  ))?;
  let sum_mu_eval = eval(plus(plus(mu0.clone(), mu1.clone()), mu2.clone()))?;

  // Numerator: (-μ_0)^x · μ_2^(y - x) · HypergeometricU[-x, 1 - x + y, -μ_1·μ_2/μ_0]
  let neg_mu0_pow_x = power(neg_mu0, xv.clone());
  // Canonicalise `y - x` to its Plus form (`-x + y`) so the final
  // Piecewise matches wolframscript's display.
  let y_minus_x_canon = eval(minus(yv.clone(), xv.clone()))?;
  let mu2_pow_y_minus_x = power(mu2.clone(), y_minus_x_canon);
  let neg_x = times(int(-1), xv.clone());
  let one_minus_x_plus_y = plus(minus(int(1), xv.clone()), yv.clone());
  let hypergeometric_u = call(
    "HypergeometricU",
    vec![neg_x, one_minus_x_plus_y, neg_mu1_mu2_over_mu0],
  );
  let numerator =
    times(times(neg_mu0_pow_x, mu2_pow_y_minus_x), hypergeometric_u);

  // Denominator: E^(μ_0+μ_1+μ_2) · x! · y!
  let exp_sum = power(e(), sum_mu_eval);
  let x_fact = factorial(xv.clone());
  let y_fact = factorial(yv.clone());
  let denominator = times(times(exp_sum, x_fact), y_fact);

  let pdf_val = divide(numerator, denominator);

  // Condition: x >= 0 && y >= 0.
  let cond = call(
    "And",
    vec![
      comparison(xv, ComparisonOp::GreaterEqual, int(0)),
      comparison(yv, ComparisonOp::GreaterEqual, int(0)),
    ],
  );

  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// Returns (Mean list, Variance list) for MultinomialDistribution
/// Mean_i = n * p_i, Variance_i = n * p_i * (1 - p_i)
pub fn multinomial_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MultinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let probs = match &dargs[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MultinomialDistribution: second argument must be a list".into(),
      ));
    }
  };

  let mut means = Vec::new();
  let mut variances = Vec::new();
  for p in &probs {
    let mean_i = times(n.clone(), p.clone());
    let var_i = times(times(n.clone(), p.clone()), minus(int(1), p.clone()));
    means.push(eval(mean_i)?);
    variances.push(eval(var_i)?);
  }

  Ok((Expr::List(means.into()), Expr::List(variances.into())))
}

/// Returns (Mean list, Variance list) for
/// NegativeMultinomialDistribution[n, {p1, …, pk}] with q = 1 - p1 - … - pk:
/// Mean_i = n p_i / q, Variance_i = n p_i (1 - Σ_{j≠i} p_j) / q^2.
pub fn negative_multinomial_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let (n, probs) = negative_multinomial_params(dargs)?;
  let q = negative_multinomial_success(&probs);

  let mut means = Vec::with_capacity(probs.len());
  let mut variances = Vec::with_capacity(probs.len());
  for p in probs.iter() {
    means.push(eval(divide(times(n.clone(), p.clone()), q.clone()))?);
    // The numerator factor 1 - Σ_{j≠i} p_j is built as q + p_i: symbolically
    // it cancels to wolframscript's subtracted form, and for machine floats
    // it reproduces wolframscript's fold order (1 - (0.2 + 0.4) + 0.2 differs
    // from 1 - 0.4 in the last bit).
    let others = plus(q.clone(), p.clone());
    variances.push(eval(divide(
      times(times(n.clone(), p.clone()), others),
      power(q.clone(), int(2)),
    ))?);
  }

  Ok((Expr::List(means.into()), Expr::List(variances.into())))
}

/// Covariance[NegativeMultinomialDistribution[n, {p1, …, pk}]] is the k×k
/// matrix with diagonal n (p_i^2/q^2 + p_i/q) and off-diagonal n p_i p_j / q^2
/// where q = 1 - p1 - … - pk.
pub fn negative_multinomial_covariance(
  dargs: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (n, probs) = negative_multinomial_params(dargs)?;
  let q = negative_multinomial_success(&probs);

  let mut rows = Vec::with_capacity(probs.len());
  for (i, pi) in probs.iter().enumerate() {
    let mut row = Vec::with_capacity(probs.len());
    for (j, pj) in probs.iter().enumerate() {
      let entry = if i == j {
        times(
          n.clone(),
          plus(
            divide(power(pi.clone(), int(2)), power(q.clone(), int(2))),
            divide(pi.clone(), q.clone()),
          ),
        )
      } else {
        divide(
          times(times(n.clone(), pi.clone()), pj.clone()),
          power(q.clone(), int(2)),
        )
      };
      row.push(eval(entry)?);
    }
    rows.push(Expr::List(row.into()));
  }
  Ok(Expr::List(rows.into()))
}

/// Returns (Mean matrix, Variance matrix) for
/// WishartMatrixDistribution[ν, Σ]: Mean = ν Σ,
/// Variance_ij = ν (σ_ij^2 + σ_ii σ_jj).
///
/// Validation mirrors wolframscript's moment-time checks (the constructor
/// itself never validates): a non-numeric / non-symmetric /
/// non-positive-definite Σ emits `posdefprm`; then a non-numeric ν or
/// ν <= p - 1 emits `bprm`. On either failure returns Err so the caller
/// leaves the call unevaluated.
pub fn wishart_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let dist_str = || {
    crate::syntax::expr_to_string(&unevaluated(
      "WishartMatrixDistribution",
      dargs,
    ))
  };
  let fail = |msg: String| -> Result<(Expr, Expr), InterpreterError> {
    crate::emit_message(&msg);
    Err(InterpreterError::EvaluationError(
      "invalid WishartMatrixDistribution parameters".into(),
    ))
  };
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WishartMatrixDistribution expects 2 arguments".into(),
    ));
  }
  let (nu, sigma) = (&dargs[0], &dargs[1]);

  // Σ: square, numeric, symmetric, positive definite.
  let posdefprm = |s: &Expr| {
    format!(
      "WishartMatrixDistribution::posdefprm: The value {} at position 2 in {} is expected to be a symmetric positive definite matrix.",
      crate::syntax::expr_to_string(s),
      dist_str()
    )
  };
  let Expr::List(rows) = sigma else {
    return fail(posdefprm(sigma));
  };
  let p = rows.len();
  let mut sig = vec![vec![0.0f64; p]; p];
  let mut sig_exprs: Vec<Vec<Expr>> = Vec::with_capacity(p);
  for (i, row) in rows.iter().enumerate() {
    let Expr::List(cells) = row else {
      return fail(posdefprm(sigma));
    };
    if cells.len() != p {
      return fail(posdefprm(sigma));
    }
    let mut row_exprs = Vec::with_capacity(p);
    for (j, cell) in cells.iter().enumerate() {
      let Some(v) = crate::functions::math_ast::try_eval_to_f64(cell) else {
        return fail(posdefprm(sigma));
      };
      sig[i][j] = v;
      row_exprs.push(cell.clone());
    }
    sig_exprs.push(row_exprs);
  }
  let scale = sig.iter().flatten().fold(1.0f64, |acc, x| acc.max(x.abs()));
  let tol = 1e-9 * scale;
  for i in 0..p {
    for j in 0..i {
      if (sig[i][j] - sig[j][i]).abs() > tol {
        return fail(posdefprm(sigma));
      }
    }
  }
  // Positive definiteness via leading principal minors (Sylvester).
  for k in 1..=p {
    let mut m: Vec<Vec<f64>> = (0..k).map(|i| sig[i][..k].to_vec()).collect();
    // Gaussian elimination determinant.
    let mut det = 1.0f64;
    for col in 0..k {
      let pivot = (col..k)
        .max_by(|&a, &b| {
          m[a][col]
            .abs()
            .partial_cmp(&m[b][col].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
      if m[pivot][col].abs() < 1e-12 * scale.max(1.0) {
        det = 0.0;
        break;
      }
      if pivot != col {
        m.swap(pivot, col);
        det = -det;
      }
      det *= m[col][col];
      for r in col + 1..k {
        let f = m[r][col] / m[col][col];
        for c in col..k {
          m[r][c] -= f * m[col][c];
        }
      }
    }
    if det <= 0.0 {
      return fail(posdefprm(sigma));
    }
  }

  // ν: numeric with ν > p - 1.
  let bprm = || {
    format!(
      "WishartMatrixDistribution::bprm: The parameters of distribution {} are not valid. Use DistributionParameterAssumptions to obtain the parameter assumptions.",
      dist_str()
    )
  };
  let Some(nu_f) = crate::functions::math_ast::try_eval_to_f64(nu) else {
    return fail(bprm());
  };
  if nu_f <= (p as f64) - 1.0 {
    return fail(bprm());
  }

  let mut mean_rows = Vec::with_capacity(p);
  let mut var_rows = Vec::with_capacity(p);
  for i in 0..p {
    let mut mean_row = Vec::with_capacity(p);
    let mut var_row = Vec::with_capacity(p);
    for j in 0..p {
      mean_row.push(eval(times(nu.clone(), sig_exprs[i][j].clone()))?);
      // ν (σ_ij^2 + σ_ii σ_jj)
      var_row.push(eval(times(
        nu.clone(),
        plus(
          power(sig_exprs[i][j].clone(), int(2)),
          times(sig_exprs[i][i].clone(), sig_exprs[j][j].clone()),
        ),
      ))?);
    }
    mean_rows.push(Expr::List(mean_row.into()));
    var_rows.push(Expr::List(var_row.into()));
  }
  Ok((Expr::List(mean_rows.into()), Expr::List(var_rows.into())))
}

/// Mean and Variance for MultivariatePoissonDistribution[theta0, {theta1, ..., thetan}]
/// Mean = {theta0 + theta1, ..., theta0 + thetan}
/// Variance = {theta0 + theta1, ..., theta0 + thetan} (same as mean)
pub fn multivariate_poisson_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MultivariatePoissonDistribution expects 2 arguments".into(),
    ));
  }
  let theta0 = dargs[0].clone();
  let thetas = match &dargs[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MultivariatePoissonDistribution: second argument must be a list"
          .into(),
      ));
    }
  };

  let mut components = Vec::new();
  for theta_i in &thetas {
    let sum = call("Plus", vec![theta0.clone(), theta_i.clone()]);
    components.push(eval(sum)?);
  }

  // Variance equals Mean for Poisson marginals
  Ok((
    Expr::List(components.clone().into()),
    Expr::List(components.into()),
  ))
}

/// The shape parameters {α1, …, α(k+1)} of a
/// `DirichletDistribution[{α1, …, α(k+1)}]` (a k-dimensional Dirichlet needs
/// at least two parameters).
fn dirichlet_alphas(
  dargs: &[Expr],
) -> Result<crate::ExprList, InterpreterError> {
  match dargs {
    [Expr::List(alphas)] if alphas.len() >= 2 => Ok(alphas.clone()),
    _ => Err(InterpreterError::EvaluationError(
      "DirichletDistribution expects one list of at least 2 shape parameters"
        .into(),
    )),
  }
}

/// The Dirichlet normalizing total α0 = α1 + … + α(k+1), unevaluated.
fn dirichlet_total(alphas: &[Expr]) -> Expr {
  unevaluated("Plus", alphas)
}

/// The common denominator of the Dirichlet second moments,
/// α0^2 (1 + α0), as an unevaluated reciprocal `α0^-2 (1 + α0)^-1`.
fn dirichlet_moment_denominator(alphas: &[Expr]) -> Expr {
  let total = dirichlet_total(alphas);
  times(
    power(total.clone(), int(-2)),
    power(plus(int(1), total), int(-1)),
  )
}

/// Mean and Variance for `DirichletDistribution[{α1, …, α(k+1)}]`. Both are
/// k-vectors (the last component is determined by the others and dropped):
/// Mean_i = αi/α0 and Variance_i = αi (α0 - αi) / (α0² (1 + α0)), the
/// numerator spelled αi·(sum of the other parameters) as Wolfram displays it.
pub fn dirichlet_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let alphas = dirichlet_alphas(dargs)?;
  let total = dirichlet_total(&alphas);
  let denom = dirichlet_moment_denominator(&alphas);

  let mut means = Vec::with_capacity(alphas.len() - 1);
  let mut variances = Vec::with_capacity(alphas.len() - 1);
  for (i, alpha_i) in alphas.iter().enumerate().take(alphas.len() - 1) {
    means.push(eval(times(alpha_i.clone(), power(total.clone(), int(-1))))?);
    let others = call(
      "Plus",
      alphas
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, a)| a.clone())
        .collect::<Vec<_>>(),
    );
    variances.push(eval(times(times(alpha_i.clone(), others), denom.clone()))?);
  }
  Ok((Expr::List(means.into()), Expr::List(variances.into())))
}

/// Covariance matrix for `DirichletDistribution[{α1, …, α(k+1)}]`: the k×k
/// matrix with diagonal (αi α0 - αi²) / (α0² (1 + α0)) — spelled
/// `-αi² + αi·α0` as Wolfram displays it — and off-diagonal
/// -αi αj / (α0² (1 + α0)).
pub fn dirichlet_covariance(dargs: &[Expr]) -> Result<Expr, InterpreterError> {
  let alphas = dirichlet_alphas(dargs)?;
  let total = dirichlet_total(&alphas);
  let denom = dirichlet_moment_denominator(&alphas);
  let k = alphas.len() - 1;

  let mut rows = Vec::with_capacity(k);
  for i in 0..k {
    let mut row = Vec::with_capacity(k);
    for j in 0..k {
      let numer = if i == j {
        plus(
          times(int(-1), power(alphas[i].clone(), int(2))),
          times(alphas[i].clone(), total.clone()),
        )
      } else {
        times(int(-1), times(alphas[i].clone(), alphas[j].clone()))
      };
      row.push(eval(times(numer, denom.clone()))?);
    }
    rows.push(Expr::List(row.into()));
  }
  Ok(Expr::List(rows.into()))
}

/// PDF[DirichletDistribution[{α1, …, α(k+1)}], {x1, …, xk}]
/// = Gamma[α0]/∏ Gamma[αi] · x1^(α1-1) ⋯ xk^(αk-1) (1-Σxi)^(α(k+1)-1)
///   on the open simplex x1 > 0 ∧ … ∧ xk > 0 ∧ 1 - Σxi > 0, and 0 outside.
fn pdf_dirichlet(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  // Anything that is not a k-component point stays unevaluated, as in
  // Wolfram (a scalar second argument just echoes back).
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("DirichletDistribution", dargs), x],
    ))
  };
  let alphas = dirichlet_alphas(dargs)?;
  let xs = match &x {
    Expr::List(items) => items.clone(),
    _ => return unevaluated(x),
  };
  if xs.len() != alphas.len() - 1 {
    return unevaluated(x);
  }

  // The last coordinate is determined by the simplex: 1 - x1 - … - xk.
  let mut last = int(1);
  for xi in xs.iter() {
    last = minus(last, xi.clone());
  }

  // x1^(α1-1) ⋯ xk^(αk-1) (1-Σxi)^(α(k+1)-1) · Gamma[α0] / ∏ Gamma[αi]
  let mut value = power(xs[0].clone(), minus(alphas[0].clone(), int(1)));
  for (xi, alpha_i) in xs.iter().zip(alphas.iter()).skip(1) {
    value = times(value, power(xi.clone(), minus(alpha_i.clone(), int(1))));
  }
  value = times(
    value,
    power(
      last.clone(),
      minus(alphas[alphas.len() - 1].clone(), int(1)),
    ),
  );
  value = times(value, gamma(dirichlet_total(&alphas)));
  for alpha_i in alphas.iter() {
    value = times(value, power(gamma(alpha_i.clone()), int(-1)));
  }

  // x1 > 0 && … && xk > 0 && 1 - Σxi > 0 (the open simplex).
  let mut conditions: Vec<Expr> = xs
    .iter()
    .map(|xi| comparison(xi.clone(), ComparisonOp::Greater, int(0)))
    .collect();
  conditions.push(comparison(last, ComparisonOp::Greater, int(0)));
  let combined = call("And", conditions);

  eval(piecewise(vec![(eval(value)?, combined)], int(0)))
}

/// PDF[NegativeBinomialDistribution[n, p], k]
/// = (1-p)^k * p^n * Binomial[k+n-1, n-1] for k >= 0
fn pdf_negative_binomial(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NegativeBinomialDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();
  // (1-p)^k * p^n * Binomial[-1+k+n, -1+n]
  let one_minus_p = minus(int(1), p.clone());
  let n_minus_1 = plus(int(-1), n.clone());
  let k_plus_n_minus_1 = call("Plus", vec![int(-1), x.clone(), n.clone()]);
  let binom = call("Binomial", vec![k_plus_n_minus_1, n_minus_1]);
  let pdf_val = times(times(power(one_minus_p, x.clone()), power(p, n)), binom);
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// PDF[HalfNormalDistribution[t], x] = Piecewise[{{(2*t)/(E^((t^2*x^2)/Pi)*Pi), x > 0}}, 0]
fn pdf_half_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "HalfNormalDistribution expects 1 argument".into(),
    ));
  }
  let t = dargs[0].clone();
  // (2*t) / (E^((t^2 * x^2) / Pi) * Pi)
  let numerator = times(int(2), t.clone());
  let exponent =
    divide(times(power(t, int(2)), power(x.clone(), int(2))), pi());
  let denominator = times(power(e(), exponent), pi());
  let pdf_val = divide(numerator, denominator);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[HalfNormalDistribution[t], x] = Piecewise[{{Erf[(t*x)/Sqrt[Pi]], x > 0}}, 0]
fn cdf_half_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "HalfNormalDistribution expects 1 argument".into(),
    ));
  }
  let t = dargs[0].clone();
  let erf_arg = divide(times(t, x.clone()), sqrt(pi()));
  let erf_val = call("Erf", vec![erf_arg]);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(erf_val, cond)], int(0)))
}

/// PDF[ChiDistribution[n], x] = Piecewise[{{2^(1-n/2) * x^(n-1) / (E^(x^2/2) * Gamma[n/2]), x > 0}}, 0]
fn pdf_chi(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();
  // 2^(1 - n/2)
  let pow2 = power(int(2), minus(int(1), divide(n.clone(), int(2))));
  // x^(n-1)
  let x_pow = power(x.clone(), minus(n.clone(), int(1)));
  // E^(x^2/2)
  let exp_part = power(e(), divide(power(x.clone(), int(2)), int(2)));
  // Gamma[n/2]
  let gamma_part = call("Gamma", vec![divide(n, int(2))]);
  let pdf_val = eval(divide(times(pow2, x_pow), times(exp_part, gamma_part)))?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(pdf_val, cond)], int(0)))
}

/// CDF[ChiDistribution[n], x] = Piecewise[{{GammaRegularized[n/2, 0, x^2/2], x > 0}}, 0]
fn cdf_chi(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ChiDistribution expects 1 argument".into(),
    ));
  }
  let n = dargs[0].clone();
  let gamma_reg = call(
    "GammaRegularized",
    vec![
      divide(n, int(2)),
      int(0),
      divide(power(x.clone(), int(2)), int(2)),
    ],
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(gamma_reg, cond)], int(0)))
}

/// PDF[StableDistribution[alpha, beta, mu, sigma], x]
/// General case returns unevaluated. Special cases:
/// - alpha=1, beta=0: Cauchy(mu, sigma)
/// - alpha=2: Normal(mu, sigma*Sqrt[2])
fn pdf_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "PDF",
      vec![unevaluated("StableDistribution", dargs), x],
    ))
  };
  let (alpha, beta, mu, sigma) = match dargs.len() {
    2 => (dargs[0].clone(), dargs[1].clone(), int(0), int(1)),
    4 => (
      dargs[0].clone(),
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
    ),
    // 5-param canonical form: StableDistribution[1, alpha, beta, mu, sigma]
    5 => (
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
      dargs[4].clone(),
    ),
    _ => {
      return unevaluated(x);
    }
  };

  // Evaluate parameters to check for special cases
  let alpha_eval = evaluate_expr_to_expr(&alpha)?;
  let beta_eval = evaluate_expr_to_expr(&beta)?;

  // alpha=1, beta=0: Cauchy distribution
  // Use expanded form: sigma / (Pi * (sigma^2 + (x - mu)^2))
  // to match Wolfram's canonical simplification
  if matches!(&alpha_eval, Expr::Integer(1))
    && matches!(&beta_eval, Expr::Integer(0))
  {
    let diff = minus(x, mu);
    let numer = sigma.clone();
    let denom = times(pi(), plus(power(sigma, int(2)), power(diff, int(2))));
    return eval(divide(numer, denom));
  }

  // alpha=2: Normal(mu, sigma*Sqrt[2])
  if matches!(&alpha_eval, Expr::Integer(2)) {
    let s = times(sigma, sqrt(int(2)));
    let cauchy_args = vec![mu, s];
    return pdf_normal(&cauchy_args, x);
  }

  // General case: return unevaluated
  unevaluated(x)
}

/// CDF[StableDistribution[alpha, beta, mu, sigma], x]
fn cdf_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |x: Expr| {
    Ok(call(
      "CDF",
      vec![unevaluated("StableDistribution", dargs), x],
    ))
  };
  let (alpha, beta, mu, sigma) = match dargs.len() {
    2 => (dargs[0].clone(), dargs[1].clone(), int(0), int(1)),
    4 => (
      dargs[0].clone(),
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
    ),
    // 5-param canonical form: StableDistribution[1, alpha, beta, mu, sigma]
    5 => (
      dargs[1].clone(),
      dargs[2].clone(),
      dargs[3].clone(),
      dargs[4].clone(),
    ),
    _ => {
      return unevaluated(x);
    }
  };

  let alpha_eval = evaluate_expr_to_expr(&alpha)?;
  let beta_eval = evaluate_expr_to_expr(&beta)?;

  // alpha=1, beta=0: Cauchy CDF = 1/2 + ArcTan[(x-mu)/sigma]/Pi
  if matches!(&alpha_eval, Expr::Integer(1))
    && matches!(&beta_eval, Expr::Integer(0))
  {
    let cauchy_args = vec![mu, sigma];
    return cdf_cauchy(&cauchy_args, x);
  }

  // alpha=2: Normal CDF
  if matches!(&alpha_eval, Expr::Integer(2)) {
    let s = times(sigma, sqrt(int(2)));
    let normal_args = vec![mu, s];
    return cdf_normal(&normal_args, x);
  }

  // General case: return unevaluated
  unevaluated(x)
}

/// PDF[ArcSinDistribution[{a, b}], x]
/// = Piecewise[{{1/(Pi*Sqrt[(x - a)*(b - x)]), a < x < b}}, 0]
fn pdf_arcsin(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSinDistribution expects 1 argument (a list {a, b})".into(),
    ));
  }
  let (a, b) = match &dargs[0] {
    Expr::List(bounds) if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone())
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ArcSinDistribution expects a list {a, b}".into(),
      ));
    }
  };

  // density = 1 / (Pi * Sqrt[(x - a) * (b - x)])
  let density = divide(
    int(1),
    times(
      pi(),
      sqrt(times(
        minus(x.clone(), a.clone()),
        minus(b.clone(), x.clone()),
      )),
    ),
  );

  // condition: a < x < b
  let cond = Expr::Comparison {
    operands: vec![a, x, b],
    operators: vec![ComparisonOp::Less, ComparisonOp::Less],
  };

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[PascalDistribution[n, p], k]
/// = (1-p)^(k-n) * p^n * Binomial[k-1, n-1] for k >= n
fn pdf_pascal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PascalDistribution expects 2 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let p = dargs[1].clone();

  // Binomial[k-1, n-1]
  let binom = call(
    "Binomial",
    vec![minus(x.clone(), int(1)), minus(n.clone(), int(1))],
  );
  // p^n
  let p_n = power(p.clone(), n.clone());
  // (1-p)^(k-n)
  let one_minus_p_k_n = power(minus(int(1), p), minus(x.clone(), n.clone()));

  let density = times(times(binom, p_n), one_minus_p_k_n);

  // k >= n
  let cond = call("GreaterEqual", vec![x, n]);

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[DagumDistribution[p, a, b], x]
/// = (a*p/x) * ((x/b)^(a*p)) / (1 + (x/b)^a)^(p+1) for x > 0
fn pdf_dagum(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "DagumDistribution expects 3 arguments".into(),
    ));
  }
  let p = dargs[0].clone();
  let a = dargs[1].clone();
  let b = dargs[2].clone();

  // a*p * x^(a*p - 1) * (1 + (x/b)^a)^(-1-p) / b^(a*p)
  let ap = times(a.clone(), p.clone());
  let x_to_ap_minus_1 = power(x.clone(), minus(ap.clone(), int(1)));
  let x_over_b_to_a = power(divide(x.clone(), b.clone()), a.clone());
  let bracket = power(plus(int(1), x_over_b_to_a), minus(int(-1), p));
  let b_to_ap = power(b, ap.clone());
  let density = divide(times(times(ap, x_to_ap_minus_1), bracket), b_to_ap);

  // x > 0
  let cond = call("Greater", vec![x, int(0)]);

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[HyperbolicDistribution[a, b, d, m], x]
/// = Sqrt[a^2 - b^2] * E^(b*(x-m) - a*Sqrt[d^2 + (x-m)^2]) / (2*a*d*BesselK[1, Sqrt[a^2-b^2]*d])
fn pdf_hyperbolic(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 4 {
    return Err(InterpreterError::EvaluationError(
      "HyperbolicDistribution expects 4 arguments".into(),
    ));
  }
  let a = dargs[0].clone();
  let b = dargs[1].clone();
  let d = dargs[2].clone();
  let m = dargs[3].clone();

  let besselk = |n: Expr, z: Expr| call("BesselK", vec![n, z]);

  let x_minus_m = minus(x, m);
  let a2_minus_b2 = minus(power(a.clone(), int(2)), power(b.clone(), int(2)));
  let sqrt_a2_minus_b2 = sqrt(a2_minus_b2);

  // numerator: Sqrt[a^2 - b^2] * E^(b*(x-m) - a*Sqrt[d^2 + (x-m)^2])
  let exponent = minus(
    times(b, x_minus_m.clone()),
    times(
      a.clone(),
      sqrt(plus(power(d.clone(), int(2)), power(x_minus_m, int(2)))),
    ),
  );
  let numerator = times(sqrt_a2_minus_b2.clone(), power(e(), exponent));

  // denominator: 2*a*d*BesselK[1, Sqrt[a^2-b^2]*d]
  let denominator = times(
    times(int(2), times(a, d.clone())),
    besselk(int(1), times(sqrt_a2_minus_b2, d)),
  );

  eval(divide(numerator, denominator))
}

/// PDF[NoncentralFRatioDistribution[n, m, l], x]
/// = m^(m/2)*n^(n/2)*x^((n-2)/2)*(m+n*x)^(-(m+n)/2)
///   *Hypergeometric1F1[(m+n)/2, n/2, l*n*x/(2*(m+n*x))]
///   / (E^(l/2)*Beta[n/2, m/2])
/// for x > 0, else 0
fn pdf_noncentral_f(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "NoncentralFRatioDistribution expects 3 arguments".into(),
    ));
  }
  let n = dargs[0].clone();
  let m = dargs[1].clone();
  let l = dargs[2].clone();

  let half = |e: Expr| divide(e, int(2));

  let beta = |a: Expr, b: Expr| call("Beta", vec![a, b]);

  let hyp1f1 =
    |a: Expr, b: Expr, z: Expr| call("Hypergeometric1F1", vec![a, b, z]);

  // numerator pieces
  let m_half = power(m.clone(), half(m.clone()));
  let n_half = power(n.clone(), half(n.clone()));
  let x_pow = power(x.clone(), half(plus(int(-2), n.clone())));
  let bracket = power(
    plus(m.clone(), times(n.clone(), x.clone())),
    half(plus(times(int(-1), m.clone()), times(int(-1), n.clone()))),
  );
  let hyp = hyp1f1(
    half(plus(m.clone(), n.clone())),
    half(n.clone()),
    divide(
      times(times(l.clone(), n), x.clone()),
      times(int(2), plus(m.clone(), times(dargs[0].clone(), x.clone()))),
    ),
  );

  let numerator =
    times(times(times(m_half, n_half), times(x_pow, bracket)), hyp);

  // denominator: E^(l/2) * Beta[n/2, m/2]
  let denominator =
    times(power(e(), half(l)), beta(half(dargs[0].clone()), half(m)));

  let density = divide(numerator, denominator);

  let cond = comparison(x, ComparisonOp::Greater, int(0));

  eval(piecewise(vec![(density, cond)], int(0)))
}

/// PDF[JohnsonDistribution["type", gamma, delta, mu, sigma], x]
fn pdf_johnson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 5 {
    return Err(InterpreterError::EvaluationError(
      "JohnsonDistribution expects 5 arguments".into(),
    ));
  }
  let type_str = match &dargs[0] {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(call(
        "PDF",
        vec![unevaluated("JohnsonDistribution", dargs), x],
      ));
    }
  };
  let gamma = dargs[1].clone();
  let delta = dargs[2].clone();
  let mu = dargs[3].clone();
  let sigma = dargs[4].clone();

  // (-mu + x): Wolfram canonical ordering
  let neg_mu_plus_x = plus(neg(mu.clone()), x.clone());

  match type_str.as_str() {
    "SN" => {
      // PDF = delta / (E^(z^2/2) * Sqrt[2*Pi] * sigma)
      // where z = gamma + delta*(-mu + x)/sigma
      let z = plus(
        gamma,
        divide(times(delta.clone(), neg_mu_plus_x), sigma.clone()),
      );
      let density = divide(
        delta,
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          sigma,
        ),
      );
      eval(density)
    }
    "SL" => {
      // PDF = delta / (E^(z^2/2) * Sqrt[2*Pi] * (-mu + x))
      // where z = gamma + delta*Log[(-mu + x)/sigma], for x > mu
      let log_t = call("Log", vec![divide(neg_mu_plus_x.clone(), sigma)]);
      let z = plus(gamma, times(delta.clone(), log_t));
      let density = divide(
        delta,
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          neg_mu_plus_x,
        ),
      );
      let cond = comparison(x, ComparisonOp::Greater, mu);
      eval(piecewise(vec![(density, cond)], int(0)))
    }
    "SU" => {
      // PDF = delta / (E^(z^2/2) * Sqrt[2*Pi] * Sqrt[sigma^2 + (-mu + x)^2])
      // where z = gamma + delta*ArcSinh[(-mu + x)/sigma]
      let arcsinh_t = call(
        "ArcSinh",
        vec![divide(neg_mu_plus_x.clone(), sigma.clone())],
      );
      let z = plus(gamma, times(delta.clone(), arcsinh_t));
      let density = divide(
        delta,
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          sqrt(plus(power(sigma, int(2)), power(neg_mu_plus_x, int(2)))),
        ),
      );
      eval(density)
    }
    "SB" => {
      // PDF = (delta*sigma) / (E^(z^2/2) * Sqrt[2*Pi] * (mu+sigma-x) * (-mu+x))
      // where z = gamma + delta*Log[(-mu+x)/(mu+sigma-x)], for mu < x < mu+sigma
      let mu_plus_sigma_minus_x =
        plus(plus(mu.clone(), sigma.clone()), neg(x.clone()));
      let log_arg =
        divide(neg_mu_plus_x.clone(), mu_plus_sigma_minus_x.clone());
      let log_t = call("Log", vec![log_arg]);
      let z = plus(gamma, times(delta.clone(), log_t));
      let density = divide(
        times(delta, sigma.clone()),
        times(
          times(
            power(e(), divide(power(z, int(2)), int(2))),
            sqrt(times(int(2), pi())),
          ),
          times(mu_plus_sigma_minus_x, neg_mu_plus_x),
        ),
      );
      let cond = comparison3(
        mu.clone(),
        ComparisonOp::Less,
        x.clone(),
        ComparisonOp::Less,
        plus(mu, sigma),
      );
      // The density is built in Wolfram's canonical factor order
      // ((mu + sigma - x)*(-mu + x)), which Woxi's evaluator would
      // reorder; keep the raw form for a symbolic x and only evaluate
      // (to fold the branch) for concrete arguments.
      if matches!(&x, Expr::Identifier(_)) {
        Ok(piecewise(vec![(density, cond)], int(0)))
      } else {
        eval(piecewise(vec![(density, cond)], int(0)))
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "JohnsonDistribution: unknown type {type_str}"
    ))),
  }
}

/// CDF[JohnsonDistribution["type", gamma, delta, mu, sigma], x]
fn cdf_johnson(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  if dargs.len() != 5 {
    return Err(InterpreterError::EvaluationError(
      "JohnsonDistribution expects 5 arguments".into(),
    ));
  }
  let type_str = match &dargs[0] {
    Expr::String(s) => s.clone(),
    _ => {
      return Ok(call(
        "CDF",
        vec![unevaluated("JohnsonDistribution", dargs), x],
      ));
    }
  };
  let gamma = dargs[1].clone();
  let delta = dargs[2].clone();
  let mu = dargs[3].clone();
  let sigma = dargs[4].clone();

  // t = (-mu + x) / sigma (canonical ordering)
  let neg_mu_plus_x = plus(neg(mu.clone()), x.clone());
  let t = divide(neg_mu_plus_x.clone(), sigma.clone());

  // CDF = Erfc[(-gamma - delta*h(t)) / Sqrt[2]] / 2
  // where h depends on type
  let h_of_t = match type_str.as_str() {
    "SN" => t.clone(),
    "SL" => call("Log", vec![t.clone()]),
    "SU" => call("ArcSinh", vec![t.clone()]),
    "SB" => {
      let mu_plus_sigma_minus_x =
        plus(plus(mu.clone(), sigma.clone()), neg(x.clone()));
      call("Log", vec![divide(neg_mu_plus_x, mu_plus_sigma_minus_x)])
    }
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "JohnsonDistribution: unknown type {type_str}"
      )));
    }
  };

  // Distribute negative sign: (-gamma - delta*h) / Sqrt[2]
  let neg_gamma = neg(gamma.clone());
  let neg_delta_h = neg(times(delta.clone(), h_of_t.clone()));
  let erfc_arg = divide(plus(neg_gamma, neg_delta_h), sqrt(int(2)));
  let cdf_val = divide(call("Erfc", vec![erfc_arg]), int(2));

  // Also build (1 + Erf[(gamma + delta*h) / Sqrt[2]]) / 2 form (used by Wolfram for some types)
  let erf_arg = divide(plus(gamma, times(delta, h_of_t)), sqrt(int(2)));
  let cdf_erf = divide(plus(int(1), call("Erf", vec![erf_arg])), int(2));

  match type_str.as_str() {
    "SN" => eval(cdf_val),
    "SU" => eval(cdf_erf), // Wolfram uses Erf form for SU
    "SL" => {
      // Wolfram splits into two regions based on mu+sigma threshold
      let cond_lower = Expr::Comparison {
        operands: vec![mu.clone(), x.clone(), plus(mu.clone(), sigma.clone())],
        operators: vec![ComparisonOp::Less, ComparisonOp::LessEqual],
      };
      let cond_upper = comparison(x, ComparisonOp::Greater, plus(mu, sigma));
      eval(piecewise(
        vec![(cdf_val, cond_lower), (cdf_erf, cond_upper)],
        int(0),
      ))
    }
    "SB" => {
      // Wolfram splits SB CDF into multiple regions
      let half_sigma = divide(sigma.clone(), int(2));
      let cond1 = comparison3(
        mu.clone(),
        ComparisonOp::Less,
        x.clone(),
        ComparisonOp::Less,
        plus(mu.clone(), half_sigma.clone()),
      );
      let cond2 = Expr::Comparison {
        operands: vec![
          plus(mu.clone(), half_sigma),
          x.clone(),
          plus(mu.clone(), sigma.clone()),
        ],
        operators: vec![ComparisonOp::LessEqual, ComparisonOp::Less],
      };
      let cond3 = comparison(x, ComparisonOp::GreaterEqual, plus(mu, sigma));
      eval(piecewise(
        vec![(cdf_val, cond1), (cdf_erf, cond2), (int(1), cond3)],
        int(0),
      ))
    }
    _ => eval(cdf_val),
  }
}

/// Return the per-variable marginal distribution for a known
/// multivariate distribution. Standard `BinormalDistribution[ρ]` and
/// `BinormalDistribution[{μ₁, μ₂}, {σ₁, σ₂}, ρ]` are supported (both
/// yield Normal marginals).
fn multivariate_marginals(
  dist_name: &str,
  dargs: &[Expr],
  vars: &[String],
) -> Option<Vec<Expr>> {
  let _ = vars; // matched only on dist shape currently
  match dist_name {
    "BinormalDistribution" => match dargs.len() {
      // BinormalDistribution[ρ]: standard normals with correlation ρ.
      1 => Some(vec![standard_normal(), standard_normal()]),
      // BinormalDistribution[{μ₁, μ₂}, {σ₁, σ₂}, ρ].
      3 => {
        let means = match &dargs[0] {
          Expr::List(items) if items.len() == 2 => items.clone(),
          _ => return None,
        };
        let sigmas = match &dargs[1] {
          Expr::List(items) if items.len() == 2 => items.clone(),
          _ => return None,
        };
        Some(vec![
          normal_distribution(&means[0], &sigmas[0]),
          normal_distribution(&means[1], &sigmas[1]),
        ])
      }
      _ => None,
    },
    _ => None,
  }
}

fn standard_normal() -> Expr {
  normal_distribution(&Expr::Integer(0), &Expr::Integer(1))
}

fn normal_distribution(mu: &Expr, sigma: &Expr) -> Expr {
  call("NormalDistribution", vec![mu.clone(), sigma.clone()])
}

/// Collect the additive terms of `expr` and dispatch each to a
/// univariate Expectation against its marginal. Returns `None` when
/// any term touches two or more distribution variables (since that
/// requires the joint pdf — not yet implemented).
fn try_separable_multivariate_expectation(
  expr: &Expr,
  vars: &[String],
  marginals: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let terms = additive_terms(expr);
  let mut acc: Vec<Expr> = Vec::new();
  for term in terms {
    let var_hits: Vec<usize> = vars
      .iter()
      .enumerate()
      .filter(|(_, v)| contains_variable(&term, v))
      .map(|(i, _)| i)
      .collect();
    if var_hits.len() > 1 {
      return Ok(None);
    }
    let contrib = if var_hits.is_empty() {
      term
    } else {
      let i = var_hits[0];
      let inner = expectation_ast(&[
        term,
        call(
          "Distributed",
          vec![Expr::Identifier(vars[i].clone()), marginals[i].clone()],
        ),
      ])?;
      // If the inner call returned unevaluated, the whole separable
      // path can't simplify cleanly — back out so the caller leaves
      // the outer Expectation symbolic.
      if matches!(
        &inner,
        Expr::FunctionCall { name, .. } if name == "Expectation"
      ) {
        return Ok(None);
      }
      inner
    };
    acc.push(contrib);
  }
  if acc.is_empty() {
    return Ok(Some(Expr::Integer(0)));
  }
  let summed = if acc.len() == 1 {
    acc.remove(0)
  } else {
    call("Plus", acc)
  };
  crate::evaluator::evaluate_expr_to_expr(&summed).map(Some)
}

fn additive_terms(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut out = Vec::new();
      for arg in args.iter() {
        out.extend(additive_terms(arg));
      }
      out
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut out = additive_terms(left);
      out.extend(additive_terms(right));
      out
    }
    other => vec![other.clone()],
  }
}

fn contains_variable(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(n) => n == var,
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| contains_variable(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_variable(left, var) || contains_variable(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_variable(operand, var),
    Expr::List(items) => items.iter().any(|a| contains_variable(a, var)),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| contains_variable(a, var))
    }
    _ => false,
  }
}

// ─── LogLikelihood ───────────────────────────────────────────────────

/// LogLikelihood[dist, {x1, x2, ...}] — the sum of Log[PDF[dist, x_i]]
/// in wolframscript's per-distribution closed form. Supports numeric
/// observations for Exponential, Poisson, Bernoulli, and Normal
/// distributions; symbolic observations (which wolframscript wraps in
/// domain Piecewise conditions) stay unevaluated.
pub fn log_likelihood_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let uneval = || unevaluated("LogLikelihood", args);
  if args.len() != 2 {
    return Ok(uneval());
  }
  let (dist_name, dargs) = match &args[0] {
    Expr::FunctionCall { name, args } => (name.as_str(), args.as_slice()),
    _ => return Ok(uneval()),
  };
  let data = match &args[1] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return Ok(uneval()),
  };
  let is_numeric = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::Real(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  if !data.iter().all(is_numeric) {
    return Ok(uneval());
  }
  let n = data.len() as i128;

  let log_of = |e: Expr| call("Log", vec![e]);
  let sum_expr =
    || -> Result<Expr, InterpreterError> { eval(unevaluated("Plus", data)) };

  match (dist_name, dargs) {
    // n*Log[a] - a*Sum[x]
    ("ExponentialDistribution", [a]) => {
      let total = sum_expr()?;
      eval(plus(
        times(int(-1), times(total, a.clone())),
        times(int(n), log_of(a.clone())),
      ))
    }
    // -n*m + Sum[x]*Log[m] - Sum[Log[x!]]
    ("PoissonDistribution", [m]) => {
      if !data
        .iter()
        .all(|e| matches!(e, Expr::Integer(v) if *v >= 0))
      {
        return Ok(uneval());
      }
      let total = sum_expr()?;
      let mut terms =
        vec![times(int(-n), m.clone()), times(total, log_of(m.clone()))];
      for x in data {
        let fact = eval(factorial(x.clone()))?;
        if !matches!(fact, Expr::Integer(1)) {
          terms.push(times(int(-1), log_of(fact)));
        }
      }
      eval(call("Plus", terms))
    }
    // zeros*Log[1 - p] + ones*Log[p] (raw: the evaluator would reorder
    // the Plus terms away from wolframscript's Log[1 - p] + 3*Log[p])
    ("BernoulliDistribution", [p]) => {
      let ones = data
        .iter()
        .filter(|e| matches!(e, Expr::Integer(1)))
        .count() as i128;
      let zeros = data
        .iter()
        .filter(|e| matches!(e, Expr::Integer(0)))
        .count() as i128;
      if ones + zeros != n {
        return Ok(uneval());
      }
      // Evaluate each term individually but keep the raw Plus order
      // (Log[1 - p] first): full evaluation reorders the sum away from
      // wolframscript's Log[1 - p] + 3*Log[p] / -Log[3/2] - 3*Log[3]
      let log_q = eval(log_of(plus(int(1), times(int(-1), p.clone()))))?;
      let log_p = eval(log_of(p.clone()))?;
      let term = |c: i128, l: Expr| -> Result<Expr, InterpreterError> {
        match c {
          1 => Ok(l),
          c => eval(times(int(c), l)),
        }
      };
      Ok(match (zeros, ones) {
        (0, o) => term(o, log_p)?,
        (z, 0) => term(z, log_q)?,
        (z, o) => plus(term(z, log_q)?, term(o, log_p)?),
      })
    }
    // -1/2*(Sum[(x - m)^2] expanded)/s^2 - n*((Log[2] + Log[Pi])/2 + Log[s])
    ("NormalDistribution", params)
      if params.is_empty() || params.len() == 2 =>
    {
      let (m, s) = if params.is_empty() {
        (int(0), int(1))
      } else {
        (params[0].clone(), params[1].clone())
      };
      // Sum[(x_i - m)^2] expanded in ascending powers of m:
      // Sum[x^2] - 2*Sum[x]*m + n*m^2
      let sq_total = eval(call(
        "Plus",
        data
          .iter()
          .map(|x| times(x.clone(), x.clone()))
          .collect::<Vec<_>>(),
      ))?;
      let total = sum_expr()?;
      let poly = plus(
        plus(sq_total, times(times(int(-2), total), m.clone())),
        times(int(n), times(m.clone(), m.clone())),
      );
      let half_log_2pi = divide(plus(log_of(int(2)), log_of(pi())), int(2));
      let result = plus(
        times(
          crate::functions::math_ast::make_rational(-1, 2),
          divide(poly, times(s.clone(), s.clone())),
        ),
        times(int(-n), plus(half_log_2pi, log_of(s.clone()))),
      );
      eval(result)
    }
    _ => Ok(uneval()),
  }
}

// ─── Exact polynomial moments for Expectation ────────────────────────

/// Decompose `expr` as a polynomial in `var` with var-free coefficients:
/// returns (k, coeff) pairs. None when any term isn't c·var^k.
fn extract_polynomial_in_var(
  expr: &Expr,
  var: &str,
) -> Option<Vec<(i128, Expr)>> {
  let terms: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().cloned().collect()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    _ => vec![expr.clone()],
  };
  let mut out: Vec<(i128, Expr)> = Vec::with_capacity(terms.len());
  for term in &terms {
    let (c, rest) = strip_constant_multiplier(term, var);
    let k = match &rest {
      Expr::Integer(1) => 0,
      Expr::Identifier(v) if v == var => 1,
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[0], Expr::Identifier(v) if v == var) =>
      {
        match &args[1] {
          Expr::Integer(k) if *k >= 1 => *k,
          _ => return None,
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(left.as_ref(), Expr::Identifier(v) if v == var) => {
        match right.as_ref() {
          Expr::Integer(k) if *k >= 1 => *k,
          _ => return None,
        }
      }
      _ if !contains_variable(&rest, var) => {
        // Fully constant term: fold rest into the coefficient
        out.push((0, times(c, rest.clone())));
        continue;
      }
      _ => return None,
    };
    if !(0..=12).contains(&k) {
      return None;
    }
    out.push((k, c));
  }
  Some(out)
}

/// The k-th raw moment E[x^k] of a supported distribution, in
/// wolframscript's exact output form. Symbolic parameters return raw
/// (unevaluated) structures where evaluation would reshuffle the
/// factored shapes; numeric parameters evaluate.
fn distribution_raw_moment(
  dist_name: &str,
  dargs: &[Expr],
  k: i128,
) -> Option<Expr> {
  if k == 0 {
    return Some(int(1));
  }
  let numeric = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::Real(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };

  match dist_name {
    // E[x^k] = p for every k >= 1: the support is {0, 1}, so x^k = x.
    // (k == 0 is handled above, returning 1.)
    "BernoulliDistribution" if dargs.len() == 1 => Some(dargs[0].clone()),
    // E[x^k] = k!/lambda^k
    "ExponentialDistribution" if dargs.len() == 1 => {
      eval(divide(int(fact(k)), power(dargs[0].clone(), int(k)))).ok()
    }
    // E[x^k] = (a^k + a^(k-1) b + ... + b^k)/(k + 1)
    "UniformDistribution" if dargs.len() == 1 => {
      let (a, b) = match &dargs[0] {
        Expr::List(bounds) if bounds.len() == 2 => {
          (bounds[0].clone(), bounds[1].clone())
        }
        _ => return None,
      };
      let term = |i: i128| -> Expr {
        // a^i * b^(k-i), omitting exponent-0/1 noise
        let pow_of = |base: &Expr, e: i128| match e {
          0 => None,
          1 => Some(base.clone()),
          e => Some(power(base.clone(), int(e))),
        };
        match (pow_of(&a, i), pow_of(&b, k - i)) {
          (Some(x), Some(y)) => times(x, y),
          (Some(x), None) => x,
          (None, Some(y)) => y,
          (None, None) => int(1),
        }
      };
      let sum = call("Plus", (0..=k).rev().map(term).collect::<Vec<_>>());
      let result = divide(sum, int(k + 1));
      if numeric(&a) && numeric(&b) {
        eval(result).ok()
      } else {
        Some(result)
      }
    }
    // E[x^k] = Sum[Binomial[k, j]*(j-1)!!*m^(k-j)*s^j, even j]
    "NormalDistribution" if dargs.is_empty() || dargs.len() == 2 => {
      let (m, s) = if dargs.is_empty() {
        (int(0), int(1))
      } else {
        (dargs[0].clone(), dargs[1].clone())
      };
      let all_numeric = numeric(&m) && numeric(&s);
      if !all_numeric {
        // Symbolic parameters: wolframscript's templates for k <= 4
        return match k {
          1 => Some(m),
          2 => {
            eval(plus(power(s.clone(), int(2)), power(m.clone(), int(2)))).ok()
          }
          // m*(m^2 + 3*s^2), kept raw so Times isn't distributed
          3 => Some(times(
            m.clone(),
            plus(
              power(m.clone(), int(2)),
              times(int(3), power(s.clone(), int(2))),
            ),
          )),
          // m^4 + 6*m^2*s^2 + 3*s^4
          4 => Some(plus(
            plus(
              power(m.clone(), int(4)),
              times(
                int(6),
                times(power(m.clone(), int(2)), power(s.clone(), int(2))),
              ),
            ),
            times(int(3), power(s.clone(), int(4))),
          )),
          _ => None,
        };
      }
      // Numeric parameters: exact binomial sum over central moments
      let pow_term = |base: &Expr, e: i128| -> Option<Expr> {
        // Omit unit factors — Power[0, 0] would be Indeterminate
        match e {
          0 => None,
          1 => Some(base.clone()),
          e => Some(power(base.clone(), int(e))),
        }
      };
      let mut terms: Vec<Expr> = Vec::new();
      let mut c_kj = 1i128; // binomial C(k,j)
      let mut fact2 = 1i128; // double factorial, (-1)! = 1
      for j in (0..=k).step_by(2) {
        if j != 0 {
          // C(k,j) = C(k,j-2) (k-j+2)/(j-1) (k-j+1)/j
          c_kj = (c_kj * (k - j + 2)) / (j - 1);
          c_kj = (c_kj * (k - j + 1)) / j;
          fact2 *= j - 1; // j = 2, 4, 6, ...
        }

        let c = c_kj * fact2;
        let mut factors = vec![int(c)];
        factors.extend(pow_term(&m, k - j));
        factors.extend(pow_term(&s, j));
        terms.push(call("Times", factors));
      }
      eval(call("Plus", terms)).ok()
    }
    // E[x^k] = a*(1 + a)*...*(k - 1 + a)*b^k
    "GammaDistribution" if dargs.len() == 2 => {
      let (a, b) = (dargs[0].clone(), dargs[1].clone());
      let mut factors: Vec<Expr> = vec![a.clone()];
      for i in 1..k {
        factors.push(plus(int(i), a.clone()));
      }
      factors.push(power(b.clone(), int(k)));
      let result = call("Times", factors);
      if numeric(&a) && numeric(&b) {
        eval(result).ok()
      } else {
        Some(result)
      }
    }
    // E[x^k] = Sum[StirlingS2[k, j]*mu^j] (Touchard polynomial)
    "PoissonDistribution" if dargs.len() == 1 => {
      let mu = dargs[0].clone();
      // Stirling numbers of the second kind via the triangle recurrence
      let mut s2 = vec![vec![0i128; (k + 1) as usize]; (k + 1) as usize];
      s2[0][0] = 1;
      for n in 1..=(k as usize) {
        for j in 1..=n {
          s2[n][j] = s2[n - 1][j - 1] + (j as i128) * s2[n - 1][j];
        }
      }
      let terms: Vec<Expr> = (1..=(k as usize))
        .map(|j| {
          let c = s2[k as usize][j];
          let p = match j {
            1 => mu.clone(),
            j => power(mu.clone(), int(j as i128)),
          };
          if c == 1 { p } else { times(int(c), p) }
        })
        .collect();
      eval(call("Plus", terms)).ok()
    }
    // ChiSquareDistribution[nu] = GammaDistribution[nu/2, 2], so
    // E[x^k] = 2^k*Pochhammer[nu/2, k] = Product_{i=0}^{k-1} (nu + 2 i).
    // wolframscript prints this as nu*(2 + nu)*(4 + nu)*...*(2(k-1) + nu).
    "ChiSquareDistribution" if dargs.len() == 1 => {
      let nu = dargs[0].clone();
      let mut factors: Vec<Expr> = Vec::with_capacity(k as usize);
      for i in 0..k {
        factors.push(if i == 0 {
          nu.clone()
        } else {
          plus(int(2 * i), nu.clone())
        });
      }
      let result = call("Times", factors);
      if numeric(&nu) {
        eval(result).ok()
      } else {
        Some(result)
      }
    }
    _ => None,
  }
}

/// Exact expectation of a polynomial integrand: Sum[c_k * E[x^k]].
/// Single power terms with unit coefficient return the moment verbatim
/// (preserving raw factored templates); combinations evaluate.
fn polynomial_expectation(
  expr: &Expr,
  var: &str,
  dist_name: &str,
  dargs: &[Expr],
) -> Option<Expr> {
  let poly = extract_polynomial_in_var(expr, var)?;
  if poly.len() == 1 && matches!(&poly[0].1, Expr::Integer(1)) {
    return distribution_raw_moment(dist_name, dargs, poly[0].0);
  }
  let mut terms: Vec<Expr> = Vec::with_capacity(poly.len());
  for (k, c) in &poly {
    let moment = distribution_raw_moment(dist_name, dargs, *k)?;
    terms.push(times(c.clone(), moment));
  }
  eval(call("Plus", terms)).ok()
}

// ─── TransformedDistribution ─────────────────────────────────────────

/// TransformedDistribution[a*x + b, x \[Distributed] dist] for numeric
/// rational a (nonzero) and b, folding linear transforms into the
/// distribution families wolframscript uses:
/// - NormalDistribution[m, s] -> NormalDistribution[b + a*m, |a|*s]
/// - UniformDistribution[{lo, hi}] -> sorted {a*lo + b, a*hi + b}
/// - ExponentialDistribution[l] (a > 0, b = 0) -> ExponentialDistribution[l/a]
/// - GammaDistribution[al, be] (a > 0, b = 0) -> GammaDistribution[al, a*be]
pub fn transformed_distribution_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated =
    |args: &[Expr]| unevaluated("TransformedDistribution", args);
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let (var, dist) = match &args[1] {
    Expr::FunctionCall { name, args: dargs }
      if name == "Distributed" && dargs.len() == 2 =>
    {
      match &dargs[0] {
        Expr::Identifier(v) => (v.clone(), dargs[1].clone()),
        _ => return Ok(unevaluated(args)),
      }
    }
    _ => return Ok(unevaluated(args)),
  };
  let (dist_name, dargs) = match &dist {
    Expr::FunctionCall { name, args: da } => (name.as_str(), da.as_slice()),
    _ => return Ok(unevaluated(args)),
  };

  // Linear transform a*x + b with exact numeric a (nonzero) and b.
  // extract_linear returns unevaluated Plus chains — fold them first.
  let (a, b) = match extract_linear(&args[0], &var) {
    Some((a, b)) => (eval(a)?, eval(b)?),
    None => return Ok(unevaluated(args)),
  };
  let as_frac = |e: &Expr| -> Option<(i128, i128)> {
    match e {
      Expr::Integer(n) => Some((*n, 1)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n, *d))
        } else {
          None
        }
      }
      _ => None,
    }
  };
  let (Some(af), Some(bf)) = (as_frac(&a), as_frac(&b)) else {
    return Ok(unevaluated(args));
  };
  if af.0 == 0 {
    return Ok(unevaluated(args));
  }
  let a_positive = af.0 > 0;

  let dist_call = |name: &str, params: Vec<Expr>| call(name, params);
  let linear = |e: &Expr| -> Result<Expr, InterpreterError> {
    // a*e + b
    eval(plus(times(a.clone(), e.clone()), b.clone()))
  };

  match (dist_name, dargs) {
    ("NormalDistribution", [m, s]) => {
      let new_m = linear(m)?;
      let abs_a = frac_to_rational_expr((af.0.abs(), af.1));
      let new_s = eval(times(abs_a, s.clone()))?;
      Ok(dist_call("NormalDistribution", vec![new_m, new_s]))
    }
    ("NormalDistribution", []) => {
      let new_m = eval(b.clone())?;
      let new_s = frac_to_rational_expr((af.0.abs(), af.1));
      Ok(dist_call("NormalDistribution", vec![new_m, new_s]))
    }
    ("UniformDistribution", [Expr::List(bounds)]) if bounds.len() == 2 => {
      let p = linear(&bounds[0])?;
      let q = linear(&bounds[1])?;
      let (lo, hi) = if a_positive { (p, q) } else { (q, p) };
      Ok(dist_call(
        "UniformDistribution",
        vec![Expr::List(vec![lo, hi].into())],
      ))
    }
    ("ExponentialDistribution", [l]) if a_positive && bf.0 == 0 => {
      let new_l = eval(divide(l.clone(), a.clone()))?;
      Ok(dist_call("ExponentialDistribution", vec![new_l]))
    }
    ("GammaDistribution", [al, be]) if a_positive && bf.0 == 0 => {
      let new_be = eval(times(a.clone(), be.clone()))?;
      Ok(dist_call("GammaDistribution", vec![al.clone(), new_be]))
    }
    _ => Ok(unevaluated(args)),
  }
}

fn frac_to_rational_expr(f: (i128, i128)) -> Expr {
  if f.1 == 1 {
    Expr::Integer(f.0)
  } else {
    crate::functions::math_ast::make_rational(f.0, f.1)
  }
}

// ─── MultinormalDistribution ─────────────────────────────────────────

/// PDF[MultinormalDistribution[mu, sigma], {v...}] for diagonal
/// covariance matrices, in wolframscript's printed form. The exponent's
/// term styles are position-dependent (the first non-unit-variance term
/// prints as -1/q*v^2, later ones as -v^2/q), so the structure mirrors
/// what the parser builds for those strings.
fn pdf_multinormal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    call(
      "PDF",
      vec![unevaluated("MultinormalDistribution", dargs), x.clone()],
    )
  };
  let (mu, sigma) = match dargs {
    [Expr::List(mu), Expr::List(rows)] => (mu, rows),
    _ => return Ok(unevaluated()),
  };
  let vars = match &x {
    Expr::List(vars) if vars.len() == mu.len() => vars,
    _ => return Ok(unevaluated()),
  };
  let k = mu.len();
  if !(2..=3).contains(&k) || sigma.len() != k {
    return Ok(unevaluated());
  }

  // Diagonal covariance with positive integer variances
  let mut variances: Vec<i128> = Vec::with_capacity(k);
  for (i, row) in sigma.iter().enumerate() {
    let cols = match row {
      Expr::List(cols) if cols.len() == k => cols,
      _ => return Ok(unevaluated()),
    };
    for (j, e) in cols.iter().enumerate() {
      match e {
        Expr::Integer(v) if i == j && *v >= 1 => variances.push(*v),
        Expr::Integer(0) if i != j => {}
        _ => return Ok(unevaluated()),
      }
    }
  }

  let pow2 = |e: Expr| power(e, int(2));
  let div2 = |a: Expr, b: Expr| divide(a, b);

  // (v - mu): canonical (-mu + v), or just v for a zero mean
  let centered = |i: usize| -> Result<Expr, InterpreterError> {
    if matches!(&mu[i], Expr::Integer(0)) {
      Ok(vars[i].clone())
    } else {
      eval(plus(times(int(-1), mu[i].clone()), vars[i].clone()))
    }
  };

  let mut terms: Vec<Expr> = Vec::with_capacity(k);
  let mut first_scaled_seen = false;
  for i in 0..k {
    let sq = pow2(centered(i)?);
    let term = if variances[i] == 1 {
      neg(sq)
    } else if !first_scaled_seen {
      first_scaled_seen = true;
      times(
        crate::functions::math_ast::make_rational(-1, variances[i]),
        sq,
      )
    } else {
      neg(div2(sq, int(variances[i])))
    };
    terms.push(term);
  }
  let exponent = div2(call("Plus", terms), int(2));
  let e_pow = power(Expr::Identifier("E".to_string()), exponent);

  // Normalizer: (2*Pi)^(k/2) * Sqrt[det]
  let det: i128 = variances.iter().product();
  let denominator = if k == 2 {
    // 2*Pi, 2*Sqrt[det]*Pi, or (2*sqrt)*Pi when det is a perfect square
    let sqrt_det = eval(call("Sqrt", vec![int(det)]))?;
    match &sqrt_det {
      Expr::Integer(s) => times(int(2 * s), pi()),
      _ => call("Times", vec![int(2), sqrt_det, pi()]),
    }
  } else {
    // k == 3: 2*Sqrt[2*det]*Pi^(3/2)
    let sqrt_part = eval(call("Sqrt", vec![int(2 * det)]))?;
    let pi_pow = power(pi(), crate::functions::math_ast::make_rational(3, 2));
    match &sqrt_part {
      Expr::Integer(s) => call("Times", vec![int(2 * s), pi_pow]),
      _ => call("Times", vec![int(2), sqrt_part, pi_pow]),
    }
  };
  let result = div2(e_pow, denominator);

  // The form above is built to mirror wolframscript's printed PDF for a
  // symbolic point ({x, y} → E^((-x^2 - y^2)/2)/(2*Pi)). At a fully numeric
  // point it still contains unevaluated pieces such as 0^2 or 1^2, so reduce
  // it to a closed numeric value (e.g. {0, 0} → 1/(2*Pi)) to match WL.
  let point_is_numeric = vars.iter().all(|v| {
    matches!(
      v,
      Expr::Integer(_)
        | Expr::BigInteger(_)
        | Expr::Real(_)
        | Expr::BigFloat(..)
    ) || matches!(v, Expr::FunctionCall { name, .. } if name == "Rational")
  });
  if point_is_numeric {
    return eval(result);
  }
  Ok(result)
}

// ─── EmpiricalDistribution / DataDistribution ────────────────────────

fn as_exact_frac(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

fn frac_cmp(a: (i128, i128), b: (i128, i128)) -> std::cmp::Ordering {
  (a.0 * b.1).cmp(&(b.0 * a.1))
}

/// EmpiricalDistribution[{data}] — wolframscript's container:
/// DataDistribution["Empirical", {weights, sorted values, False}, 1, n]
pub fn empirical_distribution_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("EmpiricalDistribution", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }
  let data = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return Ok(unevaluated(args)),
  };
  let mut values: Vec<(i128, i128)> = Vec::with_capacity(data.len());
  for e in data {
    match as_exact_frac(e) {
      Some(f) => values.push(f),
      None => return Ok(unevaluated(args)),
    }
  }
  values.sort_by(|a, b| frac_cmp(*a, *b));
  let n = values.len() as i128;
  let mut weights: Vec<Expr> = Vec::new();
  let mut uniques: Vec<Expr> = Vec::new();
  let mut i = 0usize;
  while i < values.len() {
    let v = values[i];
    let mut count = 0i128;
    while i < values.len() && values[i] == v {
      count += 1;
      i += 1;
    }
    weights.push(eval(divide(int(count), int(n)))?);
    uniques.push(if v.1 == 1 {
      Expr::Integer(v.0)
    } else {
      crate::functions::math_ast::make_rational(v.0, v.1)
    });
  }
  Ok(call(
    "DataDistribution",
    vec![
      Expr::String("Empirical".to_string()),
      Expr::List(
        vec![
          Expr::List(weights.into()),
          Expr::List(uniques.into()),
          bool_expr(false),
        ]
        .into(),
      ),
      Expr::Integer(1),
      Expr::Integer(n),
    ],
  ))
}

// ─── HistogramDistribution ────────────────────────────────────────────

fn frac_reduce(mut n: i128, mut d: i128) -> (i128, i128) {
  if d < 0 {
    n = -n;
    d = -d;
  }
  let mut a = n.abs().max(1);
  let mut b = d;
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  let g = a.max(1);
  (n / g, d / g)
}

fn frac_expr((n, d): (i128, i128)) -> Expr {
  if d == 1 {
    Expr::Integer(n)
  } else {
    crate::functions::math_ast::make_rational(n, d)
  }
}

/// `HistogramDistribution[data]` / `HistogramDistribution[data, {w}]` —
/// wolframscript's container `DataDistribution["Histogram",
/// {densities, bin edges}, 1, n]`. The bins follow the same rules as
/// `HistogramList`: automatic binning (and an explicit width whose bins are
/// *centered* on commensurate data) reports machine-real densities and
/// edges, while an explicit exact width anchored at multiples of `w` stays
/// exact (`{{1/10, 3/10, 1/10}, {0, 2, 4, 6}}`). The densities are
/// count/(n·w), so empty interior bins carry weight `0.`.
pub fn histogram_distribution_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("HistogramDistribution", args));
  let (data, width_spec) = match args {
    [Expr::List(d)] if !d.is_empty() => (d, None),
    [Expr::List(d), Expr::List(spec)] if !d.is_empty() && spec.len() == 1 => {
      (d, Some(&spec[0]))
    }
    _ => return unevaluated(),
  };
  let Some(values) = data
    .iter()
    .map(crate::functions::math_ast::expr_to_f64)
    .collect::<Option<Vec<f64>>>()
  else {
    // Multivariate (matrix) data is not implemented yet: stay silently
    // symbolic. Anything else — symbolic entries, non-numbers — is invalid
    // input and warns like wolframscript.
    if !data.iter().all(|d| matches!(d, Expr::List(_))) {
      let call = unevaluated()?;
      crate::emit_message_to_stdout(&format!(
        "HistogramDistribution::invldd: The input data {} should be a vector \
         or a matrix of real numbers or a valid TemporalData object.",
        crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
      ));
    }
    return unevaluated();
  };
  let n = values.len() as i128;

  let data_distribution = |weights: Vec<Expr>, edges: Vec<Expr>| {
    Ok(call(
      "DataDistribution",
      vec![
        Expr::String("Histogram".to_string()),
        Expr::List(
          vec![Expr::List(weights.into()), Expr::List(edges.into())].into(),
        ),
        Expr::Integer(1),
        Expr::Integer(n),
      ],
    ))
  };

  // Exact path: an explicit exact width over exact data whose bins anchor at
  // multiples of w (i.e. NOT the value-centered layout, which wolframscript
  // reports in machine reals like HistogramList does).
  if let Some(wexpr) = width_spec
    && let Some(w) = as_exact_frac(wexpr)
    && w.0 > 0
    && let Some(exact) = data
      .iter()
      .map(as_exact_frac)
      .collect::<Option<Vec<(i128, i128)>>>()
  {
    let spread = exact
      .iter()
      .any(|v| frac_cmp(*v, exact[0]) != std::cmp::Ordering::Equal);
    // v/w is an integer iff (vn·wd) divides evenly by (vd·wn).
    let centered =
      spread && exact.iter().all(|v| (v.0 * w.1) % (v.1 * w.0) == 0);
    if !centered {
      // Quotient v/w as a reduced fraction, and its floor.
      let quot = |v: (i128, i128)| frac_reduce(v.0 * w.1, v.1 * w.0);
      let floor_div = |(qn, qd): (i128, i128)| qn.div_euclid(qd);
      let mn = *exact
        .iter()
        .min_by(|a, b| frac_cmp(**a, **b))
        .expect("nonempty");
      let mx = *exact
        .iter()
        .max_by(|a, b| frac_cmp(**a, **b))
        .expect("nonempty");
      let k_lo = floor_div(quot(mn));
      let (qn, qd) = quot(mx);
      // ceil(mx/w), bumped one bin when mx lands exactly on an edge so the
      // maximum stays strictly inside the last bin.
      let k_hi = if qd == 1 {
        qn + 1
      } else {
        qn.div_euclid(qd) + 1
      };
      let num_bins = (k_hi - k_lo).max(1) as usize;
      let mut counts = vec![0i128; num_bins];
      for v in &exact {
        let idx = floor_div(quot(*v)) - k_lo;
        if (0..num_bins as i128).contains(&idx) {
          counts[idx as usize] += 1;
        }
      }
      // Density = count / (n·w); edge_i = (k_lo + i)·w.
      let weights: Vec<Expr> = counts
        .iter()
        .map(|c| frac_expr(frac_reduce(c * w.1, n * w.0)))
        .collect();
      let edges: Vec<Expr> = (0..=num_bins as i128)
        .map(|i| frac_expr(frac_reduce((k_lo + i) * w.0, w.1)))
        .collect();
      return data_distribution(weights, edges);
    }
  }

  // Machine-real path: automatic binning, a non-exact width, or the
  // value-centered layout.
  let dx_opt = match width_spec
    .map(|e| crate::functions::math_ast::expr_to_f64(e).filter(|v| *v > 0.0))
  {
    None => None,
    Some(Some(dx)) => Some(dx),
    Some(None) => return unevaluated(),
  };
  let (min_val, max_val, dx, _) =
    crate::functions::list_helpers_ast::wl_bin_spec(&values, dx_opt);
  let num_bins = ((max_val - min_val) / dx + 1e-9).floor().max(1.0) as usize;
  let mut counts = vec![0i128; num_bins];
  for &v in &values {
    if v < min_val {
      continue;
    }
    let idx = ((v - min_val) / dx).floor();
    if idx >= 0.0 && (idx as usize) < num_bins {
      counts[idx as usize] += 1;
    }
  }
  let weights: Vec<Expr> = counts
    .iter()
    .map(|c| Expr::Real(*c as f64 / (n as f64 * dx)))
    .collect();
  let edges: Vec<Expr> = (0..=num_bins)
    .map(|i| Expr::Real(min_val + i as f64 * dx))
    .collect();
  data_distribution(weights, edges)
}

/// Extract (densities, bin edges) from a histogram DataDistribution
/// (`dargs = ["Histogram", {weights, edges}, 1, n]`).
fn histogram_parts(
  dargs: &[Expr],
) -> Option<(crate::ExprList, crate::ExprList)> {
  if dargs.len() != 4 {
    return None;
  }
  match &dargs[0] {
    Expr::String(s) | Expr::Identifier(s) if s == "Histogram" => {}
    _ => return None,
  }
  let inner = match &dargs[1] {
    Expr::List(items) if items.len() == 2 => items,
    _ => return None,
  };
  match (&inner[0], &inner[1]) {
    (Expr::List(w), Expr::List(e))
      if !w.is_empty() && e.len() == w.len() + 1 =>
    {
      Some((w.clone(), e.clone()))
    }
    _ => None,
  }
}

/// PDF (density sum) or CDF (per-bin ramp sum) of a histogram
/// DataDistribution, in wolframscript's Boole form:
///
/// PDF: Σ wᵢ·Boole[loᵢ ≤ x < hiᵢ]
/// CDF: Boole[x ≥ last] + Σ (cumᵢ + wᵢ·(x − loᵢ))·Boole[loᵢ ≤ x < hiᵢ]
///
/// where cumᵢ accumulates wᵢ·widthᵢ in the stored arithmetic (machine reals
/// or exact rationals). A numeric point collapses to its value through
/// ordinary evaluation.
fn histogram_pdf_cdf(
  dargs: &[Expr],
  x: &Expr,
  cumulative: bool,
) -> Option<Result<Expr, InterpreterError>> {
  let (weights, edges) = histogram_parts(dargs)?;
  let boole = |cond: Expr| call("Boole", vec![cond]);
  // At a numeric point the sum is evaluated down to its value; at a symbolic
  // point the built form is returned as-is, preserving wolframscript's term
  // order (CDF leads with Boole[x >= last] and the bins follow ascending),
  // which ordinary Plus canonicalization would reshuffle.
  let numeric_point = crate::functions::math_ast::expr_to_f64(x);
  // PDF at a numeric point is a direct bin lookup: the stored density inside
  // a bin, and the *exact* integer 0 outside the support (wolframscript
  // returns 0, not 0., even for a real point).
  if let Some(xf) = numeric_point
    && !cumulative
  {
    let ef = |e: &Expr| crate::functions::math_ast::expr_to_f64(e);
    for (i, w) in weights.iter().enumerate() {
      let (Some(lo), Some(hi)) = (ef(&edges[i]), ef(&edges[i + 1])) else {
        return Some(Ok(Expr::Integer(0)));
      };
      if lo <= xf && xf < hi {
        return Some(Ok(w.clone()));
      }
    }
    return Some(Ok(Expr::Integer(0)));
  }
  // Flat Times/Plus FunctionCalls: the display of the raw (uncanonicalized)
  // tree matches wolframscript's, e.g. `(3*(-2 + x))/10` and `(x*Boole[…])/10`.
  let build = || -> Result<Expr, InterpreterError> {
    let mut terms: Vec<Expr> = Vec::new();
    if cumulative {
      terms.push(boole(comparison(
        x.clone(),
        ComparisonOp::GreaterEqual,
        edges[edges.len() - 1].clone(),
      )));
    }
    let mut cum: Expr = int(0);
    for (i, w) in weights.iter().enumerate() {
      let (lo, hi) = (edges[i].clone(), edges[i + 1].clone());
      let cond = comparison3(
        lo.clone(),
        ComparisonOp::LessEqual,
        x.clone(),
        ComparisonOp::Less,
        hi.clone(),
      );
      if cumulative {
        let is_zero = |e: &Expr| {
          matches!(e, Expr::Integer(0))
            || matches!(e, Expr::Real(r) if *r == 0.0)
        };
        // w·(x - lo), displayed `w*(-lo + x)` (bare `w*x` when lo is 0).
        let ramp_offset = if is_zero(&lo) {
          x.clone()
        } else {
          call("Plus", vec![eval(times(int(-1), lo.clone()))?, x.clone()])
        };
        // (cum + w·(x - lo))·Boole[…]; the first bin has no accumulated mass.
        terms.push(if is_zero(&cum) {
          call("Times", vec![w.clone(), ramp_offset, boole(cond)])
        } else {
          call(
            "Times",
            vec![
              call(
                "Plus",
                vec![cum.clone(), call("Times", vec![w.clone(), ramp_offset])],
              ),
              boole(cond),
            ],
          )
        });
        cum = eval(plus(cum, times(w.clone(), minus(hi, lo))))?;
      } else {
        terms.push(call("Times", vec![w.clone(), boole(cond)]));
      }
    }
    let sum = call("Plus", terms);
    if numeric_point.is_some() {
      eval(sum)
    } else {
      Ok(sum)
    }
  };
  Some(build())
}

/// Mean (k = 1) or raw second moment (k = 2) of a histogram
/// DataDistribution: Σ probᵢ·midᵢ and Σ probᵢ·(midᵢ² + widthᵢ²/12) with
/// probᵢ = wᵢ·widthᵢ, accumulated left-to-right in the stored arithmetic so
/// machine-real results match wolframscript bit-for-bit.
fn histogram_moment(dargs: &[Expr], k: u32) -> Option<Expr> {
  let (weights, edges) = histogram_parts(dargs)?;
  // Exact path when every stored part is an integer/rational.
  let exact_w: Option<Vec<(i128, i128)>> =
    weights.iter().map(as_exact_frac).collect();
  let exact_e: Option<Vec<(i128, i128)>> =
    edges.iter().map(as_exact_frac).collect();
  if let (Some(ws), Some(es)) = (exact_w, exact_e) {
    let mul = |a: (i128, i128), b: (i128, i128)| -> Option<(i128, i128)> {
      Some(frac_reduce(a.0.checked_mul(b.0)?, a.1.checked_mul(b.1)?))
    };
    let add = |a: (i128, i128), b: (i128, i128)| -> Option<(i128, i128)> {
      Some(frac_reduce(
        a.0.checked_mul(b.1)?.checked_add(b.0.checked_mul(a.1)?)?,
        a.1.checked_mul(b.1)?,
      ))
    };
    let mut m = (0i128, 1i128);
    for (i, w) in ws.iter().enumerate() {
      let width = add(es[i + 1], (-es[i].0, es[i].1))?;
      let prob = mul(*w, width)?;
      let mid = mul(add(es[i], es[i + 1])?, (1, 2))?;
      let term = match k {
        1 => mul(prob, mid)?,
        2 => mul(
          prob,
          add(mul(mid, mid)?, mul(mul(width, width)?, (1, 12))?)?,
        )?,
        _ => return None,
      };
      m = add(m, term)?;
    }
    return Some(frac_expr(m));
  }
  // Machine-real path, accumulated left-to-right.
  let to_f64 = crate::functions::math_ast::expr_to_f64;
  let ws: Option<Vec<f64>> = weights.iter().map(to_f64).collect();
  let es: Option<Vec<f64>> = edges.iter().map(to_f64).collect();
  let (ws, es) = (ws?, es?);
  let mut m = 0.0;
  for (i, w) in ws.iter().enumerate() {
    let width = es[i + 1] - es[i];
    let prob = w * width;
    let mid = (es[i] + es[i + 1]) / 2.0;
    m += match k {
      1 => prob * mid,
      2 => prob * (mid * mid + width * width / 12.0),
      _ => return None,
    };
  }
  Some(Expr::Real(m))
}

/// Extract (weights, values) from an empirical DataDistribution.
fn data_distribution_parts(
  dargs: &[Expr],
) -> Option<(Vec<(i128, i128)>, Vec<(i128, i128)>)> {
  if dargs.len() != 4 {
    return None;
  }
  match &dargs[0] {
    Expr::String(s) if s == "Empirical" => {}
    _ => return None,
  }
  let inner = match &dargs[1] {
    Expr::List(items) if items.len() == 3 => items,
    _ => return None,
  };
  let (w_list, v_list) = match (&inner[0], &inner[1]) {
    (Expr::List(w), Expr::List(v)) if w.len() == v.len() => (w, v),
    _ => return None,
  };
  let mut weights = Vec::with_capacity(w_list.len());
  let mut values = Vec::with_capacity(v_list.len());
  for (w, v) in w_list.iter().zip(v_list.iter()) {
    weights.push(as_exact_frac(w)?);
    values.push(as_exact_frac(v)?);
  }
  Some((weights, values))
}

/// Mean (k = 1) or raw second moment (k = 2) of an empirical
/// distribution, exactly.
pub fn data_distribution_moment(dargs: &[Expr], k: u32) -> Option<Expr> {
  // A histogram DataDistribution has its own (continuous, per-bin) moments.
  if let Some(m) = histogram_moment(dargs, k) {
    return Some(m);
  }
  let (weights, values) = data_distribution_parts(dargs)?;
  let mut num: i128 = 0;
  let mut den: i128 = 1;
  for (w, v) in weights.iter().zip(values.iter()) {
    // w * v^k
    let (vn, vd) = (v.0.checked_pow(k)?, v.1.checked_pow(k)?);
    let tn = w.0.checked_mul(vn)?;
    let td = w.1.checked_mul(vd)?;
    num = num.checked_mul(td)?.checked_add(tn.checked_mul(den)?)?;
    den = den.checked_mul(td)?;
  }
  let g = {
    let (mut a, mut b) = (num.abs().max(1), den.abs());
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a.max(1)
  };
  let (num, den) = (num / g, den / g);
  Some(if den == 1 {
    Expr::Integer(num)
  } else {
    crate::functions::math_ast::make_rational(num, den)
  })
}

/// PDF (point mass) and CDF (left-closed step sum) at a numeric point.
fn data_distribution_pdf_cdf(
  dargs: &[Expr],
  x: &Expr,
  cumulative: bool,
) -> Option<Expr> {
  let (weights, values) = data_distribution_parts(dargs)?;
  let xf = as_exact_frac(x)?;
  let mut num: i128 = 0;
  let mut den: i128 = 1;
  for (w, v) in weights.iter().zip(values.iter()) {
    let include = if cumulative {
      frac_cmp(*v, xf) != std::cmp::Ordering::Greater
    } else {
      *v == xf
    };
    if include {
      num = num * w.1 + w.0 * den;
      den *= w.1;
    }
  }
  let g = {
    let (mut a, mut b) = (num.abs().max(1), den.abs());
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a.max(1)
  };
  let (num, den) = (num / g, den / g);
  Some(if den == 1 {
    Expr::Integer(num)
  } else {
    crate::functions::math_ast::make_rational(num, den)
  })
}

// ─── ProductDistribution ─────────────────────────────────────────────

/// PDF[ProductDistribution[d1, d2], {x, y}] for pairs of standard
/// normals and integer-rate exponentials, in wolframscript's merged
/// form: one E-power with position-styled exponent terms, And-ed
/// support conditions, and the coefficient quirks (6*E^...,
/// E^.../Sqrt[2*Pi], (3*E^...)/Sqrt[2*Pi], and the lambda = 2 special
/// case E^...*Sqrt[2/Pi]).
fn pdf_product_distribution(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    call(
      "PDF",
      vec![unevaluated("ProductDistribution", dargs), x.clone()],
    )
  };
  let vars = match &x {
    Expr::List(vars) if vars.len() == dargs.len() => vars,
    _ => return Ok(unevaluated()),
  };
  if dargs.len() != 2 {
    return Ok(unevaluated());
  }

  enum Comp {
    StdNormal,
    Exponential(i128),
  }
  let mut comps: Vec<Comp> = Vec::with_capacity(2);
  for d in dargs {
    match d {
      Expr::FunctionCall { name, args } if name == "NormalDistribution" => {
        let standard = args.is_empty()
          || (args.len() == 2
            && matches!(&args[0], Expr::Integer(0))
            && matches!(&args[1], Expr::Integer(1)));
        if !standard {
          return Ok(unevaluated());
        }
        comps.push(Comp::StdNormal);
      }
      Expr::FunctionCall { name, args }
        if name == "ExponentialDistribution" && args.len() == 1 =>
      {
        match &args[0] {
          Expr::Integer(l) if *l >= 1 => comps.push(Comp::Exponential(*l)),
          _ => return Ok(unevaluated()),
        }
      }
      _ => return Ok(unevaluated()),
    }
  }

  let pow2 = |e: Expr| power(e, int(2));
  let div2 = |a: Expr, b: Expr| divide(a, b);

  // Exponent terms (position-styled for normals, like the multinormal
  // diagonal PDF)
  let mut terms: Vec<Expr> = Vec::with_capacity(2);
  let mut conds: Vec<Expr> = Vec::new();
  let mut normal_count = 0usize;
  let mut rate_product: i128 = 1;
  let mut first_scaled_seen = false;
  for (i, comp) in comps.iter().enumerate() {
    let v = vars[i].clone();
    match comp {
      Comp::StdNormal => {
        normal_count += 1;
        let sq = pow2(v);
        terms.push(if !first_scaled_seen {
          first_scaled_seen = true;
          times(crate::functions::math_ast::make_rational(-1, 2), sq)
        } else {
          neg(div2(sq, int(2)))
        });
      }
      Comp::Exponential(l) => {
        rate_product *= l;
        terms.push(if *l == 1 {
          neg(v.clone())
        } else {
          times(int(-l), v.clone())
        });
        conds.push(Expr::Comparison {
          operands: vec![v, int(0)],
          operators: vec![ComparisonOp::GreaterEqual],
        });
      }
    }
  }
  let e_pow = power(Expr::Identifier("E".to_string()), call("Plus", terms));
  let sqrt = |e: Expr| call("Sqrt", vec![e]);

  // Assemble the density with wolframscript's coefficient shapes
  let value = match normal_count {
    0 => {
      if rate_product == 1 {
        e_pow
      } else {
        times(int(rate_product), e_pow)
      }
    }
    1 => {
      if rate_product == 2 {
        // 2/Sqrt[2*Pi] folds to Sqrt[2/Pi], printed as a postfix factor
        call("Times", vec![e_pow, sqrt(div2(int(2), pi()))])
      } else if rate_product == 1 {
        div2(e_pow, sqrt(times(int(2), pi())))
      } else {
        div2(times(int(rate_product), e_pow), sqrt(times(int(2), pi())))
      }
    }
    _ => div2(e_pow, times(int(2), pi())),
  };

  if conds.is_empty() {
    return Ok(value);
  }
  let cond = if conds.len() == 1 {
    conds.remove(0)
  } else {
    call("And", conds)
  };
  Ok(call(
    "Piecewise",
    vec![
      Expr::List(vec![Expr::List(vec![value, cond].into())].into()),
      int(0),
    ],
  ))
}

/// Shared pieces for UniformSumDistribution (Irwin-Hall): wolframscript
/// prints the inclusion-exclusion sum ascending up to the midpoint and
/// the x -> n-x reflection past it, with the CDF middles expanded.
fn uniform_sum_n(dargs: &[Expr]) -> Option<i128> {
  match dargs {
    [Expr::Integer(n)] if *n >= 1 && *n <= 25 => Some(*n),
    _ => None,
  }
}

fn fact(n: i128) -> i128 {
  let mut result = 1i128;
  for i in 2..=n {
    result *= i;
  }
  result
}

/// (x - k)^p as the raw print form (-k + x)^p, or (n - k - x)^p when
/// reflected; p == 1 collapses to the bare base.
fn shifted_power(k: i128, x: &Expr, p: i128, reflect_n: Option<i128>) -> Expr {
  let base = match reflect_n {
    None if k == 0 => x.clone(),
    None => call("Plus", vec![int(-k), x.clone()]),
    Some(n) => call(
      "Plus",
      vec![int(n - k), call("Times", vec![int(-1), x.clone()])],
    ),
  };
  if p == 1 { base } else { power(base, int(p)) }
}

/// Sum_{k=0..j} (-1)^k C(n,k) (x-k)^p / denom in raw print form,
/// terms ordered k descending (the canonical Plus order).
fn inclusion_exclusion_piece(
  n: i128,
  j: i128,
  p: i128,
  denom: i128,
  x: &Expr,
  reflect: bool,
) -> Expr {
  let reflect_n = if reflect { Some(n) } else { None };
  let mut terms: Vec<Expr> = Vec::new();
  let mut c_nk = crate::functions::binomial_coeff(n, j);
  for k in (1..=j).rev() {
    if k != j {
      // C(n,k) = C(n,k+1) (k+1)/(n-k)
      c_nk = (c_nk * (k + 1)) / (n - k);
    }
    let coeff = if k % 2 == 0 { c_nk } else { -c_nk };
    terms.push(call(
      "Times",
      vec![int(coeff), shifted_power(k, x, p, reflect_n)],
    ));
  }
  terms.push(shifted_power(0, x, p, reflect_n));
  let sum = if terms.len() == 1 {
    terms.pop().unwrap()
  } else {
    call("Plus", terms)
  };
  if denom == 1 {
    sum
  } else {
    divide(sum, int(denom))
  }
}

/// Exact coefficients (as i128 fractions over fact(n)) of the expanded
/// CDF middle polynomial sum_{k<=j} (-1)^k C(n,k) (x-k)^n / n!.
/// Returns c[i] as (num, den) reduced.
fn cdf_expanded_coeffs(n: i128, j: i128) -> Vec<(i128, i128)> {
  let nf = fact(n);
  let mut coeffs = vec![0i128; (n + 1) as usize];
  let mut c_nk = 1i128; // C(n,k)
  // (-1)^0 C(n,0) (x-0)^n = x^n
  coeffs[n as usize] = 1;
  for k in 1..=j {
    // (x - k)^n = sum_i C(n,i) x^i (-k)^(n-i)
    let sign = if k % 2 == 0 { 1 } else { -1 };
    c_nk = (c_nk * (n - k + 1)) / k; // C(n,k)
    let mut c_ni = 1i128; // C(n,i)
    let mut mk_ni = 1i128; // (-k)^(n-i)
    for _ in 0..n {
      mk_ni *= -k;
    }
    for i in 0..=n {
      if i > 0 {
        c_ni = (c_ni * (n - i + 1)) / i; // C(n,i)
        mk_ni /= -k; // (-k)^(n-i)
      }
      let term = sign * c_nk * c_ni * mk_ni;
      coeffs[i as usize] += term;
    }
  }
  coeffs
    .into_iter()
    .map(|c| {
      let g = {
        let (mut a, mut b) = (c.abs(), nf);
        while b != 0 {
          (a, b) = (b, a % b);
        }
        a.max(1)
      };
      (c / g, nf / g)
    })
    .collect()
}

/// Raw term coeff * base^i for the expanded CDF pieces; i == 0 gives
/// the bare rational and unit coefficients collapse.
fn coeff_power_term(num: i128, den: i128, base: &Expr, i: i128) -> Expr {
  let rational = |num: i128, den: i128| -> Expr {
    if den == 1 {
      int(num)
    } else {
      call("Rational", vec![int(num), int(den)])
    }
  };
  let pow = |i: i128| -> Expr {
    if i == 1 {
      base.clone()
    } else {
      power(base.clone(), int(i))
    }
  };
  if i == 0 {
    return rational(num, den);
  }
  // Pull the sign out so Plus prints "- (3*x)/2" instead of "+ (-3*x)/2"
  if num < 0 {
    return neg(coeff_power_term(-num, den, base, i));
  }
  if num == 1 && den == 1 {
    pow(i)
  } else if den == 1 {
    call("Times", vec![int(num), pow(i)])
  } else {
    // (num * base^i) / den prints as (num*base^i)/den; num == 1 drops
    // the explicit factor
    let numerator = if num == 1 {
      pow(i)
    } else {
      call("Times", vec![int(num), pow(i)])
    };
    divide(numerator, int(den))
  }
}

fn pdf_uniform_sum(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("UniformSumDistribution", dargs), x])
  };
  let Some(n) = uniform_sum_n(dargs) else {
    return Ok(unevaluated(dargs, x));
  };

  // Numeric point: evaluate the inclusion-exclusion sum directly
  if let Some(j) = numeric_floor(&x) {
    if j < 0 || j >= n {
      return eval(times(int(0), x)); // 0 or 0. matching x's exactness
    }
    let piece = inclusion_exclusion_piece(n, j, n - 1, fact(n - 1), &x, false);
    return eval(piece);
  }

  if n == 1 {
    // No leading zero piece, matching wolframscript
    return Ok(piecewise(
      vec![(
        int(1),
        comparison3(
          int(0),
          ComparisonOp::LessEqual,
          x,
          ComparisonOp::LessEqual,
          int(1),
        ),
      )],
      int(0),
    ));
  }

  let mut pairs: Vec<(Expr, Expr)> =
    vec![(int(0), comparison(x.clone(), ComparisonOp::Less, int(0)))];
  for j in 0..n {
    let reflect = 2 * j > n - 1;
    let piece = if reflect {
      inclusion_exclusion_piece(n, n - 1 - j, n - 1, fact(n - 1), &x, true)
    } else {
      inclusion_exclusion_piece(n, j, n - 1, fact(n - 1), &x, false)
    };
    let cond = if j == n - 1 {
      comparison3(
        int(j),
        ComparisonOp::LessEqual,
        x.clone(),
        ComparisonOp::LessEqual,
        int(j + 1),
      )
    } else {
      comparison3(
        int(j),
        ComparisonOp::LessEqual,
        x.clone(),
        ComparisonOp::Less,
        int(j + 1),
      )
    };
    pairs.push((piece, cond));
  }
  Ok(piecewise(pairs, int(0)))
}

fn cdf_uniform_sum(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("UniformSumDistribution", dargs), x])
  };
  let Some(n) = uniform_sum_n(dargs) else {
    return Ok(unevaluated(dargs, x));
  };

  if let Some(j) = numeric_floor(&x) {
    if j < 0 {
      return eval(times(int(0), x));
    }
    if j >= n {
      return eval(plus(int(1), times(int(0), x)));
    }
    let piece = inclusion_exclusion_piece(n, j, n, fact(n), &x, false);
    return eval(piece);
  }

  let mut pairs: Vec<(Expr, Expr)> =
    vec![(int(0), comparison(x.clone(), ComparisonOp::Less, int(0)))];
  for j in 0..n {
    let piece = if j == 0 {
      inclusion_exclusion_piece(n, 0, n, fact(n), &x, false)
    } else if j == n - 1 {
      // 1 - (n - x)^n / n!
      call(
        "Plus",
        vec![
          int(1),
          call(
            "Times",
            vec![
              int(-1),
              inclusion_exclusion_piece(n, 0, n, fact(n), &x, true),
            ],
          ),
        ],
      )
    } else if 2 * j < n {
      // Expanded ascending polynomial in x
      let coeffs = cdf_expanded_coeffs(n, j);
      let terms: Vec<Expr> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, f)| f.0 != 0)
        .map(|(i, &(num, den))| coeff_power_term(num, den, &x, i as i128))
        .collect();
      call("Plus", terms)
    } else {
      // 1 - mirror(j') at u = n - x, expanded in powers of u
      let coeffs = cdf_expanded_coeffs(n, n - 1 - j);
      let u = call(
        "Plus",
        vec![int(n), call("Times", vec![int(-1), x.clone()])],
      );
      let terms: Vec<Expr> = coeffs
        .iter()
        .enumerate()
        .filter_map(|(i, &(num, den))| {
          let (num, den) = if i == 0 {
            // 1 - c_0
            (den - num, den)
          } else {
            (-num, den)
          };
          if num == 0 {
            return None;
          }
          let g = {
            let (mut a, mut b) = (num.abs(), den);
            while b != 0 {
              (a, b) = (b, a % b);
            }
            a.max(1)
          };
          Some(coeff_power_term(num / g, den / g, &u, i as i128))
        })
        .collect();
      call("Plus", terms)
    };
    let cond = if j == n - 1 {
      comparison3(
        int(j),
        ComparisonOp::LessEqual,
        x.clone(),
        ComparisonOp::LessEqual,
        int(j + 1),
      )
    } else {
      comparison3(
        int(j),
        ComparisonOp::LessEqual,
        x.clone(),
        ComparisonOp::Less,
        int(j + 1),
      )
    };
    pairs.push((piece, cond));
  }
  Ok(piecewise(pairs, int(1)))
}

/// Floor of an exact or machine numeric Expr, None for symbolic input.
fn numeric_floor(x: &Expr) -> Option<i128> {
  match x {
    Expr::Integer(v) => Some(*v),
    Expr::Real(v) => Some(v.floor() as i128),
    Expr::FunctionCall { name, args } if name == "Rational" => {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          Some(p.div_euclid(*q))
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// PDF[BetaBinomialDistribution[a, b, n], k] in wolframscript's form:
/// Binomial[n,k] Pochhammer[a,k] Pochhammer[b,n-k] / Pochhammer[a+b,n]
/// wrapped in Piecewise over 0 <= k <= n (the denominator evaluates
/// when the parameters are numeric). Numeric k gives the exact value;
/// non-integers and out-of-range points give 0.
fn pdf_beta_binomial(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("BetaBinomialDistribution", dargs), x],
    )
  };
  if dargs.len() != 3 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, b, n) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let pmf = |k: Expr| -> Expr {
    // Wolfram prints "10 - k" for numeric n but "-k + n" symbolically
    let neg_k = call("Times", vec![int(-1), k.clone()]);
    let n_minus_k = if matches!(&n, Expr::Integer(_)) {
      call("Plus", vec![n.clone(), neg_k])
    } else {
      call("Plus", vec![neg_k, n.clone()])
    };
    let numerator = call(
      "Times",
      vec![
        call("Binomial", vec![n.clone(), k.clone()]),
        call("Pochhammer", vec![a.clone(), k.clone()]),
        call("Pochhammer", vec![b.clone(), n_minus_k]),
      ],
    );
    let denominator = call(
      "Pochhammer",
      vec![call("Plus", vec![a.clone(), b.clone()]), n.clone()],
    );
    divide(numerator, denominator)
  };

  // Numeric point: exact value or 0
  if let Some(int_n) = match &n {
    Expr::Integer(v) => Some(*v),
    _ => None,
  } {
    let integer_x = match &x {
      Expr::Integer(v) => Some(*v),
      Expr::Real(v) if v.fract() == 0.0 => Some(*v as i128),
      Expr::FunctionCall { name, .. } if name == "Rational" => {
        return Ok(int(0));
      }
      Expr::Real(_) => return Ok(int(0)),
      _ => None,
    };
    if let Some(k) = integer_x {
      if k < 0 || k > int_n {
        return Ok(int(0));
      }
      return eval(pmf(int(k)));
    }
  }

  // Symbolic point: Piecewise with the denominator pre-evaluated
  match &x {
    Expr::Identifier(_) => {
      let body = pmf(x.clone());
      let body = match &body {
        Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
          op: *op,
          left: left.clone(),
          right: Box::new(eval((**right).clone())?),
        },
        other => other.clone(),
      };
      let cond = comparison3(
        int(0),
        ComparisonOp::LessEqual,
        x,
        ComparisonOp::LessEqual,
        n,
      );
      Ok(piecewise(vec![(body, cond)], int(0)))
    }
    _ => Ok(unevaluated(dargs, x)),
  }
}

/// CDF[BetaBinomialDistribution[a, b, n], x] for numeric x: partial
/// PMF sums (the symbolic form uses HypergeometricPFQ internals that
/// stay unevaluated here).
fn cdf_beta_binomial(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "CDF",
      vec![unevaluated("BetaBinomialDistribution", dargs), x],
    )
  };
  if dargs.len() != 3 {
    return Ok(unevaluated(dargs, x));
  }
  let int_n = match &dargs[2] {
    Expr::Integer(v) => *v,
    _ => return Ok(unevaluated(dargs, x)),
  };
  let Some(floor_x) = numeric_floor(&x) else {
    return Ok(unevaluated(dargs, x));
  };
  if floor_x < 0 {
    return Ok(int(0));
  }
  if floor_x >= int_n {
    return Ok(int(1));
  }
  let mut acc = int(0);
  for k in 0..=floor_x {
    let term = pdf_beta_binomial(dargs, int(k))?;
    acc = eval(plus(acc, term))?;
  }
  Ok(acc)
}

/// PDF[BetaPrimeDistribution[p, q], x] =
/// Piecewise[{{x^(p-1) (1+x)^(-p-q) / Beta[p,q], x > 0}}, 0].
/// Numeric parameters evaluate the density (with 1/Beta pre-divided so
/// rationals hoist into the numerator); symbolic parameters keep the
/// raw Beta quotient form.
/// Density of the generalized BetaPrimeDistribution with power `w` and scale
/// `s`: w (x/s)^(w p - 1) (1 + (x/s)^w)^(-p - q) / (s Beta[p, q]).
fn beta_prime_general_body(
  p: &Expr,
  q: &Expr,
  w: &Expr,
  s: &Expr,
  x: &Expr,
) -> Result<Expr, InterpreterError> {
  let xs = divide(x.clone(), s.clone());
  let x_pow = power(xs.clone(), plus(int(-1), times(w.clone(), p.clone())));
  let neg_pq = plus(times(int(-1), p.clone()), times(int(-1), q.clone()));
  let bracket = power(plus(int(1), power(xs, w.clone())), neg_pq);
  let num = times(w.clone(), times(x_pow, bracket));
  let den = times(s.clone(), call("Beta", vec![p.clone(), q.clone()]));
  eval(divide(num, den))
}

fn pdf_beta_prime(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("BetaPrimeDistribution", dargs), x])
  };
  // Generalized forms: BetaPrimeDistribution[p, q, s] (power 1, scale s) and
  // BetaPrimeDistribution[p, q, b, a] (power b, scale a).
  if dargs.len() == 3 || dargs.len() == 4 {
    let (p, q, w, s) = if dargs.len() == 3 {
      (dargs[0].clone(), dargs[1].clone(), int(1), dargs[2].clone())
    } else {
      (
        dargs[0].clone(),
        dargs[1].clone(),
        dargs[2].clone(),
        dargs[3].clone(),
      )
    };
    let body = beta_prime_general_body(&p, &q, &w, &s, &x)?;
    if let Some(xv) = ms_numeric(&x) {
      if xv <= 0.0 {
        return Ok(int(0));
      }
      return Ok(body);
    }
    if !matches!(&x, Expr::Identifier(_)) {
      return Ok(unevaluated(dargs, x));
    }
    return Ok(piecewise(
      vec![(body, comparison(x, ComparisonOp::Greater, int(0)))],
      int(0),
    ));
  }
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let (p, q) = (dargs[0].clone(), dargs[1].clone());
  let is_exact_number = |e: &Expr| -> bool {
    matches!(e, Expr::Integer(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };

  // Numeric point: 0 outside the support, exact density inside
  let numeric_x = matches!(&x, Expr::Integer(_) | Expr::Real(_))
    || matches!(&x, Expr::FunctionCall { name, .. } if name == "Rational");
  let density_at = |x: &Expr| -> Result<Expr, InterpreterError> {
    let coeff = eval(divide(int(1), call("Beta", vec![p.clone(), q.clone()])))?;
    eval(call(
      "Times",
      vec![
        coeff,
        call("Power", vec![x.clone(), plus(int(-1), p.clone())]),
        call(
          "Power",
          vec![
            plus(int(1), x.clone()),
            call(
              "Plus",
              vec![
                call("Times", vec![int(-1), p.clone()]),
                call("Times", vec![int(-1), q.clone()]),
              ],
            ),
          ],
        ),
      ],
    ))
  };
  if numeric_x {
    let positive = match &x {
      Expr::Integer(v) => *v > 0,
      Expr::Real(v) => *v > 0.0,
      Expr::FunctionCall { name, args } if name == "Rational" => {
        matches!((&args[0], &args[1]), (Expr::Integer(p), Expr::Integer(q)) if p.signum() * q.signum() > 0)
      }
      _ => false,
    };
    if !positive {
      return Ok(int(0));
    }
    return density_at(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }

  let cond = comparison(x.clone(), ComparisonOp::Greater, int(0));
  if is_exact_number(&p) && is_exact_number(&q) {
    let density = density_at(&x)?;
    return Ok(piecewise(vec![(density, cond)], int(0)));
  }
  // Symbolic parameters: raw x^(-1 + p) (1 + x)^(-p - q) / Beta[p, q]
  let density = divide(
    call(
      "Times",
      vec![
        power(x.clone(), call("Plus", vec![int(-1), p.clone()])),
        power(
          call("Plus", vec![int(1), x.clone()]),
          call(
            "Plus",
            vec![
              call("Times", vec![int(-1), p.clone()]),
              call("Times", vec![int(-1), q.clone()]),
            ],
          ),
        ),
      ],
    ),
    call("Beta", vec![p, q]),
  );
  Ok(piecewise(vec![(density, cond)], int(0)))
}

/// CDF[BetaPrimeDistribution[p, q], x] =
/// Piecewise[{{BetaRegularized[x/(1+x), p, q], x > 0}}, 0].
fn cdf_beta_prime(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("BetaPrimeDistribution", dargs), x])
  };
  // Generalized forms: CDF = BetaRegularized[x^w/(s^w + x^w), p, q] with power
  // w and scale s (w = 1 for the 3-argument form).
  if dargs.len() == 3 || dargs.len() == 4 {
    let (p, q, w, s) = if dargs.len() == 3 {
      (dargs[0].clone(), dargs[1].clone(), int(1), dargs[2].clone())
    } else {
      (
        dargs[0].clone(),
        dargs[1].clone(),
        dargs[2].clone(),
        dargs[3].clone(),
      )
    };
    let xw = power(x.clone(), w.clone());
    let ratio = eval(divide(xw.clone(), plus(power(s, w.clone()), xw)))?;
    let body = call("BetaRegularized", vec![ratio, p, q]);
    if let Some(xv) = ms_numeric(&x) {
      if xv <= 0.0 {
        return Ok(int(0));
      }
      return eval(body);
    }
    if !matches!(&x, Expr::Identifier(_)) {
      return Ok(unevaluated(dargs, x));
    }
    return Ok(piecewise(
      vec![(body, comparison(x, ComparisonOp::Greater, int(0)))],
      int(0),
    ));
  }
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let (p, q) = (dargs[0].clone(), dargs[1].clone());
  let ratio = eval(divide(x.clone(), plus(int(1), x.clone())))?;
  let regularized = call("BetaRegularized", vec![ratio, p, q]);

  let numeric_x = matches!(&x, Expr::Integer(_) | Expr::Real(_))
    || matches!(&x, Expr::FunctionCall { name, .. } if name == "Rational");
  if numeric_x {
    let positive = match &x {
      Expr::Integer(v) => *v > 0,
      Expr::Real(v) => *v > 0.0,
      Expr::FunctionCall { name, args } if name == "Rational" => {
        matches!((&args[0], &args[1]), (Expr::Integer(p), Expr::Integer(q)) if p.signum() * q.signum() > 0)
      }
      _ => false,
    };
    if !positive {
      return Ok(int(0));
    }
    return eval(regularized);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(regularized, cond)], int(0)))
}

/// PDF[NoncentralChiSquareDistribution[v, l], x] in wolframscript's
/// per-case forms: the general Hypergeometric0F1Regularized skeleton
/// for symbolic parameters, evaluated skeletons for even integer v,
/// the Cosh/Sinh collapses for v = 1 and v = 3, and the chi-square
/// forms for l = 0. Odd v >= 5 with l != 0 stays unevaluated (Wolfram
/// collapses those into growing Bessel polynomials), as do points for
/// odd v (Wolfram switches to BesselI forms there).
fn pdf_noncentral_chi_square(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("NoncentralChiSquareDistribution", dargs), x],
    )
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let (nu, lam) = (dargs[0].clone(), dargs[1].clone());
  let raw_div = |a: Expr, b: Expr| divide(a, b);
  let sqrt = |e: Expr| call("Sqrt", vec![e]);
  let int_of = |e: &Expr| -> Option<i128> {
    match e {
      Expr::Integer(v) => Some(*v),
      _ => None,
    }
  };
  let numeric_lam = matches!(&lam, Expr::Integer(_) | Expr::Real(_))
    || matches!(&lam, Expr::FunctionCall { name, .. } if name == "Rational");

  // E^((-l - x)/2)
  let e_part = |at: &Expr| -> Result<Expr, InterpreterError> {
    let inner = eval(divide(
      plus(times(int(-1), lam.clone()), times(int(-1), at.clone())),
      int(2),
    ))?;
    Ok(power(e(), inner))
  };

  let body_at = |at: &Expr| -> Result<Option<Expr>, InterpreterError> {
    // l = 0: central chi-square print forms
    let lam_is_zero = matches!(&lam, Expr::Integer(0));
    if lam_is_zero && let Some(v) = int_of(&nu) {
      if v < 1 {
        return Ok(None);
      }
      let e_pow = power(e(), raw_div(at.clone(), int(2)));
      if v % 2 == 0 {
        // x^(v/2 - 1) / (2^(v/2) (v/2 - 1)! E^(x/2))
        let coef = (1..=(v / 2 - 1)).product::<i128>().max(1)
          * 2i128.pow((v / 2) as u32);
        let num = match v / 2 - 1 {
          0 => int(1),
          1 => at.clone(),
          p => power(at.clone(), int(p)),
        };
        return Ok(Some(raw_div(num, call("Times", vec![int(coef), e_pow]))));
      }
      // odd: x^((v-2)/2) / ((v-2)!! Sqrt[2 Pi] E^(x/2)), negative
      // powers of x move into the denominator
      let dfact = (1..=(v - 2)).step_by(2).product::<i128>().max(1);
      let sqrt_2pi = sqrt(call("Times", vec![int(2), pi()]));
      let mut den_factors: Vec<Expr> = Vec::new();
      if dfact > 1 {
        den_factors.push(int(dfact));
      }
      den_factors.push(e_pow);
      den_factors.push(sqrt_2pi);
      let num = match v {
        1 => {
          den_factors.push(sqrt(at.clone()));
          int(1)
        }
        3 => sqrt(at.clone()),
        _ => power(at.clone(), call("Rational", vec![int(v - 2), int(2)])),
      };
      return Ok(Some(raw_div(num, call("Times", den_factors))));
    }

    match int_of(&nu) {
      Some(v) if v >= 2 && v % 2 == 0 && numeric_lam => {
        // Even v: evaluated 0F1Regularized skeleton
        let mut factors = vec![e_part(at)?];
        match v / 2 - 1 {
          0 => {}
          1 => factors.push(at.clone()),
          p => factors.push(power(at.clone(), int(p))),
        }
        factors.push(call(
          "Hypergeometric0F1Regularized",
          vec![
            int(v / 2),
            eval(divide(times(lam.clone(), at.clone()), int(4)))?,
          ],
        ));
        Ok(Some(raw_div(
          call("Times", factors),
          int(2i128.pow((v / 2) as u32)),
        )))
      }
      Some(1) if numeric_lam => {
        // Cosh[Sqrt[l] Sqrt[x]] / (Sqrt[2 Pi] Sqrt[x])
        let arg = eval(times(sqrt(lam.clone()), sqrt(at.clone())))?;
        Ok(Some(raw_div(
          call("Times", vec![e_part(at)?, call("Cosh", vec![arg])]),
          call(
            "Times",
            vec![sqrt(call("Times", vec![int(2), pi()])), sqrt(at.clone())],
          ),
        )))
      }
      Some(3) if numeric_lam => {
        // Sinh[Sqrt[l] Sqrt[x]] / Sqrt[2 l Pi]
        let arg = eval(times(sqrt(lam.clone()), sqrt(at.clone())))?;
        let den = eval(sqrt(times(times(int(2), lam.clone()), pi())))?;
        Ok(Some(raw_div(
          call("Times", vec![e_part(at)?, call("Sinh", vec![arg])]),
          den,
        )))
      }
      Some(_) => Ok(None),
      None if !numeric_lam || matches!(&nu, Expr::Identifier(_)) => {
        // Fully or partially symbolic: the general skeleton
        let half_nu = raw_div(nu.clone(), int(2));
        Ok(Some(raw_div(
          call(
            "Times",
            vec![
              e_part(at)?,
              power(at.clone(), call("Plus", vec![int(-1), half_nu.clone()])),
              call(
                "Hypergeometric0F1Regularized",
                vec![
                  half_nu.clone(),
                  raw_div(times(lam.clone(), at.clone()), int(4)),
                ],
              ),
            ],
          ),
          power(int(2), half_nu),
        )))
      }
      None => Ok(None),
    }
  };

  // Numeric point
  let numeric_x = matches!(&x, Expr::Integer(_) | Expr::Real(_))
    || matches!(&x, Expr::FunctionCall { name, .. } if name == "Rational");
  if numeric_x {
    let positive = match &x {
      Expr::Integer(v) => *v > 0,
      Expr::Real(v) => *v > 0.0,
      Expr::FunctionCall { name, args } if name == "Rational" => matches!(
        (&args[0], &args[1]),
        (Expr::Integer(p), Expr::Integer(q)) if p.signum() * q.signum() > 0
      ),
      _ => false,
    };
    if !positive {
      return Ok(int(0));
    }
    // Points keep the even-v / l = 0 forms; odd v uses BesselI prints
    // in Wolfram that are not reproduced here
    let odd_v = matches!(int_of(&nu), Some(v) if v % 2 == 1)
      && !matches!(&lam, Expr::Integer(0));
    if odd_v {
      return Ok(unevaluated(dargs, x));
    }
    return match body_at(&x)? {
      Some(body) => eval(body),
      None => Ok(unevaluated(dargs, x)),
    };
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  match body_at(&x)? {
    Some(body) => {
      let cond = comparison(x, ComparisonOp::Greater, int(0));
      Ok(piecewise(vec![(body, cond)], int(0)))
    }
    None => Ok(unevaluated(dargs, x)),
  }
}

/// CDF[NoncentralChiSquareDistribution[v, l], x] =
/// Piecewise[{{MarcumQ[v/2, Sqrt[l], 0, Sqrt[x]], x > 0}}, 0], with
/// the four-argument MarcumQ evaluating numerically for machine reals.
fn cdf_noncentral_chi_square(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "CDF",
      vec![unevaluated("NoncentralChiSquareDistribution", dargs), x],
    )
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let (nu, lam) = (dargs[0].clone(), dargs[1].clone());
  let marcum = |at: &Expr| -> Result<Expr, InterpreterError> {
    Ok(call(
      "MarcumQ",
      vec![
        eval(divide(nu.clone(), int(2)))?,
        eval(call("Sqrt", vec![lam.clone()]))?,
        int(0),
        call("Sqrt", vec![at.clone()]),
      ],
    ))
  };

  let numeric_x = matches!(&x, Expr::Integer(_) | Expr::Real(_))
    || matches!(&x, Expr::FunctionCall { name, .. } if name == "Rational");
  if numeric_x {
    let positive = match &x {
      Expr::Integer(v) => *v > 0,
      Expr::Real(v) => *v > 0.0,
      Expr::FunctionCall { name, args } if name == "Rational" => matches!(
        (&args[0], &args[1]),
        (Expr::Integer(p), Expr::Integer(q)) if p.signum() * q.signum() > 0
      ),
      _ => false,
    };
    if !positive {
      return Ok(int(0));
    }
    return eval(marcum(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let body = marcum(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(body, cond)], int(0)))
}

/// PDF[ExponentialPowerDistribution[k, m, s], x]: two mirrored
/// branches 1/(2 E^(((+-(x-m))/s)^k/k) k^(1/k) s Gamma[1 + 1/k]) split
/// at x >= m. For k = 2 the coefficient merges into s Sqrt[2 Pi], and
/// when both branches canonicalize identically (k = 2 with m = 0) the
/// Piecewise collapses to the single expression, as in wolframscript.
fn pdf_exponential_power(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("ExponentialPowerDistribution", dargs), x],
    )
  };
  if dargs.len() != 3 || !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let (k, m, s) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let k_is_numeric = matches!(&k, Expr::Integer(_))
    || matches!(&k, Expr::FunctionCall { name, .. } if name == "Rational");

  // 2 k^(1/k) s Gamma[1 + 1/k], with the k = 2 Sqrt[2 Pi] merge
  let coef: Vec<Expr> = if matches!(&k, Expr::Integer(2)) {
    vec![
      s.clone(),
      call("Sqrt", vec![call("Times", vec![int(2), pi()])]),
    ]
  } else {
    vec![
      int(2),
      power(k.clone(), power(k.clone(), int(-1))),
      s.clone(),
      call("Gamma", vec![plus(int(1), power(k.clone(), int(-1)))]),
    ]
  };
  let branch = |diff: Expr| -> Result<Expr, InterpreterError> {
    let exponent = divide(power(divide(diff, s.clone()), k.clone()), k.clone());
    let mut den = coef.clone();
    den.push(power(e(), exponent));
    eval(divide(int(1), call("Times", den)))
  };
  let diff_plus = call(
    "Plus",
    vec![call("Times", vec![int(-1), m.clone()]), x.clone()],
  );
  let diff_minus = call(
    "Plus",
    vec![m.clone(), call("Times", vec![int(-1), x.clone()])],
  );
  let (b_plus, b_minus) = if k_is_numeric || matches!(&k, Expr::Identifier(_)) {
    (branch(diff_plus)?, branch(diff_minus)?)
  } else {
    return Ok(unevaluated(dargs, x));
  };
  if expr_to_string(&b_plus) == expr_to_string(&b_minus) {
    return Ok(b_plus);
  }
  let cond = comparison(x, ComparisonOp::GreaterEqual, m);
  Ok(piecewise(vec![(b_plus, cond)], b_minus))
}

/// CDF[ExponentialPowerDistribution[k, m, s], x] =
/// Piecewise[{{GammaRegularized[1/k, ((m-x)/s)^k/k]/2, x < m}},
///           1 - GammaRegularized[1/k, ((-m+x)/s)^k/k]/2].
fn cdf_exponential_power(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "CDF",
      vec![unevaluated("ExponentialPowerDistribution", dargs), x],
    )
  };
  if dargs.len() != 3 || !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let (k, m, s) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let half_reg = |diff: Expr| -> Result<Expr, InterpreterError> {
    let arg = divide(power(divide(diff, s.clone()), k.clone()), k.clone());
    eval(divide(
      call("GammaRegularized", vec![power(k.clone(), int(-1)), arg]),
      int(2),
    ))
  };
  let diff_plus = call(
    "Plus",
    vec![call("Times", vec![int(-1), m.clone()]), x.clone()],
  );
  let diff_minus = call(
    "Plus",
    vec![m.clone(), call("Times", vec![int(-1), x.clone()])],
  );
  let piece = half_reg(diff_minus)?;
  let default = eval(plus(int(1), times(int(-1), half_reg(diff_plus)?)))?;
  let cond = comparison(x, ComparisonOp::Less, m);
  Ok(piecewise(vec![(piece, cond)], default))
}

/// Helper: numeric value when the Expr is an exact or machine number.
fn rice_numeric(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(v) => Some(*v as f64),
    Expr::Real(v) => Some(*v),
    Expr::FunctionCall { name, args } if name == "Rational" => {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          Some(*p as f64 / *q as f64)
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// PDF[RiceDistribution[a, b], x] =
/// Piecewise[{{E^((-a^2 - x^2)/(2 b^2)) x BesselI[0, a x/b^2]/b^2,
/// x > 0}}, 0], evaluated so numeric parameters collapse the way
/// wolframscript prints them.
fn pdf_rice(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("RiceDistribution", dargs), x])
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let pow2 = |e: &Expr| power(e.clone(), int(2));
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let exponent = divide(
      call(
        "Plus",
        vec![
          call("Times", vec![int(-1), pow2(&a)]),
          call("Times", vec![int(-1), pow2(at)]),
        ],
      ),
      call("Times", vec![int(2), pow2(&b)]),
    );
    let bessel_arg = eval(divide(times(a.clone(), at.clone()), pow2(&b)))?;
    eval(divide(
      call(
        "Times",
        vec![
          power(e(), eval(exponent)?),
          at.clone(),
          call("BesselI", vec![int(0), bessel_arg]),
        ],
      ),
      pow2(&b),
    ))
  };

  let numeric_x = rice_numeric(&x).is_some();
  if numeric_x {
    if rice_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// CDF[RiceDistribution[a, b], x] =
/// Piecewise[{{MarcumQ[1, a/b, 0, x/b], x > 0}}, 0].
fn cdf_rice(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("RiceDistribution", dargs), x])
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let marcum = |at: &Expr| -> Result<Expr, InterpreterError> {
    Ok(call(
      "MarcumQ",
      vec![
        int(1),
        eval(divide(a.clone(), b.clone()))?,
        int(0),
        eval(divide(at.clone(), b.clone()))?,
      ],
    ))
  };
  if rice_numeric(&x).is_some() {
    if rice_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return eval(marcum(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = marcum(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// Mean and variance expressions for RiceDistribution: BesselI
/// combinations for numeric parameters (via the Laguerre identity
/// L_{1/2}(-k) = e^{-k/2}((1+k) I_0(k/2) + k I_1(k/2))), the inert
/// LaguerreL[1/2, ...] forms otherwise.
pub fn rice_mean_variance(
  a: &Expr,
  b: &Expr,
) -> Result<(Expr, Expr), InterpreterError> {
  let pow2 = |e: &Expr| power(e.clone(), int(2));
  let sqrt_half_pi = call("Sqrt", vec![divide(pi(), int(2))]);
  let numeric_params = rice_numeric(a).is_some() && rice_numeric(b).is_some();
  if numeric_params {
    // k = a^2/(2 b^2) as an exact fraction p/q; wolframscript pulls
    // the common denominator out of the Bessel sum:
    // sum = (q + p) I_0(k/2) + p I_1(k/2), all integer coefficients
    let k = eval(divide(pow2(a), times(int(2), pow2(b))))?;
    let (pn, qd) = match &k {
      Expr::Integer(v) => (*v, 1i128),
      Expr::FunctionCall { name, args } if name == "Rational" => {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) => (*p, *q),
          _ => {
            return Err(InterpreterError::EvaluationError(
              "RiceDistribution: unexpected parameter form".into(),
            ));
          }
        }
      }
      _ => {
        // Real parameters: evaluate everything numerically
        let half_k = eval(divide(k.clone(), int(2)))?;
        let laguerre = call(
          "Times",
          vec![
            power(e(), eval(times(int(-1), half_k.clone()))?),
            call(
              "Plus",
              vec![
                call(
                  "Times",
                  vec![
                    eval(plus(int(1), k.clone()))?,
                    call("BesselI", vec![int(0), half_k.clone()]),
                  ],
                ),
                call(
                  "Times",
                  vec![k.clone(), call("BesselI", vec![int(1), half_k])],
                ),
              ],
            ),
          ],
        );
        let mean = eval(call(
          "Times",
          vec![b.clone(), sqrt_half_pi.clone(), laguerre.clone()],
        ))?;
        let var = eval(call(
          "Plus",
          vec![
            eval(plus(pow2(a), times(int(2), pow2(b))))?,
            call(
              "Times",
              vec![
                int(-1),
                divide(
                  call("Times", vec![pi(), pow2(b), power(laguerre, int(2))]),
                  int(2),
                ),
              ],
            ),
          ],
        ))?;
        return Ok((mean, var));
      }
    };
    if pn == 0 {
      // Rayleigh case: mean = b Sqrt[Pi/2], var = 2 b^2 - pi b^2/2
      let mean = eval(times(b.clone(), sqrt_half_pi.clone()))?;
      let var = eval(call(
        "Plus",
        vec![
          times(int(2), pow2(b)),
          call(
            "Times",
            vec![int(-1), divide(call("Times", vec![pi(), pow2(b)]), int(2))],
          ),
        ],
      ))?;
      return Ok((mean, var));
    }
    let half_k = eval(divide(k.clone(), int(2)))?;
    let sum_term = |coef: i128, order: i128| -> Expr {
      let bessel = call("BesselI", vec![int(order), half_k.clone()]);
      if coef == 1 {
        bessel
      } else {
        call("Times", vec![int(coef), bessel])
      }
    };
    let sum = call("Plus", vec![sum_term(qd + pn, 0), sum_term(pn, 1)]);
    // Mean = (b/q) Sqrt[Pi/2] sum / e^(k/2), assembled raw so the
    // factored sum is not re-canonicalized
    let r = eval(divide(b.clone(), int(qd)))?;
    let (rn, rd) = match &r {
      Expr::Integer(v) => (*v, 1i128),
      Expr::FunctionCall { name, args } if name == "Rational" => {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) => (*p, *q),
          _ => (1, 1),
        }
      }
      _ => (1, 1),
    };
    let mut num_factors: Vec<Expr> = Vec::new();
    if rn != 1 {
      num_factors.push(int(rn));
    }
    num_factors.push(sqrt_half_pi.clone());
    num_factors.push(sum.clone());
    let numerator = call("Times", num_factors);
    let e_half = power(e(), half_k.clone());
    let denominator = if rd == 1 {
      eval(e_half)?
    } else {
      eval(times(int(rd), e_half))?
    };
    // Single-factor denominators print without parentheses
    let den_str = expr_to_string(&denominator);
    let den_str = if den_str.contains('*') {
      format!("({den_str})")
    } else {
      den_str
    };
    let mean =
      Expr::Raw(format!("({})/{}", expr_to_string(&numerator), den_str));
    // Var = base - Pi sum^2 / (q^2 2 e^k / b^2)
    let base = eval(plus(pow2(a), times(int(2), pow2(b))))?;
    let denom = eval(divide(
      times(int(2 * qd * qd), power(e(), k.clone())),
      pow2(b),
    ))?;
    let var = Expr::Raw(format!(
      "{} - (Pi*({})^2)/({})",
      expr_to_string(&base),
      expr_to_string(&sum),
      expr_to_string(&denom)
    ));
    return Ok((mean, var));
  }
  // Symbolic: LaguerreL[1/2, -a^2/(2 b^2)]
  let laguerre = call(
    "LaguerreL",
    vec![
      call("Rational", vec![int(1), int(2)]),
      call(
        "Times",
        vec![
          call("Rational", vec![int(-1), int(2)]),
          pow2(a),
          power(b.clone(), int(-2)),
        ],
      ),
    ],
  );
  let mean = eval(call(
    "Times",
    vec![b.clone(), sqrt_half_pi, laguerre.clone()],
  ))?;
  let var = eval(call(
    "Plus",
    vec![
      pow2(a),
      times(int(2), pow2(b)),
      call(
        "Times",
        vec![
          int(-1),
          divide(
            call("Times", vec![pow2(b), pi(), power(laguerre, int(2))]),
            int(2),
          ),
        ],
      ),
    ],
  ))?;
  Ok((mean, var))
}

/// Numeric value of an exact or machine number Expr (for the
/// MinStable parameter branches).
fn ms_numeric(e: &Expr) -> Option<f64> {
  rice_numeric(e)
}

/// Shared raw pieces for MinStableDistribution[a, b, g]: u(x) =
/// 1 + g (a - x)/b and the (-a + x)/b argument of the Gumbel branch.
fn ms_u(a: &Expr, b: &Expr, g: &Expr, x: &Expr) -> Expr {
  // For numeric negative g wolframscript folds the sign into the
  // difference: 1 + |g| (x - a)/b prints "1 + (-1 + x)/4" rather than
  // "1 - (1 - x)/4"
  let (coef, diff) = if ms_numeric(g).is_some_and(|v| v < 0.0) {
    (
      call("Times", vec![int(-1), g.clone()]),
      call(
        "Plus",
        vec![call("Times", vec![int(-1), a.clone()]), x.clone()],
      ),
    )
  } else {
    (
      g.clone(),
      call(
        "Plus",
        vec![a.clone(), call("Times", vec![int(-1), x.clone()])],
      ),
    )
  };
  plus(int(1), divide(call("Times", vec![coef, diff]), b.clone()))
}

fn ms_z(a: &Expr, b: &Expr, x: &Expr) -> Expr {
  divide(
    call(
      "Plus",
      vec![call("Times", vec![int(-1), a.clone()]), x.clone()],
    ),
    b.clone(),
  )
}

/// PDF[MinStableDistribution[a, b, g], x] in wolframscript's forms:
/// the Gumbel branch for g == 0 (full real support) and the
/// generalized branch with support u > 0 otherwise; symbolic g keeps
/// both pieces.
fn pdf_min_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("MinStableDistribution", dargs), x])
  };
  if dargs.len() != 3 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, b, g) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let gumbel = |at: &Expr| -> Expr {
    // E^(-E^((-a + x)/b) - (a - x)/b)/b
    let z = ms_z(&a, &b, at);
    let neg_az = divide(
      call(
        "Plus",
        vec![a.clone(), call("Times", vec![int(-1), at.clone()])],
      ),
      b.clone(),
    );
    divide(
      power(
        e(),
        call(
          "Plus",
          vec![
            call("Times", vec![int(-1), power(e(), z)]),
            call("Times", vec![int(-1), neg_az]),
          ],
        ),
      ),
      b.clone(),
    )
  };
  let general = |at: &Expr| -> Expr {
    // u^(-1 - 1/g) / (b E^(u^(-1/g)))
    let u = ms_u(&a, &b, &g, at);
    let inv_g = power(g.clone(), int(-1));
    divide(
      power(
        u.clone(),
        call(
          "Plus",
          vec![int(-1), call("Times", vec![int(-1), inv_g.clone()])],
        ),
      ),
      call(
        "Times",
        vec![
          b.clone(),
          power(e(), power(u, call("Times", vec![int(-1), inv_g]))),
        ],
      ),
    )
  };

  let g_num = ms_numeric(&g);
  let numeric_x = ms_numeric(&x).is_some();
  match g_num {
    Some(gv) if gv == 0.0 => {
      if numeric_x {
        return eval(gumbel(&x));
      }
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      // Raw assembly with evaluated subparts keeps the exponent in
      // wolframscript's -E^x + x order
      let z = eval(ms_z(&a, &b, &x))?;
      let neg_az = eval(call(
        "Times",
        vec![
          int(-1),
          divide(
            call(
              "Plus",
              vec![a.clone(), call("Times", vec![int(-1), x.clone()])],
            ),
            b.clone(),
          ),
        ],
      ))?;
      let exponent = call(
        "Plus",
        vec![call("Times", vec![int(-1), power(e(), z)]), neg_az],
      );
      let body = power(e(), exponent);
      let b_is_one = matches!(&b, Expr::Integer(1));
      Ok(if b_is_one {
        body
      } else {
        divide(body, b.clone())
      })
    }
    Some(_) => {
      if numeric_x {
        // Inside the support?
        let u_val = eval(ms_u(&a, &b, &g, &x))?;
        if ms_numeric(&u_val).is_some_and(|v| v <= 0.0) {
          return Ok(int(0));
        }
        return eval(general(&x));
      }
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let cond =
        comparison(eval(ms_u(&a, &b, &g, &x))?, ComparisonOp::Greater, int(0));
      Ok(piecewise(vec![(eval(general(&x))?, cond)], int(0)))
    }
    None => {
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let p1 = (
        gumbel(&x),
        comparison(g.clone(), ComparisonOp::Equal, int(0)),
      );
      let p2 = (
        general(&x),
        call(
          "And",
          vec![
            comparison(g.clone(), ComparisonOp::NotEqual, int(0)),
            comparison(ms_u(&a, &b, &g, &x), ComparisonOp::Greater, int(0)),
          ],
        ),
      );
      Ok(piecewise(vec![p1, p2], int(0)))
    }
  }
}

/// CDF[MinStableDistribution[a, b, g], x]: 1 - E^(-E^((-a+x)/b)) for
/// g == 0; 1 - E^(-u^(-1/g)) on u > 0 otherwise, with default 1 above
/// the support for g > 0 and 0 below it for g < 0.
fn cdf_min_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("MinStableDistribution", dargs), x])
  };
  if dargs.len() != 3 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, b, g) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let one_minus = |inner: Expr| -> Expr {
    call("Plus", vec![int(1), call("Times", vec![int(-1), inner])])
  };
  let gumbel = |at: &Expr| -> Expr {
    one_minus(power(
      e(),
      call("Times", vec![int(-1), power(e(), ms_z(&a, &b, at))]),
    ))
  };
  let general = |at: &Expr| -> Expr {
    // Exponent built as UnaryOp Minus so the printer keeps
    // u^(-g^(-1)) instead of hoisting the negative power
    let u = ms_u(&a, &b, &g, at);
    one_minus(power(
      e(),
      neg(power(
        u,
        call("Times", vec![int(-1), power(g.clone(), int(-1))]),
      )),
    ))
  };

  let g_num = ms_numeric(&g);
  let numeric_x = ms_numeric(&x).is_some();
  match g_num {
    Some(gv) if gv == 0.0 => {
      if numeric_x || matches!(&x, Expr::Identifier(_)) {
        eval(gumbel(&x))
      } else {
        Ok(unevaluated(dargs, x))
      }
    }
    Some(gv) => {
      if numeric_x {
        let u_val = eval(ms_u(&a, &b, &g, &x))?;
        if ms_numeric(&u_val).is_some_and(|v| v <= 0.0) {
          return Ok(if gv > 0.0 { int(1) } else { int(0) });
        }
        return eval(general(&x));
      }
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let cond =
        comparison(eval(ms_u(&a, &b, &g, &x))?, ComparisonOp::Greater, int(0));
      Ok(piecewise(
        vec![(eval(general(&x))?, cond)],
        if gv > 0.0 { int(1) } else { int(0) },
      ))
    }
    None => {
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let u = ms_u(&a, &b, &g, &x);
      let p1 = (
        gumbel(&x),
        comparison(g.clone(), ComparisonOp::Equal, int(0)),
      );
      let p2 = (
        general(&x),
        call(
          "And",
          vec![
            comparison(g.clone(), ComparisonOp::NotEqual, int(0)),
            comparison(u.clone(), ComparisonOp::Greater, int(0)),
          ],
        ),
      );
      let p3 = (
        int(1),
        call(
          "And",
          vec![
            comparison(g.clone(), ComparisonOp::Greater, int(0)),
            comparison(u, ComparisonOp::LessEqual, int(0)),
          ],
        ),
      );
      Ok(piecewise(vec![p1, p2, p3], int(0)))
    }
  }
}

/// Mean and variance for MinStableDistribution: Gumbel constants at
/// g == 0, Gamma-function forms below the existence thresholds (g < 1
/// for the mean, 2 g < 1 for the variance), Indeterminate beyond.
fn min_stable_mean_variance(
  a: &Expr,
  b: &Expr,
  g: &Expr,
) -> Result<(Expr, Expr), InterpreterError> {
  let euler_gamma = Expr::Identifier("EulerGamma".to_string());
  let mean_gumbel = || {
    plus(
      a.clone(),
      times(int(-1), times(b.clone(), euler_gamma.clone())),
    )
  };
  let var_gumbel = || {
    divide(
      call("Times", vec![power(b.clone(), int(2)), power(pi(), int(2))]),
      int(6),
    )
  };
  let gamma_of = |inner: Expr| call("Gamma", vec![inner]);
  let one_minus_g = call(
    "Plus",
    vec![int(1), call("Times", vec![int(-1), g.clone()])],
  );
  // (b + a g - b Gamma[1 - g])/g
  let mean_general = divide(
    call(
      "Plus",
      vec![
        b.clone(),
        call("Times", vec![a.clone(), g.clone()]),
        call(
          "Times",
          vec![int(-1), b.clone(), gamma_of(one_minus_g.clone())],
        ),
      ],
    ),
    g.clone(),
  );
  // b^2 (Gamma[1 - 2g] - Gamma[1 - g]^2)/g^2
  let one_minus_2g = call(
    "Plus",
    vec![int(1), call("Times", vec![int(-2), g.clone()])],
  );
  let var_general = divide(
    call(
      "Times",
      vec![
        power(b.clone(), int(2)),
        call(
          "Plus",
          vec![
            gamma_of(one_minus_2g),
            call("Times", vec![int(-1), power(gamma_of(one_minus_g), int(2))]),
          ],
        ),
      ],
    ),
    power(g.clone(), int(2)),
  );

  match ms_numeric(g) {
    Some(gv) if gv == 0.0 => Ok((eval(mean_gumbel())?, eval(var_gumbel())?)),
    Some(gv) => {
      let mean = if gv < 1.0 {
        eval(mean_general)?
      } else {
        indeterminate()
      };
      let var = if 2.0 * gv < 1.0 {
        eval(var_general)?
      } else {
        indeterminate()
      };
      Ok((mean, var))
    }
    None => {
      let mean = piecewise(
        vec![
          (
            mean_gumbel(),
            comparison(g.clone(), ComparisonOp::Equal, int(0)),
          ),
          (
            mean_general,
            call(
              "And",
              vec![
                comparison(g.clone(), ComparisonOp::NotEqual, int(0)),
                comparison(g.clone(), ComparisonOp::Less, int(1)),
              ],
            ),
          ),
        ],
        indeterminate(),
      );
      let var = piecewise(
        vec![
          (
            var_gumbel(),
            comparison(g.clone(), ComparisonOp::Equal, int(0)),
          ),
          (
            var_general,
            call(
              "And",
              vec![
                comparison(g.clone(), ComparisonOp::NotEqual, int(0)),
                comparison(
                  times(int(2), g.clone()),
                  ComparisonOp::Less,
                  int(1),
                ),
              ],
            ),
          ),
        ],
        indeterminate(),
      );
      Ok((mean, var))
    }
  }
}

/// Mirror helpers for MaxStableDistribution: u(x) = 1 + g (-a + x)/b
/// (negative numeric g folds the sign into (a - x)) and the (a - x)/b
/// Gumbel argument.
fn msx_u(a: &Expr, b: &Expr, g: &Expr, x: &Expr) -> Expr {
  let (coef, diff) = if ms_numeric(g).is_some_and(|v| v < 0.0) {
    (
      call("Times", vec![int(-1), g.clone()]),
      call(
        "Plus",
        vec![a.clone(), call("Times", vec![int(-1), x.clone()])],
      ),
    )
  } else {
    (
      g.clone(),
      call(
        "Plus",
        vec![call("Times", vec![int(-1), a.clone()]), x.clone()],
      ),
    )
  };
  plus(int(1), divide(call("Times", vec![coef, diff]), b.clone()))
}

fn msx_z(a: &Expr, b: &Expr, x: &Expr) -> Expr {
  divide(
    call(
      "Plus",
      vec![a.clone(), call("Times", vec![int(-1), x.clone()])],
    ),
    b.clone(),
  )
}

/// PDF[MaxStableDistribution[a, b, g], x] — mirror of MinStable.
fn pdf_max_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("MaxStableDistribution", dargs), x])
  };
  if dargs.len() != 3 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, b, g) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let gumbel = |at: &Expr| -> Expr {
    // E^(-E^((a - x)/b) - (-a + x)/b)/b
    let z = msx_z(&a, &b, at);
    let neg_za = divide(
      call(
        "Plus",
        vec![call("Times", vec![int(-1), a.clone()]), at.clone()],
      ),
      b.clone(),
    );
    divide(
      power(
        e(),
        call(
          "Plus",
          vec![neg(power(e(), z)), call("Times", vec![int(-1), neg_za])],
        ),
      ),
      b.clone(),
    )
  };
  let general = |at: &Expr| -> Expr {
    let u = msx_u(&a, &b, &g, at);
    let inv_g = power(g.clone(), int(-1));
    divide(
      power(
        u.clone(),
        call(
          "Plus",
          vec![int(-1), call("Times", vec![int(-1), inv_g.clone()])],
        ),
      ),
      call(
        "Times",
        vec![
          b.clone(),
          power(e(), power(u, call("Times", vec![int(-1), inv_g]))),
        ],
      ),
    )
  };

  let g_num = ms_numeric(&g);
  let numeric_x = ms_numeric(&x).is_some();
  match g_num {
    Some(gv) if gv == 0.0 => {
      if numeric_x {
        return eval(gumbel(&x));
      }
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      // Exponent -E^z + z with z = (a - x)/b shared between both
      // terms, matching wolframscript's folded print
      let z = eval(msx_z(&a, &b, &x))?;
      let body = power(e(), call("Plus", vec![neg(power(e(), z.clone())), z]));
      let b_is_one = matches!(&b, Expr::Integer(1));
      Ok(if b_is_one {
        body
      } else {
        divide(body, b.clone())
      })
    }
    Some(_) => {
      if numeric_x {
        let u_val = eval(msx_u(&a, &b, &g, &x))?;
        if ms_numeric(&u_val).is_some_and(|v| v <= 0.0) {
          return Ok(int(0));
        }
        return eval(general(&x));
      }
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let cond =
        comparison(eval(msx_u(&a, &b, &g, &x))?, ComparisonOp::Greater, int(0));
      Ok(piecewise(vec![(eval(general(&x))?, cond)], int(0)))
    }
    None => {
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let p1 = (
        gumbel(&x),
        comparison(g.clone(), ComparisonOp::Equal, int(0)),
      );
      let p2 = (
        general(&x),
        call(
          "And",
          vec![
            comparison(g.clone(), ComparisonOp::NotEqual, int(0)),
            comparison(msx_u(&a, &b, &g, &x), ComparisonOp::Greater, int(0)),
          ],
        ),
      );
      Ok(piecewise(vec![p1, p2], int(0)))
    }
  }
}

/// CDF[MaxStableDistribution[a, b, g], x] = E^(-u^(-1/g)) on u > 0,
/// with 0 below the support for g > 0 and 1 above it for g < 0.
fn cdf_max_stable(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("MaxStableDistribution", dargs), x])
  };
  if dargs.len() != 3 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, b, g) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let gumbel =
    |at: &Expr| -> Expr { power(e(), neg(power(e(), msx_z(&a, &b, at)))) };
  let general = |at: &Expr| -> Expr {
    let u = msx_u(&a, &b, &g, at);
    power(
      e(),
      neg(power(
        u,
        call("Times", vec![int(-1), power(g.clone(), int(-1))]),
      )),
    )
  };

  let g_num = ms_numeric(&g);
  let numeric_x = ms_numeric(&x).is_some();
  match g_num {
    Some(gv) if gv == 0.0 => {
      if numeric_x {
        return eval(gumbel(&x));
      }
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      // Raw with the inner argument evaluated; an outer eval would
      // hoist E^(-x) into 1/E^x
      let z = eval(msx_z(&a, &b, &x))?;
      Ok(power(e(), neg(power(e(), z))))
    }
    Some(gv) => {
      if numeric_x {
        let u_val = eval(msx_u(&a, &b, &g, &x))?;
        if ms_numeric(&u_val).is_some_and(|v| v <= 0.0) {
          return Ok(if gv > 0.0 { int(0) } else { int(1) });
        }
        return eval(general(&x));
      }
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let cond =
        comparison(eval(msx_u(&a, &b, &g, &x))?, ComparisonOp::Greater, int(0));
      Ok(piecewise(
        vec![(eval(general(&x))?, cond)],
        if gv > 0.0 { int(0) } else { int(1) },
      ))
    }
    None => {
      if !matches!(&x, Expr::Identifier(_)) {
        return Ok(unevaluated(dargs, x));
      }
      let u = msx_u(&a, &b, &g, &x);
      let p1 = (
        gumbel(&x),
        comparison(g.clone(), ComparisonOp::Equal, int(0)),
      );
      let p2 = (
        general(&x),
        call(
          "And",
          vec![
            comparison(g.clone(), ComparisonOp::NotEqual, int(0)),
            comparison(u.clone(), ComparisonOp::Greater, int(0)),
          ],
        ),
      );
      let p3 = (
        int(0),
        call(
          "And",
          vec![
            comparison(g.clone(), ComparisonOp::Greater, int(0)),
            comparison(u, ComparisonOp::LessEqual, int(0)),
          ],
        ),
      );
      Ok(piecewise(vec![p1, p2, p3], int(1)))
    }
  }
}

/// Mean and variance for MaxStableDistribution (the variance matches
/// MinStable; the mean mirrors its sign structure).
fn max_stable_mean_variance(
  a: &Expr,
  b: &Expr,
  g: &Expr,
) -> Result<(Expr, Expr), InterpreterError> {
  let euler_gamma = Expr::Identifier("EulerGamma".to_string());
  let mean_gumbel = || plus(a.clone(), times(b.clone(), euler_gamma.clone()));
  let one_minus_g = call(
    "Plus",
    vec![int(1), call("Times", vec![int(-1), g.clone()])],
  );
  // (-b + a g + b Gamma[1 - g])/g
  let mean_general = divide(
    call(
      "Plus",
      vec![
        call("Times", vec![int(-1), b.clone()]),
        call("Times", vec![a.clone(), g.clone()]),
        call(
          "Times",
          vec![b.clone(), call("Gamma", vec![one_minus_g.clone()])],
        ),
      ],
    ),
    g.clone(),
  );
  // Variance is identical to MinStable's
  let (_, var) = min_stable_mean_variance(a, b, g)?;
  match ms_numeric(g) {
    Some(gv) if gv == 0.0 => Ok((eval(mean_gumbel())?, var)),
    Some(gv) => {
      let mean = if gv < 1.0 {
        eval(mean_general)?
      } else {
        indeterminate()
      };
      Ok((mean, var))
    }
    None => {
      let mean = piecewise(
        vec![
          (
            mean_gumbel(),
            comparison(g.clone(), ComparisonOp::Equal, int(0)),
          ),
          (
            mean_general,
            call(
              "And",
              vec![
                comparison(g.clone(), ComparisonOp::NotEqual, int(0)),
                comparison(g.clone(), ComparisonOp::Less, int(1)),
              ],
            ),
          ),
        ],
        indeterminate(),
      );
      Ok((mean, var))
    }
  }
}

/// Parse TriangularDistribution arguments: {a, b} with optional mode c
/// (default (a + b)/2; no arguments means {0, 1}).
fn triangular_params(
  dargs: &[Expr],
) -> Result<Option<(Expr, Expr, Expr)>, InterpreterError> {
  let (a, b, c) = match dargs {
    [] => (int(0), int(1), None),
    [Expr::List(bounds)] if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone(), None)
    }
    [Expr::List(bounds), c] if bounds.len() == 2 => {
      (bounds[0].clone(), bounds[1].clone(), Some(c.clone()))
    }
    _ => return Ok(None),
  };
  let c = match c {
    Some(c) => c,
    None => eval(divide(plus(a.clone(), b.clone()), int(2)))?,
  };
  Ok(Some((a, b, c)))
}

/// PDF[TriangularDistribution[{a, b}, c], x]: rising piece
/// 2(x-a)/((b-a)(c-a)) on a <= x <= c and falling piece
/// 2(b-x)/((b-a)(b-c)) on c < x <= b, with the numeric coefficient
/// evaluated but the (b - x) factor kept unexpanded, as wolframscript
/// prints it.
fn pdf_triangular(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("TriangularDistribution", dargs), x])
  };
  let Some((a, b, c)) = triangular_params(dargs)? else {
    return Ok(unevaluated(dargs, x));
  };
  let diff_xa = call(
    "Plus",
    vec![call("Times", vec![int(-1), a.clone()]), x.clone()],
  );
  let diff_bx = call(
    "Plus",
    vec![b.clone(), call("Times", vec![int(-1), x.clone()])],
  );
  let numeric = ms_numeric(&a).is_some()
    && ms_numeric(&b).is_some()
    && ms_numeric(&c).is_some();
  let (p1, p2) = if numeric {
    // Times[coef, factor] keeps the linear factor unexpanded
    let coef1 = eval(divide(
      int(2),
      times(
        plus(b.clone(), times(int(-1), a.clone())),
        plus(c.clone(), times(int(-1), a.clone())),
      ),
    ))?;
    let coef2 = eval(divide(
      int(2),
      times(
        plus(b.clone(), times(int(-1), a.clone())),
        plus(b.clone(), times(int(-1), c.clone())),
      ),
    ))?;
    (
      eval(call("Times", vec![coef1, diff_xa]))?,
      eval(call("Times", vec![coef2, diff_bx]))?,
    )
  } else {
    let den1 = call(
      "Times",
      vec![
        call(
          "Plus",
          vec![call("Times", vec![int(-1), a.clone()]), b.clone()],
        ),
        call(
          "Plus",
          vec![call("Times", vec![int(-1), a.clone()]), c.clone()],
        ),
      ],
    );
    let den2 = call(
      "Times",
      vec![
        call(
          "Plus",
          vec![call("Times", vec![int(-1), a.clone()]), b.clone()],
        ),
        call(
          "Plus",
          vec![b.clone(), call("Times", vec![int(-1), c.clone()])],
        ),
      ],
    );
    (
      divide(call("Times", vec![int(2), diff_xa]), den1),
      divide(call("Times", vec![int(2), diff_bx]), den2),
    )
  };

  if ms_numeric(&x).is_some() {
    // Pointwise: pick the piece by comparing against a, c, b
    let (xv, av, bv, cv) = (
      ms_numeric(&x).unwrap(),
      ms_numeric(&a).unwrap_or(f64::NAN),
      ms_numeric(&b).unwrap_or(f64::NAN),
      ms_numeric(&c).unwrap_or(f64::NAN),
    );
    if !numeric {
      return Ok(unevaluated(dargs, x));
    }
    if xv >= av && xv <= cv {
      return eval(p1);
    }
    if xv > cv && xv <= bv {
      return eval(p2);
    }
    return Ok(int(0));
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let cond1 = comparison3(
    a,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::LessEqual,
    c.clone(),
  );
  let cond2 = comparison3(c, ComparisonOp::Less, x, ComparisonOp::LessEqual, b);
  Ok(piecewise(vec![(p1, cond1), (p2, cond2)], int(0)))
}

/// CDF[TriangularDistribution[{a, b}, c], x] with the quadratic pieces
/// and a third {1, x > b} piece.
fn cdf_triangular(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("TriangularDistribution", dargs), x])
  };
  let Some((a, b, c)) = triangular_params(dargs)? else {
    return Ok(unevaluated(dargs, x));
  };
  let sq_xa = power(
    call(
      "Plus",
      vec![call("Times", vec![int(-1), a.clone()]), x.clone()],
    ),
    int(2),
  );
  let sq_bx = power(
    call(
      "Plus",
      vec![b.clone(), call("Times", vec![int(-1), x.clone()])],
    ),
    int(2),
  );
  let numeric = ms_numeric(&a).is_some()
    && ms_numeric(&b).is_some()
    && ms_numeric(&c).is_some();
  let (p1, p2) = if numeric {
    let coef1 = eval(divide(
      int(1),
      times(
        plus(b.clone(), times(int(-1), a.clone())),
        plus(c.clone(), times(int(-1), a.clone())),
      ),
    ))?;
    let coef2 = eval(divide(
      int(1),
      times(
        plus(b.clone(), times(int(-1), a.clone())),
        plus(b.clone(), times(int(-1), c.clone())),
      ),
    ))?;
    (
      eval(call("Times", vec![coef1, sq_xa]))?,
      call(
        "Plus",
        vec![
          int(1),
          call(
            "Times",
            vec![int(-1), eval(call("Times", vec![coef2, sq_bx]))?],
          ),
        ],
      ),
    )
  } else {
    let den1 = call(
      "Times",
      vec![
        call(
          "Plus",
          vec![call("Times", vec![int(-1), a.clone()]), b.clone()],
        ),
        call(
          "Plus",
          vec![call("Times", vec![int(-1), a.clone()]), c.clone()],
        ),
      ],
    );
    let den2 = call(
      "Times",
      vec![
        call(
          "Plus",
          vec![call("Times", vec![int(-1), a.clone()]), b.clone()],
        ),
        call(
          "Plus",
          vec![b.clone(), call("Times", vec![int(-1), c.clone()])],
        ),
      ],
    );
    (
      divide(sq_xa, den1),
      call(
        "Plus",
        vec![int(1), call("Times", vec![int(-1), divide(sq_bx, den2)])],
      ),
    )
  };

  if ms_numeric(&x).is_some() {
    if !numeric {
      return Ok(unevaluated(dargs, x));
    }
    let (xv, av, bv, cv) = (
      ms_numeric(&x).unwrap(),
      ms_numeric(&a).unwrap(),
      ms_numeric(&b).unwrap(),
      ms_numeric(&c).unwrap(),
    );
    if xv < av {
      return Ok(int(0));
    }
    if xv > bv {
      return Ok(int(1));
    }
    if xv <= cv {
      return eval(p1);
    }
    return eval(p2);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let cond1 = comparison3(
    a,
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::LessEqual,
    c.clone(),
  );
  let cond2 = comparison3(
    c,
    ComparisonOp::Less,
    x.clone(),
    ComparisonOp::LessEqual,
    b.clone(),
  );
  let cond3 = comparison(x, ComparisonOp::Greater, b);
  Ok(piecewise(
    vec![(p1, cond1), (p2, cond2), (int(1), cond3)],
    int(0),
  ))
}

/// Mean (a + b + c)/3 and variance
/// (a^2 - a b + b^2 - a c - b c + c^2)/18 for TriangularDistribution.
fn triangular_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some((a, b, c)) = triangular_params(dargs)? else {
    return Err(InterpreterError::EvaluationError(
      "TriangularDistribution expects {min, max} and an optional mode".into(),
    ));
  };
  // The general (3-parameter) formulas; Simplify collapses the symmetric
  // 2-argument form TriangularDistribution[{a, b}] (mode c = (a+b)/2) to
  // wolframscript's Mean (a+b)/2 and Variance (b-a)^2/24, while leaving the
  // genuine 3-parameter forms unchanged.
  let mean = eval(unary_fn(
    "Simplify",
    divide(plus(plus(a.clone(), b.clone()), c.clone()), int(3)),
  ))?;
  let var = eval(unary_fn(
    "Simplify",
    divide(
      call(
        "Plus",
        vec![
          power(a.clone(), int(2)),
          times(int(-1), times(a.clone(), b.clone())),
          power(b.clone(), int(2)),
          times(int(-1), times(a.clone(), c.clone())),
          times(int(-1), times(b.clone(), c.clone())),
          power(c, int(2)),
        ],
      ),
      int(18),
    ),
  ))?;
  Ok((mean, var))
}

/// Coefficient r*Sqrt[2/Pi]*num/E-part in wolframscript's canonical
/// radical form: powers of two in r's denominator merge into a
/// Sqrt[2*Pi] denominator (1/8 -> 1/(4 Sqrt[2 Pi])); everything else
/// keeps the Sqrt[2/Pi] factor.
fn maxwell_term(
  r_num: i128,
  r_den: i128,
  numerator: Expr,
  e_part: Expr,
) -> Expr {
  let mut k = 0;
  let mut m = r_den;
  while m % 2 == 0 {
    m /= 2;
    k += 1;
  }
  if k >= 1 {
    // p num / (m 2^(k-1) Sqrt[2 Pi] E-part)
    let c = m * (1 << (k - 1));
    let num_expr = if r_num == 1 {
      numerator
    } else {
      call("Times", vec![int(r_num), numerator])
    };
    let mut den_factors: Vec<Expr> = Vec::new();
    if c > 1 {
      den_factors.push(int(c));
    }
    den_factors.push(e_part);
    den_factors.push(call("Sqrt", vec![times(int(2), pi())]));
    divide(num_expr, call("Times", den_factors))
  } else {
    // (p Sqrt[2/Pi] num)/(m E-part)
    let mut num_factors: Vec<Expr> = Vec::new();
    if r_num != 1 {
      num_factors.push(int(r_num));
    }
    num_factors.push(call("Sqrt", vec![divide(int(2), pi())]));
    num_factors.push(numerator);
    let den = if m > 1 {
      call("Times", vec![int(m), e_part])
    } else {
      e_part
    };
    divide(call("Times", num_factors), den)
  }
}

/// Exact rational (p, q > 0) of an Integer/Rational Expr.
fn maxwell_rational(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(v) => Some((*v, 1)),
    Expr::FunctionCall { name, args } if name == "Rational" => {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q > 0 => Some((*p, *q)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// PDF[MaxwellDistribution[s], x] =
/// Piecewise[{{Sqrt[2/Pi] x^2 E^(-x^2/(2 s^2))/s^3, x > 0}}, 0].
fn pdf_maxwell(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("MaxwellDistribution", dargs), x])
  };
  if dargs.len() != 1 {
    return Ok(unevaluated(dargs, x));
  }
  let s = dargs[0].clone();
  let sqrt_2_pi = call("Sqrt", vec![divide(int(2), pi())]);
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let e_part = power(
      e(),
      eval(divide(
        power(at.clone(), int(2)),
        times(int(2), power(s.clone(), int(2))),
      ))?,
    );
    // Exact rational scale: 1/s^3 in wolframscript's canonical
    // radical form; symbolic or real s falls back to plain evaluation
    if ms_numeric(at).is_none()
      && let Some((sp, sq)) = maxwell_rational(&s)
    {
      // r = 1/s^3 = sq^3/sp^3 (sign of s is positive for a scale)
      return Ok(maxwell_term(
        sq * sq * sq,
        sp * sp * sp,
        power(at.clone(), int(2)),
        e_part,
      ));
    }
    eval(divide(
      call("Times", vec![sqrt_2_pi.clone(), power(at.clone(), int(2))]),
      call(
        "Times",
        vec![
          power(
            e(),
            divide(
              power(at.clone(), int(2)),
              times(int(2), power(s.clone(), int(2))),
            ),
          ),
          power(s.clone(), int(3)),
        ],
      ),
    ))
  };
  if ms_numeric(&x).is_some() {
    if ms_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// CDF[MaxwellDistribution[s], x] =
/// Piecewise[{{-Sqrt[2/Pi] x E^(-x^2/(2 s^2))/s + Erf[x/(Sqrt[2] s)],
/// x > 0}}, 0].
fn cdf_maxwell(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("MaxwellDistribution", dargs), x])
  };
  if dargs.len() != 1 {
    return Ok(unevaluated(dargs, x));
  }
  let s = dargs[0].clone();
  let sqrt_2_pi = call("Sqrt", vec![divide(int(2), pi())]);
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    if ms_numeric(at).is_none()
      && let Some((sp, sq)) = maxwell_rational(&s)
      && (sp, sq) != (1, 1)
    {
      let e_part = power(
        e(),
        eval(divide(
          power(at.clone(), int(2)),
          times(int(2), power(s.clone(), int(2))),
        ))?,
      );
      let term1 = neg(maxwell_term(sq, sp, at.clone(), e_part));
      let erf = call(
        "Erf",
        vec![eval(divide(
          at.clone(),
          call("Times", vec![call("Sqrt", vec![int(2)]), s.clone()]),
        ))?],
      );
      return Ok(call("Plus", vec![term1, erf]));
    }
    eval(call(
      "Plus",
      vec![
        call(
          "Times",
          vec![
            int(-1),
            divide(
              call("Times", vec![sqrt_2_pi.clone(), at.clone()]),
              call(
                "Times",
                vec![
                  power(
                    e(),
                    divide(
                      power(at.clone(), int(2)),
                      times(int(2), power(s.clone(), int(2))),
                    ),
                  ),
                  s.clone(),
                ],
              ),
            ),
          ],
        ),
        call(
          "Erf",
          vec![divide(
            at.clone(),
            call("Times", vec![call("Sqrt", vec![int(2)]), s.clone()]),
          )],
        ),
      ],
    ))
  };
  if ms_numeric(&x).is_some() {
    if ms_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// PDF[BirnbaumSaundersDistribution[a, l], x] =
/// Piecewise[{{(1 + l x)/(2 a E^((-1 + l x)^2/(2 a^2 l x)) Sqrt[2 Pi]
/// Sqrt[l x^3]), x > 0}}, 0].
fn pdf_birnbaum_saunders(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("BirnbaumSaundersDistribution", dargs), x],
    )
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let a = dargs[0].clone();
  let l = dargs[1].clone();
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    // exponent (-1 + l at)^2 / (2 a^2 l at)
    let exponent = divide(
      power(plus(int(-1), times(l.clone(), at.clone())), int(2)),
      times(
        times(int(2), power(a.clone(), int(2))),
        times(l.clone(), at.clone()),
      ),
    );
    let denom = times(
      times(int(2), a.clone()),
      times(
        times(
          power(e(), exponent),
          call("Sqrt", vec![times(int(2), pi())]),
        ),
        call("Sqrt", vec![times(l.clone(), power(at.clone(), int(3)))]),
      ),
    );
    eval(divide(plus(int(1), times(l.clone(), at.clone())), denom))
  };
  if ms_numeric(&x).is_some() {
    if ms_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// CDF[BirnbaumSaundersDistribution[a, l], x] =
/// Piecewise[{{(1 + Erf[(-1 + l x)/(Sqrt[2] a Sqrt[l x])])/2, x > 0}}, 0].
fn cdf_birnbaum_saunders(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "CDF",
      vec![unevaluated("BirnbaumSaundersDistribution", dargs), x],
    )
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let a = dargs[0].clone();
  let l = dargs[1].clone();
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    // Erf[(-1 + l at)/(Sqrt[2] a Sqrt[l at])]
    let erf_arg = divide(
      plus(int(-1), times(l.clone(), at.clone())),
      times(
        times(call("Sqrt", vec![int(2)]), a.clone()),
        call("Sqrt", vec![times(l.clone(), at.clone())]),
      ),
    );
    eval(divide(plus(int(1), call("Erf", vec![erf_arg])), int(2)))
  };
  if ms_numeric(&x).is_some() {
    if ms_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// PDF[LevyDistribution[m, s], x] =
/// Piecewise[{{(s/(-m + x))^(3/2)/(E^(s/(2 (-m + x))) Sqrt[2 Pi] s),
/// -m + x > 0}}, 0].
fn pdf_levy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("LevyDistribution", dargs), x])
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let m = dargs[0].clone();
  let s = dargs[1].clone();
  let shift = |at: &Expr| plus(times(int(-1), m.clone()), at.clone());
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let sh = shift(at);
    let num = power(divide(s.clone(), sh.clone()), divide(int(3), int(2)));
    let denom = times(
      times(
        power(e(), divide(s.clone(), times(int(2), sh))),
        call("Sqrt", vec![times(int(2), pi())]),
      ),
      s.clone(),
    );
    eval(divide(num, denom))
  };
  // Decide the support only when both location and point are numeric.
  if let (Some(mv), Some(xv)) = (ms_numeric(&m), ms_numeric(&x)) {
    if xv <= mv {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(shift(&x), ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// CDF[LevyDistribution[m, s], x] =
/// Piecewise[{{Erfc[Sqrt[s/(-m + x)]/Sqrt[2]], -m + x > 0}}, 0].
fn cdf_levy(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("LevyDistribution", dargs), x])
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let m = dargs[0].clone();
  let s = dargs[1].clone();
  let shift = |at: &Expr| plus(times(int(-1), m.clone()), at.clone());
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let erfc_arg = divide(
      call("Sqrt", vec![divide(s.clone(), shift(at))]),
      call("Sqrt", vec![int(2)]),
    );
    eval(call("Erfc", vec![erfc_arg]))
  };
  if let (Some(mv), Some(xv)) = (ms_numeric(&m), ms_numeric(&x)) {
    if xv <= mv {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(shift(&x), ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// PDF[LindleyDistribution[d], x] =
/// Piecewise[{{(d^2 (1 + x))/((1 + d) E^(d x)), x > 0}}, 0].
fn pdf_lindley(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("LindleyDistribution", dargs), x])
  };
  if dargs.len() != 1 {
    return Ok(unevaluated(dargs, x));
  }
  let d = dargs[0].clone();
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let num = times(power(d.clone(), int(2)), plus(int(1), at.clone()));
    let denom = times(
      plus(int(1), d.clone()),
      power(e(), times(d.clone(), at.clone())),
    );
    eval(divide(num, denom))
  };
  if ms_numeric(&x).is_some() {
    if ms_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// CDF[LindleyDistribution[d], x] =
/// Piecewise[{{1 - (1 + d + d x)/((1 + d) E^(d x)), x > 0}}, 0].
fn cdf_lindley(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("LindleyDistribution", dargs), x])
  };
  if dargs.len() != 1 {
    return Ok(unevaluated(dargs, x));
  }
  let d = dargs[0].clone();
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    // 1 - (1 + d + d x)/((1 + d) E^(d x))
    let num = plus(plus(int(1), d.clone()), times(d.clone(), at.clone()));
    let denom = times(
      plus(int(1), d.clone()),
      power(e(), times(d.clone(), at.clone())),
    );
    eval(minus(int(1), divide(num, denom)))
  };
  if ms_numeric(&x).is_some() {
    if ms_numeric(&x).is_some_and(|v| v <= 0.0) {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let piece = body(&x)?;
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  Ok(piecewise(vec![(piece, cond)], int(0)))
}

/// Mean 2 Sqrt[2/Pi] s and variance (3 Pi - 8) s^2/Pi for
/// MaxwellDistribution.
pub fn maxwell_mean_variance(
  s: &Expr,
) -> Result<(Expr, Expr), InterpreterError> {
  let mean = eval(call(
    "Times",
    vec![int(2), call("Sqrt", vec![divide(int(2), pi())]), s.clone()],
  ))?;
  let var = if ms_numeric(s).is_some() {
    eval(divide(
      call(
        "Times",
        vec![
          call("Plus", vec![int(-8), call("Times", vec![int(3), pi()])]),
          power(s.clone(), int(2)),
        ],
      ),
      pi(),
    ))?
  } else {
    // wolframscript puts the Pi-sum factor first; assembled raw since
    // evaluation would reorder it
    Expr::Raw(format!("((-8 + 3*Pi)*{}^2)/Pi", expr_to_string(s)))
  };
  Ok((mean, var))
}

/// Parse WignerSemicircleDistribution arguments: [r] or [a, r].
fn wigner_params(dargs: &[Expr]) -> Option<(Expr, Expr)> {
  match dargs {
    [r] => Some((int(0), r.clone())),
    [a, r] => Some((a.clone(), r.clone())),
    _ => None,
  }
}

/// PDF[WignerSemicircleDistribution[a, r], x] =
/// Piecewise[{{2 Sqrt[1 - (x-a)^2/r^2]/(Pi r), a - r < x < a + r}}, 0].
fn pdf_wigner_semicircle(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("WignerSemicircleDistribution", dargs), x],
    )
  };
  let Some((a, r)) = wigner_params(dargs) else {
    return Ok(unevaluated(dargs, x));
  };
  let diff = |at: &Expr| -> Expr {
    if matches!(&a, Expr::Integer(0)) {
      at.clone()
    } else {
      call(
        "Plus",
        vec![call("Times", vec![int(-1), a.clone()]), at.clone()],
      )
    }
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let inner = call(
      "Plus",
      vec![
        int(1),
        call(
          "Times",
          vec![
            int(-1),
            divide(power(diff(at), int(2)), power(r.clone(), int(2))),
          ],
        ),
      ],
    );
    eval(divide(
      call("Times", vec![int(2), call("Sqrt", vec![inner])]),
      call("Times", vec![pi(), r.clone()]),
    ))
  };
  let numeric_params = ms_numeric(&a).is_some() && ms_numeric(&r).is_some();
  if ms_numeric(&x).is_some() {
    if !numeric_params {
      return Ok(unevaluated(dargs, x));
    }
    let (xv, av, rv) = (
      ms_numeric(&x).unwrap(),
      ms_numeric(&a).unwrap(),
      ms_numeric(&r).unwrap(),
    );
    if (xv - av).abs() >= rv {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let lo = eval(plus(a.clone(), times(int(-1), r.clone())))?;
  let hi = eval(plus(a.clone(), r.clone()))?;
  let cond =
    comparison3(lo, ComparisonOp::Less, x.clone(), ComparisonOp::Less, hi);
  Ok(piecewise(vec![(body(&x)?, cond)], int(0)))
}

/// CDF[WignerSemicircleDistribution[a, r], x] =
/// 1/2 + (x-a) Sqrt[1 - (x-a)^2/r^2]/(Pi r) + ArcSin[(x-a)/r]/Pi
/// inside the support, 1 at and above a + r.
fn cdf_wigner_semicircle(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "CDF",
      vec![unevaluated("WignerSemicircleDistribution", dargs), x],
    )
  };
  let Some((a, r)) = wigner_params(dargs) else {
    return Ok(unevaluated(dargs, x));
  };
  let diff = |at: &Expr| -> Expr {
    if matches!(&a, Expr::Integer(0)) {
      at.clone()
    } else {
      call(
        "Plus",
        vec![call("Times", vec![int(-1), a.clone()]), at.clone()],
      )
    }
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let d = diff(at);
    let inner = call(
      "Plus",
      vec![
        int(1),
        call(
          "Times",
          vec![
            int(-1),
            divide(power(d.clone(), int(2)), power(r.clone(), int(2))),
          ],
        ),
      ],
    );
    // The x-before-Sqrt factor order matches wolframscript, so the
    // middle term stays raw with evaluated subparts
    let sqrt_part = call("Sqrt", vec![eval(inner)?]);
    let denom = eval(call("Times", vec![pi(), r.clone()]))?;
    let arcsin_arg = eval(divide(d.clone(), r.clone()))?;
    Ok(call(
      "Plus",
      vec![
        call("Rational", vec![int(1), int(2)]),
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          // Wolfram puts the Sqrt factor first when the shifted
          // variable leads with a number ((-3 + x)), and last when it
          // is bare x or leads with a symbol ((-a + x))
          left: Box::new(if ms_numeric(&a).is_some_and(|v| v != 0.0) {
            call("Times", vec![sqrt_part, d])
          } else {
            call("Times", vec![d, sqrt_part])
          }),
          right: Box::new(denom),
        },
        divide(call("ArcSin", vec![arcsin_arg]), pi()),
      ],
    ))
  };
  let numeric_params = ms_numeric(&a).is_some() && ms_numeric(&r).is_some();
  if ms_numeric(&x).is_some() {
    if !numeric_params {
      return Ok(unevaluated(dargs, x));
    }
    let (xv, av, rv) = (
      ms_numeric(&x).unwrap(),
      ms_numeric(&a).unwrap(),
      ms_numeric(&r).unwrap(),
    );
    if xv <= av - rv {
      return Ok(int(0));
    }
    if xv >= av + rv {
      return Ok(int(1));
    }
    return eval(body(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let lo = eval(plus(a.clone(), times(int(-1), r.clone())))?;
  let hi = eval(plus(a.clone(), r.clone()))?;
  let cond_in = comparison3(
    lo,
    ComparisonOp::Less,
    x.clone(),
    ComparisonOp::Less,
    hi.clone(),
  );
  let cond_ge = comparison(x.clone(), ComparisonOp::GreaterEqual, hi);
  Ok(piecewise(
    vec![(body(&x)?, cond_in), (int(1), cond_ge)],
    int(0),
  ))
}

/// Mean a and variance r^2/4 for WignerSemicircleDistribution.
fn wigner_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some((a, r)) = wigner_params(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "WignerSemicircleDistribution expects [r] or [a, r]".into(),
    ));
  };
  let var = eval(divide(power(r, int(2)), int(4)))?;
  Ok((a, var))
}

/// Parse SechDistribution arguments: [] is (0, 1), otherwise [m, s].
fn sech_params(dargs: &[Expr]) -> Option<(Expr, Expr)> {
  match dargs {
    [] => Some((int(0), int(1))),
    [m, s] => Some((m.clone(), s.clone())),
    _ => None,
  }
}

/// The Pi (x - m)/(2 s) argument shared by the Sech PDF and CDF.
fn sech_arg(m: &Expr, s: &Expr, x: &Expr) -> Result<Expr, InterpreterError> {
  let diff = if matches!(m, Expr::Integer(0)) {
    x.clone()
  } else {
    call(
      "Plus",
      vec![call("Times", vec![int(-1), m.clone()]), x.clone()],
    )
  };
  eval(divide(
    call("Times", vec![pi(), diff]),
    times(int(2), s.clone()),
  ))
}

/// PDF[SechDistribution[m, s], x] = Sech[Pi (x - m)/(2 s)]/(2 s).
fn pdf_sech(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("SechDistribution", dargs), x])
  };
  let Some((m, s)) = sech_params(dargs) else {
    return Ok(unevaluated(dargs, x));
  };
  if !matches!(&x, Expr::Identifier(_)) && ms_numeric(&x).is_none() {
    return Ok(unevaluated(dargs, x));
  }
  let arg = sech_arg(&m, &s, &x)?;
  eval(divide(call("Sech", vec![arg]), times(int(2), s)))
}

/// CDF[SechDistribution[m, s], x] =
/// 2 ArcTan[E^(Pi (x - m)/(2 s))]/Pi.
fn cdf_sech(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("SechDistribution", dargs), x])
  };
  let Some((m, s)) = sech_params(dargs) else {
    return Ok(unevaluated(dargs, x));
  };
  if !matches!(&x, Expr::Identifier(_)) && ms_numeric(&x).is_none() {
    return Ok(unevaluated(dargs, x));
  }
  let arg = sech_arg(&m, &s, &x)?;
  eval(divide(
    call("Times", vec![int(2), call("ArcTan", vec![power(e(), arg)])]),
    pi(),
  ))
}

/// Mean m and variance s^2 for SechDistribution.
fn sech_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let Some((m, s)) = sech_params(dargs) else {
    return Err(InterpreterError::EvaluationError(
      "SechDistribution expects no arguments or [m, s]".into(),
    ));
  };
  Ok((m, eval(power(s, int(2)))?))
}

/// PDF[MoyalDistribution[m, s], x] =
/// E^(-(1/2) E^(-(x-m)/s) - (x-m)/(2 s))/(Sqrt[2 Pi] s), printed with
/// wolframscript's sign folding: numeric nonzero m keeps the
/// (m - x)/s exponent, m = 0 and symbolic m flip into the
/// 1/E^((-m + x)/s) reciprocal.
fn pdf_moyal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("MoyalDistribution", dargs), x])
  };
  let (m, s) = match dargs {
    [] => (int(0), int(1)),
    [m, s] => (m.clone(), s.clone()),
    _ => return Ok(unevaluated(dargs, x)),
  };
  let half = || call("Rational", vec![int(-1), int(2)]);
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let folded = ms_numeric(&m).is_some_and(|v| v != 0.0);
    let exponent = if folded {
      // -1/2 E^((m - x)/s) + (m - x)/(2 s)
      let neg_z = eval(divide(
        call(
          "Plus",
          vec![m.clone(), call("Times", vec![int(-1), at.clone()])],
        ),
        s.clone(),
      ))?;
      let half_neg_z = eval(divide(
        call(
          "Plus",
          vec![m.clone(), call("Times", vec![int(-1), at.clone()])],
        ),
        times(int(2), s.clone()),
      ))?;
      call(
        "Plus",
        vec![call("Times", vec![half(), power(e(), neg_z)]), half_neg_z],
      )
    } else {
      // -1/2 1/E^((-m + x)/s) - (-m + x)/(2 s)
      let diff = if matches!(&m, Expr::Integer(0)) {
        at.clone()
      } else {
        call(
          "Plus",
          vec![call("Times", vec![int(-1), m.clone()]), at.clone()],
        )
      };
      let z = eval(divide(diff.clone(), s.clone()))?;
      let z_half = eval(divide(diff, times(int(2), s.clone())))?;
      call(
        "Plus",
        vec![
          call("Times", vec![half(), divide(int(1), power(e(), z))]),
          call("Times", vec![int(-1), z_half]),
        ],
      )
    };
    let mut den_factors: Vec<Expr> = Vec::new();
    if let Some(sv) = ms_numeric(&s) {
      if sv != 1.0 {
        den_factors.push(s.clone());
      }
    } else {
      den_factors.push(s.clone());
    }
    // wolframscript orders the numeric scale before Sqrt[2 Pi] but a
    // symbolic one after it
    let sqrt_2pi = call("Sqrt", vec![times(int(2), pi())]);
    let den = if den_factors.is_empty() {
      sqrt_2pi
    } else if ms_numeric(&s).is_some() {
      call("Times", vec![den_factors.remove(0), sqrt_2pi])
    } else {
      call("Times", vec![sqrt_2pi, den_factors.remove(0)])
    };
    Ok(divide(power(e(), exponent), den))
  };
  if ms_numeric(&x).is_some() {
    return eval(body(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  body(&x)
}

/// CDF[MoyalDistribution[m, s], x] = Erfc[E^(-(x-m)/(2 s))/Sqrt[2]].
fn cdf_moyal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("MoyalDistribution", dargs), x])
  };
  let (m, s) = match dargs {
    [] => (int(0), int(1)),
    [m, s] => (m.clone(), s.clone()),
    _ => return Ok(unevaluated(dargs, x)),
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let diff = if matches!(&m, Expr::Integer(0)) {
      at.clone()
    } else {
      call(
        "Plus",
        vec![call("Times", vec![int(-1), m.clone()]), at.clone()],
      )
    };
    let z_half = eval(divide(diff, times(int(2), s.clone())))?;
    Ok(call(
      "Erfc",
      vec![divide(
        int(1),
        call(
          "Times",
          vec![call("Sqrt", vec![int(2)]), power(e(), z_half)],
        ),
      )],
    ))
  };
  if ms_numeric(&x).is_some() {
    return eval(body(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  body(&x)
}

/// Mean m + s (EulerGamma + Log[2]) and variance Pi^2 s^2/2 for
/// MoyalDistribution (the mean is returned raw to keep wolframscript's
/// s-before-sum factor order).
pub fn moyal_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let (m, s) = match dargs {
    [] => (int(0), int(1)),
    [m, s] => (m.clone(), s.clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MoyalDistribution expects no arguments or [m, s]".into(),
      ));
    }
  };
  let sum = call(
    "Plus",
    vec![
      Expr::Identifier("EulerGamma".to_string()),
      call("Log", vec![int(2)]),
    ],
  );
  let mean = if matches!(&m, Expr::Integer(0)) && matches!(&s, Expr::Integer(1))
  {
    eval(sum)?
  } else {
    call("Plus", vec![m.clone(), call("Times", vec![s.clone(), sum])])
  };
  let var = eval(divide(
    call("Times", vec![power(pi(), int(2)), power(s, int(2))]),
    int(2),
  ))?;
  Ok((mean, var))
}

/// PDF[BorelTannerDistribution[a, n], x] =
/// Piecewise[{{a^(x-n) n x^(x-n-1)/(E^(a x) (x-n)!), x >= n}}, 0].
/// For rational a = p/q wolframscript splits the power into
/// p^(x-n) q^(n-x) and merges any matching prime-power part of n into
/// the bases (n = 2, a = 1/2 gives 2^(3-x)).
fn pdf_borel_tanner(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("BorelTannerDistribution", dargs), x],
    )
  };
  if dargs.len() != 2 {
    return Ok(unevaluated(dargs, x));
  }
  let (a, n) = (dargs[0].clone(), dargs[1].clone());

  // Numeric point: exact value or 0
  if ms_numeric(&x).is_some() {
    let (Some(xv), Some(nv)) = (ms_numeric(&x), ms_numeric(&n)) else {
      return Ok(unevaluated(dargs, x));
    };
    if xv < nv || xv.fract() != 0.0 {
      return Ok(int(0));
    }
    // a^(x-n) n x^(x-n-1) / (E^(a x) (x-n)!)
    return eval(divide(
      call(
        "Times",
        vec![
          power(a.clone(), plus(x.clone(), times(int(-1), n.clone()))),
          n.clone(),
          power(
            x.clone(),
            plus(plus(x.clone(), times(int(-1), n.clone())), int(-1)),
          ),
        ],
      ),
      call(
        "Times",
        vec![
          power(e(), times(a.clone(), x.clone())),
          factorial(plus(x.clone(), times(int(-1), n.clone()))),
        ],
      ),
    ));
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }

  // Numerator factors in wolframscript's order
  let mut factors: Vec<Expr> = Vec::new();
  let n_int = match &n {
    Expr::Integer(v) if *v >= 1 => Some(*v),
    _ => None,
  };
  match (maxwell_rational(&a), n_int) {
    (Some((p, q)), Some(nv)) if p >= 1 && q > p => {
      // Extract the p- and q-parts of n
      let extract = |mut m: i128, base: i128| -> (i128, i128) {
        if base <= 1 {
          return (0, m);
        }
        let mut k = 0;
        while m % base == 0 {
          m /= base;
          k += 1;
        }
        (k, m)
      };
      let (i, rest) = extract(nv, p);
      let (j, m) = extract(rest, q);
      if m > 1 {
        factors.push(int(m));
      }
      if p > 1 {
        // p^(x - n + i)
        factors.push(power(int(p), call("Plus", vec![int(i - nv), x.clone()])));
      }
      // q^(n - x + j)
      factors.push(power(
        int(q),
        call(
          "Plus",
          vec![int(nv + j), call("Times", vec![int(-1), x.clone()])],
        ),
      ));
      // x^(x - n - 1)
      factors.push(power(
        x.clone(),
        call("Plus", vec![int(-1 - nv), x.clone()]),
      ));
    }
    _ => {
      // Symbolic forms: a^(-n + x) n x^(-1 - n + x) or with numeric n
      let neg_n = || -> Expr {
        match n_int {
          Some(nv) => int(-nv),
          None => call("Times", vec![int(-1), n.clone()]),
        }
      };
      if let Some(nv) = n_int {
        factors.push(int(nv));
        factors.push(power(a.clone(), call("Plus", vec![int(-nv), x.clone()])));
        factors.push(power(
          x.clone(),
          call("Plus", vec![int(-1 - nv), x.clone()]),
        ));
      } else {
        factors.push(power(a.clone(), call("Plus", vec![neg_n(), x.clone()])));
        factors.push(n.clone());
        factors.push(power(
          x.clone(),
          call("Plus", vec![int(-1), neg_n(), x.clone()]),
        ));
      }
    }
  }
  let neg_n_expr = match n_int {
    Some(nv) => int(-nv),
    None => call("Times", vec![int(-1), n.clone()]),
  };
  let body = divide(
    call("Times", factors),
    call(
      "Times",
      vec![
        power(e(), eval(times(a.clone(), x.clone()))?),
        factorial(call("Plus", vec![neg_n_expr, x.clone()])),
      ],
    ),
  );
  let cond = comparison(x, ComparisonOp::GreaterEqual, n);
  Ok(piecewise(vec![(body, cond)], int(0)))
}

/// Mean n/(1 - a) and variance a n/(1 - a)^3 for
/// BorelTannerDistribution.
fn borel_tanner_mean_variance(
  a: &Expr,
  n: &Expr,
) -> Result<(Expr, Expr), InterpreterError> {
  let one_minus_a = plus(int(1), times(int(-1), a.clone()));
  let mean = eval(divide(n.clone(), one_minus_a.clone()))?;
  let var = eval(divide(
    times(a.clone(), n.clone()),
    power(one_minus_a, int(3)),
  ))?;
  Ok((mean, var))
}

/// Validate BenktanderGibratDistribution parameters: numeric b must
/// satisfy b <= a (a + 1)/2, emitting ::lsseq otherwise.
fn benktander_valid(
  a: &Expr,
  b: &Expr,
  context: &Expr,
) -> Result<bool, InterpreterError> {
  if let (Some(av), Some(bv)) = (ms_numeric(a), ms_numeric(b)) {
    let bound = av * (av + 1.0) / 2.0;
    if bv > bound {
      let bound_expr =
        eval(divide(times(a.clone(), plus(a.clone(), int(1))), int(2)))?;
      crate::emit_message(&format!(
        "BenktanderGibratDistribution::lsseq: Parameter {} at position 2 in {} is expected to be less than or equal to {}.",
        expr_to_string(b),
        expr_to_string(context),
        expr_to_string(&bound_expr)
      ));
      return Ok(false);
    }
  }
  Ok(true)
}

/// PDF[BenktanderGibratDistribution[a, b], x] on x >= 1.
fn pdf_benktander_gibrat(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let uneval = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("BenktanderGibratDistribution", dargs), x],
    )
  };
  if dargs.len() != 2 {
    return Ok(uneval(dargs, x));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let dist = unevaluated("BenktanderGibratDistribution", dargs);
  if !benktander_valid(&a, &b, &dist)? {
    return Ok(uneval(dargs, x));
  }
  let log_x = |at: &Expr| call("Log", vec![at.clone()]);
  let numeric = ms_numeric(&a).is_some() && ms_numeric(&b).is_some();

  if ms_numeric(&x).is_some() {
    if !numeric {
      return Ok(uneval(dargs, x));
    }
    if ms_numeric(&x).is_some_and(|v| v < 1.0) {
      return Ok(int(0));
    }
    // Evaluate the closed form at the point
    return eval(divide(
      call(
        "Times",
        vec![
          power(x.clone(), eval(plus(int(-2), times(int(-1), a.clone())))?),
          call(
            "Plus",
            vec![
              divide(times(int(-2), b.clone()), a.clone()),
              call(
                "Times",
                vec![
                  plus(
                    plus(int(1), a.clone()),
                    times(times(int(2), b.clone()), log_x(&x)),
                  ),
                  plus(
                    int(1),
                    divide(
                      times(times(int(2), b.clone()), log_x(&x)),
                      a.clone(),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
      power(e(), times(b.clone(), power(log_x(&x), int(2)))),
    ));
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(uneval(dargs, x));
  }

  // t1 = -2 b/a; f1 = 1 + a + 2 b Log[x]; f2 = 1 + 2 b Log[x]/a
  let t1 = eval(divide(call("Times", vec![int(-2), b.clone()]), a.clone()))?;
  let f1 = eval(call(
    "Plus",
    vec![
      int(1),
      a.clone(),
      call("Times", vec![int(2), b.clone(), log_x(&x)]),
    ],
  ))?;
  let f2 = eval(call(
    "Plus",
    vec![
      int(1),
      divide(call("Times", vec![int(2), b.clone(), log_x(&x)]), a.clone()),
    ],
  ))?;
  // Numeric parameters order f2 f1, symbolic f1 f2
  let product = if numeric {
    call("Times", vec![f2, f1])
  } else {
    call("Times", vec![f1, f2])
  };
  let bracket = call("Plus", vec![t1, product]);
  let e_part = power(e(), eval(times(b.clone(), power(log_x(&x), int(2))))?);
  let body = if numeric {
    // (...)/(E^(b Log[x]^2) x^(2 + a))
    divide(
      bracket,
      call(
        "Times",
        vec![e_part, power(x.clone(), eval(plus(int(2), a.clone()))?)],
      ),
    )
  } else {
    // (x^(-2 - a) (...))/E^(b Log[x]^2)
    divide(
      call(
        "Times",
        vec![
          power(
            x.clone(),
            call(
              "Plus",
              vec![int(-2), call("Times", vec![int(-1), a.clone()])],
            ),
          ),
          bracket,
        ],
      ),
      e_part,
    )
  };
  let cond = comparison(x, ComparisonOp::GreaterEqual, int(1));
  Ok(piecewise(vec![(body, cond)], int(0)))
}

/// CDF[BenktanderGibratDistribution[a, b], x] =
/// 1 - x^(-1 - a)(1 + 2 b Log[x]/a)/E^(b Log[x]^2) on x >= 1.
fn cdf_benktander_gibrat(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let uneval = |dargs: &[Expr], x: Expr| {
    call(
      "CDF",
      vec![unevaluated("BenktanderGibratDistribution", dargs), x],
    )
  };
  if dargs.len() != 2 {
    return Ok(uneval(dargs, x));
  }
  let (a, b) = (dargs[0].clone(), dargs[1].clone());
  let dist = unevaluated("BenktanderGibratDistribution", dargs);
  if !benktander_valid(&a, &b, &dist)? {
    return Ok(uneval(dargs, x));
  }
  let log_x = |at: &Expr| call("Log", vec![at.clone()]);
  let numeric = ms_numeric(&a).is_some() && ms_numeric(&b).is_some();
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let f2 = eval(call(
      "Plus",
      vec![
        int(1),
        divide(call("Times", vec![int(2), b.clone(), log_x(at)]), a.clone()),
      ],
    ))?;
    let e_part = power(e(), eval(times(b.clone(), power(log_x(at), int(2))))?);
    let fraction = if numeric {
      divide(
        f2,
        call(
          "Times",
          vec![e_part, power(at.clone(), eval(plus(int(1), a.clone()))?)],
        ),
      )
    } else {
      divide(
        call(
          "Times",
          vec![
            power(
              at.clone(),
              call(
                "Plus",
                vec![int(-1), call("Times", vec![int(-1), a.clone()])],
              ),
            ),
            f2,
          ],
        ),
        e_part,
      )
    };
    Ok(call(
      "Plus",
      vec![int(1), call("Times", vec![int(-1), fraction])],
    ))
  };
  if ms_numeric(&x).is_some() {
    if !numeric {
      return Ok(uneval(dargs, x));
    }
    if ms_numeric(&x).is_some_and(|v| v < 1.0) {
      return Ok(int(0));
    }
    return eval(body(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(uneval(dargs, x));
  }
  let cond = comparison(x.clone(), ComparisonOp::GreaterEqual, int(1));
  Ok(piecewise(vec![(body(&x)?, cond)], int(0)))
}

/// Mean 1 + 1/a and the Erfc-based variance for
/// BenktanderGibratDistribution.
fn benktander_gibrat_mean_variance(
  a: &Expr,
  b: &Expr,
) -> Result<(Expr, Expr), InterpreterError> {
  let dist = call("BenktanderGibratDistribution", vec![a.clone(), b.clone()]);
  if !benktander_valid(a, b, &dist)? {
    return Err(InterpreterError::EvaluationError(
      "BenktanderGibratDistribution: invalid parameters".into(),
    ));
  }
  let mean = eval(plus(int(1), power(a.clone(), int(-1))))?;
  let a_minus_1 = call("Plus", vec![int(-1), a.clone()]);
  let numeric = ms_numeric(a).is_some() && ms_numeric(b).is_some();
  let erfc = call(
    "Erfc",
    vec![eval(divide(
      a_minus_1.clone(),
      call("Times", vec![int(2), call("Sqrt", vec![b.clone()])]),
    ))?],
  );
  let e_part = power(
    e(),
    eval(divide(power(a_minus_1, int(2)), times(int(4), b.clone())))?,
  );
  let var = if numeric {
    // Sqrt[Pi/b] merges for numeric b
    eval(divide(
      call(
        "Plus",
        vec![
          int(-1),
          call(
            "Times",
            vec![
              a.clone(),
              e_part,
              call("Sqrt", vec![eval(divide(pi(), b.clone()))?]),
              erfc,
            ],
          ),
        ],
      ),
      power(a.clone(), int(2)),
    ))?
  } else {
    // (-1 + (a E^((a-1)^2/(4 b)) Sqrt[Pi] Erfc[...])/Sqrt[b])/a^2
    divide(
      call(
        "Plus",
        vec![
          int(-1),
          divide(
            call(
              "Times",
              vec![a.clone(), e_part, call("Sqrt", vec![pi()]), erfc],
            ),
            call("Sqrt", vec![b.clone()]),
          ),
        ],
      ),
      power(a.clone(), int(2)),
    )
  };
  Ok((mean, var))
}

/// PDF[GumbelDistribution[a, b], x] = E^(-E^z + z)/b with
/// z = (x - a)/b (the minimum-extreme-value Gumbel; [] is (0, 1)).
fn pdf_gumbel(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let uneval = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("GumbelDistribution", dargs), x])
  };
  let (a, b) = match dargs {
    [] => (int(0), int(1)),
    [a, b] => (a.clone(), b.clone()),
    _ => return Ok(uneval(dargs, x)),
  };
  let z_of = |at: &Expr| -> Result<Expr, InterpreterError> {
    let diff = if matches!(&a, Expr::Integer(0)) {
      at.clone()
    } else {
      call(
        "Plus",
        vec![call("Times", vec![int(-1), a.clone()]), at.clone()],
      )
    };
    eval(divide(diff, b.clone()))
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let z = z_of(at)?;
    let exponent = call("Plus", vec![neg(power(e(), z.clone())), z]);
    let body = power(e(), exponent);
    Ok(if matches!(&b, Expr::Integer(1)) {
      body
    } else {
      divide(body, b.clone())
    })
  };
  if ms_numeric(&x).is_some() {
    return eval(body(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(uneval(dargs, x));
  }
  body(&x)
}

/// CDF[GumbelDistribution[a, b], x] = 1 - E^(-E^((x - a)/b)).
fn cdf_gumbel(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("GumbelDistribution", dargs), x])
  };
  let (a, b) = match dargs {
    [] => (int(0), int(1)),
    [a, b] => (a.clone(), b.clone()),
    _ => return Ok(unevaluated(dargs, x)),
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let diff = if matches!(&a, Expr::Integer(0)) {
      at.clone()
    } else {
      call(
        "Plus",
        vec![call("Times", vec![int(-1), a.clone()]), at.clone()],
      )
    };
    let z = eval(divide(diff, b.clone()))?;
    Ok(call(
      "Plus",
      vec![
        int(1),
        call("Times", vec![int(-1), power(e(), neg(power(e(), z)))]),
      ],
    ))
  };
  if ms_numeric(&x).is_some() {
    return eval(body(&x)?);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  body(&x)
}

/// Mean a - b EulerGamma and variance b^2 Pi^2/6 for
/// GumbelDistribution.
fn gumbel_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let (a, b) = match dargs {
    [] => (int(0), int(1)),
    [a, b] => (a.clone(), b.clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GumbelDistribution expects no arguments or [a, b]".into(),
      ));
    }
  };
  let mean = eval(plus(
    a,
    times(
      int(-1),
      times(b.clone(), Expr::Identifier("EulerGamma".to_string())),
    ),
  ))?;
  let var = eval(divide(
    call("Times", vec![power(b, int(2)), power(pi(), int(2))]),
    int(6),
  ))?;
  Ok((mean, var))
}

/// PDF[SkewNormalDistribution[m, s, a], x] =
///   Erfc[-(a (x - m)/(Sqrt[2] s))] / (E^((x - m)^2/(2 s^2)) Sqrt[2 Pi] s),
/// i.e. 2/s φ((x - m)/s) Φ(a (x - m)/s) written with Erfc. For a = 0 this
/// reduces to the Normal PDF because Erfc[0] = 1.
fn pdf_skew_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let uneval = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("SkewNormalDistribution", dargs), x])
  };
  let (m, s, a) = match dargs {
    [m, s, a] => (m.clone(), s.clone(), a.clone()),
    _ => return Ok(uneval(dargs, x)),
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let xm = plus(neg(m.clone()), at.clone());
    let sqrt2 = sqrt(int(2));
    let erfc_arg = neg(divide(
      times(a.clone(), xm.clone()),
      times(sqrt2, s.clone()),
    ));
    let num = call("Erfc", vec![erfc_arg]);
    let gauss = power(
      e(),
      divide(power(xm, int(2)), times(int(2), power(s.clone(), int(2)))),
    );
    let denom = times(times(gauss, sqrt(times(int(2), pi()))), s.clone());
    eval(divide(num, denom))
  };
  if ms_numeric(&x).is_some() {
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(uneval(dargs, x));
  }
  body(&x)
}

/// CDF[SkewNormalDistribution[m, s, a], x] =
///   Erfc[(m - x)/(Sqrt[2] s)]/2 - 2 OwenT[(x - m)/s, a].
/// For a = 0, OwenT[z, 0] = 0 and this reduces to the Normal CDF.
fn cdf_skew_normal(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let uneval = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("SkewNormalDistribution", dargs), x])
  };
  let (m, s, a) = match dargs {
    [m, s, a] => (m.clone(), s.clone(), a.clone()),
    _ => return Ok(uneval(dargs, x)),
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    let phi = divide(
      call(
        "Erfc",
        vec![divide(
          minus(m.clone(), at.clone()),
          times(sqrt(int(2)), s.clone()),
        )],
      ),
      int(2),
    );
    let owen = call(
      "OwenT",
      vec![divide(minus(at.clone(), m.clone()), s.clone()), a.clone()],
    );
    eval(minus(phi, times(int(2), owen)))
  };
  if ms_numeric(&x).is_some() {
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(uneval(dargs, x));
  }
  body(&x)
}

/// Mean = m + a Sqrt[2/Pi] s / Sqrt[1 + a^2] and
/// Variance = (1 - 2 a^2/((1 + a^2) Pi)) s^2 for SkewNormalDistribution.
fn skew_normal_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let (m, s, a) = match dargs {
    [m, s, a] => (m.clone(), s.clone(), a.clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "SkewNormalDistribution expects [m, s, a]".into(),
      ));
    }
  };
  let a2 = power(a.clone(), int(2));
  let mean = eval(plus(
    m,
    divide(
      times(times(a, sqrt(divide(int(2), pi()))), s.clone()),
      sqrt(plus(int(1), a2.clone())),
    ),
  ))?;
  let var = eval(times(
    minus(
      int(1),
      divide(times(int(2), a2.clone()), times(plus(int(1), a2), pi())),
    ),
    power(s, int(2)),
  ))?;
  Ok((mean, var))
}

/// PDF[ZipfDistribution[r], x] = x^(-1-r)/Zeta[1+r] on x >= 1;
/// PDF[ZipfDistribution[n, r], x] uses HarmonicNumber[n, 1+r] on
/// 1 <= x <= n.
fn pdf_zipf(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("ZipfDistribution", dargs), x])
  };
  let (n, r) = match dargs {
    [r] => (None, r.clone()),
    [n, r] => (Some(n.clone()), r.clone()),
    _ => return Ok(unevaluated(dargs, x)),
  };
  let norm = match &n {
    None => call("Zeta", vec![eval(plus(int(1), r.clone()))?]),
    Some(n) => call(
      "HarmonicNumber",
      vec![n.clone(), eval(plus(int(1), r.clone()))?],
    ),
  };
  let body = |at: &Expr| -> Result<Expr, InterpreterError> {
    // Pre-dividing 1/norm hoists rationals out of evaluated Zeta
    // values (Zeta[2] = Pi^2/6 prints as 6/(Pi^2 x^2), not nested)
    let coeff = eval(divide(int(1), norm.clone()))?;
    eval(call(
      "Times",
      vec![
        coeff,
        power(at.clone(), eval(plus(int(-1), times(int(-1), r.clone())))?),
      ],
    ))
  };
  if ms_numeric(&x).is_some() {
    let xv = ms_numeric(&x).unwrap();
    let in_support = xv >= 1.0
      && xv.fract() == 0.0
      && n.as_ref().and_then(ms_numeric).is_none_or(|nv| xv <= nv);
    if !in_support {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let cond = match &n {
    None => comparison(x.clone(), ComparisonOp::GreaterEqual, int(1)),
    Some(n) => comparison3(
      int(1),
      ComparisonOp::LessEqual,
      x.clone(),
      ComparisonOp::LessEqual,
      n.clone(),
    ),
  };
  Ok(piecewise(vec![(body(&x)?, cond)], int(0)))
}

/// Mean and variance for ZipfDistribution: Zeta ratios with existence
/// thresholds (r > 1, r > 2) for the infinite form, HarmonicNumber
/// ratios for the bounded form.
fn zipf_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  match dargs {
    [r] => {
      let zeta = |offset: i128| -> Result<Expr, InterpreterError> {
        Ok(call("Zeta", vec![eval(plus(int(offset), r.clone()))?]))
      };
      let mean_value = divide(zeta(0)?, zeta(1)?);
      let var_value = call(
        "Plus",
        vec![
          call(
            "Times",
            vec![
              int(-1),
              divide(power(zeta(0)?, int(2)), power(zeta(1)?, int(2))),
            ],
          ),
          divide(zeta(-1)?, zeta(1)?),
        ],
      );
      match ms_numeric(r) {
        Some(rv) => Ok((
          if rv > 1.0 {
            eval(mean_value)?
          } else {
            infinity()
          },
          if rv > 2.0 {
            eval(var_value)?
          } else {
            infinity()
          },
        )),
        None => Ok((
          piecewise(
            vec![(
              mean_value,
              comparison(r.clone(), ComparisonOp::Greater, int(1)),
            )],
            infinity(),
          ),
          piecewise(
            vec![(
              var_value,
              comparison(r.clone(), ComparisonOp::Greater, int(2)),
            )],
            infinity(),
          ),
        )),
      }
    }
    [n, r] => {
      let h = |offset: i128| -> Result<Expr, InterpreterError> {
        Ok(call(
          "HarmonicNumber",
          vec![n.clone(), eval(plus(int(offset), r.clone()))?],
        ))
      };
      let mean = eval(divide(h(0)?, h(1)?))?;
      let var = eval(call(
        "Plus",
        vec![
          divide(h(-1)?, h(1)?),
          call("Times", vec![int(-1), power(divide(h(0)?, h(1)?), int(2))]),
        ],
      ))?;
      Ok((mean, var))
    }
    _ => Err(InterpreterError::EvaluationError(
      "ZipfDistribution expects 1 or 2 arguments".into(),
    )),
  }
}

/// PDF for BenfordDistribution[b] (Benford's law, base b): the probability of
/// leading digit d is Log[1 + 1/d]/Log[b] for d = 1 … b-1, and 0 otherwise.
/// Only an integer base b >= 2 is handled.
fn pdf_benford(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("PDF", vec![unevaluated("BenfordDistribution", dargs), x])
  };
  let [b] = dargs else {
    return Ok(unevaluated(dargs, x));
  };
  let Some(bv) = ms_numeric(b) else {
    return Ok(unevaluated(dargs, x));
  };
  if bv < 2.0 || bv.fract() != 0.0 {
    return Ok(unevaluated(dargs, x));
  }
  // Log[1 + 1/d] / Log[b].
  let body = |d: &Expr| -> Result<Expr, InterpreterError> {
    let inner = eval(plus(int(1), divide(int(1), d.clone())))?;
    eval(divide(
      call("Log", vec![inner]),
      call("Log", vec![b.clone()]),
    ))
  };
  if let Some(xv) = ms_numeric(&x) {
    let in_support = xv >= 1.0 && xv.fract() == 0.0 && xv <= bv - 1.0;
    if !in_support {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let cond = comparison3(
    int(1),
    ComparisonOp::LessEqual,
    x.clone(),
    ComparisonOp::Less,
    b.clone(),
  );
  Ok(piecewise(vec![(body(&x)?, cond)], int(0)))
}

/// CDF for BenfordDistribution[b]: the leading-digit probabilities telescope to
/// Log[1 + Floor[x]]/Log[b] on 1 <= x < b, with 0 below 1 and 1 at or above b.
fn cdf_benford(dargs: &[Expr], x: Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call("CDF", vec![unevaluated("BenfordDistribution", dargs), x])
  };
  let [b] = dargs else {
    return Ok(unevaluated(dargs, x));
  };
  let Some(bv) = ms_numeric(b) else {
    return Ok(unevaluated(dargs, x));
  };
  if bv < 2.0 || bv.fract() != 0.0 {
    return Ok(unevaluated(dargs, x));
  }
  // Log[1 + Floor[x]] / Log[b] at the given point.
  let body = |floor_plus_one: Expr| -> Result<Expr, InterpreterError> {
    eval(divide(
      call("Log", vec![floor_plus_one]),
      call("Log", vec![b.clone()]),
    ))
  };
  if let Some(xv) = ms_numeric(&x) {
    if xv < 1.0 {
      return Ok(int(0));
    }
    if xv >= bv {
      return Ok(int(1));
    }
    let fl = xv.floor() as i128;
    return body(int(fl + 1));
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  let floor_x = call("Floor", vec![x.clone()]);
  let main = body(eval(plus(int(1), floor_x))?)?;
  Ok(piecewise(
    vec![
      (
        main,
        comparison3(
          int(1),
          ComparisonOp::LessEqual,
          x.clone(),
          ComparisonOp::Less,
          b.clone(),
        ),
      ),
      (int(0), comparison(x.clone(), ComparisonOp::Less, int(1))),
    ],
    int(1),
  ))
}

/// Mean and variance for BenfordDistribution[b] (integer base). The mean has
/// the closed form b - Log[b!]/Log[b]; the variance is the exact second-moment
/// sum minus the squared mean (its symbolic form may differ from wolframscript
/// while remaining value-correct).
fn benford_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let [b] = dargs else {
    return Err(InterpreterError::EvaluationError(
      "BenfordDistribution expects 1 argument".into(),
    ));
  };
  let Some(bv) = ms_numeric(b) else {
    return Err(InterpreterError::EvaluationError(
      "BenfordDistribution base must be an integer".into(),
    ));
  };
  if bv < 2.0 || bv.fract() != 0.0 {
    return Err(InterpreterError::EvaluationError(
      "BenfordDistribution base must be an integer >= 2".into(),
    ));
  }
  let b_int = bv as i128;
  let log_b = call("Log", vec![b.clone()]);
  // Mean = b - Log[b!]/Log[b].
  let factorial = eval(factorial(b.clone()))?;
  let mean = eval(plus(
    b.clone(),
    times(int(-1), divide(call("Log", vec![factorial]), log_b.clone())),
  ))?;
  // Variance = Sum[d^2 Log[1 + 1/d]/Log[b], {d, 1, b-1}] - Mean^2.
  let mut terms: Vec<Expr> = Vec::new();
  for d in 1..b_int {
    let inner = eval(plus(int(1), divide(int(1), int(d))))?;
    terms.push(times(
      power(int(d), int(2)),
      divide(call("Log", vec![inner]), log_b.clone()),
    ));
  }
  let second_moment = eval(call("Plus", terms))?;
  let variance = eval(plus(
    second_moment,
    times(int(-1), power(mean.clone(), int(2))),
  ))?;
  Ok((mean, variance))
}

/// Whether BenktanderWeibullDistribution[a, b] has valid parameters: a > 0 and
/// 0 < b <= 1. Symbolic parameters are treated as valid (computed symbolically).
fn benktander_weibull_valid(a: &Expr, b: &Expr) -> bool {
  let a_ok = ms_numeric(a).is_none_or(|v| v > 0.0);
  let b_ok = ms_numeric(b).is_none_or(|v| v > 0.0 && v <= 1.0);
  a_ok && b_ok
}

/// PDF for BenktanderWeibullDistribution[a, b] on x >= 1:
/// E^((a (1 - x^b))/b) x^(b-2) (1 - b + a x^b).
fn pdf_benktander_weibull(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "PDF",
      vec![unevaluated("BenktanderWeibullDistribution", dargs), x],
    )
  };
  let [a, b] = dargs else {
    return Ok(unevaluated(dargs, x));
  };
  if !benktander_weibull_valid(a, b) {
    return Ok(unevaluated(dargs, x));
  }
  let e = || Expr::Identifier("E".to_string());
  // E^((a (1 - x^b))/b) x^(b-2) (1 - b + a x^b).
  let body = |t: &Expr| -> Result<Expr, InterpreterError> {
    let exponent = divide(
      times(
        a.clone(),
        plus(int(1), times(int(-1), power(t.clone(), b.clone()))),
      ),
      b.clone(),
    );
    let e_factor = power(e(), exponent);
    let x_factor = power(t.clone(), plus(int(-2), b.clone()));
    let poly = plus(
      plus(int(1), times(int(-1), b.clone())),
      times(a.clone(), power(t.clone(), b.clone())),
    );
    eval(times(e_factor, times(x_factor, poly)))
  };
  if let Some(xv) = ms_numeric(&x) {
    if xv < 1.0 {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  Ok(piecewise(
    vec![(
      body(&x)?,
      comparison(x.clone(), ComparisonOp::GreaterEqual, int(1)),
    )],
    int(0),
  ))
}

/// CDF for BenktanderWeibullDistribution[a, b] on x >= 1:
/// 1 - E^((a (1 - x^b))/b) x^(b-1).
fn cdf_benktander_weibull(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |dargs: &[Expr], x: Expr| {
    call(
      "CDF",
      vec![unevaluated("BenktanderWeibullDistribution", dargs), x],
    )
  };
  let [a, b] = dargs else {
    return Ok(unevaluated(dargs, x));
  };
  if !benktander_weibull_valid(a, b) {
    return Ok(unevaluated(dargs, x));
  }
  let e = || Expr::Identifier("E".to_string());
  // 1 - E^((a (1 - x^b))/b) x^(b-1).
  let body = |t: &Expr| -> Result<Expr, InterpreterError> {
    let exponent = divide(
      times(
        a.clone(),
        plus(int(1), times(int(-1), power(t.clone(), b.clone()))),
      ),
      b.clone(),
    );
    let e_factor = power(e(), exponent);
    let x_factor = power(t.clone(), plus(int(-1), b.clone()));
    eval(plus(int(1), times(int(-1), times(e_factor, x_factor))))
  };
  if let Some(xv) = ms_numeric(&x) {
    if xv < 1.0 {
      return Ok(int(0));
    }
    return body(&x);
  }
  if !matches!(&x, Expr::Identifier(_)) {
    return Ok(unevaluated(dargs, x));
  }
  Ok(piecewise(
    vec![(
      body(&x)?,
      comparison(x.clone(), ComparisonOp::GreaterEqual, int(1)),
    )],
    int(0),
  ))
}

/// Mean and variance for BenktanderWeibullDistribution[a, b]:
/// Mean = 1 + 1/a; Variance = (-1 + 2 a E^(a/b) ExpIntegralE[1 - 1/b, a/b]/b)/a^2.
fn benktander_weibull_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let [a, b] = dargs else {
    return Err(InterpreterError::EvaluationError(
      "BenktanderWeibullDistribution expects 2 arguments".into(),
    ));
  };
  let e = || Expr::Identifier("E".to_string());
  let mean = eval(plus(int(1), divide(int(1), a.clone())))?;
  // (-1 + (2 a E^(a/b) ExpIntegralE[1 - 1/b, a/b]) / b) / a^2.
  let exp_int = call(
    "ExpIntegralE",
    vec![
      plus(int(1), times(int(-1), divide(int(1), b.clone()))),
      divide(a.clone(), b.clone()),
    ],
  );
  let numer = plus(
    int(-1),
    divide(
      times(
        times(int(2), a.clone()),
        times(power(e(), divide(a.clone(), b.clone())), exp_int),
      ),
      b.clone(),
    ),
  );
  let variance = eval(divide(numer, power(a.clone(), int(2))))?;
  Ok((mean, variance))
}

/// PDF for SinghMaddalaDistribution[q, a, b] (Burr XII) on x > 0:
/// a q x^(a-1) (1 + (x/b)^a)^(-1-q) / b^a.
fn pdf_singh_maddala(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "SinghMaddalaDistribution expects 3 arguments".into(),
    ));
  }
  let (q, a, b) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let aq = times(a.clone(), q.clone());
  let x_pow = power(x.clone(), minus(a.clone(), int(1)));
  let x_over_b_to_a = power(divide(x.clone(), b.clone()), a.clone());
  let bracket = power(plus(int(1), x_over_b_to_a), minus(int(-1), q));
  let b_to_a = power(b, a);
  let density = divide(times(times(aq, x_pow), bracket), b_to_a);
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(density, cond)], int(0)))
}

/// CDF for SinghMaddalaDistribution[q, a, b] on x > 0:
/// 1 - (1 + (x/b)^a)^(-q).
fn cdf_singh_maddala(
  dargs: &[Expr],
  x: Expr,
) -> Result<Expr, InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "SinghMaddalaDistribution expects 3 arguments".into(),
    ));
  }
  let (q, a, b) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let x_over_b_to_a = power(divide(x.clone(), b), a);
  let body = minus(
    int(1),
    power(plus(int(1), x_over_b_to_a), times(int(-1), q)),
  );
  let cond = comparison(x, ComparisonOp::Greater, int(0));
  eval(piecewise(vec![(body, cond)], int(0)))
}

/// Mean and variance for SinghMaddalaDistribution[q, a, b], each valid only
/// above a moment threshold (a q > 1 for the mean, a q > 2 for the variance)
/// and Indeterminate otherwise.
fn singh_maddala_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  if dargs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "SinghMaddalaDistribution expects 3 arguments".into(),
    ));
  }
  let (q, a, b) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  let aq = times(a.clone(), q.clone());
  let inv_a = divide(int(1), a.clone());
  let two_a = divide(int(2), a.clone());
  let g_q = gamma(q.clone());
  let g_1_inv_a = gamma(plus(int(1), inv_a.clone()));
  let g_q_inv_a = gamma(plus(times(int(-1), inv_a), q.clone()));

  // Mean = b Gamma[1 + 1/a] Gamma[-1/a + q] / Gamma[q], for a q > 1.
  let mean_expr = divide(
    times(times(b.clone(), g_1_inv_a.clone()), g_q_inv_a.clone()),
    g_q.clone(),
  );
  let mean = piecewise(
    vec![(
      mean_expr,
      comparison(aq.clone(), ComparisonOp::Greater, int(1)),
    )],
    indeterminate(),
  );

  // Variance = b^2 (Gamma[1 + 2/a] Gamma[q] Gamma[-2/a + q]
  //   - Gamma[1 + 1/a]^2 Gamma[-1/a + q]^2) / Gamma[q]^2, for a q > 2.
  let g_2_a = gamma(plus(int(1), two_a.clone()));
  let g_q_2a = gamma(plus(times(int(-1), two_a), q.clone()));
  let term1 = times(times(g_2_a, g_q.clone()), g_q_2a);
  let term2 = times(power(g_1_inv_a, int(2)), power(g_q_inv_a, int(2)));
  let var_expr = divide(
    times(power(b, int(2)), minus(term1, term2)),
    power(g_q, int(2)),
  );
  let variance = piecewise(
    vec![(var_expr, comparison(aq, ComparisonOp::Greater, int(2)))],
    indeterminate(),
  );

  Ok((mean, variance))
}

/// Mean and variance for the 4-argument BetaPrimeDistribution[p, q, b, a]
/// (power b, scale a), each existing only above a moment threshold in b*q.
fn beta_prime4_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let (p, q, b, a) = (
    dargs[0].clone(),
    dargs[1].clone(),
    dargs[2].clone(),
    dargs[3].clone(),
  );
  let inv_b = divide(int(1), b.clone());
  let two_b = divide(int(2), b.clone());
  let bq = times(b.clone(), q.clone());
  let g_p = gamma(p.clone());
  let g_q = gamma(q.clone());
  let g_1b_p = gamma(plus(inv_b.clone(), p.clone()));
  let g_1b_q = gamma(plus(times(int(-1), inv_b), q.clone()));

  // Mean = a Gamma[1/b + p] Gamma[-1/b + q] / (Gamma[p] Gamma[q]), for 1 < b q.
  let mean_val = divide(
    times(times(a.clone(), g_1b_p.clone()), g_1b_q.clone()),
    times(g_p.clone(), g_q.clone()),
  );
  let mean = piecewise(
    vec![(mean_val, comparison(int(1), ComparisonOp::Less, bq.clone()))],
    infinity(),
  );

  // Variance = a^2 (Gamma[p] Gamma[2/b + p] Gamma[q] Gamma[-2/b + q]
  //   - Gamma[1/b + p]^2 Gamma[-1/b + q]^2) / (Gamma[p]^2 Gamma[q]^2), b q > 2.
  let g_2b_p = gamma(plus(two_b.clone(), p.clone()));
  let g_2b_q = gamma(plus(times(int(-1), two_b), q.clone()));
  let term1 = times(times(times(g_p.clone(), g_2b_p), g_q.clone()), g_2b_q);
  let term2 = times(power(g_1b_p, int(2)), power(g_1b_q, int(2)));
  let var_val = divide(
    times(power(a, int(2)), minus(term1, term2)),
    times(power(g_p, int(2)), power(g_q, int(2))),
  );
  let variance = piecewise(
    vec![(var_val, comparison(bq, ComparisonOp::Greater, int(2)))],
    indeterminate(),
  );

  Ok((mean, variance))
}

/// Mean and variance for the 3-argument ParetoDistribution[k, a, m]
/// (Type II / Lomax, scale k, location m).
fn pareto3_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let (k, a, m) = (dargs[0].clone(), dargs[1].clone(), dargs[2].clone());
  // Mean = k/(a - 1) + m, for a > 1.
  let mean = piecewise(
    vec![(
      plus(divide(k.clone(), plus(int(-1), a.clone())), m),
      comparison(a.clone(), ComparisonOp::Greater, int(1)),
    )],
    indeterminate(),
  );
  // Variance = a k^2 / ((a - 2)(a - 1)^2), for a > 2.
  let variance = piecewise(
    vec![(
      divide(
        times(a.clone(), power(k, int(2))),
        times(
          plus(int(-2), a.clone()),
          power(plus(int(-1), a.clone()), int(2)),
        ),
      ),
      comparison(a, ComparisonOp::Greater, int(2)),
    )],
    indeterminate(),
  );
  Ok((mean, variance))
}

/// Mean and variance for the 4-argument ParetoDistribution[k, a, g, m]
/// (extra shape g), each existing only above a threshold in a/g.
fn pareto4_mean_variance(
  dargs: &[Expr],
) -> Result<(Expr, Expr), InterpreterError> {
  let (k, a, g, m) = (
    dargs[0].clone(),
    dargs[1].clone(),
    dargs[2].clone(),
    dargs[3].clone(),
  );
  let g_a = gamma(a.clone());
  let g_amg = gamma(minus(a.clone(), g.clone())); // Gamma[a - g]
  let g_1pg = gamma(plus(int(1), g.clone())); // Gamma[1 + g]
  // Mean = m + k Gamma[a - g] Gamma[1 + g] / Gamma[a], for a > g.
  let mean = piecewise(
    vec![(
      plus(
        m,
        divide(
          times(times(k.clone(), g_amg.clone()), g_1pg.clone()),
          g_a.clone(),
        ),
      ),
      comparison(a.clone(), ComparisonOp::Greater, g.clone()),
    )],
    indeterminate(),
  );
  // Variance = k^2 (-(Gamma[a-g]^2 Gamma[1+g]^2) + Gamma[a] Gamma[a-2g] Gamma[1+2g])
  //   / Gamma[a]^2, for a > 2 g.
  let g_am2g = gamma(minus(a.clone(), times(int(2), g.clone()))); // Gamma[a - 2 g]
  let g_1p2g = gamma(plus(int(1), times(int(2), g.clone()))); // Gamma[1 + 2 g]
  let term_neg = times(power(g_amg, int(2)), power(g_1pg, int(2)));
  let term_pos = times(times(g_a.clone(), g_am2g), g_1p2g);
  let var_num =
    times(power(k, int(2)), plus(times(int(-1), term_neg), term_pos));
  let variance = piecewise(
    vec![(
      divide(var_num, power(g_a, int(2))),
      comparison(a, ComparisonOp::Greater, times(int(2), g)),
    )],
    indeterminate(),
  );
  Ok((mean, variance))
}
