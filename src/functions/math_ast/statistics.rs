#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Total[list] - Sum of all elements in a list
/// Total[list, n] - Sum across levels 1 through n
/// Total[list, {n}] - Sum at exactly level n
pub fn total_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Total expects 1 or 2 arguments".into(),
    ));
  }

  // A small helper that accepts either a finite level number or
  // `Infinity` and turns it into a `usize`.
  fn level_to_usize(e: &Expr) -> Option<usize> {
    match e {
      Expr::Identifier(s) if s == "Infinity" => Some(usize::MAX),
      other => expr_to_num(other).map(|n| n as usize),
    }
  }

  // Parse level spec from second argument
  let level_spec = if args.len() == 2 {
    match &args[1] {
      // Total[list, {n}] - exact level n
      Expr::List(items) if items.len() == 1 => {
        if let Some(n) = level_to_usize(&items[0]) {
          TotalLevelSpec::Exact(n)
        } else {
          return Ok(Expr::FunctionCall {
            name: "Total".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
      // Total[list, {n1, n2}] - sum across levels n1 through n2
      Expr::List(items) if items.len() == 2 => {
        if let (Some(n1), Some(n2)) =
          (level_to_usize(&items[0]), level_to_usize(&items[1]))
        {
          TotalLevelSpec::Range(n1, n2)
        } else {
          return Ok(Expr::FunctionCall {
            name: "Total".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
      // Total[list, Infinity]
      Expr::Identifier(s) if s == "Infinity" => {
        TotalLevelSpec::Through(usize::MAX)
      }
      // Total[list, n] - through level n
      _ => {
        if let Some(n) = expr_to_num(&args[1]) {
          TotalLevelSpec::Through(n as usize)
        } else {
          return Ok(Expr::FunctionCall {
            name: "Total".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
    }
  } else {
    TotalLevelSpec::Through(1)
  };

  match &args[0] {
    Expr::List(_) => total_with_level(&args[0], &level_spec),
    Expr::Association(pairs) => {
      let values: Vec<Expr> = pairs.iter().map(|(_, v)| v.clone()).collect();
      total_with_level(&Expr::List(values.into()), &level_spec)
    }
    // Total[x] for non-list returns x
    other => Ok(other.clone()),
  }
}

pub enum TotalLevelSpec {
  Through(usize),      // sum levels 1..=n
  Exact(usize),        // sum at exactly level n
  Range(usize, usize), // sum levels n1..=n2 (collapsed together)
}

/// Sum a list at level 1 using Plus (Apply[Plus, list])
/// Handles nested lists by recursively adding element-wise.
pub fn total_sum_level1(items: &[Expr]) -> Result<Expr, InterpreterError> {
  if items.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let mut acc = items[0].clone();
  for item in &items[1..] {
    acc = add_exprs_recursive(&acc, item)?;
  }
  Ok(acc)
}

/// Recursively add two expressions, threading over lists element-wise
pub fn add_exprs_recursive(
  a: &Expr,
  b: &Expr,
) -> Result<Expr, InterpreterError> {
  match (a, b) {
    (Expr::List(la), Expr::List(lb)) if la.len() == lb.len() => {
      let results: Result<Vec<Expr>, _> = la
        .iter()
        .zip(lb.iter())
        .map(|(x, y)| add_exprs_recursive(x, y))
        .collect();
      Ok(Expr::List(results?.into()))
    }
    _ => plus_ast(&[a.clone(), b.clone()]),
  }
}

/// Recursively apply Total with level spec
pub fn total_with_level(
  expr: &Expr,
  level_spec: &TotalLevelSpec,
) -> Result<Expr, InterpreterError> {
  match level_spec {
    TotalLevelSpec::Through(n) => total_through_level(expr, *n),
    TotalLevelSpec::Exact(n) => total_at_exact_level(expr, *n),
    TotalLevelSpec::Range(n1, n2) => total_range_levels(expr, *n1, *n2),
  }
}

/// Total[list, {n1, n2}] - sum across levels n1 through n2 (inclusive),
/// collapsing those levels into a scalar while preserving the structure
/// above n1 and (implicitly) below n2.
pub fn total_range_levels(
  expr: &Expr,
  n1: usize,
  n2: usize,
) -> Result<Expr, InterpreterError> {
  if n2 < n1 {
    return Ok(expr.clone());
  }
  if n1 <= 1 {
    // From the top level down to n2: same as summing levels 1..=n2.
    return total_through_level(expr, n2);
  }
  match expr {
    Expr::List(items) => {
      let processed: Vec<Expr> = items
        .iter()
        .map(|item| total_range_levels(item, n1 - 1, n2 - 1))
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Expr::List(processed.into()))
    }
    _ => Ok(expr.clone()),
  }
}

/// Total[list, n] - sum across levels 1 through n
/// Level 0 means no summing, level 1 means Apply[Plus, list], etc.
pub fn total_through_level(
  expr: &Expr,
  n: usize,
) -> Result<Expr, InterpreterError> {
  if n == 0 {
    return Ok(expr.clone());
  }
  match expr {
    Expr::List(items) => {
      // First, recursively process sublists for levels 2..n
      if n > 1 {
        let processed: Vec<Expr> = items
          .iter()
          .map(|item| total_through_level(item, n - 1))
          .collect::<Result<Vec<_>, _>>()?;
        total_sum_level1(&processed)
      } else {
        total_sum_level1(items)
      }
    }
    _ => Ok(expr.clone()),
  }
}

/// Total[list, {n}] - sum at exactly level n
/// Level 1 = sum the outermost list, level 2 = sum each sublist, etc.
pub fn total_at_exact_level(
  expr: &Expr,
  n: usize,
) -> Result<Expr, InterpreterError> {
  if n <= 1 {
    // Sum at this level
    match expr {
      Expr::List(items) => total_sum_level1(items),
      _ => Ok(expr.clone()),
    }
  } else {
    // Recurse into sublists, summing at deeper level
    match expr {
      Expr::List(items) => {
        let processed: Vec<Expr> = items
          .iter()
          .map(|item| total_at_exact_level(item, n - 1))
          .collect::<Result<Vec<_>, _>>()?;
        Ok(Expr::List(processed.into()))
      }
      _ => Ok(expr.clone()),
    }
  }
}

/// Mean[list] - Arithmetic mean
pub fn mean_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Mean expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::FunctionCall {
          name: "Mean".to_string(),
          args: vec![Expr::List(vec![].into())].into(),
        });
      }
      // Try to compute exact integer sum first
      let mut int_sum: Option<i128> = Some(0);
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => {
            if let Some(s) = int_sum {
              int_sum = s.checked_add(*n);
            }
          }
          Expr::Real(_) => {
            has_real = true;
            int_sum = None;
          }
          _ => {
            int_sum = None;
          }
        }
      }

      if let Some(sum) = int_sum {
        // All integers - return exact rational or integer
        let count = items.len() as i128;
        if sum % count == 0 {
          Ok(Expr::Integer(sum / count))
        } else {
          // Return as Rational
          let g = gcd_helper(sum.abs(), count);
          let num = sum / g;
          let denom = count / g;
          Ok(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num), Expr::Integer(denom)].into(),
          })
        }
      } else if has_real {
        // Has real numbers - compute float
        let mut sum = 0.0;
        for item in items {
          if let Some(n) = expr_to_num(item) {
            sum += n;
          } else {
            return Ok(Expr::FunctionCall {
              name: "Mean".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
        Ok(num_to_expr(sum / items.len() as f64))
      } else {
        // Check for list-of-lists (matrix) → compute column-wise mean
        if items.iter().all(|item| matches!(item, Expr::List(_))) {
          return mean_columnwise(items);
        }
        // Non-numeric elements - compute symbolically: Total[list] / Length[list]
        // Evaluate the sum first, then wrap in division (don't distribute)
        let sum_expr = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: items.clone(),
        };
        let evaluated_sum = crate::evaluator::evaluate_expr_to_expr(&sum_expr)?;
        let n = items.len() as i128;
        // Use BinaryOp::Divide to represent (sum) / n without distributing
        Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(evaluated_sum),
          right: Box::new(Expr::Integer(n)),
        })
      }
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if matches!(
      dist_name.as_str(),
      "NormalDistribution"
        | "UniformDistribution"
        | "ExponentialDistribution"
        | "PoissonDistribution"
        | "BernoulliDistribution"
        | "BinomialDistribution"
        | "StableDistribution"
        | "GammaDistribution"
        | "BetaDistribution"
        | "StudentTDistribution"
        | "LogNormalDistribution"
        | "ChiSquareDistribution"
        | "ParetoDistribution"
        | "WeibullDistribution"
        | "FrechetDistribution"
        | "ExtremeValueDistribution"
        | "GompertzMakehamDistribution"
        | "HalfNormalDistribution"
        | "InverseGammaDistribution"
        | "HypoexponentialDistribution"
        | "InverseGaussianDistribution"
        | "LogisticDistribution"
        | "GeometricDistribution"
        | "CauchyDistribution"
        | "DiscreteUniformDistribution"
        | "LaplaceDistribution"
        | "RayleighDistribution"
        | "NegativeBinomialDistribution"
        | "ArcSinDistribution"
        | "PascalDistribution"
        | "DagumDistribution"
        | "HyperbolicDistribution"
        | "NoncentralFRatioDistribution"
    ) =>
    {
      let (mean, _) =
        super::distributions::distribution_mean_variance_pub(dist_name, dargs)?;
      crate::evaluator::evaluate_expr_to_expr(&mean)
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "JohnsonDistribution" => {
      match super::distributions::distribution_mean_variance_pub(
        dist_name, dargs,
      ) {
        Ok((mean, _)) => crate::evaluator::evaluate_expr_to_expr(&mean),
        Err(_) => Ok(Expr::FunctionCall {
          name: "Mean".to_string(),
          args: vec![Expr::FunctionCall {
            name: dist_name.clone(),
            args: dargs.clone(),
          }]
          .into(),
        }),
      }
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "MultinomialDistribution" => {
      let (mean, _) = super::distributions::multinomial_mean_variance(dargs)?;
      Ok(mean)
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "MultivariatePoissonDistribution" => {
      let (mean, _) =
        super::distributions::multivariate_poisson_mean_variance(dargs)?;
      Ok(mean)
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "QuantityDistribution" && dargs.len() == 2 => {
      // Mean[QuantityDistribution[dist, unit]] = Quantity[Mean[dist], unit]
      let inner_mean = mean_ast(&[dargs[0].clone()])?;
      Ok(Expr::FunctionCall {
        name: "Quantity".to_string(),
        args: vec![inner_mean, dargs[1].clone()].into(),
      })
    }
    Expr::Association(pairs) => {
      let values: Vec<Expr> = pairs.iter().map(|(_, v)| v.clone()).collect();
      mean_ast(&[Expr::List(values.into())])
    }
    _ => Ok(Expr::FunctionCall {
      name: "Mean".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Mean of columns in a list-of-lists (matrix)
pub fn mean_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&crate::ExprList> = rows
    .iter()
    .filter_map(|r| {
      if let Expr::List(items) = r {
        Some(items)
      } else {
        None
      }
    })
    .collect();
  if row_vecs.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Mean".to_string(),
      args: vec![Expr::List(rows.to_vec().into())].into(),
    });
  }
  let ncols = row_vecs[0].len();
  let nrows = row_vecs.len();
  let mut col_means = Vec::new();
  for col in 0..ncols {
    let col_items: Vec<Expr> = row_vecs
      .iter()
      .map(|r| {
        if col < r.len() {
          r[col].clone()
        } else {
          Expr::Integer(0)
        }
      })
      .collect();
    let mean_result = mean_ast(&[Expr::List(col_items.into())])?;
    col_means.push(mean_result);
  }
  let _ = nrows; // used indirectly through mean_ast
  Ok(Expr::List(col_means.into()))
}

/// Variance[list] - Sample variance (unbiased, divides by n-1)
/// Variance[{1, 2, 3}] => 1
/// Variance[{1.0, 2.0, 3.0}] => 1.0
pub fn variance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Variance expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.len() < 2 {
        crate::emit_message(&format!(
          "Variance::shlen: The argument {} should have at least two elements.",
          crate::syntax::expr_to_string(&args[0])
        ));
        return Ok(Expr::FunctionCall {
          name: "Variance".to_string(),
          args: vec![args[0].clone()].into(),
        });
      }
      // Try all-integer exact path
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => int_vals.push(*n),
          Expr::Real(_) => {
            all_int = false;
            has_real = true;
            break;
          }
          _ => {
            all_int = false;
            break;
          }
        }
      }
      if all_int && !int_vals.is_empty() {
        // Exact: Variance = Sum[(xi - mean)^2] / (n-1)
        // = (n * Sum[xi^2] - (Sum[xi])^2) / (n * (n-1))
        let n = int_vals.len() as i128;
        let sum: i128 = int_vals.iter().sum();
        let sum_sq: i128 = int_vals.iter().map(|x| x * x).sum();
        let numer = n * sum_sq - sum * sum;
        let denom = n * (n - 1);
        return Ok(make_rational(numer, denom));
      }
      if has_real || !all_int {
        // Try float path first
        let mut vals = Vec::new();
        let mut all_numeric = true;
        for item in items {
          if let Some(v) = expr_to_num(item) {
            vals.push(v);
          } else {
            all_numeric = false;
            break;
          }
        }
        if all_numeric && !vals.is_empty() {
          let n = vals.len() as f64;
          let mean = vals.iter().sum::<f64>() / n;
          let var =
            vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
          return Ok(num_to_expr(var));
        }
        // Check for list-of-lists → compute column-wise
        if items.iter().all(|item| matches!(item, Expr::List(_))) {
          return variance_columnwise(items);
        }
        // Symbolic/complex path: compute Variance = Sum[Abs[xi - mean]^2] / (n-1)
        return variance_symbolic(items);
      }
      Ok(Expr::FunctionCall {
        name: "Variance".to_string(),
        args: args.to_vec().into(),
      })
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if matches!(
      dist_name.as_str(),
      "NormalDistribution"
        | "UniformDistribution"
        | "ExponentialDistribution"
        | "PoissonDistribution"
        | "BernoulliDistribution"
        | "BinomialDistribution"
        | "StableDistribution"
        | "GammaDistribution"
        | "BetaDistribution"
        | "StudentTDistribution"
        | "LogNormalDistribution"
        | "ChiSquareDistribution"
        | "ParetoDistribution"
        | "WeibullDistribution"
        | "FrechetDistribution"
        | "ExtremeValueDistribution"
        | "HalfNormalDistribution"
        | "HypoexponentialDistribution"
        | "InverseGammaDistribution"
        | "InverseGaussianDistribution"
        | "LogisticDistribution"
        | "GeometricDistribution"
        | "CauchyDistribution"
        | "DiscreteUniformDistribution"
        | "LaplaceDistribution"
        | "RayleighDistribution"
        | "NegativeBinomialDistribution"
        | "ArcSinDistribution"
        | "PascalDistribution"
        | "DagumDistribution"
        | "HyperbolicDistribution"
        | "NoncentralFRatioDistribution"
    ) =>
    {
      let (_, variance) =
        super::distributions::distribution_mean_variance_pub(dist_name, dargs)?;
      crate::evaluator::evaluate_expr_to_expr(&variance)
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "JohnsonDistribution" => {
      match super::distributions::distribution_mean_variance_pub(
        dist_name, dargs,
      ) {
        Ok((_, variance)) => crate::evaluator::evaluate_expr_to_expr(&variance),
        Err(_) => Ok(Expr::FunctionCall {
          name: "Variance".to_string(),
          args: vec![Expr::FunctionCall {
            name: dist_name.clone(),
            args: dargs.clone(),
          }]
          .into(),
        }),
      }
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "MultinomialDistribution" => {
      let (_, variance) =
        super::distributions::multinomial_mean_variance(dargs)?;
      Ok(variance)
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "MultivariatePoissonDistribution" => {
      let (_, variance) =
        super::distributions::multivariate_poisson_mean_variance(dargs)?;
      Ok(variance)
    }
    Expr::FunctionCall {
      name: dist_name,
      args: dargs,
    } if dist_name == "QuantityDistribution" && dargs.len() == 2 => {
      // Variance[QuantityDistribution[dist, unit]] = Quantity[Variance[dist], unit^2]
      let inner_var = variance_ast(&[dargs[0].clone()])?;
      let unit_sq = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(dargs[1].clone()),
        right: Box::new(Expr::Integer(2)),
      };
      Ok(Expr::FunctionCall {
        name: "Quantity".to_string(),
        args: vec![inner_var, unit_sq].into(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Variance".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Compute variance symbolically
pub fn variance_symbolic(items: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = items.len();
  if n < 2 {
    return Err(InterpreterError::EvaluationError(
      "Variance: need at least 2 elements".into(),
    ));
  }
  // Compute mean symbolically
  let mean = mean_ast(&[Expr::List(items.to_vec().into())])?;
  // Compute Sum[Abs[xi - mean]^2] / (n-1)
  let mut sum_sq_terms = Vec::new();
  for item in items {
    // (xi - mean)
    let diff = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        item.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), mean.clone()].into(),
        },
      ]
      .into(),
    };
    // Abs[xi - mean]^2
    let abs_sq = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![diff].into(),
        },
        Expr::Integer(2),
      ]
      .into(),
    };
    sum_sq_terms.push(abs_sq);
  }
  let sum_sq = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: sum_sq_terms.into(),
  };
  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      sum_sq,
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![Expr::Integer((n - 1) as i128), Expr::Integer(-1)].into(),
      },
    ]
    .into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Variance of columns in a list-of-lists (matrix)
pub fn variance_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&crate::ExprList> = rows
    .iter()
    .filter_map(|r| {
      if let Expr::List(items) = r {
        Some(items)
      } else {
        None
      }
    })
    .collect();
  if row_vecs.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Variance".to_string(),
      args: vec![Expr::List(rows.to_vec().into())].into(),
    });
  }
  let ncols = row_vecs[0].len();
  let mut col_vars = Vec::new();
  for col in 0..ncols {
    let col_items: Vec<Expr> = row_vecs
      .iter()
      .map(|r| {
        if col < r.len() {
          r[col].clone()
        } else {
          Expr::Integer(0)
        }
      })
      .collect();
    let var_result = variance_ast(&[Expr::List(col_items.into())])?;
    col_vars.push(var_result);
  }
  Ok(Expr::List(col_vars.into()))
}

/// StandardDeviation[list] - Sample standard deviation (Sqrt of Variance)
/// StandardDeviation[{1, 2, 3}] => 1
/// StandardDeviation[{1.0, 2.0, 3.0}] => 1.0
pub fn standard_deviation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StandardDeviation expects exactly 1 argument".into(),
    ));
  }
  // Emit the StandardDeviation::shlen warning directly when the list has
  // fewer than two elements, instead of falling through to Variance's
  // warning. Use ::is_quiet to suppress the inner Variance message.
  if let Expr::List(items) = &args[0]
    && items.len() < 2
  {
    crate::emit_message(&format!(
      "StandardDeviation::shlen: The argument {} should have at least two elements.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return Ok(Expr::FunctionCall {
      name: "StandardDeviation".to_string(),
      args: args.to_vec().into(),
    });
  }
  // For list-of-lists, the variance returns a list of column variances
  let var = variance_ast(args)?;
  // If variance returned unevaluated (e.g. for too-few elements), return unevaluated too
  if let Expr::FunctionCall { name, .. } = &var
    && name == "Variance"
  {
    return Ok(Expr::FunctionCall {
      name: "StandardDeviation".to_string(),
      args: args.to_vec().into(),
    });
  }
  match &var {
    Expr::List(items) => {
      // Apply Sqrt to each element
      let mut results = Vec::new();
      for item in items {
        results.push(sqrt_ast(&[item.clone()])?);
      }
      Ok(Expr::List(results.into()))
    }
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::FunctionCall { .. }
    | Expr::BinaryOp { .. } => {
      // For distribution arguments, extract even negative power factors from
      // the variance. Distribution parameters are always positive, so
      // Sqrt[a * p^(-2)] = Sqrt[a] / p (no Abs needed).
      // E.g. Variance = n*(1-p)/p^2 → SD = Sqrt[n*(1-p)] / p
      if is_distribution_arg(&args[0])
        && let Some(result) = try_sqrt_extract_denom_factors(&var)?
      {
        return Ok(result);
      }
      sqrt_ast(&[var.clone()])
    }
    _ => Ok(Expr::FunctionCall {
      name: "StandardDeviation".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Check if expr is a distribution function call.
fn is_distribution_arg(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, .. }
  if matches!(name.as_str(),
    "NormalDistribution"
      | "UniformDistribution"
      | "ExponentialDistribution"
      | "PoissonDistribution"
      | "BernoulliDistribution"
      | "GammaDistribution"
      | "BetaDistribution"
      | "StudentTDistribution"
      | "LogNormalDistribution"
      | "ChiSquareDistribution"
      | "ParetoDistribution"
      | "WeibullDistribution"
      | "GeometricDistribution"
      | "CauchyDistribution"
      | "DiscreteUniformDistribution"
      | "LaplaceDistribution"
      | "RayleighDistribution"
      | "NegativeBinomialDistribution"
      | "ArcSinDistribution"
      | "PascalDistribution"
      | "DagumDistribution"
      | "HyperbolicDistribution"
      | "NoncentralFRatioDistribution"
      | "BinomialDistribution"
      | "JohnsonDistribution"
  ))
}

/// Try to extract even negative power factors from a product and split the Sqrt.
/// E.g. Sqrt[n * (1-p) * p^(-2)] → Sqrt[n*(1-p)] / p
/// Returns None if there are no extractable factors.
fn try_sqrt_extract_denom_factors(
  var: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::polynomial_ast::{
    build_product, collect_multiplicative_factors,
  };
  use crate::syntax::BinaryOperator;

  let factors = collect_multiplicative_factors(var);
  let mut numerator_factors: Vec<Expr> = Vec::new();
  let mut denominator_factors: Vec<Expr> = Vec::new();

  for f in &factors {
    match f {
      // x^(-2n) where n > 0: extract x^n into denominator
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } if matches!(exp.as_ref(), Expr::Integer(n) if *n < 0 && n % 2 == 0) => {
        if let Expr::Integer(n) = exp.as_ref() {
          let half = (-n) / 2;
          if half == 1 {
            denominator_factors.push(*base.clone());
          } else {
            denominator_factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: base.clone(),
              right: Box::new(Expr::Integer(half)),
            });
          }
        }
      }
      // FunctionCall Power[x, -2n]
      Expr::FunctionCall {
        name: pname,
        args: pargs,
      } if pname == "Power"
        && pargs.len() == 2
        && matches!(&pargs[1], Expr::Integer(n) if *n < 0 && n % 2 == 0) =>
      {
        if let Expr::Integer(n) = &pargs[1] {
          let half = (-n) / 2;
          if half == 1 {
            denominator_factors.push(pargs[0].clone());
          } else {
            denominator_factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(pargs[0].clone()),
              right: Box::new(Expr::Integer(half)),
            });
          }
        }
      }
      _ => numerator_factors.push(f.clone()),
    }
  }

  if denominator_factors.is_empty() {
    return Ok(None);
  }

  let num_expr = if numerator_factors.is_empty() {
    Expr::Integer(1)
  } else if numerator_factors.len() == 1 {
    numerator_factors.remove(0)
  } else {
    build_product(numerator_factors)
  };

  let sqrt_num = sqrt_ast(&[num_expr])?;

  let denom_expr = if denominator_factors.len() == 1 {
    denominator_factors.remove(0)
  } else {
    build_product(denominator_factors)
  };

  Ok(Some(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(sqrt_num),
    right: Box::new(denom_expr),
  }))
}

/// GeometricMean[list] - Geometric mean: (product of elements)^(1/n)
/// GeometricMean[{2, 8}] => 4
/// GeometricMean[{1.0, 2.0, 3.0}] => 1.8171205928321397
pub fn geometric_mean_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "GeometricMean expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "GeometricMean: empty list".into(),
        ));
      }
      // List-of-lists (matrix) → column-wise geometric mean.
      if items.iter().all(|item| matches!(item, Expr::List(_))) {
        return geometric_mean_columnwise(items);
      }
      let n = items.len() as i128;
      // Exact path: if all elements are exact (integers or rationals),
      // compute product symbolically and return Power[product, 1/n]
      let product = super::times_ast(items);
      if let Ok(ref prod) = product {
        let has_real = items.iter().any(|i| matches!(i, Expr::Real(_)));
        if !has_real {
          let exponent = super::make_rational(1, n);
          return super::power_two(prod, &exponent);
        }
      }
      // Float path
      let mut vals = Vec::new();
      for item in items {
        if let Some(v) = try_eval_to_f64(item) {
          vals.push(v);
        } else {
          return Ok(Expr::FunctionCall {
            name: "GeometricMean".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
      let n_f = vals.len() as f64;
      let product: f64 = vals.iter().product();
      let result = product.powf(1.0 / n_f);
      // Check if result is an integer
      Ok(num_to_expr(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "GeometricMean".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Column-wise GeometricMean for a list-of-lists (matrix) input.
fn geometric_mean_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&crate::ExprList> = rows
    .iter()
    .filter_map(|r| {
      if let Expr::List(items) = r {
        Some(items)
      } else {
        None
      }
    })
    .collect();
  if row_vecs.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "GeometricMean".to_string(),
      args: vec![Expr::List(rows.to_vec().into())].into(),
    });
  }
  let ncols = row_vecs[0].len();
  let mut col_means = Vec::with_capacity(ncols);
  for col in 0..ncols {
    let col_items: Vec<Expr> = row_vecs
      .iter()
      .map(|r| {
        if col < r.len() {
          r[col].clone()
        } else {
          Expr::Integer(0)
        }
      })
      .collect();
    col_means.push(geometric_mean_ast(&[Expr::List(col_items.into())])?);
  }
  Ok(Expr::List(col_means.into()))
}

/// HarmonicMean[list] - Harmonic mean: n / Sum[1/xi]
/// HarmonicMean[{1, 2, 3}] => 18/11
/// HarmonicMean[{1.0, 2.0, 3.0}] => 1.6363636363636365
pub fn harmonic_mean_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "HarmonicMean expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "HarmonicMean: empty list".into(),
        ));
      }
      // Try all-integer exact path using rational arithmetic
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => {
            if *n == 0 {
              return Err(InterpreterError::EvaluationError(
                "HarmonicMean: division by zero".into(),
              ));
            }
            int_vals.push(*n);
          }
          Expr::Real(_) => {
            all_int = false;
            has_real = true;
            break;
          }
          _ => {
            all_int = false;
            break;
          }
        }
      }
      if all_int && !int_vals.is_empty() {
        // HarmonicMean = n / Sum[1/xi]
        // = n / (Sum[product/xi] / product)
        // = n * product / Sum[product/xi]
        // Use rational sum: Sum[1/xi] = numer/denom
        let n = int_vals.len() as i128;
        let mut sum_numer: i128 = 0;
        let mut sum_denom: i128 = 1;
        for &x in &int_vals {
          // Add 1/x to sum_numer/sum_denom
          // a/b + 1/x = (a*x + b) / (b*x)
          sum_numer = sum_numer * x + sum_denom;
          sum_denom *= x;
          // Simplify to avoid overflow
          let g = gcd(sum_numer.abs(), sum_denom.abs());
          sum_numer /= g;
          sum_denom /= g;
        }
        // HarmonicMean = n / (sum_numer/sum_denom) = n * sum_denom / sum_numer
        let result_numer = n * sum_denom;
        let result_denom = sum_numer;
        return Ok(make_rational(result_numer, result_denom));
      }
      if has_real {
        // Float path
        let mut vals = Vec::new();
        for item in items {
          if let Some(v) = expr_to_num(item) {
            if v == 0.0 {
              return Err(InterpreterError::EvaluationError(
                "HarmonicMean: division by zero".into(),
              ));
            }
            vals.push(v);
          } else {
            return Ok(Expr::FunctionCall {
              name: "HarmonicMean".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
        let n = vals.len() as f64;
        let sum_recip: f64 = vals.iter().map(|x| 1.0 / x).sum();
        return Ok(num_to_expr(n / sum_recip));
      }
      // List-of-lists (matrix) → column-wise harmonic mean.
      if items.iter().all(|item| matches!(item, Expr::List(_))) {
        return harmonic_mean_columnwise(items);
      }
      // Symbolic / mixed path: build n / Plus[1/x1, ..., 1/xn] and let
      // the evaluator simplify it. The output formatter renders 1/x as
      // x^(-1), which matches wolframscript's display form.
      harmonic_mean_symbolic(items)
    }
    _ => Ok(Expr::FunctionCall {
      name: "HarmonicMean".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// HarmonicMean for symbolic or mixed numeric+symbolic lists:
/// builds `n / Plus[1/x1, ..., 1/xn]` and evaluates it. A single-element
/// list collapses to the element itself.
fn harmonic_mean_symbolic(items: &[Expr]) -> Result<Expr, InterpreterError> {
  if items.len() == 1 {
    return Ok(items[0].clone());
  }
  let recips: Vec<Expr> = items
    .iter()
    .map(|x| Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(x.clone()),
    })
    .collect();
  let sum = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: recips.into(),
  })?;
  let n = items.len() as i128;
  crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::Integer(n)),
    right: Box::new(sum),
  })
}

/// Column-wise HarmonicMean for a list-of-lists (matrix) input.
fn harmonic_mean_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&crate::ExprList> = rows
    .iter()
    .filter_map(|r| {
      if let Expr::List(items) = r {
        Some(items)
      } else {
        None
      }
    })
    .collect();
  if row_vecs.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "HarmonicMean".to_string(),
      args: vec![Expr::List(rows.to_vec().into())].into(),
    });
  }
  let ncols = row_vecs[0].len();
  let mut col_means = Vec::with_capacity(ncols);
  for col in 0..ncols {
    let col_items: Vec<Expr> = row_vecs
      .iter()
      .map(|r| {
        if col < r.len() {
          r[col].clone()
        } else {
          Expr::Integer(0)
        }
      })
      .collect();
    col_means.push(harmonic_mean_ast(&[Expr::List(col_items.into())])?);
  }
  Ok(Expr::List(col_means.into()))
}

/// Covariance[list1, list2] - Sample covariance of two numeric lists
pub fn covariance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Covariance".to_string(),
      args: args.to_vec().into(),
    });
  }
  let (xs, ys) = match (&args[0], &args[1]) {
    (Expr::List(xs), Expr::List(ys))
      if xs.len() == ys.len() && xs.len() >= 2 =>
    {
      (xs, ys)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Covariance".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  // Check all elements are numeric
  let all_numeric = xs.iter().all(|x| expr_to_num(x).is_some())
    && ys.iter().all(|y| expr_to_num(y).is_some());
  if !all_numeric {
    return Ok(Expr::FunctionCall {
      name: "Covariance".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Compute means symbolically
  let mean_x = mean_ast(&[args[0].clone()])?;
  let mean_y = mean_ast(&[args[1].clone()])?;

  let n = xs.len();
  // Compute sum of (x_i - mean_x) * (y_i - mean_y) symbolically
  let mut terms = Vec::with_capacity(n);
  for (x, y) in xs.iter().zip(ys.iter()) {
    let dx = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(x.clone()),
      right: Box::new(mean_x.clone()),
    };
    let dy = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(y.clone()),
      right: Box::new(mean_y.clone()),
    };
    let product = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(dx),
      right: Box::new(dy),
    };
    let val = crate::evaluator::evaluate_expr_to_expr(&product)?;
    terms.push(val);
  }

  // Sum and divide by (n-1)
  let sum_expr = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  let sum_val = crate::evaluator::evaluate_expr_to_expr(&sum_expr)?;
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(sum_val),
    right: Box::new(Expr::Integer((n - 1) as i128)),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Transpose a rectangular `rows` of `Expr::List` rows into a Vec of columns.
/// Returns None when rows are ragged.
fn transpose_rows(rows: &[Expr]) -> Option<Vec<Vec<Expr>>> {
  let row_vecs: Vec<&crate::ExprList> = rows
    .iter()
    .filter_map(|r| {
      if let Expr::List(items) = r {
        Some(items)
      } else {
        None
      }
    })
    .collect();
  if row_vecs.len() != rows.len() {
    return None;
  }
  let ncols = row_vecs[0].len();
  if row_vecs.iter().any(|r| r.len() != ncols) {
    return None;
  }
  let mut cols: Vec<Vec<Expr>> = vec![Vec::with_capacity(rows.len()); ncols];
  for r in row_vecs {
    for (i, cell) in r.iter().enumerate() {
      cols[i].push(cell.clone());
    }
  }
  Some(cols)
}

/// Correlation[list1, list2] - Pearson correlation coefficient
pub fn correlation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Correlation".to_string(),
      args: args.to_vec().into(),
    });
  }
  // Matrix form: Correlation[A, B] for m×n A and m×p B → cross-correlation
  // matrix R[i, j] = Correlation(column_i(A), column_j(B)).
  if let (Expr::List(a_rows), Expr::List(b_rows)) = (&args[0], &args[1])
    && !a_rows.is_empty()
    && !b_rows.is_empty()
    && a_rows.len() == b_rows.len()
    && a_rows.iter().all(|r| matches!(r, Expr::List(_)))
    && b_rows.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let a_cols = transpose_rows(a_rows);
    let b_cols = transpose_rows(b_rows);
    if let (Some(a_cols), Some(b_cols)) = (a_cols, b_cols) {
      let mut rows: Vec<Expr> = Vec::with_capacity(a_cols.len());
      for ai in &a_cols {
        let mut row: Vec<Expr> = Vec::with_capacity(b_cols.len());
        for bj in &b_cols {
          let entry = correlation_ast(&[
            Expr::List(ai.clone().into()),
            Expr::List(bj.clone().into()),
          ])?;
          row.push(entry);
        }
        rows.push(Expr::List(row.into()));
      }
      return Ok(Expr::List(rows.into()));
    }
  }
  let (xs, ys) = match (&args[0], &args[1]) {
    (Expr::List(xs), Expr::List(ys))
      if xs.len() == ys.len() && xs.len() >= 2 =>
    {
      (xs, ys)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Correlation".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  // Require all elements to be numeric; otherwise leave unevaluated.
  let all_numeric = xs.iter().all(|x| expr_to_num(x).is_some())
    && ys.iter().all(|y| expr_to_num(y).is_some());
  if !all_numeric {
    // Symbolic two-element vectors collapse to a clean closed form:
    //   r = (v0-v1)(Conj(w0)-Conj(w1)) /
    //       (Sqrt[(v0-v1)(Conj(v0)-Conj(v1))] * Sqrt[(w0-w1)(Conj(w0)-Conj(w1))])
    if xs.len() == 2 && ys.len() == 2 {
      let v0 = xs[0].clone();
      let v1 = xs[1].clone();
      let w0 = ys[0].clone();
      let w1 = ys[1].clone();
      let diff = |a: &Expr, b: &Expr| Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(a.clone()),
        right: Box::new(b.clone()),
      };
      let conj = |e: &Expr| Expr::FunctionCall {
        name: "Conjugate".to_string(),
        args: vec![e.clone()].into(),
      };
      let times = |a: Expr, b: Expr| Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![a, b].into(),
      };
      let conj_diff = |a: &Expr, b: &Expr| Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(conj(a)),
        right: Box::new(conj(b)),
      };
      let v_diff = diff(&v0, &v1);
      let w_diff = diff(&w0, &w1);
      let v_conj_diff = conj_diff(&v0, &v1);
      let w_conj_diff = conj_diff(&w0, &w1);
      let numer = times(v_diff.clone(), w_conj_diff.clone());
      let sqrt_v = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![times(v_diff, v_conj_diff)].into(),
      };
      let sqrt_w = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![times(w_diff, w_conj_diff)].into(),
      };
      let denom = times(sqrt_v, sqrt_w);
      // Return without re-evaluating to preserve the Times factor order
      // inside each Sqrt (which Woxi's canonical sort would otherwise
      // reorder differently than wolframscript for some variable names).
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(numer),
        right: Box::new(denom),
      });
    }
    return Ok(Expr::FunctionCall {
      name: "Correlation".to_string(),
      args: args.to_vec().into(),
    });
  }
  // If any input is a machine-precision Real, fall through to a direct
  // f64 computation: the result is a machine number either way, and the
  // exact operation order matches wolframscript's floating-point output.
  let has_real = xs
    .iter()
    .chain(ys.iter())
    .any(|e| matches!(e, Expr::Real(_)));
  if has_real {
    let x_vals: Vec<f64> = xs.iter().map(|x| expr_to_num(x).unwrap()).collect();
    let y_vals: Vec<f64> = ys.iter().map(|y| expr_to_num(y).unwrap()).collect();
    let n = x_vals.len() as f64;
    let mean_x = x_vals.iter().sum::<f64>() / n;
    let mean_y = y_vals.iter().sum::<f64>() / n;
    let cov: f64 = x_vals
      .iter()
      .zip(y_vals.iter())
      .map(|(x, y)| (x - mean_x) * (y - mean_y))
      .sum();
    let var_x: f64 = x_vals.iter().map(|x| (x - mean_x).powi(2)).sum();
    let var_y: f64 = y_vals.iter().map(|y| (y - mean_y).powi(2)).sum();
    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
      // Wolfram emits Correlation::zerosd and leaves the expression unevaluated.
      return Ok(Expr::FunctionCall {
        name: "Correlation".to_string(),
        args: args.to_vec().into(),
      });
    }
    return Ok(num_to_expr(cov / denom));
  }
  // All inputs are exact (Integer/Rational): compute symbolically.
  // r = Cov(x,y) / Sqrt[Cov(x,x) * Cov(y,y)]
  // The (n-1) divisors in each covariance cancel in the ratio, so this is
  // equivalent to the standard Pearson formula while reusing the symbolic
  // covariance implementation to preserve exact arithmetic.
  let cov_xy = covariance_ast(&[args[0].clone(), args[1].clone()])?;
  let var_x = covariance_ast(&[args[0].clone(), args[0].clone()])?;
  let var_y = covariance_ast(&[args[1].clone(), args[1].clone()])?;
  // Guard against zero variance (constant list): Wolfram emits
  // Correlation::zerosd and leaves the expression unevaluated.
  if let (Some(vx), Some(vy)) = (expr_to_num(&var_x), expr_to_num(&var_y))
    && (vx == 0.0 || vy == 0.0)
  {
    return Ok(Expr::FunctionCall {
      name: "Correlation".to_string(),
      args: args.to_vec().into(),
    });
  }
  let var_product = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(var_x),
    right: Box::new(var_y),
  };
  let denom = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![var_product].into(),
  };
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(cov_xy),
    right: Box::new(denom),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// CentralMoment[list, r] - r-th central moment of a numeric list
pub fn central_moment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "CentralMoment".to_string(),
      args: args.to_vec().into(),
    });
  }
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let r_expr = &args[1];
  let r = match expr_to_num(r_expr) {
    Some(r) => r as i32,
    None => {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Check if all items are numeric (integer or rational or real)
  let all_numeric = items.iter().all(|item| expr_to_num(item).is_some());
  if !all_numeric {
    return Ok(Expr::FunctionCall {
      name: "CentralMoment".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Compute mean symbolically to preserve exact arithmetic
  let mean_expr = mean_ast(&[args[0].clone()])?;

  // Compute sum of (x_i - mean)^r symbolically
  let n = items.len();
  let mut terms = Vec::with_capacity(n);
  for item in items {
    // (item - mean)^r
    let diff = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(item.clone()),
      right: Box::new(mean_expr.clone()),
    };
    let powered = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(diff),
      right: Box::new(Expr::Integer(r as i128)),
    };
    let val = crate::evaluator::evaluate_expr_to_expr(&powered)?;
    terms.push(val);
  }

  // Sum and divide by n
  let sum_expr = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  let sum_val = crate::evaluator::evaluate_expr_to_expr(&sum_expr)?;
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(sum_val),
    right: Box::new(Expr::Integer(n as i128)),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Kurtosis[list] - CentralMoment[list, 4] / CentralMoment[list, 2]^2
pub fn kurtosis_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Kurtosis".to_string(),
      args: args.to_vec().into(),
    });
  }
  let m4 = central_moment_ast(&[args[0].clone(), Expr::Integer(4)])?;
  let m2 = central_moment_ast(&[args[0].clone(), Expr::Integer(2)])?;
  // Compute m4 / m2^2 symbolically
  let m2_squared = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(m2),
    right: Box::new(Expr::Integer(2)),
  };
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(m4),
    right: Box::new(m2_squared),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Skewness[list] - CentralMoment[list, 3] / CentralMoment[list, 2]^(3/2)
pub fn skewness_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Skewness".to_string(),
      args: args.to_vec().into(),
    });
  }
  let m3 = central_moment_ast(&[args[0].clone(), Expr::Integer(3)])?;
  let m2 = central_moment_ast(&[args[0].clone(), Expr::Integer(2)])?;
  // Compute m3 / m2^(3/2) symbolically
  let m2_pow = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(m2),
    right: Box::new(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(3)),
      right: Box::new(Expr::Integer(2)),
    }),
  };
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(m3),
    right: Box::new(m2_pow),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// RootMeanSquare[list] - Sqrt[Mean[list^2]]
/// RootMeanSquare[{1, 2, 3}] => Sqrt[14/3]
/// RootMeanSquare[{1.0, 2.0, 3.0}] => 2.160246899469287
pub fn root_mean_square_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RootMeanSquare expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "RootMeanSquare: empty list".into(),
        ));
      }
      // Try all-integer exact path
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => int_vals.push(*n),
          Expr::Real(_) => {
            all_int = false;
            has_real = true;
            break;
          }
          _ => {
            all_int = false;
            break;
          }
        }
      }
      if all_int && !int_vals.is_empty() {
        let n = int_vals.len() as i128;
        let sum_sq: i128 = int_vals.iter().map(|x| x * x).sum();
        // RMS = Sqrt[sum_sq / n]
        let g = gcd(sum_sq.abs(), n);
        let numer = sum_sq / g;
        let denom = n / g;
        // Check if numer/denom is a perfect square
        if denom == 1 {
          let root = (numer as f64).sqrt() as i128;
          if root * root == numer {
            return Ok(Expr::Integer(root));
          }
          return Ok(make_sqrt(Expr::Integer(numer)));
        }
        // Return Sqrt[Rational[numer, denom]]
        return Ok(make_sqrt(make_rational(numer, denom)));
      }
      if has_real || !all_int {
        let mut vals = Vec::new();
        for item in items {
          if let Some(v) = expr_to_num(item) {
            vals.push(v);
          } else {
            return Ok(Expr::FunctionCall {
              name: "RootMeanSquare".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
        let n = vals.len() as f64;
        let mean_sq = vals.iter().map(|x| x * x).sum::<f64>() / n;
        return Ok(num_to_expr(mean_sq.sqrt()));
      }
      Ok(Expr::FunctionCall {
        name: "RootMeanSquare".to_string(),
        args: args.to_vec().into(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "RootMeanSquare".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// GCD for i128 values
pub fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Extract numerator from Integer or Rational expr
pub fn expr_numerator(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(n) => Some(*n),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(n) = &args[0] {
        Some(*n)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Extract denominator from Integer or Rational expr
pub fn expr_denominator(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(_) => Some(1),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(d) = &args[1] {
        Some(*d)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Floor of a rational number num/den
pub fn rational_floor(num: i128, den: i128) -> i128 {
  if den == 0 {
    return 0; // shouldn't happen, caller checks
  }
  // Normalize sign so den > 0
  let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
  if num >= 0 {
    num / den
  } else {
    // For negative: floor division
    (num - den + 1) / den
  }
}

/// Ceiling of a rational number num/den
pub fn rational_ceil(num: i128, den: i128) -> i128 {
  if den == 0 {
    return 0;
  }
  let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
  if num >= 0 {
    (num + den - 1) / den
  } else {
    num / den
  }
}

/// Quantile[list, q] - the q-th quantile of the list
/// Quantile[list, {q1, q2, ...}] - multiple quantiles
pub fn quantile_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Quantile expects exactly 2 arguments".into(),
    ));
  }
  // Quantile[dist, q] for a distribution — first try closed-form heads,
  // then fall back to the numerical CDF-inversion for ProbabilityDistribution.
  if let Expr::FunctionCall { name, args: dargs } = &args[0] {
    if let Some(result) =
      crate::functions::math_ast::distributions::quantile_distribution_closed_form(
        name, dargs, &args[1],
      )
    {
      return Ok(result);
    }
    if let Some(result) =
      crate::functions::math_ast::quantile_distribution_numeric(
        name, dargs, &args[1],
      )?
    {
      return Ok(result);
    }
  }
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Quantile".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Sort the items numerically
  let mut sorted: Vec<&Expr> = items.iter().collect();
  sorted.sort_by(|a, b| {
    let fa = try_eval_to_f64(a);
    let fb = try_eval_to_f64(b);
    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
  });

  // Handle list of quantiles
  if let Expr::List(qs) = &args[1] {
    let results: Result<Vec<Expr>, _> =
      qs.iter().map(|q| quantile_single(&sorted, q)).collect();
    return Ok(Expr::List(results?.into()));
  }

  quantile_single(&sorted, &args[1])
}

pub fn quantile_single(
  sorted: &[&Expr],
  q: &Expr,
) -> Result<Expr, InterpreterError> {
  let n = sorted.len();
  // Default Quantile uses Type 1 (inverse of CDF)
  // Index = Ceiling[q * n]
  let q_val = match q {
    Expr::Integer(n) => *n as f64,
    Expr::Real(f) => *f,
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(num), Expr::Integer(den)) = (&rargs[0], &rargs[1]) {
        *num as f64 / *den as f64
      } else {
        return Ok(Expr::FunctionCall {
          name: "Quantile".to_string(),
          args: vec![
            Expr::List(sorted.iter().cloned().cloned().collect()),
            q.clone(),
          ]
          .into(),
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Quantile".to_string(),
        args: vec![
          Expr::List(sorted.iter().cloned().cloned().collect()),
          q.clone(),
        ]
        .into(),
      });
    }
  };

  let idx = (q_val * n as f64).ceil() as usize;
  let idx = idx.max(1).min(n);
  Ok(sorted[idx - 1].clone())
}

// ─── Moment ──────────────────────────────────────────────────────────

/// Moment[data, r] — the r-th raw moment: Sum[x_i^r] / n.
pub fn moment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Moment".to_string(),
      args: args.to_vec().into(),
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Moment".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let r = &args[1];

  // Raise each element to power r and evaluate, then compute mean
  let mut powered = Vec::with_capacity(items.len());
  for x in items {
    let p = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![x.clone(), r.clone()].into(),
    })?;
    powered.push(p);
  }

  let powered_list = Expr::List(powered.into());
  mean_ast(&[powered_list])
}

/// MeanDeviation[list] - mean absolute deviation from the mean
pub fn mean_deviation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Expr::List(items) = &args[0] {
    if items.is_empty() {
      return Err(InterpreterError::EvaluationError(
        "MeanDeviation: list must not be empty".into(),
      ));
    }
    // Compute mean
    let mean_expr = mean_ast(&[args[0].clone()])?;
    // Compute sum of |xi - mean|
    let n = items.len() as i128;
    let mut abs_devs = Vec::new();
    for item in items {
      let diff = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(item.clone()),
        right: Box::new(mean_expr.clone()),
      };
      let abs_diff = Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![diff].into(),
      };
      abs_devs.push(abs_diff);
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: abs_devs.into(),
    };
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(sum),
      right: Box::new(Expr::Integer(n)),
    };
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "MeanDeviation".to_string(),
      args: args.to_vec().into(),
    })
  }
}

/// MedianDeviation[list] - median absolute deviation from the median
pub fn median_deviation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Expr::List(items) = &args[0] {
    if items.is_empty() {
      return Err(InterpreterError::EvaluationError(
        "MedianDeviation: list must not be empty".into(),
      ));
    }
    // Compute median
    let median_expr = crate::functions::list_helpers_ast::median_ast(&args[0])?;
    // Compute absolute deviations |xi - median|
    let mut abs_devs = Vec::new();
    for item in items {
      let diff = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(item.clone()),
        right: Box::new(median_expr.clone()),
      };
      let abs_diff = Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![diff].into(),
      };
      abs_devs.push(crate::evaluator::evaluate_expr_to_expr(&abs_diff)?);
    }
    // Return median of the absolute deviations
    let devs_list = Expr::List(abs_devs.into());
    crate::functions::list_helpers_ast::median_ast(&devs_list)
  } else {
    Ok(Expr::FunctionCall {
      name: "MedianDeviation".to_string(),
      args: args.to_vec().into(),
    })
  }
}

// ─── LocationTest ─────────────────────────────────────────────────────

/// LocationTest[data] - test if mean is 0 (one-sample t-test, returns p-value)
/// LocationTest[data, mu0] - test if mean is mu0
/// LocationTest[data, mu0, "PValue"] - returns p-value
/// LocationTest[data, mu0, "TestStatistic"] - returns t-statistic
/// LocationTest[data, mu0, "TestDataTable"] - returns test data table
/// LocationTest[{data1, data2}, mu0] - two-sample t-test
pub fn location_test_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "LocationTest expects 1 to 3 arguments".into(),
    ));
  }

  // Parse mu0 (second argument, default 0)
  let mu0 = if args.len() >= 2 {
    match &args[1] {
      Expr::Identifier(s) if s == "Automatic" => 0.0,
      other => {
        if let Some(v) = try_eval_to_f64(other) {
          v
        } else {
          return Ok(Expr::FunctionCall {
            name: "LocationTest".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
    }
  } else {
    0.0
  };

  // Parse property (third argument, default "PValue")
  let property = if args.len() == 3 {
    match &args[2] {
      Expr::String(s) => s.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "LocationTest".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    "PValue".to_string()
  };

  // Determine one-sample vs two-sample
  let data = &args[0];
  match data {
    Expr::List(items) if !items.is_empty() => {
      // Check if it's a two-sample test: {{...}, {...}}
      if items.len() == 2
        && matches!(&items[0], Expr::List(_))
        && matches!(&items[1], Expr::List(_))
      {
        // Two-sample t-test
        let vals1 = extract_numeric_list(&items[0])?;
        let vals2 = extract_numeric_list(&items[1])?;
        if vals1.len() < 2 || vals2.len() < 2 {
          return Err(InterpreterError::EvaluationError(
            "LocationTest: each sample needs at least 2 elements".into(),
          ));
        }
        let (t_stat, df) = two_sample_t_test(&vals1, &vals2, mu0);
        return format_location_test_result(t_stat, df, &property, "T");
      }
      // One-sample t-test
      let vals = extract_numeric_list_flat(items)?;
      if vals.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "LocationTest: need at least 2 elements".into(),
        ));
      }
      let (t_stat, df) = one_sample_t_test(&vals, mu0);
      format_location_test_result(t_stat, df, &property, "T")
    }
    _ => Ok(Expr::FunctionCall {
      name: "LocationTest".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

fn extract_numeric_list(expr: &Expr) -> Result<Vec<f64>, InterpreterError> {
  match expr {
    Expr::List(items) => extract_numeric_list_flat(items),
    _ => Err(InterpreterError::EvaluationError(
      "LocationTest: expected a list of numbers".into(),
    )),
  }
}

fn extract_numeric_list_flat(
  items: &[Expr],
) -> Result<Vec<f64>, InterpreterError> {
  let mut vals = Vec::with_capacity(items.len());
  for item in items {
    if let Some(v) = try_eval_to_f64(item) {
      vals.push(v);
    } else {
      return Err(InterpreterError::EvaluationError(
        "LocationTest: all elements must be numeric".into(),
      ));
    }
  }
  Ok(vals)
}

fn one_sample_t_test(data: &[f64], mu0: f64) -> (f64, f64) {
  let n = data.len() as f64;
  let mean = data.iter().sum::<f64>() / n;
  let variance =
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
  let se = (variance / n).sqrt();
  let t = if se == 0.0 {
    if (mean - mu0).abs() == 0.0 {
      0.0
    } else {
      f64::INFINITY
    }
  } else {
    (mean - mu0) / se
  };
  (t, n - 1.0)
}

fn two_sample_t_test(data1: &[f64], data2: &[f64], mu0: f64) -> (f64, f64) {
  let n1 = data1.len() as f64;
  let n2 = data2.len() as f64;
  let mean1 = data1.iter().sum::<f64>() / n1;
  let mean2 = data2.iter().sum::<f64>() / n2;
  let var1 =
    data1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
  let var2 =
    data2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);
  let se = (var1 / n1 + var2 / n2).sqrt();
  let t = if se == 0.0 {
    let diff = mean1 - mean2 - mu0;
    if diff.abs() == 0.0 {
      0.0
    } else {
      f64::INFINITY
    }
  } else {
    (mean1 - mean2 - mu0) / se
  };
  // Welch-Satterthwaite degrees of freedom
  let num = (var1 / n1 + var2 / n2).powi(2);
  let denom =
    (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
  let df = if denom == 0.0 { 1.0 } else { num / denom };
  (t, df)
}

fn format_location_test_result(
  t_stat: f64,
  df: f64,
  property: &str,
  test_name: &str,
) -> Result<Expr, InterpreterError> {
  match property {
    "TestStatistic" => Ok(num_to_expr(t_stat)),
    "PValue" | _ if property == "PValue" => {
      let p = t_test_p_value(t_stat, df);
      Ok(num_to_expr(p))
    }
    "TestDataTable" => {
      let p = t_test_p_value(t_stat, df);
      // Build Grid[{{, Statistic, P-Value}, {T, t_stat, p}}, ...]
      let header = Expr::List(
        vec![
          Expr::String(String::new()),
          Expr::String("Statistic".to_string()),
          Expr::String("P\u{2010}Value".to_string()),
        ]
        .into(),
      );
      let row = Expr::List(
        vec![
          Expr::String(test_name.to_string()),
          num_to_expr(t_stat),
          num_to_expr(p),
        ]
        .into(),
      );
      Ok(Expr::FunctionCall {
        name: "Grid".to_string(),
        args: vec![
          Expr::List(vec![header, row].into()),
          Expr::FunctionCall {
            name: "Rule".to_string(),
            args: vec![
              Expr::Identifier("Alignment".to_string()),
              Expr::List(
                vec![
                  Expr::Identifier("Left".to_string()),
                  Expr::Identifier("Automatic".to_string()),
                ]
                .into(),
              ),
            ]
            .into(),
          },
          Expr::FunctionCall {
            name: "Rule".to_string(),
            args: vec![
              Expr::Identifier("Dividers".to_string()),
              Expr::List(
                vec![
                  Expr::FunctionCall {
                    name: "Rule".to_string(),
                    args: vec![
                      Expr::Integer(2),
                      Expr::FunctionCall {
                        name: "GrayLevel".to_string(),
                        args: vec![Expr::Real(0.7)].into(),
                      },
                    ]
                    .into(),
                  },
                  Expr::FunctionCall {
                    name: "Rule".to_string(),
                    args: vec![
                      Expr::Integer(2),
                      Expr::FunctionCall {
                        name: "GrayLevel".to_string(),
                        args: vec![Expr::Real(0.7)].into(),
                      },
                    ]
                    .into(),
                  },
                ]
                .into(),
              ),
            ]
            .into(),
          },
          Expr::FunctionCall {
            name: "Rule".to_string(),
            args: vec![
              Expr::Identifier("Spacings".to_string()),
              Expr::Identifier("Automatic".to_string()),
            ]
            .into(),
          },
        ]
        .into(),
      })
    }
    _ => {
      // Default to PValue for unknown properties
      let p = t_test_p_value(t_stat, df);
      Ok(num_to_expr(p))
    }
  }
}

fn t_test_p_value(t_stat: f64, df: f64) -> f64 {
  // Two-tailed p-value: p = I(df/(df+t^2), df/2, 1/2)
  if t_stat.is_infinite() {
    return 0.0;
  }
  let x = df / (df + t_stat * t_stat);
  regularized_beta_inc(x, df / 2.0, 0.5)
}

// ─── Likelihood ───────────────────────────────────────────────────────

/// Likelihood[dist, {x1, x2, ...}] - product of PDF[dist, xi] for each xi
pub fn likelihood_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Likelihood".to_string(),
      args: args.to_vec().into(),
    });
  }

  let dist = &args[0];
  let data = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Likelihood".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if data.is_empty() {
    return Ok(Expr::Integer(1));
  }

  // Check if all data is numeric (for forcing numerical evaluation)
  let all_numeric = data.iter().all(|x| try_eval_to_f64(x).is_some());

  // Compute PDF[dist, xi] for each data point
  let mut pdf_values = Vec::with_capacity(data.len());
  for xi in data {
    let pdf_expr = Expr::FunctionCall {
      name: "PDF".to_string(),
      args: vec![dist.clone(), xi.clone()].into(),
    };
    let pdf_val = crate::evaluator::evaluate_expr_to_expr(&pdf_expr)?;
    pdf_values.push(pdf_val);
  }

  // Multiply them: Times[pdf1, pdf2, ...]
  let product = if pdf_values.len() == 1 {
    pdf_values.into_iter().next().unwrap()
  } else {
    let product_expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: pdf_values.into(),
    };
    crate::evaluator::evaluate_expr_to_expr(&product_expr)?
  };

  // If all data is numeric, try to evaluate the result numerically
  if all_numeric {
    if let Some(val) = try_eval_to_f64(&product) {
      return Ok(num_to_expr(val));
    }
    // Try N[product]
    let n_expr = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![product.clone()].into(),
    };
    if let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&n_expr)
      && try_eval_to_f64(&result).is_some()
    {
      return Ok(result);
    }
  }
  Ok(product)
}

// ─── PearsonChiSquareTest ─────────────────────────────────────────────

/// PearsonChiSquareTest[data] - chi-square goodness-of-fit test (normal with estimated params)
/// PearsonChiSquareTest[data, dist] - test against a specific distribution
/// PearsonChiSquareTest[data, dist, "PValue"] - p-value
/// PearsonChiSquareTest[data, dist, "TestStatistic"] - chi-square statistic
pub fn pearson_chi_square_test_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "PearsonChiSquareTest".to_string(),
      args: args.to_vec().into(),
    });
  }

  let data = match &args[0] {
    Expr::List(items) if items.len() >= 2 => {
      let mut vals = Vec::new();
      for item in items {
        if let Some(v) = try_eval_to_f64(item) {
          vals.push(v);
        } else {
          return Ok(Expr::FunctionCall {
            name: "PearsonChiSquareTest".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
      vals
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PearsonChiSquareTest".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Determine the null distribution and number of estimated parameters
  let (dist_cdf, estimated_params): (Box<dyn Fn(f64) -> f64>, usize) = if args
    .len()
    >= 2
  {
    match &args[1] {
      Expr::Identifier(s) if s == "Automatic" => {
        // Normal with estimated mean and variance
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var =
          data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sd = var.sqrt();
        (
          Box::new(move |x: f64| {
            0.5 * (1.0 + erf_f64((x - mean) / (sd * std::f64::consts::SQRT_2)))
          }),
          2, // estimated mean and variance
        )
      }
      Expr::FunctionCall { name, args: dargs } => match name.as_str() {
        "NormalDistribution" => {
          let (mu, sigma) = match dargs.len() {
            0 => (0.0, 1.0),
            2 => {
              let mu = try_eval_to_f64(&dargs[0]).unwrap_or(0.0);
              let sigma = try_eval_to_f64(&dargs[1]).unwrap_or(1.0);
              (mu, sigma)
            }
            _ => (0.0, 1.0),
          };
          (
            Box::new(move |x: f64| {
              0.5
                * (1.0 + erf_f64((x - mu) / (sigma * std::f64::consts::SQRT_2)))
            }),
            0,
          )
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "PearsonChiSquareTest".to_string(),
            args: args.to_vec().into(),
          });
        }
      },
      _ => {
        return Ok(Expr::FunctionCall {
          name: "PearsonChiSquareTest".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    // Default: Normal with estimated parameters
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    (
      Box::new(move |x: f64| {
        0.5 * (1.0 + erf_f64((x - mean) / (sd * std::f64::consts::SQRT_2)))
      }),
      2,
    )
  };

  // Parse property
  let property = if args.len() == 3 {
    match &args[2] {
      Expr::String(s) => s.clone(),
      _ => "PValue".to_string(),
    }
  } else {
    "PValue".to_string()
  };

  let n = data.len();
  let k = (2.0 * (n as f64).powf(0.4)).floor() as usize;
  let k = k.max(2);

  // Compute CDF values for each data point and bin into k equal bins
  let mut bin_counts = vec![0usize; k];
  for &x in &data {
    let cdf_val = dist_cdf(x);
    let bin = (cdf_val * k as f64).floor() as usize;
    let bin = bin.min(k - 1);
    bin_counts[bin] += 1;
  }

  let expected = n as f64 / k as f64;
  let chi_sq: f64 = bin_counts
    .iter()
    .map(|&o| {
      let diff = o as f64 - expected;
      diff * diff / expected
    })
    .sum();

  let df = (k as f64 - 1.0 - estimated_params as f64).max(1.0);

  match property.as_str() {
    "TestStatistic" => Ok(num_to_expr(chi_sq)),
    _ => {
      // P-value from chi-square distribution: P(X > chi_sq) where X ~ ChiSquare(df)
      // Using regularized gamma: 1 - GammaRegularized(df/2, chi_sq/2)
      let p = 1.0 - regularized_gamma_lower(df / 2.0, chi_sq / 2.0);
      Ok(num_to_expr(p))
    }
  }
}

/// Regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a)
/// Used for chi-square CDF: CDF(x, df) = P(df/2, x/2)
fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
  if x <= 0.0 {
    return 0.0;
  }
  if x < a + 1.0 {
    // Series expansion
    gamma_series(a, x)
  } else {
    // Continued fraction
    1.0 - gamma_cf(a, x)
  }
}

/// Series expansion for lower incomplete gamma
fn gamma_series(a: f64, x: f64) -> f64 {
  let mut sum = 1.0 / a;
  let mut term = 1.0 / a;
  for n in 1..200 {
    term *= x / (a + n as f64);
    sum += term;
    if term.abs() < sum.abs() * 1e-15 {
      break;
    }
  }
  sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued fraction for upper incomplete gamma
fn gamma_cf(a: f64, x: f64) -> f64 {
  let mut f = x + 1.0 - a;
  if f.abs() < 1e-30 {
    f = 1e-30;
  }
  let mut c = 1.0 / 1e-30;
  let mut d = 1.0 / f;
  let mut result = d;
  for i in 1..200 {
    let an = -(i as f64) * (i as f64 - a);
    let bn = x + 2.0 * i as f64 + 1.0 - a;
    d = bn + an * d;
    if d.abs() < 1e-30 {
      d = 1e-30;
    }
    c = bn + an / c;
    if c.abs() < 1e-30 {
      c = 1e-30;
    }
    d = 1.0 / d;
    let delta = c * d;
    result *= delta;
    if (delta - 1.0).abs() < 1e-15 {
      break;
    }
  }
  result * (-x + a * x.ln() - ln_gamma(a)).exp()
}

// ─── Longitude / Latitude ─────────────────────────────────────────────

/// Longitude[GeoPosition[{lat, lon}]] or Longitude[{lat, lon}] → Quantity[lon, "AngularDegrees"]
pub fn longitude_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Longitude".to_string(),
      args: args.to_vec().into(),
    });
  }
  let coords = extract_geo_coords(&args[0]);
  match coords {
    Some((_, lon)) => Ok(Expr::FunctionCall {
      name: "Quantity".to_string(),
      args: vec![lon, Expr::String("AngularDegrees".to_string())].into(),
    }),
    None => Ok(Expr::FunctionCall {
      name: "Longitude".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Latitude[GeoPosition[{lat, lon}]] or Latitude[{lat, lon}] → Quantity[lat, "AngularDegrees"]
pub fn latitude_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Latitude".to_string(),
      args: args.to_vec().into(),
    });
  }
  let coords = extract_geo_coords(&args[0]);
  match coords {
    Some((lat, _)) => Ok(Expr::FunctionCall {
      name: "Quantity".to_string(),
      args: vec![lat, Expr::String("AngularDegrees".to_string())].into(),
    }),
    None => Ok(Expr::FunctionCall {
      name: "Latitude".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Extract (lat, lon) from GeoPosition[{lat, lon}] or {lat, lon}
fn extract_geo_coords(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::List(items) if items.len() >= 2 => {
      Some((items[0].clone(), items[1].clone()))
    }
    Expr::FunctionCall { name, args }
      if name == "GeoPosition" && args.len() == 1 =>
    {
      if let Expr::List(items) = &args[0]
        && items.len() >= 2
      {
        return Some((items[0].clone(), items[1].clone()));
      }
      None
    }
    _ => None,
  }
}

/// LatitudeLongitude[GeoPosition[{lat, lon}]] → {Quantity[lat, "AngularDegrees"], Quantity[lon, "AngularDegrees"]}
pub fn latitude_longitude_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "LatitudeLongitude".to_string(),
      args: args.to_vec().into(),
    });
  }
  match extract_geo_coords(&args[0]) {
    Some((lat, lon)) => Ok(Expr::List(
      vec![
        Expr::FunctionCall {
          name: "Quantity".to_string(),
          args: vec![lat, Expr::String("AngularDegrees".to_string())].into(),
        },
        Expr::FunctionCall {
          name: "Quantity".to_string(),
          args: vec![lon, Expr::String("AngularDegrees".to_string())].into(),
        },
      ]
      .into(),
    )),
    None => Ok(Expr::FunctionCall {
      name: "LatitudeLongitude".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

// ─── GroupGenerators ──────────────────────────────────────────────────

/// GroupGenerators[group] - return a list of generators for the given group
pub fn group_generators_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "GroupGenerators".to_string(),
      args: args.to_vec().into(),
    });
  }

  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "AbelianGroup"
    && gargs.len() == 1
    && let Some(factors) = abelian_factors(&gargs[0])
  {
    return Ok(abelian_group_generators(&factors));
  }

  // GroupGenerators[PermutationGroup[{gens}]] → the same generator list.
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "PermutationGroup"
    && gargs.len() == 1
    && matches!(&gargs[0], Expr::List(_))
  {
    return Ok(gargs[0].clone());
  }

  match &args[0] {
    Expr::FunctionCall { name, args: gargs } if gargs.len() == 1 => {
      let n = match &gargs[0] {
        Expr::Integer(n) if *n >= 1 => *n as usize,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "GroupGenerators".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      match name.as_str() {
        "SymmetricGroup" => Ok(symmetric_group_generators(n)),
        "CyclicGroup" => Ok(cyclic_group_generators(n)),
        "DihedralGroup" => Ok(dihedral_group_generators(n)),
        "AlternatingGroup" => Ok(alternating_group_generators(n)),
        _ => Ok(Expr::FunctionCall {
          name: "GroupGenerators".to_string(),
          args: args.to_vec().into(),
        }),
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "GroupGenerators".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Extract non-negative-integer factors {n1, n2, ...} from an AbelianGroup arg.
fn abelian_factors(expr: &Expr) -> Option<Vec<usize>> {
  if let Expr::List(items) = expr {
    let mut out = Vec::with_capacity(items.len());
    for item in items.iter() {
      if let Expr::Integer(n) = item
        && *n >= 1
      {
        out.push(*n as usize);
      } else {
        return None;
      }
    }
    Some(out)
  } else {
    None
  }
}

fn abelian_group_generators(factors: &[usize]) -> Expr {
  let mut gens = Vec::new();
  let mut start: i128 = 1;
  for &ni in factors {
    if ni >= 2 {
      let slots: Vec<i128> = (start..start + ni as i128).collect();
      gens.push(make_cycles(slots));
    }
    start += ni as i128;
  }
  Expr::List(gens.into())
}

/// GroupOrder[group] - the number of elements in a group.
pub fn group_order_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "GroupOrder".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && gargs.len() == 1
  {
    if name == "AbelianGroup"
      && let Some(factors) = abelian_factors(&gargs[0])
    {
      let order: i128 = factors.iter().map(|&n| n as i128).product();
      return Ok(Expr::Integer(order));
    }
    if let Expr::Integer(n) = &gargs[0]
      && *n >= 1
    {
      let n = *n;
      return Ok(match name.as_str() {
        "CyclicGroup" => Expr::Integer(n),
        "SymmetricGroup" => Expr::Integer((1..=n).product()),
        "AlternatingGroup" => {
          if n <= 1 {
            Expr::Integer(1)
          } else {
            Expr::Integer((1..=n).product::<i128>() / 2)
          }
        }
        "DihedralGroup" => {
          if n == 1 {
            Expr::Integer(2)
          } else {
            Expr::Integer(2 * n)
          }
        }
        _ => unevaluated(),
      });
    }
  }
  Ok(unevaluated())
}

/// GroupElements[group] - list of all elements in a group as cycle notation.
pub fn group_elements_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "GroupElements".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "AbelianGroup"
    && gargs.len() == 1
    && let Some(factors) = abelian_factors(&gargs[0])
  {
    return Ok(abelian_group_elements(&factors));
  }
  Ok(unevaluated())
}

fn abelian_group_elements(factors: &[usize]) -> Expr {
  // Slot ranges per factor: slot_starts[i] is the first slot of factor i.
  let mut slot_starts: Vec<i128> = Vec::with_capacity(factors.len());
  let mut start: i128 = 1;
  for &ni in factors {
    slot_starts.push(start);
    start += ni as i128;
  }

  // Iterate (k_1, ..., k_m) in lex order with the rightmost varying fastest.
  let mut powers = vec![0usize; factors.len()];
  let mut elements = Vec::new();
  loop {
    let mut cycles_for_element: Vec<Vec<i128>> = Vec::new();
    for (i, &ni) in factors.iter().enumerate() {
      let k = powers[i];
      if k == 0 || ni < 2 {
        continue;
      }
      let s = slot_starts[i];
      let slots: Vec<i128> = (s..s + ni as i128).collect();
      cycles_for_element.extend(cyclic_power_cycles(&slots, k));
    }
    elements.push(make_cycles_multi(cycles_for_element));

    // Increment the odometer from the right.
    if factors.is_empty() {
      break;
    }
    let mut idx = factors.len();
    let mut carried = true;
    while idx > 0 && carried {
      idx -= 1;
      powers[idx] += 1;
      if powers[idx] >= factors[idx] {
        powers[idx] = 0;
      } else {
        carried = false;
      }
    }
    if carried {
      break;
    }
  }
  Expr::List(elements.into())
}

/// Compute the disjoint-cycle decomposition of the k-th power of a single
/// cyclic permutation acting on `slots` (treated in order). Returns an empty
/// vec for the identity (k % n == 0).
fn cyclic_power_cycles(slots: &[i128], k: usize) -> Vec<Vec<i128>> {
  let n = slots.len();
  if n == 0 || k.is_multiple_of(n) {
    return Vec::new();
  }
  let mut visited = vec![false; n];
  let mut cycles = Vec::new();
  for start in 0..n {
    if visited[start] {
      continue;
    }
    let mut cycle = Vec::new();
    let mut i = start;
    loop {
      visited[i] = true;
      cycle.push(slots[i]);
      i = (i + k) % n;
      if i == start {
        break;
      }
    }
    if cycle.len() > 1 {
      cycles.push(cycle);
    }
  }
  cycles
}

fn make_cycles(cycle: Vec<i128>) -> Expr {
  Expr::FunctionCall {
    name: "Cycles".to_string(),
    args: vec![Expr::List(
      vec![Expr::List(cycle.into_iter().map(Expr::Integer).collect())].into(),
    )]
    .into(),
  }
}

fn make_cycles_multi(cycles: Vec<Vec<i128>>) -> Expr {
  Expr::FunctionCall {
    name: "Cycles".to_string(),
    args: vec![Expr::List(
      cycles
        .into_iter()
        .map(|c| Expr::List(c.into_iter().map(Expr::Integer).collect()))
        .collect(),
    )]
    .into(),
  }
}

fn symmetric_group_generators(n: usize) -> Expr {
  if n <= 1 {
    return Expr::List(vec![make_cycles_multi(vec![])].into());
  }
  let transposition = make_cycles(vec![1, 2]);
  let n_cycle: Vec<i128> = (1..=n as i128).collect();
  let rotation = make_cycles(n_cycle);
  Expr::List(vec![transposition, rotation].into())
}

fn cyclic_group_generators(n: usize) -> Expr {
  if n <= 1 {
    return Expr::List(vec![make_cycles_multi(vec![])].into());
  }
  let n_cycle: Vec<i128> = (1..=n as i128).collect();
  Expr::List(vec![make_cycles(n_cycle)].into())
}

fn dihedral_group_generators(n: usize) -> Expr {
  if n <= 1 {
    return Expr::List(vec![make_cycles_multi(vec![])].into());
  }
  if n == 2 {
    return Expr::List(
      vec![make_cycles(vec![1, 2]), make_cycles(vec![3, 4])].into(),
    );
  }

  // Reflection: for even n pairs (i, n+1-i); for odd n pairs (i, n+2-i) for i>=2
  let mut reflection_cycles = Vec::new();
  if n.is_multiple_of(2) {
    for i in 1..=(n / 2) {
      reflection_cycles.push(vec![i as i128, (n + 1 - i) as i128]);
    }
  } else {
    // Odd n: element 1 is fixed, pairs are (2,n), (3,n-1), ...
    for i in 2..=n.div_ceil(2) {
      reflection_cycles.push(vec![i as i128, (n + 2 - i) as i128]);
    }
  }
  let reflection = make_cycles_multi(reflection_cycles);
  let n_cycle: Vec<i128> = (1..=n as i128).collect();
  let rotation = make_cycles(n_cycle);
  Expr::List(vec![reflection, rotation].into())
}

fn alternating_group_generators(n: usize) -> Expr {
  if n <= 2 {
    return Expr::List(vec![make_cycles_multi(vec![])].into());
  }
  if n == 3 {
    return Expr::List(vec![make_cycles(vec![1, 2, 3])].into());
  }
  let three_cycle = make_cycles(vec![1, 2, 3]);
  let second_gen = if n % 2 == 1 {
    make_cycles((1..=n as i128).collect())
  } else {
    make_cycles((2..=n as i128).collect())
  };
  Expr::List(vec![three_cycle, second_gen].into())
}

// ─── DiscreteAsymptotic ───────────────────────────────────────────────

/// DiscreteAsymptotic[expr, n -> Infinity] - leading asymptotic term of expr as n -> Infinity
pub fn discrete_asymptotic_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "DiscreteAsymptotic".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Parse second argument: must be Rule[var, Infinity] or Expr::Rule { var, Infinity }
  let var_name = match &args[1] {
    Expr::FunctionCall { name, args: rargs }
      if name == "Rule" && rargs.len() == 2 =>
    {
      match (&rargs[0], &rargs[1]) {
        (Expr::Identifier(s), Expr::Identifier(inf)) if inf == "Infinity" => {
          s.clone()
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "DiscreteAsymptotic".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    } => match (pattern.as_ref(), replacement.as_ref()) {
      (Expr::Identifier(s), Expr::Identifier(inf)) if inf == "Infinity" => {
        s.clone()
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DiscreteAsymptotic".to_string(),
          args: args.to_vec().into(),
        });
      }
    },
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DiscreteAsymptotic".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let expr = &args[0];
  match discrete_asymptotic_leading(expr, &var_name) {
    Some(result) => crate::evaluator::evaluate_expr_to_expr(&result),
    None => Ok(Expr::FunctionCall {
      name: "DiscreteAsymptotic".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// Compute the leading asymptotic term of an expression as var -> Infinity.
/// Returns None if the expression cannot be handled.
fn discrete_asymptotic_leading(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    // Constants don't depend on var
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => Some(expr.clone()),
    Expr::Identifier(name) if name == var => Some(expr.clone()),
    Expr::Identifier(name) if name != var => Some(expr.clone()),

    // Factorial[var] → Stirling: var^(var+1/2) * Sqrt[2*Pi] / E^var
    Expr::FunctionCall { name, args }
      if name == "Factorial"
        && args.len() == 1
        && is_pure_var(&args[0], var) =>
    {
      Some(stirling_approx(var))
    }

    // Gamma[var] → (var-1)! ~ var^(var-1/2) * Sqrt[2*Pi] / E^var  (but actually Gamma(n) = (n-1)!)
    // Gamma[n] → n^(n - 1/2) * Sqrt[2*Pi] / E^n  leading term
    Expr::FunctionCall { name, args }
      if name == "Gamma" && args.len() == 1 && is_pure_var(&args[0], var) =>
    {
      // Gamma[n] = (n-1)! ~ n^(n-1/2) * Sqrt[2*Pi] / E^n
      let n = Expr::Identifier(var.to_string());
      Some(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          // n^(n - 1/2)
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              n.clone(),
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  n.clone(),
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
          },
          // Sqrt[2*Pi]
          make_sqrt(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]
              .into(),
          }),
          // E^(-n)
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::Constant("E".to_string()),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), n].into(),
              },
            ]
            .into(),
          },
        ]
        .into(),
      })
    }

    // HarmonicNumber[var] → Log[var]
    Expr::FunctionCall { name, args }
      if name == "HarmonicNumber"
        && args.len() == 1
        && is_pure_var(&args[0], var) =>
    {
      Some(Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::Identifier(var.to_string())].into(),
      })
    }

    // Power[base, exp] - handle var^k, k^var, etc.
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      // If neither depends on var, return as-is
      if !contains_var(&args[0], var) && !contains_var(&args[1], var) {
        return Some(expr.clone());
      }
      // var^const or const^var - already in asymptotic form
      Some(expr.clone())
    }

    // BinaryOp Power
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      ..
    } => Some(expr.clone()),

    // Plus: find the dominant term
    Expr::FunctionCall { name, args } if name == "Plus" && !args.is_empty() => {
      asymptotic_sum(args, var)
    }

    // BinaryOp Plus
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => asymptotic_sum(&[*left.clone(), *right.clone()], var),

    // BinaryOp Minus
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left,
      right,
    } => {
      let neg_right = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), *right.clone()].into(),
      };
      asymptotic_sum(&[*left.clone(), neg_right], var)
    }

    // Times: take asymptotic of each factor
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let mut result_factors = Vec::new();
      for arg in args {
        match discrete_asymptotic_leading(arg, var) {
          Some(a) => result_factors.push(a),
          None => return None,
        }
      }
      if result_factors.len() == 1 {
        Some(result_factors.into_iter().next().unwrap())
      } else {
        Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: result_factors.into(),
        })
      }
    }

    // BinaryOp Times
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let l = discrete_asymptotic_leading(left, var)?;
      let r = discrete_asymptotic_leading(right, var)?;
      Some(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![l, r].into(),
      })
    }

    // BinaryOp Divide
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      let l = discrete_asymptotic_leading(left, var)?;
      let r = discrete_asymptotic_leading(right, var)?;
      Some(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(l),
        right: Box::new(r),
      })
    }

    // Sqrt[expr]
    expr if is_sqrt(expr).is_some() => {
      let sqrt_arg = is_sqrt(expr).unwrap();
      let inner = discrete_asymptotic_leading(sqrt_arg, var)?;
      Some(make_sqrt(inner))
    }

    // Log[expr] - keep as is if it depends on var
    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      Some(expr.clone())
    }

    // Binomial[var, var/2] → 2^(1/2+var) / (Sqrt[var]*Sqrt[Pi])
    Expr::FunctionCall { name, args }
      if name == "Binomial" && args.len() == 2 =>
    {
      if is_pure_var(&args[0], var)
        && let Some(result) = asymptotic_binomial(&args[0], &args[1], var)
      {
        return Some(result);
      }
      None
    }

    // If expression doesn't contain var, it's a constant
    _ if !contains_var(expr, var) => Some(expr.clone()),

    _ => None,
  }
}

/// Check if expr is exactly the variable
fn is_pure_var(expr: &Expr, var: &str) -> bool {
  matches!(expr, Expr::Identifier(name) if name == var)
}

/// Check if expression contains the variable
fn contains_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name == var,
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => false,
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| contains_var(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_var(left, var) || contains_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_var(operand, var),
    Expr::List(items) => items.iter().any(|a| contains_var(a, var)),
    _ => false,
  }
}

/// Stirling's approximation: n! ~ n^(n+1/2) * Sqrt[2*Pi] / E^n
fn stirling_approx(var: &str) -> Expr {
  let n = Expr::Identifier(var.to_string());
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      // n^(n + 1/2)
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          n.clone(),
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              n.clone(),
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
            ]
            .into(),
          },
        ]
        .into(),
      },
      // Sqrt[2*Pi]
      make_sqrt(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string())].into(),
      }),
      // E^(-n)
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::Constant("E".to_string()),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), n].into(),
          },
        ]
        .into(),
      },
    ]
    .into(),
  }
}

/// Determine growth rate class for comparison.
/// Returns a rough numerical "growth order" for comparing dominance:
/// constants < log < polynomial < exponential < factorial
fn growth_order(expr: &Expr, var: &str) -> Option<f64> {
  if !contains_var(expr, var) {
    return Some(0.0); // constant
  }
  match expr {
    Expr::Identifier(name) if name == var => Some(1.0), // linear ~ n^1

    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      Some(0.001) // Log grows slower than any polynomial
    }

    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if is_pure_var(&args[0], var) && !contains_var(&args[1], var) {
        // var^k - polynomial
        try_eval_to_f64(&args[1]).map(|k| k)
      } else if !contains_var(&args[0], var) && contains_var(&args[1], var) {
        // const^var - exponential
        try_eval_to_f64(&args[0]).map(|base| 100.0 + base.ln())
      } else {
        Some(200.0) // var^var or similar - super-exponential
      }
    }

    Expr::FunctionCall { name, args } if name == "Times" => {
      // Product: max of growth orders (approximately)
      let mut max_order = 0.0f64;
      for arg in args {
        if let Some(o) = growth_order(arg, var) {
          max_order = max_order.max(o);
        } else {
          return None;
        }
      }
      Some(max_order)
    }

    Expr::FunctionCall { name, args }
      if name == "Factorial"
        && args.len() == 1
        && is_pure_var(&args[0], var) =>
    {
      Some(300.0) // factorial grows faster than exponential
    }

    expr if is_sqrt(expr).is_some() => {
      growth_order(is_sqrt(expr).unwrap(), var).map(|o| o * 0.5)
    }

    // BinaryOp variants
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let l = growth_order(left, var)?;
      let r = growth_order(right, var)?;
      Some(l.max(r))
    }

    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if is_pure_var(left, var) && !contains_var(right, var) {
        try_eval_to_f64(right)
      } else if !contains_var(left, var) && contains_var(right, var) {
        try_eval_to_f64(left).map(|base| 100.0 + base.ln())
      } else {
        Some(200.0)
      }
    }

    _ => None,
  }
}

/// Find the dominant term in a sum
fn asymptotic_sum(terms: &[Expr], var: &str) -> Option<Expr> {
  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return discrete_asymptotic_leading(&terms[0], var);
  }

  // Get asymptotic of each term and find the dominant one
  let mut best_expr = discrete_asymptotic_leading(&terms[0], var)?;
  let mut best_order = growth_order(&best_expr, var).unwrap_or(0.0);

  for term in &terms[1..] {
    let asym = discrete_asymptotic_leading(term, var)?;
    let order = growth_order(&asym, var).unwrap_or(0.0);
    if order > best_order {
      best_expr = asym;
      best_order = order;
    } else if (order - best_order).abs() < 1e-10 {
      // Same order - need to add them (e.g. 3n^2 + 5n^2 -> 8n^2)
      // For simplicity, if they have the same growth order, keep as sum
      best_expr = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![best_expr, asym].into(),
      };
    }
  }
  Some(best_expr)
}

/// Asymptotic of Binomial[n, n/2] → 2^(n+1/2) / (Sqrt[n]*Sqrt[Pi])
fn asymptotic_binomial(
  n_expr: &Expr,
  k_expr: &Expr,
  var: &str,
) -> Option<Expr> {
  // Check if k = n/2
  let is_half = match k_expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => is_pure_var(left, var) && matches!(**right, Expr::Integer(2)),
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      let has_half = args.iter().any(|a| {
        matches!(a, Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational" && ra.len() == 2
          && matches!((&ra[0], &ra[1]), (Expr::Integer(1), Expr::Integer(2))))
      });
      let has_var = args.iter().any(|a| is_pure_var(a, var));
      has_half && has_var
    }
    _ => false,
  };

  if !is_half || !is_pure_var(n_expr, var) {
    return None;
  }

  let n = Expr::Identifier(var.to_string());
  // 2^(1/2 + n) / (Sqrt[n] * Sqrt[Pi])
  Some(Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      // 2^(1/2 + n)
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::Integer(2),
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
              n.clone(),
            ]
            .into(),
          },
        ]
        .into(),
      },
      // 1 / (Sqrt[n] * Sqrt[Pi])
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              make_sqrt(n),
              make_sqrt(Expr::Constant("Pi".to_string())),
            ]
            .into(),
          },
          Expr::Integer(-1),
        ]
        .into(),
      },
    ]
    .into(),
  })
}

// ─── CovarianceFunction[ARMAProcess[...], s, t] ───────────────────────

/// CovarianceFunction[proc, s, t] gives the autocovariance Cov[X_s, X_t]
/// for the stochastic process `proc`. This implementation handles
/// ARMA(p, q) processes for (p, q) ∈ {(1,0), (0,1), (1,1)}, returning
/// a closed-form expression that matches wolframscript.
pub fn covariance_function_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "CovarianceFunction".to_string(),
      args: args.to_vec().into(),
    });
  }
  if let Some(result) = arma_covariance(&args[0], &args[1], &args[2]) {
    // Re-evaluate so the inner Times/Divide combine into canonical
    // negative-power form. Without this, the printer renders sub-terms
    // as the parser's literal `Times[-1, x/y]` instead of `-((x*y)/z)`.
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }
  Ok(Expr::FunctionCall {
    name: "CovarianceFunction".to_string(),
    args: args.to_vec().into(),
  })
}

/// Extract `(ar_list, ma_list, variance)` from an `ARMAProcess[...]` call,
/// stripping a leading scalar constant if present.
fn extract_arma_params(proc: &Expr) -> Option<(Vec<Expr>, Vec<Expr>, Expr)> {
  let (name, args) = match proc {
    Expr::FunctionCall { name, args } => (name.as_str(), args),
    _ => return None,
  };
  if name != "ARMAProcess" {
    return None;
  }
  let a: Vec<Expr> = args.iter().cloned().collect();
  // Strip optional leading scalar constant: ARMAProcess[c, ar, ma, v, …].
  let rest: &[Expr] = if !a.is_empty() && matches!(&a[0], Expr::List(_)) {
    &a[..]
  } else if a.len() >= 4 && matches!(&a[1], Expr::List(_)) {
    &a[1..]
  } else {
    return None;
  };
  if rest.len() < 3 {
    return None;
  }
  let ar = match &rest[0] {
    Expr::List(xs) => xs.iter().cloned().collect(),
    _ => return None,
  };
  let ma = match &rest[1] {
    Expr::List(xs) => xs.iter().cloned().collect(),
    _ => return None,
  };
  Some((ar, ma, rest[2].clone()))
}

fn arma_covariance(proc: &Expr, s: &Expr, t: &Expr) -> Option<Expr> {
  let (ar, ma, sigma2) = extract_arma_params(proc)?;
  match (ar.len(), ma.len()) {
    (1, 0) => Some(cov_ar1(&ar[0], &sigma2, s, t)),
    (0, 1) => Some(cov_ma1(&ma[0], &sigma2, s, t)),
    (1, 1) => Some(cov_arma11(&ar[0], &ma[0], &sigma2, s, t)),
    _ => None,
  }
}

/// Evaluate `e` once so Times/Divide chains collapse into the canonical
/// form Woxi prints. Piecewise leaves unevaluated branches alone when
/// the condition stays symbolic, so we have to do this before nesting.
fn eval_once(e: Expr) -> Expr {
  crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
}

// ─── AST builder helpers ────────────────────────────────────────────────

fn fc(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}
fn cf_plus(xs: Vec<Expr>) -> Expr {
  fc("Plus", xs)
}
fn cf_times(xs: Vec<Expr>) -> Expr {
  fc("Times", xs)
}
fn cf_pow(b: Expr, e: Expr) -> Expr {
  fc("Power", vec![b, e])
}
fn cf_div(n: Expr, d: Expr) -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(n),
    right: Box::new(d),
  }
}
fn cf_neg(x: Expr) -> Expr {
  cf_times(vec![Expr::Integer(-1), x])
}
fn cf_abs(x: Expr) -> Expr {
  fc("Abs", vec![x])
}
fn cf_abs_diff(s: &Expr, t: &Expr) -> Expr {
  cf_abs(cf_plus(vec![s.clone(), cf_neg(t.clone())]))
}

// ─── AR(1): (a^|s-t| σ²) / (1 - a²) ─────────────────────────────────────

fn cov_ar1(a: &Expr, sigma2: &Expr, s: &Expr, t: &Expr) -> Expr {
  let numerator = cf_times(vec![
    cf_pow(a.clone(), cf_abs_diff(s, t)),
    sigma2.clone(),
  ]);
  let denominator = cf_plus(vec![
    Expr::Integer(1),
    cf_neg(cf_pow(a.clone(), Expr::Integer(2))),
  ]);
  cf_div(numerator, denominator)
}

// ─── MA(1): Piecewise[{{bσ², |s-t|==1}, {(1+b²)σ², |s-t|==0}}, 0] ──────

fn cov_ma1(b: &Expr, sigma2: &Expr, s: &Expr, t: &Expr) -> Expr {
  let lag = cf_abs_diff(s, t);
  let case1 = Expr::List(
    vec![
      cf_times(vec![b.clone(), sigma2.clone()]),
      Expr::Comparison {
        operands: vec![lag.clone(), Expr::Integer(1)],
        operators: vec![crate::syntax::ComparisonOp::Equal],
      },
    ]
    .into(),
  );
  let case0 = Expr::List(
    vec![
      cf_times(vec![
        cf_plus(vec![Expr::Integer(1), cf_pow(b.clone(), Expr::Integer(2))]),
        sigma2.clone(),
      ]),
      Expr::Comparison {
        operands: vec![lag.clone(), Expr::Integer(0)],
        operators: vec![crate::syntax::ComparisonOp::Equal],
      },
    ]
    .into(),
  );
  fc(
    "Piecewise",
    vec![
      Expr::List(vec![case1, case0].into()),
      Expr::Integer(0),
    ],
  )
}

// ─── ARMA(1, 1) ─────────────────────────────────────────────────────────
//
// γ(h>0) = -((a^(h-1) * (a + b + a²b + ab²) * σ²) / ((a-1)(a+1)))
// γ(0)   = -(bσ²/a) - ((a + b + a²b + ab²)σ²) / ((a-1) a (a+1))

fn arma11_top_poly(a: &Expr, b: &Expr) -> Expr {
  // a + b + a^2*b + a*b^2
  cf_plus(vec![
    a.clone(),
    b.clone(),
    cf_times(vec![cf_pow(a.clone(), Expr::Integer(2)), b.clone()]),
    cf_times(vec![a.clone(), cf_pow(b.clone(), Expr::Integer(2))]),
  ])
}

fn cov_arma11(a: &Expr, b: &Expr, sigma2: &Expr, s: &Expr, t: &Expr) -> Expr {
  let lag = cf_abs_diff(s, t);
  let top_poly = arma11_top_poly(a, b);

  // Nonzero-lag branch numerator: a^(-1 + lag) * top_poly * σ²
  let nonzero_num = cf_times(vec![
    cf_pow(
      a.clone(),
      cf_plus(vec![Expr::Integer(-1), lag.clone()]),
    ),
    top_poly.clone(),
    sigma2.clone(),
  ]);
  // Denominator: (-1 + a) * (1 + a)
  let nonzero_den = cf_times(vec![
    cf_plus(vec![Expr::Integer(-1), a.clone()]),
    cf_plus(vec![Expr::Integer(1), a.clone()]),
  ]);
  let nonzero_value = eval_once(cf_neg(cf_div(nonzero_num, nonzero_den)));

  // Zero-lag (default) value: -(bσ²/a) - (top_poly σ²)/((-1+a) a (1+a))
  let default_term1 = eval_once(cf_neg(cf_div(
    cf_times(vec![b.clone(), sigma2.clone()]),
    a.clone(),
  )));
  let default_denom = cf_times(vec![
    cf_plus(vec![Expr::Integer(-1), a.clone()]),
    a.clone(),
    cf_plus(vec![Expr::Integer(1), a.clone()]),
  ]);
  let default_term2 = eval_once(cf_neg(cf_div(
    cf_times(vec![top_poly, sigma2.clone()]),
    default_denom,
  )));
  let default_value = eval_once(cf_plus(vec![default_term1, default_term2]));

  let case = Expr::List(
    vec![
      nonzero_value,
      Expr::Comparison {
        operands: vec![lag, Expr::Integer(0)],
        operators: vec![crate::syntax::ComparisonOp::Greater],
      },
    ]
    .into(),
  );
  fc(
    "Piecewise",
    vec![Expr::List(vec![case].into()), default_value],
  )
}

// ─── PowerExpand ──────────────────────────────────────────────────────
