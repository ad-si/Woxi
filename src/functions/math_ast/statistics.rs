use crate::InterpreterError;
use crate::syntax::Expr;
#[allow(unused_imports)]
use super::*;

/// Total[list] - Sum of all elements in a list
/// Total[list, n] - Sum across levels 1 through n
/// Total[list, {n}] - Sum at exactly level n
pub fn total_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Total expects 1 or 2 arguments".into(),
    ));
  }

  // Parse level spec from second argument
  let level_spec = if args.len() == 2 {
    match &args[1] {
      // Total[list, {n}] - exact level n
      Expr::List(items) if items.len() == 1 => {
        if let Some(n) = expr_to_num(&items[0]) {
          TotalLevelSpec::Exact(n as usize)
        } else {
          return Ok(Expr::FunctionCall {
            name: "Total".to_string(),
            args: args.to_vec(),
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
            args: args.to_vec(),
          });
        }
      }
    }
  } else {
    TotalLevelSpec::Through(1)
  };

  match &args[0] {
    Expr::List(_) => total_with_level(&args[0], &level_spec),
    // Total[x] for non-list returns x
    other => Ok(other.clone()),
  }
}

pub enum TotalLevelSpec {
  Through(usize), // sum levels 1..=n
  Exact(usize),   // sum at exactly level n
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
pub fn add_exprs_recursive(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  match (a, b) {
    (Expr::List(la), Expr::List(lb)) if la.len() == lb.len() => {
      let results: Result<Vec<Expr>, _> = la
        .iter()
        .zip(lb.iter())
        .map(|(x, y)| add_exprs_recursive(x, y))
        .collect();
      Ok(Expr::List(results?))
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
        Ok(Expr::List(processed))
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
        return Err(InterpreterError::EvaluationError(
          "Mean: empty list".into(),
        ));
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
            args: vec![Expr::Integer(num), Expr::Integer(denom)],
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
              args: args.to_vec(),
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
    _ => Ok(Expr::FunctionCall {
      name: "Mean".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Mean of columns in a list-of-lists (matrix)
pub fn mean_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&Vec<Expr>> = rows
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
      args: vec![Expr::List(rows.to_vec())],
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
    let mean_result = mean_ast(&[Expr::List(col_items)])?;
    col_means.push(mean_result);
  }
  let _ = nrows; // used indirectly through mean_ast
  Ok(Expr::List(col_means))
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
        return Err(InterpreterError::EvaluationError(
          "Variance: need at least 2 elements".into(),
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
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Variance".to_string(),
      args: args.to_vec(),
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
  let mean = mean_ast(&[Expr::List(items.to_vec())])?;
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
          args: vec![Expr::Integer(-1), mean.clone()],
        },
      ],
    };
    // Abs[xi - mean]^2
    let abs_sq = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![diff],
        },
        Expr::Integer(2),
      ],
    };
    sum_sq_terms.push(abs_sq);
  }
  let sum_sq = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: sum_sq_terms,
  };
  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      sum_sq,
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![Expr::Integer((n - 1) as i128), Expr::Integer(-1)],
      },
    ],
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Variance of columns in a list-of-lists (matrix)
pub fn variance_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&Vec<Expr>> = rows
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
      args: vec![Expr::List(rows.to_vec())],
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
    let var_result = variance_ast(&[Expr::List(col_items)])?;
    col_vars.push(var_result);
  }
  Ok(Expr::List(col_vars))
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
  // For list-of-lists, the variance returns a list of column variances
  let var = variance_ast(args)?;
  match &var {
    Expr::List(items) => {
      // Apply Sqrt to each element
      let mut results = Vec::new();
      for item in items {
        results.push(sqrt_ast(&[item.clone()])?);
      }
      Ok(Expr::List(results))
    }
    Expr::Integer(_) | Expr::Real(_) | Expr::FunctionCall { .. } => {
      sqrt_ast(&[var.clone()])
    }
    _ => Ok(Expr::FunctionCall {
      name: "StandardDeviation".to_string(),
      args: args.to_vec(),
    }),
  }
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
      // Float path
      let mut vals = Vec::new();
      for item in items {
        if let Some(v) = try_eval_to_f64(item) {
          vals.push(v);
        } else {
          return Ok(Expr::FunctionCall {
            name: "GeometricMean".to_string(),
            args: args.to_vec(),
          });
        }
      }
      let n = vals.len() as f64;
      let product: f64 = vals.iter().product();
      let result = product.powf(1.0 / n);
      // Check if result is an integer
      Ok(num_to_expr(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "GeometricMean".to_string(),
      args: args.to_vec(),
    }),
  }
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
      if has_real || !all_int {
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
              args: args.to_vec(),
            });
          }
        }
        let n = vals.len() as f64;
        let sum_recip: f64 = vals.iter().map(|x| 1.0 / x).sum();
        return Ok(num_to_expr(n / sum_recip));
      }
      Ok(Expr::FunctionCall {
        name: "HarmonicMean".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "HarmonicMean".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Covariance[list1, list2] - Sample covariance of two numeric lists
pub fn covariance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Covariance".to_string(),
      args: args.to_vec(),
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
        args: args.to_vec(),
      });
    }
  };
  let mut x_vals = Vec::new();
  let mut y_vals = Vec::new();
  for (x, y) in xs.iter().zip(ys.iter()) {
    match (expr_to_num(x), expr_to_num(y)) {
      (Some(xv), Some(yv)) => {
        x_vals.push(xv);
        y_vals.push(yv);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Covariance".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  let n = x_vals.len() as f64;
  let mean_x = x_vals.iter().sum::<f64>() / n;
  let mean_y = y_vals.iter().sum::<f64>() / n;
  let cov: f64 = x_vals
    .iter()
    .zip(y_vals.iter())
    .map(|(x, y)| (x - mean_x) * (y - mean_y))
    .sum::<f64>()
    / (n - 1.0);
  Ok(num_to_expr(cov))
}

/// Correlation[list1, list2] - Pearson correlation coefficient
pub fn correlation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Correlation".to_string(),
      args: args.to_vec(),
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
        name: "Correlation".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let mut x_vals = Vec::new();
  let mut y_vals = Vec::new();
  for (x, y) in xs.iter().zip(ys.iter()) {
    match (expr_to_num(x), expr_to_num(y)) {
      (Some(xv), Some(yv)) => {
        x_vals.push(xv);
        y_vals.push(yv);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Correlation".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  let n = x_vals.len() as f64;
  let mean_x = x_vals.iter().sum::<f64>() / n;
  let mean_y = y_vals.iter().sum::<f64>() / n;
  let cov: f64 = x_vals
    .iter()
    .zip(y_vals.iter())
    .map(|(x, y)| (x - mean_x) * (y - mean_y))
    .sum::<f64>();
  let var_x: f64 = x_vals.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
  let var_y: f64 = y_vals.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
  let denom = (var_x * var_y).sqrt();
  if denom == 0.0 {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  Ok(num_to_expr(cov / denom))
}

/// CentralMoment[list, r] - r-th central moment of a numeric list
pub fn central_moment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "CentralMoment".to_string(),
      args: args.to_vec(),
    });
  }
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let r = match expr_to_num(&args[1]) {
    Some(r) => r as i32,
    None => {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let mut vals = Vec::new();
  for item in items {
    if let Some(v) = expr_to_num(item) {
      vals.push(v);
    } else {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec(),
      });
    }
  }
  let n = vals.len() as f64;
  let mean = vals.iter().sum::<f64>() / n;
  let moment: f64 = vals.iter().map(|x| (x - mean).powi(r)).sum::<f64>() / n;
  Ok(num_to_expr(moment))
}

/// Kurtosis[list] - CentralMoment[list, 4] / CentralMoment[list, 2]^2
pub fn kurtosis_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Kurtosis".to_string(),
      args: args.to_vec(),
    });
  }
  let m4 = central_moment_ast(&[args[0].clone(), Expr::Integer(4)])?;
  let m2 = central_moment_ast(&[args[0].clone(), Expr::Integer(2)])?;
  match (expr_to_num(&m4), expr_to_num(&m2)) {
    (Some(m4v), Some(m2v)) if m2v != 0.0 => Ok(num_to_expr(m4v / (m2v * m2v))),
    _ => Ok(Expr::FunctionCall {
      name: "Kurtosis".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Skewness[list] - CentralMoment[list, 3] / CentralMoment[list, 2]^(3/2)
pub fn skewness_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Skewness".to_string(),
      args: args.to_vec(),
    });
  }
  let m3 = central_moment_ast(&[args[0].clone(), Expr::Integer(3)])?;
  let m2 = central_moment_ast(&[args[0].clone(), Expr::Integer(2)])?;
  match (expr_to_num(&m3), expr_to_num(&m2)) {
    (Some(m3v), Some(m2v)) if m2v != 0.0 => {
      Ok(num_to_expr(m3v / m2v.powf(1.5)))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Skewness".to_string(),
      args: args.to_vec(),
    }),
  }
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
          return Ok(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::Integer(numer)],
          });
        }
        // Return Sqrt[Rational[numer, denom]]
        return Ok(Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![make_rational(numer, denom)],
        });
      }
      if has_real || !all_int {
        let mut vals = Vec::new();
        for item in items {
          if let Some(v) = expr_to_num(item) {
            vals.push(v);
          } else {
            return Ok(Expr::FunctionCall {
              name: "RootMeanSquare".to_string(),
              args: args.to_vec(),
            });
          }
        }
        let n = vals.len() as f64;
        let mean_sq = vals.iter().map(|x| x * x).sum::<f64>() / n;
        return Ok(num_to_expr(mean_sq.sqrt()));
      }
      Ok(Expr::FunctionCall {
        name: "RootMeanSquare".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "RootMeanSquare".to_string(),
      args: args.to_vec(),
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
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Quantile".to_string(),
        args: args.to_vec(),
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
    return Ok(Expr::List(results?));
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
          ],
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Quantile".to_string(),
        args: vec![
          Expr::List(sorted.iter().cloned().cloned().collect()),
          q.clone(),
        ],
      });
    }
  };

  let idx = (q_val * n as f64).ceil() as usize;
  let idx = idx.max(1).min(n);
  Ok(sorted[idx - 1].clone())
}

// ─── PowerExpand ──────────────────────────────────────────────────────

