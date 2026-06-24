#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// If the first argument is a numeric scalar, emit
/// `<F>::rectt: Rectangular array expected at position 1 in <call>.`,
/// matching wolframscript for statistics functions (Mean, Median, Variance,
/// StandardDeviation, …) applied to a number. Non-numeric atoms (symbols,
/// strings, Infinity, True) and expressions stay unevaluated with no message.
pub fn emit_rectt_if_numeric(name: &str, args: &[Expr]) {
  if let Some(first) = args.first()
    && crate::functions::predicate_ast::is_numeric_q_pub(first)
  {
    crate::emit_message(&format!(
      "{}::rectt: Rectangular array expected at position 1 in {}.",
      name,
      crate::syntax::format_expr(
        &Expr::FunctionCall {
          name: name.to_string(),
          args: args.to_vec().into(),
        },
        crate::syntax::ExprForm::Output
      )
    ));
  }
}

/// Dimensions of a rectangular nested-list array, or `None` if the list is
/// ragged or of mixed depth. Scalars have empty dimensions.
fn array_dimensions(e: &Expr) -> Option<Vec<usize>> {
  match e {
    Expr::List(items) => {
      if items.is_empty() {
        return Some(vec![0]);
      }
      let first = array_dimensions(&items[0])?;
      for it in items.iter().skip(1) {
        if array_dimensions(it)? != first {
          return None;
        }
      }
      let mut dims = Vec::with_capacity(first.len() + 1);
      dims.push(items.len());
      dims.extend(first);
      Some(dims)
    }
    _ => Some(Vec::new()),
  }
}

/// Whether `e` is a rectangular array — a list whose elements all share the
/// same dimensions (recursively). Scalars are trivially rectangular.
pub fn is_rectangular_array(e: &Expr) -> bool {
  array_dimensions(e).is_some()
}

/// If `args[0]` is a non-rectangular list (ragged rows or mixed scalar/list
/// depth), emit `<F>::rectt: Rectangular array expected at position 1 in
/// <call>.` and return the unevaluated call. Returns `None` for a valid
/// rectangular array or a non-list argument.
pub fn rectt_if_ragged(name: &str, args: &[Expr]) -> Option<Expr> {
  if matches!(args.first(), Some(Expr::List(_)))
    && !is_rectangular_array(&args[0])
  {
    crate::emit_message(&format!(
      "{}::rectt: Rectangular array expected at position 1 in {}.",
      name,
      crate::syntax::format_expr(
        &Expr::FunctionCall {
          name: name.to_string(),
          args: args.to_vec().into(),
        },
        crate::syntax::ExprForm::Output
      )
    ));
    Some(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    })
  } else {
    None
  }
}

/// Whether every leaf of a (possibly nested) list is a real number — used by
/// Median, which requires a rectangular array of real numbers (symbols and
/// complex values are rejected, but Pi, Sin[1], 1/2, 1.5 are accepted).
fn all_leaves_real_numeric(e: &Expr) -> bool {
  match e {
    Expr::List(items) => items.iter().all(all_leaves_real_numeric),
    // A Quantity with a real magnitude is acceptable: Median sorts a list of
    // compatible quantities by magnitude.
    Expr::FunctionCall { name, args }
      if name == "Quantity" && args.len() == 2 =>
    {
      all_leaves_real_numeric(&args[0])
    }
    _ => {
      crate::functions::predicate_ast::is_numeric_q_pub(e)
        && !crate::functions::predicate_ast::is_complex_number(e)
    }
  }
}

/// If `args[0]` is a list that is not a rectangular array of real numbers,
/// emit `<F>::rectn: A rectangular array of real numbers is expected at
/// position 1 in <call>.` and return the unevaluated call. Used by Median,
/// which is stricter than Mean (it rejects ragged arrays and symbolic /
/// complex entries). Returns `None` for a non-list argument.
pub fn rectn_if_not_real_rectangular(
  name: &str,
  args: &[Expr],
) -> Option<Expr> {
  if matches!(args.first(), Some(Expr::List(_)))
    && !(is_rectangular_array(&args[0]) && all_leaves_real_numeric(&args[0]))
  {
    crate::emit_message(&format!(
      "{}::rectn: A rectangular array of real numbers is expected at position 1 in {}.",
      name,
      crate::syntax::format_expr(
        &Expr::FunctionCall {
          name: name.to_string(),
          args: args.to_vec().into(),
        },
        crate::syntax::ExprForm::Output
      )
    ));
    Some(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    })
  } else {
    None
  }
}

/// Total[list] - Sum of all elements in a list
/// Total[list, n] - Sum across levels 1 through n
/// Total[list, {n}] - Sum at exactly level n
pub fn total_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Total expects 1 or 2 arguments".into(),
    ));
  }

  // A SparseArray argument is summed over its dense form.
  if let Expr::FunctionCall { name, args: sa } = &args[0]
    && name == "SparseArray"
  {
    let dense = crate::functions::list_helpers_ast::sparse_array_ast(sa)?;
    // Guard against re-entry if normalization left it as a SparseArray.
    if !matches!(&dense, Expr::FunctionCall { name, .. } if name == "SparseArray")
    {
      let mut new_args = args.to_vec();
      new_args[0] = dense;
      return total_ast(&new_args);
    }
  }

  // The number of nested list levels of `e` — the maximum valid Total level.
  // A flat list {1, 2, 3} has depth 1; {{1, 2}, {3, 4}} has depth 2.
  fn list_total_depth(e: &Expr) -> usize {
    match e {
      Expr::List(items) => {
        1 + items.iter().map(list_total_depth).max().unwrap_or(0)
      }
      Expr::Association(pairs) => {
        1 + pairs
          .iter()
          .map(|(_, v)| list_total_depth(v))
          .max()
          .unwrap_or(0)
      }
      _ => 0,
    }
  }

  // Resolve a level component to a usize, accepting `Infinity` and negative
  // levels (counted from the deepest level: -1 is the last level, matching
  // wolframscript — Total[{{1,2},{3,4}}, {-1}] sums the innermost lists).
  let depth = list_total_depth(&args[0]) as i64;
  let resolve_level = |e: &Expr| -> Option<usize> {
    if matches!(e, Expr::Identifier(s) if s == "Infinity") {
      return Some(usize::MAX);
    }
    let n = expr_to_num(e)? as i64;
    let r = if n >= 0 { n } else { depth + n + 1 };
    if r < 0 { None } else { Some(r as usize) }
  };

  // Parse level spec from second argument
  let level_spec = if args.len() == 2 {
    match &args[1] {
      // Total[list, {n}] - exact level n
      Expr::List(items) if items.len() == 1 => {
        if let Some(n) = resolve_level(&items[0]) {
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
          (resolve_level(&items[0]), resolve_level(&items[1]))
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
        if let Some(n) = resolve_level(&args[1]) {
          TotalLevelSpec::Through(n)
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

  // Adding rows of unequal length surfaces as an internal error from the
  // list-threading helper; turn it into Total::tllen (matching wolframscript)
  // and leave the call unevaluated instead of leaking the error.
  let tllen = |result: Result<Expr, InterpreterError>| match result {
    Err(InterpreterError::EvaluationError(ref m))
      if m == "Lists must have the same length" =>
    {
      crate::emit_message(&format!(
        "Total::tllen: Lists of unequal length in {} cannot be added.",
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
      Ok(Expr::FunctionCall {
        name: "Total".to_string(),
        args: args.to_vec().into(),
      })
    }
    other => other,
  };

  // An empty top-level list totals to 0 (the additive identity) at every
  // positive level. Only a pure level-0 spec leaves it as the untouched {}.
  // (Nested empty lists are handled structurally by total_with_level — e.g.
  // Total[{{}}, {3}] stays {{}} — so this guard is restricted to the top.)
  if let Expr::List(items) = &args[0]
    && items.is_empty()
  {
    let untouched = matches!(
      level_spec,
      TotalLevelSpec::Exact(0) | TotalLevelSpec::Through(0)
    );
    return Ok(if untouched {
      args[0].clone()
    } else {
      Expr::Integer(0)
    });
  }

  match &args[0] {
    Expr::List(_) => tllen(total_with_level(&args[0], &level_spec)),
    Expr::Association(pairs) => {
      let values: Vec<Expr> = pairs.iter().map(|(_, v)| v.clone()).collect();
      tllen(total_with_level(&Expr::List(values.into()), &level_spec))
    }
    // Non-list, non-association argument. A numeric scalar (e.g. 5, Pi,
    // Sin[1], 1 + I) is its own total and is returned as-is — even with a
    // level spec. A String can never become a list, so Total emits
    // ::normal and stays unevaluated. Anything else (symbols, Infinity,
    // True, x + y, f[…], rules) stays unevaluated with no message, matching
    // wolframscript.
    other => {
      if crate::functions::predicate_ast::is_numeric_q_pub(other) {
        Ok(other.clone())
      } else {
        if matches!(other, Expr::String(_)) {
          crate::emit_message(&format!(
            "Total::normal: Nonatomic expression expected at position 1 in {}.",
            crate::syntax::format_expr(
              &Expr::FunctionCall {
                name: "Total".to_string(),
                args: args.to_vec().into(),
              },
              crate::syntax::ExprForm::Output
            )
          ));
        }
        Ok(Expr::FunctionCall {
          name: "Total".to_string(),
          args: args.to_vec().into(),
        })
      }
    }
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
    // Two lists of unequal length can't be added: raise the raw error so the
    // caller (Total) reports Total::tllen. Going through plus_ast here would
    // instead emit Thread::tdlen, which is wrong for Total.
    (Expr::List(_), Expr::List(_)) => Err(InterpreterError::EvaluationError(
      "Lists must have the same length".into(),
    )),
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
  if n == 0 {
    // Level 0 is the whole expression itself; nothing is summed.
    Ok(expr.clone())
  } else if n == 1 {
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
  // A ragged / mixed-depth list is not a rectangular array: emit ::rectt
  // and stay unevaluated rather than producing a bogus column-wise result.
  if let Some(uneval) = rectt_if_ragged("Mean", args) {
    return Ok(uneval);
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
        // Always keep a Real result (machine precision) even when the
        // mean is a whole number, matching wolframscript (e.g. 2.).
        Ok(Expr::Real(sum / items.len() as f64))
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
        // A rational sum folds to an exact rational (e.g. Mean[{1/4, 1/8}]
        // is 3/16, not the unevaluated quotient (3/8)/2)
        if let Some((num, den)) = expr_to_rational(&evaluated_sum)
          && let Some(full_den) = den.checked_mul(n)
        {
          return Ok(make_rational(num, full_den));
        }
        // Represent the mean as (sum) / n and evaluate it: a symbolic sum
        // like (a + b)/2 stays as-is (Plus does not distribute over Times),
        // while a Quantity sum collapses (Quantity[6, Meters]/2 →
        // Quantity[3, Meters]), matching wolframscript.
        crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
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
        | "ExpGammaDistribution"
        | "PoissonDistribution"
        | "BernoulliDistribution"
        | "SkellamDistribution"
        | "PolyaAeppliDistribution"
        | "BinomialDistribution"
        | "StableDistribution"
        | "GammaDistribution"
        | "ErlangDistribution"
        | "BetaDistribution"
        | "KumaraswamyDistribution"
        | "StudentTDistribution"
        | "LogNormalDistribution"
        | "ChiDistribution"
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
        | "FRatioDistribution"
        | "NoncentralFRatioDistribution"
        | "UniformSumDistribution"
        | "BetaBinomialDistribution"
        | "BetaNegativeBinomialDistribution"
        | "BetaPrimeDistribution"
        | "NoncentralChiSquareDistribution"
        | "ExponentialPowerDistribution"
        | "RiceDistribution"
        | "MinStableDistribution"
        | "MaxStableDistribution"
        | "TriangularDistribution"
        | "MaxwellDistribution"
        | "WignerSemicircleDistribution"
        | "SechDistribution"
        | "BorelTannerDistribution"
        | "BenktanderGibratDistribution"
        | "GumbelDistribution"
        | "ZipfDistribution"
        // Mean only: the symbolic Variance form factors as
        // (1 - ns/nt)*(-n + nt), but Woxi's Times canonicalization orders
        // those two factors the other way, diverging from wolframscript.
        | "HypergeometricDistribution"
    ) =>
    {
      // Invalid parameters (e.g. BenktanderGibratDistribution[1, 2]) emit a
      // message and leave the call unevaluated, matching wolframscript;
      // they must not surface as an evaluation error.
      match super::distributions::distribution_mean_variance_pub(
        dist_name, dargs,
      ) {
        Ok((mean, _)) => crate::evaluator::evaluate_expr_to_expr(&mean),
        Err(_) => Ok(Expr::FunctionCall {
          name: "Mean".to_string(),
          args: args.to_vec().into(),
        }),
      }
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
    } if dist_name == "BinormalDistribution" => {
      // Mean[BinormalDistribution[{m1, m2}, …]] = {m1, m2}.
      match super::distributions::binormal_params(dargs) {
        Some((m1, m2, ..)) => Ok(Expr::List(vec![m1, m2].into())),
        None => Ok(Expr::FunctionCall {
          name: "Mean".to_string(),
          args: args.to_vec().into(),
        }),
      }
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
    _ => {
      emit_rectt_if_numeric("Mean", args);
      Ok(Expr::FunctionCall {
        name: "Mean".to_string(),
        args: args.to_vec().into(),
      })
    }
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
  if let Some(uneval) = rectt_if_ragged("Variance", args) {
    return Ok(uneval);
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
        | "ExpGammaDistribution"
        | "PoissonDistribution"
        | "BernoulliDistribution"
        | "SkellamDistribution"
        | "PolyaAeppliDistribution"
        | "BinomialDistribution"
        | "StableDistribution"
        | "GammaDistribution"
        | "ErlangDistribution"
        | "BetaDistribution"
        | "KumaraswamyDistribution"
        | "HypergeometricDistribution"
        | "StudentTDistribution"
        | "LogNormalDistribution"
        | "ChiDistribution"
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
        | "FRatioDistribution"
        | "NoncentralFRatioDistribution"
        | "UniformSumDistribution"
        | "BetaBinomialDistribution"
        | "BetaNegativeBinomialDistribution"
        | "BetaPrimeDistribution"
        | "NoncentralChiSquareDistribution"
        | "ExponentialPowerDistribution"
        | "RiceDistribution"
        | "MinStableDistribution"
        | "MaxStableDistribution"
        | "TriangularDistribution"
        | "MaxwellDistribution"
        | "WignerSemicircleDistribution"
        | "SechDistribution"
        | "BorelTannerDistribution"
        | "BenktanderGibratDistribution"
        | "GumbelDistribution"
        | "ZipfDistribution"
    ) =>
    {
      // Invalid parameters emit a message and leave the call unevaluated
      // (matching wolframscript), never an evaluation error.
      match super::distributions::distribution_mean_variance_pub(
        dist_name, dargs,
      ) {
        Ok((_, variance)) => crate::evaluator::evaluate_expr_to_expr(&variance),
        Err(_) => Ok(Expr::FunctionCall {
          name: "Variance".to_string(),
          args: args.to_vec().into(),
        }),
      }
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
    } if dist_name == "BinormalDistribution" => {
      // Variance[BinormalDistribution[…, {s1, s2}, …]] = {s1^2, s2^2}.
      match super::distributions::binormal_params(dargs) {
        Some((_, _, s1, s2, _)) => {
          let sq = |s: Expr| Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(s),
            right: Box::new(Expr::Integer(2)),
          };
          crate::evaluator::evaluate_expr_to_expr(&Expr::List(
            vec![sq(s1), sq(s2)].into(),
          ))
        }
        None => Ok(Expr::FunctionCall {
          name: "Variance".to_string(),
          args: args.to_vec().into(),
        }),
      }
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
    _ => {
      emit_rectt_if_numeric("Variance", args);
      Ok(Expr::FunctionCall {
        name: "Variance".to_string(),
        args: args.to_vec().into(),
      })
    }
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
  if let Some(uneval) = rectt_if_ragged("StandardDeviation", args) {
    return Ok(uneval);
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
  // A numeric scalar isn't a rectangular array: emit StandardDeviation::rectt
  // directly and stay unevaluated, rather than delegating to Variance below
  // (which would emit Variance::rectt instead).
  if crate::functions::predicate_ast::is_numeric_q_pub(&args[0]) {
    emit_rectt_if_numeric("StandardDeviation", args);
    return Ok(Expr::FunctionCall {
      name: "StandardDeviation".to_string(),
      args: args.to_vec().into(),
    });
  }
  // StandardDeviation[BinormalDistribution[…, {s1, s2}, …]] = {s1, s2}. Return
  // the sigmas directly rather than Sqrt[s1^2], which Woxi cannot reduce for a
  // symbolic sigma of unknown sign.
  if let Expr::FunctionCall { name, args: dargs } = &args[0]
    && name == "BinormalDistribution"
    && let Some((_, _, s1, s2, _)) =
      super::distributions::binormal_params(dargs)
  {
    return Ok(Expr::List(vec![s1, s2].into()));
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
    | Expr::Identifier(_)
    | Expr::FunctionCall { .. }
    | Expr::BinaryOp { .. } => {
      let is_dist = is_distribution_arg(&args[0]);
      // A Piecewise variance (StudentT/Pareto/FRatio, ...) takes the square
      // root branch-by-branch: Sqrt[Piecewise[{{v, c}}, d]] threads into
      // Piecewise[{{Sqrt[v], c}}, Sqrt[d]], matching wolframscript.
      if let Some(threaded) = thread_sqrt_into_piecewise(&var, is_dist)? {
        return Ok(threaded);
      }
      // For distribution arguments, extract even negative power factors from
      // the variance. Distribution parameters are always positive, so
      // Sqrt[a * p^(-2)] = Sqrt[a] / p (no Abs needed).
      // E.g. Variance = n*(1-p)/p^2 → SD = Sqrt[n*(1-p)] / p
      if is_dist && let Some(result) = try_sqrt_extract_denom_factors(&var)? {
        // Evaluate so a residual radical coefficient folds, e.g.
        // (b-a)*1/(2 Sqrt[6]) -> (b-a)/(2 Sqrt[6]).
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
      sqrt_ast(&[var.clone()])
    }
    _ => {
      emit_rectt_if_numeric("StandardDeviation", args);
      Ok(Expr::FunctionCall {
        name: "StandardDeviation".to_string(),
        args: args.to_vec().into(),
      })
    }
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
      | "SkellamDistribution"
      | "PolyaAeppliDistribution"
      | "GammaDistribution"
      | "ErlangDistribution"
      | "BetaDistribution"
      | "KumaraswamyDistribution"
      | "StudentTDistribution"
      | "LogNormalDistribution"
      | "ChiDistribution"
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
      | "UniformSumDistribution"
      | "BetaBinomialDistribution"
      | "BetaNegativeBinomialDistribution"
      | "BetaPrimeDistribution"
      | "NoncentralChiSquareDistribution"
      | "ExponentialPowerDistribution"
      | "RiceDistribution"
      | "MinStableDistribution"
      | "MaxStableDistribution"
      | "TriangularDistribution"
      | "MaxwellDistribution"
      | "WignerSemicircleDistribution"
      | "SechDistribution"
      | "BorelTannerDistribution"
      | "BenktanderGibratDistribution"
      | "GumbelDistribution"
      | "ZipfDistribution"
      | "BinomialDistribution"
      | "JohnsonDistribution"
  ))
}

/// Try to extract even power factors from a product and split the Sqrt.
/// Positive even powers come out of the radical into the numerator, negative
/// even powers into the denominator. Because this is only used for distribution
/// variances (whose parameters are positive), `Sqrt[x^2] = x` with no `Abs`.
/// E.g. `Sqrt[n*(1-p)*p^(-2)] → Sqrt[n*(1-p)]/p` and `Sqrt[sigma^2] → sigma`.
/// Returns None if there are no extractable factors.
/// Square root of a single (distribution) value: extract even-power factors
/// when `is_dist`, otherwise plain Sqrt.
fn sqrt_of_value(v: &Expr, is_dist: bool) -> Result<Expr, InterpreterError> {
  if is_dist && let Some(r) = try_sqrt_extract_denom_factors(v)? {
    return crate::evaluator::evaluate_expr_to_expr(&r);
  }
  sqrt_ast(&[v.clone()])
}

/// If `var` is a `Piecewise[{{v1, c1}, ...}, default]`, return the StandardDeviation
/// as `Piecewise[{{Sqrt[v1], c1}, ...}, Sqrt[default]]` (square root threaded
/// into every branch). Returns None for any other shape.
fn thread_sqrt_into_piecewise(
  var: &Expr,
  is_dist: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let Expr::FunctionCall { name, args } = var else {
    return Ok(None);
  };
  if name != "Piecewise" || args.is_empty() {
    return Ok(None);
  }
  let Expr::List(pairs) = &args[0] else {
    return Ok(None);
  };
  let mut new_pairs: Vec<Expr> = Vec::with_capacity(pairs.len());
  for pair in pairs.iter() {
    let Expr::List(vc) = pair else {
      return Ok(None);
    };
    if vc.len() != 2 {
      return Ok(None);
    }
    let new_val = sqrt_of_value(&vc[0], is_dist)?;
    new_pairs.push(Expr::List(vec![new_val, vc[1].clone()].into()));
  }
  let default = if args.len() >= 2 {
    args[1].clone()
  } else {
    Expr::Integer(0)
  };
  let new_default = sqrt_of_value(&default, is_dist)?;
  let result = Expr::FunctionCall {
    name: "Piecewise".to_string(),
    args: vec![Expr::List(new_pairs.into()), new_default].into(),
  };
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?))
}

fn try_sqrt_extract_denom_factors(
  var: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::functions::polynomial_ast::{
    build_product, collect_multiplicative_factors,
  };
  use crate::syntax::BinaryOperator;

  // Build `base^half`, collapsing the exponent to a bare base when half == 1.
  let power_or_base = |base: Expr, half: i128| -> Expr {
    if half == 1 {
      base
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(Expr::Integer(half)),
      }
    }
  };

  let factors = collect_multiplicative_factors(var);
  // Factors that stay under the radical.
  let mut numerator_factors: Vec<Expr> = Vec::new();
  // Even powers pulled out of the radical, by sign of the original exponent.
  let mut pulled_numerator_factors: Vec<Expr> = Vec::new();
  let mut denominator_factors: Vec<Expr> = Vec::new();

  for f in &factors {
    // Normalize a factor to (base, exponent) when it is an integer power.
    let as_power: Option<(&Expr, i128)> = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } => match exp.as_ref() {
        Expr::Integer(n) => Some((base.as_ref(), *n)),
        _ => None,
      },
      Expr::FunctionCall {
        name: pname,
        args: pargs,
      } if pname == "Power" && pargs.len() == 2 => match &pargs[1] {
        Expr::Integer(n) => Some((&pargs[0], *n)),
        _ => None,
      },
      _ => None,
    };

    match as_power {
      // x^(2n), n > 0: pull x^n out into the numerator.
      Some((base, n)) if n > 0 && n % 2 == 0 => {
        pulled_numerator_factors.push(power_or_base(base.clone(), n / 2));
      }
      // x^(-2n), n > 0: pull x^n out into the denominator.
      Some((base, n)) if n < 0 && n % 2 == 0 => {
        denominator_factors.push(power_or_base(base.clone(), -n / 2));
      }
      _ => numerator_factors.push(f.clone()),
    }
  }

  if denominator_factors.is_empty() && pulled_numerator_factors.is_empty() {
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

  // Combine the pulled-out numerator factors with the residual radical.
  let numerator = if pulled_numerator_factors.is_empty() {
    sqrt_num
  } else {
    let pulled = if pulled_numerator_factors.len() == 1 {
      pulled_numerator_factors.remove(0)
    } else {
      build_product(pulled_numerator_factors)
    };
    // Drop a trivial Sqrt[1] = 1 factor.
    if matches!(sqrt_num, Expr::Integer(1)) {
      pulled
    } else {
      build_product(vec![pulled, sqrt_num])
    }
  };

  if denominator_factors.is_empty() {
    return Ok(Some(numerator));
  }

  let denom_expr = if denominator_factors.len() == 1 {
    denominator_factors.remove(0)
  } else {
    build_product(denominator_factors)
  };

  Ok(Some(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(numerator),
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
      // An empty list stays unevaluated (matching wolframscript) rather than
      // raising an error.
      if items.is_empty() {
        return Ok(Expr::FunctionCall {
          name: "GeometricMean".to_string(),
          args: args.to_vec().into(),
        });
      }
      // List-of-lists (matrix) → column-wise geometric mean.
      if items.iter().all(|item| matches!(item, Expr::List(_))) {
        return geometric_mean_columnwise(items);
      }
      let n = items.len() as i128;
      let has_real = items
        .iter()
        .any(|i| matches!(i, Expr::Real(_) | Expr::BigFloat(_, _)));
      // Exact path: if all elements are exact (integers or rationals),
      // compute product symbolically and return Power[product, 1/n]
      let product = super::times_ast(items);
      if let Ok(ref prod) = product
        && !has_real
      {
        let exponent = super::make_rational(1, n);
        return super::power_two(prod, &exponent);
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
      // A Real element makes the result inexact even when it is a whole number
      // (GeometricMean[{4., 9}] = 6., not 6); num_to_expr would collapse it.
      Ok(if has_real {
        Expr::Real(result)
      } else {
        num_to_expr(result)
      })
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
      // An empty list stays unevaluated (matching wolframscript).
      if items.is_empty() {
        return Ok(Expr::FunctionCall {
          name: "HarmonicMean".to_string(),
          args: args.to_vec().into(),
        });
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

/// ContraharmonicMean[list] / ContraharmonicMean[list, p] — the Lehmer mean
/// `Total[list^p] / Total[list^(p-1)]` (default `p = 2`). With the default it
/// reduces to the usual contraharmonic mean `Total[list^2] / Total[list]`. The
/// element-wise Power and the columnwise Total handle nested lists, exact
/// rationals, reals, and symbolic entries uniformly.
pub fn contraharmonic_mean_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ContraharmonicMean".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }
  // The first argument must be a non-empty list.
  match &args[0] {
    Expr::List(items) if !items.is_empty() => {}
    _ => return unevaluated(),
  }
  let list = &args[0];
  let p = if args.len() == 2 {
    // The exponent must be a scalar, not a list.
    if matches!(&args[1], Expr::List(_)) {
      crate::emit_message(&format!(
        "ContraharmonicMean::scalar: Argument {} at position 2 is not a scalar.",
        crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output)
      ));
      return unevaluated();
    }
    args[1].clone()
  } else {
    Expr::Integer(2)
  };

  use crate::evaluator::evaluate_function_call_ast;
  let p_minus_1 =
    evaluate_function_call_ast("Subtract", &[p.clone(), Expr::Integer(1)])?;
  let numerator = evaluate_function_call_ast(
    "Total",
    &[evaluate_function_call_ast("Power", &[list.clone(), p])?],
  )?;
  let denominator = evaluate_function_call_ast(
    "Total",
    &[evaluate_function_call_ast(
      "Power",
      &[list.clone(), p_minus_1],
    )?],
  )?;
  evaluate_function_call_ast("Divide", &[numerator, denominator])
}

/// AbsoluteCorrelation[v1, v2] = Mean[v1 * Conjugate[v2]] — the uncentered
/// (second-moment) correlation of two equal-length vectors. The single-argument
/// form is AbsoluteCorrelation[v, v]. Mismatched lengths emit
/// AbsoluteCorrelation::vctmat. (The matrix form is left unevaluated.)
pub fn absolute_correlation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::evaluate_function_call_ast;
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "AbsoluteCorrelation".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }
  // Single-argument form: AbsoluteCorrelation[v] == AbsoluteCorrelation[v, v].
  if args.len() == 1 {
    return absolute_correlation_ast(&[args[0].clone(), args[0].clone()]);
  }

  let (Expr::List(xs), Expr::List(ys)) = (&args[0], &args[1]) else {
    return unevaluated();
  };

  // Matrix form: with two n×p / n×q matrices (rows are observations), the
  // result is the p×q matrix whose (i, j) entry is the mean over observations
  // of column_i times Conjugate[column_j], i.e. (Transpose[m1] . Conjugate[m2])
  // / n. AbsoluteCorrelation[m] uses m2 = m. Reuse Transpose/Dot/Conjugate so
  // the scalar division threads over the matrix.
  let xs_is_matrix =
    !xs.is_empty() && xs.iter().all(|e| matches!(e, Expr::List(_)));
  let ys_is_matrix =
    !ys.is_empty() && ys.iter().all(|e| matches!(e, Expr::List(_)));
  if xs_is_matrix && ys_is_matrix {
    if xs.len() != ys.len() {
      crate::emit_message(
        "AbsoluteCorrelation::vctmat: The arguments to AbsoluteCorrelation are not a pair of vectors or a pair of matrices of equal length.",
      );
      return unevaluated();
    }
    let n = xs.len();
    let transpose_m1 =
      evaluate_function_call_ast("Transpose", &[args[0].clone()])?;
    let conj_m2 = evaluate_function_call_ast("Conjugate", &[args[1].clone()])?;
    let dot = evaluate_function_call_ast("Dot", &[transpose_m1, conj_m2])?;
    // Divide each entry directly by n rather than letting `matrix/n`
    // canonicalize to Times[matrix, 1/n]: the latter multiplies floats by the
    // rounded reciprocal and double-rounds, so a float entry would diverge from
    // wolframscript in the last ULP. A per-entry Divide matches it exactly.
    let n_expr = Expr::Integer(n as i128);
    if let Expr::List(rows) = &dot {
      let mut out_rows = Vec::with_capacity(rows.len());
      for row in rows.iter() {
        if let Expr::List(cells) = row {
          let mut out_cells = Vec::with_capacity(cells.len());
          for c in cells.iter() {
            out_cells.push(evaluate_function_call_ast(
              "Divide",
              &[c.clone(), n_expr.clone()],
            )?);
          }
          out_rows.push(Expr::List(out_cells.into()));
        } else {
          out_rows.push(evaluate_function_call_ast(
            "Divide",
            &[row.clone(), n_expr.clone()],
          )?);
        }
      }
      return Ok(Expr::List(out_rows.into()));
    }
    return evaluate_function_call_ast("Divide", &[dot, n_expr]);
  }
  // A pure vector pair has no nested lists at all; anything else (one matrix
  // and one vector, or ragged data) is an invalid argument pair.
  if xs.iter().any(|e| matches!(e, Expr::List(_)))
    || ys.iter().any(|e| matches!(e, Expr::List(_)))
  {
    crate::emit_message(
      "AbsoluteCorrelation::vctmat: The arguments to AbsoluteCorrelation are not a pair of vectors or a pair of matrices of equal length.",
    );
    return unevaluated();
  }
  if xs.is_empty() || xs.len() != ys.len() {
    crate::emit_message(
      "AbsoluteCorrelation::vctmat: The arguments to AbsoluteCorrelation are not a pair of vectors or a pair of matrices of equal length.",
    );
    return unevaluated();
  }

  // Σ x_i * Conjugate[y_i], divided by the length.
  let n = xs.len();
  let mut terms = Vec::with_capacity(n);
  for (x, y) in xs.iter().zip(ys.iter()) {
    let conj_y = evaluate_function_call_ast("Conjugate", &[y.clone()])?;
    terms.push(evaluate_function_call_ast("Times", &[x.clone(), conj_y])?);
  }
  let total = evaluate_function_call_ast("Plus", &terms)?;
  evaluate_function_call_ast("Divide", &[total, Expr::Integer(n as i128)])
}

/// Sample covariance of two equal-length scalar lists, computed symbolically
/// as `Sum[(x_i - meanx)*Conjugate[y_i - meany]] / (n - 1)`. The Conjugate on
/// the second argument matches Wolfram's (Hermitian) convention; for real data
/// it evaluates away. Works for both numeric and symbolic entries.
fn covariance_pair(xs: &[Expr], ys: &[Expr]) -> Result<Expr, InterpreterError> {
  let mean_x = mean_ast(&[Expr::List(xs.to_vec().into())])?;
  let mean_y = mean_ast(&[Expr::List(ys.to_vec().into())])?;

  let n = xs.len();
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
    let conj_dy = Expr::FunctionCall {
      name: "Conjugate".to_string(),
      args: vec![dy].into(),
    };
    let product = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(dx),
      right: Box::new(conj_dy),
    };
    let val = crate::evaluator::evaluate_expr_to_expr(&product)?;
    terms.push(val);
  }

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

/// True when every entry of the list is a numeric scalar.
fn all_numeric_scalars(items: &[Expr]) -> bool {
  items.iter().all(|e| expr_to_num(e).is_some())
}

/// Symbolic covariance of two equal-length vectors, in wolframscript's
/// canonical form. The mean of `ys` drops out because the `xs`-deviations sum
/// to zero, so the result is `sum_i (n*x_i - sum x) * Conjugate[y_i]`
/// over `n*(n-1)`. For `n == 2` wolframscript factors this as
/// `((x0 - x1)*(Conjugate[y0] - Conjugate[y1]))/2`, so build that shape
/// directly; for `n >= 3` the expanded sum already matches.
fn symbolic_covariance(
  xs: &[Expr],
  ys: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator::{Divide, Minus, Times};
  let n = xs.len();
  let conj = |e: &Expr| Expr::FunctionCall {
    name: "Conjugate".to_string(),
    args: vec![e.clone()].into(),
  };
  let result = if n == 2 {
    let dx = Expr::BinaryOp {
      op: Minus,
      left: Box::new(xs[0].clone()),
      right: Box::new(xs[1].clone()),
    };
    let dy = Expr::BinaryOp {
      op: Minus,
      left: Box::new(conj(&ys[0])),
      right: Box::new(conj(&ys[1])),
    };
    Expr::BinaryOp {
      op: Divide,
      left: Box::new(Expr::BinaryOp {
        op: Times,
        left: Box::new(dx),
        right: Box::new(dy),
      }),
      right: Box::new(Expr::Integer(2)),
    }
  } else {
    let sum_x = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: xs.to_vec().into(),
    };
    let mut terms = Vec::with_capacity(n);
    for (x, y) in xs.iter().zip(ys.iter()) {
      let coeff = Expr::BinaryOp {
        op: Minus,
        left: Box::new(Expr::BinaryOp {
          op: Times,
          left: Box::new(Expr::Integer(n as i128)),
          right: Box::new(x.clone()),
        }),
        right: Box::new(sum_x.clone()),
      };
      terms.push(Expr::BinaryOp {
        op: Times,
        left: Box::new(coeff),
        right: Box::new(conj(y)),
      });
    }
    Expr::BinaryOp {
      op: Divide,
      left: Box::new(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      }),
      right: Box::new(Expr::Integer((n * (n - 1)) as i128)),
    }
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Covariance of two equal-length vectors: an exact numeric result when both
/// are numeric, otherwise the symbolic closed form.
fn covariance_two(xs: &[Expr], ys: &[Expr]) -> Result<Expr, InterpreterError> {
  if all_numeric_scalars(xs) && all_numeric_scalars(ys) {
    covariance_pair(xs, ys)
  } else {
    symbolic_covariance(xs, ys)
  }
}

/// Covariance[list1, list2] - sample covariance of two equal-length lists.
/// Covariance[matrix] - covariance matrix of the columns of an n×p matrix.
pub fn covariance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "Covariance".to_string(),
      args: args.to_vec().into(),
    })
  };

  // Covariance[BinormalDistribution[…, {s1, s2}, rho]] is the 2×2 covariance
  // matrix {{s1^2, rho s1 s2}, {rho s1 s2, s2^2}}.
  if args.len() == 1
    && let Expr::FunctionCall { name, args: dargs } = &args[0]
    && name == "BinormalDistribution"
    && let Some((_, _, s1, s2, rho)) =
      super::distributions::binormal_params(dargs)
  {
    use crate::syntax::BinaryOperator::{Power, Times};
    let sq = |s: Expr| Expr::BinaryOp {
      op: Power,
      left: Box::new(s),
      right: Box::new(Expr::Integer(2)),
    };
    let off = Expr::BinaryOp {
      op: Times,
      left: Box::new(rho),
      right: Box::new(Expr::BinaryOp {
        op: Times,
        left: Box::new(s1.clone()),
        right: Box::new(s2.clone()),
      }),
    };
    let matrix = Expr::List(
      vec![
        Expr::List(vec![sq(s1.clone()), off.clone()].into()),
        Expr::List(vec![off, sq(s2)].into()),
      ]
      .into(),
    );
    return crate::evaluator::evaluate_expr_to_expr(&matrix);
  }

  // Single-argument matrix form: covariance matrix of the columns.
  if args.len() == 1 {
    let Expr::List(rows) = &args[0] else {
      return unevaluated();
    };
    if rows.len() < 2 {
      return unevaluated();
    }
    // A flat vector is one variable, so its covariance is its variance
    // (covariance of the variable with itself).
    if rows.iter().all(|r| !matches!(r, Expr::List(_))) {
      return covariance_two(rows, rows);
    }
    let Some(cols) = transpose_rows(rows) else {
      return unevaluated();
    };
    // Only the numeric covariance matrix is closed-formed here. The symbolic
    // matrix form is left unevaluated: its lower-triangle entries multiply a
    // plain difference by a Conjugate-difference, and Woxi's Times ordering of
    // those two factors diverges from wolframscript's (e.g. it prints
    // (Conjugate[a] - Conjugate[c])*(b - d) where WL keeps (b - d) first).
    if !cols.iter().all(|c| all_numeric_scalars(c)) {
      return unevaluated();
    }
    // Build the symmetric p×p covariance matrix.
    let mut matrix_rows = Vec::with_capacity(cols.len());
    for ci in &cols {
      let mut row = Vec::with_capacity(cols.len());
      for cj in &cols {
        row.push(covariance_pair(ci, cj)?);
      }
      matrix_rows.push(Expr::List(row.into()));
    }
    return Ok(Expr::List(matrix_rows.into()));
  }

  if args.len() != 2 {
    return unevaluated();
  }

  // Two-matrix cross-covariance: with two n×p / n×q data matrices (rows are
  // observations) the result is the p×q matrix whose (i, j) entry is the sample
  // covariance of column i of m1 with column j of m2.
  if let (Expr::List(m1), Expr::List(m2)) = (&args[0], &args[1])
    && m1.len() == m2.len()
    && m1.len() >= 2
    && m1.iter().all(|r| matches!(r, Expr::List(_)))
    && m2.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let (Some(cols1), Some(cols2)) = (transpose_rows(m1), transpose_rows(m2))
    else {
      return unevaluated();
    };
    // Only the numeric form is closed-formed here; the symbolic form has the
    // same Times-ordering divergence noted for the single-matrix case.
    if !cols1.iter().all(|c| all_numeric_scalars(c))
      || !cols2.iter().all(|c| all_numeric_scalars(c))
    {
      return unevaluated();
    }
    let mut matrix_rows = Vec::with_capacity(cols1.len());
    for ci in &cols1 {
      let mut row = Vec::with_capacity(cols2.len());
      for cj in &cols2 {
        row.push(covariance_pair(ci, cj)?);
      }
      matrix_rows.push(Expr::List(row.into()));
    }
    return Ok(Expr::List(matrix_rows.into()));
  }

  let (xs, ys) = match (&args[0], &args[1]) {
    (Expr::List(xs), Expr::List(ys))
      if xs.len() == ys.len()
        && xs.len() >= 2
        && xs.iter().all(|e| !matches!(e, Expr::List(_)))
        && ys.iter().all(|e| !matches!(e, Expr::List(_))) =>
    {
      (xs, ys)
    }
    _ => return unevaluated(),
  };
  covariance_two(xs, ys)
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
/// Average ranks (1-based) of a numeric list. Tied values share the mean of
/// the positions they occupy, so ranks stay exact (Integer or Rational).
/// Returns `None` if any element is non-numeric.
fn average_ranks(items: &[Expr]) -> Option<Vec<Expr>> {
  let n = items.len();
  let vals: Vec<f64> = items
    .iter()
    .map(crate::functions::math_ast::try_eval_to_f64)
    .collect::<Option<_>>()?;
  let mut order: Vec<usize> = (0..n).collect();
  order.sort_by(|&a, &b| {
    vals[a]
      .partial_cmp(&vals[b])
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  let mut ranks = vec![Expr::Integer(0); n];
  let mut i = 0;
  while i < n {
    let mut j = i;
    while j + 1 < n && vals[order[j + 1]] == vals[order[i]] {
      j += 1;
    }
    // Positions i+1 ..= j+1 (1-based); shared rank = mean of positions.
    let count = (j - i + 1) as i128;
    let sum_pos: i128 = ((i as i128 + 1)..=(j as i128 + 1)).sum();
    let rank = crate::functions::math_ast::make_rational(sum_pos, count);
    for &idx in &order[i..=j] {
      ranks[idx] = rank.clone();
    }
    i = j + 1;
  }
  Some(ranks)
}

/// SpearmanRho[v1, v2] — Spearman rank-correlation coefficient: the Pearson
/// correlation of the average-rank vectors of the two equal-length numeric
/// lists. Mismatched lengths or non-numeric vectors emit the `::rctneqln`
/// message and stay unevaluated.
pub fn spearman_rho_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "SpearmanRho".to_string(),
      args: args.to_vec().into(),
    })
  };
  let rctneqln = || {
    crate::emit_message(
      "SpearmanRho::rctneqln: The arguments to SpearmanRho are not a pair of vectors or matrices of equal length.",
    );
    unevaluated()
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let (x, y) = match (&args[0], &args[1]) {
    (Expr::List(x), Expr::List(y)) if x.len() == y.len() && x.len() >= 2 => {
      (x, y)
    }
    _ => return rctneqln(),
  };
  let (Some(rx), Some(ry)) = (average_ranks(x), average_ranks(y)) else {
    return rctneqln();
  };
  crate::evaluator::evaluate_function_call_ast(
    "Correlation",
    &[Expr::List(rx.into()), Expr::List(ry.into())],
  )
}

/// KendallTau[v1, v2] — Kendall's rank-correlation coefficient τ_b for two
/// equal-length numeric vectors. Defined as
///   τ_b = (C - D) / Sqrt[(n0 - n1)(n0 - n2)]
/// where C and D are the counts of concordant and discordant pairs, n0 =
/// n(n-1)/2 is the total number of pairs, and n1, n2 are the tie corrections
/// Σ t(t-1)/2 over the groups of equal values in v1 and v2 respectively.
/// Exact (integer/rational) inputs yield an exact result; machine-real inputs
/// yield a machine-real result. Constant data (zero variance in either vector)
/// emits `KendallTau::zrvr` and returns Indeterminate.
pub fn kendall_tau_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "KendallTau".to_string(),
      args: args.to_vec().into(),
    })
  };
  let rctneqln = || {
    crate::emit_message(
      "KendallTau::rctneqln: The arguments to KendallTau are not a pair of vectors or matrices of equal length.",
    );
    unevaluated()
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let (x, y) = match (&args[0], &args[1]) {
    (Expr::List(x), Expr::List(y)) if x.len() == y.len() && x.len() >= 2 => {
      (x, y)
    }
    _ => return rctneqln(),
  };
  let (Some(xf), Some(yf)) = (
    x.iter()
      .map(crate::functions::math_ast::try_eval_to_f64)
      .collect::<Option<Vec<f64>>>(),
    y.iter()
      .map(crate::functions::math_ast::try_eval_to_f64)
      .collect::<Option<Vec<f64>>>(),
  ) else {
    return rctneqln();
  };
  let n = x.len();
  // Numerator: Σ_{i<j} sign(x_j - x_i) · sign(y_j - y_i) = C - D.
  let mut num: i128 = 0;
  for i in 0..n {
    for j in (i + 1)..n {
      let sx = (xf[j] - xf[i]).partial_cmp(&0.0);
      let sy = (yf[j] - yf[i]).partial_cmp(&0.0);
      if let (Some(sx), Some(sy)) = (sx, sy) {
        use std::cmp::Ordering::*;
        let sxi = match sx {
          Greater => 1i128,
          Less => -1,
          Equal => 0,
        };
        let syi = match sy {
          Greater => 1i128,
          Less => -1,
          Equal => 0,
        };
        num += sxi * syi;
      }
    }
  }
  // Tie correction Σ t(t-1)/2 over groups of equal values.
  let tie_sum = |vals: &[f64]| -> i128 {
    let mut sorted = vals.to_vec();
    sorted
      .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut total: i128 = 0;
    let mut i = 0;
    while i < sorted.len() {
      let mut j = i;
      while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
        j += 1;
      }
      let t = (j - i + 1) as i128;
      total += t * (t - 1) / 2;
      i = j + 1;
    }
    total
  };
  let n0 = (n as i128) * (n as i128 - 1) / 2;
  let n1 = tie_sum(&xf);
  let n2 = tie_sum(&yf);
  let denom_sq = (n0 - n1) * (n0 - n2);
  if denom_sq == 0 {
    crate::emit_message(
      "KendallTau::zrvr: The input data has zero variance. The statistic cannot be computed.",
    );
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Machine-real inputs give a machine-real result; otherwise stay exact.
  let is_real = x
    .iter()
    .chain(y.iter())
    .any(|e| matches!(e, Expr::Real(_) | Expr::BigFloat(_, _)));
  if is_real {
    return Ok(Expr::Real(num as f64 / (denom_sq as f64).sqrt()));
  }
  let expr = Expr::FunctionCall {
    name: "Divide".to_string(),
    args: vec![
      Expr::Integer(num),
      Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(denom_sq)].into(),
      },
    ]
    .into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

pub fn correlation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "Correlation".to_string(),
      args: args.to_vec().into(),
    })
  };
  // Single-argument form.
  if args.len() == 1 {
    if let Expr::List(items) = &args[0]
      && items.len() >= 2
    {
      // A data matrix → the p×p correlation matrix of its columns, which
      // equals Correlation[m, m].
      if items.iter().all(|r| matches!(r, Expr::List(_))) {
        return correlation_ast(&[args[0].clone(), args[0].clone()]);
      }
      // A flat vector is one variable, so it correlates perfectly with itself.
      if items.iter().all(|r| !matches!(r, Expr::List(_))) {
        return Ok(Expr::Integer(1));
      }
    }
    return unevaluated();
  }
  if args.len() != 2 {
    return unevaluated();
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

/// A distribution is any expression whose head ends in "Distribution".
fn as_distribution(expr: &Expr) -> Option<&Expr> {
  match expr {
    Expr::FunctionCall { name, .. } if name.ends_with("Distribution") => {
      Some(expr)
    }
    _ => None,
  }
}

/// Whether the symbol `sym` occurs anywhere in `expr`.
fn expr_has_symbol(expr: &Expr, sym: &str) -> bool {
  match expr {
    Expr::Identifier(s) => s == sym,
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| expr_has_symbol(a, sym))
    }
    Expr::List(items) => items.iter().any(|a| expr_has_symbol(a, sym)),
    Expr::BinaryOp { left, right, .. } => {
      expr_has_symbol(left, sym) || expr_has_symbol(right, sym)
    }
    Expr::UnaryOp { operand, .. } => expr_has_symbol(operand, sym),
    _ => false,
  }
}

/// An integration-variable name not occurring in the distribution.
fn fresh_moment_var(dist: &Expr) -> String {
  for c in ["x", "y", "z", "u", "w"] {
    if !expr_has_symbol(dist, c) {
      return c.to_string();
    }
  }
  "xMomentVar".to_string()
}

/// Raw moment E[x^n] of a distribution via Expectation. Returns None when the
/// Expectation cannot be evaluated in closed form.
fn distribution_raw_moment(
  dist: &Expr,
  n: i128,
  var: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let powered = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(Expr::Identifier(var.to_string())),
    right: Box::new(Expr::Integer(n)),
  };
  let distributed = Expr::FunctionCall {
    name: "Distributed".to_string(),
    args: vec![Expr::Identifier(var.to_string()), dist.clone()].into(),
  };
  let exp = Expr::FunctionCall {
    name: "Expectation".to_string(),
    args: vec![powered, distributed].into(),
  };
  let result = crate::evaluator::evaluate_expr_to_expr(&exp)?;
  if matches!(&result, Expr::FunctionCall { name, .. } if name == "Expectation")
  {
    Ok(None)
  } else {
    Ok(Some(result))
  }
}

/// Moment[dist, n] / CentralMoment[dist, n] for a known distribution.
/// `central` selects between the raw moment E[x^n] and the central moment
/// E[(x - mean)^n]; the latter is assembled from raw moments by the binomial
/// theorem so the result stays exact. Returns None when unsupported.
/// The two parameters of `Head[a, b]`, or None.
fn two_params_of(dist: &Expr, head: &str) -> Option<(Expr, Expr)> {
  if let Expr::FunctionCall { name, args } = dist
    && name == head
    && args.len() == 2
  {
    Some((args[0].clone(), args[1].clone()))
  } else {
    None
  }
}

/// Parameters `(a, b)` of a `SkellamDistribution[a, b]`, or None.
fn skellam_params(dist: &Expr) -> Option<(Expr, Expr)> {
  if let Expr::FunctionCall { name, args } = dist
    && name == "SkellamDistribution"
    && args.len() == 2
  {
    Some((args[0].clone(), args[1].clone()))
  } else {
    None
  }
}

fn fact_i128(n: i128) -> i128 {
  (1..=n).product::<i128>().max(1)
}

/// Integer partitions of `n` into parts >= 2 (each part non-increasing).
fn partitions_min2(n: usize) -> Vec<Vec<usize>> {
  fn rec(
    n: usize,
    max: usize,
    cur: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
  ) {
    if n == 0 {
      out.push(cur.clone());
      return;
    }
    for part in (2..=max.min(n)).rev() {
      cur.push(part);
      rec(n - part, part, cur, out);
      cur.pop();
    }
  }
  let mut out = Vec::new();
  rec(n, n, &mut Vec::new(), &mut out);
  out
}

/// n-th central moment of `SkellamDistribution[a, b]` from its cumulants
/// `κ_j = a + (-1)^j b`: `μ_n = Σ_{λ ⊢ n, parts≥2} coef(λ) · Π_i κ_{λ_i}`,
/// where `coef(λ) = n!/(Π_i λ_i! · Π_v m_v!)` (the number of set partitions of
/// [n] with block sizes λ). Matches wolframscript's compact form.
fn skellam_central_moment(a: &Expr, b: &Expr, n: i128) -> Expr {
  if n == 0 {
    return Expr::Integer(1);
  }
  let kappa = |j: usize| -> Expr {
    if j.is_multiple_of(2) {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(a.clone()),
        right: Box::new(b.clone()),
      }
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(a.clone()),
        right: Box::new(b.clone()),
      }
    }
  };
  let mut terms: Vec<Expr> = Vec::new();
  for lambda in partitions_min2(n as usize) {
    // coef = n! / (Π part! · Π mult!)
    let mut denom = 1i128;
    for &p in &lambda {
      denom *= fact_i128(p as i128);
    }
    let mut counts: std::collections::HashMap<usize, i128> =
      std::collections::HashMap::new();
    for &p in &lambda {
      *counts.entry(p).or_insert(0) += 1;
    }
    for &m in counts.values() {
      denom *= fact_i128(m);
    }
    let coef = fact_i128(n) / denom;
    let mut factors = vec![Expr::Integer(coef)];
    for &p in &lambda {
      factors.push(kappa(p));
    }
    terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: factors.into(),
    });
  }
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  }
}

fn distribution_moment(
  dist: &Expr,
  n_expr: &Expr,
  central: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let n = match expr_to_num(n_expr) {
    Some(v) if v >= 0.0 && v.fract() == 0.0 => v as i128,
    _ => return Ok(None),
  };
  let var = fresh_moment_var(dist);

  if !central {
    return distribution_raw_moment(dist, n, &var);
  }

  // Skellam central moments have a clean cumulant-based closed form for all n
  // (the generic Expectation-of-PDF path only closes for small n).
  if let Some((a, b)) = skellam_params(dist) {
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(
      &skellam_central_moment(&a, &b, n),
    )?));
  }

  let mean = mean_ast(&[dist.clone()])?;
  // CentralMoment = Sum_{k=0}^n Binomial[n, k] (-mean)^(n-k) E[x^k]
  let mut terms = Vec::with_capacity((n + 1) as usize);
  for k in 0..=n {
    let raw = match distribution_raw_moment(dist, k, &var)? {
      Some(r) => r,
      None => return Ok(None),
    };
    let binom = Expr::Integer(binomial_i128(n, k));
    // (-mean)^(n-k); guard the n==k case to avoid 0^0 when mean == 0.
    let neg_mean_pow = if n - k == 0 {
      Expr::Integer(1)
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), mean.clone()].into(),
        }),
        right: Box::new(Expr::Integer(n - k)),
      }
    };
    terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![binom, neg_mean_pow, raw].into(),
    });
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  // Expand so symbolic-parameter results collapse to their reduced form
  // (e.g. the Normal third central moment cancels to 0, Poisson's to m).
  let expanded = Expr::FunctionCall {
    name: "Expand".to_string(),
    args: vec![sum].into(),
  };
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&expanded)?))
}

/// CentralMoment[list, r] - r-th central moment of a numeric list
pub fn central_moment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Distribution form: CentralMoment[dist, n].
  if args.len() == 2
    && let Some(dist) = as_distribution(&args[0])
    && let Some(result) = distribution_moment(dist, &args[1], true)?
  {
    return Ok(result);
  }

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

/// Cumulant[list, r] - the r-th (sample) cumulant of the data in `list`.
///
/// Computed from the raw moments mu'_j = (1/n) Sum[x_i^j] via the standard
/// moment-cumulant recursion:
///     k_n = mu'_n - Sum_{m=1}^{n-1} Binomial[n-1, m-1] k_m mu'_{n-m}
/// All arithmetic is carried out symbolically so exact rational results are
/// preserved.
pub fn cumulant_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let symbolic = || {
    Ok(Expr::FunctionCall {
      name: "Cumulant".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    return symbolic();
  }
  let r = match expr_to_num(&args[1]) {
    Some(r) if r >= 0.0 && r.fract() == 0.0 => r as usize,
    _ => return symbolic(),
  };
  if r == 0 {
    return Ok(Expr::Integer(0));
  }

  // Distribution form: Cumulant[dist, r] from the raw moments E[x^j].
  if let Some(dist) = as_distribution(&args[0]) {
    let var = fresh_moment_var(dist);
    let mut mu = Vec::with_capacity(r + 1);
    mu.push(Expr::Integer(0)); // index 0 unused
    for j in 1..=r {
      match distribution_raw_moment(dist, j as i128, &var)? {
        Some(m) => mu.push(m),
        None => return symbolic(),
      }
    }
    let result = cumulant_from_raw_moments(&mu, r)?;
    let expanded = Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![result].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&expanded);
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return symbolic(),
  };

  let n = items.len() as i128;

  // raw_moment(j) = (1/n) Sum[x_i^j], evaluated symbolically/exactly.
  let raw_moment = |j: usize| -> Result<Expr, InterpreterError> {
    let mut terms = Vec::with_capacity(items.len());
    for item in items {
      let powered = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(item.clone()),
        right: Box::new(Expr::Integer(j as i128)),
      };
      terms.push(crate::evaluator::evaluate_expr_to_expr(&powered)?);
    }
    let sum_expr = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    };
    let div = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(sum_expr),
      right: Box::new(Expr::Integer(n)),
    };
    crate::evaluator::evaluate_expr_to_expr(&div)
  };

  // Precompute raw moments mu'_1 .. mu'_r.
  let mut mu = Vec::with_capacity(r + 1);
  mu.push(Expr::Integer(0)); // index 0 unused
  for j in 1..=r {
    mu.push(raw_moment(j)?);
  }

  cumulant_from_raw_moments(&mu, r)
}

/// The r-th cumulant from raw moments mu[1..=r] via the recursion
///     k_n = mu'_n - Sum_{m=1}^{n-1} Binomial[n-1, m-1] k_m mu'_{n-m}
/// (mu[0] is an unused placeholder). All arithmetic stays symbolic.
fn cumulant_from_raw_moments(
  mu: &[Expr],
  r: usize,
) -> Result<Expr, InterpreterError> {
  let mut k: Vec<Expr> = Vec::with_capacity(r + 1);
  k.push(Expr::Integer(0)); // k[0] unused
  for nn in 1..=r {
    let mut acc = mu[nn].clone();
    for m in 1..nn {
      let binom = binomial_i128((nn - 1) as i128, (m - 1) as i128);
      let term = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(binom), k[m].clone(), mu[nn - m].clone()]
          .into(),
      };
      acc = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(acc),
        right: Box::new(term),
      };
      acc = crate::evaluator::evaluate_expr_to_expr(&acc)?;
    }
    k.push(acc);
  }
  Ok(k[r].clone())
}

/// Binomial coefficient C(n, kk) for small non-negative arguments.
fn binomial_i128(n: i128, kk: i128) -> i128 {
  if kk < 0 || kk > n {
    return 0;
  }
  let kk = kk.min(n - kk);
  let mut result: i128 = 1;
  for i in 0..kk {
    result = result * (n - i) / (i + 1);
  }
  result
}

/// Kurtosis[list] - CentralMoment[list, 4] / CentralMoment[list, 2]^2
pub fn kurtosis_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Kurtosis".to_string(),
      args: args.to_vec().into(),
    });
  }
  // Skellam: Kurtosis = 3 + 1/(a+b). Built directly because the generic
  // moment-ratio Expand mangles the multi-parameter denominator.
  if let Some((a, b)) = skellam_params(&args[0]) {
    let a_plus_b = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(a),
      right: Box::new(b),
    };
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(Expr::Integer(3)),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(a_plus_b),
      }),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }
  // PolyaAeppli: Kurtosis = 3 + (1 + 10 p + p^2)/((1 + p) t).
  if let Some((t, p)) = two_params_of(&args[0], "PolyaAeppliDistribution") {
    use crate::syntax::BinaryOperator as B;
    let bin = |op, l, r| Expr::BinaryOp {
      op,
      left: Box::new(l),
      right: Box::new(r),
    };
    let num = bin(
      B::Plus,
      bin(
        B::Plus,
        Expr::Integer(1),
        bin(B::Times, Expr::Integer(10), p.clone()),
      ),
      bin(B::Power, p.clone(), Expr::Integer(2)),
    );
    let den = bin(B::Times, bin(B::Plus, Expr::Integer(1), p), t);
    let result = bin(B::Plus, Expr::Integer(3), bin(B::Divide, num, den));
    return crate::evaluator::evaluate_expr_to_expr(&result);
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
  let result = maybe_expand_for_distribution(&args[0], result);
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// For distribution arguments, distribute the moment-ratio division so that
/// e.g. `(m + 3 m^2)/m^2` reduces to Wolfram's `3 + m^(-1)`. Numeric (list)
/// results are unaffected.
fn maybe_expand_for_distribution(arg: &Expr, result: Expr) -> Expr {
  if as_distribution(arg).is_some() {
    Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![result].into(),
    }
  } else {
    result
  }
}

/// Skewness[list] - CentralMoment[list, 3] / CentralMoment[list, 2]^(3/2)
pub fn skewness_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Skewness".to_string(),
      args: args.to_vec().into(),
    });
  }
  // Skellam: Skewness = (a-b)/(a+b)^(3/2). Built directly so the compact form
  // is preserved (the generic moment-ratio Expand would distribute it).
  if let Some((a, b)) = skellam_params(&args[0]) {
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(a.clone()),
        right: Box::new(b.clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(a),
          right: Box::new(b),
        }),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::Integer(3)),
          right: Box::new(Expr::Integer(2)),
        }),
      }),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }
  // PolyaAeppli: Skewness = (1 + 4 p + p^2)/((1 + p) Sqrt[(1 + p) t]).
  if let Some((t, p)) = two_params_of(&args[0], "PolyaAeppliDistribution") {
    use crate::syntax::BinaryOperator as B;
    let bin = |op, l, r| Expr::BinaryOp {
      op,
      left: Box::new(l),
      right: Box::new(r),
    };
    let one_plus_p = bin(B::Plus, Expr::Integer(1), p.clone());
    let num = bin(
      B::Plus,
      bin(
        B::Plus,
        Expr::Integer(1),
        bin(B::Times, Expr::Integer(4), p.clone()),
      ),
      bin(B::Power, p, Expr::Integer(2)),
    );
    let sqrt = Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![bin(B::Times, one_plus_p.clone(), t)].into(),
    };
    let den = bin(B::Times, one_plus_p, sqrt);
    let result = bin(B::Divide, num, den);
    return crate::evaluator::evaluate_expr_to_expr(&result);
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
  let result = maybe_expand_for_distribution(&args[0], result);
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
      // An empty list stays unevaluated (matching wolframscript).
      if items.is_empty() {
        return Ok(Expr::FunctionCall {
          name: "RootMeanSquare".to_string(),
          args: args.to_vec().into(),
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
          // Evaluate so the radical is reduced (e.g. Sqrt[8] -> 2 Sqrt[2]).
          return crate::evaluator::evaluate_expr_to_expr(&make_sqrt(
            Expr::Integer(numer),
          ));
        }
        // Sqrt[Rational[numer, denom]], evaluated so it reduces to Wolfram's
        // form (e.g. Sqrt[25/2] -> 5/Sqrt[2]).
        return crate::evaluator::evaluate_expr_to_expr(&make_sqrt(
          make_rational(numer, denom),
        ));
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
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Quantile expects 2 or 3 arguments".into(),
    ));
  }
  // ErlangDistribution[k, λ] == GammaDistribution[k, 1/λ]
  if let Expr::FunctionCall { name, args: dargs } = &args[0]
    && name == "ErlangDistribution"
    && dargs.len() == 2
  {
    let gamma = Expr::FunctionCall {
      name: "GammaDistribution".to_string(),
      args: super::distributions::erlang_gamma_dargs(dargs)?.into(),
    };
    let mut new_args = args.to_vec();
    new_args[0] = gamma;
    return quantile_ast(&new_args);
  }
  // Quantile[dist, {p1, p2, ...}] for a distribution head threads over the
  // probability list (the second argument of Quantile is always a scalar
  // probability, so a list there always means "evaluate at each").
  if args.len() == 2
    && matches!(&args[0], Expr::FunctionCall { .. })
    && let Expr::List(ps) = &args[1]
  {
    let results: Result<Vec<Expr>, InterpreterError> = ps
      .iter()
      .map(|p| quantile_ast(&[args[0].clone(), p.clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  // 3-argument parametric form Quantile[list, q, {{a,b},{c,d}}] (Hyndman-Fan).
  // The parameter list must be {{a,b},{c,d}} with exact numeric entries; the
  // 2-argument form is the special case {{0,0},{1,0}}.
  if args.len() == 3 {
    let params = match &args[2] {
      Expr::List(rows)
        if rows.len() == 2
          && matches!(&rows[0], Expr::List(r) if r.len() == 2)
          && matches!(&rows[1], Expr::List(r) if r.len() == 2) =>
      {
        let (Expr::List(r0), Expr::List(r1)) = (&rows[0], &rows[1]) else {
          unreachable!()
        };
        (r0[0].clone(), r0[1].clone(), r1[0].clone(), r1[1].clone())
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Quantile".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    let items = match &args[0] {
      Expr::List(items) if !items.is_empty() => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Quantile".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    let mut sorted: Vec<&Expr> = items.iter().collect();
    sorted.sort_by(|a, b| {
      try_eval_to_f64(a)
        .partial_cmp(&try_eval_to_f64(b))
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (a, b, c, d) = params;
    if let Expr::List(qs) = &args[1] {
      let results: Result<Vec<Expr>, _> = qs
        .iter()
        .map(|q| quantile_parametric(&sorted, q, &a, &b, &c, &d))
        .collect();
      return Ok(Expr::List(results?.into()));
    }
    return quantile_parametric(&sorted, &args[1], &a, &b, &c, &d);
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

/// Hyndman-Fan parametric quantile Quantile[list, q, {{a,b},{c,d}}].
/// With s = Sort[list] and n = Length[list]:
///   x = a + (n + b) q,  k = Floor[x],  frac = x - k
///   frac == 0 → s[[clamp(k)]]
///   else      → s[[clamp(k)]] + (c + d frac) (s[[clamp(k+1)]] - s[[clamp(k)]])
/// where clamp pins indices to [1, n]. All arithmetic runs through the
/// evaluator so exact rationals stay exact and machine reals propagate.
pub fn quantile_parametric(
  sorted: &[&Expr],
  q: &Expr,
  a: &Expr,
  b: &Expr,
  c: &Expr,
  d: &Expr,
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::evaluate_expr_to_expr as ev;
  let plus = |x: Expr, y: Expr| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![x, y].into(),
  };
  let times = |x: Expr, y: Expr| Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![x, y].into(),
  };
  // q must be numeric (Integer, Rational, or Real); otherwise leave symbolic.
  if try_eval_to_f64(q).is_none() {
    return Ok(Expr::FunctionCall {
      name: "Quantile".to_string(),
      args: vec![
        Expr::List(sorted.iter().cloned().cloned().collect()),
        q.clone(),
        Expr::List(
          vec![
            Expr::List(vec![a.clone(), b.clone()].into()),
            Expr::List(vec![c.clone(), d.clone()].into()),
          ]
          .into(),
        ),
      ]
      .into(),
    });
  }
  let n = sorted.len() as i128;
  // x = a + (n + b) * q
  let nb = ev(&plus(Expr::Integer(n), b.clone()))?;
  let x = ev(&plus(a.clone(), times(nb, q.clone())))?;
  // k = Floor[x]
  let k_expr = ev(&Expr::FunctionCall {
    name: "Floor".to_string(),
    args: vec![x.clone()].into(),
  })?;
  let k = match &k_expr {
    Expr::Integer(v) => *v,
    other => try_eval_to_f64(other)
      .map(|f| f.floor() as i128)
      .unwrap_or(0),
  };
  // frac = x - k
  let frac = ev(&plus(x, Expr::Integer(-k)))?;
  let frac_is_zero = matches!(&frac, Expr::Integer(0))
    || matches!(&frac, Expr::Real(f) if *f == 0.0);
  let clamp = |i: i128| -> usize { i.max(1).min(n) as usize };
  let lo = sorted[clamp(k) - 1].clone();
  if frac_is_zero {
    return Ok(lo);
  }
  let hi = sorted[clamp(k + 1) - 1].clone();
  // w = c + d * frac. When d is exactly zero there is no interpolation, so
  // the weight is just c — and the (inexact) fractional part of x must not
  // leak into the result. wolframscript keeps e.g.
  // Quantile[{1,2,3,4,5}, 0.5, {{0,0},{1,0}}] = 3 (exact), not 3., because
  // the d*frac term drops out entirely rather than producing 0.*0.5 = 0.
  let w = if matches!(d, Expr::Integer(0)) {
    c.clone()
  } else {
    ev(&plus(c.clone(), times(d.clone(), frac)))?
  };
  // result = lo + w * (hi - lo)
  let diff = ev(&plus(hi, times(Expr::Integer(-1), lo.clone())))?;
  ev(&plus(lo, times(w, diff)))
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

  // Distribution form: Moment[dist, n] = E[x^n].
  if let Some(dist) = as_distribution(&args[0])
    && let Some(result) = distribution_moment(dist, &args[1], false)?
  {
    return Ok(result);
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

/// FactorialMoment[data, r] — the r-th factorial moment:
/// Mean of the falling factorials FactorialPower[x_i, r].
/// FactorialMoment[{{x1, y1, ...}, ...}, {r1, r2, ...}] — multivariate:
/// mean of the products of per-coordinate falling factorials.
/// Closed-form factorial moment E[X(X-1)...(X-r+1)] for the standard discrete
/// distributions, in wolframscript's printed form. Returns None for orders or
/// distributions without a clean closed form here.
fn factorial_moment_of_distribution(
  name: &str,
  dargs: &[Expr],
  r: i128,
) -> Option<Expr> {
  let pow = |b: Expr, e: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(e),
  };
  let times = |fs: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: fs.into(),
  };
  let r_factorial: i128 = (1..=r).product::<i128>().max(1);
  match (name, dargs) {
    // Poisson: E[X^(r)] = lambda^r (the defining property).
    ("PoissonDistribution", [lam]) => Some(pow(lam.clone(), Expr::Integer(r))),
    // Bernoulli: X in {0,1}, so X(X-1)... = 0 for r >= 2.
    ("BernoulliDistribution", [p]) => Some(match r {
      0 => Expr::Integer(1),
      1 => p.clone(),
      _ => Expr::Integer(0),
    }),
    // Geometric: r! (1/p - 1)^r, printed by Wolfram as r! (-1 + p^(-1))^r.
    ("GeometricDistribution", [p]) => {
      let base = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![Expr::Integer(-1), pow(p.clone(), Expr::Integer(-1))].into(),
      };
      Some(times(vec![
        Expr::Integer(r_factorial),
        pow(base, Expr::Integer(r)),
      ]))
    }
    // Binomial: the falling factorial n(n-1)...(n-r+1) times p^r. Wolfram
    // prints the falling factorial as the product of (i - n) factors, e.g.
    // r=2 -> -((1 - n)*n*p^2), r=3 -> (1 - n)*(2 - n)*n*p^3.
    ("BinomialDistribution", [n, p]) => {
      if r == 0 {
        return Some(Expr::Integer(1));
      }
      let mut factors: Vec<Expr> = Vec::new();
      // (1 - n)*(2 - n)*...*((r-1) - n)
      for i in 1..r {
        factors.push(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Integer(i),
            times(vec![Expr::Integer(-1), n.clone()]),
          ]
          .into(),
        });
      }
      factors.push(n.clone());
      factors.push(pow(p.clone(), Expr::Integer(r)));
      let product = times(factors);
      // Each (i - n) flips a sign relative to the falling factorial n(n-1)...,
      // so (r-1) such factors contribute (-1)^(r-1).
      Some(if (r - 1).rem_euclid(2) == 1 {
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(product),
        }
      } else {
        product
      })
    }
    _ => None,
  }
}

pub fn factorial_moment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "FactorialMoment".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }

  // Distribution factorial moment E[X(X-1)...(X-r+1)] for a non-negative
  // integer order r.
  if let Expr::FunctionCall {
    name: dist_name,
    args: dargs,
  } = &args[0]
    && let Expr::Integer(r) = &args[1]
    && *r >= 0
    && let Some(result) = factorial_moment_of_distribution(dist_name, dargs, *r)
  {
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return Ok(unevaluated(args)),
  };

  let factorial_power = |x: &Expr, r: &Expr| {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "FactorialPower".to_string(),
      args: vec![x.clone(), r.clone()].into(),
    })
  };

  let mut terms = Vec::with_capacity(items.len());
  match &args[1] {
    // Multivariate order {r1, ..., rm}: data points must be lists of
    // matching length
    Expr::List(orders) => {
      for item in items {
        let coords = match item {
          Expr::List(coords) if coords.len() == orders.len() => coords,
          _ => return Ok(unevaluated(args)),
        };
        let mut factors = Vec::with_capacity(coords.len());
        for (x, r) in coords.iter().zip(orders.iter()) {
          factors.push(factorial_power(x, r)?);
        }
        terms.push(crate::evaluator::evaluate_expr_to_expr(
          &Expr::FunctionCall {
            name: "Times".to_string(),
            args: factors.into(),
          },
        )?);
      }
    }
    r => {
      for x in items {
        terms.push(factorial_power(x, r)?);
      }
    }
  }

  mean_ast(&[Expr::List(terms.into())])
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
      // Symmetric/Cyclic/Alternating groups accept n = 0 (trivial group);
      // DihedralGroup requires a positive degree in wolframscript.
      let min_n = if name == "DihedralGroup" { 1 } else { 0 };
      let n = match &gargs[0] {
        Expr::Integer(n) if *n >= min_n => *n as usize,
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
    {
      // Symmetric/Cyclic/Alternating groups accept n = 0 (trivial group,
      // order 1); DihedralGroup requires a positive degree.
      let min_n = if name == "DihedralGroup" { 1 } else { 0 };
      if let Expr::Integer(n) = &gargs[0]
        && *n >= min_n
      {
        let n = *n;
        return Ok(match name.as_str() {
          "CyclicGroup" => Expr::Integer(n.max(1)),
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
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "DihedralGroup"
    && gargs.len() == 1
    && let Expr::Integer(n) = &gargs[0]
    && *n >= 1
  {
    return Ok(dihedral_group_elements(*n as usize));
  }
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "AlternatingGroup"
    && gargs.len() == 1
    && let Expr::Integer(n) = &gargs[0]
    && *n >= 0
  {
    return Ok(alternating_group_elements(*n as usize));
  }
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "CyclicGroup"
    && gargs.len() == 1
    && let Expr::Integer(n) = &gargs[0]
    && *n >= 0
  {
    return Ok(cyclic_group_elements(*n as usize));
  }
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "SymmetricGroup"
    && gargs.len() == 1
    && let Expr::Integer(n) = &gargs[0]
    && *n >= 0
  {
    return Ok(symmetric_group_elements(*n as usize));
  }
  Ok(unevaluated())
}

/// GroupElements[AlternatingGroup[n]] - all even permutations of {1, ..., n}
/// in canonical cycle notation, ordered lexicographically by their image list
/// (i.e. by PermutationList), matching wolframscript.
fn alternating_group_elements(n: usize) -> Expr {
  // n <= 1: the trivial group, just the identity Cycles[{}].
  if n <= 1 {
    return Expr::List(vec![make_cycles_multi(Vec::new())].into());
  }
  // Enumerate all permutations of {1, ..., n} in lexicographic image order,
  // keeping only the even ones (sign +1).
  let mut image: Vec<i128> = (1..=n as i128).collect();
  let mut elements: Vec<Expr> = Vec::new();
  loop {
    if permutation_is_even(&image) {
      elements.push(images_to_cycles(&image));
    }
    if !next_permutation(&mut image) {
      break;
    }
  }
  Expr::List(elements.into())
}

/// Whether the permutation given as a 1-based image list is even (sign +1).
fn permutation_is_even(image: &[i128]) -> bool {
  let n = image.len();
  let mut visited = vec![false; n];
  let mut transpositions = 0usize;
  for start in 0..n {
    if visited[start] {
      continue;
    }
    let mut len = 0usize;
    let mut cur = start;
    while !visited[cur] {
      visited[cur] = true;
      cur = (image[cur] - 1) as usize;
      len += 1;
    }
    // A cycle of length L contributes (L - 1) transpositions.
    transpositions += len - 1;
  }
  transpositions.is_multiple_of(2)
}

/// In-place next lexicographic permutation of `image`; returns false when the
/// sequence is already the last (descending) permutation.
fn next_permutation(image: &mut [i128]) -> bool {
  let n = image.len();
  if n < 2 {
    return false;
  }
  let mut i = n - 1;
  while i > 0 && image[i - 1] >= image[i] {
    i -= 1;
  }
  if i == 0 {
    return false;
  }
  let mut j = n - 1;
  while image[j] <= image[i - 1] {
    j -= 1;
  }
  image.swap(i - 1, j);
  image[i..].reverse();
  true
}

/// Convert a permutation given as an image list (`image[i-1]` is the image of
/// point `i`, 1-based values) into canonical `Cycles[{...}]` form: each cycle
/// starts at its smallest element, cycles are ordered by that smallest element,
/// and fixed points are dropped.
fn images_to_cycles(image: &[i128]) -> Expr {
  let n = image.len();
  let mut visited = vec![false; n];
  let mut cycles: Vec<Vec<i128>> = Vec::new();
  for start in 0..n {
    if visited[start] {
      continue;
    }
    let mut cycle = Vec::new();
    let mut cur = start;
    loop {
      visited[cur] = true;
      cycle.push((cur + 1) as i128);
      cur = (image[cur] - 1) as usize;
      if cur == start {
        break;
      }
    }
    if cycle.len() >= 2 {
      cycles.push(cycle);
    }
  }
  // Each cycle already starts at its smallest element (we enter cycles at the
  // smallest unvisited point) and cycles are produced in ascending order.
  make_cycles_multi(cycles)
}

/// GroupElements[DihedralGroup[n]] - all 2n symmetries of a regular n-gon as
/// permutations of {1, ..., m}, in canonical cycle notation, ordered exactly
/// as wolframscript does (lexicographically by image list).
fn dihedral_group_elements(n: usize) -> Expr {
  // Point set and base permutations match wolframscript's conventions:
  // - DihedralGroup[1] acts on 2 points: identity and the swap (1 2).
  // - DihedralGroup[2] acts on 4 points: the Klein four-group {e,(12),(34),(12)(34)}.
  // - DihedralGroup[n>=3] acts on n points as the symmetries of a regular n-gon.
  let mut elements: Vec<Vec<i128>> = Vec::new();
  if n == 1 {
    elements.push(vec![1, 2]); // identity
    elements.push(vec![2, 1]); // reflection
  } else if n == 2 {
    // Klein four-group acting on {1,2,3,4}.
    for image in [
      vec![1, 2, 3, 4],
      vec![2, 1, 3, 4],
      vec![1, 2, 4, 3],
      vec![2, 1, 4, 3],
    ] {
      elements.push(image);
    }
  } else {
    // Rotation r: i -> (i mod n) + 1. Reflection s: fixes 1, reverses 2..n.
    let rotate = |image: &[i128]| -> Vec<i128> {
      // Apply r after the given permutation: r(image[i]).
      image.iter().map(|&v| (v % n as i128) + 1).collect()
    };
    // Build identity image.
    let identity: Vec<i128> = (1..=n as i128).collect();
    // Reflection image: s(1)=1, s(i)=n+2-i for i>=2.
    let reflection: Vec<i128> = (1..=n as i128)
      .map(|i| if i == 1 { 1 } else { n as i128 + 2 - i })
      .collect();
    // Rotations r^k and reflections r^k . s for k = 0..n-1.
    let mut cur_rot = identity.clone();
    let mut cur_ref = reflection.clone();
    for _ in 0..n {
      elements.push(cur_rot.clone());
      elements.push(cur_ref.clone());
      cur_rot = rotate(&cur_rot);
      cur_ref = rotate(&cur_ref);
    }
  }
  // Sort elements lexicographically by their image list (wolframscript order).
  elements.sort();
  elements.dedup();
  let cycle_exprs: Vec<Expr> =
    elements.iter().map(|img| images_to_cycles(img)).collect();
  Expr::List(cycle_exprs.into())
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
  // S_0 and S_1 are trivial: no generators.
  if n <= 1 {
    return Expr::List(vec![].into());
  }
  // S_2 is generated by the single transposition (1 2); the n-cycle would
  // coincide with it, so wolframscript returns just {Cycles[{{1, 2}}]}.
  if n == 2 {
    return Expr::List(vec![make_cycles(vec![1, 2])].into());
  }
  let transposition = make_cycles(vec![1, 2]);
  let n_cycle: Vec<i128> = (1..=n as i128).collect();
  let rotation = make_cycles(n_cycle);
  Expr::List(vec![transposition, rotation].into())
}

fn cyclic_group_generators(n: usize) -> Expr {
  // C_0 and C_1 are trivial: no generators.
  if n <= 1 {
    return Expr::List(vec![].into());
  }
  let n_cycle: Vec<i128> = (1..=n as i128).collect();
  Expr::List(vec![make_cycles(n_cycle)].into())
}

fn dihedral_group_generators(n: usize) -> Expr {
  if n <= 1 {
    // DihedralGroup[1] has order 2: a single reflection swapping points 1 and 2.
    return Expr::List(vec![make_cycles(vec![1, 2])].into());
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
  // A_0, A_1 and A_2 are trivial: no generators.
  if n <= 2 {
    return Expr::List(vec![].into());
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

/// CovarianceFunction[data, h] gives the sample autocovariance of a numeric
/// time series at lag `h`:
///   γ(h) = (1/n) · Σ_{t=1}^{n-|h|} (x_t − x̄)(x_{t+|h|} − x̄),  n = Length[data].
/// (Note the 1/n normalization, not 1/(n−h).) The autocovariance is symmetric,
/// so negative lags equal their magnitude. A lag whose magnitude is not less
/// than the series length emits `CovarianceFunction::bdlag` and stays
/// unevaluated. Returns `None` for non-numeric data or a non-integer lag so the
/// caller leaves the call unevaluated.
pub fn covariance_function_data(
  data: &Expr,
  lag: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::List(items) = data else {
    return None;
  };
  let Expr::Integer(h) = lag else {
    return None;
  };
  let h = *h;
  if items.is_empty() || !all_numeric_scalars(items) {
    return None;
  }
  let n = items.len();
  // The lag magnitude must be strictly less than the series length.
  if h.unsigned_abs() >= n as u128 {
    crate::emit_message(&format!(
      "CovarianceFunction::bdlag: The lag specification {h} should be a symbol, an integer with magnitude less than the length of the data or a range specification indicating such integers."
    ));
    return Some(Ok(Expr::FunctionCall {
      name: "CovarianceFunction".to_string(),
      args: vec![data.clone(), lag.clone()].into(),
    }));
  }
  let mean = match mean_ast(&[Expr::List(items.to_vec().into())]) {
    Ok(m) => m,
    Err(e) => return Some(Err(e)),
  };
  let dev = |x: &Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Minus,
    left: Box::new(x.clone()),
    right: Box::new(mean.clone()),
  };
  let h_us = h.unsigned_abs() as usize;
  let mut terms = Vec::new();
  for t in 0..(n - h_us) {
    terms.push(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(dev(&items[t])),
      right: Box::new(dev(&items[t + h_us])),
    });
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(sum),
    right: Box::new(Expr::Integer(n as i128)),
  };
  Some(crate::evaluator::evaluate_expr_to_expr(&result))
}

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
  let numerator =
    cf_times(vec![cf_pow(a.clone(), cf_abs_diff(s, t)), sigma2.clone()]);
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
    vec![Expr::List(vec![case1, case0].into()), Expr::Integer(0)],
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
    cf_pow(a.clone(), cf_plus(vec![Expr::Integer(-1), lag.clone()])),
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

// ─── CharacteristicFunction ──────────────────────────────────────────

/// CharacteristicFunction[dist, t] — E[e^(i t X)] for the supported
/// distribution constructors.
///
/// Templates whose evaluated form round-trips to wolframscript's print
/// are built and evaluated; the NormalDistribution[m, s] and
/// UniformDistribution forms are returned as raw structures because the
/// evaluator's canonical ordering would reshuffle them (e.g.
/// E^(-1/2*(s^2*t^2) + I*m*t) instead of E^(I*m*t - (s^2*t^2)/2)).
pub fn characteristic_function_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "CharacteristicFunction".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let t = args[1].clone();

  // Identifier (not Constant): the output formatter's imaginary-unit
  // special cases match Identifier("I")
  let i_unit = || Expr::Identifier("I".to_string());
  let e_sym = || Expr::Identifier("E".to_string());
  let call = |name: &str, fargs: Vec<Expr>| Expr::FunctionCall {
    name: name.to_string(),
    args: fargs.into(),
  };
  let neg = |e: Expr| Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(e),
  };
  let pow = |b: Expr, e: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(e),
  };
  let div = |n: Expr, d: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(n),
    right: Box::new(d),
  };
  // E^(I*t) and E^(I*c*t)
  let e_it = |factors: Vec<Expr>| {
    let mut f = vec![i_unit()];
    f.extend(factors);
    f.push(t.clone());
    pow(e_sym(), call("Times", f))
  };

  let (dist_name, dargs) = match &args[0] {
    Expr::FunctionCall { name, args } => (name.as_str(), args.as_slice()),
    _ => return Ok(unevaluated(args)),
  };

  // Raw templates only make sense while everything stays symbolic
  // (parameter lists like UniformDistribution's {a, b} count when all
  // their elements are symbols)
  fn is_symbolic_param(e: &Expr) -> bool {
    match e {
      Expr::Identifier(_) => true,
      Expr::List(items) => {
        items.iter().all(|i| matches!(i, Expr::Identifier(_)))
      }
      _ => false,
    }
  }
  let symbolic =
    matches!(&t, Expr::Identifier(_)) && dargs.iter().all(is_symbolic_param);

  let template: Option<(Expr, bool)> = match (dist_name, dargs) {
    // E^(-1/2*t^2)
    ("NormalDistribution", []) => Some((
      pow(
        e_sym(),
        call(
          "Times",
          vec![
            crate::functions::math_ast::make_rational_pub(-1, 2),
            pow(t.clone(), Expr::Integer(2)),
          ],
        ),
      ),
      false,
    )),
    // E^(I*m*t - (s^2*t^2)/2)
    ("NormalDistribution", [m, s]) => Some((
      pow(
        e_sym(),
        call(
          "Plus",
          vec![
            call("Times", vec![i_unit(), m.clone(), t.clone()]),
            neg(div(
              call(
                "Times",
                vec![
                  pow(s.clone(), Expr::Integer(2)),
                  pow(t.clone(), Expr::Integer(2)),
                ],
              ),
              Expr::Integer(2),
            )),
          ],
        ),
      ),
      true,
    )),
    // a/(a - I*t)
    ("ExponentialDistribution", [a]) => Some((
      div(
        a.clone(),
        call(
          "Plus",
          vec![
            a.clone(),
            call("Times", vec![Expr::Integer(-1), i_unit(), t.clone()]),
          ],
        ),
      ),
      false,
    )),
    // E^((-1 + E^(I*t))*m)
    ("PoissonDistribution", [m]) => Some((
      pow(
        e_sym(),
        call(
          "Times",
          vec![
            call("Plus", vec![Expr::Integer(-1), e_it(vec![])]),
            m.clone(),
          ],
        ),
      ),
      false,
    )),
    // 1 - p + E^(I*t)*p
    ("BernoulliDistribution", [p]) => Some((
      call(
        "Plus",
        vec![
          Expr::Integer(1),
          call("Times", vec![Expr::Integer(-1), p.clone()]),
          call("Times", vec![e_it(vec![]), p.clone()]),
        ],
      ),
      false,
    )),
    // (1 - p + E^(I*t)*p)^n
    ("BinomialDistribution", [n, p]) => Some((
      pow(
        call(
          "Plus",
          vec![
            Expr::Integer(1),
            call("Times", vec![Expr::Integer(-1), p.clone()]),
            call("Times", vec![e_it(vec![]), p.clone()]),
          ],
        ),
        n.clone(),
      ),
      false,
    )),
    // p/(1 - E^(I*t)*(1 - p))
    ("GeometricDistribution", [p]) => Some((
      div(
        p.clone(),
        call(
          "Plus",
          vec![
            Expr::Integer(1),
            call(
              "Times",
              vec![
                Expr::Integer(-1),
                e_it(vec![]),
                call(
                  "Plus",
                  vec![
                    Expr::Integer(1),
                    call("Times", vec![Expr::Integer(-1), p.clone()]),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
      false,
    )),
    // (1 - I*b*t)^(-a) — raw: the evaluator's canonical Times order
    // would print b*I*t
    ("GammaDistribution", [a, b]) => Some((
      pow(
        call(
          "Plus",
          vec![
            Expr::Integer(1),
            neg(call("Times", vec![i_unit(), b.clone(), t.clone()])),
          ],
        ),
        neg(a.clone()),
      ),
      true,
    )),
    // (-I*(-1 + E^(I*t)))/t
    ("UniformDistribution", []) => Some((
      div(
        call(
          "Times",
          vec![
            neg(i_unit()),
            call("Plus", vec![Expr::Integer(-1), e_it(vec![])]),
          ],
        ),
        t.clone(),
      ),
      false,
    )),
    // (-I*(-E^(I*a*t) + E^(I*b*t)))/((-a + b)*t)
    ("UniformDistribution", [Expr::List(bounds)]) if bounds.len() == 2 => {
      let (a, b) = (bounds[0].clone(), bounds[1].clone());
      Some((
        div(
          call(
            "Times",
            vec![
              neg(i_unit()),
              call(
                "Plus",
                vec![neg(e_it(vec![a.clone()])), e_it(vec![b.clone()])],
              ),
            ],
          ),
          call("Times", vec![call("Plus", vec![neg(a), b]), t.clone()]),
        ),
        true,
      ))
    }
    _ => None,
  };

  match template {
    // Raw templates print in wolframscript's exact form; evaluating
    // them would re-canonicalize. With numeric arguments fall back to
    // evaluation so e.g. CharacteristicFunction[NormalDistribution[], 0]
    // folds to 1.
    Some((expr, raw)) if raw && symbolic => Ok(expr),
    Some((expr, _)) => crate::evaluator::evaluate_expr_to_expr(&expr),
    None => Ok(unevaluated(args)),
  }
}

// ─── MomentGeneratingFunction ────────────────────────────────────────

/// MomentGeneratingFunction[dist, t] — E[e^(t X)] for the supported
/// distribution constructors. Structurally the CharacteristicFunction with
/// the imaginary unit dropped (MGF(t) = CF(-I t)), but Wolfram canonicalizes
/// the MGF forms differently, so the templates are built to match its print.
pub fn moment_generating_function_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "MomentGeneratingFunction".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let t = args[1].clone();

  let e_sym = || Expr::Identifier("E".to_string());
  let call = |name: &str, fargs: Vec<Expr>| Expr::FunctionCall {
    name: name.to_string(),
    args: fargs.into(),
  };
  let neg = |e: Expr| Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(e),
  };
  let pow = |b: Expr, e: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(e),
  };
  let div = |n: Expr, d: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(n),
    right: Box::new(d),
  };
  // E^t and E^(c*t)
  let e_t = |factors: Vec<Expr>| {
    if factors.is_empty() {
      pow(e_sym(), t.clone())
    } else {
      let mut f = factors;
      f.push(t.clone());
      pow(e_sym(), call("Times", f))
    }
  };
  // 1 - p, written as Plus[1, Times[-1, p]]
  let one_minus = |p: &Expr| {
    call(
      "Plus",
      vec![
        Expr::Integer(1),
        call("Times", vec![Expr::Integer(-1), p.clone()]),
      ],
    )
  };

  let (dist_name, dargs) = match &args[0] {
    Expr::FunctionCall { name, args } => (name.as_str(), args.as_slice()),
    _ => return Ok(unevaluated(args)),
  };

  fn is_symbolic_param(e: &Expr) -> bool {
    match e {
      Expr::Identifier(_) => true,
      Expr::List(items) => {
        items.iter().all(|i| matches!(i, Expr::Identifier(_)))
      }
      _ => false,
    }
  }
  let symbolic =
    matches!(&t, Expr::Identifier(_)) && dargs.iter().all(is_symbolic_param);

  let template: Option<(Expr, bool)> = match (dist_name, dargs) {
    // E^(t^2/2)
    ("NormalDistribution", []) => Some((
      pow(
        e_sym(),
        div(pow(t.clone(), Expr::Integer(2)), Expr::Integer(2)),
      ),
      false,
    )),
    // E^(m*t + (s^2*t^2)/2)
    ("NormalDistribution", [m, s]) => Some((
      pow(
        e_sym(),
        call(
          "Plus",
          vec![
            call("Times", vec![m.clone(), t.clone()]),
            div(
              call(
                "Times",
                vec![
                  pow(s.clone(), Expr::Integer(2)),
                  pow(t.clone(), Expr::Integer(2)),
                ],
              ),
              Expr::Integer(2),
            ),
          ],
        ),
      ),
      true,
    )),
    // a/(a - t)
    ("ExponentialDistribution", [a]) => Some((
      div(
        a.clone(),
        call(
          "Plus",
          vec![a.clone(), call("Times", vec![Expr::Integer(-1), t.clone()])],
        ),
      ),
      false,
    )),
    // E^((-1 + E^t)*m)
    ("PoissonDistribution", [m]) => Some((
      pow(
        e_sym(),
        call(
          "Times",
          vec![
            call("Plus", vec![Expr::Integer(-1), e_t(vec![])]),
            m.clone(),
          ],
        ),
      ),
      true,
    )),
    // 1 - p + E^t*p
    ("BernoulliDistribution", [p]) => Some((
      call(
        "Plus",
        vec![
          Expr::Integer(1),
          call("Times", vec![Expr::Integer(-1), p.clone()]),
          call("Times", vec![e_t(vec![]), p.clone()]),
        ],
      ),
      false,
    )),
    // (1 + (-1 + E^t)*p)^n
    ("BinomialDistribution", [n, p]) => Some((
      pow(
        call(
          "Plus",
          vec![
            Expr::Integer(1),
            call(
              "Times",
              vec![
                call("Plus", vec![Expr::Integer(-1), e_t(vec![])]),
                p.clone(),
              ],
            ),
          ],
        ),
        n.clone(),
      ),
      true,
    )),
    // p/(1 - E^t*(1 - p))
    ("GeometricDistribution", [p]) => Some((
      div(
        p.clone(),
        call(
          "Plus",
          vec![
            Expr::Integer(1),
            call("Times", vec![Expr::Integer(-1), e_t(vec![]), one_minus(p)]),
          ],
        ),
      ),
      true,
    )),
    // (1 - b*t)^(-a)
    ("GammaDistribution", [a, b]) => Some((
      pow(
        call(
          "Plus",
          vec![
            Expr::Integer(1),
            neg(call("Times", vec![b.clone(), t.clone()])),
          ],
        ),
        neg(a.clone()),
      ),
      true,
    )),
    // (-1 + E^t)/t
    ("UniformDistribution", []) => Some((
      div(
        call("Plus", vec![Expr::Integer(-1), e_t(vec![])]),
        t.clone(),
      ),
      false,
    )),
    // (-E^(a*t) + E^(b*t))/((-a + b)*t)
    ("UniformDistribution", [Expr::List(bounds)]) if bounds.len() == 2 => {
      let (a, b) = (bounds[0].clone(), bounds[1].clone());
      Some((
        div(
          call(
            "Plus",
            vec![neg(e_t(vec![a.clone()])), e_t(vec![b.clone()])],
          ),
          call("Times", vec![call("Plus", vec![neg(a), b]), t.clone()]),
        ),
        true,
      ))
    }
    _ => None,
  };

  match template {
    Some((expr, raw)) if raw && symbolic => Ok(expr),
    Some((expr, _)) => crate::evaluator::evaluate_expr_to_expr(&expr),
    None => Ok(unevaluated(args)),
  }
}

// ─── CumulantGeneratingFunction ──────────────────────────────────────

/// View `e` as a power, returning (base, exponent) for either the BinaryOp or
/// FunctionCall spelling of Power.
fn as_power_pair(e: &Expr) -> Option<(&Expr, &Expr)> {
  match e {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => Some((left, right)),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

fn is_e_base(e: &Expr) -> bool {
  matches!(e, Expr::Constant(c) | Expr::Identifier(c) if c == "E")
}

/// CumulantGeneratingFunction[dist, t] = Log[MomentGeneratingFunction[...]],
/// simplified the way Wolfram prints it. For most distributions this is a
/// structural transform of the MGF:
///   E^X        → X
///   base^(-a)  → -(a Log[base])
///   base^exp   → exp Log[base]
///   otherwise  → Log[mgf]
/// Geometric and the two-parameter Uniform are canonicalized differently by
/// Wolfram, so they get explicit templates.
pub fn cumulant_generating_function_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "CumulantGeneratingFunction".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let t = args[1].clone();

  let e_sym = || Expr::Identifier("E".to_string());
  let call = |name: &str, fargs: Vec<Expr>| Expr::FunctionCall {
    name: name.to_string(),
    args: fargs.into(),
  };
  let neg = |e: Expr| Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand: Box::new(e),
  };
  let pow = |b: Expr, e: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(e),
  };
  let div = |n: Expr, d: Expr| Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(n),
    right: Box::new(d),
  };
  let log = |e: Expr| Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![e].into(),
  };

  let (dist_name, dargs) = match &args[0] {
    Expr::FunctionCall { name, args } => (name.as_str(), args.as_slice()),
    _ => return Ok(unevaluated(args)),
  };

  fn is_symbolic_param(e: &Expr) -> bool {
    match e {
      Expr::Identifier(_) => true,
      Expr::List(items) => {
        items.iter().all(|i| matches!(i, Expr::Identifier(_)))
      }
      _ => false,
    }
  }
  let symbolic =
    matches!(&t, Expr::Identifier(_)) && dargs.iter().all(is_symbolic_param);

  // Distributions whose CGF is NOT a clean structural transform of the MGF.
  let special: Option<Expr> = match (dist_name, dargs) {
    // -t - Log[1 - (1 - E^(-t))/p]
    ("GeometricDistribution", [p]) => Some(call(
      "Plus",
      vec![
        neg(t.clone()),
        neg(log(call(
          "Plus",
          vec![
            Expr::Integer(1),
            neg(div(
              call(
                "Plus",
                vec![Expr::Integer(1), neg(pow(e_sym(), neg(t.clone())))],
              ),
              p.clone(),
            )),
          ],
        ))),
      ],
    )),
    // a*t + Log[(-1 + E^((-a + b)*t))/((-a + b)*t)]
    ("UniformDistribution", [Expr::List(bounds)]) if bounds.len() == 2 => {
      let (a, b) = (bounds[0].clone(), bounds[1].clone());
      let span = || call("Plus", vec![neg(a.clone()), b.clone()]);
      let span_t = || call("Times", vec![span(), t.clone()]);
      Some(call(
        "Plus",
        vec![
          call("Times", vec![a.clone(), t.clone()]),
          log(div(
            call("Plus", vec![Expr::Integer(-1), pow(e_sym(), span_t())]),
            span_t(),
          )),
        ],
      ))
    }
    _ => None,
  };
  if let Some(expr) = special {
    return if symbolic {
      Ok(expr)
    } else {
      crate::evaluator::evaluate_expr_to_expr(&expr)
    };
  }

  // General case: derive from the MGF.
  let mgf = moment_generating_function_ast(args)?;
  if matches!(&mgf, Expr::FunctionCall { name, .. }
    if name == "MomentGeneratingFunction")
  {
    return Ok(unevaluated(args));
  }
  let cgf = if let Some((base, exp)) = as_power_pair(&mgf) {
    if is_e_base(base) {
      exp.clone()
    } else if let Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } = exp
    {
      neg(call("Times", vec![(**operand).clone(), log(base.clone())]))
    } else {
      call("Times", vec![exp.clone(), log(base.clone())])
    }
  } else {
    log(mgf)
  };

  if symbolic {
    Ok(cgf)
  } else {
    crate::evaluator::evaluate_expr_to_expr(&cgf)
  }
}

// ─── CorrelationFunction ─────────────────────────────────────────────

/// CorrelationFunction[data, k] — the sample autocorrelation at lag k:
/// Sum[(x_i - mean)(x_{i+|k|} - mean)] / Sum[(x_i - mean)^2].
/// Symmetric in the lag sign; |k| must be smaller than the data length
/// (wolframscript emits CorrelationFunction::bdlag otherwise).
pub fn correlation_function_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "CorrelationFunction".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let data = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return Ok(unevaluated(args)),
  };
  let is_numeric = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::Real(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  if !data.iter().all(is_numeric) {
    return Ok(unevaluated(args));
  }
  let k = match &args[1] {
    Expr::Integer(k) => *k,
    _ => return Ok(unevaluated(args)),
  };
  let n = data.len() as i128;
  if k.abs() >= n {
    crate::emit_message(&format!(
      "CorrelationFunction::bdlag: The lag specification {k} should be a \
       symbol, an integer with magnitude less than the length of the data \
       or a range specification indicating such integers.",
    ));
    return Ok(unevaluated(args));
  }
  let lag = k.unsigned_abs() as usize;

  let mean = mean_ast(&[args[0].clone()])?;
  let dev = |x: &Expr| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      x.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), mean.clone()].into(),
      },
    ]
    .into(),
  };
  let sum = |terms: Vec<Expr>| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    })
  };
  let len = data.len();
  let num_terms: Vec<Expr> = (0..len - lag)
    .map(|i| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![dev(&data[i]), dev(&data[i + lag])].into(),
    })
    .collect();
  let den_terms: Vec<Expr> = data
    .iter()
    .map(|x| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![dev(x), dev(x)].into(),
    })
    .collect();
  let numerator = sum(num_terms)?;
  let denominator = sum(den_terms)?;

  // Constant data: 0/0 — wolframscript returns a bare Indeterminate
  // without the Power::infy/Infinity::indet messages a literal division
  // would emit
  let is_zero = matches!(&denominator, Expr::Integer(0))
    || matches!(&denominator, Expr::Real(v) if *v == 0.0);
  if is_zero {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(numerator),
    right: Box::new(denominator),
  })
}

/// ZTest[data] / ZTest[data, var] / ZTest[data, var, mu0] /
/// ZTest[data, var, mu0, "property"] - two-sided one-sample z-test.
/// The second argument is the KNOWN VARIANCE (Automatic or omitted
/// estimates the sample variance); the p-value is
/// Erfc[|z|/Sqrt[2]] with z = (mean - mu0)/Sqrt[var/n]. Properties:
/// "PValue" (default) and "TestStatistic".
pub fn ztest_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "ZTest".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() || args.len() > 4 {
    return Ok(unevaluated(args));
  }
  let bad_data = |args: &[Expr]| {
    crate::emit_message(&format!(
      "ZTest::rctndm1: The argument {} at position 1 should be a rectangular array of real numbers with length greater than the dimension of the array or two such arrays of equal dimensionality.",
      crate::syntax::expr_to_string(&args[0])
    ));
    Ok(Expr::FunctionCall {
      name: "ZTest".to_string(),
      args: args.to_vec().into(),
    })
  };
  let data: Vec<f64> = match &args[0] {
    Expr::List(items) if items.len() >= 2 => {
      match items
        .iter()
        .map(crate::functions::math_ast::numeric_utils::try_eval_to_f64)
        .collect::<Option<Vec<f64>>>()
      {
        Some(v) => v,
        None => return bad_data(args),
      }
    }
    _ => return bad_data(args),
  };
  let n = data.len() as f64;
  let mean = data.iter().sum::<f64>() / n;
  let variance = match args.get(1) {
    None => None,
    Some(Expr::Identifier(s)) if s == "Automatic" => None,
    Some(e) => {
      match crate::functions::math_ast::numeric_utils::try_eval_to_f64(e) {
        Some(v) if v > 0.0 => Some(v),
        _ => return Ok(unevaluated(args)),
      }
    }
  }
  .unwrap_or_else(|| {
    data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
  });
  let mu0 = match args.get(2) {
    None => 0.0,
    Some(e) => {
      match crate::functions::math_ast::numeric_utils::try_eval_to_f64(e) {
        Some(v) => v,
        None => return Ok(unevaluated(args)),
      }
    }
  };
  let z = (mean - mu0) / (variance / n).sqrt();
  match args.get(3) {
    None => Ok(Expr::Real(
      crate::functions::math_ast::numeric_utils::erfc_cf(
        z.abs() / std::f64::consts::SQRT_2,
      ),
    )),
    Some(Expr::String(p)) if p == "PValue" => Ok(Expr::Real(
      crate::functions::math_ast::numeric_utils::erfc_cf(
        z.abs() / std::f64::consts::SQRT_2,
      ),
    )),
    Some(Expr::String(p)) if p == "TestStatistic" => Ok(Expr::Real(z)),
    _ => Ok(unevaluated(args)),
  }
}

/// FisherRatioTest[data] / FisherRatioTest[data, sigma0^2] /
/// FisherRatioTest[data, sigma0^2, "property"] - one-sample variance
/// test. The statistic is Total[(x - mean)^2]/sigma0^2 (chi-square
/// with n-1 degrees of freedom under the null) and the p-value is the
/// two-sided 2 Min[F[T], 1 - F[T]]. A non-positive or non-numeric
/// second argument emits FisherRatioTest::sigmnt.
pub fn fisher_ratio_test_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "FisherRatioTest".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() || args.len() > 3 {
    return Ok(unevaluated(args));
  }
  let bad_data = |args: &[Expr]| {
    crate::emit_message(&format!(
      "FisherRatioTest::vctnln1: The argument {} at position 1 should be a vector of real numbers with length greater than 1 or a list containing two such vectors.",
      crate::syntax::expr_to_string(&args[0])
    ));
    Ok(Expr::FunctionCall {
      name: "FisherRatioTest".to_string(),
      args: args.to_vec().into(),
    })
  };
  let data: Vec<f64> = match &args[0] {
    Expr::List(items) if items.len() >= 2 => {
      match items
        .iter()
        .map(crate::functions::math_ast::numeric_utils::try_eval_to_f64)
        .collect::<Option<Vec<f64>>>()
      {
        Some(v) => v,
        None => return bad_data(args),
      }
    }
    _ => return bad_data(args),
  };
  let sigma2 = match args.get(1) {
    None => 1.0,
    Some(e) => {
      match crate::functions::math_ast::numeric_utils::try_eval_to_f64(e) {
        Some(v) if v > 0.0 => v,
        _ => {
          crate::emit_message(&format!(
            "FisherRatioTest::sigmnt: The argument {} should be a positive number.",
            crate::syntax::expr_to_string(&args[1])
          ));
          return Ok(unevaluated(args));
        }
      }
    }
  };
  let n = data.len() as f64;
  let mean = data.iter().sum::<f64>() / n;
  let t = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / sigma2;
  match args.get(2) {
    None => {}
    Some(Expr::String(p)) if p == "PValue" => {}
    Some(Expr::String(p)) if p == "TestStatistic" => {
      return Ok(Expr::Real(t));
    }
    _ => return Ok(unevaluated(args)),
  }
  // Two-sided p-value from the chi-square(n-1) CDF
  let upper = crate::functions::math_ast::gamma::gamma_regularized_numeric(
    (n - 1.0) / 2.0,
    t / 2.0,
  );
  let lower = 1.0 - upper;
  Ok(Expr::Real(2.0 * lower.min(upper)))
}

/// GroupElements[CyclicGroup[n]] - the n rotations, ordered by power of the
/// generator (which coincides with lexicographic image-list order).
fn cyclic_group_elements(n: usize) -> Expr {
  if n <= 1 {
    return Expr::List(vec![make_cycles_multi(Vec::new())].into());
  }
  let mut elements = Vec::with_capacity(n);
  for k in 0..n {
    let image: Vec<i128> = (0..n).map(|i| ((i + k) % n + 1) as i128).collect();
    elements.push(images_to_cycles(&image));
  }
  Expr::List(elements.into())
}

/// GroupElements[SymmetricGroup[n]] - all n! permutations in lexicographic
/// image-list order, matching wolframscript.
fn symmetric_group_elements(n: usize) -> Expr {
  if n <= 1 {
    return Expr::List(vec![make_cycles_multi(Vec::new())].into());
  }
  let mut image: Vec<i128> = (1..=n as i128).collect();
  let mut elements: Vec<Expr> = Vec::new();
  loop {
    elements.push(images_to_cycles(&image));
    if !next_permutation(&mut image) {
      break;
    }
  }
  Expr::List(elements.into())
}

/// Cycle lengths (excluding fixed points) of a `Cycles[{{...}, ...}]` expr.
/// Returns None when the expr is not a well-formed Cycles object.
fn cycles_expr_lengths(e: &Expr) -> Option<Vec<usize>> {
  if let Expr::FunctionCall { name, args } = e
    && name == "Cycles"
    && args.len() == 1
    && let Expr::List(cycles) = &args[0]
  {
    let mut lengths = Vec::with_capacity(cycles.len());
    for c in cycles.iter() {
      if let Expr::List(points) = c {
        lengths.push(points.len());
      } else {
        return None;
      }
    }
    Some(lengths)
  } else {
    None
  }
}

/// CycleIndexPolynomial[group, {x1, x2, ...}] - the cycle index polynomial
/// (1/|G|) * Sum over elements of Prod x_l^(number of l-cycles), counting
/// fixed points as 1-cycles. Variables beyond the supplied list are 1
/// (matching wolframscript).
pub fn cycle_index_polynomial_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "CycleIndexPolynomial".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated());
  }
  let call_str = || {
    crate::syntax::format_expr(
      &Expr::FunctionCall {
        name: "CycleIndexPolynomial".to_string(),
        args: args.to_vec().into(),
      },
      crate::syntax::ExprForm::Output,
    )
  };

  // Resolve the group to (element list, degree of the action)
  let resolved: Option<(Expr, usize)> = match &args[0] {
    Expr::FunctionCall { name, args: gargs } if gargs.len() == 1 => {
      match name.as_str() {
        "CyclicGroup" | "SymmetricGroup" | "AlternatingGroup"
        | "DihedralGroup" => {
          let min_n = if name == "DihedralGroup" { 1 } else { 0 };
          match &gargs[0] {
            Expr::Integer(n) if *n >= min_n => {
              let n = *n as usize;
              let elements = match name.as_str() {
                "CyclicGroup" => cyclic_group_elements(n),
                "SymmetricGroup" => symmetric_group_elements(n),
                "AlternatingGroup" => alternating_group_elements(n),
                _ => dihedral_group_elements(n),
              };
              Some((elements, n))
            }
            _ => None,
          }
        }
        "AbelianGroup" => abelian_factors(&gargs[0]).map(|factors| {
          let degree = factors.iter().sum();
          (abelian_group_elements(&factors), degree)
        }),
        "PermutationGroup" => permutation_group_closure(&gargs[0])
          .map(|(elements, degree)| (Expr::List(elements.into()), degree)),
        _ => None,
      }
    }
    _ => None,
  };
  let (elements, degree) = match resolved {
    Some(r) => r,
    None => {
      crate::emit_message(&format!(
        "CycleIndexPolynomial::grp: {} is not a valid group.",
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
      return Ok(unevaluated());
    }
  };
  let vars = match &args[1] {
    Expr::List(v) => v,
    _ => {
      crate::emit_message(&format!(
        "CycleIndexPolynomial::list: List expected at position 2 in {}.",
        call_str()
      ));
      return Ok(unevaluated());
    }
  };
  let element_list = match &elements {
    Expr::List(items) => items,
    _ => return Ok(unevaluated()),
  };
  let order = element_list.len() as i128;
  if order == 0 {
    return Ok(unevaluated());
  }

  // Aggregate cycle types: counts of (multiplicity per cycle length)
  let mut type_counts: std::collections::BTreeMap<Vec<usize>, i128> =
    Default::default();
  for e in element_list.iter() {
    let lengths = match cycles_expr_lengths(e) {
      Some(l) => l,
      None => return Ok(unevaluated()),
    };
    let moved: usize = lengths.iter().sum();
    let mut mult = vec![0usize; degree + 1];
    if degree >= moved {
      mult[1] += degree - moved; // fixed points are 1-cycles
    }
    for l in lengths {
      if l <= degree {
        mult[l] += 1;
      }
    }
    *type_counts.entry(mult).or_insert(0) += 1;
  }

  let var_for =
    |k: usize| -> Expr { vars.get(k - 1).cloned().unwrap_or(Expr::Integer(1)) };
  let terms: Vec<Expr> = type_counts
    .into_iter()
    .map(|(mult, count)| {
      let mut factors: Vec<Expr> = vec![Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(count), Expr::Integer(order)].into(),
      }];
      for (k, &m) in mult.iter().enumerate().skip(1) {
        if m == 0 {
          continue;
        }
        let base = var_for(k);
        factors.push(if m == 1 {
          base
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(base),
            right: Box::new(Expr::Integer(m as i128)),
          }
        });
      }
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors.into(),
      }
    })
    .collect();
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  })
}

/// Expand PermutationGroup generators into the full group via BFS closure.
/// Returns the elements (as Cycles exprs, ordered lexicographically by
/// image list) and the degree (largest moved point among the generators).
fn permutation_group_closure(gens: &Expr) -> Option<(Vec<Expr>, usize)> {
  let gen_list = match gens {
    Expr::List(g) => g,
    _ => return None,
  };
  // Degree: largest point appearing in any generator
  let mut degree = 0usize;
  let mut gen_lengths: Vec<Vec<Vec<usize>>> = Vec::new();
  for g in gen_list.iter() {
    if let Expr::FunctionCall { name, args } = g
      && name == "Cycles"
      && args.len() == 1
      && let Expr::List(cycles) = &args[0]
    {
      let mut parsed = Vec::new();
      for c in cycles.iter() {
        if let Expr::List(points) = c {
          let mut cyc = Vec::with_capacity(points.len());
          for p in points.iter() {
            if let Expr::Integer(v) = p
              && *v >= 1
            {
              degree = degree.max(*v as usize);
              cyc.push(*v as usize);
            } else {
              return None;
            }
          }
          parsed.push(cyc);
        } else {
          return None;
        }
      }
      gen_lengths.push(parsed);
    } else {
      return None;
    }
  }
  // Generators as image lists over 1..degree
  let to_image = |cycles: &[Vec<usize>]| -> Vec<usize> {
    let mut image: Vec<usize> = (1..=degree).collect();
    for cyc in cycles {
      for w in 0..cyc.len() {
        image[cyc[w] - 1] = cyc[(w + 1) % cyc.len()];
      }
    }
    image
  };
  let gen_images: Vec<Vec<usize>> =
    gen_lengths.iter().map(|c| to_image(c)).collect();
  let identity: Vec<usize> = (1..=degree).collect();
  let mut seen: std::collections::BTreeSet<Vec<usize>> =
    std::collections::BTreeSet::new();
  seen.insert(identity.clone());
  let mut queue = std::collections::VecDeque::new();
  queue.push_back(identity);
  while let Some(cur) = queue.pop_front() {
    for g in &gen_images {
      // compose: apply cur first, then g
      let next: Vec<usize> = cur.iter().map(|&v| g[v - 1]).collect();
      if seen.insert(next.clone()) {
        if seen.len() > 100_000 {
          return None;
        }
        queue.push_back(next);
      }
    }
  }
  let elements: Vec<Expr> = seen
    .into_iter()
    .map(|image| {
      let image_i128: Vec<i128> = image.iter().map(|&v| v as i128).collect();
      images_to_cycles(&image_i128)
    })
    .collect();
  Some((elements, degree))
}

/// Convert a Cycles expression into a 1-based image list over 1..n.
fn cycles_to_image(e: &Expr, n: usize) -> Option<Vec<usize>> {
  let mut image: Vec<usize> = (1..=n).collect();
  if let Expr::FunctionCall { name, args } = e
    && name == "Cycles"
    && args.len() == 1
    && let Expr::List(cycles) = &args[0]
  {
    for c in cycles.iter() {
      let Expr::List(points) = c else {
        return None;
      };
      let pts: Vec<usize> = points
        .iter()
        .map(|p| match p {
          Expr::Integer(v) if *v >= 1 && (*v as usize) <= n => {
            Some(*v as usize)
          }
          _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
      for w in 0..pts.len() {
        image[pts[w] - 1] = pts[(w + 1) % pts.len()];
      }
    }
    Some(image)
  } else {
    None
  }
}

/// The elements of a supported group as (image lists, degree), in the
/// same order GroupElements uses.
fn group_element_images(group: &Expr) -> Option<(Vec<Vec<usize>>, usize)> {
  if let Expr::FunctionCall { name, args } = group
    && args.len() == 1
  {
    if name == "PermutationGroup" {
      let (elements, degree) = permutation_group_closure(&args[0])?;
      let images = elements
        .iter()
        .map(|e| cycles_to_image(e, degree))
        .collect::<Option<Vec<_>>>()?;
      return Some((images, degree));
    }
    let min_n = if name == "DihedralGroup" { 1 } else { 0 };
    if let Expr::Integer(n) = &args[0]
      && *n >= min_n
    {
      let n = *n as usize;
      let elements = match name.as_str() {
        "CyclicGroup" => cyclic_group_elements(n),
        "SymmetricGroup" => symmetric_group_elements(n),
        "AlternatingGroup" => alternating_group_elements(n),
        "DihedralGroup" => dihedral_group_elements(n),
        _ => return None,
      };
      let degree = n.max(1);
      if let Expr::List(items) = &elements {
        let images = items
          .iter()
          .map(|e| cycles_to_image(e, degree))
          .collect::<Option<Vec<_>>>()?;
        return Some((images, degree));
      }
    }
  }
  None
}

/// Whether an expression is shaped like a group object (for ::grp messages).
fn looks_like_group(e: &Expr) -> bool {
  matches!(
    e,
    Expr::FunctionCall { name, .. }
      if matches!(
        name.as_str(),
        "SymmetricGroup"
          | "AlternatingGroup"
          | "CyclicGroup"
          | "DihedralGroup"
          | "AbelianGroup"
          | "PermutationGroup"
      )
  )
}

/// GroupMultiplicationTable[g] — entry (i, j) is the position (in
/// GroupElements order) of the product of elements i and j (left-to-right
/// composition, as in PermutationProduct).
pub fn group_multiplication_table_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "GroupMultiplicationTable".to_string(),
    args: args.to_vec().into(),
  };
  let Some((images, _)) = group_element_images(&args[0]) else {
    if !looks_like_group(&args[0]) {
      crate::emit_message(&format!(
        "GroupMultiplicationTable::grp: {} is not a valid group.",
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
    }
    return Ok(unevaluated());
  };
  let index: std::collections::HashMap<&Vec<usize>, usize> = images
    .iter()
    .enumerate()
    .map(|(i, img)| (img, i + 1))
    .collect();
  let mut rows = Vec::with_capacity(images.len());
  for a in &images {
    let mut row = Vec::with_capacity(images.len());
    for b in &images {
      // apply a, then b
      let product: Vec<usize> = a.iter().map(|&v| b[v - 1]).collect();
      match index.get(&product) {
        Some(&i) => row.push(Expr::Integer(i as i128)),
        None => return Ok(unevaluated()),
      }
    }
    rows.push(Expr::List(row.into()));
  }
  Ok(Expr::List(rows.into()))
}

/// GroupStabilizer[g, {p1, ...}] — the pointwise stabilizer subgroup.
/// Supported for the named groups, matching wolframscript's canonical
/// generator choices (SymmetricGroup: transposition + full cycle on the
/// remaining points; AlternatingGroup: consecutive 3-cycles; Cyclic and
/// Dihedral: the at-most-one nontrivial fixing element). The generic
/// PermutationGroup form follows wolframscript's internal Schreier-Sims
/// generator selection and is not replicated.
pub fn group_stabilizer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "GroupStabilizer".to_string(),
    args: args.to_vec().into(),
  };
  if !looks_like_group(&args[0]) {
    crate::emit_message(&format!(
      "GroupStabilizer::grp: {} is not a valid group.",
      crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
    ));
    return Ok(unevaluated());
  }
  let Expr::List(pt_items) = &args[1] else {
    return Ok(unevaluated());
  };
  // The empty point list stabilizes nothing: the whole group
  if pt_items.is_empty() {
    return Ok(args[0].clone());
  }
  let mut pts: Vec<usize> = Vec::new();
  for p in pt_items.iter() {
    match p {
      Expr::Integer(v) if *v >= 1 => pts.push(*v as usize),
      _ => return Ok(unevaluated()),
    }
  }
  let permutation_group = |gens: Vec<Expr>| Expr::FunctionCall {
    name: "PermutationGroup".to_string(),
    args: vec![Expr::List(gens.into())].into(),
  };
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && gargs.len() == 1
    && let Expr::Integer(n) = &gargs[0]
    && *n >= 0
  {
    let n = *n as usize;
    let remaining: Vec<i128> = (1..=n as i128)
      .filter(|v| !pts.contains(&(*v as usize)))
      .collect();
    match name.as_str() {
      "SymmetricGroup" => {
        let mut gens = Vec::new();
        if remaining.len() >= 2 {
          gens.push(make_cycles(vec![remaining[0], remaining[1]]));
        }
        if remaining.len() >= 3 {
          gens.push(make_cycles(remaining.clone()));
        }
        return Ok(permutation_group(gens));
      }
      "AlternatingGroup" => {
        let mut gens = Vec::new();
        for w in remaining.windows(3) {
          gens.push(make_cycles(w.to_vec()));
        }
        return Ok(permutation_group(gens));
      }
      "CyclicGroup" | "DihedralGroup" => {
        // Enumerate and keep the (at most one) nontrivial element fixing
        // every point
        if let Some((images, _)) = group_element_images(&args[0]) {
          let mut gens = Vec::new();
          for img in &images {
            let fixes = pts.iter().all(|&p| p > img.len() || img[p - 1] == p);
            let identity = img.iter().enumerate().all(|(i, &v)| v == i + 1);
            if fixes && !identity {
              let img_i128: Vec<i128> =
                img.iter().map(|&v| v as i128).collect();
              gens.push(images_to_cycles(&img_i128));
            }
          }
          return Ok(permutation_group(gens));
        }
      }
      _ => {}
    }
  }
  // PermutationGroup: the stabilizer is computed by enumeration when its
  // generator choice is forced (at most one nontrivial element);
  // wolframscript's generator selection for larger stabilizers follows
  // its internal Schreier-Sims chains and is not replicated.
  if let Expr::FunctionCall { name, args: gargs } = &args[0]
    && name == "PermutationGroup"
    && gargs.len() == 1
    && let Some((images, _)) = group_element_images(&args[0])
  {
    let fixing: Vec<&Vec<usize>> = images
      .iter()
      .filter(|img| pts.iter().all(|&p| p > img.len() || img[p - 1] == p))
      .collect();
    if fixing.len() <= 2 {
      let gens: Vec<Expr> = fixing
        .iter()
        .filter(|img| img.iter().enumerate().any(|(i, &v)| v != i + 1))
        .map(|img| {
          let img_i128: Vec<i128> = img.iter().map(|&v| v as i128).collect();
          images_to_cycles(&img_i128)
        })
        .collect();
      return Ok(permutation_group(gens));
    }
  }
  Ok(unevaluated())
}
