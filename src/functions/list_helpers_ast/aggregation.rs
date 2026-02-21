#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based AllTrue: check if predicate is true for all elements.
/// AllTrue[{a, b, c}, pred] -> True if pred[x] is True for all x
pub fn all_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AllTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) != Some(true) {
      return Ok(bool_to_expr(false));
    }
  }

  Ok(bool_to_expr(true))
}

/// AST-based AnyTrue: check if predicate is true for any element.
/// AnyTrue[{a, b, c}, pred] -> True if pred[x] is True for any x
pub fn any_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AnyTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(bool_to_expr(true));
    }
  }

  Ok(bool_to_expr(false))
}

/// AST-based NoneTrue: check if predicate is false for all elements.
/// NoneTrue[{a, b, c}, pred] -> True if pred[x] is False for all x
pub fn none_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NoneTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(bool_to_expr(false));
    }
  }

  Ok(bool_to_expr(true))
}

/// AST-based CountBy: count elements by the value of a function.
/// CountBy[{a, b, c}, f] -> association of f[x] -> count
pub fn count_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CountBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, i128> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if let Some(count) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, 1);
    }
  }

  // Build association preserving order
  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let count = counts[&k];
      let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
      (key_expr, Expr::Integer(count))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based GroupBy: group elements by the value of a function.
/// GroupBy[{a, b, c}, f] -> association of f[x] -> {elements with that f value}
pub fn group_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "GroupBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut groups: HashMap<String, Vec<Expr>> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if let Some(group) = groups.get_mut(&key_str) {
      group.push(item.clone());
    } else {
      order.push(key_str.clone());
      groups.insert(key_str, vec![item.clone()]);
    }
  }

  // Build association preserving order
  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let items = groups.remove(&k).unwrap();
      let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
      (key_expr, Expr::List(items))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based Median: calculate median of a list.
pub fn median_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Median".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Median: list is empty".into(),
    ));
  }

  // Check for list-of-lists (matrix) input → columnwise median
  if items.iter().all(|item| matches!(item, Expr::List(_))) {
    let rows: Vec<&Vec<Expr>> = items
      .iter()
      .filter_map(|item| {
        if let Expr::List(row) = item {
          Some(row)
        } else {
          None
        }
      })
      .collect();
    if !rows.is_empty() {
      let ncols = rows[0].len();
      if rows.iter().all(|r| r.len() == ncols) {
        let mut result = Vec::new();
        for col in 0..ncols {
          let column: Vec<Expr> = rows.iter().map(|r| r[col].clone()).collect();
          let col_median = median_ast(&Expr::List(column))?;
          result.push(col_median);
        }
        return Ok(Expr::List(result));
      }
    }
  }

  // Check if all items are integers
  let all_integers = items.iter().all(|i| matches!(i, Expr::Integer(_)));

  if all_integers {
    // Sort integer values
    let mut int_values: Vec<i128> = items
      .iter()
      .filter_map(|i| {
        if let Expr::Integer(n) = i {
          Some(*n)
        } else {
          None
        }
      })
      .collect();
    int_values.sort();

    let len = int_values.len();
    if len % 2 == 1 {
      Ok(Expr::Integer(int_values[len / 2]))
    } else {
      // Average of two middle values
      let a = int_values[len / 2 - 1];
      let b = int_values[len / 2];
      let sum = a + b;
      if sum % 2 == 0 {
        Ok(Expr::Integer(sum / 2))
      } else {
        // Return as Rational
        fn gcd(a: i128, b: i128) -> i128 {
          if b == 0 { a } else { gcd(b, a % b) }
        }
        let g = gcd(sum.abs(), 2);
        Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sum / g), Expr::Integer(2 / g)],
        })
      }
    }
  } else {
    // Check if any Real inputs - preserve Real type
    let has_real = items.iter().any(|i| matches!(i, Expr::Real(_)));

    // Extract numeric values as f64
    let mut values: Vec<f64> = Vec::new();
    for item in items {
      if let Some(n) = expr_to_f64(item) {
        values.push(n);
      } else {
        return Ok(Expr::FunctionCall {
          name: "Median".to_string(),
          args: vec![list.clone()],
        });
      }
    }

    values
      .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = values.len();
    let result = if len % 2 == 1 {
      values[len / 2]
    } else {
      (values[len / 2 - 1] + values[len / 2]) / 2.0
    };

    // Preserve Real type if inputs had Real
    if has_real {
      Ok(Expr::Real(result))
    } else {
      Ok(f64_to_expr(result))
    }
  }
}

/// AST-based TakeLargest: take n largest elements.
pub fn take_largest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  // Extract numeric values with indices
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = expr_to_f64(item) {
      keyed.push((v, item.clone()));
    } else {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  }

  keyed
    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

  let take = (n as usize).min(keyed.len());
  let result: Vec<Expr> =
    keyed.into_iter().take(take).map(|(_, e)| e).collect();

  Ok(Expr::List(result))
}

/// AST-based TakeSmallest: take n smallest elements.
pub fn take_smallest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  // Extract numeric values with indices
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = expr_to_f64(item) {
      keyed.push((v, item.clone()));
    } else {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  }

  keyed
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

  let take = (n as usize).min(keyed.len());
  let result: Vec<Expr> =
    keyed.into_iter().take(take).map(|(_, e)| e).collect();

  Ok(Expr::List(result))
}

/// AST-based MinMax: return {min, max} of a list.
pub fn min_max_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MinMax".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![
      Expr::Identifier("Infinity".to_string()),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      },
    ]));
  }

  let mut min_val = f64::INFINITY;
  let mut max_val = f64::NEG_INFINITY;

  for item in items {
    if let Some(n) = expr_to_f64(item) {
      if n < min_val {
        min_val = n;
      }
      if n > max_val {
        max_val = n;
      }
    } else {
      return Ok(Expr::FunctionCall {
        name: "MinMax".to_string(),
        args: vec![list.clone()],
      });
    }
  }

  Ok(Expr::List(vec![f64_to_expr(min_val), f64_to_expr(max_val)]))
}

/// Gather[list] - gathers elements into sublists of identical elements
pub fn gather_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Gather expects a list argument".into(),
      ));
    }
  };
  let mut groups: Vec<Vec<Expr>> = Vec::new();
  for item in items {
    let found = groups.iter_mut().find(|g| {
      crate::syntax::expr_to_string(&g[0])
        == crate::syntax::expr_to_string(item)
    });
    if let Some(group) = found {
      group.push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// GatherBy[list, f] - gathers elements into sublists by applying f
pub fn gather_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GatherBy expects a list as first argument".into(),
      ));
    }
  };
  let mut groups: Vec<(String, Vec<Expr>)> = Vec::new();
  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    let found = groups.iter_mut().find(|(k, _)| *k == key_str);
    if let Some((_, group)) = found {
      group.push(item.clone());
    } else {
      groups.push((key_str, vec![item.clone()]));
    }
  }
  Ok(Expr::List(
    groups.into_iter().map(|(_, v)| Expr::List(v)).collect(),
  ))
}

/// Split[list] - splits into sublists of identical consecutive elements
pub fn split_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Split expects a list argument".into(),
      ));
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let last_group = groups.last().unwrap();
    if crate::syntax::expr_to_string(&last_group[0])
      == crate::syntax::expr_to_string(item)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// Split[list, test] - splits list where consecutive elements satisfy test function
pub fn split_with_test_ast(
  list: &Expr,
  test: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Split".to_string(),
        args: vec![list.clone(), test.clone()],
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let last_item = groups.last().unwrap().last().unwrap();
    let test_result = apply_func_to_two_args(test, last_item, item)?;
    let passes = matches!(
      &test_result,
      Expr::Identifier(name) if name == "True"
    );
    if passes {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// SplitBy[list, f] - splits into sublists of consecutive elements with same f value
pub fn split_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "SplitBy expects a list as first argument".into(),
      ));
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut prev_key = apply_func_ast(func, &items[0])?;
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let key = apply_func_ast(func, item)?;
    if crate::syntax::expr_to_string(&key)
      == crate::syntax::expr_to_string(&prev_key)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
      prev_key = key;
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// BinCounts[data, {min, max, dx}] - count data points in equal-width bins
/// Bins are [min, min+dx), [min+dx, min+2dx), ..., [max-dx, max)
pub fn bin_counts_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BinCounts".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract numeric values from data, skip non-numeric
  let values: Vec<f64> = data.iter().filter_map(expr_to_f64).collect();

  let (min_val, max_val, dx) = if args.len() == 1 {
    // BinCounts[data] - default dx=1, aligned to integer boundaries
    if values.is_empty() {
      return Ok(Expr::List(vec![]));
    }
    let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dx = 1.0;
    let mut lo = (data_min / dx).floor() * dx;
    if (data_min - lo).abs() < 1e-12 {
      lo -= dx;
    }
    let mut hi = (data_max / dx).ceil() * dx;
    if (data_max - hi).abs() < 1e-12 {
      hi += dx;
    }
    (lo, hi, dx)
  } else if args.len() == 2 {
    match &args[1] {
      // BinCounts[data, dx]
      Expr::Integer(dx_int) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![]));
        }
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_int as f64;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      Expr::Real(dx_f) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![]));
        }
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_f;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      // BinCounts[data, {min, max, dx}]
      Expr::List(spec) if spec.len() == 3 => {
        let min_v = match expr_to_f64(&spec[0]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let max_v = match expr_to_f64(&spec[1]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let dx = match expr_to_f64(&spec[2]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec(),
            });
          }
        };
        (min_v, max_v, dx)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "BinCounts".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "BinCounts".to_string(),
      args: args.to_vec(),
    });
  };

  if dx <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "BinCounts: bin width must be positive".into(),
    ));
  }

  let num_bins = ((max_val - min_val) / dx).round() as usize;
  let mut counts = vec![0i128; num_bins];

  for &v in &values {
    if v >= min_val && v < max_val {
      let bin = ((v - min_val) / dx) as usize;
      let bin = bin.min(num_bins - 1);
      counts[bin] += 1;
    }
  }

  Ok(Expr::List(counts.into_iter().map(Expr::Integer).collect()))
}

/// Round a positive value to the nearest "nice" number (1, 2, 5, 10, 20, 50, …)
/// using log-scale distance to decide which is closest.
fn nice_number(x: f64) -> f64 {
  if x <= 0.0 {
    return 1.0;
  }
  let exp = x.log10().floor() as i32;
  let base = 10f64.powi(exp);
  let candidates = [1.0 * base, 2.0 * base, 5.0 * base, 10.0 * base];
  let log_x = x.ln();
  candidates
    .iter()
    .copied()
    .min_by(|a, b| {
      (a.ln() - log_x)
        .abs()
        .partial_cmp(&(b.ln() - log_x).abs())
        .unwrap()
    })
    .unwrap()
}

/// Compute the interquartile range (Q3 - Q1) of a sorted slice.
fn interquartile_range(sorted: &[f64]) -> f64 {
  let n = sorted.len();
  if n < 2 {
    return 0.0;
  }
  let q1 = wolfram_quantile(sorted, 0.25);
  let q3 = wolfram_quantile(sorted, 0.75);
  q3 - q1
}

/// Wolfram's default Quantile: h = n*p, j = floor(h), g = h - j.
/// If g > 0: x_{j+1}, else x_j (1-based indexing).
fn wolfram_quantile(sorted: &[f64], p: f64) -> f64 {
  let n = sorted.len();
  if n == 0 {
    return 0.0;
  }
  let h = n as f64 * p;
  let j = h.floor() as usize;
  let g = h - j as f64;
  if g > 1e-12 {
    // x_{j+1}, 0-based: sorted[j]
    sorted[j.min(n - 1)]
  } else {
    // x_j, 0-based: sorted[j-1], but j could be 0
    if j == 0 {
      sorted[0]
    } else {
      sorted[(j - 1).min(n - 1)]
    }
  }
}

/// HistogramList[data] - returns {bin_edges, counts}
/// HistogramList[data, {dx}] - explicit bin width
/// HistogramList[data, {min, max, dx}] - explicit bin specification
pub fn histogram_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HistogramList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let values: Vec<f64> = data.iter().filter_map(expr_to_f64).collect();

  if values.is_empty() {
    return Ok(Expr::List(vec![Expr::List(vec![]), Expr::List(vec![])]));
  }

  let (min_val, max_val, dx) = if args.len() == 1 {
    // Auto-binning: Freedman-Diaconis rule with nice number rounding
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let data_min = sorted[0];
    let data_max = sorted[sorted.len() - 1];
    let n = sorted.len() as f64;

    let iqr = interquartile_range(&sorted);
    let dx = if iqr > 0.0 {
      nice_number(2.0 * iqr / n.cbrt())
    } else if data_max > data_min {
      // Fallback: use range / Sturges' rule
      let sturges_bins = (n.log2() + 1.0).ceil().max(1.0);
      nice_number((data_max - data_min) / sturges_bins)
    } else {
      // All values identical
      let v = data_min.abs();
      if v == 0.0 { 1.0 } else { nice_number(v) }
    };

    let lo = (data_min / dx).floor() * dx;
    let mut hi = (data_max / dx).ceil() * dx;
    // Ensure data_max is strictly inside the last bin
    if (data_max - hi).abs() < 1e-12 * dx.max(1.0) {
      hi += dx;
    }
    (lo, hi, dx)
  } else if args.len() == 2 {
    match &args[1] {
      // HistogramList[data, {dx}]
      Expr::List(spec) if spec.len() == 1 => {
        let dx = match expr_to_f64(&spec[0]) {
          Some(v) if v > 0.0 => v,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = (data_min / dx).floor() * dx;
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 * dx.max(1.0) {
          hi += dx;
        }
        (lo, hi, dx)
      }
      // HistogramList[data, {min, max, dx}]
      Expr::List(spec) if spec.len() == 3 => {
        let min_v = match expr_to_f64(&spec[0]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let max_v = match expr_to_f64(&spec[1]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let dx = match expr_to_f64(&spec[2]) {
          Some(v) if v > 0.0 => v,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        (min_v, max_v, dx)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "HistogramList".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "HistogramList".to_string(),
      args: args.to_vec(),
    });
  };

  if dx <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "HistogramList: bin width must be positive".into(),
    ));
  }

  let num_bins = ((max_val - min_val) / dx).round() as usize;
  if num_bins == 0 {
    return Ok(Expr::List(vec![Expr::List(vec![]), Expr::List(vec![])]));
  }

  let mut counts = vec![0i128; num_bins];
  for &v in &values {
    if v >= min_val && v <= max_val {
      let bin = ((v - min_val) / dx) as usize;
      let bin = bin.min(num_bins - 1);
      counts[bin] += 1;
    }
  }

  // Build bin edges list
  let mut edges = Vec::with_capacity(num_bins + 1);
  for i in 0..=num_bins {
    edges.push(f64_to_expr(min_val + i as f64 * dx));
  }

  Ok(Expr::List(vec![
    Expr::List(edges),
    Expr::List(counts.into_iter().map(Expr::Integer).collect()),
  ]))
}

/// TakeLargestBy[list, f, n] - take the n largest elements sorted by f
pub fn take_largest_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "TakeLargestBy expects exactly 3 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let f = &args[1];
  let n = match &args[2] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Compute f[item] for each item
  let mut with_keys: Vec<(Expr, Expr)> = Vec::new();
  for item in list {
    let key = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![f.clone(), Expr::List(vec![item.clone()])],
    })?;
    with_keys.push((key, item.clone()));
  }
  // Sort descending by key (largest first)
  // compare_exprs returns 1 if first < second in canonical order
  with_keys.sort_by(|a, b| {
    let ord = compare_exprs(&a.0, &b.0);
    // ord > 0 means a.key < b.key, so b should come first (descending)
    if ord > 0 {
      std::cmp::Ordering::Greater
    } else if ord < 0 {
      std::cmp::Ordering::Less
    } else {
      std::cmp::Ordering::Equal
    }
  });
  let result: Vec<Expr> =
    with_keys.into_iter().take(n).map(|(_, v)| v).collect();
  Ok(Expr::List(result))
}

/// TakeSmallestBy[list, f, n] - take the n smallest elements sorted by f
pub fn take_smallest_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "TakeSmallestBy expects exactly 3 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let f = &args[1];
  let n = match &args[2] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Compute f[item] for each item
  let mut with_keys: Vec<(Expr, Expr)> = Vec::new();
  for item in list {
    let key = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![f.clone(), Expr::List(vec![item.clone()])],
    })?;
    with_keys.push((key, item.clone()));
  }
  // Sort ascending by key (smallest first)
  // compare_exprs returns 1 if first < second in canonical order
  with_keys.sort_by(|a, b| {
    let ord = compare_exprs(&a.0, &b.0);
    // ord > 0 means a.key < b.key, so a should come first (ascending)
    if ord > 0 {
      std::cmp::Ordering::Less
    } else if ord < 0 {
      std::cmp::Ordering::Greater
    } else {
      std::cmp::Ordering::Equal
    }
  });
  let result: Vec<Expr> =
    with_keys.into_iter().take(n).map(|(_, v)| v).collect();
  Ok(Expr::List(result))
}
