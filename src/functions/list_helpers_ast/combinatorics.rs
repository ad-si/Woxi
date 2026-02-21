#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Permutations: generate all permutations of a list.
/// Permutations[{a, b, c}] -> all permutations
/// Permutations[{a, b, c}, {k}] -> all permutations of length k
pub fn permutations_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Permutations".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = items.len();

  if args.len() >= 2 {
    match &args[1] {
      // {k} means exactly k-permutations
      Expr::List(spec) if spec.len() == 1 => {
        let k = expr_to_i128(&spec[0]).unwrap_or(n as i128) as usize;
        if k > n {
          return Ok(Expr::List(vec![]));
        }
        let mut result = Vec::new();
        let indices: Vec<usize> = (0..n).collect();
        generate_k_permutations(
          &indices,
          k,
          &mut vec![],
          &mut vec![false; n],
          &items,
          &mut result,
        );
        return Ok(Expr::List(result));
      }
      // Plain integer k means all permutations of length 0 through k
      _ => {
        let max_k = expr_to_i128(&args[1]).unwrap_or(n as i128) as usize;
        let max_k = max_k.min(n);
        let mut result = Vec::new();
        let indices: Vec<usize> = (0..n).collect();
        for k in 0..=max_k {
          generate_k_permutations(
            &indices,
            k,
            &mut vec![],
            &mut vec![false; n],
            &items,
            &mut result,
          );
        }
        return Ok(Expr::List(result));
      }
    }
  }

  // Default: full permutations (length n)
  let mut result = Vec::new();
  let indices: Vec<usize> = (0..n).collect();
  generate_k_permutations(
    &indices,
    n,
    &mut vec![],
    &mut vec![false; n],
    &items,
    &mut result,
  );
  Ok(Expr::List(result))
}

/// Helper to generate k-permutations
fn generate_k_permutations(
  _indices: &[usize],
  k: usize,
  current: &mut Vec<usize>,
  used: &mut Vec<bool>,
  items: &[Expr],
  result: &mut Vec<Expr>,
) {
  if current.len() == k {
    let perm: Vec<Expr> = current.iter().map(|&i| items[i].clone()).collect();
    result.push(Expr::List(perm));
    return;
  }
  for i in 0..items.len() {
    if !used[i] {
      used[i] = true;
      current.push(i);
      generate_k_permutations(_indices, k, current, used, items, result);
      current.pop();
      used[i] = false;
    }
  }
}

/// AST-based Subsets: generate subsets of a list.
/// Subsets[{a, b, c}] -> all subsets
/// Subsets[{a, b, c}, {k}] -> all subsets of size k
pub fn subsets_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subsets".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Generate all subsets based on level spec (args[1])
  let mut result = if args.len() >= 2 {
    let is_all = matches!(&args[1], Expr::Identifier(s) if s == "All");
    if is_all {
      // All subsets
      let n = items.len();
      let mut r = Vec::new();
      for k in 0..=n {
        generate_combinations(&items, k, 0, &mut vec![], &mut r);
      }
      r
    } else {
      match &args[1] {
        // Subsets[list, {k}] - subsets of exactly size k
        Expr::List(spec) if spec.len() == 1 => {
          if let Some(k) = expr_to_i128(&spec[0]) {
            let k = k.max(0) as usize;
            let mut r = Vec::new();
            generate_combinations(&items, k, 0, &mut vec![], &mut r);
            r
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsets".to_string(),
              args: args.to_vec(),
            });
          }
        }
        // Subsets[list, {min, max}] - subsets of sizes min through max
        Expr::List(spec) if spec.len() == 2 => {
          if let (Some(min), Some(max)) =
            (expr_to_i128(&spec[0]), expr_to_i128(&spec[1]))
          {
            let min = min.max(0) as usize;
            let max = max.min(items.len() as i128).max(0) as usize;
            let mut r = Vec::new();
            for k in min..=max {
              generate_combinations(&items, k, 0, &mut vec![], &mut r);
            }
            r
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsets".to_string(),
              args: args.to_vec(),
            });
          }
        }
        // Subsets[list, {min, max, step}] - subsets of sizes min, min+step, ...
        Expr::List(spec) if spec.len() == 3 => {
          if let (Some(min), Some(max), Some(step)) = (
            expr_to_i128(&spec[0]),
            expr_to_i128(&spec[1]),
            expr_to_i128(&spec[2]),
          ) {
            let min = min.max(0) as usize;
            let max = max.min(items.len() as i128).max(0) as usize;
            let step = step.max(1) as usize;
            let mut r = Vec::new();
            let mut k = min;
            while k <= max {
              generate_combinations(&items, k, 0, &mut vec![], &mut r);
              k += step;
            }
            r
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsets".to_string(),
              args: args.to_vec(),
            });
          }
        }
        // Subsets[list, n] - all subsets up to size n
        _ => {
          if let Some(max_k) = expr_to_i128(&args[1]) {
            let max_k = max_k.min(items.len() as i128).max(0) as usize;
            let mut r = Vec::new();
            for k in 0..=max_k {
              generate_combinations(&items, k, 0, &mut vec![], &mut r);
            }
            r
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsets".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
    }
  } else {
    // Subsets[list] - all subsets
    let n = items.len();
    let mut r = Vec::new();
    for k in 0..=n {
      generate_combinations(&items, k, 0, &mut vec![], &mut r);
    }
    r
  };

  // Apply 3rd argument: plain integer = max count, list = Part specification
  if args.len() >= 3 {
    match &args[2] {
      Expr::List(_) => {
        // Part specification like {25} or {15, 1, -2}
        result = apply_part_spec(&result, &args[2])?;
      }
      _ => {
        // Plain integer = max count (take first n subsets)
        if let Some(n) = expr_to_i128(&args[2]) {
          let n = n.max(0) as usize;
          result.truncate(n);
        }
      }
    }
  }

  Ok(Expr::List(result))
}

/// Apply a Part specification to select elements from a list of subsets.
/// Spec can be: {i} for single element, {start, end} for range, {start, end, step} for stepped range.
fn apply_part_spec(
  items: &[Expr],
  spec: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  match spec {
    Expr::List(indices) if indices.len() == 1 => {
      // {i} - take single element (1-indexed)
      if let Some(i) = expr_to_i128(&indices[0]) {
        let idx = if i > 0 {
          (i - 1) as usize
        } else if i < 0 {
          let len = items.len() as i128;
          (len + i) as usize
        } else {
          return Err(InterpreterError::EvaluationError(
            "Part: index 0 is not valid".into(),
          ));
        };
        if idx < items.len() {
          Ok(vec![items[idx].clone()])
        } else {
          Ok(vec![])
        }
      } else {
        Ok(items.to_vec())
      }
    }
    Expr::List(indices) if indices.len() == 2 => {
      // {start, end} - range (1-indexed)
      if let (Some(start), Some(end)) =
        (expr_to_i128(&indices[0]), expr_to_i128(&indices[1]))
      {
        let len = items.len() as i128;
        let s = if start > 0 { start - 1 } else { len + start };
        let e = if end > 0 { end - 1 } else { len + end };
        let mut selected = Vec::new();
        if s <= e {
          for i in s..=e {
            if i >= 0 && (i as usize) < items.len() {
              selected.push(items[i as usize].clone());
            }
          }
        } else {
          // Reverse range
          let mut i = s;
          while i >= e {
            if i >= 0 && (i as usize) < items.len() {
              selected.push(items[i as usize].clone());
            }
            i -= 1;
          }
        }
        Ok(selected)
      } else {
        Ok(items.to_vec())
      }
    }
    Expr::List(indices) if indices.len() == 3 => {
      // {start, end, step} - stepped range (1-indexed)
      if let (Some(start), Some(end), Some(step)) = (
        expr_to_i128(&indices[0]),
        expr_to_i128(&indices[1]),
        expr_to_i128(&indices[2]),
      ) {
        let len = items.len() as i128;
        let s = if start > 0 { start - 1 } else { len + start };
        let e = if end > 0 { end - 1 } else { len + end };
        let mut selected = Vec::new();
        if step > 0 {
          let mut i = s;
          while i <= e {
            if i >= 0 && (i as usize) < items.len() {
              selected.push(items[i as usize].clone());
            }
            i += step;
          }
        } else if step < 0 {
          let mut i = s;
          while i >= e {
            if i >= 0 && (i as usize) < items.len() {
              selected.push(items[i as usize].clone());
            }
            i += step; // step is negative
          }
        }
        Ok(selected)
      } else {
        Ok(items.to_vec())
      }
    }
    _ => {
      // Single integer: take element at that index
      if let Some(i) = expr_to_i128(spec) {
        let idx = if i > 0 {
          (i - 1) as usize
        } else if i < 0 {
          let len = items.len() as i128;
          (len + i) as usize
        } else {
          return Err(InterpreterError::EvaluationError(
            "Part: index 0 is not valid".into(),
          ));
        };
        if idx < items.len() {
          Ok(vec![items[idx].clone()])
        } else {
          Ok(vec![])
        }
      } else {
        Ok(items.to_vec())
      }
    }
  }
}

/// Helper to generate combinations (subsets of size k)
fn generate_combinations(
  items: &[Expr],
  k: usize,
  start: usize,
  current: &mut Vec<Expr>,
  result: &mut Vec<Expr>,
) {
  if current.len() == k {
    result.push(Expr::List(current.clone()));
    return;
  }
  for i in start..items.len() {
    current.push(items[i].clone());
    generate_combinations(items, k, i + 1, current, result);
    current.pop();
  }
}

/// Subsequences[list] - all contiguous subsequences
/// Subsequences[list, {n}] - contiguous subsequences of length n
/// Subsequences[list, {nmin, nmax}] - lengths in range
/// Subsequences[{a, b, c}] => {{}, {a}, {b}, {c}, {a, b}, {b, c}, {a, b, c}}
pub fn subsequences_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Subsequences expects 1 or 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subsequences".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = items.len();

  // Determine min and max lengths
  let (min_len, max_len) = if args.len() == 2 {
    match &args[1] {
      Expr::List(spec) => {
        if spec.len() == 1 {
          // {n} - exactly length n
          if let Expr::Integer(k) = &spec[0] {
            let k = *k as usize;
            (k, k)
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsequences".to_string(),
              args: args.to_vec(),
            });
          }
        } else if spec.len() == 2 {
          // {nmin, nmax}
          if let (Expr::Integer(lo), Expr::Integer(hi)) = (&spec[0], &spec[1]) {
            (*lo as usize, *hi as usize)
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsequences".to_string(),
              args: args.to_vec(),
            });
          }
        } else {
          return Ok(Expr::FunctionCall {
            name: "Subsequences".to_string(),
            args: args.to_vec(),
          });
        }
      }
      Expr::Integer(_) | Expr::BigInteger(_) => {
        let k = expr_to_i128(&args[1]).unwrap_or(0) as usize;
        (k, k)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Subsequences".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    (0, n)
  };

  let mut result = Vec::new();
  for len in min_len..=max_len.min(n) {
    if len == 0 {
      result.push(Expr::List(vec![]));
    } else {
      for start in 0..=(n - len) {
        result.push(Expr::List(items[start..start + len].to_vec()));
      }
    }
  }
  Ok(Expr::List(result))
}
