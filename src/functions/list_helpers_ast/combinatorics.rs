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
      // {kmin, kmax} means permutations of lengths kmin..=kmax
      Expr::List(spec) if spec.len() == 2 => {
        let kmin = expr_to_i128(&spec[0]).unwrap_or(0).max(0) as usize;
        let kmax = expr_to_i128(&spec[1]).unwrap_or(n as i128) as usize;
        let kmax = kmax.min(n);
        let mut result = Vec::new();
        let indices: Vec<usize> = (0..n).collect();
        if kmin <= kmax {
          for k in kmin..=kmax {
            generate_k_permutations(
              &indices,
              k,
              &mut vec![],
              &mut vec![false; n],
              &items,
              &mut result,
            );
          }
        }
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

/// Helper to generate k-permutations.
///
/// When the input contains duplicate elements, only distinct permutations
/// are emitted (matching Wolfram's `Permutations`). This is achieved by
/// skipping any item at the current recursion level that is structurally
/// equal to a previously tried item at the same level.
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
  // Track which item "values" we've already used at this recursion level
  // so that {1, 1, 2} doesn't produce {1, 1, 2} twice.
  let mut seen_at_level: std::collections::HashSet<String> =
    std::collections::HashSet::new();
  for i in 0..items.len() {
    if used[i] {
      continue;
    }
    let key = crate::syntax::expr_to_string(&items[i]);
    if !seen_at_level.insert(key) {
      continue;
    }
    used[i] = true;
    current.push(i);
    generate_k_permutations(_indices, k, current, used, items, result);
    current.pop();
    used[i] = false;
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

// ─── Groupings ──────────────────────────────────────────────────────

/// Groupings[n, k] or Groupings[list, k]
/// Generate all ways to group elements using an operator of arity k.
pub fn groupings_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Groupings expects 2 arguments".into(),
    ));
  }

  let arity = match &args[1] {
    Expr::Integer(k) if *k >= 2 => *k as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Groupings".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let elements: Vec<Expr> = match &args[0] {
    Expr::Integer(n) if *n >= 1 => (1..=*n).map(Expr::Integer).collect(),
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Groupings".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = elements.len();
  let results = groupings_recursive(&elements, arity);
  Ok(Expr::List(
    results
      .into_iter()
      .map(|g| {
        if n == 1 {
          // Single element: return just the element (not wrapped in list)
          g
        } else {
          g
        }
      })
      .collect(),
  ))
}

/// Check if n elements can form a valid k-ary tree
fn can_group(n: usize, k: usize) -> bool {
  if n == 1 {
    return true;
  }
  if n < k {
    return false;
  }
  // n = 1 + m*(k-1) for some m >= 1
  (n - 1).is_multiple_of(k - 1)
}

/// Generate all binary groupings of a contiguous slice of elements
/// Uses interleaved ordering to match Wolfram Language output
fn groupings_binary(elements: &[Expr]) -> Vec<Expr> {
  let n = elements.len();
  if n == 1 {
    return vec![elements[0].clone()];
  }
  if n == 2 {
    return vec![Expr::List(elements.to_vec())];
  }

  let mut results = Vec::new();

  // Collect all (left_size, right_size) pairs and their groupings
  // For binary trees: split into left (i elements) and right (n-i elements)
  // Valid sizes: i where can_group(i,2) && can_group(n-i,2)
  // Iterate from largest left to smallest to match Wolfram ordering
  let mut split_pairs: Vec<(usize, usize)> = Vec::new();
  for i in (1..n).rev() {
    if can_group(i, 2) && can_group(n - i, 2) {
      split_pairs.push((i, n - i));
    }
  }

  // Group symmetric pairs for interleaving
  // Process pairs: (large, small) interleaved with (small, large)
  let mut processed = vec![false; split_pairs.len()];
  for idx in 0..split_pairs.len() {
    if processed[idx] {
      continue;
    }
    let (l1, r1) = split_pairs[idx];

    // Find complement pair (r1, l1) if different
    let complement_idx = if l1 != r1 {
      split_pairs.iter().position(|&(l, r)| l == r1 && r == l1)
    } else {
      None
    };

    let left_groupings_1 = groupings_binary(&elements[..l1]);
    let right_groupings_1 = groupings_binary(&elements[l1..]);

    if let Some(c_idx) = complement_idx {
      processed[c_idx] = true;
      let (l2, _r2) = split_pairs[c_idx];
      let left_groupings_2 = groupings_binary(&elements[..l2]);
      let right_groupings_2 = groupings_binary(&elements[l2..]);

      // Build full results for both splits
      let mut results_1 = Vec::new();
      for lg in &left_groupings_1 {
        for rg in &right_groupings_1 {
          results_1.push(Expr::List(vec![lg.clone(), rg.clone()]));
        }
      }
      let mut results_2 = Vec::new();
      for lg in &left_groupings_2 {
        for rg in &right_groupings_2 {
          results_2.push(Expr::List(vec![lg.clone(), rg.clone()]));
        }
      }

      // Interleave results
      let max_len = results_1.len().max(results_2.len());
      for i in 0..max_len {
        if i < results_1.len() {
          results.push(results_1[i].clone());
        }
        if i < results_2.len() {
          results.push(results_2[i].clone());
        }
      }
    } else {
      // Self-symmetric split (l1 == r1) or no complement
      for lg in &left_groupings_1 {
        for rg in &right_groupings_1 {
          results.push(Expr::List(vec![lg.clone(), rg.clone()]));
        }
      }
    }
    processed[idx] = true;
  }

  results
}

/// Generate all k-ary groupings of a contiguous slice of elements
fn groupings_recursive(elements: &[Expr], k: usize) -> Vec<Expr> {
  let n = elements.len();
  if n == 1 {
    return vec![elements[0].clone()];
  }
  if k == 2 {
    return groupings_binary(elements);
  }
  if n < k || !can_group(n, k) {
    return vec![];
  }
  if n == k {
    return vec![Expr::List(elements.to_vec())];
  }

  // For k > 2: split into k groups
  let mut results = Vec::new();
  generate_splits(elements, k, 0, &mut Vec::new(), &mut results, k);
  results
}

/// Generate all ways to split elements[start..] into `remaining` groups (for k > 2)
fn generate_splits(
  elements: &[Expr],
  k: usize,
  start: usize,
  current_groups: &mut Vec<Vec<Expr>>,
  results: &mut Vec<Expr>,
  remaining: usize,
) {
  let n = elements.len();
  if remaining == 0 {
    if start == n {
      let mut group_results: Vec<Vec<Expr>> = current_groups
        .iter()
        .map(|g| groupings_recursive(g, k))
        .collect();
      let mut product = vec![Vec::new()];
      for group in &mut group_results {
        let mut new_product = Vec::new();
        for existing in &product {
          for item in group.iter() {
            let mut combo = existing.clone();
            combo.push(item.clone());
            new_product.push(combo);
          }
        }
        product = new_product;
      }
      for combo in product {
        results.push(Expr::List(combo));
      }
    }
    return;
  }

  let remaining_elements = n - start;
  if remaining == 1 {
    let group: Vec<Expr> = elements[start..].to_vec();
    if can_group(group.len(), k) {
      current_groups.push(group);
      generate_splits(elements, k, n, current_groups, results, 0);
      current_groups.pop();
    }
    return;
  }

  let max_size = remaining_elements - (remaining - 1);
  for size in (1..=max_size).rev() {
    if can_group(size, k) {
      let group: Vec<Expr> = elements[start..start + size].to_vec();
      current_groups.push(group);
      generate_splits(
        elements,
        k,
        start + size,
        current_groups,
        results,
        remaining - 1,
      );
      current_groups.pop();
    }
  }
}
