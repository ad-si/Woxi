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
        args: args.to_vec().into(),
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
          return Ok(Expr::List(vec![].into()));
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
        return Ok(Expr::List(result.into()));
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
        return Ok(Expr::List(result.into()));
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
        return Ok(Expr::List(result.into()));
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
  Ok(Expr::List(result.into()))
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
    result.push(Expr::List(perm.into()));
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
        args: args.to_vec().into(),
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
              args: args.to_vec().into(),
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
              args: args.to_vec().into(),
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
              args: args.to_vec().into(),
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
              args: args.to_vec().into(),
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

  Ok(Expr::List(result.into()))
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
    result.push(Expr::List(current.clone().into()));
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
        args: args.to_vec().into(),
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
              args: args.to_vec().into(),
            });
          }
        } else if spec.len() == 2 {
          // {nmin, nmax}
          if let (Expr::Integer(lo), Expr::Integer(hi)) = (&spec[0], &spec[1]) {
            (*lo as usize, *hi as usize)
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsequences".to_string(),
              args: args.to_vec().into(),
            });
          }
        } else {
          return Ok(Expr::FunctionCall {
            name: "Subsequences".to_string(),
            args: args.to_vec().into(),
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
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    (0, n)
  };

  let mut result = Vec::new();
  for len in min_len..=max_len.min(n) {
    if len == 0 {
      result.push(Expr::List(vec![].into()));
    } else {
      for start in 0..=(n - len) {
        result.push(Expr::List(items[start..start + len].to_vec().into()));
      }
    }
  }
  Ok(Expr::List(result.into()))
}

// ─── Groupings ──────────────────────────────────────────────────────

/// A grouping operator: either an anonymous arity (head = `List`) or a
/// named head `f` with the specified arity.
#[derive(Clone, Debug)]
struct GroupingOp {
  /// Head used when wrapping children. `None` means use `Expr::List`
  /// (the integer-arity form `Groupings[list, k]`).
  head: Option<String>,
  arity: usize,
}

impl GroupingOp {
  fn wrap(&self, children: Vec<Expr>) -> Expr {
    match &self.head {
      Some(name) => Expr::FunctionCall {
        name: name.clone(),
        args: children.into(),
      },
      None => Expr::List(children.into()),
    }
  }
}

/// Parse the second argument of `Groupings` into a list of operators.
/// Returns `None` if the argument doesn't match any recognised form.
fn parse_grouping_ops(arg: &Expr) -> Option<Vec<GroupingOp>> {
  match arg {
    // Plain integer k -> single anonymous arity-k operator.
    Expr::Integer(k) if *k >= 2 => Some(vec![GroupingOp {
      head: None,
      arity: *k as usize,
    }]),
    // Single Rule `f -> k`.
    Expr::Rule {
      pattern,
      replacement,
    } => parse_single_rule(pattern, replacement).map(|op| vec![op]),
    // List of Rules `{f -> k, ...}` or singleton `{f -> k}`.
    Expr::List(items) if !items.is_empty() => {
      let mut ops = Vec::with_capacity(items.len());
      for it in items.iter() {
        if let Expr::Rule {
          pattern,
          replacement,
        } = it
        {
          ops.push(parse_single_rule(pattern, replacement)?);
        } else {
          return None;
        }
      }
      Some(ops)
    }
    _ => None,
  }
}

fn parse_single_rule(pattern: &Expr, replacement: &Expr) -> Option<GroupingOp> {
  let head = match pattern {
    Expr::Identifier(s) => s.clone(),
    _ => return None,
  };
  let arity = match replacement {
    Expr::Integer(k) if *k >= 2 => *k as usize,
    _ => return None,
  };
  Some(GroupingOp {
    head: Some(head),
    arity,
  })
}

/// Groupings[n, k], Groupings[list, k], Groupings[list, f -> k],
/// Groupings[list, {f -> k, ...}].
pub fn groupings_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Groupings expects 2 arguments".into(),
    ));
  }

  let ops = match parse_grouping_ops(&args[1]) {
    Some(ops) => ops,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Groupings".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let elements: Vec<Expr> = match &args[0] {
    Expr::Integer(n) if *n >= 1 => (1..=*n).map(Expr::Integer).collect(),
    Expr::List(items) => items.to_vec(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Groupings".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Single anonymous arity: use the existing optimised path so the
  // canonical output of `Groupings[list, k]` is byte-identical.
  if ops.len() == 1 && ops[0].head.is_none() {
    let arity = ops[0].arity;
    let results = groupings_recursive(&elements, arity);
    return Ok(Expr::List(results.into()));
  }

  // Single named operator: reuse the anonymous path, then rewrap each
  // List as the named head.
  if ops.len() == 1
    && let Some(head) = &ops[0].head
  {
    let arity = ops[0].arity;
    let results = groupings_recursive(&elements, arity);
    let rewrapped: Vec<Expr> =
      results.into_iter().map(|t| rewrap_lists(&t, head)).collect();
    return Ok(Expr::List(rewrapped.into()));
  }

  // General multi-operator case.
  Ok(Expr::List(groupings_multi(&elements, &ops).into()))
}

/// Recursively replace every `Expr::List` with `head[...]`. Leaves all
/// other AST nodes untouched. Used to lift the integer-arity output to a
/// named-operator output.
fn rewrap_lists(expr: &Expr, head: &str) -> Expr {
  match expr {
    Expr::List(items) => {
      let new_items: Vec<Expr> =
        items.iter().map(|i| rewrap_lists(i, head)).collect();
      Expr::FunctionCall {
        name: head.to_string(),
        args: new_items.into(),
      }
    }
    _ => expr.clone(),
  }
}

/// Generate all groupings of `elements` using any of the operators in
/// `ops` at each internal node. Returns one entry per distinct tree.
///
/// Operators are tried at the root in descending-arity order so that
/// higher-arity ops (used at most once in shallow trees) emit before
/// lower-arity ones; this matches wolframscript's canonical output
/// regardless of the order rules were written in.
fn groupings_multi(elements: &[Expr], ops: &[GroupingOp]) -> Vec<Expr> {
  let n = elements.len();
  if n == 1 {
    return vec![elements[0].clone()];
  }

  let mut sorted_ops: Vec<&GroupingOp> = ops.iter().collect();
  sorted_ops.sort_by(|a, b| b.arity.cmp(&a.arity));

  let mut results = Vec::new();
  for op in &sorted_ops {
    if n < op.arity {
      continue;
    }
    let op_trees = groupings_op_at_root(elements, op, ops);
    results.extend(op_trees);
  }
  results
}

/// All trees that have `op` at the root, with subtrees built from any
/// operator in `all_ops`.
///
/// The split compositions are grouped by their partition (sorted
/// descending) and the partitions are visited by descending max element;
/// within each partition the compositions are taken in decreasing-lex
/// order. Trees from compositions sharing a partition are then
/// interleaved shape-major / composition-minor: the i-th tree from each
/// composition is emitted before the (i+1)-th. This reproduces
/// wolframscript's canonical ordering on the documented examples.
fn groupings_op_at_root(
  elements: &[Expr],
  op: &GroupingOp,
  all_ops: &[GroupingOp],
) -> Vec<Expr> {
  let n = elements.len();
  let k = op.arity;

  let mut compositions = generate_compositions(n, k);
  // Sort: by partition (max-element descending), then by composition
  // (lex descending).
  compositions.sort_by(|a, b| {
    let pa = partition_key(a);
    let pb = partition_key(b);
    pb.cmp(&pa).then_with(|| b.cmp(a))
  });

  // Group consecutive compositions sharing the same partition.
  let mut groups: Vec<Vec<&Vec<usize>>> = Vec::new();
  for c in &compositions {
    let pk = partition_key(c);
    if let Some(last) = groups.last_mut()
      && partition_key(last[0]) == pk
    {
      last.push(c);
    } else {
      groups.push(vec![c]);
    }
  }

  let mut results = Vec::new();
  for group in groups {
    // For each composition in the group, build all its child-cross-product
    // trees in row-major order.
    let mut per_comp: Vec<Vec<Expr>> = Vec::with_capacity(group.len());
    for comp in &group {
      per_comp.push(build_trees_for_composition(elements, comp, op, all_ops));
    }
    // Interleave: take the i-th tree from each composition (in order)
    // before the (i+1)-th. Lengths may differ if children of compositions
    // in the same partition produce different numbers of inner shapes —
    // skip empties.
    let max_len = per_comp.iter().map(|v| v.len()).max().unwrap_or(0);
    for i in 0..max_len {
      for trees in &per_comp {
        if i < trees.len() {
          results.push(trees[i].clone());
        }
      }
    }
  }

  results
}

/// All trees for a single composition: cross product of the children's
/// groupings, ordered row-major over the child positions.
fn build_trees_for_composition(
  elements: &[Expr],
  comp: &[usize],
  op: &GroupingOp,
  all_ops: &[GroupingOp],
) -> Vec<Expr> {
  // Children's groupings in slice order.
  let mut start = 0;
  let mut child_groupings: Vec<Vec<Expr>> = Vec::with_capacity(comp.len());
  for &sz in comp {
    let slice = &elements[start..start + sz];
    child_groupings.push(groupings_multi(slice, all_ops));
    start += sz;
  }
  // If any child has no valid groupings, this composition contributes
  // nothing.
  if child_groupings.iter().any(|v| v.is_empty()) {
    return Vec::new();
  }

  // Row-major cross product.
  let mut acc: Vec<Vec<Expr>> = vec![Vec::new()];
  for child_list in &child_groupings {
    let mut next: Vec<Vec<Expr>> = Vec::with_capacity(acc.len() * child_list.len());
    for prefix in &acc {
      for child in child_list {
        let mut new_prefix = prefix.clone();
        new_prefix.push(child.clone());
        next.push(new_prefix);
      }
    }
    acc = next;
  }
  acc.into_iter().map(|kids| op.wrap(kids)).collect()
}

/// The "partition" of a composition: its parts sorted descending.
fn partition_key(comp: &[usize]) -> Vec<usize> {
  let mut v = comp.to_vec();
  v.sort_unstable_by(|a, b| b.cmp(a));
  v
}

/// All compositions of `n` into `k` positive parts (each ≥ 1), in lex
/// order.
fn generate_compositions(n: usize, k: usize) -> Vec<Vec<usize>> {
  let mut out = Vec::new();
  let mut cur = vec![0usize; k];
  fn recurse(
    cur: &mut Vec<usize>,
    idx: usize,
    remaining: usize,
    k: usize,
    out: &mut Vec<Vec<usize>>,
  ) {
    if idx == k - 1 {
      if remaining >= 1 {
        cur[idx] = remaining;
        out.push(cur.clone());
      }
      return;
    }
    let max = remaining.saturating_sub(k - 1 - idx);
    for v in 1..=max {
      cur[idx] = v;
      recurse(cur, idx + 1, remaining - v, k, out);
    }
  }
  recurse(&mut cur, 0, n, k, &mut out);
  out
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
    return vec![Expr::List(elements.to_vec().into())];
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
          results_1.push(Expr::List(vec![lg.clone(), rg.clone()].into()));
        }
      }
      let mut results_2 = Vec::new();
      for lg in &left_groupings_2 {
        for rg in &right_groupings_2 {
          results_2.push(Expr::List(vec![lg.clone(), rg.clone()].into()));
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
          results.push(Expr::List(vec![lg.clone(), rg.clone()].into()));
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
    return vec![Expr::List(elements.to_vec().into())];
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
        results.push(Expr::List(combo.into()));
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
