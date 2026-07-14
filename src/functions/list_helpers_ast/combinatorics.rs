#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;
use crate::syntax::unevaluated;

/// Unified Permutations: `Permutations[list]` (full-length only),
/// `Permutations[list, n]` (lengths 0..n), `Permutations[list, {n}]`,
/// `{nmin, nmax}` (nmax may be Infinity), `{nmin, nmax, dn}` (dn
/// nonzero, possibly negative), All and Infinity. Duplicate elements
/// yield distinct permutations only; general heads keep their head.
/// Atoms emit ::normal and invalid specs emit ::nninfseq.
pub fn permutations_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("Permutations", args);
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  let (items, head): (&[Expr], Option<&str>) = match &args[0] {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      crate::emit_message(&format!(
        "Permutations::normal: Nonatomic expression expected at position 1 in {}.",
        show(&original())
      ));
      return Ok(original());
    }
  };
  let len = items.len() as i128;

  let strict_int = |e: &Expr| -> Option<i128> {
    match e {
      Expr::Integer(n) => Some(*n),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i128()
      }
      _ => None,
    }
  };
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };
  let nonneg = |e: &Expr| strict_int(e).filter(|n| *n >= 0);
  let nonneg_or_inf = |e: &Expr| {
    if is_infinity(e) {
      Some(i128::MAX)
    } else {
      nonneg(e)
    }
  };

  let sizes: Option<Vec<i128>> = match args.get(1) {
    None => Some(vec![len]),
    Some(e) if matches!(e, Expr::Identifier(s) if s == "All") => {
      Some((0..=len).collect())
    }
    Some(e) if is_infinity(e) => Some((0..=len).collect()),
    Some(Expr::List(spec)) if spec.len() == 1 => {
      nonneg(&spec[0]).map(|n| vec![n])
    }
    Some(Expr::List(spec)) if spec.len() == 2 => {
      match (nonneg(&spec[0]), nonneg_or_inf(&spec[1])) {
        (Some(lo), Some(hi)) => Some((lo..=hi.min(len)).collect()),
        _ => None,
      }
    }
    Some(Expr::List(spec)) if spec.len() == 3 => {
      match (
        nonneg(&spec[0]),
        nonneg_or_inf(&spec[1]),
        strict_int(&spec[2]).filter(|s| *s != 0),
      ) {
        (Some(lo), Some(hi), Some(dn)) => {
          let mut seq = Vec::new();
          let mut k = lo;
          let hi_eff = if dn > 0 { hi.min(len.max(lo)) } else { hi };
          while (dn > 0 && k <= hi_eff) || (dn < 0 && k >= hi_eff) {
            seq.push(k);
            k += dn;
          }
          Some(seq)
        }
        _ => None,
      }
    }
    Some(e) => nonneg(e).map(|n| (0..=n.min(len)).collect()),
  };
  let Some(sizes) = sizes else {
    crate::emit_message(&format!(
      "Permutations::nninfseq: Position 2 of {} must be All, Infinity, nmax, {{nmin}}, {{nmin, nmax}} or {{nmin, nmax, dn}}, where nmin is a non-negative integer, nmax is non-negative integer or Infinity and dn is a nonzero integer.",
      show(&original())
    ));
    return Ok(original());
  };

  let n = items.len();
  let mut result: Vec<Expr> = Vec::new();
  let indices: Vec<usize> = (0..n).collect();
  for k in sizes {
    if (0..=len).contains(&k) {
      generate_k_permutations(
        &indices,
        k as usize,
        &mut vec![],
        &mut vec![false; n],
        items,
        &mut result,
      );
    }
  }

  if let Some(h) = head {
    result = result
      .into_iter()
      .map(|perm| match &perm {
        Expr::List(elems) => Expr::FunctionCall {
          name: h.to_string(),
          args: elems.clone(),
        },
        _ => perm,
      })
      .collect();
  }
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

/// Unified Subsets: `Subsets[list]`, `Subsets[list, nspec]`, and
/// `Subsets[list, nspec, s]`. nspec may be All, Infinity, nmax, {n},
/// {nmin, nmax} (nmax may be Infinity) or {nmin, nmax, dn} (dn nonzero,
/// possibly negative). s follows Take semantics (negative counts from
/// the end) with a ::take warning when the sequence is clipped. General
/// heads keep their head; atoms emit ::normal; invalid specs emit
/// ::nninfseq (position 2) or ::seq (position 3).
pub fn subsets_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("Subsets", args);
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  let (items, head): (&[Expr], Option<&str>) = match &args[0] {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      crate::emit_message(&format!(
        "Subsets::normal: Nonatomic expression expected at position 1 in {}.",
        show(&original())
      ));
      return Ok(original());
    }
  };
  let len = items.len() as i128;

  let strict_int = |e: &Expr| -> Option<i128> {
    match e {
      Expr::Integer(n) => Some(*n),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i128()
      }
      _ => None,
    }
  };
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };
  let nonneg = |e: &Expr| strict_int(e).filter(|n| *n >= 0);
  let nonneg_or_inf = |e: &Expr| {
    if is_infinity(e) {
      Some(i128::MAX)
    } else {
      nonneg(e)
    }
  };

  // The size sequence requested by nspec.
  let sizes: Option<Vec<i128>> = match args.get(1) {
    None => Some((0..=len).collect()),
    Some(e) if matches!(e, Expr::Identifier(s) if s == "All") => {
      Some((0..=len).collect())
    }
    Some(e) if is_infinity(e) => Some((0..=len).collect()),
    Some(Expr::List(spec)) if spec.len() == 1 => {
      nonneg(&spec[0]).map(|n| vec![n])
    }
    Some(Expr::List(spec)) if spec.len() == 2 => {
      match (nonneg(&spec[0]), nonneg_or_inf(&spec[1])) {
        (Some(lo), Some(hi)) => Some((lo..=hi.min(len)).collect()),
        _ => None,
      }
    }
    Some(Expr::List(spec)) if spec.len() == 3 => {
      match (
        nonneg(&spec[0]),
        nonneg_or_inf(&spec[1]),
        strict_int(&spec[2]).filter(|s| *s != 0),
      ) {
        (Some(lo), Some(hi), Some(dn)) => {
          let mut seq = Vec::new();
          let mut k = lo;
          let hi_eff = if dn > 0 { hi.min(len.max(lo)) } else { hi };
          while (dn > 0 && k <= hi_eff) || (dn < 0 && k >= hi_eff) {
            seq.push(k);
            k += dn;
          }
          Some(seq)
        }
        _ => None,
      }
    }
    Some(e) => nonneg(e).map(|n| (0..=n.min(len)).collect()),
  };
  let Some(sizes) = sizes else {
    crate::emit_message(&format!(
      "Subsets::nninfseq: Position 2 of {} must be All, Infinity, nmax, {{nmin}}, {{nmin, nmax}} or {{nmin, nmax, dn}}, where nmin is a non-negative integer, nmax is non-negative integer or Infinity and dn is a nonzero integer.",
      show(&original())
    ));
    return Ok(original());
  };

  let mut result: Vec<Expr> = Vec::new();
  for k in sizes {
    if (0..=len).contains(&k) {
      generate_combinations(items, k as usize, 0, &mut vec![], &mut result);
    }
  }

  // The take argument follows Take semantics, clipping with a warning.
  if let Some(take) = args.get(2) {
    enum TakeSpec {
      All,
      Nothing,
      Triple(i128, i128, i128),
    }
    let parsed = match take {
      Expr::Identifier(s) if s == "All" => Some(TakeSpec::All),
      Expr::Identifier(s) if s == "None" => Some(TakeSpec::Nothing),
      Expr::List(spec) if spec.len() == 1 => {
        strict_int(&spec[0]).map(|m| TakeSpec::Triple(m, m, 1))
      }
      Expr::List(spec) if spec.len() == 2 => {
        match (strict_int(&spec[0]), strict_int(&spec[1])) {
          (Some(m), Some(n)) => Some(TakeSpec::Triple(m, n, 1)),
          _ => None,
        }
      }
      Expr::List(spec) if spec.len() == 3 => {
        match (
          strict_int(&spec[0]),
          strict_int(&spec[1]),
          strict_int(&spec[2]).filter(|s| *s != 0),
        ) {
          (Some(m), Some(n), Some(s)) => Some(TakeSpec::Triple(m, n, s)),
          _ => None,
        }
      }
      other => strict_int(other).map(|m| {
        if m >= 0 {
          TakeSpec::Triple(1, m, 1)
        } else {
          TakeSpec::Triple(m, -1, 1)
        }
      }),
    };
    let Some(parsed) = parsed else {
      crate::emit_message(&format!(
        "Subsets::seq: Position 3 of {} must be All, None, m, {{m}}, {{m, n}} or {{m, n, s}}, where m and n are integers, and s is a nonzero integer.",
        show(&original())
      ));
      return Ok(original());
    };
    match parsed {
      TakeSpec::All => {}
      TakeSpec::Nothing => result.clear(),
      TakeSpec::Triple(m, n, s) => {
        let total = result.len() as i128;
        let resolve = |x: i128| if x < 0 { total + 1 + x } else { x };
        let (mr, nr) = (resolve(m), resolve(n));
        let mut taken: Vec<Expr> = Vec::new();
        let mut clipped = false;
        let mut i = mr;
        while (s > 0 && i <= nr) || (s < 0 && i >= nr) {
          if i >= 1 && i <= total {
            taken.push(result[(i - 1) as usize].clone());
          } else {
            clipped = true;
          }
          i += s;
        }
        if clipped {
          let spec_display = Expr::List(
            vec![Expr::Integer(m), Expr::Integer(n), Expr::Integer(s)].into(),
          );
          let two_arg = Expr::FunctionCall {
            name: "Subsets".to_string(),
            args: vec![args[0].clone(), args[1].clone()].into(),
          };
          crate::emit_message(&format!(
            "Subsets::take: Warning: not all elements were found when attempting to take the sequence {} from {}, which has length {}.",
            show(&spec_display),
            show(&two_arg),
            total
          ));
        }
        result = taken;
      }
    }
  }

  // General heads keep their head on every subset.
  if let Some(h) = head {
    result = result
      .into_iter()
      .map(|subset| match &subset {
        Expr::List(elems) => Expr::FunctionCall {
          name: h.to_string(),
          args: elems.clone(),
        },
        _ => subset,
      })
      .collect();
  }
  Ok(Expr::List(result.into()))
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
/// Subsequences[list, n] - contiguous subsequences of length 0 through n
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
      return Ok(unevaluated("Subsequences", args));
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
            return Ok(unevaluated("Subsequences", args));
          }
        } else if spec.len() == 2 {
          // {nmin, nmax}
          if let (Expr::Integer(lo), Expr::Integer(hi)) = (&spec[0], &spec[1]) {
            (*lo as usize, *hi as usize)
          } else {
            return Ok(unevaluated("Subsequences", args));
          }
        } else {
          return Ok(unevaluated("Subsequences", args));
        }
      }
      Expr::Integer(_) | Expr::BigInteger(_) => {
        // Subsequences[list, n] gives lengths 0 through n, not exactly n.
        let k = expr_to_i128(&args[1]).unwrap_or(0) as usize;
        (0, k)
      }
      _ => {
        return Ok(unevaluated("Subsequences", args));
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
      return Ok(unevaluated("Groupings", args));
    }
  };

  let elements: Vec<Expr> = match &args[0] {
    Expr::Integer(n) if *n >= 1 => (1..=*n).map(Expr::Integer).collect(),
    Expr::List(items) => items.to_vec(),
    _ => {
      return Ok(unevaluated("Groupings", args));
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
    let rewrapped: Vec<Expr> = results
      .into_iter()
      .map(|t| rewrap_lists(&t, head))
      .collect();
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
    let mut next: Vec<Vec<Expr>> =
      Vec::with_capacity(acc.len() * child_list.len());
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
