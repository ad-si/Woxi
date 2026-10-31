#[allow(unused_imports)]
use super::utilities::*;
use super::*;

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
    // Rational/Complex are atoms — their internal args must not permute.
    Expr::FunctionCall {
      name,
      args: fc_args,
    } if !crate::functions::list_helpers_ast::sorting::is_atomic_arg(
      &args[0],
    ) =>
    {
      (fc_args.as_slice(), Some(name.as_str()))
    }
    _ => {
      crate::functions::list_helpers_ast::sorting::emit_nonatomic_normal_message(
        "Permutations",
        args,
      );
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
    Some(Expr::Identifier(s)) if s == "All" => Some((0..=len).collect()),
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

  // Refuse results that cannot be represented or built, as WL does:
  // ::fac when the length is at least 21! (checked without computing it),
  // ::len when the exact count is not a machine integer, ::toobig when it
  // is but the list would not fit in memory.
  let sizes: Vec<usize> = sizes
    .into_iter()
    .filter(|k| (0..=len).contains(k))
    .map(|k| k as usize)
    .collect();
  let mut multiplicities: Vec<usize> = Vec::new();
  let mut group_of: std::collections::HashMap<String, usize> =
    std::collections::HashMap::new();
  for item in items {
    let next = group_of.len();
    let g = *group_of.entry(expr_to_string(item)).or_insert(next);
    if g == multiplicities.len() {
      multiplicities.push(0);
    }
    multiplicities[g] += 1;
  }
  if multiplicities.len() >= 21 && sizes.iter().any(|&k| k >= 21) {
    crate::emit_message(&format!(
      "Permutations::fac: In {} there are at least 21 distinct elements in the input list, and the requested permutation lengths include one that is at least 21. The result cannot be computed because it has length at least 21 factorial, which is not a machine integer.",
      show(&original())
    ));
    return Ok(original());
  }
  let counts = multiset_permutation_counts(
    &multiplicities,
    sizes.iter().copied().max().unwrap_or(0),
  );
  let total: BigInt = sizes.iter().map(|&k| &counts[k]).sum();
  if total > BigInt::from(i64::MAX) {
    crate::emit_message(&format!(
      "Permutations::len: {} cannot be computed because its length is {}, which is not a machine integer.",
      show(&original()),
      total
    ));
    return Ok(original());
  }
  let nodes: BigInt = sizes
    .iter()
    .map(|&k| &counts[k] * BigInt::from(k + 1))
    .sum();
  if nodes > BigInt::from(MAX_PERMUTATIONS_NODES) {
    crate::emit_message(&format!(
      "Permutations::toobig: Insufficient memory available to evaluate {}.",
      show(&original())
    ));
    return Ok(original());
  }

  let n = items.len();
  let mut result: Vec<Expr> = Vec::with_capacity(total.try_into().unwrap_or(0));
  for k in sizes {
    generate_k_permutations(
      k,
      &mut vec![],
      &mut vec![false; n],
      items,
      &mut result,
    );
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

/// `Combinatorica\`UnrankPermutation[r, l]` / `Combinatorica\`UnrankPermutation[r, n]`
/// — the legacy Combinatorica package's permutation unranking: the
/// permutation of `l` (or of `Range[n]`) of rank `r` in lexicographic order,
/// counting from 0 (the list itself) and taking `r` modulo `Length[l]!`
/// (so `Length[l]! - 1`, or -1, is its reverse). Each factorial-number-system
/// digit of the rank selects how many of the still-unused elements, taken in
/// `l`'s own order, to skip before picking the next one.
pub fn combinatorica_unrank_permutation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use num_traits::ToPrimitive;

  let original = || unevaluated("Combinatorica`UnrankPermutation", args);
  if args.len() != 2 {
    return Ok(original());
  }

  let rank = match &args[0] {
    Expr::Integer(r) => BigInt::from(*r),
    Expr::BigInteger(r) => r.clone(),
    _ => return Ok(original()),
  };

  let items: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.to_vec(),
    Expr::Integer(n) if *n >= 0 => (1..=*n).map(Expr::Integer).collect(),
    _ => return Ok(original()),
  };

  let n = items.len();
  let mut n_factorial = BigInt::from(1u32);
  for k in 2..=n {
    n_factorial *= k as u64;
  }
  let rank = ((rank % &n_factorial) + &n_factorial) % &n_factorial;

  let mut remaining = items;
  let mut result = Vec::with_capacity(n);
  let mut rem_rank = rank;
  for i in 0..n {
    let remaining_len = n - i;
    let mut weight = BigInt::from(1u32);
    for k in 2..remaining_len {
      weight *= k as u64;
    }
    let quotient: BigInt = &rem_rank / &weight;
    let idx: usize = quotient.to_usize().unwrap_or(0);
    rem_rank -= &weight * BigInt::from(idx);
    result.push(remaining.remove(idx));
  }
  Ok(Expr::List(result.into()))
}

/// `Combinatorica\`Permutations[l]` / `Combinatorica\`Permutations[n]` — the
/// legacy Combinatorica package's `Permutations`, extended (unlike the
/// built-in of the same name) to accept a bare non-negative integer `n`
/// meaning `Range[n]`. For a list argument the two agree: both enumerate
/// permutations in the same lexicographic order (the order
/// `Combinatorica\`UnrankPermutation` already relies on), so a list simply
/// delegates to the built-in's algorithm. A non-list, non-integer argument
/// is left symbolic, matching that Combinatorica itself only defines the
/// function on that domain.
pub fn combinatorica_permutations_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("Combinatorica`Permutations", args);
  if args.len() != 1 {
    return Ok(original());
  }
  let items: Option<Vec<Expr>> = match &args[0] {
    Expr::List(items) => Some(items.to_vec()),
    Expr::Integer(n) if *n >= 0 => Some((1..=*n).map(Expr::Integer).collect()),
    _ => None,
  };
  let Some(items) = items else {
    return Ok(original());
  };
  permutations_ast(&[Expr::List(items.into())])
}

/// `Combinatorica\`NextPermutation[l]` — the permutation following `l` in
/// lexicographic (canonical) order; the last one wraps around to the sorted
/// list, as in Combinatorica.
pub fn combinatorica_next_permutation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use std::cmp::Ordering::Less;
  let [Expr::List(items)] = args else {
    return Ok(unevaluated("Combinatorica`NextPermutation", args));
  };
  let mut v: Vec<Expr> = items.to_vec();
  let cmp = super::sorting::canonical_cmp;
  // The rightmost ascent i, then the rightmost element above v[i].
  let Some(i) = (0..v.len().saturating_sub(1))
    .rev()
    .find(|&i| cmp(&v[i], &v[i + 1]) == Less)
  else {
    v.sort_by(cmp);
    return Ok(Expr::List(v.into()));
  };
  let j = (i + 1..v.len())
    .rev()
    .find(|&j| cmp(&v[i], &v[j]) == Less)
    .unwrap();
  v.swap(i, j);
  v[i + 1..].reverse();
  Ok(Expr::List(v.into()))
}

/// `Combinatorica\`Derangements[l]` / `Combinatorica\`Derangements[n]` — all
/// permutations of `l` (or of `Range[n]`) that leave no element in its
/// original position, in the same order `Combinatorica\`Permutations`
/// enumerates them.
pub fn combinatorica_derangements_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("Combinatorica`Derangements", args);
  if args.len() != 1 {
    return Ok(original());
  }
  let items: Option<Vec<Expr>> = match &args[0] {
    Expr::List(items) => Some(items.to_vec()),
    Expr::Integer(n) if *n >= 0 => Some((1..=*n).map(Expr::Integer).collect()),
    _ => None,
  };
  let Some(items) = items else {
    return Ok(original());
  };
  let all = permutations_ast(&[Expr::List(items.clone().into())])?;
  let perms: Vec<Expr> = match &all {
    Expr::List(perms) => perms.iter().cloned().collect(),
    other => {
      unreachable!(
        "permutations_ast always returns a List for a List argument, got {other:?}"
      )
    }
  };
  let kept: Vec<Expr> = perms
    .into_iter()
    .filter(|perm| match perm {
      Expr::List(elems) => elems
        .iter()
        .zip(items.iter())
        .all(|(a, b)| !crate::evaluator::pattern_matching::expr_equal(a, b)),
      _ => true,
    })
    .collect();
  Ok(Expr::List(kept.into()))
}

/// `Combinatorica\`SymmetricGroup[n]`, `AlternatingGroup[n]`,
/// `CyclicGroup[n]` and `DihedralGroup[n]` — the elements of the group as a
/// list of permutations of `Range[n]`. The symmetric group lists every
/// permutation in lexicographic order, the alternating group keeps the even
/// ones, the cyclic group the `n` left rotations of `Range[n]` and the
/// dihedral group those rotations followed by their reversals (without
/// repeating a permutation that already appeared).
pub fn combinatorica_group_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated(name, args);
  let n = match args {
    [Expr::Integer(n)] if *n >= 1 => *n as usize,
    _ => return Ok(original()),
  };
  let to_list = |perm: &[i128]| {
    Expr::List(perm.iter().map(|&k| Expr::Integer(k)).collect())
  };
  let identity: Vec<i128> = (1..=n as i128).collect();
  let rotations: Vec<Vec<i128>> = (0..n)
    .map(|k| {
      let mut r = identity.clone();
      r.rotate_left(k);
      r
    })
    .collect();
  let perms: Vec<Vec<i128>> = match name {
    "Combinatorica`CyclicGroup" => rotations,
    "Combinatorica`DihedralGroup" => {
      let mut out = rotations.clone();
      for r in rotations {
        let mut rev = r;
        rev.reverse();
        if !out.contains(&rev) {
          out.push(rev);
        }
      }
      out
    }
    _ => {
      let all = permutations_ast(&[to_list(&identity)])?;
      let Expr::List(all) = &all else {
        return Ok(original());
      };
      let as_ints: Vec<Vec<i128>> = all
        .iter()
        .filter_map(|p| match p {
          Expr::List(items) => items
            .iter()
            .map(|e| match e {
              Expr::Integer(k) => Some(*k),
              _ => None,
            })
            .collect(),
          _ => None,
        })
        .collect();
      if name == "Combinatorica`AlternatingGroup" {
        as_ints
          .into_iter()
          .filter(|p| {
            let inversions = (0..p.len())
              .flat_map(|i| (i + 1..p.len()).map(move |j| (i, j)))
              .filter(|&(i, j)| p[i] > p[j])
              .count();
            inversions % 2 == 0
          })
          .collect()
      } else {
        as_ints
      }
    }
  };
  Ok(Expr::List(perms.iter().map(|p| to_list(p)).collect()))
}

/// Largest `Permutations` result, counted in `Expr` nodes (each
/// permutation's elements plus its own `List`), that is built rather than
/// refused with ::toobig. WL's bound is the machine's free memory; an
/// owned `Expr` tree costs far more per element than a packed array, so
/// Woxi's is lower: `Permutations[Range[10]]` (~40M nodes) still builds,
/// `Permutations[Range[11]]` (~479M) is refused.
const MAX_PERMUTATIONS_NODES: u64 = 1 << 26;

/// `counts[k]` = number of distinct length-`k` permutations of a multiset
/// with the given element multiplicities, for `k` in `0..=kmax`. Adds one
/// group at a time: placing `j` copies of the new element among `i + j`
/// positions multiplies each length-`i` arrangement by `Binomial[i + j, j]`.
fn multiset_permutation_counts(
  multiplicities: &[usize],
  kmax: usize,
) -> Vec<BigInt> {
  let mut counts = vec![BigInt::from(1u32)];
  for &m in multiplicities {
    let new_len = (counts.len() - 1 + m).min(kmax) + 1;
    let mut next = vec![BigInt::ZERO; new_len];
    for (i, c) in counts.iter().enumerate() {
      let mut binom = BigInt::from(1u32);
      for j in 0..=m.min(new_len - 1 - i) {
        if j > 0 {
          binom = binom * (i + j) / j;
        }
        next[i + j] += c * &binom;
      }
    }
    counts = next;
  }
  counts.resize(kmax + 1, BigInt::ZERO);
  counts
}

/// Helper to generate k-permutations.
///
/// When the input contains duplicate elements, only distinct permutations
/// are emitted (matching Wolfram's `Permutations`). This is achieved by
/// skipping any item at the current recursion level that is structurally
/// equal to a previously tried item at the same level.
fn generate_k_permutations(
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
    let key = expr_to_string(&items[i]);
    if !seen_at_level.insert(key) {
      continue;
    }
    used[i] = true;
    current.push(i);
    generate_k_permutations(k, current, used, items, result);
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
    // Rational/Complex are atoms — their internal args must not enumerate.
    Expr::FunctionCall {
      name,
      args: fc_args,
    } if !crate::functions::list_helpers_ast::sorting::is_atomic_arg(
      &args[0],
    ) =>
    {
      (fc_args.as_slice(), Some(name.as_str()))
    }
    _ => {
      crate::functions::list_helpers_ast::sorting::emit_nonatomic_normal_message(
        "Subsets",
        args,
      );
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
    Some(Expr::Identifier(s)) if s == "All" => Some((0..=len).collect()),
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

  // Subsets come in size order, each size in lexicographic order of
  // positions. A take spec picks some of them by index; those are unranked
  // directly rather than enumerating them all (`Subsets[Range[32], {16},
  // {k}]` names one of 601080390).
  let layers: Vec<usize> = sizes
    .iter()
    .filter(|k| (0..=len).contains(*k))
    .map(|&k| k as usize)
    .collect();
  let layer_counts: Option<Vec<u128>> = layers
    .iter()
    .map(|&k| binomial_u128(items.len(), k))
    .collect();
  let total_count: Option<u128> = layer_counts
    .as_ref()
    .and_then(|c| c.iter().try_fold(0u128, |a, &b| a.checked_add(b)));
  let materialize = || {
    let mut all = Vec::new();
    for &k in &layers {
      generate_combinations(items, k, 0, &mut vec![], &mut all);
    }
    all
  };
  let nth_subset = |mut rank: u128| -> Expr {
    let counts = layer_counts.as_deref().unwrap_or(&[]);
    for (&k, &count) in layers.iter().zip(counts) {
      if rank < count {
        return Expr::List(unrank_combination(items, k, rank).into());
      }
      rank -= count;
    }
    Expr::List(Vec::new().into())
  };
  let mut result: Vec<Expr> = if args.get(2).is_some() && total_count.is_some()
  {
    Vec::new()
  } else {
    materialize()
  };

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
      TakeSpec::All => {
        if result.is_empty() {
          result = materialize();
        }
      }
      TakeSpec::Nothing => result.clear(),
      TakeSpec::Triple(m, n, s) => {
        let total = match total_count {
          Some(t) => i128::try_from(t).unwrap_or(i128::MAX),
          None => result.len() as i128,
        };
        let lazy = total_count.is_some();
        let resolve = |x: i128| if x < 0 { total + 1 + x } else { x };
        let (mr, nr) = (resolve(m), resolve(n));
        let mut taken: Vec<Expr> = Vec::new();
        let mut clipped = false;
        let mut i = mr;
        while (s > 0 && i <= nr) || (s < 0 && i >= nr) {
          if i >= 1 && i <= total {
            taken.push(if lazy {
              nth_subset((i - 1) as u128)
            } else {
              result[(i - 1) as usize].clone()
            });
          } else {
            clipped = true;
          }
          i += s;
        }
        if clipped {
          let spec_display = Expr::List(
            vec![Expr::Integer(m), Expr::Integer(n), Expr::Integer(s)].into(),
          );
          let two_arg = call("Subsets", vec![args[0].clone(), args[1].clone()]);
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

/// C(n, k) in u128, None on overflow.
fn binomial_u128(n: usize, k: usize) -> Option<u128> {
  if k > n {
    return Some(0);
  }
  let k = k.min(n - k);
  let mut acc: u128 = 1;
  for i in 0..k {
    // acc * (n - i) / (i + 1) stays an integer at every step.
    acc = acc.checked_mul((n - i) as u128)? / (i as u128 + 1);
  }
  Some(acc)
}

/// The size-`k` subset of `items` at 0-based `rank` in lexicographic order of
/// positions (the order `Subsets` lists them in).
fn unrank_combination(items: &[Expr], k: usize, mut rank: u128) -> Vec<Expr> {
  let n = items.len();
  let mut out = Vec::with_capacity(k);
  let mut next = 0;
  for slot in 0..k {
    let remaining = k - slot - 1;
    while next < n {
      let with_next =
        binomial_u128(n - next - 1, remaining).unwrap_or(u128::MAX);
      if rank < with_next {
        break;
      }
      rank -= with_next;
      next += 1;
    }
    if next >= n {
      break;
    }
    out.push(items[next].clone());
    next += 1;
  }
  out
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
  let Expr::List(items) = &args[0] else {
    return Ok(unevaluated("Subsequences", args));
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
      Some(name) => call(name.as_str(), children),
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
      for it in items {
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

  let Some(ops) = parse_grouping_ops(&args[1]) else {
    return Ok(unevaluated("Groupings", args));
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
      call(head, new_items)
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
  sorted_ops.sort_by_key(|op| std::cmp::Reverse(op.arity));

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
    let max_len = per_comp.iter().map(std::vec::Vec::len).max().unwrap_or(0);
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
  if child_groupings.iter().any(std::vec::Vec::is_empty) {
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
    let complement_idx = if l1 == r1 {
      None
    } else {
      split_pairs.iter().position(|&(l, r)| l == r1 && r == l1)
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

fn combinatorica_is_true(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(s) if s == "True")
}

/// `Combinatorica\`RandomPermutation[n]` / `Combinatorica\`RandomPermutation[l]`
/// — the legacy package's uniformly random permutation, returned as a plain
/// list (the built-in `RandomPermutation` returns a `Cycles` object instead).
/// An integer `n` means `Range[n]`; anything else is left symbolic.
pub fn combinatorica_random_permutation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use rand::seq::SliceRandom;

  let original = || unevaluated("Combinatorica`RandomPermutation", args);
  if args.len() != 1 {
    return Ok(original());
  }
  let mut items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    Expr::Integer(n) if *n >= 0 => (1..=*n).map(Expr::Integer).collect(),
    _ => return Ok(original()),
  };
  crate::with_rng(|rng| items.shuffle(rng));
  Ok(Expr::List(items.into()))
}

/// `Combinatorica\`StableMarriage[mpref, fpref]` — the man-optimal stable
/// matching (Gale–Shapley, men proposing). `mpref[[i]]` lists the women in
/// man `i`'s order of preference and `fpref[[j]]` the men in woman `j`'s;
/// the result's `i`-th entry is the woman engaged to man `i`. Left symbolic
/// unless both are square lists of permutations of `Range[n]`.
pub fn combinatorica_stable_marriage_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("Combinatorica`StableMarriage", args);
  if args.len() != 2 {
    return Ok(original());
  }
  let parse = |e: &Expr| -> Option<Vec<Vec<usize>>> {
    let Expr::List(rows) = e else { return None };
    let n = rows.len();
    let mut out = Vec::with_capacity(n);
    for row in rows {
      let Expr::List(items) = row else { return None };
      if items.len() != n {
        return None;
      }
      let mut seen = vec![false; n];
      let mut v = Vec::with_capacity(n);
      for it in items {
        let Expr::Integer(k) = it else { return None };
        if *k < 1 || *k as usize > n || seen[*k as usize - 1] {
          return None;
        }
        seen[*k as usize - 1] = true;
        v.push(*k as usize - 1);
      }
      out.push(v);
    }
    Some(out)
  };
  let (Some(mpref), Some(fpref)) = (parse(&args[0]), parse(&args[1])) else {
    return Ok(original());
  };
  let n = mpref.len();
  if fpref.len() != n {
    return Ok(original());
  }
  // rank[w][m]: position of man m in woman w's list (lower is better).
  let mut rank = vec![vec![0usize; n]; n];
  for (w, prefs) in fpref.iter().enumerate() {
    for (pos, &m) in prefs.iter().enumerate() {
      rank[w][m] = pos;
    }
  }
  let mut next = vec![0usize; n];
  let mut fiance: Vec<Option<usize>> = vec![None; n];
  let mut free: Vec<usize> = (0..n).rev().collect();
  while let Some(m) = free.pop() {
    let w = mpref[m][next[m]];
    next[m] += 1;
    match fiance[w] {
      None => fiance[w] = Some(m),
      Some(cur) if rank[w][m] < rank[w][cur] => {
        fiance[w] = Some(m);
        free.push(cur);
      }
      Some(_) => free.push(m),
    }
  }
  let mut wife = vec![0i128; n];
  for (w, m) in fiance.iter().enumerate() {
    wife[m.expect("every woman is matched")] = w as i128 + 1;
  }
  Ok(Expr::List(wife.into_iter().map(Expr::Integer).collect()))
}

/// `Combinatorica\`RandomKSubset[l, k]` — a uniformly random `k`-element
/// subset of `l` (or of `Range[l]` for an integer), kept in `l`'s own order.
/// Left symbolic unless `0 <= k <= Length[l]`.
pub fn combinatorica_random_k_subset_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use rand::seq::index::sample;

  let original = || unevaluated("Combinatorica`RandomKSubset", args);
  if args.len() != 2 {
    return Ok(original());
  }
  let items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    Expr::Integer(n) if *n >= 0 => (1..=*n).map(Expr::Integer).collect(),
    _ => return Ok(original()),
  };
  let k = match &args[1] {
    Expr::Integer(k) if *k >= 0 && (*k as usize) <= items.len() => *k as usize,
    _ => return Ok(original()),
  };
  let mut picked: Vec<usize> =
    crate::with_rng(|rng| sample(rng, items.len(), k).into_vec());
  picked.sort_unstable();
  Ok(Expr::List(
    picked.into_iter().map(|i| items[i].clone()).collect(),
  ))
}

/// `Combinatorica\`RandomSubset[l]` — a uniformly random subset of `l` (or
/// of `Range[l]` for an integer), kept in `l`'s own order.
pub fn combinatorica_random_subset_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let items: Vec<Expr> = match args {
    [Expr::List(items)] => items.to_vec(),
    [Expr::Integer(n)] if *n >= 0 => (1..=*n).map(Expr::Integer).collect(),
    _ => return Ok(unevaluated("Combinatorica`RandomSubset", args)),
  };
  let keep: Vec<bool> = crate::with_rng(|rng| {
    use rand::RngExt;
    (0..items.len()).map(|_| rng.random_bool(0.5)).collect()
  });
  Ok(Expr::List(
    items
      .into_iter()
      .zip(keep)
      .filter(|(_, k)| *k)
      .map(|(e, _)| e)
      .collect(),
  ))
}

/// `Combinatorica\`ToCycles[p]` — the cycles of the permutation `p`, each
/// started from its smallest unvisited point `i` and written
/// `{p[i], p[p[i]], …, i}`.
pub fn combinatorica_to_cycles_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("Combinatorica`ToCycles", args);
  let [Expr::List(items)] = args else {
    return Ok(original());
  };
  let n = items.len();
  let perm: Option<Vec<usize>> = items
    .iter()
    .map(|e| match e {
      Expr::Integer(k) if *k >= 1 && (*k as usize) <= n => {
        Some(*k as usize - 1)
      }
      _ => None,
    })
    .collect();
  let Some(perm) = perm else {
    return Ok(original());
  };
  let mut seen = vec![false; n];
  let mut cycles = Vec::new();
  for start in 0..n {
    if seen[start] {
      continue;
    }
    let mut cycle = Vec::new();
    let mut j = perm[start];
    loop {
      if seen[j] {
        return Ok(original());
      }
      seen[j] = true;
      cycle.push(Expr::Integer(j as i128 + 1));
      if j == start {
        break;
      }
      j = perm[j];
    }
    cycles.push(Expr::List(cycle.into()));
  }
  Ok(Expr::List(cycles.into()))
}

/// `Combinatorica\`Orbits[g, x]` — the orbits of the elements of `x` under
/// the permutation group `g` acting by `Permute` (`e[[p]]`): each orbit is
/// the images of the first element left, and the elements are taken in the
/// order `Complement` leaves them. `OrbitRepresentatives` is the first of
/// each orbit.
pub fn combinatorica_orbits_ast(
  args: &[Expr],
  representatives: bool,
) -> Result<Expr, InterpreterError> {
  let head = if representatives {
    "Combinatorica`OrbitRepresentatives"
  } else {
    "Combinatorica`Orbits"
  };
  let [Expr::List(group), Expr::List(x)] = args else {
    return Ok(unevaluated(head, args));
  };
  let mut rest: Vec<Expr> = x.to_vec();
  let mut out = Vec::new();
  while !rest.is_empty() {
    let first = rest[0].clone();
    let mut orbit = Vec::with_capacity(group.len());
    for g in group {
      orbit.push(crate::evaluator::evaluate_function_call_ast(
        "Part",
        &[first.clone(), g.clone()],
      )?);
    }
    let remaining = crate::evaluator::evaluate_function_call_ast(
      "Complement",
      &[Expr::List(rest.into()), Expr::List(orbit.clone().into())],
    )?;
    let Expr::List(ref remaining) = remaining else {
      return Ok(unevaluated(head, args));
    };
    rest = remaining.to_vec();
    out.push(if representatives {
      first
    } else {
      Expr::List(orbit.into())
    });
  }
  Ok(Expr::List(out.into()))
}

/// `Combinatorica\`KSetPartitions[l, k]` — the partitions of `l` (or of
/// `Range[l]` for an integer) into `k` nonempty blocks. Partitions with the
/// last element alone come first, then those with it added to each block of
/// a partition of the remaining elements.
pub fn combinatorica_k_set_partitions_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  fn partitions(n: usize, k: usize) -> Vec<Vec<Vec<usize>>> {
    if n == 0 {
      return if k == 0 { vec![vec![]] } else { vec![] };
    }
    if k == 0 || k > n {
      return vec![];
    }
    let mut out = Vec::new();
    for mut p in partitions(n - 1, k - 1) {
      p.push(vec![n - 1]);
      out.push(p);
    }
    for p in partitions(n - 1, k) {
      for i in 0..p.len() {
        let mut q = p.clone();
        q[i].push(n - 1);
        out.push(q);
      }
    }
    out
  }

  let original = || unevaluated("Combinatorica`KSetPartitions", args);
  let items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    Expr::Integer(n) if (0..=20).contains(n) => {
      (1..=*n).map(Expr::Integer).collect()
    }
    _ => return Ok(original()),
  };
  let k = match &args[1] {
    Expr::Integer(k) if *k >= 0 => *k as usize,
    _ => return Ok(original()),
  };
  Ok(Expr::List(
    partitions(items.len(), k)
      .into_iter()
      .map(|p| {
        Expr::List(
          p.into_iter()
            .map(|b| {
              Expr::List(b.into_iter().map(|i| items[i].clone()).collect())
            })
            .collect(),
        )
      })
      .collect(),
  ))
}

/// `Combinatorica\`Josephus[n, m]`: for `n` people in a circle with every
/// `m`-th one eliminated, the round in which each person goes (the inverse
/// of the elimination order).
pub fn combinatorica_josephus_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (Expr::Integer(n), Expr::Integer(m)) = (&args[0], &args[1]) else {
    return Ok(unevaluated("Combinatorica`Josephus", args));
  };
  if *n < 0 || *m < 1 || *n > 1_000_000 {
    return Ok(unevaluated("Combinatorica`Josephus", args));
  }
  let n = *n as usize;
  let mut circle: Vec<usize> = (0..n).collect();
  let mut round = vec![0i128; n];
  let mut at = 0usize;
  for k in 1..=n {
    at = ((at as i128 + m - 1) % circle.len() as i128) as usize;
    round[circle.remove(at)] = k as i128;
    if !circle.is_empty() {
      at %= circle.len();
    }
  }
  Ok(Expr::List(
    round
      .into_iter()
      .map(Expr::Integer)
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// `Combinatorica\`Backtrack[space, partialQ, solutionQ, flag]` — the legacy
/// package's depth-first search over `space`, a list of candidate lists (one
/// per position). `partialQ` prunes partial solutions, `solutionQ` accepts
/// complete ones. With `flag` `All` it returns every solution; otherwise
/// (`One`, the default) the first one, or `{}` if there is none. As in the
/// package, the first position is never tested with `partialQ`. Both
/// predicates default to always-true.
pub fn combinatorica_backtrack_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::function_application::apply_function_to_arg;

  let original = || unevaluated("Combinatorica`Backtrack", args);
  if args.is_empty() || args.len() > 4 {
    return Ok(original());
  }
  let Expr::List(space) = &args[0] else {
    return Ok(original());
  };
  let partial_q = args.get(1);
  let solution_q = args.get(2);
  let find_all = matches!(
    args.get(3),
    Some(Expr::Identifier(s)) if s == "All" || s == "Combinatorica`All"
  );

  let mut candidates: Vec<&[Expr]> = Vec::with_capacity(space.len());
  for level in space {
    let Expr::List(level) = level else {
      return Ok(original());
    };
    candidates.push(level.as_slice());
  }
  let n = candidates.len();
  if n == 0 || candidates[0].is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  let test =
    |f: Option<&Expr>, partial: &[Expr]| -> Result<bool, InterpreterError> {
      match f {
        None => Ok(true),
        Some(f) => Ok(combinatorica_is_true(&apply_function_to_arg(
          f,
          &Expr::List(partial.to_vec().into()),
        )?)),
      }
    };

  // `index[i]` is the 1-based candidate chosen at level `i`; `v` is the
  // 1-based level being extended (0 ends the search, `n + 1` means a
  // complete candidate is ready).
  let mut index = vec![0usize; n];
  index[0] = 1;
  let mut v: usize = 2;
  let mut all: Vec<Expr> = Vec::new();
  let mut first: Option<Expr> = None;
  while v > 0 {
    if v <= n {
      let mut done = false;
      while !done && index[v - 1] < candidates[v - 1].len() {
        index[v - 1] += 1;
        let partial: Vec<Expr> = (0..v)
          .map(|i| candidates[i][index[i] - 1].clone())
          .collect();
        done = test(partial_q, &partial)?;
      }
      if done {
        v += 1;
      } else {
        index[v - 1] = 0;
        v -= 1;
      }
    }
    if v > n {
      let solution: Vec<Expr> = (0..n)
        .map(|i| candidates[i][index[i] - 1].clone())
        .collect();
      if test(solution_q, &solution)? {
        if find_all {
          all.push(Expr::List(solution.into()));
        } else {
          first = Some(Expr::List(solution.into()));
          break;
        }
      }
      v -= 1;
    }
  }

  if find_all {
    Ok(Expr::List(all.into()))
  } else {
    Ok(first.unwrap_or_else(|| Expr::List(vec![].into())))
  }
}
