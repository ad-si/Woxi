#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Tally: count occurrences of each element.
/// Tally[{a, b, a, c, b, a}] -> {{a, 3}, {b, 2}, {c, 1}}
pub fn tally_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  // On an association, tally the values.
  if let Expr::Association(pairs) = list {
    let values: Vec<Expr> = pairs.iter().map(|(_, v)| v.clone()).collect();
    return tally_ast(&Expr::List(values.into()));
  }
  let items = match list {
    Expr::List(items) => items,
    _ => {
      let call = Expr::FunctionCall {
        name: "Tally".to_string(),
        args: vec![list.clone()].into(),
      };
      crate::emit_message(&format!(
        "Tally::list: List expected at position 1 in {}.",
        crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
      ));
      return Ok(call);
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, (Expr, i128)> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if let Some((_, count)) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, (item.clone(), 1));
    }
  }

  let pairs: Vec<Expr> = order
    .into_iter()
    .map(|k| {
      let (expr, count) = counts.remove(&k).unwrap();
      Expr::List(vec![expr, Expr::Integer(count)].into())
    })
    .collect();

  Ok(Expr::List(pairs.into()))
}

/// AST-based Tally with a custom equivalence test.
/// Tally[list, test] groups elements using `test[a, b]` to decide
/// whether `b` joins the group represented by `a` (the first element
/// seen for that group). Groups are reported in first-seen order.
pub fn tally_with_test_ast(
  list: &Expr,
  test: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tally".to_string(),
        args: vec![list.clone(), test.clone()].into(),
      });
    }
  };

  // Representatives keep insertion order; parallel counts vector.
  let mut reps: Vec<Expr> = Vec::new();
  let mut counts: Vec<i128> = Vec::new();

  'outer: for item in items {
    for (i, rep) in reps.iter().enumerate() {
      let result = apply_func_to_two_args(test, rep, item)?;
      if matches!(result, Expr::Identifier(ref s) if s == "True") {
        counts[i] += 1;
        continue 'outer;
      }
    }
    reps.push(item.clone());
    counts.push(1);
  }

  let pairs: Vec<Expr> = reps
    .into_iter()
    .zip(counts)
    .map(|(rep, count)| Expr::List(vec![rep, Expr::Integer(count)].into()))
    .collect();

  Ok(Expr::List(pairs.into()))
}

///// Counts[list] - Returns association of distinct elements with their counts
/// Counts[{a, b, a, c, b, a}] -> <|a -> 3, b -> 2, c -> 1|>
pub fn counts_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  // On an association, count the values.
  if let Expr::Association(pairs) = list {
    let values: Vec<Expr> = pairs.iter().map(|(_, v)| v.clone()).collect();
    return counts_ast(&Expr::List(values.into()));
  }
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Counts".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, (Expr, i128)> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if let Some((_, count)) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, (item.clone(), 1));
    }
  }

  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let (expr, count) = counts.remove(&k).unwrap();
      (expr, Expr::Integer(count))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based DeleteDuplicates: remove duplicate elements.
/// DeleteDuplicates[{a, b, a, c, b}] -> {a, b, c}
/// DeleteDuplicates[list, test] uses `test[a, b]` to decide if `b`
/// collapses into an already-kept representative `a`.
pub fn delete_duplicates_ast(
  list: &Expr,
  test: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  // On an association, deduplicate by value, keeping the first key->value pair
  // for each distinct value.
  if let Expr::Association(pairs) = list {
    if let Some(test_fn) = test {
      let mut kept: Vec<(Expr, Expr)> = Vec::new();
      'outer: for (k, v) in pairs.iter() {
        for (_, rep_v) in &kept {
          let r = apply_func_to_two_args(test_fn, rep_v, v)?;
          if matches!(r, Expr::Identifier(ref s) if s == "True") {
            continue 'outer;
          }
        }
        kept.push((k.clone(), v.clone()));
      }
      return Ok(Expr::Association(kept));
    }
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    let mut result: Vec<(Expr, Expr)> = Vec::new();
    for (k, v) in pairs.iter() {
      if seen.insert(crate::syntax::expr_to_string(v)) {
        result.push((k.clone(), v.clone()));
      }
    }
    return Ok(Expr::Association(result));
  }

  // DeleteDuplicates works on any expression with parts, preserving the head
  // (DeleteDuplicates[f[1, 2, 2, 3]] -> f[1, 2, 3]).
  let (items, head_name): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    _ => {
      let mut call_args = vec![list.clone()];
      if let Some(t) = test {
        call_args.push(t.clone());
      }
      // An atomic argument is invalid: emit ::normal, matching WL.
      if is_atomic_arg(list) {
        emit_nonatomic_normal_message("DeleteDuplicates", &call_args);
      }
      return Ok(Expr::FunctionCall {
        name: "DeleteDuplicates".to_string(),
        args: call_args.into(),
      });
    }
  };

  let kept: Vec<Expr> = if let Some(test_fn) = test {
    // Custom equivalence: keep the first element from each equivalence
    // class, in first-seen order.
    let mut reps: Vec<Expr> = Vec::new();
    'outer: for item in items {
      for rep in &reps {
        let r = apply_func_to_two_args(test_fn, rep, item)?;
        if matches!(r, Expr::Identifier(ref s) if s == "True") {
          continue 'outer;
        }
      }
      reps.push(item.clone());
    }
    reps
  } else {
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    let mut result = Vec::new();
    for item in items {
      if seen.insert(crate::syntax::expr_to_string(item)) {
        result.push(item.clone());
      }
    }
    result
  };

  // Preserve the original head, then evaluate the rebuilt expression so a
  // head with its own rules reduces; inert heads stay symbolic.
  match head_name {
    Some(name) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name,
        args: kept.into(),
      })
    }
    None => Ok(Expr::List(kept.into())),
  }
}

/// AST-based Union: combine lists and remove duplicates.
///
/// Supports `Union[l1, l2, ..., SameTest -> f]`. When `SameTest` is
/// specified, elements are combined and sorted first, then the result is
/// deduplicated by keeping the first occurrence of each equivalence class
/// under `f[#1, #2]`.
/// Split Union/Intersection/Complement arguments into subject element
/// slices, their common head and an optional SameTest function. Atomic
/// subjects emit ::normal and mismatched heads emit ::heads; in both
/// cases the unevaluated call is returned as the Err value.
#[allow(clippy::type_complexity)]
fn collect_set_subjects<'a>(
  fname: &str,
  args: &'a [Expr],
) -> Result<(Vec<&'a [Expr]>, Option<&'a str>, Option<&'a Expr>), Expr> {
  let unevaluated = || Expr::FunctionCall {
    name: fname.to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  let mut same_test: Option<&Expr> = None;
  let mut slices: Vec<&[Expr]> = Vec::new();
  let mut heads: Vec<(usize, Option<&str>)> = Vec::new();
  for (i, a) in args.iter().enumerate() {
    match a {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "SameTest") =>
      {
        same_test = Some(replacement.as_ref());
      }
      Expr::List(items) => {
        slices.push(items.as_slice());
        heads.push((i, None));
      }
      Expr::FunctionCall {
        name,
        args: fc_args,
      } => {
        slices.push(fc_args.as_slice());
        heads.push((i, Some(name.as_str())));
      }
      _ => {
        crate::emit_message(&format!(
          "{}::normal: Nonatomic expression expected at position {} in {}.",
          fname,
          i + 1,
          show(&unevaluated())
        ));
        return Err(unevaluated());
      }
    }
  }
  if let Some((first_pos, first_head)) = heads.first().copied() {
    for (i, h) in heads.iter().skip(1) {
      if *h != first_head {
        crate::emit_message(&format!(
          "{}::heads: Heads {} and {} at positions {} and {} are expected to be the same.",
          fname,
          h.unwrap_or("List"),
          first_head.unwrap_or("List"),
          i + 1,
          first_pos + 1
        ));
        return Err(unevaluated());
      }
    }
  }
  let head = heads.first().and_then(|(_, h)| *h);
  Ok((slices, head, same_test))
}

/// Sort canonically via compare_exprs (1 = first argument precedes).
fn sort_canonical(items: &mut [Expr]) {
  items.sort_by(|a, b| match compare_exprs(a, b) {
    n if n > 0 => std::cmp::Ordering::Less,
    n if n < 0 => std::cmp::Ordering::Greater,
    _ => std::cmp::Ordering::Equal,
  });
}

pub fn union_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  use std::collections::HashSet;

  let (slices, head, same_test) = match collect_set_subjects("Union", lists) {
    Ok(t) => t,
    Err(unevaluated) => return Ok(unevaluated),
  };
  let wrap = |v: Vec<Expr>| -> Expr {
    match head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: v.into(),
      },
      None => Expr::List(v.into()),
    }
  };

  let mut result = Vec::new();

  if same_test.is_none() {
    let mut seen: HashSet<String> = HashSet::new();
    for items in &slices {
      for item in *items {
        let key_str = crate::syntax::expr_to_string(item);
        if seen.insert(key_str) {
          result.push(item.clone());
        }
      }
    }
  } else {
    // Collect every element from every list, without plain
    // deduplication -- equivalence is decided later by SameTest.
    for items in &slices {
      result.extend(items.iter().cloned());
    }
  }

  // Union sorts its result in Mathematica
  sort_canonical(&mut result);

  if let Some(test) = same_test {
    // Deduplicate by equivalence. Each sorted element is compared
    // against every already-kept representative; the first
    // representative that `test[item, rep]` is True for absorbs it.
    // The new candidate goes on the *left* of `test` to match
    // wolframscript: `Union[{1,2,3,4}, SameTest->Greater]` yields
    // `{1}` because `Greater[2,1]`, `Greater[3,1]`, `Greater[4,1]`
    // all hold, while `Less` keeps every element since
    // `Less[k, 1]` is False for every later `k`.
    let mut reps: Vec<Expr> = Vec::new();
    'outer: for item in result.into_iter() {
      for rep in &reps {
        let r = apply_func_to_two_args(test, &item, rep)?;
        if matches!(r, Expr::Identifier(ref s) if s == "True") {
          continue 'outer;
        }
      }
      reps.push(item);
    }
    return Ok(wrap(reps));
  }

  Ok(wrap(result))
}

/// AST-based Intersection: find common elements.
pub fn intersection_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // SameTest -> fn makes two elements "equal" iff `fn[a, b]` evaluates
  // to True. Matches wolframscript's
  // `Intersection[{1,2,3}, {2,3,4}, SameTest->Less]` → `{3}` (Less[1,2]
  // is True so 1 collapses with 2; Less[2,3] is True so 2 collapses with
  // 3; net result is whatever remains after pairwise SameTest dedup).
  let (slices, head, same_test) =
    match collect_set_subjects("Intersection", lists) {
      Ok(t) => t,
      Err(unevaluated) => return Ok(unevaluated),
    };
  let wrap = |v: Vec<Expr>| -> Expr {
    match head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: v.into(),
      },
      None => Expr::List(v.into()),
    }
  };

  if let Some(test) = same_test {
    return intersection_with_same_test(&slices, test).map(wrap);
  }

  use std::collections::HashSet;

  let Some(first_items) = slices.first() else {
    return Ok(wrap(Vec::new()));
  };

  let mut common: HashSet<String> = first_items
    .iter()
    .map(crate::syntax::expr_to_string)
    .collect();

  // Intersect with each subsequent list
  for items in slices.iter().skip(1) {
    let list_set: HashSet<String> =
      items.iter().map(crate::syntax::expr_to_string).collect();
    common = common.intersection(&list_set).cloned().collect();
  }

  // Collect matching elements and sort canonically (like Mathematica)
  let mut result: Vec<Expr> = first_items
    .iter()
    .filter(|item| common.contains(&crate::syntax::expr_to_string(item)))
    .cloned()
    .collect();
  sort_canonical(&mut result);
  result.dedup_by(|a, b| {
    crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
  });

  Ok(wrap(result))
}

/// Intersection with a custom SameTest function. Two elements `a` and `b`
/// from different lists count as equal iff `test[a, b]` evaluates to True.
/// Within the result, elements are deduplicated pairwise via the same
/// test. Element order in the result follows the first list.
fn intersection_with_same_test(
  slices: &[&[Expr]],
  test: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  let first_items: Vec<Expr> = match slices.first() {
    Some(items) => items.to_vec(),
    None => return Ok(Vec::new()),
  };

  let test_eq = |a: &Expr, b: &Expr| -> bool {
    let call = Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![test.clone(), Expr::List(vec![a.clone(), b.clone()].into())]
        .into(),
    };
    matches!(
      crate::evaluator::evaluate_expr_to_expr(&call).ok(),
      Some(Expr::Identifier(ref s)) if s == "True"
    )
  };

  // Filter first list down to elements that have a test-match in every
  // subsequent list.
  let mut filtered: Vec<Expr> = Vec::new();
  for item in &first_items {
    let keep = slices
      .iter()
      .skip(1)
      .all(|other| other.iter().any(|o| test_eq(item, o)));
    if keep {
      filtered.push(item.clone());
    }
  }

  // Pairwise dedup the result using the same test. Keep the LAST element
  // of each equivalence class so `{1,2,3}` with `Less` collapses left to
  // right and leaves `3` (Less[1,2] = True drops 1, Less[2,3] = True drops 2).
  let mut deduped: Vec<Expr> = Vec::new();
  for item in &filtered {
    let mut replaced = false;
    for existing in deduped.iter_mut() {
      if test_eq(existing, item) {
        *existing = item.clone();
        replaced = true;
        break;
      }
    }
    if !replaced {
      deduped.push(item.clone());
    }
  }

  Ok(deduped)
}

/// AST-based Complement: elements in first list not in others.
pub fn complement_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  use std::collections::HashSet;

  let (slices, head, _same_test) =
    match collect_set_subjects("Complement", lists) {
      Ok(t) => t,
      Err(unevaluated) => return Ok(unevaluated),
    };

  let Some(first_items) = slices.first() else {
    return Ok(Expr::List(vec![].into()));
  };

  // Get elements to exclude from all lists after the first
  let mut exclude: HashSet<String> = HashSet::new();
  for items in slices.iter().skip(1) {
    for item in *items {
      exclude.insert(crate::syntax::expr_to_string(item));
    }
  }

  // Filter first list, also remove duplicates and sort canonically.
  // (Accent collation — å/ä/ö after z — lives in wolfram_string_order,
  // so `Complement[Alphabet["Swedish"], Alphabet["English"]]` lands on
  // `{å, ä, ö}` rather than the codepoint-sorted `{ä, å, ö}`.)
  let mut seen = HashSet::new();
  let mut result: Vec<Expr> = first_items
    .iter()
    .filter(|item| {
      let s = crate::syntax::expr_to_string(item);
      !exclude.contains(&s) && seen.insert(s)
    })
    .cloned()
    .collect();
  sort_canonical(&mut result);

  match head {
    Some(name) => Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: result.into(),
    }),
    None => Ok(Expr::List(result.into())),
  }
}

/// AST-based DeleteDuplicatesBy: remove duplicates by key function.
pub fn delete_duplicates_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteDuplicatesBy".to_string(),
        args: vec![list.clone(), func.clone()].into(),
      });
    }
  };

  use std::collections::HashSet;
  let mut seen: HashSet<String> = HashSet::new();
  let mut result = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if seen.insert(key_str) {
      result.push(item.clone());
    }
  }

  Ok(Expr::List(result.into()))
}

/// DeleteAdjacentDuplicates[list] - removes consecutive duplicate elements
pub fn delete_adjacent_duplicates_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DeleteAdjacentDuplicates expects exactly 1 argument".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteAdjacentDuplicates".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  let mut result = vec![items[0].clone()];
  for item in items.iter().skip(1) {
    if crate::syntax::expr_to_string(item)
      != crate::syntax::expr_to_string(result.last().unwrap())
    {
      result.push(item.clone());
    }
  }
  Ok(Expr::List(result.into()))
}

/// Commonest[list] - returns the most common element(s)
/// Commonest[list, n] - returns the n most common elements
pub fn commonest_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Commonest expects 1 or 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Commonest".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Default form Commonest[list] is equivalent to picking the single
  // most common count tier (all elements tied at the maximum count).
  let mut tier_mode = false;
  let n = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n >= 1 => *n as usize,
      // UpTo[n] caps at min(n, distinct_count).
      Expr::FunctionCall { name, args: a }
        if name == "UpTo" && a.len() == 1 =>
      {
        match &a[0] {
          Expr::Integer(n) if *n >= 0 => *n as usize,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Commonest".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Commonest".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    tier_mode = true;
    usize::MAX
  };

  // Count occurrences, recording first-appearance index for stable ordering.
  // counts: Vec<(key, item, count, first_appearance)>
  let mut counts: Vec<(String, &Expr, usize, usize)> = Vec::new();
  for (idx, item) in items.iter().enumerate() {
    let key = crate::syntax::expr_to_string(item);
    if let Some(entry) = counts.iter_mut().find(|(k, _, _, _)| k == &key) {
      entry.2 += 1;
    } else {
      counts.push((key, item, 1, idx));
    }
  }

  // Sort by (-count, first_appearance) so the highest-count elements come
  // first and ties break by where they first appeared in the input.
  let mut ranked = counts.clone();
  ranked.sort_by(|a, b| b.2.cmp(&a.2).then(a.3.cmp(&b.3)));

  // Default form returns just the top count tier.
  let limit = if tier_mode {
    let max_count = ranked.first().map(|t| t.2).unwrap_or(0);
    ranked.iter().take_while(|t| t.2 == max_count).count()
  } else {
    n.min(ranked.len())
  };

  // Mark the top `limit` elements (by first-appearance index) as chosen,
  // then emit them in input order.
  let mut chosen: std::collections::HashSet<usize> =
    std::collections::HashSet::new();
  for (_, _, _, fa) in ranked.iter().take(limit) {
    chosen.insert(*fa);
  }
  let mut result: Vec<Expr> = counts
    .iter()
    .filter(|(_, _, _, fa)| chosen.contains(fa))
    .map(|(_, item, _, _)| (*item).clone())
    .collect();
  // `counts` was already built in first-appearance order, so `result`
  // is too — but we kept the iteration explicit for clarity.
  let _ = &mut result; // satisfy clippy if ever flipped
  Ok(Expr::List(result.into()))
}

/// AST-based SymmetricDifference: elements in an odd number of the input lists.
/// Result is sorted and deduplicated.
pub fn symmetric_difference_ast(
  lists: &[Expr],
) -> Result<Expr, InterpreterError> {
  use std::collections::HashMap;

  if lists.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Count how many lists contain each element (dedup within each list)
  let mut membership_count: HashMap<String, (usize, Expr)> = HashMap::new();

  for list in lists {
    let items = match list {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "SymmetricDifference".to_string(),
          args: lists.to_vec().into(),
        });
      }
    };

    // Deduplicate within this list
    let mut seen_in_list = std::collections::HashSet::new();
    for item in items {
      let key = crate::syntax::expr_to_string(item);
      if seen_in_list.insert(key.clone()) {
        membership_count
          .entry(key)
          .and_modify(|(count, _)| *count += 1)
          .or_insert((1, item.clone()));
      }
    }
  }

  // Keep elements that appear in an odd number of lists
  let mut result: Vec<Expr> = membership_count
    .into_values()
    .filter(|(count, _)| count % 2 == 1)
    .map(|(_, expr)| expr)
    .collect();

  // Sort canonically (matching Complement/Union behavior)
  result.sort_by(|a, b| {
    crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
  });

  Ok(Expr::List(result.into()))
}
