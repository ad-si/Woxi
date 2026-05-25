#[allow(unused_imports)]
use super::*;
use crate::functions::list_helpers_ast;

/// Parse the `m` argument of NestWhile[f, x, test, m, ...]. Returns `All` for
/// the symbol `All`, `Last(n)` for a positive integer, and `None` otherwise.
fn parse_nest_while_m(expr: &Expr) -> Option<list_helpers_ast::NestWhileM> {
  match expr {
    Expr::Identifier(s) if s == "All" => {
      Some(list_helpers_ast::NestWhileM::All)
    }
    Expr::Integer(n) if *n >= 1 => {
      Some(list_helpers_ast::NestWhileM::Last(*n as usize))
    }
    _ => None,
  }
}

/// Check recursively whether an expression contains pattern elements (Blank, Pattern, etc.)
fn has_pattern_element(expr: &Expr) -> bool {
  match expr {
    Expr::Pattern { .. }
    | Expr::PatternOptional { .. }
    | Expr::PatternTest { .. } => true,
    Expr::FunctionCall { name, args } => {
      matches!(
        name.as_str(),
        "Blank"
          | "BlankSequence"
          | "BlankNullSequence"
          | "Pattern"
          | "Alternatives"
          | "PatternTest"
          | "Condition"
          | "Repeated"
          | "RepeatedNull"
          | "Except"
      ) || args.iter().any(has_pattern_element)
    }
    Expr::List(items) => items.iter().any(has_pattern_element),
    _ => false,
  }
}

/// Check if a pattern contains sequence-matching elements (BlankSequence, BlankNullSequence,
/// Repeated, RepeatedNull) that can match variable numbers of list elements.
fn has_sequence_pattern(expr: &Expr) -> bool {
  match expr {
    Expr::Pattern { blank_type, .. } => *blank_type >= 2,
    Expr::PatternTest { blank_type, .. } => *blank_type >= 2,
    Expr::FunctionCall { name, .. } => matches!(
      name.as_str(),
      "BlankSequence" | "BlankNullSequence" | "Repeated" | "RepeatedNull"
    ),
    _ => false,
  }
}

/// Parse `ExcludedForms -> {pat1, pat2, ...}` into the list of patterns.
/// Returns `None` if the argument is not an `ExcludedForms` rule.
fn parse_excluded_forms(arg: &Expr) -> Option<Vec<Expr>> {
  let (lhs, rhs) = match arg {
    Expr::Rule {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref()),
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return None,
  };
  match lhs {
    Expr::Identifier(s) if s == "ExcludedForms" => {}
    _ => return None,
  }
  match rhs {
    Expr::List(items) => Some(items.to_vec()),
    other => Some(vec![other.clone()]),
  }
}

/// Splice the children of `expr` at the given position vectors. Positions are
/// 1-based, may be negative (counted from the end), and apply at the level of
/// the position's last index. Used by FlattenAt.
fn flatten_at_positions(expr: &Expr, positions: &[Vec<i128>]) -> Expr {
  let children: &[Expr] = match expr {
    Expr::List(items) => items,
    Expr::FunctionCall { args, .. } => args,
    _ => return expr.clone(),
  };
  let len = children.len() as i128;
  let mut groups: std::collections::HashMap<usize, Vec<Vec<i128>>> =
    Default::default();
  let mut flatten_here: std::collections::HashSet<usize> = Default::default();
  for pos in positions {
    if pos.is_empty() {
      continue;
    }
    let first = pos[0];
    let idx = if first < 0 { len + first + 1 } else { first };
    if idx < 1 || idx > len {
      continue;
    }
    let i = idx as usize;
    if pos.len() == 1 {
      flatten_here.insert(i);
    } else {
      groups.entry(i).or_default().push(pos[1..].to_vec());
    }
  }
  let mut result: Vec<Expr> = Vec::new();
  for (i, child) in children.iter().enumerate() {
    let idx = i + 1;
    let new_child = if let Some(deeper) = groups.get(&idx) {
      flatten_at_positions(child, deeper)
    } else {
      child.clone()
    };
    if flatten_here.contains(&idx) {
      match &new_child {
        Expr::List(items) => result.extend(items.iter().cloned()),
        Expr::FunctionCall { args, .. } => result.extend(args.iter().cloned()),
        _ => result.push(new_child),
      }
    } else {
      result.push(new_child);
    }
  }
  match expr {
    Expr::List(_) => Expr::List(result.into()),
    Expr::FunctionCall { name, .. } => Expr::FunctionCall {
      name: name.clone(),
      args: result.into(),
    },
    _ => expr.clone(),
  }
}

pub fn dispatch_list_operations(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Map" | "ParallelMap" if args.len() == 2 => {
      return Some(list_helpers_ast::map_ast(&args[0], &args[1]));
    }
    "Map" if args.len() == 3 => {
      return Some(list_helpers_ast::map_with_level_ast(
        &args[0], &args[1], &args[2],
      ));
    }
    "MapAll" if args.len() == 2 => {
      return Some(map_all_ast(&args[0], &args[1]));
    }
    "MapAt" if args.len() == 3 => {
      return Some(list_helpers_ast::map_at_ast(&args[0], &args[1], &args[2]));
    }
    "SelectFirst" if args.len() >= 2 && args.len() <= 3 => {
      return Some(list_helpers_ast::select_first_ast(args));
    }
    "DuplicateFreeQ" if args.len() == 1 => {
      if let Expr::List(items) = &args[0] {
        let mut seen = std::collections::HashSet::new();
        for item in items {
          let s = expr_to_string(item);
          if !seen.insert(s) {
            return Some(Ok(Expr::Identifier("False".to_string())));
          }
        }
        return Some(Ok(Expr::Identifier("True".to_string())));
      }
    }
    "TakeList" if args.len() == 2 => {
      // Pull the children of args[0] and remember its head so each sublist
      // can be wrapped in the same head (List or any other symbol).
      let (head, items): (Option<String>, Vec<Expr>) = match &args[0] {
        Expr::List(xs) => (None, xs.to_vec()),
        Expr::FunctionCall { name, args: xs } => {
          (Some(name.clone()), xs.to_vec())
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "TakeList".to_string(),
            args: args.to_vec().into(),
          }));
        }
      };
      let Expr::List(specs) = &args[1] else {
        return Some(Ok(Expr::FunctionCall {
          name: "TakeList".to_string(),
          args: args.to_vec().into(),
        }));
      };
      let wrap = |slice: Vec<Expr>| -> Expr {
        match &head {
          None => Expr::List(slice.into()),
          Some(h) => Expr::FunctionCall {
            name: h.clone(),
            args: slice.into(),
          },
        }
      };
      // Walk a (start, end) window over `items`, consuming front or back
      // depending on the sign / form of each spec.
      let mut start: usize = 0;
      let mut end: usize = items.len();
      let mut result: Vec<Expr> = Vec::with_capacity(specs.len());
      for spec in specs.iter() {
        let remaining = end - start;
        match spec {
          Expr::Integer(n) if *n >= 0 => {
            let n = *n as usize;
            if n > remaining {
              crate::emit_message(&format!(
                "TakeList::take: Cannot take {} elements from a list of length {}.",
                n, remaining
              ));
              return Some(Ok(Expr::FunctionCall {
                name: "TakeList".to_string(),
                args: args.to_vec().into(),
              }));
            }
            let chunk: Vec<Expr> = items[start..start + n].to_vec();
            start += n;
            result.push(wrap(chunk));
          }
          Expr::Integer(n) => {
            // n < 0: take last |n| of the remaining slice
            let k = (-*n) as usize;
            if k > remaining {
              crate::emit_message(&format!(
                "TakeList::take: Cannot take {} elements from a list of length {}.",
                k, remaining
              ));
              return Some(Ok(Expr::FunctionCall {
                name: "TakeList".to_string(),
                args: args.to_vec().into(),
              }));
            }
            let chunk: Vec<Expr> = items[end - k..end].to_vec();
            end -= k;
            result.push(wrap(chunk));
          }
          Expr::Identifier(s) if s == "All" => {
            let chunk: Vec<Expr> = items[start..end].to_vec();
            start = end;
            result.push(wrap(chunk));
          }
          Expr::FunctionCall {
            name: upto,
            args: uargs,
          } if upto == "UpTo" && uargs.len() == 1 => {
            let Some(m) = (match &uargs[0] {
              Expr::Integer(n) if *n >= 0 => Some(*n as usize),
              _ => None,
            }) else {
              return Some(Ok(Expr::FunctionCall {
                name: "TakeList".to_string(),
                args: args.to_vec().into(),
              }));
            };
            let take = m.min(remaining);
            let chunk: Vec<Expr> = items[start..start + take].to_vec();
            start += take;
            result.push(wrap(chunk));
          }
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "TakeList".to_string(),
              args: args.to_vec().into(),
            }));
          }
        }
      }
      return Some(Ok(Expr::List(result.into())));
    }
    "FlattenAt" if args.len() == 1 => {
      // Operator form FlattenAt[pos] — return unevaluated for currying.
      return Some(Ok(Expr::FunctionCall {
        name: "FlattenAt".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "FlattenAt" if args.len() == 2 => {
      let unevaluated = || {
        Ok(Expr::FunctionCall {
          name: "FlattenAt".to_string(),
          args: args.to_vec().into(),
        })
      };
      // Parse second arg into a list of position vectors.
      let positions: Vec<Vec<i128>> = match &args[1] {
        Expr::Integer(n) => vec![vec![*n]],
        Expr::List(pos_list) => {
          if pos_list.is_empty() {
            return Some(Ok(args[0].clone()));
          }
          let all_ints = pos_list.iter().all(|p| matches!(p, Expr::Integer(_)));
          let all_lists = pos_list.iter().all(|p| matches!(p, Expr::List(_)));
          if all_ints {
            let v: Vec<i128> = pos_list
              .iter()
              .map(|p| match p {
                Expr::Integer(n) => *n,
                _ => 0,
              })
              .collect();
            vec![v]
          } else if all_lists {
            let mut out: Vec<Vec<i128>> = Vec::new();
            for p in pos_list {
              if let Expr::List(inner) = p {
                if !inner.iter().all(|x| matches!(x, Expr::Integer(_))) {
                  return Some(unevaluated());
                }
                let v: Vec<i128> = inner
                  .iter()
                  .map(|x| match x {
                    Expr::Integer(n) => *n,
                    _ => 0,
                  })
                  .collect();
                out.push(v);
              }
            }
            out
          } else {
            return Some(unevaluated());
          }
        }
        _ => return Some(unevaluated()),
      };
      match &args[0] {
        Expr::List(_) | Expr::FunctionCall { .. } => {
          return Some(Ok(flatten_at_positions(&args[0], &positions)));
        }
        _ => return Some(unevaluated()),
      }
    }
    "InversePermutation" if args.len() == 1 => {
      if let Expr::List(perm) = &args[0] {
        let n = perm.len();
        let mut inv = vec![Expr::Integer(0); n];
        let mut valid = true;
        for (i, p) in perm.iter().enumerate() {
          if let Expr::Integer(val) = p {
            let idx = *val as usize;
            if idx >= 1 && idx <= n {
              inv[idx - 1] = Expr::Integer((i + 1) as i128);
            } else {
              valid = false;
              break;
            }
          } else {
            valid = false;
            break;
          }
        }
        if valid {
          return Some(Ok(Expr::List(inv.into())));
        }
      }
      // InversePermutation[Cycles[{cycle1, cycle2, ...}]] — reverse each
      // cycle, rotate so its smallest element is first, drop fixed points,
      // and sort cycles by their smallest element.
      if let Expr::FunctionCall {
        name: cname,
        args: cargs,
      } = &args[0]
        && cname == "Cycles"
        && cargs.len() == 1
        && let Expr::List(cycle_list) = &cargs[0]
      {
        let mut out_cycles: Vec<Vec<i128>> =
          Vec::with_capacity(cycle_list.len());
        let mut valid = true;
        for cycle in cycle_list.iter() {
          let Expr::List(c) = cycle else {
            valid = false;
            break;
          };
          let mut ints: Vec<i128> = Vec::with_capacity(c.len());
          for e in c.iter() {
            if let Expr::Integer(n) = e {
              ints.push(*n);
            } else {
              valid = false;
              break;
            }
          }
          if !valid {
            break;
          }
          if ints.len() < 2 {
            continue;
          }
          ints.reverse();
          let min_idx = ints
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| *v)
            .map(|(i, _)| i)
            .unwrap_or(0);
          ints.rotate_left(min_idx);
          out_cycles.push(ints);
        }
        if valid {
          out_cycles.sort_by_key(|c| c[0]);
          let cycle_exprs: Vec<Expr> = out_cycles
            .into_iter()
            .map(|c| Expr::List(c.into_iter().map(Expr::Integer).collect()))
            .collect();
          return Some(Ok(Expr::FunctionCall {
            name: "Cycles".to_string(),
            args: vec![Expr::List(cycle_exprs.into())].into(),
          }));
        }
      }
    }
    "MovingMedian" if args.len() == 2 => {
      if let Expr::List(items) = &args[0]
        && let Some(r) = match &args[1] {
          Expr::Integer(n) if *n >= 1 => Some(*n as usize),
          _ => None,
        }
      {
        let n = items.len();
        if r > n {
          crate::emit_message(&format!(
            "MovingMedian::arg2: The second argument {} must be a positive integer less than or equal to the length {} of the first argument.",
            r, n
          ));
          return Some(Ok(Expr::FunctionCall {
            name: "MovingMedian".to_string(),
            args: args.to_vec().into(),
          }));
        }
        let mut result = Vec::with_capacity(n - r + 1);
        for i in 0..=(n - r) {
          let window = Expr::List(items[i..i + r].to_vec().into());
          match list_helpers_ast::median_ast(&window) {
            Ok(val) => result.push(val),
            Err(e) => return Some(Err(e)),
          }
        }
        return Some(Ok(Expr::List(result.into())));
      }
    }
    "MovingMap" if args.len() == 3 => {
      // MovingMap[f, list, n] - apply f to sublists of length n+1
      if let Expr::List(items) = &args[1]
        && let Some(n) = match &args[2] {
          Expr::Integer(n) => Some(*n as usize),
          _ => None,
        }
      {
        let window_size = n + 1;
        if window_size > items.len() {
          return Some(Ok(Expr::List(vec![].into())));
        }
        let f = &args[0];
        let mut results = Vec::new();
        for i in 0..=(items.len() - window_size) {
          let sublist = Expr::List(items[i..i + window_size].to_vec().into());
          // Construct f[sublist] using Map-like application
          let applied = match f {
            Expr::Identifier(fname) => Expr::FunctionCall {
              name: fname.clone(),
              args: vec![sublist].into(),
            },
            _ => {
              // For pure functions etc, use general application
              Expr::FunctionCall {
                name: expr_to_string(f),
                args: vec![sublist].into(),
              }
            }
          };
          match crate::evaluator::evaluate_expr_to_expr(&applied) {
            Ok(val) => results.push(val),
            Err(e) => return Some(Err(e)),
          }
        }
        return Some(Ok(Expr::List(results.into())));
      }
    }
    "Select" if args.len() == 2 => {
      return Some(list_helpers_ast::select_ast(&args[0], &args[1], None));
    }
    "Select" if args.len() == 3 => {
      return Some(list_helpers_ast::select_ast(
        &args[0],
        &args[1],
        Some(&args[2]),
      ));
    }
    "AllSameBy" if args.len() == 2 => {
      return Some(list_helpers_ast::all_same_by_ast(args));
    }
    "AllTrue" if args.len() == 2 => {
      return Some(list_helpers_ast::all_true_ast(&args[0], &args[1]));
    }
    "AllMatch" if (2..=3).contains(&args.len()) => {
      return Some(list_helpers_ast::all_match_ast(args));
    }
    "AllMatch" if args.len() == 1 => {
      // Operator form: return unevaluated for currying
      return Some(Ok(Expr::FunctionCall {
        name: "AllMatch".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "AnyTrue" if args.len() == 2 => {
      return Some(list_helpers_ast::any_true_ast(&args[0], &args[1]));
    }
    "AnyMatch" if (2..=3).contains(&args.len()) => {
      return Some(list_helpers_ast::any_match_ast(args));
    }
    "AnyMatch" if args.len() == 1 => {
      // Operator form: return unevaluated for currying
      return Some(Ok(Expr::FunctionCall {
        name: "AnyMatch".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "NoneTrue" if args.len() == 2 => {
      return Some(list_helpers_ast::none_true_ast(&args[0], &args[1]));
    }
    "Fold" if args.len() == 2 || args.len() == 3 => {
      if args.len() == 3 {
        return Some(list_helpers_ast::fold_ast(&args[0], &args[1], &args[2]));
      }
      // Fold[f, {a, b, c, ...}] = Fold[f, a, {b, c, ...}]
      // Also handles Fold[f, g[a, b, c, ...]] with arbitrary heads.
      let (items, head): (&[Expr], Option<&str>) = match &args[1] {
        Expr::List(items) => (items.as_slice(), None),
        Expr::FunctionCall { name, args: fargs } => {
          (fargs.as_slice(), Some(name.as_str()))
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Fold".to_string(),
            args: args.to_vec().into(),
          }));
        }
      };
      if items.is_empty() {
        // Fold[f, {}] is unevaluated in Wolfram Language
        return Some(Ok(Expr::FunctionCall {
          name: "Fold".to_string(),
          args: args.to_vec().into(),
        }));
      }
      let init = items[0].clone();
      let rest = match head {
        Some(h) => Expr::FunctionCall {
          name: h.to_string(),
          args: items[1..].to_vec().into(),
        },
        None => Expr::List(items[1..].to_vec().into()),
      };
      return Some(list_helpers_ast::fold_ast(&args[0], &init, &rest));
    }
    "CountBy" if args.len() == 2 => {
      return Some(list_helpers_ast::count_by_ast(&args[0], &args[1]));
    }
    "GroupBy" if args.len() == 2 || args.len() == 3 => {
      let result = list_helpers_ast::group_by_ast(&args[0], &args[1]);
      if args.len() == 3 {
        // GroupBy[list, f, reducer] - apply reducer to each group
        return Some(result.and_then(|grouped| match &grouped {
          Expr::Association(pairs) => {
            let new_pairs: Result<Vec<(Expr, Expr)>, InterpreterError> = pairs
              .iter()
              .map(|(k, v)| {
                let reduced =
                  crate::functions::list_helpers_ast::apply_func_ast(
                    &args[2], v,
                  )?;
                Ok((k.clone(), reduced))
              })
              .collect();
            Ok(Expr::Association(new_pairs?))
          }
          _ => Ok(grouped),
        }));
      }
      return Some(result);
    }
    "SortBy" if args.len() == 2 => {
      return Some(list_helpers_ast::sort_by_ast(&args[0], &args[1]));
    }
    "Ordering" if !args.is_empty() && args.len() <= 3 => {
      return Some(list_helpers_ast::ordering_ast(args));
    }
    "Nest" if args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[2]) {
        return Some(list_helpers_ast::nest_ast(&args[0], &args[1], n));
      }
    }
    "NestList" if args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[2]) {
        return Some(list_helpers_ast::nest_list_ast(&args[0], &args[1], n));
      }
    }
    "FixedPoint" if args.len() >= 2 => {
      let max_iter = if args.len() == 3 {
        expr_to_i128(&args[2])
      } else {
        None
      };
      return Some(list_helpers_ast::fixed_point_ast(
        &args[0], &args[1], max_iter,
      ));
    }
    "Cases" if args.len() == 2 => {
      return Some(list_helpers_ast::cases_ast(&args[0], &args[1]));
    }
    "Cases" if args.len() == 3 => {
      // Cases accepts trailing options like Heads -> True. If the 3rd arg is
      // a Rule, treat it as an option and fall through to the 2-arg path
      // with a default levelspec of {1}.
      if let Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } = &args[2]
      {
        let heads_on = matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Heads")
          && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True");
        if heads_on {
          return Some(list_helpers_ast::cases_heads_ast(&args[0], &args[1]));
        }
        return Some(list_helpers_ast::cases_ast(&args[0], &args[1]));
      }
      return Some(list_helpers_ast::cases_with_level_ast(
        &args[0], &args[1], &args[2], None,
      ));
    }
    "Cases" if args.len() == 4 => {
      return Some(list_helpers_ast::cases_with_level_ast(
        &args[0],
        &args[1],
        &args[2],
        Some(&args[3]),
      ));
    }
    // FirstCase[list, pattern] or FirstCase[list, pattern, default]
    // FirstCase[list, pattern :> rhs] or FirstCase[list, pattern :> rhs, default]
    "FirstCase" if args.len() >= 2 && args.len() <= 3 => {
      return Some(list_helpers_ast::first_case_ast(args));
    }
    "Position" if args.len() == 2 => {
      return Some(list_helpers_ast::position_ast(
        &args[0], &args[1], None, None,
      ));
    }
    "Position" if args.len() == 3 => {
      return Some(list_helpers_ast::position_ast(
        &args[0],
        &args[1],
        Some(&args[2]),
        None,
      ));
    }
    "Position" if args.len() == 4 => {
      return Some(list_helpers_ast::position_ast(
        &args[0],
        &args[1],
        Some(&args[2]),
        Some(&args[3]),
      ));
    }
    "FirstPosition" if args.len() >= 2 => {
      return Some(list_helpers_ast::first_position_ast(args));
    }
    "MapIndexed" if args.len() == 2 => {
      return Some(list_helpers_ast::map_indexed_ast(&args[0], &args[1]));
    }
    "MapIndexed" if args.len() == 3 => {
      return Some(list_helpers_ast::map_indexed_with_level_ast(
        &args[0], &args[1], &args[2],
      ));
    }
    "MapIndexed" if args.len() == 4 => {
      return Some(list_helpers_ast::map_indexed_with_level_heads_ast(
        &args[0], &args[1], &args[2], &args[3],
      ));
    }
    "Tally" if args.len() == 1 => {
      return Some(list_helpers_ast::tally_ast(&args[0]));
    }
    "Tally" if args.len() == 2 => {
      return Some(list_helpers_ast::tally_with_test_ast(&args[0], &args[1]));
    }
    "Counts" if args.len() == 1 => {
      return Some(list_helpers_ast::counts_ast(&args[0]));
    }
    "BinCounts" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::bin_counts_ast(args));
    }
    "BinLists" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::bin_lists_ast(args));
    }
    "HistogramList" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::histogram_list_ast(args));
    }
    "DeleteDuplicates" if args.len() == 1 => {
      return Some(list_helpers_ast::delete_duplicates_ast(&args[0], None));
    }
    "DeleteDuplicates" if args.len() == 2 => {
      return Some(list_helpers_ast::delete_duplicates_ast(
        &args[0],
        Some(&args[1]),
      ));
    }
    "Union" => {
      return Some(list_helpers_ast::union_ast(args));
    }
    "Intersection" => {
      return Some(list_helpers_ast::intersection_ast(args));
    }
    "Complement" => {
      return Some(list_helpers_ast::complement_ast(args));
    }
    "SymmetricDifference" if args.len() >= 2 => {
      return Some(list_helpers_ast::symmetric_difference_ast(args));
    }
    "Dimensions" | "TensorDimensions" if args.len() == 1 || args.len() == 2 => {
      return Some(list_helpers_ast::dimensions_ast(args));
    }
    "Delete" if args.len() == 2 => {
      return Some(list_helpers_ast::delete_ast(args));
    }
    "Order" if args.len() == 2 => {
      // Order[e1, e2]: 1 if e1 < e2, -1 if e1 > e2, 0 if equal (canonical ordering)
      let result =
        crate::functions::list_helpers_ast::compare_exprs(&args[0], &args[1]);
      return Some(Ok(Expr::Integer(result as i128)));
    }
    "OrderedQ" if args.len() == 1 => {
      return Some(list_helpers_ast::ordered_q_ast(args));
    }
    "DeleteAdjacentDuplicates" if args.len() == 1 => {
      return Some(list_helpers_ast::delete_adjacent_duplicates_ast(args));
    }
    "Commonest" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::commonest_ast(args));
    }
    "ClusteringComponents" if args.len() == 1 => {
      return Some(list_helpers_ast::clustering_components_ast(&args[0]));
    }
    "FindClusters" if !args.is_empty() && args.len() <= 3 => {
      return Some(list_helpers_ast::find_clusters_ast_n(args));
    }
    "ComposeList" if args.len() == 2 => {
      return Some(list_helpers_ast::compose_list_ast(args));
    }
    "ContainsOnly" if args.len() == 2 || args.len() == 3 => {
      return Some(list_helpers_ast::contains_only_ast(args));
    }
    "Pick" if args.len() == 2 || args.len() == 3 => {
      return Some(list_helpers_ast::pick_ast(args));
    }
    "LengthWhile" if args.len() == 2 => {
      return Some(list_helpers_ast::length_while_ast(args));
    }
    "TakeLargestBy" if args.len() == 3 => {
      return Some(list_helpers_ast::take_largest_by_ast(args));
    }
    "TakeSmallestBy" if args.len() == 3 => {
      return Some(list_helpers_ast::take_smallest_by_ast(args));
    }

    // Additional AST-native list functions
    "Table" | "ParallelTable" if args.len() == 2 => {
      return Some(list_helpers_ast::table_ast(&args[0], &args[1]));
    }
    "Table" | "ParallelTable" if args.len() >= 3 => {
      // Multi-dimensional Table: Table[expr, iter1, iter2, ...]
      // Nest from innermost to outermost
      return Some(list_helpers_ast::table_multi_ast(&args[0], &args[1..]));
    }
    "MapThread" if args.len() == 2 || args.len() == 3 => {
      let level = if args.len() == 3 {
        match &args[2] {
          Expr::Integer(n) if *n >= 1 => Some(*n as usize),
          _ => None,
        }
      } else {
        None
      };
      return Some(
        match list_helpers_ast::map_thread_ast(&args[0], &args[1], level) {
          Err(InterpreterError::EvaluationError(msg))
            if msg.contains("same length") =>
          {
            crate::emit_message(&format!(
              "MapThread::mptc: Incompatible dimensions of objects in MapThread[{}, {}].",
              crate::syntax::expr_to_string(&args[0]),
              crate::syntax::expr_to_string(&args[1]),
            ));
            Ok(Expr::FunctionCall {
              name: "MapThread".to_string(),
              args: args.to_vec().into(),
            })
          }
          other => other,
        },
      );
    }
    "Downsample" if args.len() == 2 || args.len() == 3 => {
      if let Expr::List(items) = &args[0]
        && let Some(n) = expr_to_i128(&args[1])
        && n >= 1
      {
        let offset = if args.len() == 3 {
          expr_to_i128(&args[2]).unwrap_or(1)
        } else {
          1
        };
        let n = n as usize;
        let offset = (offset - 1).max(0) as usize;
        let result: Vec<Expr> =
          items.iter().skip(offset).step_by(n).cloned().collect();
        return Some(Ok(Expr::List(result.into())));
      }
    }
    "BlockMap" if args.len() == 3 || args.len() == 4 => {
      // BlockMap[f, list, n] or BlockMap[f, list, n, offset]
      if let Some(n) = expr_to_i128(&args[2]) {
        let d = if args.len() == 4 {
          expr_to_i128(&args[3])
        } else {
          None
        };
        return Some(
          list_helpers_ast::partition_ast(&args[1], n, d, None, None).and_then(
            |partitioned| list_helpers_ast::map_ast(&args[0], &partitioned),
          ),
        );
      }
    }
    "Partition" if args.len() >= 2 && args.len() <= 5 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        let d = if args.len() >= 3 {
          expr_to_i128(&args[2])
        } else {
          None
        };
        // args[3] is alignment spec {kL, kR}, args[4] is pad element
        let (align, pad) = if args.len() == 5 {
          (Some(&args[3]), Some(&args[4]))
        } else if args.len() == 4 {
          (Some(&args[3]), None)
        } else {
          (None, None)
        };
        return Some(list_helpers_ast::partition_ast(
          &args[0], n, d, align, pad,
        ));
      }
      // Multi-dimensional form: Partition[tensor, {n1, n2, ...}, d]
      // partitions each dimension in turn with block sizes `n_i` and a
      // uniform offset `d`.
      if let Expr::List(ns) = &args[1]
        && let Some(sizes) =
          ns.iter().map(expr_to_i128).collect::<Option<Vec<_>>>()
        && sizes.iter().all(|&s| s > 0)
      {
        let offsets: Option<Vec<i128>> = if args.len() >= 3 {
          match &args[2] {
            Expr::Integer(n) => Some(vec![*n; sizes.len()]),
            Expr::List(ds) => {
              let ds: Option<Vec<i128>> = ds.iter().map(expr_to_i128).collect();
              match ds {
                Some(v)
                  if v.len() == sizes.len() && v.iter().all(|&x| x > 0) =>
                {
                  Some(v)
                }
                _ => None,
              }
            }
            _ => None,
          }
        } else {
          Some(sizes.clone())
        };
        if let Some(offsets) = offsets {
          return Some(list_helpers_ast::partition_multi_dim_ast(
            &args[0], &sizes, &offsets,
          ));
        }
      }
    }
    "Permutations" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::permutations_ast(args));
    }
    "Signature" if args.len() == 1 => {
      if let Expr::List(items) = &args[0] {
        // Check for duplicates first
        let strs: Vec<String> =
          items.iter().map(crate::syntax::expr_to_string).collect();
        for i in 0..strs.len() {
          for j in (i + 1)..strs.len() {
            if strs[i] == strs[j] {
              return Some(Ok(Expr::Integer(0)));
            }
          }
        }
        // Count inversions to determine signature
        let mut inversions = 0;
        for i in 0..strs.len() {
          for j in (i + 1)..strs.len() {
            if strs[i] > strs[j] {
              inversions += 1;
            }
          }
        }
        return Some(Ok(Expr::Integer(if inversions % 2 == 0 {
          1
        } else {
          -1
        })));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Signature".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "Subsets" if !args.is_empty() && args.len() <= 3 => {
      return Some(list_helpers_ast::subsets_ast(args));
    }
    "SubsetPosition" if args.len() == 2 => {
      return Some(list_helpers_ast::subset_position_ast(&args[0], &args[1]));
    }
    "SubsetCases" if args.len() == 2 || args.len() == 3 => {
      let max_count = if args.len() == 3 {
        match &args[2] {
          Expr::Integer(n) => Some(*n as usize),
          Expr::Identifier(s) if s == "Infinity" => None,
          _ => None,
        }
      } else {
        None
      };
      return Some(list_helpers_ast::subset_cases_ast(
        &args[0], &args[1], max_count,
      ));
    }
    "SubsetCount" if args.len() == 2 => {
      return Some(list_helpers_ast::subset_count_ast(&args[0], &args[1]));
    }
    "Subsequences" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::subsequences_ast(args));
    }
    "Groupings" if args.len() == 2 => {
      return Some(list_helpers_ast::groupings_ast(args));
    }
    "PeakDetect" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::peak_detect_ast(args));
    }
    "SparseArray" if !args.is_empty() => {
      // Normalize to canonical form: SparseArray[Automatic, dims, default, rules].
      // Use Normal[] to expand to a dense nested list.
      return Some(list_helpers_ast::sparse_array_normalize_ast(args));
    }
    "Normal" if args.len() == 1 => {
      // Normal[FittedModel[...]] extracts the fitted expression
      if let Expr::FunctionCall {
        name,
        args: fm_args,
      } = &args[0]
        && name == "FittedModel"
      {
        return Some(
          crate::functions::linear_algebra_ast::fitted_model_normal(fm_args),
        );
      }
      // Normal[NumericArray[list, type]] returns the underlying list
      // (`{{1,2},{3,4}}`), discarding the dtype tag — wolframscript does
      // the same.
      if let Expr::FunctionCall {
        name,
        args: na_args,
      } = &args[0]
        && name == "NumericArray"
        && (na_args.len() == 1 || na_args.len() == 2)
        && matches!(&na_args[0], Expr::List(_))
      {
        return Some(Ok(na_args[0].clone()));
      }
      // Normal[ByteArray["base64"]] extracts the byte list
      if let Expr::FunctionCall {
        name,
        args: ba_args,
      } = &args[0]
        && name == "ByteArray"
        && ba_args.len() == 1
      {
        if let Expr::String(b64) = &ba_args[0] {
          use base64::Engine;
          let engine = base64::engine::general_purpose::STANDARD;
          if let Ok(decoded) = engine.decode(b64) {
            let bytes: Vec<Expr> =
              decoded.iter().map(|b| Expr::Integer(*b as i128)).collect();
            return Some(Ok(Expr::List(bytes.into())));
          }
        }
        // Fallback: if it's already a list (shouldn't happen but be safe)
        if matches!(&ba_args[0], Expr::List(_)) {
          return Some(Ok(ba_args[0].clone()));
        }
      }
      // Normal[SparseArray[...]] expands to a regular list
      if let Expr::FunctionCall {
        name,
        args: sa_args,
      } = &args[0]
        && name == "SparseArray"
      {
        return Some(list_helpers_ast::sparse_array_ast(sa_args));
      }
      // Normal[Dataset[data, ...]] extracts the data
      if let Expr::FunctionCall {
        name,
        args: ds_args,
      } = &args[0]
        && name == "Dataset"
        && !ds_args.is_empty()
      {
        return Some(Ok(ds_args[0].clone()));
      }
      // Normal[Tabular[data, ...]] extracts the data
      if let Expr::FunctionCall {
        name,
        args: tab_args,
      } = &args[0]
        && name == "Tabular"
        && !tab_args.is_empty()
      {
        // For column-oriented data (Association with list values), transpose to rows
        if let Expr::Association(pairs) = &tab_args[0]
          && !pairs.is_empty()
          && pairs.iter().all(|(_, v)| matches!(v, Expr::List(_)))
        {
          // Determine number of rows from the longest column
          let num_rows = pairs
            .iter()
            .map(|(_, v)| {
              if let Expr::List(items) = v {
                items.len()
              } else {
                0
              }
            })
            .max()
            .unwrap_or(0);
          // Build row-oriented associations
          let mut rows = Vec::new();
          for i in 0..num_rows {
            let mut row_pairs = Vec::new();
            for (k, v) in pairs {
              let val = if let Expr::List(items) = v {
                items.get(i).cloned().unwrap_or(Expr::FunctionCall {
                  name: "Missing".to_string(),
                  args: vec![].into(),
                })
              } else {
                v.clone()
              };
              row_pairs.push((k.clone(), val));
            }
            rows.push(Expr::Association(row_pairs));
          }
          return Some(Ok(Expr::List(rows.into())));
        }
        return Some(Ok(tab_args[0].clone()));
      }
      // Normal[SeriesData[x, x0, {c0, c1, ...}, nmin, nmax, den]]
      // => sum(c_i * (x - x0)^(nmin + i), i=0..len-1) when den=1
      if let Expr::FunctionCall {
        name,
        args: sd_args,
      } = &args[0]
        && name == "SeriesData"
        && sd_args.len() == 6
      {
        let var = &sd_args[0];
        let x0 = &sd_args[1];
        let coeffs = match &sd_args[2] {
          Expr::List(c) => c,
          _ => return Some(Ok(args[0].clone())),
        };
        let nmin = match &sd_args[3] {
          Expr::Integer(n) => *n,
          _ => return Some(Ok(args[0].clone())),
        };

        if coeffs.is_empty() {
          return Some(Ok(Expr::Integer(0)));
        }

        let is_zero_center = matches!(x0, Expr::Integer(0));

        // Build the base expression: x or (-x0 + x)
        let base = if is_zero_center {
          var.clone()
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(x0.clone()),
            }),
            right: Box::new(var.clone()),
          }
        };

        // Build terms: c_i * base^(nmin + i). Recursively apply Normal to
        // any inner `SeriesData` coefficient so multivariate Series like
        // `Series[Exp[x-y], {x,0,2}, {y,0,2}] // Normal` collapse to a
        // genuine bivariate polynomial.
        let mut terms: Vec<Expr> = Vec::new();
        for (i, coeff) in coeffs.iter().enumerate() {
          if matches!(coeff, Expr::Integer(0)) {
            continue;
          }
          let coeff_normalised = if matches!(
            coeff,
            Expr::FunctionCall { name, args } if name == "SeriesData" && args.len() == 6
          ) {
            let inner = Expr::FunctionCall {
              name: "Normal".to_string(),
              args: vec![coeff.clone()].into(),
            };
            match evaluate_expr_to_expr(&inner) {
              Ok(v) => v,
              Err(e) => return Some(Err(e)),
            }
          } else {
            coeff.clone()
          };
          let power = nmin + i as i128;
          // base^power
          let base_pow = if power == 0 {
            None
          } else if power == 1 {
            Some(base.clone())
          } else {
            Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(base.clone()),
              right: Box::new(Expr::Integer(power)),
            })
          };

          // Build c * x^n in Mathematica's canonical form:
          // Rational[-a,b]*x^n => -(a*x^n)/b  which prints as -(a*x^n)/b
          let term = match base_pow {
            None => coeff_normalised,
            Some(bp) => {
              // Evaluate the Times to get canonical form
              let t = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(coeff_normalised),
                right: Box::new(bp),
              };
              match evaluate_expr_to_expr(&t) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
              }
            }
          };
          terms.push(term);
        }

        if terms.is_empty() {
          return Some(Ok(Expr::Integer(0)));
        }

        // Combine terms with Plus, preserving order (low to high power)
        let result = terms
          .into_iter()
          .reduce(|acc, t| Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(acc),
            right: Box::new(t),
          })
          .unwrap();
        return Some(Ok(result));
      }
      // Normal[<|k -> v, ...|>] converts Association to List of rules.
      // A `RuleDelayed { pattern == key, replacement }` value is the marker
      // for an entry that was originally `key :> value`; preserve it as
      // `key :> value` in the output list.
      if let Expr::Association(pairs) = &args[0] {
        let rules: Vec<Expr> = pairs
          .iter()
          .map(|(k, v)| match v {
            Expr::RuleDelayed {
              pattern,
              replacement,
            } if crate::syntax::assoc_marker_matches(k, pattern) => {
              Expr::RuleDelayed {
                pattern: Box::new(k.clone()),
                replacement: replacement.clone(),
              }
            }
            _ => Expr::Rule {
              pattern: Box::new(k.clone()),
              replacement: Box::new(v.clone()),
            },
          })
          .collect();
        return Some(Ok(Expr::List(rules.into())));
      }
      // Normal[FunctionCall{Association, args}] converts to List
      if let Expr::FunctionCall {
        name,
        args: assoc_args,
      } = &args[0]
        && name == "Association"
      {
        return Some(Ok(Expr::List(assoc_args.clone())));
      }
      // For other expressions, recursively convert Associations in arguments
      return Some(Ok(normal_convert_associations(&args[0])));
    }
    "First" if args.len() == 1 || args.len() == 2 => {
      let default = if args.len() == 2 {
        Some(&args[1])
      } else {
        None
      };
      return Some(list_helpers_ast::first_ast(&args[0], default));
    }
    "Last" if args.len() == 1 || args.len() == 2 => {
      let default = if args.len() == 2 {
        Some(&args[1])
      } else {
        None
      };
      return Some(list_helpers_ast::last_ast(&args[0], default));
    }
    "Rest" if args.len() == 1 => {
      return Some(list_helpers_ast::rest_ast(&args[0]));
    }
    "Most" if args.len() == 1 => {
      return Some(list_helpers_ast::most_ast(&args[0]));
    }
    "Take" if args.len() >= 2 => {
      return Some(list_helpers_ast::take_multi_ast(&args[0], &args[1..]));
    }
    "Drop" if args.len() == 2 => {
      return Some(list_helpers_ast::drop_ast(&args[0], &args[1]));
    }
    "Drop" if args.len() == 3 => {
      return Some(list_helpers_ast::drop_multi_ast(
        &args[0], &args[1], &args[2],
      ));
    }
    "ArrayRules" if args.len() == 1 || args.len() == 2 => {
      return Some(array_rules_ast(args));
    }
    "TakeDrop" if args.len() == 2 => {
      let taken = list_helpers_ast::take_multi_ast(&args[0], &args[1..]);
      let dropped = list_helpers_ast::drop_ast(&args[0], &args[1]);
      return Some(match (taken, dropped) {
        (Ok(t), Ok(d)) => Ok(Expr::List(vec![t, d].into())),
        (Err(e), _) | (_, Err(e)) => Err(e),
      });
    }
    "ArrayFlatten" if args.len() == 1 => {
      return Some(array_flatten_ast(&args[0]));
    }
    "Flatten" if args.len() == 1 => {
      return Some(list_helpers_ast::flatten_ast(&args[0]));
    }
    "Flatten" if args.len() == 2 => {
      // Flatten[expr, n] or Flatten[expr, Infinity, head]
      if let Expr::Identifier(id) = &args[1] {
        if id == "Infinity" {
          // Flatten[expr, Infinity] same as Flatten[expr]
          return Some(list_helpers_ast::flatten_ast(&args[0]));
        }
        // Flatten[expr, head] — treat identifier as head
        return Some(list_helpers_ast::flatten_head_ast(
          &args[0],
          i128::MAX,
          id,
        ));
      }
      // Check for dimension spec: Flatten[list, {{2}, {1}}]
      if let Expr::List(outer) = &args[1]
        && !outer.is_empty()
        && matches!(&outer[0], Expr::List(_))
      {
        // Parse dimension spec: each element is a list of level numbers
        let mut dim_spec: Vec<Vec<usize>> = Vec::new();
        let mut valid = true;
        for item in outer {
          if let Expr::List(levels) = item {
            let mut group: Vec<usize> = Vec::new();
            for level in levels {
              if let Some(n) = expr_to_i128(level) {
                group.push(n as usize);
              } else {
                valid = false;
                break;
              }
            }
            dim_spec.push(group);
          } else {
            valid = false;
            break;
          }
          if !valid {
            break;
          }
        }
        if valid {
          return Some(list_helpers_ast::flatten_dims_ast(&args[0], &dim_spec));
        }
      }
      // Flat level list: Flatten[list, {1, 2, ...}] — merges those levels
      // into a single level. Equivalent to Flatten[list, {{1, 2, ...}}].
      if let Expr::List(levels) = &args[1]
        && !levels.is_empty()
        && levels.iter().all(|e| expr_to_i128(e).is_some())
      {
        let group: Vec<usize> = levels
          .iter()
          .filter_map(|e| expr_to_i128(e).map(|n| n as usize))
          .collect();
        return Some(list_helpers_ast::flatten_dims_ast(&args[0], &[group]));
      }
      if let Some(n) = expr_to_i128(&args[1]) {
        return Some(list_helpers_ast::flatten_level_ast(&args[0], n));
      }
    }
    "Flatten" if args.len() == 3 => {
      // Flatten[expr, depth, head]
      let depth = match &args[1] {
        Expr::Identifier(id) if id == "Infinity" => i128::MAX,
        _ => expr_to_i128(&args[1]).unwrap_or(i128::MAX),
      };
      if let Expr::Identifier(head) = &args[2] {
        return Some(list_helpers_ast::flatten_head_ast(&args[0], depth, head));
      }
    }
    "Level" if args.len() == 2 => {
      return Some(list_helpers_ast::level_ast(&args[0], &args[1], false));
    }
    "Level" if args.len() == 3 => {
      // Extract Heads option
      let include_heads = match &args[2] {
        Expr::Rule {
          pattern,
          replacement,
        } => {
          if let Expr::Identifier(name) = pattern.as_ref() {
            if name == "Heads" {
              matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True")
            } else {
              false
            }
          } else {
            false
          }
        }
        _ => false,
      };
      return Some(list_helpers_ast::level_ast(
        &args[0],
        &args[1],
        include_heads,
      ));
    }
    "Reverse" if args.len() == 1 => {
      return Some(list_helpers_ast::reverse_ast(&args[0]));
    }
    "Reverse" if args.len() == 2 => {
      return Some(list_helpers_ast::reverse_level_ast(&args[0], &args[1]));
    }
    "Sort" if args.len() == 1 => {
      return Some(list_helpers_ast::sort_ast(&args[0]));
    }
    "Sort" if args.len() == 2 => {
      // Sort[list, p] - sort using comparator p
      // p[a, b] returns True if a should come before b
      let (items, head_name) = match &args[0] {
        Expr::List(items) => (Some(items.clone()), None),
        Expr::FunctionCall { name, args } => {
          (Some(args.clone()), Some(name.clone()))
        }
        _ => (None, None),
      };
      if let Some(mut sorted) = items {
        let wrap = |items: Vec<Expr>| -> Expr {
          match &head_name {
            None => Expr::List(items.into()),
            Some(name) => Expr::FunctionCall {
              name: name.clone(),
              args: items.into(),
            },
          }
        };
        // Fast path: the two bare symbol comparators that Sort is most
        // commonly called with.
        if let Expr::Identifier(name) = &args[1] {
          match name.as_str() {
            "Greater" => {
              sorted.sort_by(|a, b| {
                list_helpers_ast::canonical_cmp(a, b).reverse()
              });
              return Some(Ok(wrap(sorted.to_vec())));
            }
            "Less" => {
              sorted.sort_by(list_helpers_ast::canonical_cmp);
              return Some(Ok(wrap(sorted.to_vec())));
            }
            _ => {}
          }
        }
        // General path: evaluate p[a, b] for each comparison via the
        // normal function-application machinery, so pure Functions
        // (`#1 > #2 &`), NamedFunctions, and curried calls all work.
        let comparator = args[1].clone();
        sorted.sort_by(|a, b| {
          let result =
            crate::functions::list_helpers_ast::apply_func_to_two_args(
              &comparator,
              a,
              b,
            );
          match result {
            Ok(Expr::Identifier(ref s)) if s == "True" => {
              std::cmp::Ordering::Less
            }
            _ => std::cmp::Ordering::Greater,
          }
        });
        return Some(Ok(wrap(sorted.to_vec())));
      }
    }
    "ReverseSort" if args.len() == 1 || args.len() == 2 => {
      // ReverseSort[list] sorts then reverses
      // ReverseSort[list, p] sorts by p then reverses
      let mut sorted = match evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Sort".to_string(),
        args: args.to_vec().into(),
      }) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      if let Expr::List(ref mut items) = sorted {
        items.reverse();
        return Some(Ok(Expr::List(std::mem::take(items))));
      }
      return Some(Ok(sorted));
    }
    "List" => {
      // List[a, b, c] is equivalent to {a, b, c}
      return Some(Ok(Expr::List(args.to_vec().into())));
    }
    "Range" => {
      return Some(list_helpers_ast::range_ast(args));
    }
    "PowerRange" if args.len() == 2 || args.len() == 3 => {
      return Some(list_helpers_ast::power_range_ast(args));
    }
    "Accumulate" if args.len() == 1 => {
      return Some(list_helpers_ast::accumulate_ast(&args[0]));
    }
    "AnglePath" => {
      return Some(list_helpers_ast::angle_path_ast(args));
    }
    "Differences" if args.len() == 1 => {
      return Some(list_helpers_ast::differences_ast(&args[0]));
    }
    "Differences" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return Some(list_helpers_ast::differences_n_ast(&args[0], n as usize));
      }
      if let Expr::List(spec_items) = &args[1] {
        let spec: Option<Vec<usize>> = spec_items
          .iter()
          .map(|e| expr_to_i128(e).map(|n| n as usize))
          .collect();
        if let Some(spec) = spec {
          return Some(list_helpers_ast::differences_spec_ast(&args[0], &spec));
        }
      }
    }
    "Ratios" if args.len() == 1 || args.len() == 2 => {
      let n = if args.len() == 2 {
        match expr_to_i128(&args[1]) {
          Some(n) if n >= 0 => n as usize,
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "Ratios".to_string(),
              args: args.to_vec().into(),
            }));
          }
        }
      } else {
        1
      };
      if let Expr::List(items) = &args[0] {
        let mut current = items.clone();
        for _ in 0..n {
          if current.len() < 2 {
            return Some(Ok(Expr::List(vec![].into())));
          }
          let mut next = Vec::with_capacity(current.len() - 1);
          for i in 1..current.len() {
            let ratio = match evaluate_expr_to_expr(&Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(current[i].clone()),
              right: Box::new(current[i - 1].clone()),
            }) {
              Ok(v) => v,
              Err(e) => return Some(Err(e)),
            };
            next.push(ratio);
          }
          current = next.into();
        }
        return Some(Ok(Expr::List(current)));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Ratios".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "Scan" if args.len() == 2 => {
      return Some(list_helpers_ast::scan_ast(&args[0], &args[1]));
    }
    "FoldList" if args.len() == 2 || args.len() == 3 => {
      if args.len() == 3 {
        return Some(list_helpers_ast::fold_list_ast(
          &args[0], &args[1], &args[2],
        ));
      }
      // FoldList[f, {a, b, c, ...}] = FoldList[f, a, {b, c, ...}]
      // Also handles FoldList[f, g[a, b, c, ...]] with arbitrary heads.
      let (items, head): (&[Expr], Option<&str>) = match &args[1] {
        Expr::List(items) => (items.as_slice(), None),
        Expr::FunctionCall { name, args: fargs } => {
          (fargs.as_slice(), Some(name.as_str()))
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "FoldList".to_string(),
            args: args.to_vec().into(),
          }));
        }
      };
      if items.is_empty() {
        return Some(Ok(match head {
          Some(h) => Expr::FunctionCall {
            name: h.to_string(),
            args: vec![].into(),
          },
          None => Expr::List(vec![].into()),
        }));
      }
      let init = items[0].clone();
      let rest = match head {
        Some(h) => Expr::FunctionCall {
          name: h.to_string(),
          args: items[1..].to_vec().into(),
        },
        None => Expr::List(items[1..].to_vec().into()),
      };
      return Some(list_helpers_ast::fold_list_ast(&args[0], &init, &rest));
    }
    "FixedPointList" if args.len() >= 2 => {
      let max_iter = if args.len() == 3 {
        expr_to_i128(&args[2])
      } else {
        None
      };
      return Some(list_helpers_ast::fixed_point_list_ast(
        &args[0], &args[1], max_iter,
      ));
    }
    "Transpose" if args.len() == 1 => {
      return Some(match list_helpers_ast::transpose_ast(&args[0]) {
        Err(InterpreterError::EvaluationError(msg))
          if msg.contains("same length") =>
        {
          crate::emit_message(&format!(
            "Transpose::nmtx: The first two levels of {} cannot be transposed.",
            crate::syntax::expr_to_string(&args[0])
          ));
          Ok(Expr::FunctionCall {
            name: "Transpose".to_string(),
            args: args.to_vec().into(),
          })
        }
        other => other,
      });
    }
    "Transpose" if args.len() == 2 => {
      if let Expr::List(perm) = &args[1] {
        return Some(list_helpers_ast::transpose_perm_ast(&args[0], perm));
      }
    }
    "Diagonal" if args.len() == 1 || args.len() == 2 => {
      let offset = if args.len() == 2 {
        match &args[1] {
          Expr::Integer(n) => *n as i64,
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "Diagonal".to_string(),
              args: args.to_vec().into(),
            }));
          }
        }
      } else {
        0
      };
      if let Expr::List(rows) = &args[0] {
        let mut result = Vec::new();
        let nrows = rows.len() as i64;
        for (i, row) in rows.iter().enumerate() {
          if let Expr::List(cols) = row {
            let j = i as i64 + offset;
            if j >= 0 && (j as usize) < cols.len() && (i as i64) < nrows {
              result.push(cols[j as usize].clone());
            }
          }
        }
        return Some(Ok(Expr::List(result.into())));
      }
    }
    "Riffle" if args.len() == 2 => {
      return Some(list_helpers_ast::riffle_ast(&args[0], &args[1]));
    }
    "Riffle" if args.len() == 3 => {
      return Some(list_helpers_ast::riffle_extended_ast(
        &args[0], &args[1], &args[2],
      ));
    }
    "RotateLeft" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return Some(list_helpers_ast::rotate_left_ast(&args[0], n));
      }
      if let Expr::List(shifts) = &args[1] {
        return Some(list_helpers_ast::rotate_multi_ast(
          &args[0], shifts, true,
        ));
      }
    }
    "RotateLeft" if args.len() == 1 => {
      return Some(list_helpers_ast::rotate_left_ast(&args[0], 1));
    }
    "RotateRight" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return Some(list_helpers_ast::rotate_right_ast(&args[0], n));
      }
      if let Expr::List(shifts) = &args[1] {
        return Some(list_helpers_ast::rotate_multi_ast(
          &args[0], shifts, false,
        ));
      }
    }
    "RotateRight" if args.len() == 1 => {
      return Some(list_helpers_ast::rotate_right_ast(&args[0], 1));
    }
    "PadLeft" if args.len() == 1 => {
      // PadLeft[{{}, {1, 2}, {1, 2, 3}}] - auto-pad ragged array
      if let Expr::List(items) = &args[0] {
        let max_len = items
          .iter()
          .filter_map(|item| match item {
            Expr::List(sub) => Some(sub.len()),
            _ => None,
          })
          .max()
          .unwrap_or(0);
        let padded: Vec<Expr> = items
          .iter()
          .map(|item| {
            list_helpers_ast::pad_left_ast(
              item,
              max_len as i128,
              &Expr::Integer(0),
              None,
            )
            .unwrap_or_else(|_| item.clone())
          })
          .collect();
        return Some(Ok(Expr::List(padded.into())));
      }
    }
    "PadLeft" if args.len() >= 2 => {
      // Multi-dim form: PadLeft[list, {n1, n2, ...}, pad?, margin?]
      if let Expr::List(dim_items) = &args[1] {
        let ns_opt: Option<Vec<i128>> =
          dim_items.iter().map(expr_to_i128).collect();
        if let Some(ns) = ns_opt {
          let pad = if args.len() >= 3 {
            args[2].clone()
          } else {
            Expr::Integer(0)
          };
          // 4th-argument margin: scalar broadcasts to every dim, list
          // maps 1:1 per dimension.
          let margins: Vec<i128> = if args.len() >= 4 {
            match &args[3] {
              Expr::List(ms) => ms.iter().filter_map(expr_to_i128).collect(),
              _ => expr_to_i128(&args[3])
                .map(|m| vec![m; ns.len()])
                .unwrap_or_default(),
            }
          } else {
            Vec::new()
          };
          return Some(list_helpers_ast::pad_left_multidim_with_margin(
            &args[0], &ns, &pad, &margins,
          ));
        }
      }
      if let Some(n) = expr_to_i128(&args[1]) {
        let pad = if args.len() >= 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        let offset = if args.len() >= 4 {
          expr_to_i128(&args[3])
        } else {
          None
        };
        return Some(list_helpers_ast::pad_left_ast(&args[0], n, &pad, offset));
      }
    }
    "PadRight" if args.len() == 1 => {
      // PadRight[{{}, {1, 2}, {1, 2, 3}}] - auto-pad ragged array
      if let Expr::List(items) = &args[0] {
        let max_len = items
          .iter()
          .filter_map(|item| match item {
            Expr::List(sub) => Some(sub.len()),
            _ => None,
          })
          .max()
          .unwrap_or(0);
        let padded: Vec<Expr> = items
          .iter()
          .map(|item| {
            list_helpers_ast::pad_right_ast(
              item,
              max_len as i128,
              &Expr::Integer(0),
              None,
            )
            .unwrap_or_else(|_| item.clone())
          })
          .collect();
        return Some(Ok(Expr::List(padded.into())));
      }
    }
    "PadRight" if args.len() >= 2 => {
      if let Expr::List(dim_items) = &args[1] {
        let ns_opt: Option<Vec<i128>> =
          dim_items.iter().map(expr_to_i128).collect();
        if let Some(ns) = ns_opt {
          let pad = if args.len() >= 3 {
            args[2].clone()
          } else {
            Expr::Integer(0)
          };
          // 4th argument is the per-dimension margin. Scalar margin
          // broadcasts to every dimension; a list of margins maps 1:1.
          let margins: Vec<i128> = if args.len() >= 4 {
            match &args[3] {
              Expr::List(ms) => ms.iter().filter_map(expr_to_i128).collect(),
              _ => expr_to_i128(&args[3])
                .map(|m| vec![m; ns.len()])
                .unwrap_or_default(),
            }
          } else {
            Vec::new()
          };
          return Some(list_helpers_ast::pad_right_multidim_with_margin(
            &args[0], &ns, &pad, &margins,
          ));
        }
      }
      if let Some(n) = expr_to_i128(&args[1]) {
        let pad = if args.len() >= 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        let offset = if args.len() >= 4 {
          expr_to_i128(&args[3])
        } else {
          None
        };
        return Some(list_helpers_ast::pad_right_ast(
          &args[0], n, &pad, offset,
        ));
      }
    }
    "Join" => {
      // Check if last argument is an integer level spec
      if args.len() >= 3
        && let Expr::Integer(n) = &args[args.len() - 1]
      {
        let level = *n as usize;
        return Some(list_helpers_ast::join_at_level_ast(
          &args[..args.len() - 1],
          level,
        ));
      }
      return Some(list_helpers_ast::join_ast(args));
    }
    "Append" if args.len() == 2 => {
      return Some(list_helpers_ast::append_ast(&args[0], &args[1]));
    }
    "Prepend" if args.len() == 2 => {
      return Some(list_helpers_ast::prepend_ast(&args[0], &args[1]));
    }
    "DeleteDuplicatesBy" if args.len() == 2 => {
      return Some(list_helpers_ast::delete_duplicates_by_ast(
        &args[0], &args[1],
      ));
    }
    "Median" if args.len() == 1 => {
      return Some(list_helpers_ast::median_ast(&args[0]));
    }
    "Count" if args.len() == 2 => {
      return Some(list_helpers_ast::count_ast(&args[0], &args[1]));
    }
    "Count" if args.len() == 3 => {
      return Some(list_helpers_ast::count_ast_level(
        &args[0],
        &args[1],
        Some(&args[2]),
      ));
    }
    "ConstantArray" if args.len() == 2 => {
      return Some(list_helpers_ast::constant_array_ast(&args[0], &args[1]));
    }
    "NestWhile" if (3..=6).contains(&args.len()) => {
      // NestWhile[f, x, test]              — plain (m = 1)
      // NestWhile[f, x, test, m]           — m = number of recent values to
      //                                        pass to test (positive integer
      //                                        or `All`)
      // NestWhile[f, x, test, m, max]      — max is the maximum iteration cap
      // NestWhile[f, x, test, m, max, n]   — n extra iterations (or -|n|
      //                                        steps back) once test fails
      let m = if args.len() >= 4 {
        parse_nest_while_m(&args[3])?
      } else {
        list_helpers_ast::NestWhileM::Last(1)
      };
      let max_iter = if args.len() >= 5 {
        expr_to_i128(&args[4])
      } else {
        None
      };
      let extra_n = if args.len() == 6 {
        expr_to_i128(&args[5])?
      } else {
        0
      };
      return Some(list_helpers_ast::nest_while_ast(
        &args[0], &args[1], &args[2], m, max_iter, extra_n,
      ));
    }
    "NestWhileList" if (3..=6).contains(&args.len()) => {
      let m = if args.len() >= 4 {
        parse_nest_while_m(&args[3])?
      } else {
        list_helpers_ast::NestWhileM::Last(1)
      };
      let max_iter = if args.len() >= 5 {
        expr_to_i128(&args[4])
      } else {
        None
      };
      let extra_n = if args.len() == 6 {
        expr_to_i128(&args[5])?
      } else {
        0
      };
      return Some(list_helpers_ast::nest_while_list_ast(
        &args[0], &args[1], &args[2], m, max_iter, extra_n,
      ));
    }
    "Thread" if args.len() == 1 => {
      return Some(match list_helpers_ast::thread_ast(&args[0], None) {
        Err(InterpreterError::EvaluationError(msg))
          if msg.contains("same length") =>
        {
          crate::emit_message(&format!(
            "Thread::tdlen: Objects of unequal length in {} cannot be combined.",
            crate::syntax::expr_to_string(&args[0])
          ));
          Ok(args[0].clone())
        }
        other => other,
      });
    }
    "Thread" if args.len() == 2 => {
      let head = if let Expr::Identifier(head) = &args[1] {
        Some(head.as_str())
      } else {
        None
      };
      return Some(match list_helpers_ast::thread_ast(&args[0], head) {
        Err(InterpreterError::EvaluationError(msg))
          if msg.contains("same length") =>
        {
          crate::emit_message(&format!(
            "Thread::tdlen: Objects of unequal length in {} cannot be combined.",
            crate::syntax::expr_to_string(&args[0])
          ));
          Ok(args[0].clone())
        }
        other => other,
      });
    }
    "Through" if args.len() == 1 => {
      return Some(list_helpers_ast::through_ast(&args[0], None));
    }
    "Through" if args.len() == 2 => {
      // Through[expr, h] - only apply if head of expr matches h
      let head_filter = crate::syntax::expr_to_string(&args[1]);
      return Some(list_helpers_ast::through_ast(&args[0], Some(&head_filter)));
    }
    "Operate" if args.len() == 2 || args.len() == 3 => {
      let p = &args[0];
      let expr = &args[1];
      let n = if args.len() == 3 {
        expr_to_i128(&args[2]).unwrap_or(1)
      } else {
        1
      };
      if n == 0 {
        return Some(
          Ok(Expr::FunctionCall {
            name: "".to_string(),
            args: vec![expr.clone()].into(),
          })
          .map(|_| Expr::FunctionCall {
            name: crate::syntax::expr_to_string(p),
            args: vec![expr.clone()].into(),
          }),
        );
      }
      // For n >= 1, we need to wrap the head at depth n.
      // For n == 1 (default): f[a, b] -> p[f][a, b]
      // When depth exceeds the expression's nesting (including atoms at any
      // depth), return the expression unchanged (matches wolframscript).
      //
      // Expressions like `f[a][b][c]` can arrive as a FunctionCall whose
      // `name` field is a literal string "f[a][b]" (the Woxi parser leaves
      // some deeply-nested calls in this form). Detect that shape and
      // re-parse the name so the recursion can peel the nesting correctly.
      fn decode_complex_head(name: &str) -> Option<Expr> {
        if !name.contains('[') {
          return None;
        }
        crate::syntax::string_to_expr(name).ok()
      }
      fn wrap_head_at_depth(expr: &Expr, p: &Expr, depth: i128) -> Expr {
        if depth == 0 {
          Expr::FunctionCall {
            name: crate::syntax::expr_to_string(p),
            args: vec![expr.clone()].into(),
          }
        } else {
          match expr {
            Expr::FunctionCall { name, args } => {
              let head_expr = decode_complex_head(name)
                .unwrap_or_else(|| Expr::Identifier(name.clone()));
              let wrapped_head = wrap_head_at_depth(&head_expr, p, depth - 1);
              Expr::CurriedCall {
                func: Box::new(wrapped_head),
                args: args.to_vec(),
              }
            }
            Expr::CurriedCall { func, args } => {
              let wrapped_func = wrap_head_at_depth(func, p, depth - 1);
              Expr::CurriedCall {
                func: Box::new(wrapped_func),
                args: args.clone(),
              }
            }
            _ => expr.clone(),
          }
        }
      }
      return Some(Ok(wrap_head_at_depth(expr, p, n)));
    }
    "TakeLargest" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return Some(list_helpers_ast::take_largest_ast(&args[0], n));
      }
    }
    "TakeLargest" if args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[1])
        && let Some(forms) = parse_excluded_forms(&args[2])
      {
        return Some(list_helpers_ast::take_largest_excluded_ast(
          &args[0], n, &forms,
        ));
      }
    }
    "TakeSmallest" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return Some(list_helpers_ast::take_smallest_ast(&args[0], n));
      }
    }
    "TakeSmallest" if args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[1])
        && let Some(forms) = parse_excluded_forms(&args[2])
      {
        return Some(list_helpers_ast::take_smallest_excluded_ast(
          &args[0], n, &forms,
        ));
      }
    }
    "MinimalBy" if args.len() == 2 || args.len() == 3 => {
      let n = if args.len() == 3 {
        expr_to_i128(&args[2])
      } else {
        None
      };
      return Some(list_helpers_ast::minimal_by_ast(&args[0], &args[1], n));
    }
    "MaximalBy" if args.len() == 2 || args.len() == 3 => {
      let n = if args.len() == 3 {
        expr_to_i128(&args[2])
      } else {
        None
      };
      return Some(list_helpers_ast::maximal_by_ast(&args[0], &args[1], n));
    }
    "ArrayDepth" if args.len() == 1 => {
      return Some(list_helpers_ast::array_depth_ast(&args[0]));
    }
    "TensorRank" if args.len() == 1 => {
      return Some(list_helpers_ast::tensor_rank_ast(&args[0]));
    }
    "TensorSymmetry" if args.len() == 1 => {
      return Some(list_helpers_ast::tensor_symmetry_ast(&args[0]));
    }
    "TensorContract" if args.len() == 2 => {
      return Some(list_helpers_ast::tensor_contract_ast(args));
    }
    "ArrayQ" if !args.is_empty() && args.len() <= 3 => {
      let is_array = match list_helpers_ast::array_q_ast(&args[0]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      if !matches!(&is_array, Expr::Identifier(s) if s == "True") {
        return Some(Ok(is_array));
      }
      // Determine array depth.
      let depth = match list_helpers_ast::dimensions_ast(&[args[0].clone()]) {
        Ok(Expr::List(ref d)) => d.len(),
        _ => return Some(Ok(is_array)),
      };
      if args.len() >= 2 {
        // ArrayQ[expr, n] - depth must equal n
        if let Some(n) = expr_to_i128(&args[1]) {
          if depth != n as usize {
            return Some(Ok(Expr::Identifier("False".to_string())));
          }
        } else {
          return Some(Ok(is_array));
        }
      }
      if args.len() == 3 {
        // ArrayQ[expr, n, test] - every leaf at depth `n` must pass `test`
        let test = &args[2];
        let leaves_pass =
          list_helpers_ast::all_leaves_pass_test(&args[0], depth, test);
        return Some(Ok(Expr::Identifier(
          (if leaves_pass { "True" } else { "False" }).to_string(),
        )));
      }
      return Some(Ok(Expr::Identifier("True".to_string())));
    }
    "VectorQ" if args.len() == 1 => {
      return Some(list_helpers_ast::vector_q_ast(&args[0]));
    }
    "MatrixQ" if args.len() == 1 => {
      return Some(list_helpers_ast::matrix_q_ast(&args[0]));
    }
    "MatrixQ" if args.len() == 2 => {
      return Some(list_helpers_ast::matrix_q_with_test_ast(
        &args[0], &args[1],
      ));
    }
    "ContainsAny" if args.len() == 2 => {
      if let (Expr::List(list1), Expr::List(list2)) = (&args[0], &args[1]) {
        let set1: std::collections::HashSet<String> =
          list1.iter().map(expr_to_string).collect();
        let result = list2.iter().any(|x| set1.contains(&expr_to_string(x)));
        return Some(Ok(Expr::Identifier(
          if result { "True" } else { "False" }.to_string(),
        )));
      }
    }
    "ContainsAll" if args.len() == 2 => {
      if let (Expr::List(list1), Expr::List(list2)) = (&args[0], &args[1]) {
        let set1: std::collections::HashSet<String> =
          list1.iter().map(expr_to_string).collect();
        let result = list2.iter().all(|x| set1.contains(&expr_to_string(x)));
        return Some(Ok(Expr::Identifier(
          if result { "True" } else { "False" }.to_string(),
        )));
      }
    }
    "ContainsNone" if args.len() == 2 => {
      if let (Expr::List(list1), Expr::List(list2)) = (&args[0], &args[1]) {
        let set1: std::collections::HashSet<String> =
          list1.iter().map(expr_to_string).collect();
        let result = !list2.iter().any(|x| set1.contains(&expr_to_string(x)));
        return Some(Ok(Expr::Identifier(
          if result { "True" } else { "False" }.to_string(),
        )));
      }
    }
    "SquareMatrixQ" if args.len() == 1 => {
      let result = match &args[0] {
        Expr::List(rows) if !rows.is_empty() => {
          let nrows = rows.len();
          rows
            .iter()
            .all(|r| matches!(r, Expr::List(cols) if cols.len() == nrows))
        }
        _ => false,
      };
      return Some(Ok(if result {
        Expr::Identifier("True".to_string())
      } else {
        Expr::Identifier("False".to_string())
      }));
    }
    "TakeWhile" if args.len() == 2 => {
      return Some(list_helpers_ast::take_while_ast(&args[0], &args[1]));
    }
    "Do" if args.len() == 2 => {
      return Some(list_helpers_ast::do_ast(&args[0], &args[1]));
    }
    "Do" if args.len() > 2 => {
      // Multi-iterator Do: Do[body, {i, ...}, {j, ...}, ...] is a single
      // construct in Wolfram. Break[] and Return[] exit the entire Do, not
      // just the innermost iterator, so we cannot lower it to nested Do
      // calls (each of which would catch Break/Return at its own level).
      return Some(list_helpers_ast::do_multi_ast(&args[0], &args[1..]));
    }
    "For" if args.len() == 3 || args.len() == 4 => {
      return Some(for_ast(args));
    }
    "DeleteCases" if args.len() == 2 => {
      return Some(list_helpers_ast::delete_cases_ast(&args[0], &args[1]));
    }
    "DeleteCases" if args.len() == 3 => {
      // DeleteCases[list, pattern, levelspec]
      return Some(list_helpers_ast::delete_cases_with_level_ast(
        &args[0], &args[1], &args[2],
      ));
    }
    "DeleteCases" if args.len() == 4 => {
      // DeleteCases[list, pattern, levelspec, n]
      // For now, level spec is applied but count is ignored in level-aware version
      // Fall back to count-only version for simple cases
      let max_count = expr_to_i128(&args[3]);
      return Some(list_helpers_ast::delete_cases_with_count_ast(
        &args[0], &args[1], max_count,
      ));
    }
    "MinMax" if args.len() == 1 || args.len() == 2 => {
      return Some(list_helpers_ast::min_max_ast(args));
    }
    // Indexed[expr, i]            ≡ Part[expr, i]     for a concrete List expr
    // Indexed[expr, {i, j, ...}]  ≡ Part[expr, i, j, ...]
    // Otherwise the call stays unevaluated, with the index normalised
    // into a singleton list (matching wolframscript's canonical form).
    "Indexed" if args.len() == 2 => {
      let unevaluated_with_list_index = || {
        let idx = match &args[1] {
          Expr::List(_) => args[1].clone(),
          other => Expr::List(vec![other.clone()].into()),
        };
        Expr::FunctionCall {
          name: "Indexed".to_string(),
          args: vec![args[0].clone(), idx].into(),
        }
      };
      // Collect the indices from the second arg. A non-list index is
      // treated as a single-element index list.
      let idx_specs: Vec<&Expr> = match &args[1] {
        Expr::List(items) => items.iter().collect(),
        other => vec![other],
      };
      if idx_specs.is_empty() {
        return Some(Ok(args[0].clone()));
      }
      // Validate every index is a nonzero integer; otherwise stay
      // unevaluated (the original unsimplified call).
      let mut ints: Vec<i128> = Vec::with_capacity(idx_specs.len());
      for spec in &idx_specs {
        match spec {
          Expr::Integer(n) if *n != 0 => ints.push(*n),
          Expr::Integer(_) => {
            crate::emit_message(
              "Indexed::ind: The index 0 is not a nonzero integer.",
            );
            return Some(Ok(Expr::FunctionCall {
              name: "Indexed".to_string(),
              args: args.to_vec().into(),
            }));
          }
          _ => {
            return Some(Ok(unevaluated_with_list_index()));
          }
        }
      }
      // Walk into the data one level per index. A non-List head means
      // we can't resolve concretely — fall back to canonical Indexed.
      let mut current = args[0].clone();
      for n in &ints {
        let Expr::List(items) = &current else {
          return Some(Ok(unevaluated_with_list_index()));
        };
        let len = items.len() as i128;
        let pos: i128 = if *n > 0 { *n - 1 } else { len + *n };
        if pos < 0 || pos >= len {
          crate::emit_message(&format!(
            "Indexed::partw: Part {} of {} does not exist.",
            n,
            crate::syntax::expr_to_string(&current)
          ));
          return Some(Ok(Expr::FunctionCall {
            name: "Indexed".to_string(),
            args: args.to_vec().into(),
          }));
        }
        current = items[pos as usize].clone();
      }
      return Some(Ok(current));
    }
    "Part" if args.len() >= 2 => {
      let mut part_expr = Expr::Part {
        expr: Box::new(args[0].clone()),
        index: Box::new(args[1].clone()),
      };
      for idx in &args[2..] {
        part_expr = Expr::Part {
          expr: Box::new(part_expr),
          index: Box::new(idx.clone()),
        };
      }
      return Some(evaluate_expr_to_expr(&part_expr));
    }
    "Insert" if args.len() == 3 => {
      return Some(list_helpers_ast::insert_ast(&args[0], &args[1], &args[2]));
    }
    "Array" if args.len() >= 2 && args.len() <= 4 => {
      if args.len() == 2
        && let Some(n) = expr_to_i128(&args[1])
      {
        return Some(list_helpers_ast::array_ast(&args[0], n));
      }
      if matches!(&args[1], Expr::List(_)) || args.len() > 2 {
        return Some(list_helpers_ast::array_multi_ast(args));
      }
    }
    "Gather" if args.len() == 1 => {
      return Some(list_helpers_ast::gather_ast(&args[0]));
    }
    "GatherBy" if args.len() >= 2 => {
      // GatherBy[list, f1, f2, ...] is equivalent to GatherBy[list, {f1, f2, ...}]
      let func = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::List(args[1..].to_vec().into())
      };
      return Some(list_helpers_ast::gather_by_ast(&func, &args[0]));
    }
    "Split" if args.len() == 1 || args.len() == 2 => {
      if args.len() == 1 {
        return Some(list_helpers_ast::split_ast(&args[0]));
      }
      return Some(list_helpers_ast::split_with_test_ast(&args[0], &args[1]));
    }
    "SplitBy" if args.len() == 2 => {
      return Some(list_helpers_ast::split_by_ast(&args[1], &args[0]));
    }
    "Extract" if args.len() == 2 || args.len() == 3 => {
      let head = if args.len() == 3 {
        Some(&args[2])
      } else {
        None
      };
      return Some(list_helpers_ast::extract_ast(&args[0], &args[1], head));
    }
    "Catenate" if args.len() == 1 => {
      return Some(list_helpers_ast::catenate_ast(&args[0]));
    }
    "Apply" if args.len() == 2 => {
      return Some(list_helpers_ast::apply_ast(&args[0], &args[1]));
    }
    "Apply" if args.len() == 3 => {
      return Some(list_helpers_ast::apply_at_level_ast(
        &args[0], &args[1], &args[2],
      ));
    }
    "MapApply" if args.len() == 2 => {
      return Some(
        crate::evaluator::function_application::apply_map_apply_ast(
          &args[0], &args[1],
        ),
      );
    }
    "Identity" if args.len() == 1 => {
      return Some(list_helpers_ast::identity_ast(&args[0]));
    }
    // Composition[] -> Identity
    "Composition" if args.is_empty() => {
      return Some(Ok(Expr::Identifier("Identity".to_string())));
    }
    // Composition[f] -> f
    "Composition" if args.len() == 1 => {
      return Some(Ok(args[0].clone()));
    }
    // Composition[f, Composition[g, h], k] -> Composition[f, g, h, k]
    "Composition" if args.len() >= 2 => {
      let mut flat = Vec::new();
      for arg in args {
        if let Expr::FunctionCall { name: n, args: a } = arg
          && n == "Composition"
        {
          flat.extend(a.iter().cloned());
          continue;
        }
        flat.push(arg.clone());
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Composition".to_string(),
        args: flat.into(),
      }));
    }
    // RightComposition[] -> Identity
    "RightComposition" if args.is_empty() => {
      return Some(Ok(Expr::Identifier("Identity".to_string())));
    }
    // RightComposition[f] -> f
    "RightComposition" if args.len() == 1 => {
      return Some(Ok(args[0].clone()));
    }
    // RightComposition[f, RightComposition[g, h], k] -> RightComposition[f, g, h, k]
    "RightComposition" if args.len() >= 2 => {
      let mut flat = Vec::new();
      for arg in args {
        if let Expr::FunctionCall { name: n, args: a } = arg
          && n == "RightComposition"
        {
          flat.extend(a.iter().cloned());
          continue;
        }
        flat.push(arg.clone());
      }
      return Some(Ok(Expr::FunctionCall {
        name: "RightComposition".to_string(),
        args: flat.into(),
      }));
    }
    "Outer" if args.len() >= 3 => {
      // Outer[f, list1, list2, ..., n] or Outer[f, list1, list2, ..., n1, n2, ...]
      // Detect trailing integer level specifications.
      let rest = &args[1..];
      // Count how many list args there are (at least 1).
      // Lists come first, then optional integer level specs at the end.
      // We need at least 1 list. Find where integers start from the end.
      let num_rest = rest.len();
      let mut num_level_args = 0;
      for i in (0..num_rest).rev() {
        if matches!(&rest[i], Expr::Integer(_)) {
          num_level_args += 1;
        } else {
          break;
        }
      }
      // Must have at least 1 list arg
      let num_lists = num_rest - num_level_args;
      if num_lists == 0 {
        num_level_args = 0; // all args are lists (integers can be list elements)
      }
      let (lists_in, level_args) = if num_level_args > 0 {
        (&rest[..num_lists], &rest[num_lists..])
      } else {
        (rest, &rest[0..0])
      };

      // Parse level specs
      let levels: Vec<usize> = level_args
        .iter()
        .filter_map(|e| {
          if let Expr::Integer(n) = e {
            Some(*n as usize)
          } else {
            None
          }
        })
        .collect();

      // Convert any SparseArray argument to its Normal (dense-list) form so
      // Outer can treat it as a regular nested list. Wolfram handles the
      // mixed case `Outer[Times, SparseArray[...], {c, d}]` the same way.
      let mut had_sparse = false;
      let lists_owned: Vec<Expr> = lists_in
        .iter()
        .map(|e| {
          if let Expr::FunctionCall {
            name,
            args: sa_args,
          } = e
            && name == "SparseArray"
          {
            had_sparse = true;
            list_helpers_ast::sparse_array_ast(sa_args)
              .unwrap_or_else(|_| e.clone())
          } else {
            e.clone()
          }
        })
        .collect();

      let dense = list_helpers_ast::outer_ast_with_levels(
        &args[0],
        &lists_owned,
        &levels,
      );
      // For `Times` over any SparseArray input, wolframscript collapses the
      // result into a single SparseArray with default 0 (since Times[…, 0]
      // = 0 makes every product involving a zero default to zero).
      // Other heads keep the dense nested form.
      if had_sparse
        && matches!(&args[0], Expr::Identifier(s) if s == "Times")
        && let Ok(d) = &dense
        && let Some(sparse) =
          dense_to_sparse_array_with_default(d, &Expr::Integer(0))
      {
        return Some(Ok(sparse));
      }
      // For non-Times functions, when the LAST argument is a SparseArray,
      // wolframscript wraps the leaf level (corresponding to that last
      // SparseArray's dims) as SparseArray with the function applied.
      // Outer dense iteration is unchanged for the earlier args.
      if !lists_in.is_empty()
        && matches!(lists_in.last(), Some(Expr::FunctionCall { name, .. }) if name == "SparseArray")
        && !matches!(&args[0], Expr::Identifier(s) if s == "Times")
        && let Some(sparse_last) = lists_in.last()
        && let Some(sa_data) = parse_sparse_array_data(sparse_last)
        && let Some(nested) = build_outer_with_sparse_last(
          &args[0],
          &lists_owned[..lists_owned.len() - 1],
          &sa_data,
        )
      {
        return Some(Ok(nested));
      }
      return Some(dense);
    }
    "TensorProduct" if args.len() >= 2 => {
      // TensorProduct[v1, v2, ...] = Outer[Times, v1, v2, ...]
      let times = Expr::Identifier("Times".to_string());
      return Some(list_helpers_ast::outer_ast(&times, args));
    }
    "Inner" if args.len() == 3 => {
      let plus = Expr::Identifier("Plus".to_string());
      return Some(list_helpers_ast::inner_ast(
        &args[0], &args[1], &args[2], &plus,
      ));
    }
    "Inner" if args.len() == 4 => {
      return Some(list_helpers_ast::inner_ast(
        &args[0], &args[1], &args[2], &args[3],
      ));
    }
    "ReplacePart" if args.len() == 2 => {
      return Some(list_helpers_ast::replace_part_ast(&args[0], &args[1]));
    }
    "Nearest" if (2..=3).contains(&args.len()) => {
      return Some(nearest_ast(args));
    }
    // ArrayPad[array, n] — pad with 0
    // ArrayPad[array, n, val] — pad with val
    // ArrayPad[array, {left, right}] — asymmetric padding
    // ArrayPad[array, {left, right}, val] — asymmetric padding with val
    // Negative padding trims elements
    "ArrayPad" if args.len() >= 2 && args.len() <= 3 => {
      return Some(array_pad_ast(args));
    }
    // ArrayReshape[list, {d1, d2, ...}] — reshape a flat list into given dimensions
    // ArrayReshape[list, dims, padding] — pad trailing slots with the given value(s)
    "ArrayReshape" if args.len() == 2 || args.len() == 3 => {
      return Some(array_reshape_ast(args));
    }
    // PositionIndex[list] — association mapping values to their positions
    "PositionIndex" if args.len() == 1 => {
      return Some(position_index_ast(&args[0]));
    }
    // ListConvolve[kernel, list] — discrete convolution
    "ListConvolve" if args.len() == 2 => {
      return Some(list_convolve_ast(&args[0], &args[1]));
    }
    // ListCorrelate[kernel, list] — discrete cross-correlation
    "ListCorrelate" if args.len() == 2 => {
      return Some(list_correlate_ast(&args[0], &args[1]));
    }
    // CountsBy[list, f] — count elements grouped by f
    "CountsBy" if args.len() == 2 => {
      if let Expr::List(ref elems) = args[0] {
        let f = &args[1];
        let mut keys: Vec<Expr> = Vec::new();
        let mut counts: Vec<i128> = Vec::new();
        for elem in elems {
          let key = crate::evaluator::apply_function_to_arg(f, elem)
            .unwrap_or_else(|_| elem.clone());
          let key_str = crate::syntax::expr_to_string(&key);
          if let Some(pos) = keys
            .iter()
            .position(|k| crate::syntax::expr_to_string(k) == key_str)
          {
            counts[pos] += 1;
          } else {
            keys.push(key);
            counts.push(1);
          }
        }
        let pairs: Vec<(Expr, Expr)> = keys
          .into_iter()
          .zip(counts)
          .map(|(k, c)| (k, Expr::Integer(c)))
          .collect();
        return Some(Ok(Expr::Association(pairs)));
      }
    }
    // FoldPairList[f, x, list] — fold with pair output {emit, newState}
    "FoldPairList" if args.len() == 3 => {
      if let Expr::List(ref elems) = args[2] {
        let f = &args[0];
        let mut state = args[1].clone();
        let mut results = Vec::new();
        for elem in elems {
          // Apply f[state, elem] — build function call expression
          let applied = match f {
            Expr::Function { body } => crate::syntax::substitute_slots(
              body,
              &[state.clone(), elem.clone()],
            ),
            Expr::Identifier(fname) => Expr::FunctionCall {
              name: fname.clone(),
              args: vec![state.clone(), elem.clone()].into(),
            },
            _ => Expr::FunctionCall {
              name: expr_to_string(f),
              args: vec![state.clone(), elem.clone()].into(),
            },
          };
          let result = crate::evaluator::evaluate_expr_to_expr(&applied)
            .unwrap_or(applied);
          if let Expr::List(ref pair) = result {
            if pair.len() == 2 {
              results.push(pair[0].clone());
              state = pair[1].clone();
            } else {
              return Some(Ok(result));
            }
          } else {
            return Some(Ok(result));
          }
        }
        return Some(Ok(Expr::List(results.into())));
      }
    }
    // JoinAcross[list1, list2, key] — join associations on a common key
    "JoinAcross" if args.len() == 3 => {
      if let (Expr::List(l1), Expr::List(l2)) = (&args[0], &args[1]) {
        let key_str = crate::syntax::expr_to_string(&args[2]);
        let mut results = Vec::new();
        for a1 in l1 {
          let key_val = get_assoc_value(a1, &key_str);
          if let Some(ref kv) = key_val {
            for a2 in l2 {
              let key_val2 = get_assoc_value(a2, &key_str);
              if let Some(ref kv2) = key_val2
                && crate::syntax::expr_to_string(kv)
                  == crate::syntax::expr_to_string(kv2)
              {
                // Merge the two associations
                let merged = merge_associations(a1, a2);
                results.push(merged);
              }
            }
          }
        }
        return Some(Ok(Expr::List(results.into())));
      }
    }
    // CountDistinct[list] — count unique elements
    "CountDistinct" if args.len() == 1 => {
      if let Expr::List(ref elems) = args[0] {
        let mut seen = std::collections::HashSet::new();
        for e in elems {
          seen.insert(expr_to_string(e));
        }
        return Some(Ok(Expr::Integer(seen.len() as i128)));
      }
    }
    // SequencePosition[list, sublist] — find positions of subsequence (overlapping)
    "SequencePosition" if args.len() == 2 => {
      if !matches!(&args[0], Expr::List(_)) {
        crate::emit_message(&format!(
          "SequencePosition::list: List expected at position 1 in SequencePosition[{}, {}].",
          crate::syntax::expr_to_string(&args[0]),
          crate::syntax::expr_to_string(&args[1])
        ));
        return Some(Ok(Expr::FunctionCall {
          name: "SequencePosition".to_string(),
          args: args.to_vec().into(),
        }));
      }
      if let (Expr::List(list), Expr::List(sub)) = (&args[0], &args[1]) {
        if sub.is_empty() {
          return Some(Ok(Expr::List(vec![].into())));
        }
        let sub_len = sub.len();
        let sub_strs: Vec<String> = sub.iter().map(expr_to_string).collect();
        let mut results: Vec<Expr> = Vec::new();
        for i in 0..list.len() {
          if i + sub_len > list.len() {
            break;
          }
          let mut matches = true;
          for j in 0..sub_len {
            if expr_to_string(&list[i + j]) != sub_strs[j] {
              matches = false;
              break;
            }
          }
          if matches {
            results.push(Expr::List(
              vec![
                Expr::Integer((i + 1) as i128),
                Expr::Integer((i + sub_len) as i128),
              ]
              .into(),
            ));
          }
        }
        return Some(Ok(Expr::List(results.into())));
      }
    }
    // SequenceCases[list, sublist] — find matching subsequences
    // Supports: plain list, Condition[list, test], Rule/RuleDelayed[list, rhs]
    "SequenceCases" if args.len() == 2 => {
      if !matches!(&args[0], Expr::List(_)) {
        crate::emit_message(&format!(
          "SequenceCases::list: List expected at position 1 in SequenceCases[{}, {}].",
          crate::syntax::expr_to_string(&args[0]),
          crate::syntax::expr_to_string(&args[1])
        ));
        return Some(Ok(Expr::FunctionCall {
          name: "SequenceCases".to_string(),
          args: args.to_vec().into(),
        }));
      }
      if let Expr::List(list) = &args[0] {
        // Extract the list pattern and optional replacement from
        // Condition, Rule, or RuleDelayed wrappers
        let (match_pat, replacement) = match &args[1] {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => (pattern.as_ref(), Some(replacement.as_ref())),
          _ => (&args[1], None),
        };

        // Unwrap `Pattern[name, inner]` (from `name : inner` binding) so the
        // inner list-pattern controls length calculations; `match_pat` keeps
        // the Pattern so bindings still flow through `match_pattern`.
        let mut list_pat = match_pat;
        loop {
          match list_pat {
            Expr::FunctionCall {
              name,
              args: inner_args,
            } if name == "Pattern" && inner_args.len() == 2 => {
              list_pat = &inner_args[1];
            }
            Expr::FunctionCall {
              name,
              args: cond_args,
            } if name == "Condition" && cond_args.len() == 2 => {
              list_pat = &cond_args[0];
            }
            _ => break,
          }
        }

        // Get the sub-elements for length calculations
        let sub = match list_pat {
          Expr::List(items) => items,
          _ => return Some(Ok(Expr::List(vec![].into()))),
        };

        if sub.is_empty() {
          return Some(Ok(Expr::List(vec![].into())));
        }

        let has_patterns = sub.iter().any(has_pattern_element);
        let has_sequence = sub.iter().any(has_sequence_pattern);

        if has_patterns {
          let mut results: Vec<Expr> = Vec::new();
          let mut i = 0;
          while i < list.len() {
            let mut matched = false;
            let remaining = list.len() - i;
            let min_len = if has_sequence { 1 } else { sub.len() };
            let try_max = if has_sequence { remaining } else { sub.len() };
            if remaining < min_len {
              break;
            }
            let try_max = try_max.min(remaining);
            let range: Vec<usize> = (min_len..=try_max).rev().collect();
            for len in range {
              let subseq = Expr::List(list[i..i + len].to_vec().into());
              // Use match_pattern which handles Condition properly
              if let Some(bindings) =
                crate::evaluator::pattern_matching::match_pattern(
                  &subseq, match_pat,
                )
              {
                if let Some(repl) = replacement {
                  // Rule/RuleDelayed: apply bindings to replacement
                  match crate::evaluator::pattern_matching::apply_bindings(
                    repl, &bindings,
                  ) {
                    Ok(result) => results.push(result),
                    Err(_) => results.push(subseq),
                  }
                } else {
                  results.push(subseq);
                }
                i += len;
                matched = true;
                break;
              }
            }
            if !matched {
              i += 1;
            }
          }
          return Some(Ok(Expr::List(results.into())));
        } else {
          // Literal subsequence match
          let sub_len = sub.len();
          let sub_strs: Vec<String> = sub.iter().map(expr_to_string).collect();
          let mut results: Vec<Expr> = Vec::new();
          let mut i = 0;
          while i + sub_len <= list.len() {
            let mut matches = true;
            for j in 0..sub_len {
              if expr_to_string(&list[i + j]) != sub_strs[j] {
                matches = false;
                break;
              }
            }
            if matches {
              results.push(Expr::List(list[i..i + sub_len].to_vec().into()));
              i += sub_len;
            } else {
              i += 1;
            }
          }
          return Some(Ok(Expr::List(results.into())));
        }
      }
    }
    // SequenceCount[list, sublist] — count non-overlapping occurrences
    "SequenceCount" if args.len() == 2 => {
      if let (Expr::List(list), Expr::List(sub)) = (&args[0], &args[1]) {
        if sub.is_empty() {
          return Some(Ok(Expr::Integer(0)));
        }
        let sub_len = sub.len();
        let sub_strs: Vec<String> = sub.iter().map(expr_to_string).collect();
        let mut count = 0i128;
        let mut i = 0;
        while i + sub_len <= list.len() {
          let mut matches = true;
          for j in 0..sub_len {
            if expr_to_string(&list[i + j]) != sub_strs[j] {
              matches = false;
              break;
            }
          }
          if matches {
            count += 1;
            i += sub_len;
          } else {
            i += 1;
          }
        }
        return Some(Ok(Expr::Integer(count)));
      }
    }
    // KeySortBy[assoc, f] — sort association by applying f to keys
    "KeySortBy" if args.len() == 2 => {
      if let Expr::Association(pairs) = &args[0] {
        let func = &args[1];
        let mut indexed: Vec<(usize, Expr)> = pairs
          .iter()
          .enumerate()
          .map(|(i, (k, _))| {
            let applied = evaluate_expr_to_expr(&Expr::FunctionCall {
              name: expr_to_string(func),
              args: vec![k.clone()].into(),
            })
            .unwrap_or(k.clone());
            (i, applied)
          })
          .collect();
        indexed.sort_by(|a, b| {
          let sa = expr_to_string(&a.1);
          let sb = expr_to_string(&b.1);
          // Try numeric comparison first
          let na: Result<f64, _> = sa.parse();
          let nb: Result<f64, _> = sb.parse();
          if let (Ok(a_val), Ok(b_val)) = (na, nb) {
            a_val
              .partial_cmp(&b_val)
              .unwrap_or(std::cmp::Ordering::Equal)
          } else {
            sa.cmp(&sb)
          }
        });
        let sorted_pairs: Vec<(Expr, Expr)> =
          indexed.iter().map(|(i, _)| pairs[*i].clone()).collect();
        return Some(Ok(Expr::Association(sorted_pairs)));
      }
    }

    // CenterArray[nspec]            ≡ CenterArray[1, nspec]
    // CenterArray[a, nspec]         centers a within an array of dimensions nspec
    // CenterArray[a, nspec, pad]    pads with `pad` instead of 0
    //
    // a is either a scalar or a nested rectangular list; nspec is an
    // Integer (1-D) or a list of Integers (multi-D). For each dimension,
    // left padding is floor((n - k)/2) and right padding is the rest, so
    // scalars in an even-length dimension sit just left of middle.
    "CenterArray" if (1..=3).contains(&args.len()) => {
      // Normalise to (a, nspec, pad).
      let (a, nspec, pad) = if args.len() == 1 {
        (Expr::Integer(1), args[0].clone(), Expr::Integer(0))
      } else if args.len() == 2 {
        (args[0].clone(), args[1].clone(), Expr::Integer(0))
      } else {
        (args[0].clone(), args[1].clone(), args[2].clone())
      };
      // Parse nspec into a list of dimension lengths.
      let dims: Vec<usize> = match &nspec {
        Expr::Integer(n) if *n >= 0 => vec![*n as usize],
        Expr::List(items) => {
          let mut out = Vec::with_capacity(items.len());
          let mut ok = true;
          for item in items.iter() {
            if let Expr::Integer(n) = item
              && *n >= 0
            {
              out.push(*n as usize);
            } else {
              ok = false;
              break;
            }
          }
          if !ok {
            return None;
          }
          out
        }
        _ => return None,
      };
      // Inspect the block's shape against the requested rank. A scalar
      // input is treated as a rank-0 block; lists must be exactly as
      // deeply nested as `dims` and rectangular at every level.
      fn block_shape(e: &Expr, depth: usize) -> Option<Vec<usize>> {
        if depth == 0 {
          if matches!(e, Expr::List(_)) {
            return None;
          }
          return Some(Vec::new());
        }
        let Expr::List(items) = e else {
          return None;
        };
        let mut shape = vec![items.len()];
        if items.is_empty() {
          for _ in 1..depth {
            shape.push(0);
          }
          return Some(shape);
        }
        let inner_shape = block_shape(&items[0], depth - 1)?;
        for child in items.iter().skip(1) {
          if block_shape(child, depth - 1).as_deref() != Some(&inner_shape) {
            return None;
          }
        }
        shape.extend(inner_shape);
        Some(shape)
      }
      // If a is a scalar (non-List), it's a rank-0 block. If a is a
      // list, we expect its rank to match `dims.len()`.
      let block_dims: Vec<usize> = if !matches!(a, Expr::List(_)) {
        vec![1; dims.len()]
      } else {
        block_shape(&a, dims.len())?
      };
      // Per-dimension layout: how much to truncate from the block's
      // front (block_start), the effective block size after truncation
      // (effective_k), and how many pad cells go before/after the block
      // along this axis.
      struct Axis {
        block_start: usize,
        effective_k: usize,
        left_pad: usize,
      }
      let axes: Vec<Axis> = dims
        .iter()
        .zip(block_dims.iter())
        .map(|(d, k)| {
          if *k <= *d {
            Axis {
              block_start: 0,
              effective_k: *k,
              left_pad: (*d - *k) / 2,
            }
          } else {
            // Truncate the block; div_ceil matches wolframscript on
            // odd parity (CenterArray[{a,b,c}, 2] == {b, c}).
            Axis {
              block_start: (*k - *d).div_ceil(2),
              effective_k: *d,
              left_pad: 0,
            }
          }
        })
        .collect();
      // Fetch an element of the (possibly nested) block at a multi-index.
      fn fetch(block: &Expr, idx: &[usize]) -> Expr {
        if idx.is_empty() {
          return block.clone();
        }
        if let Expr::List(items) = block {
          return fetch(&items[idx[0]], &idx[1..]);
        }
        block.clone()
      }
      fn build(
        dims: &[usize],
        axes: &[Axis],
        block: &Expr,
        pad: &Expr,
        block_idx: &mut Vec<usize>,
        is_scalar_block: bool,
      ) -> Expr {
        if dims.is_empty() {
          if is_scalar_block {
            return block.clone();
          }
          return fetch(block, block_idx);
        }
        let n = dims[0];
        let ax = &axes[0];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
          if i >= ax.left_pad && i < ax.left_pad + ax.effective_k {
            block_idx.push(ax.block_start + (i - ax.left_pad));
            out.push(build(
              &dims[1..],
              &axes[1..],
              block,
              pad,
              block_idx,
              is_scalar_block,
            ));
            block_idx.pop();
          } else {
            out.push(pad_block(&dims[1..], pad));
          }
        }
        Expr::List(out.into())
      }
      fn pad_block(dims: &[usize], pad: &Expr) -> Expr {
        if dims.is_empty() {
          return pad.clone();
        }
        let inner = pad_block(&dims[1..], pad);
        Expr::List(vec![inner; dims[0]].into())
      }
      let is_scalar = !matches!(a, Expr::List(_));
      let mut block_idx = Vec::with_capacity(dims.len());
      return Some(Ok(build(
        &dims,
        &axes,
        &a,
        &pad,
        &mut block_idx,
        is_scalar,
      )));
    }
    // ReverseSortBy[list, f] — sort list in reverse order by applying f
    "ReverseSortBy" if args.len() == 2 => {
      if let Expr::List(items) = &args[0] {
        let func = &args[1];
        let mut indexed: Vec<(usize, String)> = items
          .iter()
          .enumerate()
          .map(|(i, item)| {
            let key = apply_function_to_arg(func, item)
              .unwrap_or_else(|_| item.clone());
            let key_str = expr_to_string(&key);
            (i, key_str)
          })
          .collect();
        indexed.sort_by(|a, b| {
          let fa = a.1.parse::<f64>();
          let fb = b.1.parse::<f64>();
          if let (Ok(va), Ok(vb)) = (fa, fb) {
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
          } else {
            b.1.cmp(&a.1)
          }
        });
        let result: Vec<Expr> =
          indexed.iter().map(|(i, _)| items[*i].clone()).collect();
        return Some(Ok(Expr::List(result.into())));
      }
    }
    // IntersectingQ[list1, list2] — True if lists share any element
    "IntersectingQ" if args.len() == 2 => {
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1]) {
        let a_strs: Vec<String> = a.iter().map(expr_to_string).collect();
        let has_common = b.iter().any(|e| a_strs.contains(&expr_to_string(e)));
        return Some(Ok(Expr::Identifier(
          if has_common { "True" } else { "False" }.to_string(),
        )));
      }
    }
    // DisjointQ[list1, list2] — True if lists share no common elements
    "DisjointQ" if args.len() == 2 => {
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1]) {
        let a_strs: Vec<String> = a.iter().map(expr_to_string).collect();
        let has_common = b.iter().any(|e| a_strs.contains(&expr_to_string(e)));
        return Some(Ok(Expr::Identifier(
          if has_common { "False" } else { "True" }.to_string(),
        )));
      }
    }
    // FindPermutation[e1, e2] — find permutation that maps e1 to e2.
    // Accepts Lists or any two FunctionCalls with the same head.
    "FindPermutation" if args.len() == 2 => {
      // Pull `&[Expr]` from a List or from a FunctionCall, requiring both
      // sides to share the same head (List vs List, or head[..] vs head[..]).
      fn elements(expr: &Expr) -> Option<(Option<&str>, &[Expr])> {
        match expr {
          Expr::List(items) => Some((None, items)),
          Expr::FunctionCall { name, args } => {
            Some((Some(name.as_str()), args))
          }
          _ => None,
        }
      }
      let (left, right) = (elements(&args[0]), elements(&args[1]));
      if let (Some((hl, a)), Some((hr, b))) = (left, right)
        && hl == hr
        && a.len() == b.len()
      {
        let n = a.len();
        let a_strs: Vec<String> = a.iter().map(expr_to_string).collect();
        let b_strs: Vec<String> = b.iter().map(expr_to_string).collect();
        let mut perm = vec![0usize; n];
        let mut valid = true;
        for (i, bs) in b_strs.iter().enumerate() {
          if let Some(pos) = a_strs.iter().position(|x| x == bs) {
            perm[pos] = i + 1;
          } else {
            valid = false;
            break;
          }
        }
        if valid {
          // Convert to cycles notation
          let mut visited = vec![false; n];
          let mut cycles = Vec::new();
          for start in 0..n {
            if visited[start] || perm[start] == start + 1 {
              visited[start] = true;
              continue;
            }
            let mut cycle = Vec::new();
            let mut curr = start;
            while !visited[curr] {
              visited[curr] = true;
              cycle.push(Expr::Integer((curr + 1) as i128));
              curr = perm[curr] - 1;
            }
            if cycle.len() > 1 {
              cycles.push(Expr::List(cycle.into()));
            }
          }
          return Some(Ok(Expr::FunctionCall {
            name: "Cycles".to_string(),
            args: vec![Expr::List(cycles.into())].into(),
          }));
        }
      }
    }
    // KeyMemberQ[assoc, key] — True if key exists in association
    "KeyMemberQ" if args.len() == 2 => {
      if let Expr::Association(pairs) = &args[0] {
        let key_str = expr_to_string(&args[1]);
        let found = pairs.iter().any(|(k, _)| expr_to_string(k) == key_str);
        return Some(Ok(Expr::Identifier(
          if found { "True" } else { "False" }.to_string(),
        )));
      }
    }
    // Cycles[{cyc1, ...}] — canonicalise: drop length-1 cycles, rotate
    // each cycle to start with its smallest element, sort cycles by
    // that first element. Matches Mathematica's canonical form so
    // structurally distinct inputs like Cycles[{{4, 10, 2, 5}, {9}}]
    // and Cycles[{{2, 5, 4, 10}}] compare equal.
    "Cycles" if args.len() == 1 => {
      if let Some(cycles) = cycles_arg(&Expr::FunctionCall {
        name: "Cycles".to_string(),
        args: args.to_vec().into(),
      }) {
        let mut canonical: Vec<Vec<i128>> = Vec::with_capacity(cycles.len());
        for cycle in cycles.iter() {
          if cycle.len() <= 1 {
            continue;
          }
          // Rotate so the smallest element comes first.
          let (min_idx, _) =
            cycle.iter().enumerate().min_by_key(|(_, v)| **v).unwrap();
          let mut rotated: Vec<i128> = cycle[min_idx..].to_vec();
          rotated.extend_from_slice(&cycle[..min_idx]);
          canonical.push(rotated);
        }
        // Sort cycles by their first element.
        canonical.sort_by_key(|c| c[0]);
        let cycle_exprs: Vec<Expr> = canonical
          .into_iter()
          .map(|c| {
            Expr::List(
              c.into_iter().map(Expr::Integer).collect::<Vec<_>>().into(),
            )
          })
          .collect();
        return Some(Ok(Expr::FunctionCall {
          name: "Cycles".to_string(),
          args: vec![Expr::List(cycle_exprs.into())].into(),
        }));
      }
    }

    // PermutationOrder[perm] — order (smallest n such that perm^n = identity)
    "PermutationOrder" if args.len() == 1 => {
      // Cycles[{cycle1, ...}] form: order is LCM of cycle lengths.
      if let Some(cycles) = cycles_arg(&args[0]) {
        let mut order: i128 = 1;
        for cycle in cycles {
          let len = cycle.len() as i128;
          if len > 0 {
            order = lcm_i128(order, len);
          }
        }
        return Some(Ok(Expr::Integer(order)));
      }
      if let Expr::List(perm) = &args[0] {
        // Permutation as list form
        let n = perm.len();
        let mut indices = Vec::with_capacity(n);
        let mut valid = true;
        for p in perm {
          if let Expr::Integer(v) = p {
            indices.push(*v as usize);
          } else {
            valid = false;
            break;
          }
        }
        if valid {
          // Find cycle lengths, order = LCM of cycle lengths
          let mut visited = vec![false; n];
          let mut order: i128 = 1;
          for start in 0..n {
            if visited[start] {
              continue;
            }
            let mut cycle_len: i128 = 0;
            let mut curr = start;
            while !visited[curr] {
              visited[curr] = true;
              cycle_len += 1;
              curr = indices[curr] - 1;
            }
            order = lcm_i128(order, cycle_len);
          }
          return Some(Ok(Expr::Integer(order)));
        }
      }
    }
    // PermutationPower[perm, n] — apply permutation n times
    "PermutationPower" if args.len() == 2 => {
      if let (Expr::List(perm), Some(n)) = (&args[0], expr_to_i128(&args[1])) {
        let len = perm.len();
        let mut indices = Vec::with_capacity(len);
        let mut valid = true;
        for p in perm {
          if let Expr::Integer(v) = p {
            indices.push(*v as usize);
          } else {
            valid = false;
            break;
          }
        }
        if valid {
          // For negative n, use inverse first
          let (indices, n) = if n < 0 {
            // Compute inverse
            let mut inv = vec![0usize; len];
            for (i, &idx) in indices.iter().enumerate() {
              inv[idx - 1] = i + 1;
            }
            (inv, -n)
          } else {
            (indices, n)
          };
          // Apply permutation n times efficiently using cycle decomposition
          let mut result = vec![0usize; len];
          let mut visited = vec![false; len];
          for start in 0..len {
            if visited[start] {
              continue;
            }
            // Trace cycle
            let mut cycle = Vec::new();
            let mut curr = start;
            while !visited[curr] {
              visited[curr] = true;
              cycle.push(curr);
              curr = indices[curr] - 1;
            }
            let cycle_len = cycle.len();
            let shift = (n as usize) % cycle_len;
            for (i, &pos) in cycle.iter().enumerate() {
              result[pos] = cycle[(i + shift) % cycle_len] + 1;
            }
          }
          let result_exprs: Vec<Expr> = result
            .into_iter()
            .map(|v| Expr::Integer(v as i128))
            .collect();
          return Some(Ok(Expr::List(result_exprs.into())));
        }
      }
      // PermutationPower[Cycles[{...}], n] — apply cycle-form permutation n times,
      // returning canonical Cycles form.
      if let (
        Expr::FunctionCall {
          name: cname,
          args: cargs,
        },
        Some(n),
      ) = (&args[0], expr_to_i128(&args[1]))
        && cname == "Cycles"
        && cargs.len() == 1
        && let Expr::List(cycle_list) = &cargs[0]
      {
        let mut max_elem: usize = 0;
        let mut valid = true;
        for cycle in cycle_list.iter() {
          let Expr::List(c) = cycle else {
            valid = false;
            break;
          };
          for e in c.iter() {
            if let Expr::Integer(v) = e {
              if *v >= 1 {
                let u = *v as usize;
                if u > max_elem {
                  max_elem = u;
                }
              } else {
                valid = false;
                break;
              }
            } else {
              valid = false;
              break;
            }
          }
          if !valid {
            break;
          }
        }
        if valid {
          // Build the underlying permutation map perm[i] = σ(i) for i in 1..=N
          let mut perm: Vec<usize> = (0..=max_elem).collect();
          for cycle in cycle_list.iter() {
            if let Expr::List(c) = cycle {
              let ints: Vec<usize> = c
                .iter()
                .filter_map(|e| {
                  if let Expr::Integer(v) = e {
                    Some(*v as usize)
                  } else {
                    None
                  }
                })
                .collect();
              if ints.len() >= 2 {
                for i in 0..ints.len() - 1 {
                  perm[ints[i]] = ints[i + 1];
                }
                perm[ints[ints.len() - 1]] = ints[0];
              }
            }
          }
          // Invert if n < 0
          let (perm, n_abs) = if n < 0 {
            let mut inv = vec![0usize; max_elem + 1];
            for i in 1..=max_elem {
              inv[perm[i]] = i;
            }
            inv[0] = 0;
            (inv, (-n) as u128)
          } else {
            (perm, n as u128)
          };
          // Compute σ^n_abs by decomposing into disjoint cycles and shifting
          // each cycle by n_abs mod cycle_len. This is the same trick used
          // for the list-form branch but adapted to a 1-indexed map.
          let mut result: Vec<usize> = (0..=max_elem).collect();
          let mut visited = vec![false; max_elem + 1];
          visited[0] = true;
          for start in 1..=max_elem {
            if visited[start] {
              continue;
            }
            let mut cycle = Vec::new();
            let mut curr = start;
            while !visited[curr] {
              visited[curr] = true;
              cycle.push(curr);
              curr = perm[curr];
            }
            let cycle_len = cycle.len();
            let shift = (n_abs % cycle_len as u128) as usize;
            for (i, &pos) in cycle.iter().enumerate() {
              result[pos] = cycle[(i + shift) % cycle_len];
            }
          }
          // Build canonical Cycles from result: extract cycles, skip fixed
          // points, rotate each so its smallest element comes first, and
          // sort cycles by smallest element.
          let mut visited2 = vec![false; max_elem + 1];
          visited2[0] = true;
          let mut out_cycles: Vec<Vec<i128>> = Vec::new();
          for start in 1..=max_elem {
            if visited2[start] {
              continue;
            }
            let mut cycle = Vec::new();
            let mut curr = start;
            while !visited2[curr] {
              visited2[curr] = true;
              cycle.push(curr as i128);
              curr = result[curr];
            }
            if cycle.len() >= 2 {
              let min_idx = cycle
                .iter()
                .enumerate()
                .min_by_key(|(_, v)| *v)
                .map(|(i, _)| i)
                .unwrap_or(0);
              cycle.rotate_left(min_idx);
              out_cycles.push(cycle);
            }
          }
          out_cycles.sort_by_key(|c| c[0]);
          let cycle_exprs: Vec<Expr> = out_cycles
            .into_iter()
            .map(|c| Expr::List(c.into_iter().map(Expr::Integer).collect()))
            .collect();
          return Some(Ok(Expr::FunctionCall {
            name: "Cycles".to_string(),
            args: vec![Expr::List(cycle_exprs.into())].into(),
          }));
        }
      }
    }
    // PermutationLength[perm] — number of non-fixed points
    "PermutationLength" if args.len() == 1 => {
      // Cycles form: sum of cycle lengths (Cycles canonicalises away
      // length-1 cycles, so every listed element is moved).
      if let Some(cycles) = cycles_arg(&args[0]) {
        let total: i128 = cycles.iter().map(|c| c.len() as i128).sum();
        return Some(Ok(Expr::Integer(total)));
      }
      if let Expr::List(perm) = &args[0] {
        let mut count: i128 = 0;
        for (i, p) in perm.iter().enumerate() {
          if let Expr::Integer(v) = p
            && *v as usize != i + 1
          {
            count += 1;
          }
        }
        return Some(Ok(Expr::Integer(count)));
      }
    }
    // PermutationListQ[list] — True if list is a valid permutation
    "PermutationListQ" if args.len() == 1 => {
      if let Expr::List(perm) = &args[0] {
        let n = perm.len();
        let mut seen = vec![false; n + 1];
        let mut valid = true;
        for p in perm {
          if let Expr::Integer(v) = p {
            let v = *v as usize;
            if v >= 1 && v <= n && !seen[v] {
              seen[v] = true;
            } else {
              valid = false;
              break;
            }
          } else {
            valid = false;
            break;
          }
        }
        return Some(Ok(Expr::Identifier(
          if valid { "True" } else { "False" }.to_string(),
        )));
      }
      // Non-list input
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    // FoldWhileList[f, x, list, test] — fold while test is True, returning intermediate results
    "FoldWhileList" if args.len() == 4 => {
      if let Expr::List(items) = &args[2] {
        let f = &args[0];
        let mut acc = args[1].clone();
        let test = &args[3];
        let mut results = vec![acc.clone()];
        for item in items {
          // Build f[acc, item] and evaluate
          let call = match f {
            Expr::Identifier(name) => Expr::FunctionCall {
              name: name.clone(),
              args: vec![acc.clone(), item.clone()].into(),
            },
            Expr::Function { body } => crate::syntax::substitute_slots(
              body,
              &[acc.clone(), item.clone()],
            ),
            _ => Expr::FunctionCall {
              name: expr_to_string(f),
              args: vec![acc.clone(), item.clone()].into(),
            },
          };
          let new_acc = evaluate_expr_to_expr(&call).unwrap_or(call);
          // Test the new value
          // Include the new value, then check the test.
          // If the test fails, we still include this value (Wolfram behavior).
          acc = new_acc;
          results.push(acc.clone());
          let test_result = apply_function_to_arg(test, &acc)
            .unwrap_or(Expr::Identifier("False".to_string()));
          let test_str = expr_to_string(&test_result);
          if test_str != "True" {
            break;
          }
        }
        return Some(Ok(Expr::List(results.into())));
      }
    }
    // PermutationCyclesQ[Cycles[{...}]] — True if valid Cycles form
    "PermutationCyclesQ" if args.len() == 1 => {
      if let Expr::FunctionCall {
        name: cname,
        args: cargs,
      } = &args[0]
        && cname == "Cycles"
        && cargs.len() == 1
        && let Expr::List(cycles) = &cargs[0]
      {
        let mut valid = true;
        let mut seen = std::collections::HashSet::new();
        for cycle in cycles {
          if let Expr::List(c) = cycle {
            for elem in c {
              if let Expr::Integer(v) = elem {
                if *v < 1 || !seen.insert(*v) {
                  valid = false;
                  break;
                }
              } else {
                valid = false;
                break;
              }
            }
          } else {
            valid = false;
          }
          if !valid {
            break;
          }
        }
        return Some(Ok(Expr::Identifier(
          if valid { "True" } else { "False" }.to_string(),
        )));
      }
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    // PermutationSupport[perm] — set of elements moved by the permutation
    "PermutationSupport" if args.len() == 1 => {
      // Cycles form: sorted union of integers across all cycles.
      if let Some(cycles) = cycles_arg(&args[0]) {
        let mut support: Vec<i128> = cycles.iter().flatten().copied().collect();
        support.sort();
        support.dedup();
        return Some(Ok(Expr::List(
          support
            .into_iter()
            .map(Expr::Integer)
            .collect::<Vec<_>>()
            .into(),
        )));
      }
      if let Expr::List(perm) = &args[0] {
        let mut support = Vec::new();
        for (i, p) in perm.iter().enumerate() {
          if let Expr::Integer(v) = p
            && *v as usize != i + 1
          {
            support.push(Expr::Integer((i + 1) as i128));
          }
        }
        return Some(Ok(Expr::List(support.into())));
      }
    }
    // PermutationMax[perm] — largest element moved by the permutation
    "PermutationMax" if args.len() == 1 => {
      // Cycles form: max element across all cycles.
      if let Some(cycles) = cycles_arg(&args[0]) {
        let max_val = cycles.iter().flatten().copied().max();
        return Some(Ok(Expr::Integer(max_val.unwrap_or(0))));
      }
      if let Expr::List(perm) = &args[0] {
        let mut max_val: Option<i128> = None;
        for (i, p) in perm.iter().enumerate() {
          if let Expr::Integer(v) = p
            && *v as usize != i + 1
          {
            let idx = (i + 1) as i128;
            max_val = Some(max_val.map_or(idx, |m: i128| m.max(idx)));
          }
        }
        if let Some(m) = max_val {
          return Some(Ok(Expr::Integer(m)));
        }
        return Some(Ok(Expr::Integer(0)));
      }
    }
    // PermutationMin[perm] — smallest element moved by the permutation
    "PermutationMin" if args.len() == 1 => {
      // Cycles form: min element across all cycles.
      if let Some(cycles) = cycles_arg(&args[0]) {
        if let Some(min_val) = cycles.iter().flatten().copied().min() {
          return Some(Ok(Expr::Integer(min_val)));
        }
        // Empty Cycles → wolframscript returns Infinity (no moved points).
        return Some(Ok(Expr::Identifier("Infinity".to_string())));
      }
      if let Expr::List(perm) = &args[0] {
        let mut min_val: Option<i128> = None;
        for (i, p) in perm.iter().enumerate() {
          if let Expr::Integer(v) = p
            && *v as usize != i + 1
          {
            let idx = (i + 1) as i128;
            min_val = Some(min_val.map_or(idx, |m: i128| m.min(idx)));
          }
        }
        if let Some(m) = min_val {
          return Some(Ok(Expr::Integer(m)));
        }
        return Some(Ok(Expr::FunctionCall {
          name: "Infinity".to_string(),
          args: vec![].into(),
        }));
      }
    }
    // Splice[list] and Splice[list, head] — stay unevaluated; splicing is done
    // by the enclosing context (List evaluation or flatten_sequences).
    "Splice" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "Splice".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // SubsetMap[f, list, positions] — apply f to elements at positions, put results back
    "SubsetMap" if args.len() == 3 => {
      if let (Expr::List(items), Expr::List(positions)) = (&args[1], &args[2]) {
        let f = &args[0];
        // Extract elements at given positions
        let pos_indices: Vec<usize> = positions
          .iter()
          .filter_map(|p| {
            if let Expr::Integer(v) = p {
              Some(*v as usize)
            } else {
              None
            }
          })
          .collect();
        let subset: Vec<Expr> = pos_indices
          .iter()
          .filter_map(|&idx| items.get(idx - 1).cloned())
          .collect();
        // Apply f to the subset
        let mapped = apply_function_to_arg(f, &Expr::List(subset.into()))
          .unwrap_or(Expr::List(vec![].into()));
        // Put results back
        if let Expr::List(mapped_items) = &mapped {
          let mut result = items.clone();
          for (i, &pos) in pos_indices.iter().enumerate() {
            if pos >= 1 && pos <= result.len() && i < mapped_items.len() {
              result[pos - 1] = mapped_items[i].clone();
            }
          }
          return Some(Ok(Expr::List(result)));
        }
      }
    }
    // Assert — returns unevaluated (matches Wolfram default behavior without AssertTools package)
    _ => {}
  }
  None
}

/// Extract a value from an association by key string
fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

fn lcm_i128(a: i128, b: i128) -> i128 {
  if a == 0 || b == 0 {
    return 0;
  }
  let g = gcd_i128(a.abs(), b.abs());
  (a / g * b).abs()
}

/// If `expr` is `Cycles[{{...}, {...}, ...}]` with all-integer cycles,
/// return each cycle as `Vec<i128>`. Returns `None` for any other shape
/// so callers can fall through to list-form handling.
fn cycles_arg(expr: &Expr) -> Option<Vec<Vec<i128>>> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Cycles" || args.len() != 1 {
    return None;
  }
  let Expr::List(cycles) = &args[0] else {
    return None;
  };
  let mut result = Vec::with_capacity(cycles.len());
  for cycle in cycles.iter() {
    let Expr::List(items) = cycle else {
      return None;
    };
    let mut nums = Vec::with_capacity(items.len());
    for item in items.iter() {
      if let Expr::Integer(n) = item {
        nums.push(*n);
      } else {
        return None;
      }
    }
    result.push(nums);
  }
  Some(result)
}

fn get_assoc_value(assoc: &Expr, key: &str) -> Option<Expr> {
  if let Expr::Association(pairs) = assoc {
    for (k, v) in pairs {
      let k_str = expr_to_string(k);
      if k_str == key || k_str == key.trim_matches('"') {
        return Some(v.clone());
      }
    }
  }
  None
}

/// Merge two associations, with the first taking priority for duplicate keys
fn merge_associations(a1: &Expr, a2: &Expr) -> Expr {
  let mut pairs: Vec<(Expr, Expr)> = Vec::new();
  let mut seen_keys: Vec<String> = Vec::new();

  if let Expr::Association(items) = a1 {
    for (k, v) in items {
      let k_str = expr_to_string(k);
      seen_keys.push(k_str);
      pairs.push((k.clone(), v.clone()));
    }
  }

  if let Expr::Association(items) = a2 {
    for (k, v) in items {
      let k_str = expr_to_string(k);
      if !seen_keys.contains(&k_str) {
        pairs.push((k.clone(), v.clone()));
      }
    }
  }

  Expr::Association(pairs)
}

fn array_pad_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let arr = &args[0];
  let pad_val = if args.len() >= 3 {
    args[2].clone()
  } else {
    Expr::Integer(0)
  };
  let unevaluated = || Expr::FunctionCall {
    name: "ArrayPad".to_string(),
    args: args.to_vec().into(),
  };

  // Try per-dimension form: {{m1, n1}, {m2, n2}, ...} or {{m1}, {m2}, ...}
  // where the spec list has length equal to the array's rank.
  if let Expr::List(spec_items) = &args[1]
    && spec_items.iter().all(|s| matches!(s, Expr::List(_)))
    && !spec_items.is_empty()
  {
    let mut per_dim: Vec<(i128, i128)> = Vec::with_capacity(spec_items.len());
    for s in spec_items.iter() {
      let Expr::List(inner) = s else {
        return Ok(unevaluated());
      };
      let pair = match inner.len() {
        1 => match &inner[0] {
          Expr::Integer(n) => (*n, *n),
          _ => return Ok(unevaluated()),
        },
        2 => match (&inner[0], &inner[1]) {
          (Expr::Integer(l), Expr::Integer(r)) => (*l, *r),
          _ => return Ok(unevaluated()),
        },
        _ => return Ok(unevaluated()),
      };
      per_dim.push(pair);
    }
    return pad_array_per_dim(arr, &per_dim, &pad_val);
  }

  // Parse padding spec: integer n or {left, right}
  let (left, right) = match &args[1] {
    Expr::Integer(n) => {
      let n = *n;
      (n, n)
    }
    Expr::List(items) if items.len() == 2 => {
      let l = match &items[0] {
        Expr::Integer(n) => *n,
        _ => return Ok(unevaluated()),
      };
      let r = match &items[1] {
        Expr::Integer(n) => *n,
        _ => return Ok(unevaluated()),
      };
      (l, r)
    }
    _ => return Ok(unevaluated()),
  };

  pad_array(arr, left, right, &pad_val)
}

/// Apply `(left, right)` padding per outer dimension. `per_dim[0]` pads the
/// outermost dim; remaining entries pad inner dims. If the spec is shorter
/// than the array's rank, inner dims are left unchanged.
fn pad_array_per_dim(
  arr: &Expr,
  per_dim: &[(i128, i128)],
  pad_val: &Expr,
) -> Result<Expr, InterpreterError> {
  if per_dim.is_empty() {
    return Ok(arr.clone());
  }
  let (left, right) = per_dim[0];
  let rest = &per_dim[1..];

  let Expr::List(items) = arr else {
    return Ok(arr.clone());
  };

  // First recursively pad each child along the remaining dimensions.
  let mut padded_children: Vec<Expr> = items
    .iter()
    .map(|item| pad_array_per_dim(item, rest, pad_val))
    .collect::<Result<Vec<_>, _>>()?;

  // Build a pad element for this dimension: a fully-zero block with the
  // shape of one padded child.
  let pad_block = if let Some(first) = padded_children.first() {
    zero_block_like(first, pad_val)
  } else {
    pad_val.clone()
  };

  // Add or trim entries at the front.
  if left >= 0 {
    let mut prefix: Vec<Expr> = vec![pad_block.clone(); left as usize];
    prefix.append(&mut padded_children);
    padded_children = prefix;
  } else {
    let trim = (-left) as usize;
    if trim < padded_children.len() {
      padded_children = padded_children[trim..].to_vec();
    } else {
      padded_children = vec![];
    }
  }

  // Add or trim entries at the back.
  if right >= 0 {
    for _ in 0..right {
      padded_children.push(pad_block.clone());
    }
  } else {
    let trim = (-right) as usize;
    if trim < padded_children.len() {
      padded_children.truncate(padded_children.len() - trim);
    } else {
      padded_children = vec![];
    }
  }

  Ok(Expr::List(padded_children.into()))
}

/// Build an Expr with the same nested-List shape as `template`, filled with `pad_val`.
fn zero_block_like(template: &Expr, pad_val: &Expr) -> Expr {
  match template {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|it| zero_block_like(it, pad_val))
        .collect::<Vec<_>>()
        .into(),
    ),
    _ => pad_val.clone(),
  }
}

fn pad_array(
  arr: &Expr,
  left: i128,
  right: i128,
  pad_val: &Expr,
) -> Result<Expr, InterpreterError> {
  match arr {
    Expr::List(items) => {
      // Check if this is a multi-dimensional array (items are lists)
      let is_nested = items.iter().all(|item| matches!(item, Expr::List(_)));

      if is_nested && !items.is_empty() {
        // Multi-dimensional: pad each sub-array, then add padding rows
        let mut padded_items: Vec<Expr> = items
          .iter()
          .map(|item| pad_array(item, left, right, pad_val))
          .collect::<Result<Vec<_>, _>>()?;

        // Figure out the width of padded sub-arrays
        let inner_len = if let Expr::List(inner) = &padded_items[0] {
          inner.len()
        } else {
          0
        };

        // Create padding row
        let pad_row = Expr::List(vec![pad_val.clone(); inner_len].into());

        // Add/remove rows at top and bottom
        if left >= 0 {
          let mut prefix = vec![pad_row.clone(); left as usize];
          prefix.append(&mut padded_items);
          padded_items = prefix;
        } else {
          let trim = (-left) as usize;
          if trim < padded_items.len() {
            padded_items = padded_items[trim..].to_vec();
          } else {
            padded_items = vec![];
          }
        }

        if right >= 0 {
          for _ in 0..right {
            padded_items.push(pad_row.clone());
          }
        } else {
          let trim = (-right) as usize;
          if trim < padded_items.len() {
            padded_items.truncate(padded_items.len() - trim);
          } else {
            padded_items = vec![];
          }
        }

        Ok(Expr::List(padded_items.into()))
      } else {
        // 1D array
        let mut result = items.clone();

        if left >= 0 {
          let mut prefix: Vec<Expr> = vec![pad_val.clone(); left as usize];
          prefix.extend(result.iter().cloned());
          result = prefix.into();
        } else {
          let trim = (-left) as usize;
          if trim < result.len() {
            result = result.slice(trim..);
          } else {
            result = crate::ExprList::new();
          }
        }

        if right >= 0 {
          for _ in 0..right {
            result.push(pad_val.clone());
          }
        } else {
          let trim = (-right) as usize;
          let cur_len = result.len();
          if trim < cur_len {
            result.truncate(cur_len - trim);
          } else {
            result = crate::ExprList::new();
          }
        }

        Ok(Expr::List(result))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "ArrayPad".to_string(),
      args: vec![arr.clone()].into(),
    }),
  }
}

/// Convert Associations to Lists of rules within an expression.
/// Recurses into FunctionCall args and List items but not into Rule values,
/// matching Wolfram's Normal behavior.
fn normal_convert_associations(expr: &Expr) -> Expr {
  match expr {
    Expr::Association(pairs) => {
      // See `Normal` dispatch: `RuleDelayed{pattern==key, replacement}` is the
      // marker for an originally-delayed entry.
      let rules: Vec<Expr> = pairs
        .iter()
        .map(|(k, v)| match v {
          Expr::RuleDelayed {
            pattern,
            replacement,
          } if crate::syntax::assoc_marker_matches(k, pattern) => {
            Expr::RuleDelayed {
              pattern: Box::new(k.clone()),
              replacement: replacement.clone(),
            }
          }
          _ => Expr::Rule {
            pattern: Box::new(k.clone()),
            replacement: Box::new(v.clone()),
          },
        })
        .collect();
      Expr::List(rules.into())
    }
    Expr::FunctionCall { name, args } if name == "Association" => {
      Expr::List(args.clone())
    }
    // NumericArray / ByteArray unwrap to their underlying list payload.
    Expr::FunctionCall { name, args }
      if (name == "NumericArray" || name == "ByteArray") && args.len() == 1 =>
    {
      normal_convert_associations(&args[0])
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(normal_convert_associations).collect(),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(normal_convert_associations).collect())
    }
    _ => expr.clone(),
  }
}

/// ArrayFlatten[{{block11, block12, ...}, {block21, ...}, ...}]
/// Combines a matrix of sub-matrices (blocks) into a single matrix.
/// Scalar entries (e.g. 0) are expanded to zero/constant matrices
/// of the appropriate dimensions inferred from neighboring blocks.
fn array_flatten_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  // arg should be a list of rows, where each row is a list of blocks (sub-matrices)
  let block_rows = match arg {
    Expr::List(rows) => rows,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ArrayFlatten".to_string(),
        args: vec![arg.clone()].into(),
      });
    }
  };

  if block_rows.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // First, determine the grid dimensions (number of block rows and columns)
  let n_block_rows = block_rows.len();
  let mut n_block_cols = 0;

  // Collect all blocks as raw expressions in a 2D grid
  let mut block_grid: Vec<Vec<&Expr>> = Vec::new();
  for block_row in block_rows {
    let blocks_in_row = match block_row {
      Expr::List(blocks) => blocks,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "ArrayFlatten".to_string(),
          args: vec![arg.clone()].into(),
        });
      }
    };
    if n_block_cols == 0 {
      n_block_cols = blocks_in_row.len();
    }
    block_grid.push(blocks_in_row.iter().collect());
  }

  // Helper: get the dimensions (rows, cols) of a block if it's a matrix
  fn block_dims(block: &Expr) -> Option<(usize, usize)> {
    match block {
      Expr::List(rows) => {
        if rows.is_empty() {
          return Some((0, 0));
        }
        match &rows[0] {
          Expr::List(cols) => Some((rows.len(), cols.len())),
          _ => None, // 1D list, not a matrix block
        }
      }
      _ => None, // Scalar
    }
  }

  // Determine the row height for each block-row and column width
  // for each block-column by scanning actual matrix blocks.
  let mut row_heights: Vec<Option<usize>> = vec![None; n_block_rows];
  let mut col_widths: Vec<Option<usize>> = vec![None; n_block_cols];

  for (i, row) in block_grid.iter().enumerate() {
    for (j, block) in row.iter().enumerate() {
      if let Some((h, w)) = block_dims(block) {
        if row_heights[i].is_none() {
          row_heights[i] = Some(h);
        }
        if col_widths[j].is_none() {
          col_widths[j] = Some(w);
        }
      }
    }
  }

  // Default any undetermined dimension to 1
  let row_heights: Vec<usize> =
    row_heights.into_iter().map(|h| h.unwrap_or(1)).collect();
  let col_widths: Vec<usize> =
    col_widths.into_iter().map(|w| w.unwrap_or(1)).collect();

  // Parse each block into a matrix, expanding scalars to the
  // appropriate size
  let mut all_block_rows: Vec<Vec<Vec<Vec<Expr>>>> = Vec::new();

  for (i, row) in block_grid.iter().enumerate() {
    let mut parsed_blocks: Vec<Vec<Vec<Expr>>> = Vec::new();
    for (j, block) in row.iter().enumerate() {
      let matrix = match block {
        Expr::List(rows) => {
          let mut m: Vec<Vec<Expr>> = Vec::new();
          for r in rows {
            match r {
              Expr::List(cols) => m.push(cols.to_vec()),
              other => m.push(vec![other.clone()]),
            }
          }
          m
        }
        // Scalar: expand to a matrix of the right size filled
        // with this value (commonly 0)
        scalar => {
          let h = row_heights[i];
          let w = col_widths[j];
          vec![vec![(*scalar).clone(); w]; h]
        }
      };
      parsed_blocks.push(matrix);
    }
    all_block_rows.push(parsed_blocks);
  }

  // Build the result matrix by combining blocks
  let mut result: Vec<Vec<Expr>> = Vec::new();

  for block_row in &all_block_rows {
    if block_row.is_empty() {
      continue;
    }
    // Number of rows in this block-row (determined by first block)
    let n_rows = block_row[0].len();

    for r in 0..n_rows {
      let mut result_row: Vec<Expr> = Vec::new();
      for block in block_row {
        if r < block.len() {
          result_row.extend_from_slice(&block[r]);
        }
      }
      result.push(result_row);
    }
  }

  Ok(Expr::List(
    result.into_iter().map(|v| Expr::List(v.into())).collect(),
  ))
}

/// Distance between two expressions. Falls back to absolute scalar difference
/// and, for equal-length numeric lists, the Euclidean norm.
fn nearest_distance(a: &Expr, b: &Expr) -> Option<f64> {
  // Colors compare via Euclidean distance on their RGB triple.
  // `GrayLevel[g]` is treated as `RGBColor[g, g, g]`. Mixed
  // RGBColor↔GrayLevel comparisons work because both lift to a
  // 3-element float vector.
  if let (Some(ra), Some(rb)) = (color_to_rgb(a), color_to_rgb(b)) {
    let mut sum = 0.0;
    for (x, y) in ra.iter().zip(rb.iter()) {
      let dx = x - y;
      sum += dx * dx;
    }
    return Some(sum.sqrt());
  }
  match (a, b) {
    (Expr::List(va), Expr::List(vb)) if va.len() == vb.len() => {
      let mut sum = 0.0;
      for (x, y) in va.iter().zip(vb.iter()) {
        let dx = expr_to_f64(x)? - expr_to_f64(y)?;
        sum += dx * dx;
      }
      Some(sum.sqrt())
    }
    _ => {
      let av = expr_to_f64(a)?;
      let bv = expr_to_f64(b)?;
      Some((av - bv).abs())
    }
  }
}

/// Lift a colour expression to a 3-element RGB float vector.
/// Recognises `RGBColor[r, g, b]`, `RGBColor[r, g, b, a]` (alpha
/// dropped), and `GrayLevel[g]` (mapped to `[g, g, g]`).
fn color_to_rgb(e: &Expr) -> Option<[f64; 3]> {
  if let Expr::FunctionCall { name, args } = e {
    if name == "RGBColor" && (args.len() == 3 || args.len() == 4) {
      let r = expr_to_f64(&args[0])?;
      let g = expr_to_f64(&args[1])?;
      let b = expr_to_f64(&args[2])?;
      return Some([r, g, b]);
    }
    if name == "GrayLevel" && (args.len() == 1 || args.len() == 2) {
      let g = expr_to_f64(&args[0])?;
      return Some([g, g, g]);
    }
  }
  None
}

/// Nearest[list, x] - find elements of list nearest to x
/// Nearest[list, x, n] - find n nearest elements
/// Nearest[points -> values, x] - return the labels whose points are closest
fn nearest_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Rule form: Nearest[points -> labels, target]. Distances are measured on
  // the `points` list, but the result is drawn from the matching `labels`.
  let (items_owned, labels): (Vec<Expr>, Option<Vec<Expr>>) = match &args[0] {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let pts = match pattern.as_ref() {
        Expr::List(v) => v.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Nearest".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      let lbls = match replacement.as_ref() {
        Expr::List(v) if v.len() == pts.len() => v.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Nearest".to_string(),
            args: args.to_vec().into(),
          });
        }
      };
      (pts.to_vec(), Some(lbls.to_vec()))
    }
    Expr::List(v) => {
      // `{point1 -> label1, point2 -> label2, …}` is the list-of-rules
      // form: split into separate point and label vectors. If every
      // element is a `Rule` / `RuleDelayed`, treat this as the labelled
      // form so the result is drawn from the labels rather than the
      // points themselves.
      let all_rules = !v.is_empty()
        && v
          .iter()
          .all(|e| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. }));
      if all_rules {
        let mut pts = Vec::with_capacity(v.len());
        let mut lbls = Vec::with_capacity(v.len());
        for r in v {
          match r {
            Expr::Rule {
              pattern,
              replacement,
            }
            | Expr::RuleDelayed {
              pattern,
              replacement,
            } => {
              pts.push((**pattern).clone());
              lbls.push((**replacement).clone());
            }
            _ => unreachable!(),
          }
        }
        (pts, Some(lbls))
      } else {
        (v.to_vec(), None)
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Nearest".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let items = &items_owned;

  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Multi-target form: when `target` is a List and each item in it
  // produces a valid distance against `items[0]`, recurse per-target
  // and return a list of results — `Nearest[items, {t1, t2, …}]` →
  // `{Nearest[items, t1], Nearest[items, t2], …}`. This handles e.g.
  // `Nearest[{colors…}, {Orange, Gray}]` where each target is itself
  // a colour. We skip this branch when `target` matches `items[0]`
  // dimensionally as a single vector (the common scalar-list case).
  if let Expr::List(targets) = &args[1]
    && !targets.is_empty()
    && nearest_distance(&items[0], &args[1]).is_none()
    && targets
      .iter()
      .all(|t| nearest_distance(&items[0], t).is_some())
  {
    let mut sub_args = args.to_vec();
    let mut results = Vec::with_capacity(targets.len());
    for t in targets {
      sub_args[1] = t.clone();
      results.push(nearest_ast(&sub_args)?);
    }
    return Ok(Expr::List(results.into()));
  }

  let target = &args[1];

  // Parse the optional third argument. Accepts:
  //   n            — up to n closest elements
  //   All          — all elements, sorted by distance
  //   {n, r}       — up to n elements within radius r
  //   {All, r}     — all elements within radius r
  let (n, radius) = if args.len() >= 3 {
    match &args[2] {
      Expr::Integer(k) => (Some(*k as usize), None),
      Expr::Identifier(s) if s == "All" => (None, None),
      Expr::List(pair) if pair.len() == 2 => {
        let count = match &pair[0] {
          Expr::Integer(k) if *k >= 1 => Some(*k as usize),
          Expr::Identifier(s) if s == "All" => None,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Nearest".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        let r = match expr_to_f64(&pair[1]) {
          Some(r) if r >= 0.0 => r,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Nearest".to_string(),
              args: args.to_vec().into(),
            });
          }
        };
        (count, Some(r))
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Nearest".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    (Some(1), None) // default is just the single closest (and ties)
  };

  // Compute distance for each element (scalar or equal-length vector)
  let mut distances: Vec<(usize, f64)> = items
    .iter()
    .enumerate()
    .filter_map(|(i, item)| nearest_distance(item, target).map(|d| (i, d)))
    .collect();

  if distances.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Nearest".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Sort by distance, then by original order for ties
  distances
    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

  // Apply the radius filter first, then the count limit.
  let filtered: Vec<&(usize, f64)> = match radius {
    Some(r) => distances
      .iter()
      .take_while(|(_, d)| *d <= r + 1e-15)
      .collect(),
    None => distances.iter().collect(),
  };

  let pick = |i: usize| -> Expr {
    match &labels {
      Some(l) => l[i].clone(),
      None => items[i].clone(),
    }
  };

  match (args.len() >= 3, n) {
    // Bare 2-arg Nearest: return the tied-for-closest group.
    (false, _) => {
      let min_dist = filtered[0].1;
      let result: Vec<Expr> = filtered
        .iter()
        .take_while(|(_, d)| (*d - min_dist).abs() < 1e-15)
        .map(|(i, _)| pick(*i))
        .collect();
      Ok(Expr::List(result.into()))
    }
    // Count limit provided (possibly together with a radius).
    (true, Some(k)) => {
      let result: Vec<Expr> =
        filtered.iter().take(k).map(|(i, _)| pick(*i)).collect();
      Ok(Expr::List(result.into()))
    }
    // `All` (possibly together with a radius): keep everything that passed
    // the radius filter.
    (true, None) => {
      let result: Vec<Expr> = filtered.iter().map(|(i, _)| pick(*i)).collect();
      Ok(Expr::List(result.into()))
    }
  }
}

/// Try to convert an Expr to f64 for distance computation
fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1]) {
        Some(*a as f64 / *b as f64)
      } else {
        None
      }
    }
    _ => None,
  }
}

fn array_reshape_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Flatten the input list
  let flat = flatten_to_vec(&args[0]);

  // Parse dimensions
  let dims = match &args[1] {
    Expr::List(items) => {
      let mut d = Vec::new();
      for item in items {
        match item {
          Expr::Integer(n) if *n > 0 => d.push(*n as usize),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "ArrayReshape".to_string(),
              args: args.to_vec().into(),
            });
          }
        }
      }
      d
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ArrayReshape".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  if dims.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Optional padding: a scalar or a list of elements to cycle through.
  // `ArrayReshape[list, dims, pad]` fills trailing slots with the padding
  // values, cycling through the list if needed. Defaults to `0`.
  let pad: Vec<Expr> = if args.len() >= 3 {
    match &args[2] {
      Expr::List(items) if !items.is_empty() => items.to_vec(),
      Expr::List(_) => vec![Expr::Integer(0)],
      other => vec![other.clone()],
    }
  } else {
    vec![Expr::Integer(0)]
  };

  // Build the reshaped array, padding with `pad` if needed
  let mut idx = 0;
  Ok(build_reshaped(&flat, &dims, 0, &mut idx, &pad))
}

fn flatten_to_vec(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::List(items) => items.iter().flat_map(flatten_to_vec).collect(),
    _ => vec![expr.clone()],
  }
}

fn build_reshaped(
  flat: &[Expr],
  dims: &[usize],
  depth: usize,
  idx: &mut usize,
  pad: &[Expr],
) -> Expr {
  if depth == dims.len() - 1 {
    // Leaf level: collect dims[depth] elements
    let n = dims[depth];
    let mut row = Vec::with_capacity(n);
    for _ in 0..n {
      if *idx < flat.len() {
        row.push(flat[*idx].clone());
      } else {
        let pad_idx = (*idx - flat.len()) % pad.len();
        row.push(pad[pad_idx].clone());
      }
      *idx += 1;
    }
    Expr::List(row.into())
  } else {
    let n = dims[depth];
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
      result.push(build_reshaped(flat, dims, depth + 1, idx, pad));
    }
    Expr::List(result.into())
  }
}

fn position_index_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  let items = match expr {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PositionIndex".to_string(),
        args: vec![expr.clone()].into(),
      });
    }
  };

  // Build ordered map: value -> list of positions (1-indexed)
  let mut map: Vec<(Expr, Vec<i128>)> = Vec::new();
  for (i, item) in items.iter().enumerate() {
    let pos = (i + 1) as i128;
    let item_str = crate::syntax::expr_to_string(item);
    if let Some(entry) = map
      .iter_mut()
      .find(|(k, _)| crate::syntax::expr_to_string(k) == item_str)
    {
      entry.1.push(pos);
    } else {
      map.push((item.clone(), vec![pos]));
    }
  }

  // Convert to Association
  let rules: Vec<(Expr, Expr)> = map
    .into_iter()
    .map(|(key, positions)| {
      let pos_list =
        Expr::List(positions.into_iter().map(Expr::Integer).collect());
      (key, pos_list)
    })
    .collect();

  Ok(Expr::Association(rules))
}

fn list_convolve_ast(
  kernel: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let ker = match kernel {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListConvolve".to_string(),
        args: vec![kernel.clone(), list.clone()].into(),
      });
    }
  };
  let data = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListConvolve".to_string(),
        args: vec![kernel.clone(), list.clone()].into(),
      });
    }
  };

  let k = ker.len();
  let n = data.len();
  if k == 0 || n == 0 || k > n {
    return Ok(Expr::List(vec![].into()));
  }

  let out_len = n - k + 1;
  let mut result = Vec::with_capacity(out_len);

  for i in 0..out_len {
    // Sum kernel[k-1-j] * data[i+j] for j in 0..k (kernel is reversed for convolution)
    let mut terms = Vec::with_capacity(k);
    for j in 0..k {
      let product = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![ker[k - 1 - j].clone(), data[i + j].clone()].into(),
      };
      terms.push(product);
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    };
    let evaluated = evaluate_expr_to_expr(&sum).unwrap_or(sum);
    result.push(evaluated);
  }

  Ok(Expr::List(result.into()))
}

fn list_correlate_ast(
  kernel: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let ker = match kernel {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListCorrelate".to_string(),
        args: vec![kernel.clone(), list.clone()].into(),
      });
    }
  };
  let data = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListCorrelate".to_string(),
        args: vec![kernel.clone(), list.clone()].into(),
      });
    }
  };

  let k = ker.len();
  let n = data.len();
  if k == 0 || n == 0 || k > n {
    return Ok(Expr::List(vec![].into()));
  }

  let out_len = n - k + 1;
  let mut result = Vec::with_capacity(out_len);

  for i in 0..out_len {
    // Sum kernel[j] * data[i+j] for j in 0..k (no reversal)
    let mut terms = Vec::with_capacity(k);
    for j in 0..k {
      let product = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![ker[j].clone(), data[i + j].clone()].into(),
      };
      terms.push(product);
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    };
    let evaluated = evaluate_expr_to_expr(&sum).unwrap_or(sum);
    result.push(evaluated);
  }

  Ok(Expr::List(result.into()))
}

/// ArrayRules[array] - returns non-default elements as position -> value rules
fn array_rules_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let default_val = if args.len() == 2 {
    args[1].clone()
  } else {
    Expr::Integer(0)
  };

  // Handle SparseArray[...] — normalize to canonical form and emit
  // sorted rules plus a trailing `{_, _, ...} -> default` pattern rule.
  if let Expr::FunctionCall {
    name,
    args: sa_args,
  } = &args[0]
    && name == "SparseArray"
    && !sa_args.is_empty()
  {
    let normalized =
      crate::functions::list_helpers_ast::sparse_array_normalize_ast(sa_args)?;
    if let Expr::FunctionCall { name: n2, args: na } = &normalized
      && n2 == "SparseArray"
      && na.len() == 4
      && matches!(&na[0], Expr::Identifier(s) if s == "Automatic")
    {
      let dims: Vec<usize> = match &na[1] {
        Expr::List(items) => items
          .iter()
          .filter_map(|it| match it {
            Expr::Integer(n) if *n >= 0 => Some(*n as usize),
            _ => None,
          })
          .collect(),
        _ => Vec::new(),
      };
      let depth = dims.len().max(1);
      let sa_default = if args.len() == 2 {
        default_val.clone()
      } else {
        na[2].clone()
      };
      // Extract (position, value) pairs from the CSR-form fourth argument.
      let extracted =
        crate::functions::list_helpers_ast::sparse_array_extract_rules(
          &dims, &na[3],
        );
      let mut rules: Vec<Expr> = extracted
        .into_iter()
        .map(|(pos, val)| Expr::Rule {
          pattern: Box::new(Expr::List(
            pos.into_iter().map(Expr::Integer).collect(),
          )),
          replacement: Box::new(val),
        })
        .collect();
      let blanks: Vec<Expr> = (0..depth)
        .map(|_| Expr::Pattern {
          name: String::new(),
          head: None,
          blank_type: 1,
        })
        .collect();
      rules.push(Expr::Rule {
        pattern: Box::new(Expr::List(blanks.into())),
        replacement: Box::new(sa_default),
      });
      return Ok(Expr::List(rules.into()));
    }
    // Normalization failed — fall through to the default-value handling.
  }

  let mut rules: Vec<Expr> = Vec::new();

  fn collect_rules(
    expr: &Expr,
    indices: &mut Vec<i128>,
    rules: &mut Vec<Expr>,
    default_val: &Expr,
  ) {
    match expr {
      Expr::List(items) => {
        for (i, item) in items.iter().enumerate() {
          indices.push((i + 1) as i128);
          collect_rules(item, indices, rules, default_val);
          indices.pop();
        }
      }
      _ => {
        if expr_to_string(expr) != expr_to_string(default_val) {
          let pos =
            Expr::List(indices.iter().map(|&i| Expr::Integer(i)).collect());
          rules.push(Expr::Rule {
            pattern: Box::new(pos),
            replacement: Box::new(expr.clone()),
          });
        }
      }
    }
  }

  let mut indices = Vec::new();
  collect_rules(&args[0], &mut indices, &mut rules, &default_val);

  // Add the default pattern rule: {_, _, ...} -> default
  let depth = array_depth(&args[0]);
  let blanks: Vec<Expr> = (0..depth)
    .map(|_| Expr::Pattern {
      name: String::new(),
      head: None,
      blank_type: 1,
    })
    .collect();
  rules.push(Expr::Rule {
    pattern: Box::new(Expr::List(blanks.into())),
    replacement: Box::new(default_val),
  });

  Ok(Expr::List(rules.into()))
}

fn array_depth(expr: &Expr) -> usize {
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        1
      } else {
        1 + array_depth(&items[0])
      }
    }
    _ => 0,
  }
}

/// Convert a fully-dense nested-list `expr` into a `SparseArray[Automatic, …]`
/// with the given default value. Only used by Outer's `Times` collapse path
/// (cases 470 and 473), so the layout exactly matches wolframscript's
/// CSR-like inner form: `{1, {{rowPtr}, {colIndices…}}, {values…}}` for
/// rank ≥ 2 and `{1, {{0, count}, {{idx}…}}, {values…}}` for rank 1.
fn dense_to_sparse_array_with_default(
  expr: &Expr,
  default: &Expr,
) -> Option<Expr> {
  let dims = sparse_dims(expr)?;
  if dims.is_empty() {
    return None;
  }
  let default_str = expr_to_string(default);
  let mut entries: Vec<(Vec<usize>, Expr)> = Vec::new();
  let mut idx_buf: Vec<usize> = Vec::with_capacity(dims.len());
  collect_non_default_entries(expr, &mut idx_buf, &mut entries, &default_str);
  Some(build_sparse_array_csr(&dims, default, &entries))
}

/// Walk a nested-list `expr`, collecting the dimension at each level.
/// Returns `None` if any sublist's length disagrees with its sibling — i.e.
/// the expression isn't a proper rectangular tensor and can't be sparsified.
fn sparse_dims(expr: &Expr) -> Option<Vec<usize>> {
  match expr {
    Expr::List(items) => {
      let mut dims = vec![items.len()];
      if items.is_empty() {
        return Some(dims);
      }
      let inner = sparse_dims(&items[0])?;
      for it in &items[1..] {
        let other = sparse_dims(it)?;
        if other != inner {
          return None;
        }
      }
      dims.extend(inner);
      Some(dims)
    }
    _ => Some(vec![]),
  }
}

fn collect_non_default_entries(
  expr: &Expr,
  idx: &mut Vec<usize>,
  out: &mut Vec<(Vec<usize>, Expr)>,
  default_str: &str,
) {
  match expr {
    Expr::List(items) => {
      for (i, it) in items.iter().enumerate() {
        idx.push(i + 1);
        collect_non_default_entries(it, idx, out, default_str);
        idx.pop();
      }
    }
    _ => {
      if expr_to_string(expr) != default_str {
        out.push((idx.clone(), expr.clone()));
      }
    }
  }
}

fn build_sparse_array_csr(
  dims: &[usize],
  default: &Expr,
  entries: &[(Vec<usize>, Expr)],
) -> Expr {
  let dims_list =
    Expr::List(dims.iter().map(|&d| Expr::Integer(d as i128)).collect());
  let k = dims.len();
  let n = dims[0];
  let make_outer = |inner: Expr| Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: vec![
      Expr::Identifier("Automatic".to_string()),
      dims_list.clone(),
      default.clone(),
      inner,
    ]
    .into(),
  };
  if entries.is_empty() {
    let row_ptr = if k == 1 {
      Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into())
    } else {
      Expr::List(vec![Expr::Integer(0); n + 1].into())
    };
    let inner = Expr::List(
      vec![
        Expr::Integer(1),
        Expr::List(vec![row_ptr, Expr::List(vec![].into())].into()),
        Expr::List(vec![].into()),
      ]
      .into(),
    );
    return make_outer(inner);
  }
  let mut sorted: Vec<(Vec<usize>, Expr)> = entries.to_vec();
  sorted.sort_by(|a, b| a.0.cmp(&b.0));
  if k == 1 {
    let row_ptr = Expr::List(
      vec![Expr::Integer(0), Expr::Integer(sorted.len() as i128)].into(),
    );
    let col_indices = Expr::List(
      sorted
        .iter()
        .map(|(idx, _)| Expr::List(vec![Expr::Integer(idx[0] as i128)].into()))
        .collect(),
    );
    let values = Expr::List(sorted.iter().map(|(_, v)| v.clone()).collect());
    let inner = Expr::List(
      vec![
        Expr::Integer(1),
        Expr::List(vec![row_ptr, col_indices].into()),
        values,
      ]
      .into(),
    );
    return make_outer(inner);
  }
  let mut row_counts = vec![0i128; n];
  let mut col_indices_list: Vec<Expr> = Vec::with_capacity(sorted.len());
  let mut values_list: Vec<Expr> = Vec::with_capacity(sorted.len());
  for (idx, v) in &sorted {
    let row = idx[0] - 1;
    row_counts[row] += 1;
    let col_idx: Vec<Expr> =
      idx[1..].iter().map(|&i| Expr::Integer(i as i128)).collect();
    col_indices_list.push(Expr::List(col_idx.into()));
    values_list.push(v.clone());
  }
  let mut row_ptr = vec![Expr::Integer(0)];
  let mut acc = 0i128;
  for c in row_counts {
    acc += c;
    row_ptr.push(Expr::Integer(acc));
  }
  let inner = Expr::List(
    vec![
      Expr::Integer(1),
      Expr::List(
        vec![
          Expr::List(row_ptr.into()),
          Expr::List(col_indices_list.into()),
        ]
        .into(),
      ),
      Expr::List(values_list.into()),
    ]
    .into(),
  );
  make_outer(inner)
}

/// Parsed structure of a `SparseArray[Automatic, dims, default, payload]`
/// expression — used by Outer's nested-leaf path to apply the user
/// function to each non-default value while preserving the default-vs-
/// non-default distinction.
struct ParsedSparseArray {
  dims: Vec<usize>,
  default: Expr,
  /// Each entry: (1-based multi-index, value).
  entries: Vec<(Vec<usize>, Expr)>,
}

fn parse_sparse_array_data(expr: &Expr) -> Option<ParsedSparseArray> {
  let Expr::FunctionCall { name, args: sa } = expr else {
    return None;
  };
  if name != "SparseArray" {
    return None;
  }
  // Re-normalize so we always start from canonical 4-arg form.
  let canonical = list_helpers_ast::sparse_array_normalize_ast(sa).ok()?;
  let Expr::FunctionCall {
    name: cname,
    args: ca,
  } = &canonical
  else {
    return None;
  };
  if cname != "SparseArray" || ca.len() != 4 {
    return None;
  }
  if !matches!(&ca[0], Expr::Identifier(s) if s == "Automatic") {
    return None;
  }
  let dims: Vec<usize> = match &ca[1] {
    Expr::List(items) => {
      let mut d = Vec::with_capacity(items.len());
      for it in items {
        match it {
          Expr::Integer(n) if *n >= 0 => d.push(*n as usize),
          _ => return None,
        }
      }
      d
    }
    _ => return None,
  };
  let default = ca[2].clone();
  let raw_entries = list_helpers_ast::sparse_array_extract_rules(&dims, &ca[3]);
  let entries: Vec<(Vec<usize>, Expr)> = raw_entries
    .into_iter()
    .map(|(idx, v)| (idx.into_iter().map(|i| i as usize).collect(), v))
    .collect();
  Some(ParsedSparseArray {
    dims,
    default,
    entries,
  })
}

/// Build the `Outer[func, lists…, sa]` result where `sa` is the last
/// argument and stays as `SparseArray` at the leaves. Walks each dense
/// `lists` arg through every nest level (accumulating one scalar at each
/// leaf), then once all outer args have contributed a scalar, builds the
/// inner `SparseArray` with `func` applied to the accumulated values and
/// each entry/default of `sa`.
fn build_outer_with_sparse_last(
  func: &Expr,
  outer_lists: &[Expr],
  sa: &ParsedSparseArray,
) -> Option<Expr> {
  fn walk_arg(
    func: &Expr,
    arg: &Expr,
    accumulated: &mut Vec<Expr>,
    rest: &[Expr],
    sa: &ParsedSparseArray,
  ) -> Option<Expr> {
    match arg {
      Expr::List(items) => {
        let mut results = Vec::with_capacity(items.len());
        for it in items {
          results.push(walk_arg(func, it, accumulated, rest, sa)?);
        }
        Some(Expr::List(results.into()))
      }
      _ => {
        accumulated.push(arg.clone());
        let r = process_outer_args(func, rest, accumulated, sa);
        accumulated.pop();
        r
      }
    }
  }

  fn process_outer_args(
    func: &Expr,
    rest: &[Expr],
    accumulated: &mut Vec<Expr>,
    sa: &ParsedSparseArray,
  ) -> Option<Expr> {
    if rest.is_empty() {
      return Some(build_inner_sparse(func, accumulated, sa));
    }
    walk_arg(func, &rest[0], accumulated, &rest[1..], sa)
  }

  let mut acc = Vec::with_capacity(outer_lists.len());
  process_outer_args(func, outer_lists, &mut acc, sa)
}

fn build_inner_sparse(
  func: &Expr,
  outer_vals: &[Expr],
  sa: &ParsedSparseArray,
) -> Expr {
  let func_name = match func {
    Expr::Identifier(s) => s.clone(),
    _ => crate::syntax::expr_to_string(func),
  };
  let make_call = |trailing: Expr| {
    let mut call_args = Vec::with_capacity(outer_vals.len() + 1);
    call_args.extend(outer_vals.iter().cloned());
    call_args.push(trailing);
    let call = Expr::FunctionCall {
      name: func_name.clone(),
      args: call_args.into(),
    };
    crate::evaluator::evaluate_expr_to_expr(&call).unwrap_or(call)
  };
  let new_default = make_call(sa.default.clone());
  let new_entries: Vec<(Vec<usize>, Expr)> = sa
    .entries
    .iter()
    .map(|(idx, v)| (idx.clone(), make_call(v.clone())))
    .collect();
  build_sparse_array_csr(&sa.dims, &new_default, &new_entries)
}
