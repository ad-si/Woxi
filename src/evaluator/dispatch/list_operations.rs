#[allow(unused_imports)]
use super::*;
use crate::functions::list_helpers_ast;

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
      if let (Expr::List(items), Expr::List(lengths)) = (&args[0], &args[1]) {
        let mut result = Vec::new();
        let mut pos = 0;
        let mut valid = true;
        for len_expr in lengths {
          if let Expr::Integer(n) = len_expr {
            let n = *n as usize;
            if pos + n > items.len() {
              valid = false;
              break;
            }
            result.push(Expr::List(items[pos..pos + n].to_vec()));
            pos += n;
          } else {
            valid = false;
            break;
          }
        }
        if valid {
          return Some(Ok(Expr::List(result)));
        }
      }
    }
    "FlattenAt" if args.len() == 2 => {
      if let Expr::List(items) = &args[0] {
        let len = items.len() as i128;
        let positions: Vec<usize> = match &args[1] {
          Expr::Integer(n) => {
            let idx = if *n < 0 { len + n + 1 } else { *n };
            if idx >= 1 && idx <= len {
              vec![idx as usize]
            } else {
              return Some(Ok(Expr::FunctionCall {
                name: "FlattenAt".to_string(),
                args: args.to_vec(),
              }));
            }
          }
          Expr::List(pos_list) => {
            let mut idxs = Vec::new();
            for p in pos_list {
              if let Expr::Integer(n) = p {
                let idx = if *n < 0 { len + n + 1 } else { *n };
                if idx >= 1 && idx <= len {
                  idxs.push(idx as usize);
                }
              }
            }
            idxs
          }
          _ => vec![],
        };
        let pos_set: std::collections::HashSet<usize> =
          positions.into_iter().collect();
        let mut result = Vec::new();
        for (i, item) in items.iter().enumerate() {
          if pos_set.contains(&(i + 1)) {
            if let Expr::List(sub) = item {
              result.extend(sub.iter().cloned());
            } else {
              result.push(item.clone());
            }
          } else {
            result.push(item.clone());
          }
        }
        return Some(Ok(Expr::List(result)));
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
          return Some(Ok(Expr::List(inv)));
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
            args: args.to_vec(),
          }));
        }
        let mut result = Vec::with_capacity(n - r + 1);
        for i in 0..=(n - r) {
          let window = Expr::List(items[i..i + r].to_vec());
          match list_helpers_ast::median_ast(&window) {
            Ok(val) => result.push(val),
            Err(e) => return Some(Err(e)),
          }
        }
        return Some(Ok(Expr::List(result)));
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
          return Some(Ok(Expr::List(vec![])));
        }
        let f = &args[0];
        let mut results = Vec::new();
        for i in 0..=(items.len() - window_size) {
          let sublist = Expr::List(items[i..i + window_size].to_vec());
          // Construct f[sublist] using Map-like application
          let applied = match f {
            Expr::Identifier(fname) => Expr::FunctionCall {
              name: fname.clone(),
              args: vec![sublist],
            },
            _ => {
              // For pure functions etc, use general application
              Expr::FunctionCall {
                name: expr_to_string(f),
                args: vec![sublist],
              }
            }
          };
          match crate::evaluator::evaluate_expr_to_expr(&applied) {
            Ok(val) => results.push(val),
            Err(e) => return Some(Err(e)),
          }
        }
        return Some(Ok(Expr::List(results)));
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
        args: args.to_vec(),
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
        args: args.to_vec(),
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
            args: args.to_vec(),
          }));
        }
      };
      if items.is_empty() {
        // Fold[f, {}] is unevaluated in Wolfram Language
        return Some(Ok(Expr::FunctionCall {
          name: "Fold".to_string(),
          args: args.to_vec(),
        }));
      }
      let init = items[0].clone();
      let rest = match head {
        Some(h) => Expr::FunctionCall {
          name: h.to_string(),
          args: items[1..].to_vec(),
        },
        None => Expr::List(items[1..].to_vec()),
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
    "ComposeList" if args.len() == 2 => {
      return Some(list_helpers_ast::compose_list_ast(args));
    }
    "ContainsOnly" if args.len() == 2 => {
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
              args: args.to_vec(),
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
        return Some(Ok(Expr::List(result)));
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
        args: args.to_vec(),
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
            return Some(Ok(Expr::List(bytes)));
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
                  args: vec![],
                })
              } else {
                v.clone()
              };
              row_pairs.push((k.clone(), val));
            }
            rows.push(Expr::Association(row_pairs));
          }
          return Some(Ok(Expr::List(rows)));
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

        // Build terms: c_i * base^(nmin + i)
        let mut terms: Vec<Expr> = Vec::new();
        for (i, coeff) in coeffs.iter().enumerate() {
          if matches!(coeff, Expr::Integer(0)) {
            continue;
          }
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
            None => coeff.clone(),
            Some(bp) => {
              // Evaluate the Times to get canonical form
              let t = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(coeff.clone()),
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
      // Normal[<|k -> v, ...|>] converts Association to List of rules
      if let Expr::Association(pairs) = &args[0] {
        let rules: Vec<Expr> = pairs
          .iter()
          .map(|(k, v)| Expr::Rule {
            pattern: Box::new(k.clone()),
            replacement: Box::new(v.clone()),
          })
          .collect();
        return Some(Ok(Expr::List(rules)));
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
        (Ok(t), Ok(d)) => Ok(Expr::List(vec![t, d])),
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
            None => Expr::List(items),
            Some(name) => Expr::FunctionCall {
              name: name.clone(),
              args: items,
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
              return Some(Ok(wrap(sorted)));
            }
            "Less" => {
              sorted.sort_by(list_helpers_ast::canonical_cmp);
              return Some(Ok(wrap(sorted)));
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
        return Some(Ok(wrap(sorted)));
      }
    }
    "ReverseSort" if args.len() == 1 || args.len() == 2 => {
      // ReverseSort[list] sorts then reverses
      // ReverseSort[list, p] sorts by p then reverses
      let mut sorted = match evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Sort".to_string(),
        args: args.to_vec(),
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
      return Some(Ok(Expr::List(args.to_vec())));
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
              args: args.to_vec(),
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
            return Some(Ok(Expr::List(vec![])));
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
          current = next;
        }
        return Some(Ok(Expr::List(current)));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Ratios".to_string(),
        args: args.to_vec(),
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
            args: args.to_vec(),
          }));
        }
      };
      if items.is_empty() {
        return Some(Ok(match head {
          Some(h) => Expr::FunctionCall {
            name: h.to_string(),
            args: vec![],
          },
          None => Expr::List(vec![]),
        }));
      }
      let init = items[0].clone();
      let rest = match head {
        Some(h) => Expr::FunctionCall {
          name: h.to_string(),
          args: items[1..].to_vec(),
        },
        None => Expr::List(items[1..].to_vec()),
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
            args: args.to_vec(),
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
              args: args.to_vec(),
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
        return Some(Ok(Expr::List(result)));
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
        return Some(Ok(Expr::List(padded)));
      }
    }
    "PadLeft" if args.len() >= 2 => {
      // Multi-dim form: PadLeft[list, {n1, n2, ...}, pad?]
      if let Expr::List(dim_items) = &args[1] {
        let ns_opt: Option<Vec<i128>> =
          dim_items.iter().map(expr_to_i128).collect();
        if let Some(ns) = ns_opt {
          let pad = if args.len() >= 3 {
            args[2].clone()
          } else {
            Expr::Integer(0)
          };
          return Some(list_helpers_ast::pad_left_multidim(
            &args[0], &ns, &pad,
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
        return Some(Ok(Expr::List(padded)));
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
          return Some(list_helpers_ast::pad_right_multidim(
            &args[0], &ns, &pad,
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
      // NestWhile[f, x, test]              — plain
      // NestWhile[f, x, test, m]           — m = supply-last-m (currently only
      //                                        m == 1 is supported)
      // NestWhile[f, x, test, m, max]      — max is the maximum iteration cap
      // NestWhile[f, x, test, m, max, n]   — n extra iterations (or -|n|
      //                                        steps back) once test fails
      if args.len() >= 4 && !matches!(&args[3], Expr::Integer(1)) {
        return None; // let the default unevaluated path handle m > 1
      }
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
        &args[0], &args[1], &args[2], max_iter, extra_n,
      ));
    }
    "NestWhileList" if (3..=6).contains(&args.len()) => {
      if args.len() >= 4 && !matches!(&args[3], Expr::Integer(1)) {
        return None;
      }
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
        &args[0], &args[1], &args[2], max_iter, extra_n,
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
            args: vec![expr.clone()],
          })
          .map(|_| Expr::FunctionCall {
            name: crate::syntax::expr_to_string(p),
            args: vec![expr.clone()],
          }),
        );
      }
      // For n >= 1, we need to wrap the head at depth n
      // For n == 1 (default): f[a, b] -> p[f][a, b]
      fn wrap_head_at_depth(expr: &Expr, p: &Expr, depth: i128) -> Expr {
        if depth == 0 {
          Expr::FunctionCall {
            name: crate::syntax::expr_to_string(p),
            args: vec![expr.clone()],
          }
        } else {
          match expr {
            Expr::FunctionCall { name, args } => {
              let wrapped_head = wrap_head_at_depth(
                &Expr::Identifier(name.clone()),
                p,
                depth - 1,
              );
              Expr::CurriedCall {
                func: Box::new(wrapped_head),
                args: args.clone(),
              }
            }
            Expr::CurriedCall { func, args } => {
              let wrapped_func = wrap_head_at_depth(func, p, depth - 1);
              Expr::CurriedCall {
                func: Box::new(wrapped_func),
                args: args.clone(),
              }
            }
            _ => Expr::FunctionCall {
              name: crate::syntax::expr_to_string(p),
              args: vec![expr.clone()],
            },
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
    "TakeSmallest" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return Some(list_helpers_ast::take_smallest_ast(&args[0], n));
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
    "ArrayQ" if !args.is_empty() && args.len() <= 3 => {
      let is_array = match list_helpers_ast::array_q_ast(&args[0]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      if args.len() >= 2 {
        // ArrayQ[expr, n] - check if expr is an array of depth n
        if matches!(&is_array, Expr::Identifier(s) if s == "True")
          && let Ok(dims) = list_helpers_ast::dimensions_ast(&[args[0].clone()])
          && let Expr::List(d) = &dims
          && let Some(n) = expr_to_i128(&args[1])
        {
          return Some(Ok(if d.len() == n as usize {
            Expr::Identifier("True".to_string())
          } else {
            Expr::Identifier("False".to_string())
          }));
        }
        return Some(Ok(is_array));
      }
      return Some(Ok(is_array));
    }
    "VectorQ" if args.len() == 1 => {
      return Some(list_helpers_ast::vector_q_ast(&args[0]));
    }
    "MatrixQ" if args.len() == 1 => {
      return Some(list_helpers_ast::matrix_q_ast(&args[0]));
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
      // Multi-iterator Do: Do[body, {i, ...}, {j, ...}, ...]
      // Nest the iterators: outermost is first iterator, innermost is last
      // Build a nested Do: Do[Do[body, last_iter], ..., first_iter]
      let body = &args[0];
      let iters = &args[1..];
      let mut nested = body.clone();
      for iter in iters.iter().rev() {
        nested = Expr::FunctionCall {
          name: "Do".to_string(),
          args: vec![nested, iter.clone()],
        };
      }
      return Some(evaluate_expr_to_expr(&nested));
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
        Expr::List(args[1..].to_vec())
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
    "Extract" if args.len() == 2 => {
      return Some(list_helpers_ast::extract_ast(&args[0], &args[1]));
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
        args: flat,
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
        args: flat,
      }));
    }
    "Outer" if args.len() >= 3 => {
      return Some(list_helpers_ast::outer_ast(&args[0], &args[1..]));
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
              args: vec![state.clone(), elem.clone()],
            },
            _ => Expr::FunctionCall {
              name: expr_to_string(f),
              args: vec![state.clone(), elem.clone()],
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
        return Some(Ok(Expr::List(results)));
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
        return Some(Ok(Expr::List(results)));
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
      if let (Expr::List(list), Expr::List(sub)) = (&args[0], &args[1]) {
        if sub.is_empty() {
          return Some(Ok(Expr::List(vec![])));
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
            results.push(Expr::List(vec![
              Expr::Integer((i + 1) as i128),
              Expr::Integer((i + sub_len) as i128),
            ]));
          }
        }
        return Some(Ok(Expr::List(results)));
      }
    }
    // SequenceCases[list, sublist] — find matching subsequences
    // Supports: plain list, Condition[list, test], Rule/RuleDelayed[list, rhs]
    "SequenceCases" if args.len() == 2 => {
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

        // Get the inner list pattern (unwrapping Condition if needed)
        let list_pat = match match_pat {
          Expr::List(_) => match_pat,
          Expr::FunctionCall {
            name,
            args: cond_args,
          } if name == "Condition" && cond_args.len() == 2 => &cond_args[0],
          _ => match_pat,
        };

        // Get the sub-elements for length calculations
        let sub = match list_pat {
          Expr::List(items) => items,
          _ => return Some(Ok(Expr::List(vec![]))),
        };

        if sub.is_empty() {
          return Some(Ok(Expr::List(vec![])));
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
              let subseq = Expr::List(list[i..i + len].to_vec());
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
          return Some(Ok(Expr::List(results)));
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
              results.push(Expr::List(list[i..i + sub_len].to_vec()));
              i += sub_len;
            } else {
              i += 1;
            }
          }
          return Some(Ok(Expr::List(results)));
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
              args: vec![k.clone()],
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

    // CenterArray[list, n] — center list within an array of size n, padding with 0
    "CenterArray" if args.len() >= 2 && args.len() <= 3 => {
      if let (Expr::List(items), Some(n)) = (&args[0], expr_to_i128(&args[1])) {
        let n = n as usize;
        let pad = if args.len() == 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        let m = items.len();
        if n <= m {
          // Take n elements centered: use ceiling division for the offset
          let start = (m - n).div_ceil(2);
          return Some(Ok(Expr::List(items[start..start + n].to_vec())));
        }
        let left_pad = (n - m) / 2;
        let right_pad = n - m - left_pad;
        let mut result = vec![pad.clone(); left_pad];
        result.extend_from_slice(items);
        result.extend(vec![pad; right_pad]);
        return Some(Ok(Expr::List(result)));
      }
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
        return Some(Ok(Expr::List(result)));
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
    // FindPermutation[list1, list2] — find permutation that maps list1 to list2
    "FindPermutation" if args.len() == 2 => {
      if let (Expr::List(a), Expr::List(b)) = (&args[0], &args[1])
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
              cycles.push(Expr::List(cycle));
            }
          }
          return Some(Ok(Expr::FunctionCall {
            name: "Cycles".to_string(),
            args: vec![Expr::List(cycles)],
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
    // PermutationOrder[perm] — order (smallest n such that perm^n = identity)
    "PermutationOrder" if args.len() == 1 => {
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
          return Some(Ok(Expr::List(result_exprs)));
        }
      }
    }
    // PermutationLength[perm] — number of non-fixed points
    "PermutationLength" if args.len() == 1 => {
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
              args: vec![acc.clone(), item.clone()],
            },
            Expr::Function { body } => crate::syntax::substitute_slots(
              body,
              &[acc.clone(), item.clone()],
            ),
            _ => Expr::FunctionCall {
              name: expr_to_string(f),
              args: vec![acc.clone(), item.clone()],
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
        return Some(Ok(Expr::List(results)));
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
      if let Expr::List(perm) = &args[0] {
        let mut support = Vec::new();
        for (i, p) in perm.iter().enumerate() {
          if let Expr::Integer(v) = p
            && *v as usize != i + 1
          {
            support.push(Expr::Integer((i + 1) as i128));
          }
        }
        return Some(Ok(Expr::List(support)));
      }
    }
    // PermutationMax[perm] — largest element moved by the permutation
    "PermutationMax" if args.len() == 1 => {
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
          args: vec![],
        }));
      }
    }
    // Splice[list] — splice a list into the enclosing List (at evaluation level, acts like Sequence)
    "Splice" if args.len() == 1 => {
      if let Expr::List(items) = &args[0] {
        // Splice[{a, b, c}] evaluates to Sequence[a, b, c]
        return Some(Ok(Expr::FunctionCall {
          name: "Sequence".to_string(),
          args: items.clone(),
        }));
      }
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
        let mapped = apply_function_to_arg(f, &Expr::List(subset))
          .unwrap_or(Expr::List(vec![]));
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

  // Parse padding spec: integer n or {left, right}
  let (left, right) = match &args[1] {
    Expr::Integer(n) => {
      let n = *n;
      (n, n)
    }
    Expr::List(items) if items.len() == 2 => {
      let l = match &items[0] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "ArrayPad".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let r = match &items[1] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "ArrayPad".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (l, r)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ArrayPad".to_string(),
        args: args.to_vec(),
      });
    }
  };

  pad_array(arr, left, right, &pad_val)
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
        let pad_row = Expr::List(vec![pad_val.clone(); inner_len]);

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

        Ok(Expr::List(padded_items))
      } else {
        // 1D array
        let mut result = items.clone();

        if left >= 0 {
          let mut prefix = vec![pad_val.clone(); left as usize];
          prefix.append(&mut result);
          result = prefix;
        } else {
          let trim = (-left) as usize;
          if trim < result.len() {
            result = result[trim..].to_vec();
          } else {
            result = vec![];
          }
        }

        if right >= 0 {
          for _ in 0..right {
            result.push(pad_val.clone());
          }
        } else {
          let trim = (-right) as usize;
          if trim < result.len() {
            result.truncate(result.len() - trim);
          } else {
            result = vec![];
          }
        }

        Ok(Expr::List(result))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "ArrayPad".to_string(),
      args: vec![arr.clone()],
    }),
  }
}

/// Convert Associations to Lists of rules within an expression.
/// Recurses into FunctionCall args and List items but not into Rule values,
/// matching Wolfram's Normal behavior.
fn normal_convert_associations(expr: &Expr) -> Expr {
  match expr {
    Expr::Association(pairs) => {
      let rules: Vec<Expr> = pairs
        .iter()
        .map(|(k, v)| Expr::Rule {
          pattern: Box::new(k.clone()),
          replacement: Box::new(v.clone()),
        })
        .collect();
      Expr::List(rules)
    }
    Expr::FunctionCall { name, args } if name == "Association" => {
      Expr::List(args.clone())
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
        args: vec![arg.clone()],
      });
    }
  };

  if block_rows.is_empty() {
    return Ok(Expr::List(vec![]));
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
          args: vec![arg.clone()],
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
              Expr::List(cols) => m.push(cols.clone()),
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

  Ok(Expr::List(result.into_iter().map(Expr::List).collect()))
}

/// Nearest[list, x] - find elements of list nearest to x
/// Nearest[list, x, n] - find n nearest elements
fn nearest_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Nearest".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
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
              args: args.to_vec(),
            });
          }
        };
        let r = match expr_to_f64(&pair[1]) {
          Some(r) if r >= 0.0 => r,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Nearest".to_string(),
              args: args.to_vec(),
            });
          }
        };
        (count, Some(r))
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Nearest".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    (Some(1), None) // default is just the single closest (and ties)
  };

  // Compute distance for each element
  let target_f = expr_to_f64(target);
  let mut distances: Vec<(usize, f64)> = items
    .iter()
    .enumerate()
    .filter_map(|(i, item)| {
      let item_f = expr_to_f64(item);
      match (target_f, item_f) {
        (Some(t), Some(v)) => Some((i, (v - t).abs())),
        _ => None,
      }
    })
    .collect();

  if distances.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Nearest".to_string(),
      args: args.to_vec(),
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

  match (args.len() >= 3, n) {
    // Bare 2-arg Nearest: return the tied-for-closest group.
    (false, _) => {
      let min_dist = filtered[0].1;
      let result: Vec<Expr> = filtered
        .iter()
        .take_while(|(_, d)| (*d - min_dist).abs() < 1e-15)
        .map(|(i, _)| items[*i].clone())
        .collect();
      Ok(Expr::List(result))
    }
    // Count limit provided (possibly together with a radius).
    (true, Some(k)) => {
      let result: Vec<Expr> = filtered
        .iter()
        .take(k)
        .map(|(i, _)| items[*i].clone())
        .collect();
      Ok(Expr::List(result))
    }
    // `All` (possibly together with a radius): keep everything that passed
    // the radius filter.
    (true, None) => {
      let result: Vec<Expr> =
        filtered.iter().map(|(i, _)| items[*i].clone()).collect();
      Ok(Expr::List(result))
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
              args: args.to_vec(),
            });
          }
        }
      }
      d
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ArrayReshape".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if dims.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Optional padding: a scalar or a list of elements to cycle through.
  // `ArrayReshape[list, dims, pad]` fills trailing slots with the padding
  // values, cycling through the list if needed. Defaults to `0`.
  let pad: Vec<Expr> = if args.len() >= 3 {
    match &args[2] {
      Expr::List(items) if !items.is_empty() => items.clone(),
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
    Expr::List(row)
  } else {
    let n = dims[depth];
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
      result.push(build_reshaped(flat, dims, depth + 1, idx, pad));
    }
    Expr::List(result)
  }
}

fn position_index_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  let items = match expr {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PositionIndex".to_string(),
        args: vec![expr.clone()],
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
        args: vec![kernel.clone(), list.clone()],
      });
    }
  };
  let data = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListConvolve".to_string(),
        args: vec![kernel.clone(), list.clone()],
      });
    }
  };

  let k = ker.len();
  let n = data.len();
  if k == 0 || n == 0 || k > n {
    return Ok(Expr::List(vec![]));
  }

  let out_len = n - k + 1;
  let mut result = Vec::with_capacity(out_len);

  for i in 0..out_len {
    // Sum kernel[k-1-j] * data[i+j] for j in 0..k (kernel is reversed for convolution)
    let mut terms = Vec::with_capacity(k);
    for j in 0..k {
      let product = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![ker[k - 1 - j].clone(), data[i + j].clone()],
      };
      terms.push(product);
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    };
    let evaluated = evaluate_expr_to_expr(&sum).unwrap_or(sum);
    result.push(evaluated);
  }

  Ok(Expr::List(result))
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
        args: vec![kernel.clone(), list.clone()],
      });
    }
  };
  let data = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListCorrelate".to_string(),
        args: vec![kernel.clone(), list.clone()],
      });
    }
  };

  let k = ker.len();
  let n = data.len();
  if k == 0 || n == 0 || k > n {
    return Ok(Expr::List(vec![]));
  }

  let out_len = n - k + 1;
  let mut result = Vec::with_capacity(out_len);

  for i in 0..out_len {
    // Sum kernel[j] * data[i+j] for j in 0..k (no reversal)
    let mut terms = Vec::with_capacity(k);
    for j in 0..k {
      let product = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![ker[j].clone(), data[i + j].clone()],
      };
      terms.push(product);
    }
    let sum = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    };
    let evaluated = evaluate_expr_to_expr(&sum).unwrap_or(sum);
    result.push(evaluated);
  }

  Ok(Expr::List(result))
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
        pattern: Box::new(Expr::List(blanks)),
        replacement: Box::new(sa_default),
      });
      return Ok(Expr::List(rules));
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
    pattern: Box::new(Expr::List(blanks)),
    replacement: Box::new(default_val),
  });

  Ok(Expr::List(rules))
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
