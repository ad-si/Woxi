#[allow(unused_imports)]
use super::*;
use crate::functions::list_helpers_ast;

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
    "MovingMap" if args.len() == 3 => {
      // MovingMap[f, list, n] - apply f to sublists of length n+1
      if let Expr::List(items) = &args[1] {
        if let Some(n) = match &args[2] {
          Expr::Integer(n) => Some(*n as usize),
          _ => None,
        } {
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
    "AllTrue" if args.len() == 2 => {
      return Some(list_helpers_ast::all_true_ast(&args[0], &args[1]));
    }
    "AnyTrue" if args.len() == 2 => {
      return Some(list_helpers_ast::any_true_ast(&args[0], &args[1]));
    }
    "NoneTrue" if args.len() == 2 => {
      return Some(list_helpers_ast::none_true_ast(&args[0], &args[1]));
    }
    "Fold" if args.len() == 2 || args.len() == 3 => {
      if args.len() == 3 {
        return Some(list_helpers_ast::fold_ast(&args[0], &args[1], &args[2]));
      }
      // Fold[f, {a, b, c, ...}] = Fold[f, a, {b, c, ...}]
      if let Expr::List(items) = &args[1] {
        if items.is_empty() {
          return Some(Ok(Expr::List(vec![])));
        }
        let init = items[0].clone();
        let rest = Expr::List(items[1..].to_vec());
        return Some(list_helpers_ast::fold_ast(&args[0], &init, &rest));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Fold".to_string(),
        args: args.to_vec(),
      }));
    }
    "CountBy" if args.len() == 2 => {
      return Some(list_helpers_ast::count_by_ast(&args[0], &args[1]));
    }
    "GroupBy" if args.len() == 2 => {
      return Some(list_helpers_ast::group_by_ast(&args[0], &args[1]));
    }
    "SortBy" if args.len() == 2 => {
      return Some(list_helpers_ast::sort_by_ast(&args[0], &args[1]));
    }
    "Ordering" if !args.is_empty() && args.len() <= 2 => {
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
        &args[0], &args[1], &args[2],
      ));
    }
    "Position" if args.len() == 2 => {
      return Some(list_helpers_ast::position_ast(&args[0], &args[1]));
    }
    "FirstPosition" if args.len() >= 2 => {
      return Some(list_helpers_ast::first_position_ast(args));
    }
    "MapIndexed" if args.len() == 2 => {
      return Some(list_helpers_ast::map_indexed_ast(&args[0], &args[1]));
    }
    "Tally" if args.len() == 1 => {
      return Some(list_helpers_ast::tally_ast(&args[0]));
    }
    "Counts" if args.len() == 1 => {
      return Some(list_helpers_ast::counts_ast(&args[0]));
    }
    "BinCounts" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::bin_counts_ast(args));
    }
    "HistogramList" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::histogram_list_ast(args));
    }
    "DeleteDuplicates" if args.len() == 1 => {
      return Some(list_helpers_ast::delete_duplicates_ast(&args[0]));
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
    "Dimensions" if args.len() == 1 => {
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
      return Some(list_helpers_ast::map_thread_ast(&args[0], &args[1], level));
    }
    "Downsample" if args.len() == 2 || args.len() == 3 => {
      if let Expr::List(items) = &args[0] {
        if let Some(n) = expr_to_i128(&args[1]) {
          if n >= 1 {
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
        return Some(list_helpers_ast::partition_ast(&args[1], n, d).and_then(
          |partitioned| list_helpers_ast::map_ast(&args[0], &partitioned),
        ));
      }
    }
    "Partition" if args.len() == 2 || args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        let d = if args.len() == 3 {
          expr_to_i128(&args[2])
        } else {
          None
        };
        return Some(list_helpers_ast::partition_ast(&args[0], n, d));
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
    "Subsequences" if !args.is_empty() && args.len() <= 2 => {
      return Some(list_helpers_ast::subsequences_ast(args));
    }
    "SparseArray" if !args.is_empty() => {
      // Return SparseArray unevaluated (like Wolfram); use Normal[] to expand
      return Some(Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec(),
      }));
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
      if let Expr::List(items) = &args[0] {
        let cmp_name = crate::syntax::expr_to_string(&args[1]);
        let mut sorted = items.clone();
        match cmp_name.as_str() {
          "Greater" => {
            sorted
              .sort_by(|a, b| list_helpers_ast::canonical_cmp(a, b).reverse());
            return Some(Ok(Expr::List(sorted)));
          }
          "Less" => {
            sorted.sort_by(list_helpers_ast::canonical_cmp);
            return Some(Ok(Expr::List(sorted)));
          }
          _ => {
            // Custom comparator: evaluate p[a, b] for each comparison
            sorted.sort_by(|a, b| {
              let result = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: cmp_name.clone(),
                args: vec![a.clone(), b.clone()],
              });
              match result {
                Ok(Expr::Identifier(ref s)) if s == "True" => {
                  std::cmp::Ordering::Less
                }
                _ => std::cmp::Ordering::Greater,
              }
            });
            return Some(Ok(Expr::List(sorted)));
          }
        }
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
    }
    "Ratios" if args.len() == 1 => {
      if let Expr::List(items) = &args[0] {
        if items.len() < 2 {
          return Some(Ok(Expr::List(vec![])));
        }
        let mut result = Vec::with_capacity(items.len() - 1);
        for i in 1..items.len() {
          let ratio = match evaluate_expr_to_expr(&Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(items[i].clone()),
            right: Box::new(items[i - 1].clone()),
          }) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
          };
          result.push(ratio);
        }
        return Some(Ok(Expr::List(result)));
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
      if let Expr::List(items) = &args[1] {
        if items.is_empty() {
          return Some(Ok(Expr::List(vec![])));
        }
        let init = items[0].clone();
        let rest = Expr::List(items[1..].to_vec());
        return Some(list_helpers_ast::fold_list_ast(&args[0], &init, &rest));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "FoldList".to_string(),
        args: args.to_vec(),
      }));
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
      return Some(list_helpers_ast::transpose_ast(&args[0]));
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
    "NestWhile" if args.len() >= 3 => {
      let max_iter = if args.len() == 4 {
        expr_to_i128(&args[3])
      } else {
        None
      };
      return Some(list_helpers_ast::nest_while_ast(
        &args[0], &args[1], &args[2], max_iter,
      ));
    }
    "NestWhileList" if args.len() >= 3 => {
      let max_iter = if args.len() == 4 {
        expr_to_i128(&args[3])
      } else {
        None
      };
      return Some(list_helpers_ast::nest_while_list_ast(
        &args[0], &args[1], &args[2], max_iter,
      ));
    }
    "Thread" if args.len() == 1 => {
      return Some(list_helpers_ast::thread_ast(&args[0], None));
    }
    "Thread" if args.len() == 2 => {
      if let Expr::Identifier(head) = &args[1] {
        return Some(list_helpers_ast::thread_ast(&args[0], Some(head)));
      }
      return Some(list_helpers_ast::thread_ast(&args[0], None));
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
    "MinimalBy" if args.len() == 2 => {
      return Some(list_helpers_ast::minimal_by_ast(&args[0], &args[1]));
    }
    "MaximalBy" if args.len() == 2 => {
      return Some(list_helpers_ast::maximal_by_ast(&args[0], &args[1]));
    }
    "ArrayDepth" if args.len() == 1 => {
      return Some(list_helpers_ast::array_depth_ast(&args[0]));
    }
    "TensorRank" if args.len() == 1 => {
      return Some(list_helpers_ast::tensor_rank_ast(&args[0]));
    }
    "ArrayQ" if args.len() == 1 => {
      return Some(list_helpers_ast::array_q_ast(&args[0]));
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
    "DeleteCases" if args.len() == 3 || args.len() == 4 => {
      // DeleteCases[list, pattern, levelspec] or DeleteCases[list, pattern, levelspec, n]
      // For now, levelspec is ignored (treated as level 1)
      let max_count = if args.len() == 4 {
        expr_to_i128(&args[3])
      } else {
        None
      };
      return Some(list_helpers_ast::delete_cases_with_count_ast(
        &args[0], &args[1], max_count,
      ));
    }
    "MinMax" if args.len() == 1 => {
      return Some(list_helpers_ast::min_max_ast(&args[0]));
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
    "GatherBy" if args.len() == 2 => {
      return Some(list_helpers_ast::gather_by_ast(&args[1], &args[0]));
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
    "ArrayReshape" if args.len() == 2 => {
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
          let applied = Expr::FunctionCall {
            name: crate::syntax::expr_to_string(f),
            args: vec![elem.clone()],
          };
          let key = crate::evaluator::evaluate_expr_to_expr(&applied)
            .unwrap_or(applied);
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
        let pairs: Vec<Expr> = keys
          .into_iter()
          .zip(counts.into_iter())
          .map(|(k, c)| Expr::FunctionCall {
            name: "Rule".to_string(),
            args: vec![k, Expr::Integer(c)],
          })
          .collect();
        let assoc = Expr::FunctionCall {
          name: "Association".to_string(),
          args: pairs,
        };
        return Some(crate::evaluator::evaluate_expr_to_expr(&assoc));
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
            Expr::Function { body } => {
              let substituted = crate::syntax::substitute_slots(
                body,
                &[state.clone(), elem.clone()],
              );
              substituted
            }
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
              if let Some(ref kv2) = key_val2 {
                if crate::syntax::expr_to_string(kv)
                  == crate::syntax::expr_to_string(kv2)
                {
                  // Merge the two associations
                  let merged = merge_associations(a1, a2);
                  results.push(merged);
                }
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
    // SequenceCount[list, sublist] — count non-overlapping occurrences
    "SequenceCount" if args.len() == 2 => {
      if let (Expr::List(list), Expr::List(sub)) = (&args[0], &args[1]) {
        if sub.is_empty() {
          return Some(Ok(Expr::Integer(0)));
        }
        let sub_len = sub.len();
        let sub_strs: Vec<String> =
          sub.iter().map(|e| expr_to_string(e)).collect();
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

    _ => {}
  }
  None
}

/// Extract a value from an association by key string
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

  // Parse the block matrix structure
  let mut all_block_rows: Vec<Vec<Vec<Vec<Expr>>>> = Vec::new();

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

    let mut parsed_blocks: Vec<Vec<Vec<Expr>>> = Vec::new();
    for block in blocks_in_row {
      match block {
        Expr::List(rows) => {
          let mut matrix: Vec<Vec<Expr>> = Vec::new();
          for row in rows {
            match row {
              Expr::List(cols) => matrix.push(cols.clone()),
              // Scalar treated as 1x1 matrix
              other => matrix.push(vec![other.clone()]),
            }
          }
          parsed_blocks.push(matrix);
        }
        // Scalar treated as 1x1
        other => parsed_blocks.push(vec![vec![other.clone()]]),
      }
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
  let n = if args.len() >= 3 {
    match &args[2] {
      Expr::Integer(n) => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Nearest".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    items.len() // return all, sorted by distance (we'll take the closest group)
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

  if args.len() >= 3 {
    // Return exactly n nearest
    let result: Vec<Expr> = distances
      .iter()
      .take(n)
      .map(|(i, _)| items[*i].clone())
      .collect();
    Ok(Expr::List(result))
  } else {
    // Return all elements tied for the minimum distance
    let min_dist = distances[0].1;
    let result: Vec<Expr> = distances
      .iter()
      .take_while(|(_, d)| (*d - min_dist).abs() < 1e-15)
      .map(|(i, _)| items[*i].clone())
      .collect();
    Ok(Expr::List(result))
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

  // Build the reshaped array, padding with 0 if needed
  let mut idx = 0;
  Ok(build_reshaped(&flat, &dims, 0, &mut idx))
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
) -> Expr {
  if depth == dims.len() - 1 {
    // Leaf level: collect dims[depth] elements
    let n = dims[depth];
    let mut row = Vec::with_capacity(n);
    for _ in 0..n {
      if *idx < flat.len() {
        row.push(flat[*idx].clone());
      } else {
        row.push(Expr::Integer(0));
      }
      *idx += 1;
    }
    Expr::List(row)
  } else {
    let n = dims[depth];
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
      result.push(build_reshaped(flat, dims, depth + 1, idx));
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
