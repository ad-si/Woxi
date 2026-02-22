#[allow(unused_imports)]
use super::*;
use crate::functions::list_helpers_ast;

pub fn dispatch_list_operations(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Map" if args.len() == 2 => {
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
    "Table" if args.len() == 2 => {
      return Some(list_helpers_ast::table_ast(&args[0], &args[1]));
    }
    "Table" if args.len() >= 3 => {
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
      // For other expressions, Normal is identity
      return Some(Ok(args[0].clone()));
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
                Ok(Expr::Identifier(s)) if s == "True" => {
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
      let sorted = match evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Sort".to_string(),
        args: args.to_vec(),
      }) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      if let Expr::List(mut items) = sorted {
        items.reverse();
        return Some(Ok(Expr::List(items)));
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
    "ArrayQ" if args.len() == 1 => {
      return Some(list_helpers_ast::array_q_ast(&args[0]));
    }
    "VectorQ" if args.len() == 1 => {
      return Some(list_helpers_ast::vector_q_ast(&args[0]));
    }
    "MatrixQ" if args.len() == 1 => {
      return Some(list_helpers_ast::matrix_q_ast(&args[0]));
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
      let mut result = list_helpers_ast::part_ast(&args[0], &args[1]);
      for idx in &args[2..] {
        match result {
          Ok(ref expr) => result = list_helpers_ast::part_ast(expr, idx),
          Err(_) => break,
        }
      }
      return Some(result);
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

    _ => {}
  }
  None
}
