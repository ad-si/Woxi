#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::make_sqrt;

pub fn dispatch_linear_algebra_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Dot" if args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::dot_ast(args));
    }
    "Det" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::det_ast(args));
    }
    "Permanent" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::permanent_ast(args));
    }
    "Minors" if !args.is_empty() && args.len() <= 3 => {
      return Some(crate::functions::linear_algebra_ast::minors_ast(args));
    }
    "Inverse" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::inverse_ast(args));
    }
    "PseudoInverse" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::pseudo_inverse_ast(
        args,
      ));
    }
    "Orthogonalize" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::orthogonalize_ast(
        args,
      ));
    }
    "DistanceMatrix" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::list_helpers_ast::distance_matrix_ast(
        args,
      ));
    }
    "LinearSolve" if args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::linear_solve_ast(
        args,
      ));
    }
    "LeastSquares" if args.len() == 2 => {
      // LeastSquares[A, b] = Inverse[Transpose[A] . A] . Transpose[A] . b
      // when A has full column rank. For rank-deficient matrices the normal
      // equations have no inverse; fall back to PseudoInverse[A] . b which
      // yields the minimum-norm least-squares solution (matching Wolfram).
      let is_matrix = matches!(&args[0], Expr::List(rows)
        if !rows.is_empty() && rows.iter().all(|r| matches!(r, Expr::List(_))));
      let is_vector = matches!(&args[1], Expr::List(_));
      if is_matrix && is_vector {
        use crate::evaluator::evaluate_function_call_ast as eval;
        let a = args[0].clone();
        let b = args[1].clone();
        let at = match eval("Transpose", &[a.clone()]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let ata = match eval("Dot", &[at.clone(), a.clone()]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let atb = match eval("Dot", &[at, b.clone()]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        // Try Inverse[A^T A] · A^T b. If A^T A is singular, Inverse leaves
        // the call unevaluated as `Inverse[<matrix>]`; in that case use
        // PseudoInverse[A] · b instead.
        let inv_ata = match eval("Inverse", &[ata.clone()]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let normal_form_failed = matches!(
          &inv_ata,
          Expr::FunctionCall { name, .. } if name == "Inverse"
        );
        if normal_form_failed {
          let pinv = match eval("PseudoInverse", &[a]) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
          };
          return Some(eval("Dot", &[pinv, b]));
        }
        return Some(eval("Dot", &[inv_ata, atb]));
      }
    }
    "Tr" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::tr_ast(args));
    }
    "IdentityMatrix" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::identity_matrix_ast(
        args,
      ));
    }
    "UnitVector" if args.len() == 1 || args.len() == 2 => {
      // UnitVector[k] is shorthand for UnitVector[2, k].
      let (n_opt, k_opt) = if args.len() == 1 {
        (Some(2i128), expr_to_i128(&args[0]))
      } else {
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
      };
      if let (Some(n), Some(k)) = (n_opt, k_opt)
        && n > 0
      {
        if k >= 1 && k <= n {
          let mut vec = vec![Expr::Integer(0); n as usize];
          vec[(k - 1) as usize] = Expr::Integer(1);
          return Some(Ok(Expr::List(vec.into())));
        }
        // Out-of-range direction: a non-positive k is ::intpm (the direction
        // argument must be a positive integer), while a too-large positive k
        // is ::nokun. Both stay unevaluated, matching wolframscript.
        let call = Expr::FunctionCall {
          name: "UnitVector".to_string(),
          args: args.to_vec().into(),
        };
        if k < 1 {
          crate::emit_message(&format!(
            "UnitVector::intpm: Positive machine-sized integer expected at position {} in {}.",
            args.len(),
            crate::syntax::expr_to_string(&call)
          ));
        } else {
          crate::emit_message(&format!(
            "UnitVector::nokun: There is no unit vector in direction {k} in {n} dimensions."
          ));
        }
        return Some(Ok(call));
      }
    }
    // BoxMatrix[r] / BoxMatrix[{r1, …, rn}] — the all-ones "box" structuring
    // element. A scalar r is the 2D box of radius r; a list of radii gives an
    // n-D box of shape (2ri+1).
    "BoxMatrix" if args.len() == 1 => {
      let radii: Option<Vec<i128>> = match &args[0] {
        Expr::List(elems) => elems.iter().map(cross_radius).collect(),
        other => cross_radius(other).map(|r| vec![r, r]),
      };
      if let Some(radii) = radii
        && !radii.is_empty()
      {
        let dims: Vec<usize> =
          radii.iter().map(|r| (2 * r + 1) as usize).collect();
        return Some(Ok(build_ones_tensor(&dims)));
      }
    }
    "DiagonalMatrix" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::diagonal_matrix_ast(
        args,
      ));
    }
    // A list of radii gives the rectangular / n-D diamond; a scalar keeps the
    // existing square-diamond implementation.
    "DiamondMatrix" if args.len() == 1 && matches!(&args[0], Expr::List(_)) => {
      if let Expr::List(elems) = &args[0]
        && let Some(radii) =
          elems.iter().map(cross_radius).collect::<Option<Vec<_>>>()
        && !radii.is_empty()
      {
        return Some(Ok(build_diamond_matrix(&radii)));
      }
    }
    "DiamondMatrix" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::diamond_matrix_ast(
        args,
      ));
    }
    // A list of radii gives the elliptical / n-D disk; a scalar keeps the
    // existing round-disk implementation.
    "DiskMatrix" if args.len() == 1 && matches!(&args[0], Expr::List(_)) => {
      if let Expr::List(elems) = &args[0]
        && let Some(radii) =
          elems.iter().map(cross_radius).collect::<Option<Vec<_>>>()
        && !radii.is_empty()
      {
        return Some(Ok(build_disk_matrix(&radii)));
      }
    }
    "DiskMatrix" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::disk_matrix_ast(args));
    }
    "HilbertMatrix" if args.len() == 1 => {
      // Accept either HilbertMatrix[n] (square) or HilbertMatrix[{m, n}]
      // (rectangular). Both fill entry (i, j) with 1/(i + j - 1).
      let dims = match &args[0] {
        Expr::List(items) if items.len() == 2 => {
          match (expr_to_i128(&items[0]), expr_to_i128(&items[1])) {
            (Some(m), Some(n)) if m >= 0 && n >= 0 => {
              Some((m as usize, n as usize))
            }
            _ => None,
          }
        }
        other => expr_to_i128(other)
          .filter(|n| *n >= 0)
          .map(|n| (n as usize, n as usize)),
      };
      if let Some((m, n)) = dims {
        let mut rows = Vec::with_capacity(m);
        for i in 0..m {
          let mut row = Vec::with_capacity(n);
          for j in 0..n {
            let denom = (i + j + 1) as i128;
            row.push(crate::functions::math_ast::make_rational_pub(1, denom));
          }
          rows.push(Expr::List(row.into()));
        }
        return Some(Ok(Expr::List(rows.into())));
      }
    }
    // ToeplitzMatrix[col, row] — first column `col`, first row `row`. Entry
    // (i, j) is col[i-j] on/below the diagonal and row[j-i] above it. The
    // result is len(col) x len(row); the shared corner uses col[0].
    "ToeplitzMatrix" if args.len() == 2 => {
      if let (Expr::List(col), Expr::List(row)) = (&args[0], &args[1])
        && !col.is_empty()
        && !row.is_empty()
      {
        let (h, w) = (col.len(), row.len());
        let mut rows = Vec::with_capacity(h);
        for i in 0..h {
          let mut r = Vec::with_capacity(w);
          for j in 0..w {
            r.push(if i >= j {
              col[i - j].clone()
            } else {
              row[j - i].clone()
            });
          }
          rows.push(Expr::List(r.into()));
        }
        return Some(Ok(Expr::List(rows.into())));
      }
    }
    "ToeplitzMatrix" if args.len() == 1 => {
      if let Expr::List(elems) = &args[0] {
        let n = elems.len();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
          let mut row = Vec::with_capacity(n);
          for j in 0..n {
            let idx = i.abs_diff(j);
            row.push(elems[idx].clone());
          }
          rows.push(Expr::List(row.into()));
        }
        return Some(Ok(Expr::List(rows.into())));
      }
      // ToeplitzMatrix[n] — integer form. First row and column are
      // 1..n, so entry (i, j) = |i - j| + 1.
      if let Some(n) = expr_to_i128(&args[0])
        && n >= 0
      {
        let n = n as usize;
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
          let mut row = Vec::with_capacity(n);
          for j in 0..n {
            row.push(Expr::Integer((i.abs_diff(j) + 1) as i128));
          }
          rows.push(Expr::List(row.into()));
        }
        return Some(Ok(Expr::List(rows.into())));
      }
    }
    "UnitaryMatrixQ" | "OrthogonalMatrixQ" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        if n == 0 {
          return Some(Ok(Expr::Identifier("False".to_string())));
        }
        let is_square = rows.iter().all(|r| {
          if let Expr::List(cols) = r {
            cols.len() == n
          } else {
            false
          }
        });
        if !is_square {
          return Some(Ok(Expr::Identifier("False".to_string())));
        }
        let transpose =
          crate::functions::list_helpers_ast::transpose_ast(&args[0]);
        if let Ok(t) = transpose {
          let dot = crate::functions::linear_algebra_ast::dot_ast(&[
            t,
            args[0].clone(),
          ]);
          if let Ok(product) = dot
            && let Ok(evaluated) =
              crate::evaluator::evaluate_expr_to_expr(&product)
          {
            let identity =
              crate::functions::linear_algebra_ast::identity_matrix_ast(&[
                Expr::Integer(n as i128),
              ]);
            if let Ok(id) = identity {
              let result = crate::syntax::expr_to_string(&evaluated)
                == crate::syntax::expr_to_string(&id);
              return Some(Ok(Expr::Identifier(
                if result { "True" } else { "False" }.to_string(),
              )));
            }
          }
        }
      }
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    "ReflectionMatrix" if args.len() == 1 => {
      if let Expr::List(v) = &args[0] {
        let n = v.len();
        if n > 0 {
          let dot_vv = Expr::FunctionCall {
            name: "Dot".to_string(),
            args: vec![args[0].clone(), args[0].clone()].into(),
          };
          let mut rows = Vec::with_capacity(n);
          for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
              let delta = if i == j {
                Expr::Integer(1)
              } else {
                Expr::Integer(0)
              };
              let vi_vj = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(v[i].clone()),
                right: Box::new(v[j].clone()),
              };
              let two_vi_vj = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Integer(2)),
                right: Box::new(vi_vj),
              };
              let frac = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(two_vi_vj),
                right: Box::new(dot_vv.clone()),
              };
              let entry = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Minus,
                left: Box::new(delta),
                right: Box::new(frac),
              };
              row.push(entry);
            }
            rows.push(Expr::List(row.into()));
          }
          let result = Expr::List(rows.into());
          return Some(crate::evaluator::evaluate_expr_to_expr(&result));
        }
      }
    }
    "LeviCivitaTensor" if args.len() == 1 => {
      return Some(
        crate::functions::linear_algebra_ast::levi_civita_tensor_sparse_ast(
          args,
        ),
      );
    }
    "LeviCivitaTensor" if args.len() == 2 => {
      if matches!(&args[1], Expr::Identifier(h) if h == "List") {
        return Some(
          crate::functions::linear_algebra_ast::levi_civita_tensor_ast(
            &args[..1],
          ),
        );
      }
      if matches!(&args[1], Expr::Identifier(h) if h == "SparseArray") {
        return Some(
          crate::functions::linear_algebra_ast::levi_civita_tensor_sparse_ast(
            &args[..1],
          ),
        );
      }
    }
    "Eigenvalues" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::eigenvalues_ast(args));
    }
    "Eigenvalues" if args.len() == 2 => {
      return Some(
        crate::functions::linear_algebra_ast::eigenvalues_count_ast(args),
      );
    }
    "PrincipalComponents" if args.len() == 1 => {
      return Some(
        crate::functions::linear_algebra_ast::principal_components_ast(args),
      );
    }
    "SmithDecomposition" if args.len() == 1 => {
      return Some(
        crate::functions::linear_algebra_ast::smith_decomposition_ast(args),
      );
    }
    "SymmetricMatrixQ" if args.len() == 1 => {
      return Some(crate::functions::list_helpers_ast::symmetric_matrix_q_ast(
        &args[0],
      ));
    }
    "PositiveDefiniteMatrixQ" if args.len() == 1 => {
      // Non-square arguments are not positive definite. Guard before calling
      // eigenvalues_ast, which would otherwise emit a spurious matsq message.
      if !is_square_matrix_expr(&args[0]) {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      // Compute eigenvalues and check all are strictly positive
      let eigenvals_result =
        crate::functions::linear_algebra_ast::eigenvalues_ast(args);
      if let Ok(list_expr) = eigenvals_result
        && let Expr::List(eigenvals) = &list_expr
      {
        for ev in eigenvals {
          let n_val =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "N".to_string(),
              args: vec![ev.clone()].into(),
            });
          match n_val {
            Ok(Expr::Real(f)) if f > 0.0 => continue,
            Ok(Expr::Integer(n)) if n > 0 => continue,
            _ => return Some(Ok(Expr::Identifier("False".to_string()))),
          }
        }
        return Some(Ok(Expr::Identifier("True".to_string())));
      }
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    "IndefiniteMatrixQ" if args.len() == 1 => {
      // A matrix is "explicitly indefinite" iff its Hermitian part
      // H = (m + ConjugateTranspose[m])/2 has at least one strictly
      // positive eigenvalue AND at least one strictly negative eigenvalue.
      // Anything that can't be confirmed indefinite returns False.
      let false_res = || Some(Ok(Expr::Identifier("False".to_string())));
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        if n == 0 {
          return false_res();
        }
        // Require a square matrix (every row a list of length n).
        for row in rows {
          match row {
            Expr::List(cols) if cols.len() == n => {}
            _ => return false_res(),
          }
        }
        // Build the Hermitian part: (m + ConjugateTranspose[m]) / 2.
        let conj_t = Expr::FunctionCall {
          name: "ConjugateTranspose".to_string(),
          args: vec![args[0].clone()].into(),
        };
        let herm = Expr::FunctionCall {
          name: "Divide".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![args[0].clone(), conj_t].into(),
            },
            Expr::Integer(2),
          ]
          .into(),
        };
        let herm_eval =
          evaluate_expr_to_expr(&herm).unwrap_or_else(|_| herm.clone());
        if let Ok(Expr::List(ref eigenvals)) =
          crate::functions::linear_algebra_ast::eigenvalues_ast(&[herm_eval])
        {
          let mut has_pos = false;
          let mut has_neg = false;
          for ev in eigenvals.iter() {
            let n_val = evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "N".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Re".to_string(),
                args: vec![ev.clone()].into(),
              }]
              .into(),
            });
            match n_val {
              Ok(Expr::Real(f)) if f > 0.0 => has_pos = true,
              Ok(Expr::Real(f)) if f < 0.0 => has_neg = true,
              Ok(Expr::Integer(v)) if v > 0 => has_pos = true,
              Ok(Expr::Integer(v)) if v < 0 => has_neg = true,
              _ => {}
            }
          }
          if has_pos && has_neg {
            return Some(Ok(Expr::Identifier("True".to_string())));
          }
        }
        return false_res();
      }
      return false_res();
    }
    "Eigenvectors" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::eigenvectors_ast(
        args,
      ));
    }
    "Eigenvectors" if args.len() == 2 => {
      return Some(
        crate::functions::linear_algebra_ast::eigenvectors_count_ast(args),
      );
    }
    "Eigensystem" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::eigensystem_ast(args));
    }
    "Eigensystem" if args.len() == 2 => {
      return Some(
        crate::functions::linear_algebra_ast::eigensystem_count_ast(args),
      );
    }
    "RowReduce" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::row_reduce_ast(args));
    }
    "MatrixRank" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::matrix_rank_ast(args));
    }
    "NullSpace" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::null_space_ast(args));
    }
    "ConjugateTranspose" if args.len() == 1 => {
      return Some(
        crate::functions::linear_algebra_ast::conjugate_transpose_ast(args),
      );
    }
    "DesignMatrix" if args.len() == 3 => {
      return Some(crate::functions::linear_algebra_ast::design_matrix_ast(
        args,
      ));
    }
    "Fit" if args.len() == 3 => {
      return Some(crate::functions::linear_algebra_ast::fit_ast(args));
    }
    "LinearModelFit" if args.len() == 1 || args.len() == 3 => {
      return Some(crate::functions::linear_algebra_ast::linear_model_fit_ast(
        args,
      ));
    }
    "LogitModelFit" if args.len() == 3 => {
      return Some(crate::functions::linear_algebra_ast::logit_model_fit_ast(
        args,
      ));
    }
    "FindFit" if args.len() == 4 => {
      return Some(crate::functions::linear_algebra_ast::find_fit_ast(args));
    }
    "Cross" if !args.is_empty() => {
      return Some(crate::functions::linear_algebra_ast::cross_ast(args));
    }
    // DotProduct[a, b] is the legacy VectorAnalysis spelling of Dot[a, b];
    // The legacy `VectorAnalysis` package functions — DotProduct,
    // CrossProduct, ScalarTripleProduct, Coordinates, SetCoordinates,
    // CoordinatesToCartesian, CoordinatesFromCartesian — are deliberately
    // left unevaluated to match wolframscript's `-code` behaviour, which
    // does not load that legacy package.
    "TensorWedge" if !args.is_empty() => {
      return Some(crate::functions::linear_algebra_ast::tensor_wedge_ast(
        args,
      ));
    }
    "VectorAngle" if args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::vector_angle_ast(
        args,
      ));
    }
    "CoordinateTransform" if args.len() == 2 => {
      return Some(
        crate::functions::linear_algebra_ast::coordinate_transform_ast(args),
      );
    }
    "DiceDissimilarity"
    | "JaccardDissimilarity"
    | "MatchingDissimilarity"
    | "RogersTanimotoDissimilarity"
    | "RussellRaoDissimilarity"
    | "SokalSneathDissimilarity"
    | "YuleDissimilarity"
      if args.len() == 2 =>
    {
      return Some(binary_dissimilarity_ast(name, &args[0], &args[1]));
    }
    "UpperTriangularize" if args.len() == 1 || args.len() == 2 => {
      return Some(
        crate::functions::linear_algebra_ast::upper_triangularize_ast(args),
      );
    }
    "LowerTriangularize" if args.len() == 1 || args.len() == 2 => {
      return Some(
        crate::functions::linear_algebra_ast::lower_triangularize_ast(args),
      );
    }
    "KroneckerProduct" if args.len() >= 2 => {
      return Some(kronecker_product_ast(args));
    }
    "CharacteristicPolynomial" if args.len() == 2 => {
      // CharacteristicPolynomial[A, x] = Det[A - x*IdentityMatrix[n]]
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        if n > 0
          && rows
            .iter()
            .all(|r| matches!(r, Expr::List(cols) if cols.len() == n))
        {
          let x = &args[1];
          // Build A - x*I
          let mut new_rows = Vec::with_capacity(n);
          for (i, row) in rows.iter().enumerate() {
            if let Expr::List(cols) = row {
              let mut new_cols = Vec::with_capacity(n);
              for (j, elem) in cols.iter().enumerate() {
                if i == j {
                  // a_ij - x
                  let entry = Expr::FunctionCall {
                    name: "Plus".to_string(),
                    args: vec![
                      elem.clone(),
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: vec![Expr::Integer(-1), x.clone()].into(),
                      },
                    ]
                    .into(),
                  };
                  new_cols.push(entry);
                } else {
                  new_cols.push(elem.clone());
                }
              }
              new_rows.push(Expr::List(new_cols.into()));
            }
          }
          let modified_mat = Expr::List(new_rows.into());
          let det_result =
            crate::functions::linear_algebra_ast::det_ast(&[modified_mat]);
          match det_result {
            Ok(det_expr) => {
              return Some(evaluate_expr_to_expr(&det_expr));
            }
            Err(e) => return Some(Err(e)),
          }
        }
      }
    }
    "MatrixMinimalPolynomial" if args.len() == 2 => {
      // A non-(square matrix) first argument is a matsq error in wolframscript
      // (returns unevaluated). A square matrix that we cannot reduce — e.g. one
      // with symbolic entries — simply stays unevaluated without a message.
      let is_square = matches!(&args[0], Expr::List(rows)
        if !rows.is_empty()
          && rows.iter().all(|r| matches!(r, Expr::List(c) if c.len() == rows.len())));
      if !is_square {
        crate::emit_message(&format!(
          "MatrixMinimalPolynomial::matsq: Argument {} at position 1 is not \
           a nonempty square matrix.",
          crate::syntax::expr_to_string(&args[0])
        ));
        return Some(Ok(Expr::FunctionCall {
          name: "MatrixMinimalPolynomial".to_string(),
          args: args.to_vec().into(),
        }));
      }
      if let Some(result) = matrix_minimal_polynomial(&args[0], &args[1]) {
        return Some(result);
      }
    }
    "Projection" if args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::projection_ast(args));
    }
    "LatticeReduce" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::lattice_reduce_ast(
        args,
      ));
    }
    "FindIntegerNullVector" if !args.is_empty() && args.len() <= 2 => {
      return Some(
        crate::functions::linear_algebra_ast::find_integer_null_vector_ast(
          args,
        ),
      );
    }
    "MatrixExp" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::matrix_function_ast(
        args,
        "MatrixExp",
        "Exp",
      ));
    }
    // MatrixExp[m, v] computes MatrixExp[m] . v (the action of the matrix
    // exponential on the vector v), avoiding forming the full exponential
    // when only its product with v is wanted. The second argument must be a
    // nonempty vector (a flat list of scalars).
    "MatrixExp" if args.len() == 2 => {
      let is_vector = matches!(&args[1],
        Expr::List(items)
          if !items.is_empty()
            && items.iter().all(|e| !matches!(e, Expr::List(_))));
      if !is_vector {
        crate::emit_message(&format!(
          "MatrixExp::vector: Argument {} at position 2 is not a nonempty \
           vector.",
          crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output)
        ));
        return Some(Ok(Expr::FunctionCall {
          name: "MatrixExp".to_string(),
          args: args.to_vec().into(),
        }));
      }
      let exp = match crate::functions::linear_algebra_ast::matrix_function_ast(
        std::slice::from_ref(&args[0]),
        "MatrixExp",
        "Exp",
      ) {
        Ok(e) => e,
        Err(e) => return Some(Err(e)),
      };
      // If the exponential could not be computed it stays a MatrixExp[...]
      // call; surface the original two-argument form unevaluated.
      if matches!(&exp, Expr::FunctionCall { name, .. } if name == "MatrixExp")
      {
        return Some(Ok(Expr::FunctionCall {
          name: "MatrixExp".to_string(),
          args: args.to_vec().into(),
        }));
      }
      return Some(evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Dot".to_string(),
        args: vec![exp, args[1].clone()].into(),
      }));
    }
    "MatrixLog" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::matrix_function_ast(
        args,
        "MatrixLog",
        "Log",
      ));
    }
    "MatrixPower" if args.len() == 2 => {
      // Validate the matrix independently of the exponent: a non-square or
      // non-matrix concrete first argument emits MatrixPower::matsq (a valid
      // matrix with a symbolic exponent still falls through to the symbolic
      // handling below).
      let valid_matrix = matches!(
        crate::functions::linear_algebra_ast::expr_to_matrix(&args[0]),
        Some(ref m)
          if crate::functions::linear_algebra_ast::is_nonempty_square(m)
      );
      if !valid_matrix {
        return Some(Ok(
          crate::functions::linear_algebra_ast::matsq_unevaluated(
            "MatrixPower",
            args,
          ),
        ));
      }
      if let Some(n) = expr_to_i128(&args[1]) {
        let mat = &args[0];
        // Check matrix is a list of lists and is square
        if let Expr::List(rows) = mat {
          let size = rows.len();
          if size == 0 {
            return Some(Ok(mat.clone()));
          }
          // Check all rows have same length = size
          let is_square = rows.iter().all(|r| {
            if let Expr::List(cols) = r {
              cols.len() == size
            } else {
              false
            }
          });
          if !is_square {
            return Some(Ok(Expr::FunctionCall {
              name: "MatrixPower".to_string(),
              args: args.to_vec().into(),
            }));
          }

          if n == 0 {
            // Return identity matrix
            return Some(
              crate::functions::linear_algebra_ast::identity_matrix_ast(&[
                Expr::Integer(size as i128),
              ]),
            );
          }

          let (base, exp) = if n < 0 {
            // Compute inverse first
            match crate::functions::linear_algebra_ast::inverse_ast(&[
              mat.clone()
            ]) {
              Ok(inv) => (inv, (-n) as u64),
              Err(e) => return Some(Err(e)),
            }
          } else {
            (mat.clone(), n as u64)
          };

          // Exponentiation by squaring
          let mut result = None;
          let mut power = base;
          let mut exp = exp;
          while exp > 0 {
            if exp & 1 == 1 {
              result = Some(match result {
                None => power.clone(),
                Some(r) => {
                  match crate::functions::linear_algebra_ast::dot_ast(&[
                    r,
                    power.clone(),
                  ]) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                  }
                }
              });
            }
            exp >>= 1;
            if exp > 0 {
              power = match crate::functions::linear_algebra_ast::dot_ast(&[
                power.clone(),
                power,
              ]) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
              };
            }
          }

          return Some(Ok(result.unwrap_or(mat.clone())));
        }
      }
      // Symbolic n with a matrix that splits (after a row/column
      // permutation) into multiple connected components: handle each
      // block independently. Blocks of size 1 reduce to `entry^n`;
      // size-2 blocks use the closed form for 2×2 powers. Single-
      // block matrices fall through to the unevaluated head.
      if let Expr::List(rows) = &args[0]
        && !rows.is_empty()
        && let Some(result) = matrix_power_block_symbolic(rows, &args[1])
      {
        return Some(Ok(result));
      }
      // Symbolic: return unevaluated
      return Some(Ok(Expr::FunctionCall {
        name: "MatrixPower".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "CellularAutomaton" if args.len() == 3 => {
      return Some(
        crate::functions::cellular_automaton_ast::cellular_automaton_ast(args),
      );
    }
    "TuringMachine" if args.len() == 3 => {
      return Some(crate::functions::turing_machine_ast::turing_machine_ast(
        args,
      ));
    }
    // RotationMatrix[{u, v}] — the rotation taking vector u to the direction
    // of v. In 2D this is the planar rotation by the signed angle from u to v:
    // cos = (u.v)/(|u||v|), sin = (ux vy - uy vx)/(|u||v|). Higher-dimensional
    // vector pairs are left unevaluated rather than mishandled as an angle.
    "RotationMatrix"
      if args.len() == 1
        && matches!(&args[0], Expr::List(p)
          if p.len() == 2 && p.iter().all(|e| matches!(e, Expr::List(_)))) =>
    {
      let Expr::List(pair) = &args[0] else {
        unreachable!()
      };
      if let (Expr::List(u), Expr::List(v)) = (&pair[0], &pair[1])
        && u.len() == 2
        && v.len() == 2
      {
        let times = |a: Expr, b: Expr| Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![a, b].into(),
        };
        let plus = |a: Expr, b: Expr| Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![a, b].into(),
        };
        let sq = |a: Expr| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![a, Expr::Integer(2)].into(),
        };
        let neg = |a: Expr| times(Expr::Integer(-1), a);
        let (ux, uy, vx, vy) =
          (u[0].clone(), u[1].clone(), v[0].clone(), v[1].clone());
        let dot =
          plus(times(ux.clone(), vx.clone()), times(uy.clone(), vy.clone()));
        let cross = plus(
          times(ux.clone(), vy.clone()),
          neg(times(uy.clone(), vx.clone())),
        );
        let normsq = times(plus(sq(ux), sq(uy)), plus(sq(vx), sq(vy)));
        let inv_d = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![normsq].into(),
            },
            Expr::Integer(-1),
          ]
          .into(),
        };
        let cos = times(dot, inv_d.clone());
        let sin = times(cross, inv_d);
        let mat = Expr::List(
          vec![
            Expr::List(vec![cos.clone(), neg(sin.clone())].into()),
            Expr::List(vec![sin, cos].into()),
          ]
          .into(),
        );
        return Some(evaluate_expr_to_expr(&mat));
      }
      // A higher-dimensional vector pair: not implemented, stay unevaluated.
      return Some(Ok(Expr::FunctionCall {
        name: "RotationMatrix".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "RotationMatrix" if args.len() == 1 => {
      // 2D rotation matrix: {{Cos[θ], -Sin[θ]}, {Sin[θ], Cos[θ]}}
      let theta = &args[0];
      let cos = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![theta.clone()].into(),
      };
      let sin = Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![theta.clone()].into(),
      };
      let neg_sin = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), sin.clone()].into(),
      };
      let mat = Expr::List(
        vec![
          Expr::List(vec![cos.clone(), neg_sin].into()),
          Expr::List(vec![sin, cos].into()),
        ]
        .into(),
      );
      return Some(evaluate_expr_to_expr(&mat));
    }
    "RotationMatrix" if args.len() == 2 => {
      // 3D rotation matrix around axis {ux, uy, uz} by angle theta
      // Rodrigues' rotation formula
      if let Expr::List(axis) = &args[1]
        && axis.len() == 3
      {
        let theta = &args[0];
        let c = Expr::FunctionCall {
          name: "Cos".to_string(),
          args: vec![theta.clone()].into(),
        };
        let s = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![theta.clone()].into(),
        };
        let one_minus_c = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), c.clone()].into(),
            },
          ]
          .into(),
        };
        let ux = &axis[0];
        let uy = &axis[1];
        let uz = &axis[2];

        // Build the rotation matrix using Rodrigues' formula
        // R_ij = cos(θ) δ_ij + (1-cos(θ)) u_i u_j - sin(θ) ε_ijk u_k
        let make_entry = |i: usize, j: usize| -> Expr {
          let u = [ux, uy, uz];
          let delta = if i == j {
            Expr::Integer(1)
          } else {
            Expr::Integer(0)
          };
          // Cross product sign for ε_ijk: (i,j)→k with sign
          let cross_term = match (i, j) {
            (0, 1) => Some((Expr::Integer(-1), uz)), // -sin*uz
            (1, 0) => Some((Expr::Integer(1), uz)),  // +sin*uz
            (0, 2) => Some((Expr::Integer(1), uy)),  // +sin*uy
            (2, 0) => Some((Expr::Integer(-1), uy)), // -sin*uy
            (1, 2) => Some((Expr::Integer(-1), ux)), // -sin*ux
            (2, 1) => Some((Expr::Integer(1), ux)),  // +sin*ux
            _ => None,
          };

          let mut terms = vec![
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![c.clone(), delta].into(),
            },
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![one_minus_c.clone(), u[i].clone(), u[j].clone()]
                .into(),
            },
          ];
          if let Some((sign, uk)) = cross_term {
            terms.push(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![sign, s.clone(), uk.clone()].into(),
            });
          }
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms.into(),
          }
        };

        let mat = Expr::List(
          vec![
            Expr::List(
              vec![make_entry(0, 0), make_entry(0, 1), make_entry(0, 2)].into(),
            ),
            Expr::List(
              vec![make_entry(1, 0), make_entry(1, 1), make_entry(1, 2)].into(),
            ),
            Expr::List(
              vec![make_entry(2, 0), make_entry(2, 1), make_entry(2, 2)].into(),
            ),
          ]
          .into(),
        );
        return Some(evaluate_expr_to_expr(&mat));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "RotationMatrix".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "EulerMatrix" if args.len() == 1 || args.len() == 2 => {
      return Some(euler_matrix_ast(args));
    }
    "LUDecomposition" if args.len() == 1 => {
      return Some(lu_decomposition_ast(&args[0]));
    }
    // TranslationTransform[{v1, v2, ...}] → TransformationFunction[augmented identity + translation]
    "TranslationTransform" if args.len() == 1 => {
      if let Expr::List(v) = &args[0] {
        let n = v.len();
        // Build (n+1)x(n+1) augmented identity matrix with translation in last column
        let mut rows = Vec::with_capacity(n + 1);
        for i in 0..n {
          let mut row = vec![Expr::Integer(0); n + 1];
          row[i] = Expr::Integer(1);
          row[n] = v[i].clone();
          rows.push(Expr::List(row.into()));
        }
        // Last row: all zeros except 1 in bottom-right
        let mut last_row = vec![Expr::Integer(0); n + 1];
        last_row[n] = Expr::Integer(1);
        rows.push(Expr::List(last_row.into()));
        let matrix = Expr::List(rows.into());
        let evaluated =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "TransformationFunction".to_string(),
            args: vec![matrix].into(),
          });
        return Some(evaluated);
      }
      return Some(Ok(Expr::FunctionCall {
        name: "TranslationTransform".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // RescalingTransform[{{min,max},...}] → maps the bounding box to the unit cube.
    // RescalingTransform[{{min,max},...}, {{ymin,ymax},...}] → maps to the target box.
    // Built as a TransformationFunction with an (n+1)x(n+1) augmented affine matrix:
    //   out_i = ymin_i + (ymax_i - ymin_i)/(max_i - min_i) * (x_i - min_i)
    // i.e. diagonal scale s_i = (ymax_i - ymin_i)/(max_i - min_i) and
    //      translation t_i = ymin_i - min_i * s_i.
    "RescalingTransform" if args.len() == 1 || args.len() == 2 => {
      let src = if let Expr::List(v) = &args[0] {
        v
      } else {
        return Some(Ok(Expr::FunctionCall {
          name: "RescalingTransform".to_string(),
          args: args.to_vec().into(),
        }));
      };
      // Optional target box.
      let tgt = if args.len() == 2 {
        if let Expr::List(v) = &args[1] {
          if v.len() != src.len() {
            return Some(Ok(Expr::FunctionCall {
              name: "RescalingTransform".to_string(),
              args: args.to_vec().into(),
            }));
          }
          Some(v)
        } else {
          return Some(Ok(Expr::FunctionCall {
            name: "RescalingTransform".to_string(),
            args: args.to_vec().into(),
          }));
        }
      } else {
        None
      };
      let n = src.len();
      // Each source/target entry must be a 2-element {lo, hi} list.
      let pair = |e: &Expr| -> Option<(Expr, Expr)> {
        if let Expr::List(p) = e
          && p.len() == 2
        {
          return Some((p[0].clone(), p[1].clone()));
        }
        None
      };
      let mut scales: Vec<Expr> = Vec::with_capacity(n);
      let mut translates: Vec<Expr> = Vec::with_capacity(n);
      for i in 0..n {
        let (min_i, max_i) = match pair(&src[i]) {
          Some(p) => p,
          None => {
            return Some(Ok(Expr::FunctionCall {
              name: "RescalingTransform".to_string(),
              args: args.to_vec().into(),
            }));
          }
        };
        let (ymin_i, ymax_i) = match &tgt {
          Some(t) => match pair(&t[i]) {
            Some(p) => p,
            None => {
              return Some(Ok(Expr::FunctionCall {
                name: "RescalingTransform".to_string(),
                args: args.to_vec().into(),
              }));
            }
          },
          None => (Expr::Integer(0), Expr::Integer(1)),
        };
        // scale = (ymax - ymin)/(max - min)
        let num = Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![ymax_i.clone(), ymin_i.clone()].into(),
        };
        let den = Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![max_i.clone(), min_i.clone()].into(),
        };
        let scale = Expr::FunctionCall {
          name: "Divide".to_string(),
          args: vec![num, den].into(),
        };
        // translate = ymin - min * scale
        let translate = Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![
            ymin_i,
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![min_i, scale.clone()].into(),
            },
          ]
          .into(),
        };
        scales.push(scale);
        translates.push(translate);
      }
      // Assemble the (n+1)x(n+1) augmented matrix.
      let mut rows = Vec::with_capacity(n + 1);
      for i in 0..n {
        let mut row = vec![Expr::Integer(0); n + 1];
        row[i] = scales[i].clone();
        row[n] = translates[i].clone();
        rows.push(Expr::List(row.into()));
      }
      let mut last_row = vec![Expr::Integer(0); n + 1];
      last_row[n] = Expr::Integer(1);
      rows.push(Expr::List(last_row.into()));
      let matrix = Expr::List(rows.into());
      return Some(crate::evaluator::evaluate_expr_to_expr(
        &Expr::FunctionCall {
          name: "TransformationFunction".to_string(),
          args: vec![matrix].into(),
        },
      ));
    }
    // TransformationMatrix[TransformationFunction[m]] → m. Extracts the
    // homogeneous matrix from a transformation; any other argument (a plain
    // matrix, a non-transform) stays unevaluated, matching wolframscript.
    "TransformationMatrix" if args.len() == 1 => {
      if let Expr::FunctionCall {
        name: tf_name,
        args: tf_args,
      } = &args[0]
        && tf_name == "TransformationFunction"
        && tf_args.len() == 1
      {
        return Some(Ok(tf_args[0].clone()));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "TransformationMatrix".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // RotationTransform[angle] → TransformationFunction[2D rotation matrix in homogeneous coords]
    // RotationTransform[angle, {cx, cy}] → rotation about point {cx, cy}
    "RotationTransform" if args.len() == 1 || args.len() == 2 => {
      // Two-vector form RotationTransform[{u, v}] (2D): the rotation taking the
      // direction of u to the direction of v. Build it directly from the
      // normalized dot/cross products so the angle is never materialized.
      //   cos = (u.v)/(|u||v|),  sin = (u1 v2 - u2 v1)/(|u||v|)
      if args.len() == 1
        && let Expr::List(pair) = &args[0]
        && pair.len() == 2
        && let (Expr::List(u), Expr::List(v)) = (&pair[0], &pair[1])
        && u.len() == 2
        && v.len() == 2
      {
        let times = |a: Expr, b: Expr| Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![a, b].into(),
        };
        let plus = |a: Expr, b: Expr| Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![a, b].into(),
        };
        let sq = |e: &Expr| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![e.clone(), Expr::Integer(2)].into(),
        };
        let sqrt = |e: Expr| Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![e].into(),
        };
        let dot = plus(
          times(u[0].clone(), v[0].clone()),
          times(u[1].clone(), v[1].clone()),
        );
        let cross = plus(
          times(u[0].clone(), v[1].clone()),
          times(Expr::Integer(-1), times(u[1].clone(), v[0].clone())),
        );
        let nu = sqrt(plus(sq(&u[0]), sq(&u[1])));
        let nv = sqrt(plus(sq(&v[0]), sq(&v[1])));
        let denom = times(nu, nv);
        let cos_t = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(dot),
          right: Box::new(denom.clone()),
        };
        let sin_t = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(cross),
          right: Box::new(denom),
        };
        let neg_sin = times(Expr::Integer(-1), sin_t.clone());
        let matrix = Expr::List(
          vec![
            Expr::List(vec![cos_t.clone(), neg_sin, Expr::Integer(0)].into()),
            Expr::List(vec![sin_t, cos_t, Expr::Integer(0)].into()),
            Expr::List(
              vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(1)].into(),
            ),
          ]
          .into(),
        );
        return Some(crate::evaluator::evaluate_expr_to_expr(
          &Expr::FunctionCall {
            name: "TransformationFunction".to_string(),
            args: vec![matrix].into(),
          },
        ));
      }
      let theta = &args[0];
      let cos_t = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![theta.clone()].into(),
      };
      let sin_t = Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![theta.clone()].into(),
      };
      let neg_sin_t = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), sin_t.clone()].into(),
      };
      // Default last column is the zero translation
      let (tx, ty) = if args.len() == 2 {
        if let Expr::List(center) = &args[1]
          && center.len() == 2
        {
          let cx = center[0].clone();
          let cy = center[1].clone();
          // Translation column for rotation about {cx,cy}:
          // tx = cx - cx*cos + cy*sin
          // ty = cy - cx*sin - cy*cos
          let neg_cx_cos = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), cx.clone(), cos_t.clone()].into(),
          };
          let cy_sin = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![cy.clone(), sin_t.clone()].into(),
          };
          let tx_e = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![cx.clone(), neg_cx_cos, cy_sin].into(),
          };
          let neg_cx_sin = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), cx.clone(), sin_t.clone()].into(),
          };
          let neg_cy_cos = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), cy.clone(), cos_t.clone()].into(),
          };
          let ty_e = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![cy.clone(), neg_cx_sin, neg_cy_cos].into(),
          };
          (tx_e, ty_e)
        } else {
          // Non-list or wrong-dim center: leave unevaluated.
          return Some(Ok(Expr::FunctionCall {
            name: "RotationTransform".to_string(),
            args: args.to_vec().into(),
          }));
        }
      } else {
        (Expr::Integer(0), Expr::Integer(0))
      };
      // Build the 3x3 homogeneous rotation matrix:
      // {{Cos[x], -Sin[x], tx}, {Sin[x], Cos[x], ty}, {0, 0, 1}}
      let matrix = Expr::List(
        vec![
          Expr::List(vec![cos_t.clone(), neg_sin_t, tx].into()),
          Expr::List(vec![sin_t, cos_t, ty].into()),
          Expr::List(
            vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(1)].into(),
          ),
        ]
        .into(),
      );
      // Evaluate the matrix to simplify trig functions (e.g. Cos[Pi/4] → 1/Sqrt[2])
      let evaluated =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "TransformationFunction".to_string(),
          args: vec![matrix].into(),
        });
      return Some(evaluated);
    }
    // AffineTransform[m] or AffineTransform[{m, v}] →
    //   TransformationFunction[ (n+1)x(n+1) augmented matrix ]
    // where the top-left nxn block is m, the first n entries of the last
    // column are the translation vector v (default zero), and the last row
    // is {0, ..., 0, 1}. Applied to a point p it computes m.p + v.
    "AffineTransform" if args.len() == 1 => {
      // Helper: is `e` a matrix (list of lists)?
      fn is_matrix(e: &Expr) -> bool {
        matches!(e, Expr::List(rows)
          if !rows.is_empty() && rows.iter().all(|r| matches!(r, Expr::List(_))))
      }
      // Helper: is `e` a flat vector (list of non-lists)?
      fn is_vector(e: &Expr) -> bool {
        matches!(e, Expr::List(items)
          if items.iter().all(|i| !matches!(i, Expr::List(_))))
      }

      // Determine the matrix m and optional translation vector v.
      let (m_rows, v_opt): (&[Expr], Option<&[Expr]>) = match &args[0] {
        // {m, v} form: 2-element list, first a matrix, second a vector.
        Expr::List(outer)
          if outer.len() == 2
            && is_matrix(&outer[0])
            && is_vector(&outer[1]) =>
        {
          if let (Expr::List(m), Expr::List(v)) = (&outer[0], &outer[1]) {
            (m.as_ref(), Some(v.as_ref()))
          } else {
            unreachable!()
          }
        }
        // Plain matrix m form.
        Expr::List(m) if is_matrix(&args[0]) => (m.as_ref(), None),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "AffineTransform".to_string(),
            args: args.to_vec().into(),
          }));
        }
      };

      let n = m_rows.len();
      // Validate that m is square (n x n) and v (if present) has length n.
      let m_ok = m_rows
        .iter()
        .all(|r| matches!(r, Expr::List(c) if c.len() == n));
      let v_ok = v_opt.map(|v| v.len() == n).unwrap_or(true);
      if !m_ok || !v_ok {
        return Some(Ok(Expr::FunctionCall {
          name: "AffineTransform".to_string(),
          args: args.to_vec().into(),
        }));
      }

      // Build the (n+1)x(n+1) augmented matrix.
      let mut rows = Vec::with_capacity(n + 1);
      for i in 0..n {
        let mut row = match &m_rows[i] {
          Expr::List(c) => c.to_vec(),
          _ => unreachable!(),
        };
        // Append the translation entry for this row (default 0).
        let t = v_opt.map(|v| v[i].clone()).unwrap_or(Expr::Integer(0));
        row.push(t);
        rows.push(Expr::List(row.into()));
      }
      let mut last_row = vec![Expr::Integer(0); n + 1];
      last_row[n] = Expr::Integer(1);
      rows.push(Expr::List(last_row.into()));
      let matrix = Expr::List(rows.into());
      let evaluated =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "TransformationFunction".to_string(),
          args: vec![matrix].into(),
        });
      return Some(evaluated);
    }
    // LinearFractionalTransform[{m, v, w, b}] → the projective transform
    //   p |-> (m.p + v) / (w.p + b),
    // represented by the (n+1)x(n+1) augmented matrix
    //   {{ m , v },
    //    { w , b }}
    // whose last row is the homogeneous denominator (so application divides by
    // w.p + b). LinearFractionalTransform[m] uses a full homogeneous matrix m.
    "LinearFractionalTransform" if args.len() == 1 => {
      fn is_matrix(e: &Expr) -> bool {
        matches!(e, Expr::List(rows)
          if !rows.is_empty() && rows.iter().all(|r| matches!(r, Expr::List(_))))
      }
      fn is_vector(e: &Expr) -> bool {
        matches!(e, Expr::List(items)
          if items.iter().all(|i| !matches!(i, Expr::List(_))))
      }
      let unevaluated = || {
        Some(Ok(Expr::FunctionCall {
          name: "LinearFractionalTransform".to_string(),
          args: args.to_vec().into(),
        }))
      };
      match &args[0] {
        // {m, v, w, b} form.
        Expr::List(parts)
          if parts.len() == 4
            && is_matrix(&parts[0])
            && is_vector(&parts[1])
            && is_vector(&parts[2])
            && !matches!(&parts[3], Expr::List(_)) =>
        {
          let (Expr::List(m), Expr::List(v), Expr::List(w)) =
            (&parts[0], &parts[1], &parts[2])
          else {
            unreachable!()
          };
          let b = &parts[3];
          let n = m.len();
          // Validate dimensions: m is n x n, v and w have length n.
          let dims_ok =
            m.iter().all(|r| matches!(r, Expr::List(c) if c.len() == n))
              && v.len() == n
              && w.len() == n;
          if !dims_ok {
            return unevaluated();
          }
          let mut rows = Vec::with_capacity(n + 1);
          for i in 0..n {
            let Expr::List(c) = &m[i] else { unreachable!() };
            let mut row = c.to_vec();
            row.push(v[i].clone());
            rows.push(Expr::List(row.into()));
          }
          let mut last_row = w.to_vec();
          last_row.push(b.clone());
          rows.push(Expr::List(last_row.into()));
          let matrix = Expr::List(rows.into());
          return Some(crate::evaluator::evaluate_expr_to_expr(
            &Expr::FunctionCall {
              name: "TransformationFunction".to_string(),
              args: vec![matrix].into(),
            },
          ));
        }
        // Plain (n+1)x(n+1) homogeneous matrix form.
        Expr::List(_) if is_matrix(&args[0]) => {
          return Some(crate::evaluator::evaluate_expr_to_expr(
            &Expr::FunctionCall {
              name: "TransformationFunction".to_string(),
              args: vec![args[0].clone()].into(),
            },
          ));
        }
        _ => return unevaluated(),
      }
    }
    // ScalingTransform[{s1, s2, ...}] or ScalingTransform[{s1, s2, ...}, {c1, c2, ...}]
    "ScalingTransform" if args.len() == 1 || args.len() == 2 => {
      if let Expr::List(scales) = &args[0] {
        let n = scales.len();
        let center = if args.len() == 2 {
          if let Expr::List(c) = &args[1] {
            if c.len() == n { Some(c) } else { None }
          } else {
            None
          }
        } else {
          None
        };

        // Build (n+1)x(n+1) homogeneous scaling matrix
        let mut rows = Vec::with_capacity(n + 1);
        for i in 0..n {
          let mut row = vec![Expr::Integer(0); n + 1];
          row[i] = scales[i].clone();
          // Translation column: ci - si*ci = ci*(1 - si)
          if let Some(c) = center {
            let translation = Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Minus,
              left: Box::new(c[i].clone()),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(scales[i].clone()),
                right: Box::new(c[i].clone()),
              }),
            };
            row[n] = translation;
          }
          rows.push(Expr::List(row.into()));
        }
        let mut last_row = vec![Expr::Integer(0); n + 1];
        last_row[n] = Expr::Integer(1);
        rows.push(Expr::List(last_row.into()));
        let matrix = Expr::List(rows.into());
        let evaluated =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "TransformationFunction".to_string(),
            args: vec![matrix].into(),
          });
        return Some(evaluated);
      }
      return Some(Ok(Expr::FunctionCall {
        name: "ScalingTransform".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // ReflectionTransform[v] → reflection in the hyperplane through the origin
    // perpendicular to v. The linear part is M = I - 2 (v⊗v)/(v·v), embedded in
    // an (n+1)×(n+1) homogeneous matrix. ReflectionTransform[v, pt] reflects in
    // the hyperplane through pt instead, giving the translation pt - M·pt.
    "ReflectionTransform" if args.len() == 1 || args.len() == 2 => {
      let unevaluated = || {
        Some(Ok(Expr::FunctionCall {
          name: "ReflectionTransform".to_string(),
          args: args.to_vec().into(),
        }))
      };
      let Expr::List(v) = &args[0] else {
        return unevaluated();
      };
      let n = v.len();
      let center = if args.len() == 2 {
        match &args[1] {
          Expr::List(c) if c.len() == n => Some(c),
          _ => return unevaluated(),
        }
      } else {
        None
      };
      if n == 0 {
        return unevaluated();
      }
      let power = |b: Expr, e: i128| Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b, Expr::Integer(e)].into(),
      };
      let times = |factors: Vec<Expr>| Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors.into(),
      };
      let plus = |terms: Vec<Expr>| Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      };
      // v·v
      let vv = plus(v.iter().map(|vi| power(vi.clone(), 2)).collect());
      // Linear part M[i][j] = delta_ij - 2 v_i v_j / (v·v).
      let m_entry = |i: usize, j: usize| {
        let off = times(vec![
          Expr::Integer(-2),
          v[i].clone(),
          v[j].clone(),
          power(vv.clone(), -1),
        ]);
        if i == j {
          plus(vec![Expr::Integer(1), off])
        } else {
          off
        }
      };
      let mut rows = Vec::with_capacity(n + 1);
      for i in 0..n {
        let mut row: Vec<Expr> = (0..n).map(|j| m_entry(i, j)).collect();
        // Translation column: pt_i - sum_j M[i][j] pt_j (0 when no center).
        let translation = match center {
          Some(c) => {
            let mut terms = vec![c[i].clone()];
            for (j, cj) in c.iter().enumerate() {
              terms.push(times(vec![
                Expr::Integer(-1),
                m_entry(i, j),
                cj.clone(),
              ]));
            }
            plus(terms)
          }
          None => Expr::Integer(0),
        };
        row.push(translation);
        rows.push(Expr::List(row.into()));
      }
      let mut last_row = vec![Expr::Integer(0); n + 1];
      last_row[n] = Expr::Integer(1);
      rows.push(Expr::List(last_row.into()));
      let matrix = Expr::List(rows.into());
      return Some(crate::evaluator::evaluate_expr_to_expr(
        &Expr::FunctionCall {
          name: "TransformationFunction".to_string(),
          args: vec![matrix].into(),
        },
      ));
    }
    // ShearingTransform[phi, e, n] → TransformationFunction[ homogeneous shear ]
    // A point x is mapped to x + Tan[phi] (nhat·x) ep, where nhat = n/Norm[n]
    // is the unit normal and ep is the unit vector along the component of the
    // direction e perpendicular to n (i.e. e with its n-component removed,
    // then normalized). The shear matrix is therefore I + Tan[phi] (ep ⊗ nhat),
    // augmented to an (d+1)x(d+1) homogeneous matrix.
    "ShearingTransform" if args.len() == 3 => {
      if let (Expr::List(e), Expr::List(n)) = (&args[1], &args[2])
        && e.len() == n.len()
        && !e.is_empty()
      {
        let phi = &args[0];
        let d = e.len();
        // Norm[v] = Sqrt[Plus @@ (v^2)]
        let norm = |v: &[Expr]| -> Expr {
          let squares: Vec<Expr> = v
            .iter()
            .map(|c| Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![c.clone(), Expr::Integer(2)].into(),
            })
            .collect();
          Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::FunctionCall {
              name: "Plus".to_string(),
              args: squares.into(),
            }]
            .into(),
          }
        };
        let times = |factors: Vec<Expr>| Expr::FunctionCall {
          name: "Times".to_string(),
          args: factors.into(),
        };
        let recip = |x: Expr| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![x, Expr::Integer(-1)].into(),
        };
        // nhat = n / Norm[n]
        let norm_n = norm(n);
        let nhat: Vec<Expr> = n
          .iter()
          .map(|c| times(vec![c.clone(), recip(norm_n.clone())]))
          .collect();
        // e·nhat
        let e_dot_nhat = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: e
            .iter()
            .zip(nhat.iter())
            .map(|(ei, ni)| times(vec![ei.clone(), ni.clone()]))
            .collect::<Vec<_>>()
            .into(),
        };
        // eperp = e - (e·nhat) nhat ; ep = eperp / Norm[eperp]
        let eperp: Vec<Expr> = e
          .iter()
          .zip(nhat.iter())
          .map(|(ei, ni)| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              ei.clone(),
              times(vec![Expr::Integer(-1), e_dot_nhat.clone(), ni.clone()]),
            ]
            .into(),
          })
          .collect();
        let norm_eperp = norm(&eperp);
        let ep: Vec<Expr> = eperp
          .iter()
          .map(|c| times(vec![c.clone(), recip(norm_eperp.clone())]))
          .collect();
        let tan_phi = Expr::FunctionCall {
          name: "Tan".to_string(),
          args: vec![phi.clone()].into(),
        };
        // Build (d+1)x(d+1) homogeneous matrix.
        let mut rows = Vec::with_capacity(d + 1);
        for i in 0..d {
          let mut row = Vec::with_capacity(d + 1);
          for j in 0..d {
            // off[i][j] = Tan[phi] * ep[i] * nhat[j]
            let off =
              times(vec![tan_phi.clone(), ep[i].clone(), nhat[j].clone()]);
            let entry = if i == j {
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![Expr::Integer(1), off].into(),
              }
            } else {
              off
            };
            // Simplify so that e.g. 1 + (-1/2) collapses to 1/2 (matching
            // Wolfram), while symbolic forms like Sqrt[3/5] are preserved.
            row.push(Expr::FunctionCall {
              name: "Simplify".to_string(),
              args: vec![entry].into(),
            });
          }
          row.push(Expr::Integer(0)); // zero translation column
          rows.push(Expr::List(row.into()));
        }
        let mut last_row = vec![Expr::Integer(0); d + 1];
        last_row[d] = Expr::Integer(1);
        rows.push(Expr::List(last_row.into()));
        let mut matrix = Expr::List(rows.into());
        // If any input is inexact (contains a Real), the whole matrix is
        // numeric in Wolfram — promote every entry to machine precision.
        fn is_inexact(e: &Expr) -> bool {
          match e {
            Expr::Real(_) | Expr::BigFloat(_, _) => true,
            Expr::List(items) => items.iter().any(is_inexact),
            Expr::FunctionCall { args, .. } => args.iter().any(is_inexact),
            _ => false,
          }
        }
        if args.iter().any(is_inexact) {
          matrix = Expr::FunctionCall {
            name: "N".to_string(),
            args: vec![matrix].into(),
          };
        }
        let evaluated =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "TransformationFunction".to_string(),
            args: vec![matrix].into(),
          });
        return Some(evaluated);
      }
      return Some(Ok(Expr::FunctionCall {
        name: "ShearingTransform".to_string(),
        args: args.to_vec().into(),
      }));
    }
    "Adjugate" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == n => Some(cols.to_vec()),
            _ => None,
          })
          .collect();
        if matrix.len() == n && n > 0 {
          let mut result = Vec::with_capacity(n);
          for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
              let mut sub = Vec::with_capacity(n - 1);
              for r in 0..n {
                if r == j {
                  continue;
                }
                let mut sub_row = Vec::with_capacity(n - 1);
                for c in 0..n {
                  if c == i {
                    continue;
                  }
                  sub_row.push(matrix[r][c].clone());
                }
                sub.push(Expr::List(sub_row.into()));
              }
              let minor_matrix = Expr::List(sub.into());
              let det =
                crate::functions::linear_algebra_ast::det_ast(&[minor_matrix]);
              match det {
                Ok(d) => {
                  let sign = if (i + j) % 2 == 0 { 1 } else { -1 };
                  let cofactor = evaluate_expr_to_expr(&Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Times,
                    left: Box::new(Expr::Integer(sign)),
                    right: Box::new(d),
                  })
                  .unwrap_or(Expr::Integer(0));
                  row.push(cofactor);
                }
                Err(e) => return Some(Err(e)),
              }
            }
            result.push(Expr::List(row.into()));
          }
          return Some(Ok(Expr::List(result.into())));
        }
      }
    }
    "QRDecomposition" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::qr_decomposition_ast(
        args,
      ));
    }
    "CholeskyDecomposition" if args.len() == 1 => {
      return Some(
        crate::functions::linear_algebra_ast::cholesky_decomposition_ast(args),
      );
    }
    "JordanDecomposition" if args.len() == 1 => {
      return Some(
        crate::functions::linear_algebra_ast::jordan_decomposition_ast(args),
      );
    }
    "SingularValueList" if args.len() == 1 || args.len() == 2 => {
      return Some(
        crate::functions::linear_algebra_ast::singular_value_list_ast(args),
      );
    }
    "SingularValueDecomposition" if args.len() == 1 => {
      return Some(
        crate::functions::linear_algebra_ast::singular_value_decomposition_ast(
          args,
        ),
      );
    }
    "DiagonalizableMatrixQ" if args.len() == 1 => {
      // A matrix is diagonalizable if eigenvalues have correct multiplicities
      // For numeric matrices: check if eigenvalues list has n elements (counting multiplicity)
      // and that the matrix has n linearly independent eigenvectors.
      // Simplified approach: compute eigenvalues via characteristic polynomial,
      // then check if algebraic multiplicity == geometric multiplicity for each.
      // For 2x2: diagonalizable iff eigenvalues are distinct OR matrix is already diagonal.
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        // Check it's a square matrix
        for row in rows {
          if let Expr::List(cols) = row {
            if cols.len() != n {
              return Some(Ok(Expr::Identifier("False".to_string())));
            }
          } else {
            return None;
          }
        }
        // Compute eigenvalues
        if let Ok(Expr::List(ref eigenvals)) =
          crate::functions::linear_algebra_ast::eigenvalues_ast(args)
        {
          // If we got n eigenvalues, check if all distinct
          if eigenvals.len() == n {
            // Check for repeated eigenvalues
            let mut has_repeated = false;
            for i in 0..eigenvals.len() {
              for j in (i + 1)..eigenvals.len() {
                let ei = expr_to_string(&eigenvals[i]);
                let ej = expr_to_string(&eigenvals[j]);
                if ei == ej {
                  has_repeated = true;
                  break;
                }
              }
              if has_repeated {
                break;
              }
            }
            if !has_repeated {
              // All eigenvalues distinct => diagonalizable
              return Some(Ok(Expr::Identifier("True".to_string())));
            }
            // Repeated eigenvalues: check if matrix is already diagonal
            let mut is_diagonal = true;
            for (i, row) in rows.iter().enumerate() {
              if let Expr::List(cols) = row {
                for (j, val) in cols.iter().enumerate() {
                  if i != j && !is_zero_expr(val) {
                    is_diagonal = false;
                    break;
                  }
                }
              }
              if !is_diagonal {
                break;
              }
            }
            return Some(Ok(Expr::Identifier(
              if is_diagonal { "True" } else { "False" }.to_string(),
            )));
          }
        }
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
    }
    "PositiveSemidefiniteMatrixQ" if args.len() == 1 => {
      // Non-square arguments (scalar, vector, ragged) are not positive
      // semidefinite — answer False rather than staying unevaluated.
      if !is_square_matrix_expr(&args[0]) {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      // Check if all eigenvalues are non-negative
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        for row in rows {
          if let Expr::List(cols) = row {
            if cols.len() != n {
              return Some(Ok(Expr::Identifier("False".to_string())));
            }
          } else {
            return None;
          }
        }
        if let Ok(Expr::List(ref eigenvals)) =
          crate::functions::linear_algebra_ast::eigenvalues_ast(args)
        {
          for ev in eigenvals {
            // Evaluate the eigenvalue to check if it's non-negative
            let evaluated = evaluate_expr_to_expr(ev).unwrap_or(ev.clone());
            match &evaluated {
              Expr::Integer(v) => {
                if *v < 0 {
                  return Some(Ok(Expr::Identifier("False".to_string())));
                }
              }
              Expr::Real(v) => {
                if *v < 0.0 {
                  return Some(Ok(Expr::Identifier("False".to_string())));
                }
              }
              Expr::FunctionCall {
                name: fname,
                args: fargs,
              } if fname == "Rational" && fargs.len() == 2 => {
                if let Expr::Integer(n) = &fargs[0]
                  && *n < 0
                {
                  return Some(Ok(Expr::Identifier("False".to_string())));
                }
              }
              _ => {
                // Try numeric evaluation
                let nval = evaluate_expr_to_expr(&Expr::FunctionCall {
                  name: "N".to_string(),
                  args: vec![evaluated.clone()].into(),
                });
                if let Ok(Expr::Real(v)) = nval
                  && v < 0.0
                {
                  return Some(Ok(Expr::Identifier("False".to_string())));
                }
              }
            }
          }
          return Some(Ok(Expr::Identifier("True".to_string())));
        }
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
    }
    "NegativeDefiniteMatrixQ" if args.len() == 1 => {
      // Non-square arguments are not negative definite — answer False.
      if !is_square_matrix_expr(&args[0]) {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      // All eigenvalues must be strictly negative
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        for row in rows {
          if let Expr::List(cols) = row {
            if cols.len() != n {
              return Some(Ok(Expr::Identifier("False".to_string())));
            }
          } else {
            return None;
          }
        }
        if let Ok(Expr::List(ref eigenvals)) =
          crate::functions::linear_algebra_ast::eigenvalues_ast(args)
        {
          for ev in eigenvals {
            let evaluated = evaluate_expr_to_expr(ev).unwrap_or(ev.clone());
            let is_neg = match &evaluated {
              Expr::Integer(v) => *v < 0,
              Expr::Real(v) => *v < 0.0,
              Expr::FunctionCall { name, args: fargs }
                if name == "Rational" && fargs.len() == 2 =>
              {
                if let Expr::Integer(n) = &fargs[0] {
                  *n < 0
                } else {
                  false
                }
              }
              _ => {
                let nval = evaluate_expr_to_expr(&Expr::FunctionCall {
                  name: "N".to_string(),
                  args: vec![evaluated.clone()].into(),
                });
                if let Ok(Expr::Real(v)) = nval {
                  v < 0.0
                } else {
                  false
                }
              }
            };
            if !is_neg {
              return Some(Ok(Expr::Identifier("False".to_string())));
            }
          }
          return Some(Ok(Expr::Identifier("True".to_string())));
        }
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
    }
    "NegativeSemidefiniteMatrixQ" if args.len() == 1 => {
      // Non-square arguments are not negative semidefinite — answer False.
      if !is_square_matrix_expr(&args[0]) {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      // All eigenvalues must be <= 0
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        for row in rows {
          if let Expr::List(cols) = row {
            if cols.len() != n {
              return Some(Ok(Expr::Identifier("False".to_string())));
            }
          } else {
            return None;
          }
        }
        if let Ok(Expr::List(ref eigenvals)) =
          crate::functions::linear_algebra_ast::eigenvalues_ast(args)
        {
          for ev in eigenvals {
            let evaluated = evaluate_expr_to_expr(ev).unwrap_or(ev.clone());
            let is_pos = match &evaluated {
              Expr::Integer(v) => *v > 0,
              Expr::Real(v) => *v > 0.0,
              Expr::FunctionCall { name, args: fargs }
                if name == "Rational" && fargs.len() == 2 =>
              {
                if let Expr::Integer(n) = &fargs[0] {
                  *n > 0
                } else {
                  false
                }
              }
              _ => {
                let nval = evaluate_expr_to_expr(&Expr::FunctionCall {
                  name: "N".to_string(),
                  args: vec![evaluated.clone()].into(),
                });
                if let Ok(Expr::Real(v)) = nval {
                  v > 0.0
                } else {
                  false
                }
              }
            };
            if is_pos {
              return Some(Ok(Expr::Identifier("False".to_string())));
            }
          }
          return Some(Ok(Expr::Identifier("True".to_string())));
        }
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
    }
    "HermitianMatrixQ" if args.len() == 1 => {
      // Hermitian: M == ConjugateTranspose[M]
      // For real matrices, this is SymmetricMatrixQ
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == n => Some(cols.to_vec()),
            _ => None,
          })
          .collect();
        if matrix.len() == n {
          for i in 0..n {
            for j in i..n {
              // Check M[i][j] == Conjugate[M[j][i]]
              let conj = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Conjugate".to_string(),
                args: vec![matrix[j][i].clone()].into(),
              })
              .unwrap_or(matrix[j][i].clone());
              let diff = evaluate_expr_to_expr(&Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Plus,
                left: Box::new(matrix[i][j].clone()),
                right: Box::new(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(Expr::Integer(-1)),
                  right: Box::new(conj),
                }),
              })
              .unwrap_or(Expr::Integer(1));
              if !is_zero_expr(&diff) {
                return Some(Ok(Expr::Identifier("False".to_string())));
              }
            }
          }
          return Some(Ok(Expr::Identifier("True".to_string())));
        }
      }
      // Not a square matrix → False (a predicate, like SymmetricMatrixQ).
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    "AntihermitianMatrixQ" if args.len() == 1 => {
      // Antihermitian: M == -ConjugateTranspose[M]
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == n => Some(cols.to_vec()),
            _ => None,
          })
          .collect();
        if matrix.len() == n {
          for i in 0..n {
            for j in i..n {
              // Check M[i][j] == -Conjugate[M[j][i]]
              let conj = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Conjugate".to_string(),
                args: vec![matrix[j][i].clone()].into(),
              })
              .unwrap_or(matrix[j][i].clone());
              let sum = evaluate_expr_to_expr(&Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Plus,
                left: Box::new(matrix[i][j].clone()),
                right: Box::new(conj),
              })
              .unwrap_or(Expr::Integer(1));
              if !is_zero_expr(&sum) {
                return Some(Ok(Expr::Identifier("False".to_string())));
              }
            }
          }
          return Some(Ok(Expr::Identifier("True".to_string())));
        }
      }
      // Not a square matrix → False (a predicate, like SymmetricMatrixQ).
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    "NormalMatrixQ" if args.len() == 1 => {
      // Normal: M.M^H == M^H.M where M^H is conjugate transpose
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == n => Some(cols.to_vec()),
            _ => None,
          })
          .collect();
        if matrix.len() == n {
          // Build conjugate transpose
          let mut ct = vec![vec![Expr::Integer(0); n]; n];
          for i in 0..n {
            for j in 0..n {
              ct[i][j] = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Conjugate".to_string(),
                args: vec![matrix[j][i].clone()].into(),
              })
              .unwrap_or(matrix[j][i].clone());
            }
          }
          // Compute M.M^H and M^H.M
          let ct_list = Expr::List(
            ct.iter().map(|r| Expr::List(r.clone().into())).collect(),
          );
          let m_list = args[0].clone();
          let mmh = crate::functions::linear_algebra_ast::dot_ast(&[
            m_list.clone(),
            ct_list.clone(),
          ]);
          let mhm =
            crate::functions::linear_algebra_ast::dot_ast(&[ct_list, m_list]);
          if let (Ok(a), Ok(b)) = (mmh, mhm) {
            let a_str = expr_to_string(&a);
            let b_str = expr_to_string(&b);
            return Some(Ok(Expr::Identifier(
              if a_str == b_str { "True" } else { "False" }.to_string(),
            )));
          }
        }
      }
      // Not a square matrix → False (a predicate, like SymmetricMatrixQ).
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    // DiagonalMatrixQ[m] — True if m is diagonal (nonzeros only on the main
    // diagonal). DiagonalMatrixQ[m, k] allows nonzeros only on the k-th
    // diagonal (j - i == k; k > 0 super-, k < 0 sub-diagonal). Works on
    // rectangular matrices, not just square ones.
    "DiagonalMatrixQ" if args.len() == 1 || args.len() == 2 => {
      let k: i64 = if args.len() == 2 {
        match &args[1] {
          Expr::Integer(n) => *n as i64,
          // A non-integer band specification is left unevaluated.
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "DiagonalMatrixQ".to_string(),
              args: args.to_vec().into(),
            }));
          }
        }
      } else {
        0
      };
      if let Expr::List(rows) = &args[0] {
        let mut is_diag = true;
        let mut ncols: Option<usize> = None;
        'diag: for (i, r) in rows.iter().enumerate() {
          let Expr::List(row) = r else {
            is_diag = false;
            break;
          };
          // Require a rectangular matrix (consistent row length).
          match ncols {
            None => ncols = Some(row.len()),
            Some(c) if c != row.len() => {
              is_diag = false;
              break;
            }
            _ => {}
          }
          for (j, entry) in row.iter().enumerate() {
            if (j as i64 - i as i64) != k && !matches!(entry, Expr::Integer(0))
            {
              let evaluated =
                evaluate_expr_to_expr(entry).unwrap_or(entry.clone());
              if !matches!(evaluated, Expr::Integer(0)) {
                is_diag = false;
                break 'diag;
              }
            }
          }
        }
        return Some(Ok(Expr::Identifier(
          if is_diag { "True" } else { "False" }.to_string(),
        )));
      }
      // A non-list argument (scalar, symbol, …) is not a matrix → False.
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    // UpperTriangularMatrixQ[m] — True if m is upper triangular
    "UpperTriangularMatrixQ" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let mut is_upper = true;
        'upper: for i in 0..n {
          if let Expr::List(row) = &rows[i] {
            for j in 0..i.min(row.len()) {
              if !matches!(&row[j], Expr::Integer(0)) {
                let evaluated =
                  evaluate_expr_to_expr(&row[j]).unwrap_or(row[j].clone());
                if !matches!(evaluated, Expr::Integer(0)) {
                  is_upper = false;
                  break 'upper;
                }
              }
            }
          } else {
            is_upper = false;
            break;
          }
        }
        return Some(Ok(Expr::Identifier(
          if is_upper { "True" } else { "False" }.to_string(),
        )));
      }
      // A non-list argument (scalar, symbol, …) is not a matrix → False.
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    // LowerTriangularMatrixQ[m] — True if m is lower triangular
    "LowerTriangularMatrixQ" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let mut is_lower = true;
        'lower: for i in 0..n {
          if let Expr::List(row) = &rows[i] {
            for j in (i + 1)..row.len() {
              if !matches!(&row[j], Expr::Integer(0)) {
                let evaluated =
                  evaluate_expr_to_expr(&row[j]).unwrap_or(row[j].clone());
                if !matches!(evaluated, Expr::Integer(0)) {
                  is_lower = false;
                  break 'lower;
                }
              }
            }
          } else {
            is_lower = false;
            break;
          }
        }
        return Some(Ok(Expr::Identifier(
          if is_lower { "True" } else { "False" }.to_string(),
        )));
      }
      // A non-list argument (scalar, symbol, …) is not a matrix → False.
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    // AntisymmetricMatrixQ[m] — True if m is antisymmetric (m[i][j] == -m[j][i])
    "AntisymmetricMatrixQ" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let mut is_antisymmetric = true;
        'outer: for i in 0..n {
          if let Expr::List(row_i) = &rows[i] {
            if row_i.len() != n {
              is_antisymmetric = false;
              break;
            }
            for j in 0..n {
              if let Expr::List(row_j) = &rows[j] {
                // Check m[i][j] + m[j][i] == 0
                let sum = Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![row_i[j].clone(), row_j[i].clone()].into(),
                };
                let evaluated = evaluate_expr_to_expr(&sum).unwrap_or(sum);
                if !matches!(evaluated, Expr::Integer(0)) {
                  is_antisymmetric = false;
                  break 'outer;
                }
              } else {
                is_antisymmetric = false;
                break 'outer;
              }
            }
          } else {
            is_antisymmetric = false;
            break;
          }
        }
        return Some(Ok(Expr::Identifier(
          if is_antisymmetric { "True" } else { "False" }.to_string(),
        )));
      }
    }
    // HankelMatrix[{c1,...,cn}] — Hankel matrix where entry (i,j) = c[i+j-1]
    // HankelMatrix[n] — n×n integer matrix with column/row 1..n.
    "HankelMatrix" if !args.is_empty() && args.len() <= 2 => {
      if args.len() == 1 {
        if let Expr::List(col) = &args[0] {
          let n = col.len();
          let mut rows = Vec::with_capacity(n);
          for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
              let idx = i + j;
              if idx < n {
                row.push(col[idx].clone());
              } else {
                row.push(Expr::Integer(0));
              }
            }
            rows.push(Expr::List(row.into()));
          }
          return Some(Ok(Expr::List(rows.into())));
        }
        // HankelMatrix[n] — entry (i, j) = i + j - 1 when in bounds, else 0.
        if let Some(n) = expr_to_i128(&args[0])
          && n >= 0
        {
          let n = n as usize;
          let mut rows = Vec::with_capacity(n);
          for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
              let idx = i + j;
              if idx < n {
                row.push(Expr::Integer((idx + 1) as i128));
              } else {
                row.push(Expr::Integer(0));
              }
            }
            rows.push(Expr::List(row.into()));
          }
          return Some(Ok(Expr::List(rows.into())));
        }
      } else if let (Expr::List(col), Expr::List(row_data)) =
        (&args[0], &args[1])
      {
        // HankelMatrix[col, row] — first column is col, last row is row_data
        let n = col.len();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
          let mut row = Vec::with_capacity(n);
          for j in 0..n {
            if i + j < n {
              row.push(col[i + j].clone());
            } else {
              // Use row_data for the remainder
              let rd_idx = i + j - n + 1;
              if rd_idx < row_data.len() {
                row.push(row_data[rd_idx].clone());
              } else {
                row.push(Expr::Integer(0));
              }
            }
          }
          rows.push(Expr::List(row.into()));
        }
        return Some(Ok(Expr::List(rows.into())));
      }
    }
    // HadamardMatrix[n] — Hadamard matrix of size n (must be power of 2)
    "HadamardMatrix" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0]) {
        let n = n as usize;
        if n > 0 && (n & (n - 1)) == 0 {
          // n is a power of 2: Sylvester construction, normalized by 1/Sqrt[n]
          let mat = hadamard_sylvester(n);
          // Evaluate 1/Sqrt[n] first
          let sqrt_n_expr = make_sqrt(Expr::Integer(n as i128));
          let sqrt_n =
            evaluate_expr_to_expr(&sqrt_n_expr).unwrap_or(sqrt_n_expr);
          // Compute 1/Sqrt[n] once (evaluated)
          let inv_sqrt_n_expr = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(sqrt_n.clone()),
          };
          let inv_sqrt_n =
            evaluate_expr_to_expr(&inv_sqrt_n_expr).unwrap_or(inv_sqrt_n_expr);

          let result_rows: Vec<Expr> = mat
            .into_iter()
            .map(|row| {
              Expr::List(
                row
                  .into_iter()
                  .map(|v| {
                    if v == 1 {
                      inv_sqrt_n.clone()
                    } else if v == -1 {
                      // Use Times[-1, inv_sqrt_n] so display matches -(1/Sqrt[n])
                      let entry = Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: vec![Expr::Integer(-1), inv_sqrt_n.clone()]
                          .into(),
                      };
                      evaluate_expr_to_expr(&entry).unwrap_or(entry)
                    } else {
                      // v / Sqrt[n]
                      let entry = Expr::BinaryOp {
                        op: crate::syntax::BinaryOperator::Divide,
                        left: Box::new(Expr::Integer(v)),
                        right: Box::new(sqrt_n.clone()),
                      };
                      evaluate_expr_to_expr(&entry).unwrap_or(entry)
                    }
                  })
                  .collect(),
              )
            })
            .collect();
          return Some(Ok(Expr::List(result_rows.into())));
        }
      }
    }
    // ScalingMatrix[{s1, s2, ...}] — diagonal scaling matrix
    "ScalingMatrix" if args.len() == 1 => {
      if let Expr::List(scales) = &args[0] {
        let n = scales.len();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
          let mut row = vec![Expr::Integer(0); n];
          row[i] = scales[i].clone();
          rows.push(Expr::List(row.into()));
        }
        return Some(Ok(Expr::List(rows.into())));
      }
    }
    // ScalingMatrix[s, v] — scale by factor s along the direction of v, leaving
    // the orthogonal complement fixed: M = I + (s-1)/(v.v) (v outer v).
    "ScalingMatrix" if args.len() == 2 => {
      if let Expr::List(v) = &args[1]
        && !v.is_empty()
      {
        let times = |xs: Vec<Expr>| Expr::FunctionCall {
          name: "Times".to_string(),
          args: xs.into(),
        };
        let plus = |xs: Vec<Expr>| Expr::FunctionCall {
          name: "Plus".to_string(),
          args: xs.into(),
        };
        let sq = |a: Expr| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![a, Expr::Integer(2)].into(),
        };
        let vdotv = plus(v.iter().cloned().map(sq).collect());
        let s_minus_1 = plus(vec![args[0].clone(), Expr::Integer(-1)]);
        let inv = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![vdotv, Expr::Integer(-1)].into(),
        };
        let factor = times(vec![s_minus_1, inv]);
        let n = v.len();
        let rows: Vec<Expr> = (0..n)
          .map(|i| {
            Expr::List(
              (0..n)
                .map(|j| {
                  let delta = Expr::Integer(i128::from(i == j));
                  let off =
                    times(vec![factor.clone(), v[i].clone(), v[j].clone()]);
                  plus(vec![delta, off])
                })
                .collect(),
            )
          })
          .collect();
        return Some(evaluate_expr_to_expr(&Expr::List(rows.into())));
      }
    }
    // CrossMatrix[r] / CrossMatrix[{r1, …, rn}] — the n-dimensional "cross"
    // structuring element (the morphology kernel). A scalar r is the 2D cross
    // of radius r in both directions. Each entry is 1 when at most one
    // coordinate differs from the center, else 0. Non-numeric radii emit
    // ::notre and stay unevaluated.
    "CrossMatrix" if args.len() == 1 => {
      let radii: Option<Vec<i128>> = match &args[0] {
        Expr::List(elems) => elems.iter().map(cross_radius).collect(),
        other => cross_radius(other).map(|r| vec![r, r]),
      };
      if let Some(radii) = radii
        && !radii.is_empty()
      {
        return Some(Ok(build_cross_matrix(&radii)));
      }
      crate::emit_message(&format!(
        "CrossMatrix::notre: The first argument {} must be a non-complex number or a list of non-complex numbers.",
        crate::syntax::expr_to_string(&args[0])
      ));
      return Some(Ok(Expr::FunctionCall {
        name: "CrossMatrix".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // FourierMatrix[n] — discrete Fourier transform matrix
    // Entry (j,k) = omega^((j-1)*(k-1)) / sqrt(n), omega = e^(2*pi*i/n)
    // Uses Cos + I*Sin for exact symbolic results
    "FourierMatrix" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0])
        && n > 0
      {
        // wolframscript materialises FourierMatrix for n up to about
        // 600 and switches to a lazy StructuredArray placeholder above
        // that. Woxi enumerates each (j, k) entry symbolically, so the
        // 640k symbolic Cos/Sin calls for n=800 would otherwise time
        // out. Match wolframscript by leaving the call unevaluated for
        // n > 600.
        const FOURIER_MATRIX_MAX_MATERIALIZE: i128 = 600;
        if n > FOURIER_MATRIX_MAX_MATERIALIZE {
          return Some(Ok(Expr::FunctionCall {
            name: "FourierMatrix".to_string(),
            args: args.to_vec().into(),
          }));
        }
        let n = n as usize;
        let inv_sqrt_n = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::Integer(n as i128),
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
            },
          ]
          .into(),
        };
        let mut rows = Vec::with_capacity(n);
        for j in 0..n {
          let mut row = Vec::with_capacity(n);
          for k in 0..n {
            let exp = ((j * k) % n) as i128;
            if exp == 0 {
              // entry = 1/sqrt(n)
              row.push(inv_sqrt_n.clone());
            } else {
              // angle = 2*Pi*exp/n, compute (Cos[angle] + I*Sin[angle]) / Sqrt[n]
              // Simplify the fraction 2*exp/n
              let num = 2 * exp;
              let den = n as i128;
              let g = gcd_i128(num.abs(), den);
              let snum = num / g;
              let sden = den / g;
              let angle = if sden == 1 {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![
                    Expr::Integer(snum),
                    Expr::Identifier("Pi".to_string()),
                  ]
                  .into(),
                }
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![
                    Expr::FunctionCall {
                      name: "Rational".to_string(),
                      args: vec![Expr::Integer(snum), Expr::Integer(sden)]
                        .into(),
                    },
                    Expr::Identifier("Pi".to_string()),
                  ]
                  .into(),
                }
              };
              // Build (Cos[angle] + I*Sin[angle]) / Sqrt[n]
              let cos_part = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![angle.clone()].into(),
              };
              let sin_part = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::Identifier("I".to_string()),
                  Expr::FunctionCall {
                    name: "Sin".to_string(),
                    args: vec![angle].into(),
                  },
                ]
                .into(),
              };
              let omega = Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![cos_part, sin_part].into(),
              };
              let entry = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![omega, inv_sqrt_n.clone()].into(),
              };
              row.push(entry);
            }
          }
          rows.push(Expr::List(row.into()));
        }
        let result = Expr::List(rows.into());
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // Symmetrize[matrix] — symmetrize a square matrix. wolframscript returns
    // a SymmetrizedArray whose StructuredData stores only the upper-triangle
    // entries (positions {i, j} with i ≤ j) of (M + M^T)/2, marked with a
    // `Symmetric[{1, 2}]` tag.
    "Symmetrize" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let mut matrix: Vec<Vec<Expr>> = Vec::new();
        let mut all_ok = true;
        for row in rows {
          if let Expr::List(cols) = row {
            if cols.len() != n {
              all_ok = false;
              break;
            }
            matrix.push(cols.to_vec());
          } else {
            all_ok = false;
            break;
          }
        }
        if all_ok && n > 0 {
          let mut rules: Vec<Expr> = Vec::new();
          for i in 0..n {
            for j in i..n {
              let entry = if i == j {
                matrix[i][j].clone()
              } else {
                let sum = Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![matrix[i][j].clone(), matrix[j][i].clone()].into(),
                };
                let half = Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![
                    Expr::FunctionCall {
                      name: "Rational".to_string(),
                      args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
                    },
                    sum,
                  ]
                  .into(),
                };
                match evaluate_expr_to_expr(&half) {
                  Ok(val) => val,
                  Err(_) => half,
                }
              };
              let pos = Expr::List(
                vec![
                  Expr::Integer((i + 1) as i128),
                  Expr::Integer((j + 1) as i128),
                ]
                .into(),
              );
              rules.push(Expr::FunctionCall {
                name: "Rule".to_string(),
                args: vec![pos, entry].into(),
              });
            }
          }
          let dims = Expr::List(
            vec![Expr::Integer(n as i128), Expr::Integer(n as i128)].into(),
          );
          let symmetric_tag = Expr::FunctionCall {
            name: "Symmetric".to_string(),
            args: vec![Expr::List(
              vec![Expr::Integer(1), Expr::Integer(2)].into(),
            )]
            .into(),
          };
          let inner_list =
            Expr::List(vec![Expr::List(rules.into()), symmetric_tag].into());
          let structured_data = Expr::FunctionCall {
            name: "StructuredArray`StructuredData".to_string(),
            args: vec![dims, inner_list].into(),
          };
          return Some(Ok(Expr::FunctionCall {
            name: "SymmetrizedArray".to_string(),
            args: vec![structured_data].into(),
          }));
        }
      }
    }
    _ => {}
  }
  None
}

/// Parse a CrossMatrix radius: a non-negative integer, or a real/rational
/// numeric value rounded to the nearest non-negative integer. Returns None for
/// anything non-numeric or negative.
fn cross_radius(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(n) if *n >= 0 => Some(*n),
    Expr::Real(r) if *r >= 0.0 => Some(r.round() as i128),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(num), Expr::Integer(den)) = (&args[0], &args[1])
        && *den != 0
      {
        let v = *num as f64 / *den as f64;
        if v >= 0.0 {
          return Some(v.round() as i128);
        }
      }
      None
    }
    _ => None,
  }
}

/// Build the n-dimensional CrossMatrix structuring element for the given radii.
/// Dimension k has length 2*r_k+1 with center index r_k; an entry is 1 when at
/// most one coordinate differs from its center, else 0.
fn build_cross_matrix(radii: &[i128]) -> Expr {
  let centers: Vec<usize> = radii.iter().map(|r| *r as usize).collect();
  let dims: Vec<usize> = radii.iter().map(|r| (2 * r + 1) as usize).collect();
  fn rec(
    dims: &[usize],
    centers: &[usize],
    idx: &mut Vec<usize>,
    depth: usize,
  ) -> Expr {
    if depth == dims.len() {
      let off = idx
        .iter()
        .zip(centers.iter())
        .filter(|(i, c)| i != c)
        .count();
      return Expr::Integer(i128::from(off <= 1));
    }
    let mut row = Vec::with_capacity(dims[depth]);
    for i in 0..dims[depth] {
      idx.push(i);
      row.push(rec(dims, centers, idx, depth + 1));
      idx.pop();
    }
    Expr::List(row.into())
  }
  rec(&dims, &centers, &mut Vec::with_capacity(dims.len()), 0)
}

/// Build the n-dimensional DiamondMatrix (L1-ball) structuring element for the
/// given radii. Dimension k has length 2*r_k+1 with center r_k; a cell is 1
/// when sum_k |idx_k - r_k| / (r_k + 1/2) <= 1. The test is done with integer
/// arithmetic: with A_k = 2*r_k+1 and P = prod_k A_k, the cell is included when
/// sum_k 2*|d_k|*(P/A_k) <= P.
fn build_diamond_matrix(radii: &[i128]) -> Expr {
  let a: Vec<i128> = radii.iter().map(|r| 2 * r + 1).collect();
  let p: i128 = a.iter().product();
  let dims: Vec<usize> = a.iter().map(|&v| v as usize).collect();
  let centers: Vec<i128> = radii.to_vec();
  fn rec(
    dims: &[usize],
    centers: &[i128],
    a: &[i128],
    p: i128,
    idx: &mut Vec<i128>,
    depth: usize,
  ) -> Expr {
    if depth == dims.len() {
      let lhs: i128 = (0..centers.len())
        .map(|k| 2 * (idx[k] - centers[k]).abs() * (p / a[k]))
        .sum();
      return Expr::Integer(i128::from(lhs <= p));
    }
    let mut row = Vec::with_capacity(dims[depth]);
    for i in 0..dims[depth] {
      idx.push(i as i128);
      row.push(rec(dims, centers, a, p, idx, depth + 1));
      idx.pop();
    }
    Expr::List(row.into())
  }
  rec(
    &dims,
    &centers,
    &a,
    p,
    &mut Vec::with_capacity(dims.len()),
    0,
  )
}

/// Build the n-dimensional DiskMatrix (L2-ball) structuring element for the
/// given radii. Dimension k has length 2*r_k+1 with center r_k; a cell is 1
/// when sum_k (|idx_k - r_k| / (r_k + 1/2))^2 <= 1. With A_k = 2*r_k+1 and
/// P = prod_k A_k the integer test is sum_k 4*d_k^2*(P/A_k)^2 <= P^2.
fn build_disk_matrix(radii: &[i128]) -> Expr {
  let a: Vec<i128> = radii.iter().map(|r| 2 * r + 1).collect();
  let p: i128 = a.iter().product();
  let dims: Vec<usize> = a.iter().map(|&v| v as usize).collect();
  let centers: Vec<i128> = radii.to_vec();
  fn rec(
    dims: &[usize],
    centers: &[i128],
    a: &[i128],
    p: i128,
    idx: &mut Vec<i128>,
    depth: usize,
  ) -> Expr {
    if depth == dims.len() {
      let lhs: i128 = (0..centers.len())
        .map(|k| {
          let d = (idx[k] - centers[k]).abs();
          let w = p / a[k];
          4 * d * d * w * w
        })
        .sum();
      return Expr::Integer(i128::from(lhs <= p * p));
    }
    let mut row = Vec::with_capacity(dims[depth]);
    for i in 0..dims[depth] {
      idx.push(i as i128);
      row.push(rec(dims, centers, a, p, idx, depth + 1));
      idx.pop();
    }
    Expr::List(row.into())
  }
  rec(
    &dims,
    &centers,
    &a,
    p,
    &mut Vec::with_capacity(dims.len()),
    0,
  )
}

/// Build an all-ones tensor of the given shape (nested Lists). An empty shape
/// yields the scalar 1.
fn build_ones_tensor(dims: &[usize]) -> Expr {
  match dims {
    [] => Expr::Integer(1),
    [first, rest @ ..] => {
      let inner = build_ones_tensor(rest);
      Expr::List(vec![inner; *first].into())
    }
  }
}

/// Hadamard matrix via Sylvester construction for powers of 2,
/// then reordered by sequency (Walsh ordering) to match Wolfram's HadamardMatrix.
fn hadamard_sylvester(n: usize) -> Vec<Vec<i128>> {
  if n == 1 {
    return vec![vec![1]];
  }
  // Build Sylvester construction
  let half = hadamard_sylvester(n / 2);
  let h = n / 2;
  let mut result = vec![vec![0i128; n]; n];
  for i in 0..h {
    for j in 0..h {
      result[i][j] = half[i][j];
      result[i][j + h] = half[i][j];
      result[i + h][j] = half[i][j];
      result[i + h][j + h] = -half[i][j];
    }
  }
  // Reorder rows by sequency (number of sign changes)
  let mut rows_with_seq: Vec<(usize, Vec<i128>)> = result
    .into_iter()
    .map(|row| {
      let seq = row.windows(2).filter(|w| w[0] != w[1]).count();
      (seq, row)
    })
    .collect();
  rows_with_seq.sort_by_key(|(seq, _)| *seq);
  rows_with_seq.into_iter().map(|(_, row)| row).collect()
}

/// LUDecomposition[m] — compute LU decomposition with partial pivoting.
/// Returns {lu_combined, pivots, 0} where lu_combined stores L (below diagonal)
/// and U (on and above diagonal), and pivots is the row permutation (1-indexed).
fn lu_decomposition_ast(mat: &Expr) -> Result<Expr, InterpreterError> {
  let rows = match mat {
    Expr::List(rows) => rows,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LUDecomposition".to_string(),
        args: vec![mat.clone()].into(),
      });
    }
  };

  let n = rows.len();
  if n == 0 {
    return Ok(Expr::List(
      vec![
        Expr::List(vec![].into()),
        Expr::List(vec![].into()),
        Expr::Integer(0),
      ]
      .into(),
    ));
  }

  // Extract matrix elements
  let mut matrix: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for row in rows {
    if let Expr::List(cols) = row {
      if cols.len() != n {
        return Err(InterpreterError::EvaluationError(
          "LUDecomposition: matrix must be square".into(),
        ));
      }
      matrix.push(cols.to_vec());
    } else {
      return Ok(Expr::FunctionCall {
        name: "LUDecomposition".to_string(),
        args: vec![mat.clone()].into(),
      });
    }
  }

  // Partial pivoting LU decomposition
  let mut pivots: Vec<usize> = (0..n).collect();

  for k in 0..n {
    // Find pivot (first non-zero in column k, from row k downward)
    // For symbolic: just check if it's Integer(0)
    let mut pivot_row = k;
    for i in k..n {
      if !is_zero_expr(&matrix[i][k]) {
        pivot_row = i;
        break;
      }
    }

    // Swap rows
    if pivot_row != k {
      matrix.swap(k, pivot_row);
      pivots.swap(k, pivot_row);
    }

    // Compute L and U entries
    let pivot_val = matrix[k][k].clone();
    if is_zero_expr(&pivot_val) {
      continue; // Singular matrix, skip
    }

    for i in (k + 1)..n {
      // L[i][k] = A[i][k] / A[k][k]
      let l_ik = evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          matrix[i][k].clone(),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![pivot_val.clone(), Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      })
      .unwrap_or(matrix[i][k].clone());

      // Update row i: A[i][j] -= L[i][k] * A[k][j] for j > k
      for j in (k + 1)..n {
        let product = evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![l_ik.clone(), matrix[k][j].clone()].into(),
        })
        .unwrap_or(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![l_ik.clone(), matrix[k][j].clone()].into(),
        });

        let new_val = evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            matrix[i][j].clone(),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), product].into(),
            },
          ]
          .into(),
        })
        .unwrap_or(matrix[i][j].clone());

        matrix[i][j] = new_val;
      }

      // Store L factor in place of A[i][k]
      matrix[i][k] = l_ik;
    }
  }

  // Build result
  let lu_matrix =
    Expr::List(matrix.into_iter().map(|v| Expr::List(v.into())).collect());

  let pivot_list = Expr::List(
    pivots
      .into_iter()
      .map(|p| Expr::Integer((p + 1) as i128)) // 1-indexed
      .collect(),
  );

  Ok(Expr::List(
    vec![lu_matrix, pivot_list, Expr::Integer(0)].into(),
  ))
}

fn is_zero_expr(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(0) => true,
    Expr::Real(v) => *v == 0.0,
    _ => false,
  }
}

/// True iff `expr` is a non-empty square matrix: a list of n rows, each a list
/// of exactly n elements. Used by the definiteness predicates so a scalar,
/// vector, or non-square matrix answers False instead of erroring or staying
/// unevaluated (matching wolframscript).
fn is_square_matrix_expr(expr: &Expr) -> bool {
  if let Expr::List(rows) = expr {
    let n = rows.len();
    n > 0
      && rows
        .iter()
        .all(|r| matches!(r, Expr::List(c) if c.len() == n))
  } else {
    false
  }
}

/// Classification of a KroneckerProduct operand by tensor rank.
enum KronArg<'a> {
  /// Rank-1 tensor (vector of scalars).
  Vector(Vec<&'a Expr>),
  /// Rank-2 tensor (matrix, list of equal-length rows).
  Matrix(Vec<Vec<&'a Expr>>),
}

/// Classify an `Expr` as a KroneckerProduct vector or matrix operand.
/// Returns `None` for anything that is not a vector or rectangular matrix.
fn classify_kron_arg(expr: &Expr) -> Option<KronArg<'_>> {
  let rows = match expr {
    Expr::List(rows) => rows,
    _ => return None,
  };
  if rows.is_empty() {
    return None;
  }
  // Matrix if every element is itself a list; vector if none are.
  if rows.iter().all(|r| matches!(r, Expr::List(_))) {
    let mat: Vec<Vec<&Expr>> = rows
      .iter()
      .map(|r| match r {
        Expr::List(cols) => cols.iter().collect(),
        _ => unreachable!(),
      })
      .collect();
    // Require rectangular shape.
    let n = mat[0].len();
    if mat.iter().any(|r| r.len() != n) {
      return None;
    }
    Some(KronArg::Matrix(mat))
  } else if rows.iter().any(|r| matches!(r, Expr::List(_))) {
    // Ragged (mix of lists and scalars) — not a valid tensor.
    None
  } else {
    Some(KronArg::Vector(rows.iter().collect()))
  }
}

fn kron_times(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  crate::functions::math_ast::times_ast(&[a.clone(), b.clone()])
}

/// KroneckerProduct of two operands of rank 1 or 2, matching wolframscript:
/// - vector⊗vector → m×n outer-product matrix
/// - vector⊗matrix → (m·p)×q matrix
/// - matrix⊗vector → m×(n·q) matrix
/// - matrix⊗matrix → (m·p)×(n·q) block matrix
fn kronecker_product_pair(
  a: &KronArg<'_>,
  b: &KronArg<'_>,
) -> Result<Expr, InterpreterError> {
  let result: Vec<Expr> = match (a, b) {
    (KronArg::Vector(u), KronArg::Vector(v)) => {
      let mut rows = Vec::with_capacity(u.len());
      for &ui in u {
        let mut row = Vec::with_capacity(v.len());
        for &vj in v {
          row.push(kron_times(ui, vj)?);
        }
        rows.push(Expr::List(row.into()));
      }
      rows
    }
    (KronArg::Vector(u), KronArg::Matrix(m)) => {
      // (|u|·rows)×cols : result[i*p+k][l] = u[i] * m[k][l]
      let mut rows = Vec::with_capacity(u.len() * m.len());
      for &ui in u {
        for mk in m {
          let mut row = Vec::with_capacity(mk.len());
          for &mkl in mk {
            row.push(kron_times(ui, mkl)?);
          }
          rows.push(Expr::List(row.into()));
        }
      }
      rows
    }
    (KronArg::Matrix(m), KronArg::Vector(v)) => {
      // rows×(cols·|v|) : result[i][j*q+l] = m[i][j] * v[l]
      let mut rows = Vec::with_capacity(m.len());
      for mi in m {
        let mut row = Vec::with_capacity(mi.len() * v.len());
        for &mij in mi {
          for &vl in v {
            row.push(kron_times(mij, vl)?);
          }
        }
        rows.push(Expr::List(row.into()));
      }
      rows
    }
    (KronArg::Matrix(a), KronArg::Matrix(b)) => {
      let (m, n) = (a.len(), a[0].len());
      let (p, q) = (b.len(), b[0].len());
      let mut rows = Vec::with_capacity(m * p);
      for i in 0..m {
        for k in 0..p {
          let mut row = Vec::with_capacity(n * q);
          for j in 0..n {
            for l in 0..q {
              row.push(kron_times(a[i][j], b[k][l])?);
            }
          }
          rows.push(Expr::List(row.into()));
        }
      }
      rows
    }
  };
  Ok(Expr::List(result.into()))
}

/// KroneckerProduct[a, b, ...] — generalized tensor (Kronecker) product.
/// Supports vectors and matrices and folds left over any number of args.
fn kronecker_product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "KroneckerProduct".to_string(),
    args: args.to_vec().into(),
  };

  // Classify the first operand.
  let mut acc_owned = args[0].clone();
  if classify_kron_arg(&acc_owned).is_none() {
    return Ok(unevaluated());
  }

  for next in &args[1..] {
    let a = match classify_kron_arg(&acc_owned) {
      Some(a) => a,
      None => return Ok(unevaluated()),
    };
    let b = match classify_kron_arg(next) {
      Some(b) => b,
      None => return Ok(unevaluated()),
    };
    acc_owned = kronecker_product_pair(&a, &b)?;
  }

  Ok(acc_owned)
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
  a = a.abs();
  b = b.abs();
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Binary dissimilarity functions for binary (0/1) vectors.
fn binary_dissimilarity_ast(
  name: &str,
  a: &Expr,
  b: &Expr,
) -> Result<Expr, InterpreterError> {
  let (list_a, list_b) = match (a, b) {
    (Expr::List(la), Expr::List(lb)) if la.len() == lb.len() => (la, lb),
    _ => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![a.clone(), b.clone()].into(),
      });
    }
  };

  // Count n11 (both 1), n10 (a=1,b=0), n01 (a=0,b=1), n00 (both 0)
  let mut n11: i128 = 0;
  let mut n10: i128 = 0;
  let mut n01: i128 = 0;
  let mut n00: i128 = 0;

  // Accept Integer (0/1) or Boolean (True/False); other entries keep the
  // call symbolic.
  fn as_bool(e: &Expr) -> Option<bool> {
    match e {
      Expr::Integer(v) => Some(*v != 0),
      Expr::Identifier(s) if s == "True" => Some(true),
      Expr::Identifier(s) if s == "False" => Some(false),
      _ => None,
    }
  }
  for (ai, bi) in list_a.iter().zip(list_b.iter()) {
    let (Some(av), Some(bv)) = (as_bool(ai), as_bool(bi)) else {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![a.clone(), b.clone()].into(),
      });
    };
    match (av, bv) {
      (true, true) => n11 += 1,
      (true, false) => n10 += 1,
      (false, true) => n01 += 1,
      (false, false) => n00 += 1,
    }
  }

  let (num, den) = match name {
    "DiceDissimilarity" => (n10 + n01, 2 * n11 + n10 + n01),
    "JaccardDissimilarity" => (n10 + n01, n11 + n10 + n01),
    "MatchingDissimilarity" => (n10 + n01, n11 + n10 + n01 + n00),
    "RogersTanimotoDissimilarity" => {
      (2 * (n10 + n01), 2 * (n10 + n01) + n11 + n00)
    }
    "RussellRaoDissimilarity" => (n10 + n01 + n00, n11 + n10 + n01 + n00),
    "SokalSneathDissimilarity" => (2 * (n10 + n01), n11 + 2 * (n10 + n01)),
    "YuleDissimilarity" => (2 * n10 * n01, n11 * n00 + n10 * n01),
    _ => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![a.clone(), b.clone()].into(),
      });
    }
  };

  if den == 0 {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  Ok(crate::functions::math_ast::make_rational(num, den))
}

/// `MatrixMinimalPolynomial[A, x]` — the monic polynomial of least degree
/// `m(x)` such that `m(A)` is the zero matrix. Computed via the Krylov method:
/// the powers `I, A, A^2, …` are flattened into column vectors and the first
/// linear dependence among them (found with `NullSpace`) gives the coefficients
/// of the minimal polynomial. Returns `None` (leaving the head unevaluated)
/// when the argument is not a square numeric matrix or no dependence is found.
fn matrix_minimal_polynomial(
  matrix: &Expr,
  x: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let rows = match matrix {
    Expr::List(rows) => rows,
    _ => return None,
  };
  let n = rows.len();
  if n == 0
    || !rows
      .iter()
      .all(|r| matches!(r, Expr::List(c) if c.len() == n))
  {
    return None;
  }

  // Build the powers A^0 = I, A^1, …, A^n.
  let identity = crate::functions::linear_algebra_ast::identity_matrix_ast(&[
    Expr::Integer(n as i128),
  ])
  .ok()?;
  let mut powers: Vec<Expr> = Vec::with_capacity(n + 1);
  powers.push(identity);
  for i in 1..=n {
    let prod = evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Dot".to_string(),
      args: vec![powers[i - 1].clone(), matrix.clone()].into(),
    })
    .ok()?;
    powers.push(prod);
  }

  // Flatten each power into a length-n^2 vector of entries.
  let flat: Vec<Vec<Expr>> = powers
    .iter()
    .map(|p| match p {
      Expr::List(prows) => {
        let mut v = Vec::with_capacity(n * n);
        for pr in prows.iter() {
          match pr {
            Expr::List(pc) => v.extend(pc.iter().cloned()),
            _ => return None,
          }
        }
        Some(v)
      }
      _ => None,
    })
    .collect::<Option<_>>()?;
  let dim = n * n;

  // Find the smallest k where {flat[0], …, flat[k]} are linearly dependent.
  for k in 1..=n {
    // Column-stack the first k+1 flattened powers: M is dim×(k+1).
    let m_rows: Vec<Expr> = (0..dim)
      .map(|r| Expr::List((0..=k).map(|i| flat[i][r].clone()).collect()))
      .collect();
    let m = Expr::List(m_rows.into());
    let ns = crate::functions::linear_algebra_ast::null_space_ast(&[m]).ok()?;
    let Expr::List(basis) = &ns else { return None };
    if let Some(Expr::List(coeffs)) = basis.first()
      && coeffs.len() == k + 1
    {
      // Normalise to monic by dividing through by the leading coefficient.
      let lead = coeffs[k].clone();
      let mut terms: Vec<Expr> = Vec::with_capacity(k + 1);
      for (i, c) in coeffs.iter().enumerate() {
        let coeff = Expr::FunctionCall {
          name: "Divide".to_string(),
          args: vec![c.clone(), lead.clone()].into(),
        };
        let term = if i == 0 {
          coeff
        } else {
          let xpow = if i == 1 {
            x.clone()
          } else {
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![x.clone(), Expr::Integer(i as i128)].into(),
            }
          };
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![coeff, xpow].into(),
          }
        };
        terms.push(term);
      }
      let poly = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      };
      return Some(evaluate_expr_to_expr(&poly));
    }
  }
  None
}

/// `MatrixPower[m, n]` with a symbolic exponent `n` on a matrix that
/// splits — after a row/column permutation — into multiple connected
/// blocks. Returns `Some(result)` only when *every* block has size 1
/// (entry ↦ entry^n) or size 2 (closed form via eigenvalues). Single-
/// block matrices (size ≥ 2) return `None` so the caller falls through
/// to the unevaluated head.
fn matrix_power_block_symbolic(rows: &[Expr], n_expr: &Expr) -> Option<Expr> {
  let size = rows.len();
  // Validate the matrix is square.
  let matrix: Vec<Vec<Expr>> = {
    let mut m = Vec::with_capacity(size);
    for row in rows {
      let cols = if let Expr::List(c) = row {
        c
      } else {
        return None;
      };
      if cols.len() != size {
        return None;
      }
      m.push(cols.to_vec());
    }
    m
  };

  // Build adjacency: index i and j are connected iff M[i][j] != 0 OR
  // M[j][i] != 0. Self-loops (i == j) don't create edges; an isolated
  // index forms a size-1 component.
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); size];
  for i in 0..size {
    for j in 0..size {
      if i == j {
        continue;
      }
      if !matches!(&matrix[i][j], Expr::Integer(0))
        || !matches!(&matrix[j][i], Expr::Integer(0))
      {
        adj[i].push(j);
      }
    }
  }

  // Find connected components by BFS.
  let mut comp_of: Vec<Option<usize>> = vec![None; size];
  let mut components: Vec<Vec<usize>> = Vec::new();
  for start in 0..size {
    if comp_of[start].is_some() {
      continue;
    }
    let cid = components.len();
    let mut comp: Vec<usize> = Vec::new();
    let mut queue: Vec<usize> = vec![start];
    while let Some(u) = queue.pop() {
      if comp_of[u].is_some() {
        continue;
      }
      comp_of[u] = Some(cid);
      comp.push(u);
      for &v in &adj[u] {
        if comp_of[v].is_none() {
          queue.push(v);
        }
      }
    }
    comp.sort_unstable();
    components.push(comp);
  }

  // Single-component matrix of size ≥ 2: leave to the caller (stays
  // unevaluated). A 1×1 single component (size = 1) is also handled
  // here as `M[0][0]^n`.
  if components.len() == 1 && size > 1 {
    return None;
  }

  // Bail out unless every block has size 1 or 2.
  if components.iter().any(|c| c.len() > 2) {
    return None;
  }

  // Build the result matrix, initialised to zero.
  let zero = Expr::Integer(0);
  let mut result: Vec<Vec<Expr>> = vec![vec![zero.clone(); size]; size];

  for comp in &components {
    match comp.len() {
      1 => {
        let i = comp[0];
        let entry = matrix[i][i].clone();
        result[i][i] =
          match crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(entry),
            right: Box::new(n_expr.clone()),
          }) {
            Ok(v) => v,
            Err(_) => return None,
          };
      }
      2 => {
        let i = comp[0];
        let j = comp[1];
        let block = [
          [matrix[i][i].clone(), matrix[i][j].clone()],
          [matrix[j][i].clone(), matrix[j][j].clone()],
        ];
        let block_n = matrix_power_2x2_symbolic_block(&block, n_expr)?;
        result[i][i] = block_n[0][0].clone();
        result[i][j] = block_n[0][1].clone();
        result[j][i] = block_n[1][0].clone();
        result[j][j] = block_n[1][1].clone();
      }
      _ => unreachable!(),
    }
  }

  // Assemble into nested-List Expr.
  let result_rows: Vec<Expr> = result
    .into_iter()
    .map(|row| Expr::List(row.into()))
    .collect();
  Some(Expr::List(result_rows.into()))
}

/// Closed form for `M^n` of a 2×2 matrix with distinct eigenvalues
/// via the diagonalisation `M^n = c_1·λ_1^n + c_2·λ_2^n` where
/// `c_1 = (M − λ_2 I)/(λ_1 − λ_2)` and `c_2 = (λ_1 I − M)/(λ_1 − λ_2)`.
///
/// Only handles 2×2 integer-entry blocks with a negative discriminant
/// `trace² − 4·det < 0` whose eigenvalues are `μ ± k·I` for some
/// integer `k`. Real-eigenvalue blocks (positive disc) are left
/// unevaluated — wolframscript's output there involves `Sqrt[disc]^n`
/// terms that Woxi doesn't reduce to wolframscript's canonical form.
fn matrix_power_2x2_symbolic_block(
  m: &[[Expr; 2]; 2],
  n_expr: &Expr,
) -> Option<[[Expr; 2]; 2]> {
  use crate::syntax::BinaryOperator;
  // Extract integer entries.
  let a = crate::functions::math_ast::expr_to_i128(&m[0][0])?;
  let b = crate::functions::math_ast::expr_to_i128(&m[0][1])?;
  let c = crate::functions::math_ast::expr_to_i128(&m[1][0])?;
  let d = crate::functions::math_ast::expr_to_i128(&m[1][1])?;

  let trace = a + d;
  let det = a * d - b * c;
  let disc = trace * trace - 4 * det;
  if disc >= 0 {
    // Real eigenvalues or repeated eigenvalue: leave unevaluated.
    return None;
  }
  let pos = -disc;
  // We need 4·det − trace² to be a perfect square so the eigenvalues
  // are `(trace ± k·I)/2` with `k` an integer. (Audit case: trace = 0,
  // det = 1, pos = 4, k = 2.)
  let k = (pos as f64).sqrt().round() as i128;
  if k * k != pos {
    return None;
  }
  // Both `trace + k` and `trace − k` must be even so the eigenvalues
  // have integer real and imaginary parts after halving.
  if (trace + k).rem_euclid(2) != 0 || (trace - k).rem_euclid(2) != 0 {
    return None;
  }
  // λ_1 = (trace + k·I)/2 = re_part + im_part·I
  // λ_2 = (trace − k·I)/2 = re_part − im_part·I
  let re_part = trace / 2;
  let im_part = k / 2;
  if re_part * 2 != trace || im_part * 2 != k {
    // Half-integer real or imaginary part — out of scope.
    return None;
  }

  // Build `Power[Complex[re, im], n]` directly so Woxi doesn't
  // canonicalise it through `Power[Times[-1, I], n] → Times[-1,
  // Power[I, n]]`, which is mathematically wrong for symbolic n.
  let lam_power = |re: i128, im: i128| -> Expr {
    let base = if re == 0 && im == 1 {
      Expr::Identifier("I".to_string())
    } else if re == 0 {
      Expr::FunctionCall {
        name: "Complex".to_string(),
        args: vec![Expr::Integer(0), Expr::Integer(im)].into(),
      }
    } else {
      Expr::FunctionCall {
        name: "Complex".to_string(),
        args: vec![Expr::Integer(re), Expr::Integer(im)].into(),
      }
    };
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base),
      right: Box::new(n_expr.clone()),
    }
  };
  let lam1_n = lam_power(re_part, im_part);
  let lam2_n = lam_power(re_part, -im_part);

  // `λ_1 − λ_2 = k·I`. Use Complex form for the same reason as above.
  let lambda_diff = if k == 1 {
    Expr::Identifier("I".to_string())
  } else {
    Expr::FunctionCall {
      name: "Complex".to_string(),
      args: vec![Expr::Integer(0), Expr::Integer(k)].into(),
    }
  };
  // For each scalar Complex factor `q = (a + b·I) / (k·I)` we'd build
  // and evaluate. Instead, compute the coefficient matrices c_1, c_2
  // numerically (each entry is a Complex rational), then assemble the
  // result via Plus of scalar·Power terms.

  let lambda_1 = Expr::FunctionCall {
    name: "Complex".to_string(),
    args: vec![Expr::Integer(re_part), Expr::Integer(im_part)].into(),
  };
  let lambda_2 = Expr::FunctionCall {
    name: "Complex".to_string(),
    args: vec![Expr::Integer(re_part), Expr::Integer(-im_part)].into(),
  };
  let mat_expr = |ai: i128| Expr::Integer(ai);
  let eval = crate::evaluator::evaluate_expr_to_expr;

  // c1_ij = (M[i,j] − λ_2·δ_ij) / (λ_1 − λ_2)
  // c2_ij = (λ_1·δ_ij − M[i,j]) / (λ_1 − λ_2)
  let build_entry = |entry_int: i128, is_diag: bool| -> Option<Expr> {
    let entry = mat_expr(entry_int);
    let c1_num_raw = if is_diag {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(entry.clone()),
        right: Box::new(lambda_2.clone()),
      }
    } else {
      entry.clone()
    };
    let c2_num_raw = if is_diag {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(lambda_1.clone()),
        right: Box::new(entry.clone()),
      }
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), entry.clone()].into(),
      }
    };
    let c1 = eval(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(c1_num_raw),
      right: Box::new(lambda_diff.clone()),
    })
    .ok()?;
    let c2 = eval(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(c2_num_raw),
      right: Box::new(lambda_diff.clone()),
    })
    .ok()?;
    // Build `c1 * λ_1^n + c2 * λ_2^n` without evaluating the Power
    // sub-expressions (so `(-I)^n` survives intact).
    let t1 = if matches!(&c1, Expr::Integer(1)) {
      lam1_n.clone()
    } else if matches!(&c1, Expr::Integer(0)) {
      Expr::Integer(0)
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![c1, lam1_n.clone()].into(),
      }
    };
    let t2 = if matches!(&c2, Expr::Integer(1)) {
      lam2_n.clone()
    } else if matches!(&c2, Expr::Integer(0)) {
      Expr::Integer(0)
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![c2, lam2_n.clone()].into(),
      }
    };
    let sum = if matches!(&t1, Expr::Integer(0)) {
      t2
    } else if matches!(&t2, Expr::Integer(0)) {
      t1
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![t1, t2].into(),
      }
    };
    eval(&sum).ok()
  };

  let m00 = build_entry(a, true)?;
  let m01 = build_entry(b, false)?;
  let m10 = build_entry(c, false)?;
  let m11 = build_entry(d, true)?;
  Some([[m00, m01], [m10, m11]])
}

/// Builds an elementary 3x3 active rotation matrix about the given Cartesian
/// axis (1 = x, 2 = y, 3 = z) by `angle`.
fn elementary_rotation(axis: i128, angle: &Expr) -> Expr {
  let c = Expr::FunctionCall {
    name: "Cos".to_string(),
    args: vec![angle.clone()].into(),
  };
  let s = Expr::FunctionCall {
    name: "Sin".to_string(),
    args: vec![angle.clone()].into(),
  };
  let neg_s = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), s.clone()].into(),
  };
  let zero = Expr::Integer(0);
  let one = Expr::Integer(1);
  let rows: [[Expr; 3]; 3] = match axis {
    1 => [
      [one.clone(), zero.clone(), zero.clone()],
      [zero.clone(), c.clone(), neg_s.clone()],
      [zero.clone(), s.clone(), c.clone()],
    ],
    2 => [
      [c.clone(), zero.clone(), s.clone()],
      [zero.clone(), one.clone(), zero.clone()],
      [neg_s.clone(), zero.clone(), c.clone()],
    ],
    // axis 3 (z) is the default for any other value
    _ => [
      [c.clone(), neg_s.clone(), zero.clone()],
      [s.clone(), c.clone(), zero.clone()],
      [zero.clone(), zero.clone(), one.clone()],
    ],
  };
  Expr::List(
    rows
      .into_iter()
      .map(|r| Expr::List(r.to_vec().into()))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// EulerMatrix[{a, b, c}] - rotation matrix R(a).R(b).R(c) using the default
/// {3, 2, 3} (ZYZ) axis convention. EulerMatrix[{a, b, c}, {n1, n2, n3}]
/// uses the explicitly given axis sequence.
pub fn euler_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let angles = match &args[0] {
    Expr::List(items) if items.len() == 3 => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EulerMatrix".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let axes: [i128; 3] = if args.len() == 2 {
    match &args[1] {
      Expr::List(items) if items.len() == 3 => {
        let mut out = [3i128; 3];
        for (i, item) in items.iter().enumerate() {
          match item {
            Expr::Integer(n) => out[i] = *n,
            _ => {
              return Ok(Expr::FunctionCall {
                name: "EulerMatrix".to_string(),
                args: args.to_vec().into(),
              });
            }
          }
        }
        out
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "EulerMatrix".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    // Wolfram's default Euler angle convention is {3, 2, 3} (ZYZ).
    [3, 2, 3]
  };

  let r1 = elementary_rotation(axes[0], &angles[0]);
  let r2 = elementary_rotation(axes[1], &angles[1]);
  let r3 = elementary_rotation(axes[2], &angles[2]);

  // R(axes[0], a) . R(axes[1], b) . R(axes[2], c)
  let r12 = crate::functions::linear_algebra_ast::dot_ast(&[r1, r2])?;
  let r123 = crate::functions::linear_algebra_ast::dot_ast(&[r12, r3])?;
  evaluate_expr_to_expr(&r123)
}
