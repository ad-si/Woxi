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
    "LinearSolve" if args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::linear_solve_ast(
        args,
      ));
    }
    "LeastSquares" if args.len() == 2 => {
      // LeastSquares[A, b] = Inverse[Transpose[A] . A] . Transpose[A] . b
      // when A has full column rank. Delegate to existing Transpose, Dot,
      // and Inverse so the result is exact for rational matrices. Keep
      // symbolic inputs unevaluated.
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
        let atb = match eval("Dot", &[at, b]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let inv_ata = match eval("Inverse", &[ata]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
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
      let (n, k) = if args.len() == 1 {
        // UnitVector[k] is shorthand for UnitVector[2, k]
        (2i128, expr_to_i128(&args[0]).unwrap_or(0))
      } else {
        (
          expr_to_i128(&args[0]).unwrap_or(0),
          expr_to_i128(&args[1]).unwrap_or(0),
        )
      };
      if n > 0 && k >= 1 && k <= n {
        let mut vec = vec![Expr::Integer(0); n as usize];
        vec[(k - 1) as usize] = Expr::Integer(1);
        return Some(Ok(Expr::List(vec)));
      }
    }
    "BoxMatrix" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0])
        && n >= 0
      {
        let size = (2 * n + 1) as usize;
        let row = Expr::List(vec![Expr::Integer(1); size]);
        return Some(Ok(Expr::List(vec![row; size])));
      }
    }
    "DiagonalMatrix" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::diagonal_matrix_ast(
        args,
      ));
    }
    "DiamondMatrix" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::diamond_matrix_ast(
        args,
      ));
    }
    "DiskMatrix" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::disk_matrix_ast(args));
    }
    "HilbertMatrix" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0])
        && n >= 0
      {
        let n = n as usize;
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
          let mut row = Vec::with_capacity(n);
          for j in 0..n {
            let denom = (i + j + 1) as i128;
            row.push(crate::functions::math_ast::make_rational_pub(1, denom));
          }
          rows.push(Expr::List(row));
        }
        return Some(Ok(Expr::List(rows)));
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
          rows.push(Expr::List(row));
        }
        return Some(Ok(Expr::List(rows)));
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
            args: vec![args[0].clone(), args[0].clone()],
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
            rows.push(Expr::List(row));
          }
          let result = Expr::List(rows);
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
              args: vec![ev.clone()],
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
    "Eigenvectors" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::eigenvectors_ast(
        args,
      ));
    }
    "Eigensystem" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::eigensystem_ast(args));
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
    "LinearModelFit" if args.len() == 3 => {
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
    "Cross" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::cross_ast(args));
    }
    // DotProduct[a, b] is the legacy VectorAnalysis spelling of Dot[a, b];
    // CrossProduct[a, b] is the same for Cross[a, b]. wolframscript's
    // `-code` mode leaves them unevaluated (the package isn't loaded),
    // but the natural reduction is what mathics expects.
    "DotProduct" if args.len() == 2 || args.len() == 3 => {
      let dot_call = Expr::FunctionCall {
        name: "Dot".to_string(),
        args: vec![args[0].clone(), args[1].clone()],
      };
      return Some(evaluate_expr_to_expr(&dot_call));
    }
    "CrossProduct" if args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::cross_ast(args));
    }
    // ScalarTripleProduct[a, b, c] = a · (b × c). Originally part of the
    // legacy VectorAnalysis package — wolframscript's `-code` mode also
    // leaves it unevaluated, so the value-level reduction here matches
    // the mathics expectation (and is consistent with how Cross/Dot are
    // already evaluated eagerly).
    "ScalarTripleProduct" if args.len() == 3 => {
      let cross_bc = match crate::functions::linear_algebra_ast::cross_ast(&[
        args[1].clone(),
        args[2].clone(),
      ]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let dot_call = Expr::FunctionCall {
        name: "Dot".to_string(),
        args: vec![args[0].clone(), cross_bc],
      };
      return Some(evaluate_expr_to_expr(&dot_call));
    }
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
    "KroneckerProduct" if args.len() == 2 => {
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
                        args: vec![Expr::Integer(-1), x.clone()],
                      },
                    ],
                  };
                  new_cols.push(entry);
                } else {
                  new_cols.push(elem.clone());
                }
              }
              new_rows.push(Expr::List(new_cols));
            }
          }
          let modified_mat = Expr::List(new_rows);
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
    "MatrixPower" if args.len() == 2 => {
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
              args: args.to_vec(),
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
      // Symbolic: return unevaluated
      return Some(Ok(Expr::FunctionCall {
        name: "MatrixPower".to_string(),
        args: args.to_vec(),
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
    "RotationMatrix" if args.len() == 1 => {
      // 2D rotation matrix: {{Cos[θ], -Sin[θ]}, {Sin[θ], Cos[θ]}}
      let theta = &args[0];
      let cos = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![theta.clone()],
      };
      let sin = Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![theta.clone()],
      };
      let neg_sin = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), sin.clone()],
      };
      let mat = Expr::List(vec![
        Expr::List(vec![cos.clone(), neg_sin]),
        Expr::List(vec![sin, cos]),
      ]);
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
          args: vec![theta.clone()],
        };
        let s = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![theta.clone()],
        };
        let one_minus_c = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), c.clone()],
            },
          ],
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
              args: vec![c.clone(), delta],
            },
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![one_minus_c.clone(), u[i].clone(), u[j].clone()],
            },
          ];
          if let Some((sign, uk)) = cross_term {
            terms.push(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![sign, s.clone(), uk.clone()],
            });
          }
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms,
          }
        };

        let mat = Expr::List(vec![
          Expr::List(vec![
            make_entry(0, 0),
            make_entry(0, 1),
            make_entry(0, 2),
          ]),
          Expr::List(vec![
            make_entry(1, 0),
            make_entry(1, 1),
            make_entry(1, 2),
          ]),
          Expr::List(vec![
            make_entry(2, 0),
            make_entry(2, 1),
            make_entry(2, 2),
          ]),
        ]);
        return Some(evaluate_expr_to_expr(&mat));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "RotationMatrix".to_string(),
        args: args.to_vec(),
      }));
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
          rows.push(Expr::List(row));
        }
        // Last row: all zeros except 1 in bottom-right
        let mut last_row = vec![Expr::Integer(0); n + 1];
        last_row[n] = Expr::Integer(1);
        rows.push(Expr::List(last_row));
        let matrix = Expr::List(rows);
        let evaluated =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "TransformationFunction".to_string(),
            args: vec![matrix],
          });
        return Some(evaluated);
      }
      return Some(Ok(Expr::FunctionCall {
        name: "TranslationTransform".to_string(),
        args: args.to_vec(),
      }));
    }
    // RotationTransform[angle] → TransformationFunction[2D rotation matrix in homogeneous coords]
    "RotationTransform" if args.len() == 1 => {
      let theta = &args[0];
      let cos_t = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![theta.clone()],
      };
      let sin_t = Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![theta.clone()],
      };
      let neg_sin_t = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), sin_t.clone()],
      };
      // Build the 3x3 homogeneous rotation matrix:
      // {{Cos[x], -Sin[x], 0}, {Sin[x], Cos[x], 0}, {0, 0, 1}}
      let matrix = Expr::List(vec![
        Expr::List(vec![cos_t.clone(), neg_sin_t, Expr::Integer(0)]),
        Expr::List(vec![sin_t, cos_t, Expr::Integer(0)]),
        Expr::List(vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(1)]),
      ]);
      // Evaluate the matrix to simplify trig functions (e.g. Cos[Pi/4] → 1/Sqrt[2])
      let evaluated =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "TransformationFunction".to_string(),
          args: vec![matrix],
        });
      return Some(evaluated);
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
          rows.push(Expr::List(row));
        }
        let mut last_row = vec![Expr::Integer(0); n + 1];
        last_row[n] = Expr::Integer(1);
        rows.push(Expr::List(last_row));
        let matrix = Expr::List(rows);
        let evaluated =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "TransformationFunction".to_string(),
            args: vec![matrix],
          });
        return Some(evaluated);
      }
      return Some(Ok(Expr::FunctionCall {
        name: "ScalingTransform".to_string(),
        args: args.to_vec(),
      }));
    }
    "Adjugate" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == n => Some(cols.clone()),
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
                sub.push(Expr::List(sub_row));
              }
              let minor_matrix = Expr::List(sub);
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
            result.push(Expr::List(row));
          }
          return Some(Ok(Expr::List(result)));
        }
      }
    }
    "QRDecomposition" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::qr_decomposition_ast(
        args,
      ));
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
                  args: vec![evaluated.clone()],
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
                  args: vec![evaluated.clone()],
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
                  args: vec![evaluated.clone()],
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
            Expr::List(cols) if cols.len() == n => Some(cols.clone()),
            _ => None,
          })
          .collect();
        if matrix.len() == n {
          for i in 0..n {
            for j in i..n {
              // Check M[i][j] == Conjugate[M[j][i]]
              let conj = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Conjugate".to_string(),
                args: vec![matrix[j][i].clone()],
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
    }
    "AntihermitianMatrixQ" if args.len() == 1 => {
      // Antihermitian: M == -ConjugateTranspose[M]
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == n => Some(cols.clone()),
            _ => None,
          })
          .collect();
        if matrix.len() == n {
          for i in 0..n {
            for j in i..n {
              // Check M[i][j] == -Conjugate[M[j][i]]
              let conj = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Conjugate".to_string(),
                args: vec![matrix[j][i].clone()],
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
    }
    "NormalMatrixQ" if args.len() == 1 => {
      // Normal: M.M^H == M^H.M where M^H is conjugate transpose
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let matrix: Vec<Vec<Expr>> = rows
          .iter()
          .filter_map(|r| match r {
            Expr::List(cols) if cols.len() == n => Some(cols.clone()),
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
                args: vec![matrix[j][i].clone()],
              })
              .unwrap_or(matrix[j][i].clone());
            }
          }
          // Compute M.M^H and M^H.M
          let ct_list =
            Expr::List(ct.iter().map(|r| Expr::List(r.clone())).collect());
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
    }
    // DiagonalMatrixQ[m] — True if m is diagonal
    "DiagonalMatrixQ" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        let mut is_diag = true;
        'diag: for i in 0..n {
          if let Expr::List(row) = &rows[i] {
            if row.len() != n {
              is_diag = false;
              break;
            }
            for j in 0..n {
              if i != j && !matches!(&row[j], Expr::Integer(0)) {
                let evaluated =
                  evaluate_expr_to_expr(&row[j]).unwrap_or(row[j].clone());
                if !matches!(evaluated, Expr::Integer(0)) {
                  is_diag = false;
                  break 'diag;
                }
              }
            }
          } else {
            is_diag = false;
            break;
          }
        }
        return Some(Ok(Expr::Identifier(
          if is_diag { "True" } else { "False" }.to_string(),
        )));
      }
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
                  args: vec![row_i[j].clone(), row_j[i].clone()],
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
            rows.push(Expr::List(row));
          }
          return Some(Ok(Expr::List(rows)));
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
          rows.push(Expr::List(row));
        }
        return Some(Ok(Expr::List(rows)));
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
                        args: vec![Expr::Integer(-1), inv_sqrt_n.clone()],
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
          return Some(Ok(Expr::List(result_rows)));
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
          rows.push(Expr::List(row));
        }
        return Some(Ok(Expr::List(rows)));
      }
    }
    // CrossMatrix[{a, b, c}] — skew-symmetric matrix such that CrossMatrix[v].u == Cross[v, u]
    // Returns {{0, -c, b}, {c, 0, -a}, {-b, a, 0}} for numeric vectors.
    // Wolfram requires numeric input and returns SparseArray; we return a plain list.
    "CrossMatrix" if args.len() == 1 => {
      if let Expr::List(elems) = &args[0]
        && elems.len() == 3 && elems.iter().all(|e| {
        matches!(e, Expr::Integer(_) | Expr::Real(_))
          || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
      }) {
        let a = &elems[0];
        let b = &elems[1];
        let c = &elems[2];
        let neg = |e: &Expr| -> Expr {
          match evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), e.clone()],
          }) {
            Ok(result) => result,
            Err(_) => Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), e.clone()],
            },
          }
        };
        let row0 = Expr::List(vec![Expr::Integer(0), neg(c), b.clone()]);
        let row1 = Expr::List(vec![c.clone(), Expr::Integer(0), neg(a)]);
        let row2 = Expr::List(vec![neg(b), a.clone(), Expr::Integer(0)]);
        return Some(Ok(Expr::List(vec![row0, row1, row2])));
      }
    }
    // FourierMatrix[n] — discrete Fourier transform matrix
    // Entry (j,k) = omega^((j-1)*(k-1)) / sqrt(n), omega = e^(2*pi*i/n)
    // Uses Cos + I*Sin for exact symbolic results
    "FourierMatrix" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0])
        && n > 0
      {
        let n = n as usize;
        let inv_sqrt_n = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::Integer(n as i128),
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(-1), Expr::Integer(2)],
            },
          ],
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
                  ],
                }
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![
                    Expr::FunctionCall {
                      name: "Rational".to_string(),
                      args: vec![Expr::Integer(snum), Expr::Integer(sden)],
                    },
                    Expr::Identifier("Pi".to_string()),
                  ],
                }
              };
              // Build (Cos[angle] + I*Sin[angle]) / Sqrt[n]
              let cos_part = Expr::FunctionCall {
                name: "Cos".to_string(),
                args: vec![angle.clone()],
              };
              let sin_part = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::Identifier("I".to_string()),
                  Expr::FunctionCall {
                    name: "Sin".to_string(),
                    args: vec![angle],
                  },
                ],
              };
              let omega = Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![cos_part, sin_part],
              };
              let entry = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![omega, inv_sqrt_n.clone()],
              };
              row.push(entry);
            }
          }
          rows.push(Expr::List(row));
        }
        let result = Expr::List(rows);
        return Some(evaluate_expr_to_expr(&result));
      }
    }
    // Symmetrize[matrix] — symmetrize a square matrix: (M + M^T) / 2
    // Wolfram returns SymmetrizedArray; we return a plain list (Normal form).
    "Symmetrize" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0] {
        let n = rows.len();
        // Check it's a square matrix of numeric values
        let mut matrix: Vec<Vec<Expr>> = Vec::new();
        let mut all_ok = true;
        for row in rows {
          if let Expr::List(cols) = row {
            if cols.len() != n {
              all_ok = false;
              break;
            }
            matrix.push(cols.clone());
          } else {
            all_ok = false;
            break;
          }
        }
        if all_ok && n > 0 {
          let mut result_rows = Vec::with_capacity(n);
          for i in 0..n {
            let mut result_cols = Vec::with_capacity(n);
            for j in 0..n {
              // (M[i,j] + M[j,i]) / 2
              let sum = Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![matrix[i][j].clone(), matrix[j][i].clone()],
              };
              let half = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(1), Expr::Integer(2)],
                  },
                  sum,
                ],
              };
              match evaluate_expr_to_expr(&half) {
                Ok(val) => result_cols.push(val),
                Err(_) => result_cols.push(half),
              }
            }
            result_rows.push(Expr::List(result_cols));
          }
          return Some(Ok(Expr::List(result_rows)));
        }
      }
    }
    _ => {}
  }
  None
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
        args: vec![mat.clone()],
      });
    }
  };

  let n = rows.len();
  if n == 0 {
    return Ok(Expr::List(vec![
      Expr::List(vec![]),
      Expr::List(vec![]),
      Expr::Integer(0),
    ]));
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
      matrix.push(cols.clone());
    } else {
      return Ok(Expr::FunctionCall {
        name: "LUDecomposition".to_string(),
        args: vec![mat.clone()],
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
            args: vec![pivot_val.clone(), Expr::Integer(-1)],
          },
        ],
      })
      .unwrap_or(matrix[i][k].clone());

      // Update row i: A[i][j] -= L[i][k] * A[k][j] for j > k
      for j in (k + 1)..n {
        let product = evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![l_ik.clone(), matrix[k][j].clone()],
        })
        .unwrap_or(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![l_ik.clone(), matrix[k][j].clone()],
        });

        let new_val = evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            matrix[i][j].clone(),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), product],
            },
          ],
        })
        .unwrap_or(matrix[i][j].clone());

        matrix[i][j] = new_val;
      }

      // Store L factor in place of A[i][k]
      matrix[i][k] = l_ik;
    }
  }

  // Build result
  let lu_matrix = Expr::List(matrix.into_iter().map(Expr::List).collect());

  let pivot_list = Expr::List(
    pivots
      .into_iter()
      .map(|p| Expr::Integer((p + 1) as i128)) // 1-indexed
      .collect(),
  );

  Ok(Expr::List(vec![lu_matrix, pivot_list, Expr::Integer(0)]))
}

fn is_zero_expr(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(0) => true,
    Expr::Real(v) => *v == 0.0,
    _ => false,
  }
}

/// KroneckerProduct[A, B] — tensor (Kronecker) product of two matrices.
fn kronecker_product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let a_rows = match &args[0] {
    Expr::List(rows) => rows,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "KroneckerProduct".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let b_rows = match &args[1] {
    Expr::List(rows) => rows,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "KroneckerProduct".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract as matrix of expressions
  let a: Vec<Vec<&Expr>> = a_rows
    .iter()
    .filter_map(|r| match r {
      Expr::List(cols) => Some(cols.iter().collect()),
      _ => None,
    })
    .collect();
  let b: Vec<Vec<&Expr>> = b_rows
    .iter()
    .filter_map(|r| match r {
      Expr::List(cols) => Some(cols.iter().collect()),
      _ => None,
    })
    .collect();

  if a.is_empty() || b.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let m = a.len();
  let n = a[0].len();
  let p = b.len();
  let q = b[0].len();

  let mut result = Vec::with_capacity(m * p);
  for i in 0..m {
    for k in 0..p {
      let mut row = Vec::with_capacity(n * q);
      for j in 0..n {
        for l in 0..q {
          // result[i*p+k][j*q+l] = a[i][j] * b[k][l]
          let product = crate::functions::math_ast::times_ast(&[
            a[i][j].clone(),
            b[k][l].clone(),
          ])?;
          row.push(product);
        }
      }
      result.push(Expr::List(row));
    }
  }

  Ok(Expr::List(result))
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
        args: vec![a.clone(), b.clone()],
      });
    }
  };

  // Count n11 (both 1), n10 (a=1,b=0), n01 (a=0,b=1), n00 (both 0)
  let mut n11: i128 = 0;
  let mut n10: i128 = 0;
  let mut n01: i128 = 0;
  let mut n00: i128 = 0;

  for (ai, bi) in list_a.iter().zip(list_b.iter()) {
    let av = match ai {
      Expr::Integer(v) => *v,
      _ => {
        return Ok(Expr::FunctionCall {
          name: name.to_string(),
          args: vec![a.clone(), b.clone()],
        });
      }
    };
    let bv = match bi {
      Expr::Integer(v) => *v,
      _ => {
        return Ok(Expr::FunctionCall {
          name: name.to_string(),
          args: vec![a.clone(), b.clone()],
        });
      }
    };
    match (av != 0, bv != 0) {
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
        args: vec![a.clone(), b.clone()],
      });
    }
  };

  if den == 0 {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  Ok(crate::functions::math_ast::make_rational(num, den))
}
