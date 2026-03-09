#[allow(unused_imports)]
use super::*;

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
    "Tr" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::tr_ast(args));
    }
    "IdentityMatrix" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::identity_matrix_ast(
        args,
      ));
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
    "DiagonalMatrix" if args.len() == 1 => {
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
    "LeviCivitaTensor" if args.len() == 2 => {
      if matches!(&args[1], Expr::Identifier(h) if h == "List") {
        return Some(
          crate::functions::linear_algebra_ast::levi_civita_tensor_ast(
            &args[..1],
          ),
        );
      }
    }
    "Eigenvalues" if args.len() == 1 => {
      return Some(crate::functions::linear_algebra_ast::eigenvalues_ast(args));
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
    "Fit" if args.len() == 3 => {
      return Some(crate::functions::linear_algebra_ast::fit_ast(args));
    }
    "FindFit" if args.len() == 4 => {
      return Some(crate::functions::linear_algebra_ast::find_fit_ast(args));
    }
    "Cross" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::cross_ast(args));
    }
    "VectorAngle" if args.len() == 2 => {
      return Some(crate::functions::linear_algebra_ast::vector_angle_ast(
        args,
      ));
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
    _ => {}
  }
  None
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
